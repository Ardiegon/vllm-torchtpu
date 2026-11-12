# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch bridge for fused MoE based on fused_moe_func in fused_moe_gmm."""

import enum
import functools
from collections.abc import Callable
from typing import Any

import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.gmm_v6e_planar_t_b11 import (
    FUSED_DECODE_MOE_MAX_TOKENS,
    FUSED_VERIFY_MOE_MAX_TOKENS,
    repack_w2_planar_t,
    repack_w13_planar_t,
)
from vllm_torchtpu.kimi_k3_opt_modes import (
    normalize_compute_polish_mode,
    normalize_spec_kda_mode,
)
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    quantize_to_native_fp4_kmajor,
    requant_unpack_kmajor,
    resolve_onehot_permute_threshold,
    unpack_fp4_to_e2m1,
)

_kernel_instance_counter = 0
_FusedDecodeOptions = tuple[str, bool, int]
_fused_moe_kernel_cache: dict[
    tuple[int, str, bool, Any, bool, _FusedDecodeOptions], Callable
] = {}
_load_kmajor_fp4_op = None
_planar_t_load_ops: dict[str, Callable] = {}
_requant_kmajor_fp4_ops: dict[int, Callable] = {}
_quantize_native_fp4_kmajor_ops: dict[tuple[int, bool], Callable] = {}


def _resolve_activation_name(activation: Any) -> str:
    if isinstance(activation, enum.Enum):
        return activation.value
    return activation.value if hasattr(activation, "value") else str(activation)


def get_fused_moe_activation(activation, moe_config) -> str:
    """Encode vLLM's activation config for the Pallas GMM kernel."""
    activation = _resolve_activation_name(activation)
    if activation != "situ":
        return activation

    beta = moe_config.activation_situ_beta
    linear_beta = moe_config.activation_situ_linear_beta
    assert beta is not None
    return f"situ:{beta}:{'none' if linear_beta is None else linear_beta}"


class TpuMoEActivationMixin:
    """Holds the Pallas GMM kernel's activation string for one MoE layer.

    Written by process_weights_after_loading, read by apply_monolithic.
    vLLM builds one quant method per MoE layer, so the method is the right
    owner. Defines no __init__, so it does not disturb the upstream
    __init__s these methods deliberately bypass; mix it in ahead of the
    upstream class.
    """

    _tpu_activation_str: str | None = None

    def _resolve_tpu_activation(self, layer) -> str:
        """Encode this layer's activation for the kernel.

        Overridden where a method needs a different encoding -- the string
        is not uniformly recomputable from layer.activation.
        """
        return get_fused_moe_activation(layer.activation, layer.moe_config)

    def _set_tpu_activation(self, layer) -> str:
        """Resolve and store; returns it, since callers also use it locally."""
        self._tpu_activation_str = self._resolve_tpu_activation(layer)
        return self._tpu_activation_str


def load_kmajor_fp4(w_u8: torch.Tensor) -> torch.Tensor:
    """One-time load transform: packed uint8 ``[..., N, K//2]`` -> native
    fp4 ``torch.float4_e2m1fn_x2`` ``[..., K, N]`` (K-major).

    Unpacks (bitcast uint8 -> float4_e2m1fn) and transposes the contracting axis
    to gmm_v2's K-major layout ONCE at load. The result is stored as a native-fp4
    Parameter, so the forward hands it straight to gmm_v2 with no per-forward
    unpack or transpose (which keeps the 480B HBM footprint bounded). Requires
    torch_tpu's native ``torch.float4_e2m1fn_x2`` dtype
    (google-pytorch/torch_tpu#1560).
    """
    global _load_kmajor_fp4_op
    if _load_kmajor_fp4_op is None:
        _load_kmajor_fp4_op = pallas.jax_op(
            "pallas::nvfp4_load_kmajor", unpack_fp4_to_e2m1
        )
    return _load_kmajor_fp4_op(w_u8)


def _planar_t_load_op(name: str, fn: Callable) -> Callable:
    """The cached torch bridge of a planar_t load transform."""
    op = _planar_t_load_ops.get(name)
    if op is None:
        op = _planar_t_load_ops[name] = pallas.jax_op(name, fn)
    return op


def load_w13_planar_t_mxfp4(
    w13_u8: torch.Tensor, s13_u8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time load transform of MXFP4 gate+up weights to the planar_t layout.

    Args:
      w13_u8: `[E, 2*N, K/2]` uint8 MXFP4 nibble pairs (gate rows first).
      s13_u8: `[E, 2*N, K/32]` uint8 E8M0 scales.

    Returns:
      `(w13, s13)`: `[E, K, N]` uint8 planar_t nibbles and
      `[E, 2*(K/32), 1, N]` uint8 transposed scales, the operands the
      float8_e4m3b11fnuz decode kernels read
      (`kernels.megablox.gmm_v6e_planar_t_b11.repack_w13_planar_t`).
    """
    return _planar_t_load_op("pallas::mxfp4_load_w13_planar_t", repack_w13_planar_t)(
        w13_u8, s13_u8
    )


def load_w2_planar_t_mxfp4(
    w2_u8: torch.Tensor, s2_u8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time load transform of MXFP4 down-projection weights to planar_t.

    Args:
      w2_u8: `[E, K, N/2]` uint8 MXFP4 nibble pairs.
      s2_u8: `[E, K, N/32]` uint8 E8M0 scales.

    Returns:
      `(w2, s2)`: `[E, N/2, K]` uint8 planar_t nibbles and `[E, N/32, 1, K]`
      uint8 transposed scales
      (`kernels.megablox.gmm_v6e_planar_t_b11.repack_w2_planar_t`).
    """
    return _planar_t_load_op("pallas::mxfp4_load_w2_planar_t", repack_w2_planar_t)(
        w2_u8, s2_u8
    )


def prepare_routes(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    pipelined: bool,
    mask_padding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Readies pre-selected routes for the fused MoE bridges.

    Casts the ids to int32 and the weights to the activation dtype, then,
    with `mask_padding`, zeroes the routing weights of padded tokens
    (`token_padding.zero_routing_weights_for_padding`; `pipelined` callers
    pass rank-local tensors).

    Returns:
      `(topk_weights, topk_ids)`.
    """
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(x.dtype)
    if mask_padding:
        topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
            topk_ids, topk_weights, is_local_tensor=pipelined
        )
    return topk_weights, topk_ids


def requant_load_kmajor_fp4(
    w_u8: torch.Tensor, scale_f: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """W4A8 requant + K-major load in one JAX op (matches tpu-inference). Takes
    the packed uint8 weight ``[..., N, K//2]`` + its fused fp32 block-16 scale
    ``[..., N, K//16]``; returns native-fp4 ``torch.float4_e2m1fn_x2`` ``[...,
    K, N]`` requantized to ``block`` plus the fp32 kernel scale ``[...,
    K//block, 1, N]``. Doing the dequant+requant in JAX (vs torch) avoids
    materializing the dequantized weight on the host and rounds to fp4 with the
    same cast the kernel reads."""
    op = _requant_kmajor_fp4_ops.get(block)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_requant_kmajor_b{block}",
            functools.partial(requant_unpack_kmajor, block=block),
        )
        _requant_kmajor_fp4_ops[block] = op
    return op(w_u8, scale_f)


def quantize_native_fp4_kmajor(
    w: torch.Tensor, block: int, pack: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an already-dequantized, K-major float weight to native FP4.

    ``pack=False`` hands the weight back as ``torch.float4_e2m1fn_x2`` rather
    than packed uint8 -- the form the fused EP MoE kernel takes. The two are
    distinct traced programs with distinct op names, so the cache is keyed on
    both: one qualname serving two closures would silently dispatch to
    whichever registered first.
    """
    key = (block, pack)
    op = _quantize_native_fp4_kmajor_ops.get(key)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_quantize_native_kmajor_b{block}"
            f"{'' if pack else '_unpacked'}",
            functools.partial(quantize_to_native_fp4_kmajor, block=block, pack=pack),
        )
        _quantize_native_fp4_kmajor_ops[key] = op
    return op(w)


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _resolve_fused_decode_options() -> _FusedDecodeOptions:
    """Settings of the fused small-batch MXFP4 MoE kernel.

    Returns `(token_trays_mode, rank3_scales, fused_decode_max_tokens)` for
    `fused_moe_func`, read from the Kimi K3 `TPU_KIMI_OPT_COMPUTE_POLISH`
    (`trays`, `copy_opt` tokens) and `TPU_KIMI_OPT_SPEC_KDA` (`"o6b"` lets
    the kernel serve speculative verify batches of up to
    FUSED_VERIFY_MOE_MAX_TOKENS tokens). Resolved when the op is built,
    outside the trace, and part of the op cache key.
    """
    _, polish_tokens = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    spec_kda_mode = normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA)
    return (
        "trays" if "trays" in polish_tokens else "off",
        "copy_opt" in polish_tokens,
        FUSED_VERIFY_MOE_MAX_TOKENS
        if spec_kda_mode == "o6b"
        else FUSED_DECODE_MOE_MAX_TOKENS,
    )


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    skip_padded_tokens: bool,
    fused_decode_options: _FusedDecodeOptions,
    rhs_quant_dtype=None,
):
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    # TODO: once the pod's vLLM includes --moe-backend (added upstream in
    # vLLM PR #33807, 2026-03-01; absent in the 0.19.0 we run), register the
    # SparseCore vs plain variants through that selector + the MoE oracle
    # instead of this env flag -- the MoE analog of how #191 moved batched
    # RPA onto --attention-backend CUSTOM. Note MoEBackend has no CUSTOM slot
    # yet, so that also needs an upstream OOT-backend hook.
    use_sparse_core = envs.USE_MOE_SPARSE_CORE

    token_trays_mode, rank3_scales, fused_decode_max_tokens = fused_decode_options

    # Pass experts_start as a runtime tensor rather than static closure arg so every EP
    # rank compiles an identical program.
    # See fa8faaf5 for background on rank-uniform sharding in compiled Pallas graphs.
    wrapped_fn = functools.partial(
        fused_moe_func,
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        token_trays_mode=token_trays_mode,
        rank3_scales=rank3_scales,
        fused_decode_max_tokens=fused_decode_max_tokens,
    )

    fused_moe_kernel_impl = pallas.jax_op(op_name, wrapped_fn)

    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (
        topk,
        activation,
        use_ep,
        rhs_quant_dtype,
        skip_padded_tokens,
        fused_decode_options,
    )
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
):
    if skip_padded_tokens is None:
        skip_padded_tokens = envs.TPU_MOE_SKIP_PADDED_TOKENS
    fused_decode_options = _resolve_fused_decode_options()
    cache_key = (
        topk,
        activation,
        use_ep,
        rhs_quant_dtype,
        skip_padded_tokens,
        fused_decode_options,
    )
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        fused_decode_options=fused_decode_options,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP) -- see
    ``moe_routing.get_experts_start_buffer``. It must be a 0-d int32 tensor
    (a persistent buffer registered once per layer), not a Python int: every
    EP rank owns a different value, and passing it as real tensor data (vs.
    binding it as a Python int into the compiled closure) keeps the compiled
    program identical across ranks. The kernel remaps global ids to local ids
    with an elementwise subtract and masks non-local experts. When present,
    the sparse-core parity path dispatches ragged gather/gather-reduce.
    """
    use_ep = experts_start is not None
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )
    if experts_start is None:
        # The compiled program for use_ep=False never reads this operand
        # (see the `if use_ep:` guard in fused_moe_func), but the custom op
        # still needs a concrete tensor to call with a fixed arity.
        experts_start = torch.zeros((), dtype=torch.int32, device=hidden_states.device)
    return fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        experts_start,
    )
