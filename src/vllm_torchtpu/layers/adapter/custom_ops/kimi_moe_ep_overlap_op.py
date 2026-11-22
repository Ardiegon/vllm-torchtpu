# Copyright 2026 Google LLC
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
"""Kimi K3 MoE layer exit with the EP dispatch overlapped by shared experts.

`build_kimi_moe_ep_overlap_op` registers one sharded torch custom op (a
`jax.shard_map` over the `('dp', 'tp')` mesh via `pallas.jax_op`) that runs a
whole Kimi K3 MoE block for expert parallelism, holding its Pallas DMA
semaphores internally:

1. Dispatch. Each rank owns a `[B_dp, latent_dim / tp]` slice of the routed
   latent. Small batches (at most `max_dma_tokens` tokens across DP) gather it
   in two axial stages: a one-shot remote-DMA all-gather over `'tp'`, then a
   non-blocking remote-DMA all-gather over `'dp'` of the latent plus the
   byte-packed routing metadata, with the shared-expert gate/up matmul and
   `SituAndMul` activation running between start and wait. Larger batches use
   XLA all-gathers on contiguous axes instead.
2. Routed experts (`fused_moe_func`) on the gathered tokens.
3. Combine. A one-shot `'dp'` reduce-scatter, a `'tp'` psum guarded to small
   widths (XLA collectives above the caps), the exact `RMSNorm(latent_dim)` of
   the routed output, and the slice of this rank's `latent_dim / tp` columns.

The op returns `[B_dp, shared_inter_tp + latent_dim / tp]`: the shared
activation and the normalized routed slice, concatenated. The Pallas and JAX
building blocks live in `kernels/kimi_k3/exit_fold.py`. The routed experts must
be in the planar_t MXFP4 layout (uint8 weights and scales, no biases).

Serving note (`LIBTPU_INIT_ARGS`, consumed once by libtpu at PJRT startup):
overlapping the split `'dp'` remote-DMA dispatch with the shared-expert matmul
and fusing surrounding XLA collectives on TPU v6e uses::

    --xla_msa_preferred_overlap_to_async_copy_ratio=12
    --xla_tpu_enable_async_collective_fusion=true
    --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_tp_mesh import DP_AXIS_NAME, TP_AXIS_NAME
from vllm_torchtpu.kernels.collectives.one_shot_dma import (
    DEFAULT_MAX_DMA_TOKENS,
    latency_all_gather_cols,
)
from vllm_torchtpu.kernels.kimi_k3.exit_fold import (
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    compute_shared_gu_and_act,
    dispatch_overlapping_shared_experts,
    hierarchical_combine_rms_slice,
    ragged_dp_dispatch_combine_rms_slice,
    unpack_axial_gathered,
)
from vllm_torchtpu.kernels.megablox.gmm_v6e_planar_t_b11 import (
    is_planar_t_b11_layout,
    moe_gmm_v6e_planar_t_b11,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    ragged_gather_reduce_v2,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    _counting_sort_positions,
    fused_moe_func,
    resolve_onehot_permute_threshold,
)


@dataclasses.dataclass(frozen=True)
class MoeOverlapConfig:
    """Static configuration of one EP-overlap op, resolved before tracing.

    Hashable, so it also keys the op cache.
    """

    # Activation string of the routed experts (see `fused_moe_func`).
    routed_activation: str
    topk: int
    # Whether the routed output passes `RMSNorm(latent_dim)`, and its epsilon.
    use_norm: bool
    rms_eps: float
    # `SituAndMul` parameters of the shared experts.
    situ_beta: float
    situ_linear_beta: float | None
    # Largest batch the fused small-batch MXFP4 MoE kernel serves.
    fused_decode_max_tokens: int


# Built ops by (op name, device ids, config, routed-MoE env settings), so
# every layer of one shape reuses one custom op.
_OVERLAP_OPS: dict[tuple, Callable[..., torch.Tensor]] = {}


def _operand_names(use_norm: bool) -> tuple[str, ...]:
    """Operands of the op, in call order."""
    names = [
        "hidden_states",  # [dp * B_dp, H]
        "routed_down_local",  # [dp * B_dp, tp * d_local]
        "packed_meta",  # [dp * B_dp, 3 * topk]
        "w_shared_gu",  # [H, tp * 2 * s_inter_tp]
        "w13",  # [dp, tp, E_local, ...]
        "w2",  # [dp, tp, E_local, ...]
        "w13_scale",  # [dp, tp, E_local, ...]
        "w2_scale",  # [dp, tp, E_local, ...]
    ]
    if use_norm:
        names.append("norm_weight")  # [latent_dim]
    names += ["experts_start", "tp_rank"]  # [dp, tp] each
    return tuple(names)


def _annotate_operands(fn: Callable, names: tuple[str, ...], annotation: type) -> None:
    """Give `fn(*args)` the explicit signature `jax_op` and its fake need."""
    fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
            )
            for name in names
        ],
        return_annotation=annotation,
    )


def _make_moe_ep_overlap_jax_fn(
    mesh: Mesh,
    cfg: MoeOverlapConfig,
    *,
    max_dma_tokens: int,
    use_sparse_core: bool,
    onehot_permute_threshold: int,
) -> tuple[Callable[..., jax.Array], tuple[PartitionSpec, ...], PartitionSpec]:
    """The SPMD JAX function of the op with its input and output specs."""
    dp_size, tp_size = mesh.devices.shape
    topk = cfg.topk
    names = _operand_names(cfg.use_norm)

    per_rank = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    dp_rows = PartitionSpec(DP_AXIS_NAME, None)
    specs = {
        "hidden_states": dp_rows,
        "routed_down_local": per_rank,
        "packed_meta": dp_rows,
        "w_shared_gu": PartitionSpec(None, TP_AXIS_NAME),
        "norm_weight": PartitionSpec(None),
    }
    in_specs = tuple(specs.get(name, per_rank) for name in names)
    out_spec = per_rank

    def _local(*local_args: jax.Array) -> jax.Array:
        named = dict(zip(names, local_args, strict=True))
        local_hidden = named["hidden_states"]
        local_rdown = named["routed_down_local"]
        local_meta = named["packed_meta"]
        local_w_shared_gu = named["w_shared_gu"]
        local_norm_w = (
            named["norm_weight"] if cfg.use_norm else jnp.ones((1,), jnp.bfloat16)
        )

        b_dp, d_local = local_rdown.shape
        latent_dim = tp_size * d_local
        b_global = dp_size * b_dp
        dp_lanes = latent_dim + 3 * topk

        # 1. Dispatch + shared gate/up compute + routed experts + combine.
        w13 = jnp.squeeze(named["w13"], axis=(0, 1))
        w2 = jnp.squeeze(named["w2"], axis=(0, 1))
        w13_scale = jnp.squeeze(named["w13_scale"], axis=(0, 1))
        w2_scale = jnp.squeeze(named["w2_scale"], axis=(0, 1))

        ragged_min_tokens = max(max_dma_tokens, cfg.fused_decode_max_tokens)
        if (
            b_global > ragged_min_tokens
            and dp_size > 1
            and is_planar_t_b11_layout(w13, w13_scale, rhs_quant_dtype=None)
        ):
            # Prefill / large batch with multi-DP: compact ragged EP all-to-all
            # over 'dp' (dispatches only routes belonging to this TP column into
            # expert-major local buffers, avoiding full [b_global * topk, H]
            # allocations and padded DP all-gather / reduce-scatter).
            shared_act = compute_shared_gu_and_act(
                local_hidden,
                local_w_shared_gu,
                situ_beta=cfg.situ_beta,
                situ_linear_beta=cfg.situ_linear_beta,
            )
            lat_tp = lax.all_gather(local_rdown, TP_AXIS_NAME, axis=1, tiled=True)
            routed_local = ragged_dp_dispatch_combine_rms_slice(
                lat_tp,
                local_meta,
                w13,
                w2,
                w13_scale,
                w2_scale,
                local_norm_w,
                named["tp_rank"],
                dp_axis_name=DP_AXIS_NAME,
                tp_axis_name=TP_AXIS_NAME,
                dp_size=dp_size,
                tp_size=tp_size,
                topk=topk,
                routed_activation=cfg.routed_activation,
                use_norm=cfg.use_norm,
                rms_eps=cfg.rms_eps,
                use_sparse_core=use_sparse_core,
                counting_sort_fn=_counting_sort_positions,
                ragged_gather_fn=ragged_gather_v2,
                ragged_gather_reduce_fn=ragged_gather_reduce_v2,
                moe_gmm_fn=moe_gmm_v6e_planar_t_b11,
            )
            return jnp.concatenate((shared_act, routed_local), axis=-1)

        if b_global > max_dma_tokens:
            # Prefill / large batch fallback (single-DP or non-planar weights):
            # XLA all-gathers on contiguous axes.
            shared_act = compute_shared_gu_and_act(
                local_hidden,
                local_w_shared_gu,
                situ_beta=cfg.situ_beta,
                situ_linear_beta=cfg.situ_linear_beta,
            )
            lat_tp = lax.all_gather(local_rdown, TP_AXIS_NAME, axis=1, tiled=True)
            packed_dp = jnp.concatenate((lat_tp, local_meta[:, : 3 * topk]), axis=-1)
            gathered_dp = lax.all_gather(packed_dp, DP_AXIS_NAME, axis=0, tiled=True)
        else:
            # Small batch: one-shot TP all-gather of the latent, then the async
            # DP dispatch of latent + metadata overlapped with shared experts.
            lat_tp = latency_all_gather_cols(
                local_rdown,
                TP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_TP_LATENT_AG,
            )
            packed_dp = jnp.concatenate((lat_tp, local_meta[:, : 3 * topk]), axis=-1)
            gathered_dp_3d, shared_act = dispatch_overlapping_shared_experts(
                packed_dp,
                DP_AXIS_NAME,
                local_hidden,
                local_w_shared_gu,
                rows=b_dp,
                orig_lanes=dp_lanes,
                situ_beta=cfg.situ_beta,
                situ_linear_beta=cfg.situ_linear_beta,
            )
            gathered_dp = gathered_dp_3d.reshape(b_global, dp_lanes)
        ag_latent, ag_ids, ag_weights = unpack_axial_gathered(
            gathered_dp, latent_dim=latent_dim, topk=topk
        )

        # 2. Routed experts.
        moe_out = fused_moe_func(
            ag_latent,
            w13,
            w2,
            w13_scale,
            w2_scale,
            None,
            None,
            ag_weights,
            ag_ids,
            named["experts_start"][0, 0],
            topk=topk,
            activation=cfg.routed_activation,
            use_ep=True,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_permute_threshold,
            rhs_quant_dtype=None,
            skip_padded_tokens=True,
            fused_decode_max_tokens=cfg.fused_decode_max_tokens,
            ep_size=dp_size * tp_size,
        )

        # 3. Exact combine + RMSNorm(latent_dim) + [latent_local] slice.
        routed_local = hierarchical_combine_rms_slice(
            moe_out,
            local_norm_w,
            named["tp_rank"],
            dp_axis_name=DP_AXIS_NAME,
            tp_axis_name=TP_AXIS_NAME,
            tp_size=tp_size,
            latent_dim=latent_dim,
            use_norm=cfg.use_norm,
            rms_eps=cfg.rms_eps,
            max_dma_tokens=max_dma_tokens,
        )

        # One contiguous [B_dp, s_inter_tp + latent_local] output.
        return jnp.concatenate((shared_act, routed_local), axis=-1)

    def _jax_fn(*global_args: jax.Array) -> jax.Array:
        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_vma=False,
        )(*global_args)
        return lax.with_sharding_constraint(out, NamedSharding(mesh, out_spec))

    _annotate_operands(_jax_fn, names, jax.Array)
    return _jax_fn, in_specs, out_spec


def build_kimi_moe_ep_overlap_op(
    mesh: Mesh,
    cfg: MoeOverlapConfig,
    *,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> Callable[..., torch.Tensor]:
    """Build (or reuse the cached) EP-overlap op for one MoE layer shape."""
    dp_size, tp_size = mesh.devices.shape
    dev_ids = tuple(int(d.id) for d in mesh.devices.flat)
    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot_permute_threshold = int(resolve_onehot_permute_threshold())
    op_name = (
        f"pallas::kimi_moe_ep_overlap_k{cfg.topk}_situ"
        f"_n{int(cfg.use_norm)}_m{cfg.fused_decode_max_tokens}"
        f"_cap{max_dma_tokens}_dp{dp_size}_tp{tp_size}"
    )
    cache_key = (op_name, dev_ids, cfg, use_sparse_core, onehot_permute_threshold)
    if cache_key in _OVERLAP_OPS:
        return _OVERLAP_OPS[cache_key]

    jax_fn, in_specs, out_spec = _make_moe_ep_overlap_jax_fn(
        mesh,
        cfg,
        max_dma_tokens=max_dma_tokens,
        use_sparse_core=use_sparse_core,
        onehot_permute_threshold=onehot_permute_threshold,
    )
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as other jax_op call sites pass them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        op_name,
        jax_fn,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(out_spec,),
    )

    def _fake_fn(*fake_args: torch.Tensor) -> torch.Tensor:
        hidden, rdown, _, w_shared_gu = fake_args[:4]
        shared_inter_tp = w_shared_gu.shape[1] // 2
        return hidden.new_empty((hidden.shape[0], shared_inter_tp + rdown.shape[1]))

    _annotate_operands(_fake_fn, _operand_names(cfg.use_norm), torch.Tensor)
    op.register_fake(_fake_fn)
    _OVERLAP_OPS[cache_key] = op
    return op
