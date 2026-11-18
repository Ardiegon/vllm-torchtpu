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
`jax.shard_map` over the `('dp', 'tp')` mesh via `sharded_jax_op`) that runs a
whole Kimi K3 MoE block for expert parallelism:

1. Dispatch. Each rank owns a `[B_dp, latent_dim / tp]` slice of the routed
   latent. XLA all-gathers on contiguous axes collect it over `'tp'`, then
   the latent plus the byte-packed routing metadata over `'dp'`, next to
   the shared-expert gate/up matmul and gated activation.
2. Routed experts (`fused_moe_func`) on the gathered tokens.
3. Combine. A `'dp'` reduce-scatter, a `'tp'` psum, the exact
   `RMSNorm(latent_dim)` of the routed output, and the slice of this rank's
   `latent_dim / tp` columns.

The op returns `[B_dp, shared_inter_tp + latent_dim / tp]`: the shared
activation and the normalized routed slice, concatenated. The JAX building
blocks live in `kernels/kimi_k3/exit_fold.py`. The routed experts must be in
the planar_t MXFP4 layout (uint8 weights and scales, no biases).

Serving note (`LIBTPU_INIT_ARGS`, consumed once by libtpu at PJRT startup):
overlapping async XLA collectives with the shared-expert compute and fusing
adjacent all-gathers on TPU v6e uses::

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
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_tp_mesh import DP_AXIS_NAME, TP_AXIS_NAME
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.kimi_k3.exit_fold import (
    compute_shared_gu_and_act,
    hierarchical_combine_rms_slice,
    unpack_axial_gathered,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
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
    """Give `fn(*args)` the explicit signature `sharded_jax_op` needs."""
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
    use_sparse_core: bool,
    onehot_permute_threshold: int,
) -> tuple[Callable[..., jax.Array], tuple[PartitionSpec, ...], PartitionSpec]:
    """The SPMD JAX function of the op with its input and output specs."""
    _, tp_size = mesh.devices.shape
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

        latent_dim = tp_size * local_rdown.shape[1]

        # 1. Dispatch + shared gate/up compute. XLA all-gathers on contiguous
        # axes (TP on axis 1, then DP on axis 0), so no 4D transpose is needed.
        shared_act = compute_shared_gu_and_act(
            local_hidden,
            local_w_shared_gu,
            situ_beta=cfg.situ_beta,
            situ_linear_beta=cfg.situ_linear_beta,
        )
        lat_tp = lax.all_gather(local_rdown, TP_AXIS_NAME, axis=1, tiled=True)
        packed_dp = jnp.concatenate((lat_tp, local_meta[:, : 3 * topk]), axis=-1)
        gathered_dp = lax.all_gather(packed_dp, DP_AXIS_NAME, axis=0, tiled=True)
        ag_latent, ag_ids, ag_weights = unpack_axial_gathered(
            gathered_dp, latent_dim=latent_dim, topk=topk
        )

        # 2. Routed experts.
        moe_out = fused_moe_func(
            ag_latent,
            jnp.squeeze(named["w13"], axis=(0, 1)),
            jnp.squeeze(named["w2"], axis=(0, 1)),
            jnp.squeeze(named["w13_scale"], axis=(0, 1)),
            jnp.squeeze(named["w2_scale"], axis=(0, 1)),
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
        )

        # One contiguous [B_dp, s_inter_tp + latent_local] output.
        return jnp.concatenate((shared_act, routed_local), axis=-1)

    def _jax_fn(*global_args: jax.Array) -> jax.Array:
        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_vma=False,
        )(*global_args)

    _annotate_operands(_jax_fn, names, jax.Array)
    return _jax_fn, in_specs, out_spec


def build_kimi_moe_ep_overlap_op(
    mesh: Mesh,
    cfg: MoeOverlapConfig,
) -> Callable[..., torch.Tensor]:
    """Build (or reuse the cached) EP-overlap op for one MoE layer shape."""
    dp_size, tp_size = mesh.devices.shape
    dev_ids = tuple(int(d.id) for d in mesh.devices.flat)
    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot_permute_threshold = int(resolve_onehot_permute_threshold())
    op_name = (
        f"pallas::kimi_moe_ep_overlap_k{cfg.topk}_situ"
        f"_n{int(cfg.use_norm)}_m{cfg.fused_decode_max_tokens}"
        f"_dp{dp_size}_tp{tp_size}"
    )
    cache_key = (op_name, dev_ids, cfg, use_sparse_core, onehot_permute_threshold)
    if cache_key in _OVERLAP_OPS:
        return _OVERLAP_OPS[cache_key]

    jax_fn, in_specs, out_spec = _make_moe_ep_overlap_jax_fn(
        mesh,
        cfg,
        use_sparse_core=use_sparse_core,
        onehot_permute_threshold=onehot_permute_threshold,
    )
    op = sharded_jax_op(
        op_name,
        jax_fn,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_spec,
    )
    _OVERLAP_OPS[cache_key] = op
    return op
