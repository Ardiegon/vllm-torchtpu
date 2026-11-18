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

`build_kimi_moe_ep_overlap_op` registers one torch custom op (a
`jax.shard_map` over the `('dp', 'tp')` mesh) that runs a whole Kimi K3 MoE
block for expert parallelism, holding its Pallas DMA semaphores internally:

1. Dispatch. Each rank owns a `[B_dp, latent_dim / tp]` slice of the routed
   latent. XLA all-gathers on contiguous axes collect it over `'tp'`, then
   the latent plus the packed routing metadata over `'dp'`, next to the
   shared-expert gate/up matmul and gated activation.
2. Routed experts (`fused_moe_func`) on the gathered tokens.
3. Combine. A `'dp'` reduce-scatter, a `'tp'` psum, the exact
   `RMSNorm(latent_dim)` of the routed output, and the slice of this rank's
   `latent_dim / tp` columns.

The op returns `[B_dp, shared_inter_tp + latent_dim / tp]`: the shared
activation and the normalized routed slice, concatenated. The Pallas and
JAX building blocks live in `kernels/kimi_k3/exit_fold.py`. The routed
experts must be in the planar_t MXFP4 layout (uint8 weights and scales, no
biases).
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state as ps

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.kimi_k3.exit_fold import (
    compute_shared_gu_and_act,
    hierarchical_combine_rms_slice,
    unpack_axial_gathered,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"
DP_TP_MESH_AXES: tuple[str, str] = (DP_AXIS_NAME, TP_AXIS_NAME)

_MESH_2D_CACHE: dict[tuple[int, int], Mesh] = {}


def validate_dp_tp_group_ranks(
    dp_size: int,
    tp_size: int,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> tuple[int, int]:
    """Check that `dp_ranks` and `tp_ranks` match a `(dp_size, tp_size)` grid."""
    if dp_size < 1 or tp_size < 1:
        raise ValueError(
            "dp_size and tp_size must be >= 1, got "
            f"dp_size={dp_size}, tp_size={tp_size}"
        )
    if (global_rank is None or world_size is None) and not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized; pass explicit "
            "global_rank and world_size."
        )
    rank = dist.get_rank() if global_rank is None else global_rank
    world = dist.get_world_size() if world_size is None else world_size
    expected_world = dp_size * tp_size
    if world != expected_world:
        raise ValueError(
            f"2D ('dp', 'tp') mesh requires world_size == dp_size * tp_size "
            f"({dp_size} * {tp_size} = {expected_world}), got world_size={world}."
        )
    if not (0 <= rank < world):
        raise ValueError(f"global_rank={rank} out of bounds for world_size={world}.")
    dp_rank, tp_rank = divmod(rank, tp_size)
    expected_dp_ranks = tuple(d * tp_size + tp_rank for d in range(dp_size))
    expected_tp_ranks = tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
    if dp_ranks is not None and tuple(int(r) for r in dp_ranks) != expected_dp_ranks:
        raise ValueError(
            f"dp_group.ranks mismatch on global_rank={rank}: got {tuple(dp_ranks)}, "
            f"expected {expected_dp_ranks}."
        )
    if tp_ranks is not None and tuple(int(r) for r in tp_ranks) != expected_tp_ranks:
        raise ValueError(
            f"tp_group.ranks mismatch on global_rank={rank}: got {tuple(tp_ranks)}, "
            f"expected {expected_tp_ranks}."
        )
    return dp_rank, tp_rank


def build_dp_tp_mesh(
    dp_size: int | None = None,
    tp_size: int | None = None,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
) -> Mesh:
    """Build (or return cached) rank-ordered 2D `('dp', 'tp')` JAX mesh."""
    if dp_size is None or tp_size is None:
        if not ps.model_parallel_is_initialized():
            raise RuntimeError(
                "vLLM parallel_state is not initialized; pass explicit dp_size/tp_size."
            )
        v_dp_ranks = tuple(int(r) for r in ps.get_dp_group().ranks)
        v_tp_ranks = tuple(int(r) for r in ps.get_tp_group().ranks)
        dp_size = len(v_dp_ranks) if dp_size is None else dp_size
        tp_size = len(v_tp_ranks) if tp_size is None else tp_size
        dp_ranks = v_dp_ranks if dp_ranks is None else dp_ranks
        tp_ranks = v_tp_ranks if tp_ranks is None else tp_ranks
    elif dp_ranks is None and tp_ranks is None and ps.model_parallel_is_initialized():
        dp_ranks = tuple(int(r) for r in ps.get_dp_group().ranks)
        tp_ranks = tuple(int(r) for r in ps.get_tp_group().ranks)

    cache_key = (dp_size, tp_size)
    if torch.compiler.is_compiling():
        if cache_key not in _MESH_2D_CACHE:
            raise RuntimeError(
                "build_dp_tp_mesh must be called before torch.compile tracing "
                f"(no cached mesh found for dp_size={dp_size}, tp_size={tp_size})."
            )
        return _MESH_2D_CACHE[cache_key]

    dp_rank, tp_rank = validate_dp_tp_group_ranks(
        dp_size, tp_size, dp_ranks=dp_ranks, tp_ranks=tp_ranks
    )
    if cache_key not in _MESH_2D_CACHE:
        mesh = pallas.get_pallas_mesh(
            axis_names=DP_TP_MESH_AXES, mesh_shape=(dp_size, tp_size)
        )
        logger.info(
            "Built (dp, tp) mesh | shape=(dp=%d, tp=%d) | rank=%d -> "
            "(dp=%d, tp=%d) | dev_id=%d",
            dp_size,
            tp_size,
            dp_rank * tp_size + tp_rank,
            dp_rank,
            tp_rank,
            int(mesh.devices[dp_rank, tp_rank].id),
        )
        _MESH_2D_CACHE[cache_key] = mesh
    return _MESH_2D_CACHE[cache_key]


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
