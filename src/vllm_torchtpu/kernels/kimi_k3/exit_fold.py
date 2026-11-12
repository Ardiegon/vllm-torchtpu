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
"""Kimi K3 Phase O4 unified-`@jax_op` split async EP DMA overlap & exact exit-fold.

Implements (`1_v6e_128_design_doc.md §7.2 Step 5`, `D87`, `D89`):
  1. Single-`@jax_op` (`torch.tpu.pallas.jax_op` + `jax.shard_map` over
     `mesh_2d = ('dp', 'tp')`) execution holding `pltpu.SemaphoreType.DMA`
     internally across:
       - `async_ep_dispatch_start` (`cid=1`): non-blocking Pallas remote-DMA
         all-gather of `(routed_down_local, packed_meta_shard)` across
         `('dp', 'tp')` (`'tp'` when `dp_size == 1`),
       - Overlapped `shared_experts` gate+up matmul (`hidden @ w_shared_gu`) +
         gated activation (`SituAndMul` / `SiluAndMul`) fenced between
         `ep_dispatch_start` and `ep_dispatch_wait` via zero-cost
         `jax.lax.optimization_barrier` edges so XLA strictly schedules
         `[ep_dispatch_start] -> [w_shared_gu + act_fn] -> [ep_dispatch_wait]`,
       - `async_ep_dispatch_wait` (`cid=1`): semaphore receive wait + full send
         drain (`0` residual semaphore count) + FTZ-safe byte-lane unpack,
       - Routed MoE compute (`fused_moe_func` -> `fused_decode_moe_v6e_planar_t_b11`
         for `T_global <= 64`, `moe_gmm` fallback for `T_global > 64`, with zero
         Python branching on `hidden_states.shape[0]`), and
       - Mathematically exact `routed_expert_norm` (`RMSNorm(latent_dim)`)
         combine + `[latent_local]` exit-fold (`O4-A` vs `O4-B`):
           * `O4-A`: `world`-way slot-owner `psum_scatter` + exact local
             `RMSNorm(latent_dim)` + barrier-free `'tp'` `seam_exit_split`
             (`cid=6`), deleting all `routed_out` `TP` `all-reduce`s (`-92` on
             `93L`, `-4` on `L5`).
           * `O4-B`: 1-shot `'dp'` `latency_psum_scatter_rows` (`cid=2`) +
             width-guarded `'tp'` `latency_psum` (`cid=3`) + in-`@jax_op`
             fused `RMSNorm(latent_dim)` + `[latent_local]` slice.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import torch
from jax import lax
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.kimi_k3.pallas_dma import (
    COLLECTIVE_ID_MOE_DP_COMBINE_RS,
    COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    COLLECTIVE_ID_MOE_OWNER_SEAM_SPLIT,
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
    DEFAULT_MAX_DMA_TOKENS,
    DEFAULT_VMEM_MAX_TOKENS,
    DP_AXIS_NAME,
    TP_AXIS_NAME,
    _normalize_axis_name,
    barrier_and_allocate,
    call_kernel,
    kimi_guarded_all_gather_cols,
    kimi_guarded_all_gather_rows,
    kimi_guarded_psum,
    kimi_guarded_psum_scatter_rows,
    seam_pieces_pack,
    seam_pieces_tile,
    seam_pieces_unpack,
)
from vllm_torchtpu.kernels.kimi_k3.polish import (
    active_trays_mode,
    use_copy_opt,
    vmem_chain_mode,
)

Ref = Any

DEFAULT_C2_COMBINE_KIND = "o4b"
DEFAULT_C2_DISPATCH_KIND = "axial"


def normalize_exit_fold_mode(raw: str | None) -> str:
    """Normalize ``TPU_KIMI_OPT_EXIT_FOLD`` to a canonical mode token.

    Supported modes:
      - ``"off"`` (aliases: ``""``, ``"0"``, ``"false"``, ``"no"``, ``"none"``)
      - ``"o4a"``: flat world async EP overlap + world owner combine + local RMSNorm + TP ``seam_exit_split`` (`cid=6`)
      - ``"o4b"``: flat world async EP overlap + DP ``psum_scatter`` (`cid=2`) + TP ``latency_psum`` (`cid=3`) + fused RMSNorm + slice
      - ``"o4d"`` (alias: ``"c2b"``): 2-stage axial ``TP -> DP`` async EP overlap (`cid=0` + `cid=1`) + ``O4-B`` combine (`cid=2` + `cid=3`) + 1D axial prefill fallback
      - ``"o4a_sync"``: synchronous world EP dispatch + ``O4-A`` exit-fold (ablation)
      - ``"o4b_sync"``: synchronous world EP dispatch + ``O4-B`` exit-fold (ablation)
      - ``"o4d_sync"``: synchronous 2-stage axial ``TP -> DP`` dispatch + ``O4-B`` exit-fold (ablation)
      - ``"c2"`` (aliases: ``"auto"``, ``"1"``, ``"true"``, ``"yes"``): Cycle-2b verified axial preset (``"o4d"``)
    """
    val = (raw or "off").strip().lower()
    if val in ("", "0", "false", "no", "off", "none"):
        return "off"
    if val == "c2b":
        return "o4d"
    if val in ("o4a", "o4b", "o4d", "o4a_sync", "o4b_sync", "o4d_sync"):
        return val
    if val in ("1", "true", "yes", "auto", "c2"):
        return "c2"
    raise ValueError(
        f"Invalid TPU_KIMI_OPT_EXIT_FOLD={raw!r}; expected 'off', 'o4a', 'o4b', "
        "'o4d'/'c2b', 'o4a_sync', 'o4b_sync', 'o4d_sync', or 'c2'/'auto'/'1'."
    )


def is_exit_fold_active(raw: str | None = None) -> bool:
    """Return True when ``TPU_KIMI_OPT_EXIT_FOLD`` enables the O4 split-weight path."""
    src = envs.TPU_KIMI_OPT_EXIT_FOLD if raw is None else raw
    return normalize_exit_fold_mode(src) != "off"


def resolve_effective_exit_fold_plan(
    mode: str,
    *,
    dp_size: int,
    tp_size: int,
    b_dp: int,
) -> tuple[str, bool, str]:
    """Map canonical ``exit_fold_mode`` + static shape to ``(combine_kind, async_dispatch, dispatch_kind)``."""
    del dp_size, tp_size, b_dp
    norm = normalize_exit_fold_mode(mode)
    if norm == "o4a":
        return "o4a", True, "world"
    if norm == "o4b":
        return "o4b", True, "world"
    if norm == "o4d":
        return "o4b", True, "axial"
    if norm == "o4a_sync":
        return "o4a", False, "world"
    if norm == "o4b_sync":
        return "o4b", False, "world"
    if norm == "o4d_sync":
        return "o4b", False, "axial"
    if norm == "c2":
        return DEFAULT_C2_COMBINE_KIND, True, DEFAULT_C2_DISPATCH_KIND
    raise ValueError(f"resolve_effective_exit_fold_plan called with inactive mode={mode!r}")


def resolve_effective_exit_fold_mode(
    mode: str,
    *,
    dp_size: int,
    tp_size: int,
    b_dp: int,
) -> tuple[str, bool]:
    """Map canonical ``exit_fold_mode`` + static shape to ``(combine_kind, async_dispatch)``."""
    combine_kind, async_dispatch, _ = resolve_effective_exit_fold_plan(
        mode,
        dp_size=dp_size,
        tp_size=tp_size,
        b_dp=b_dp,
    )
    return combine_kind, async_dispatch


# =============================================================================
# Exact JAX RMSNorm & Shared-Expert Activations (`SituAndMul` / `SiluAndMul`)
# =============================================================================


def _jax_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Exact TPU RMSNorm matching ``vllm.ir.ops.layernorm.rms_norm`` (`<= 1` bf16 ULP)."""
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    y_f32 = x_f32 * lax.rsqrt(var + float(eps))
    y_bf16 = lax.reduce_precision(y_f32, exponent_bits=8, mantissa_bits=7).astype(
        jnp.bfloat16
    )
    return (y_bf16 * weight.astype(jnp.bfloat16)).astype(jnp.bfloat16)


def _apply_shared_activation(
    x: jax.Array,
    *,
    act_kind: str,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    """Apply Kimi `SituAndMul` or `SiluAndMul` to `[B_dp, 2 * inter_tp]`."""
    gate, up = jnp.split(x, 2, axis=-1)
    if act_kind == "situ":
        gf = gate.astype(jnp.float32)
        beta = float(situ_beta)
        gate_act = (beta * jnp.tanh(gf / beta) * jax.nn.sigmoid(gf)).astype(
            jnp.bfloat16
        )
        if situ_linear_beta is not None:
            uf = up.astype(jnp.float32)
            lbeta = float(situ_linear_beta)
            up_act = (lbeta * jnp.tanh(uf / lbeta)).astype(jnp.bfloat16)
        else:
            up_act = up
        return (gate_act * up_act).astype(jnp.bfloat16)
    if act_kind == "silu":
        gate_act = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        return (gate_act * up).astype(jnp.bfloat16)
    raise ValueError(f"Unsupported shared_experts activation {act_kind!r}")


def _compute_shared_gu_and_act(
    local_hidden: jax.Array,
    local_w_shared_gu: jax.Array,
    *,
    act_kind: str,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    with jax.named_scope("kimi_shared_gu_dot"):
        shared_gu = jnp.dot(
            local_hidden,
            local_w_shared_gu,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
    with jax.named_scope("kimi_shared_gu_act"):
        return _apply_shared_activation(
            shared_gu,
            act_kind=act_kind,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )


# =============================================================================
# Split Non-Blocking Pallas Async EP Dispatch (`start` + `wait`, `cid=1`)
# =============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AsyncDispatchState:
    """Pinned HBM send/recv buffers + DMA semaphores kept strictly inside `@jax_op`."""

    x_send: Any
    x_recv: Any
    data_send_sem: Ref
    data_recv_sem: Ref


@jax.named_scope("async_ep_dispatch")
def async_ep_dispatch_start(
    send_shard: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    collective_id: int = COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
) -> tuple[AsyncDispatchState | jax.Array, jax.Array]:
    """Barrier-allocate buffers and launch non-blocking 1-shot remote-DMA gather."""
    axis_key = _normalize_axis_name(axis_name)
    world_size = jax.lax.axis_size(axis_key)
    if world_size == 1:
        return send_shard[None, :, :], jnp.zeros((1,), dtype=jnp.int32)

    x_packed = seam_pieces_pack(send_shard[None, :, :], axis_size=1)
    recv_shape = (world_size, *x_packed.shape[1:])

    def _start_kernel(
        x_ref: Any,
        state_ref: AsyncDispatchState,
        start_tok_ref: Any,
        *,
        local_sem: Any,
    ) -> None:
        with jax.named_scope("dispatch_barrier_wait"):
            barrier_sem = pltpu.get_barrier_semaphore()
            pl.semaphore_wait(barrier_sem, world_size)

        my_id = jnp.uint32(jax.lax.axis_index(axis_key))
        with jax.named_scope("stage_send_and_local_recv"):
            pltpu.make_async_copy(
                src_ref=x_ref, dst_ref=state_ref.x_send, sem=local_sem
            ).start()
            pltpu.make_async_copy(
                src_ref=x_ref,
                dst_ref=state_ref.x_recv.at[pl.ds(my_id, 1)],
                sem=local_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=x_ref, dst_ref=state_ref.x_send, sem=local_sem
            ).wait()
            pltpu.make_async_copy(
                src_ref=x_ref,
                dst_ref=state_ref.x_recv.at[pl.ds(my_id, 1)],
                sem=local_sem,
            ).wait()

        with jax.named_scope("launch_async_remote_dispatch"):
            for i in range(world_size):

                @pl.when(i != my_id)
                def _(i: int = i) -> None:
                    pltpu.make_async_remote_copy(
                        src_ref=state_ref.x_send.at[pl.ds(0, 1)],
                        dst_ref=state_ref.x_recv.at[pl.ds(my_id, 1)],
                        send_sem=state_ref.data_send_sem,
                        recv_sem=state_ref.data_recv_sem,
                        device_id={axis_key: i},
                    ).start()

        start_tok_ref[0] = jnp.int32(0)

    with xla_metadata.set_xla_metadata(
        async_type="collective-start",
        latency_metadata="300000",
    ):
        init_state = barrier_and_allocate(
            AsyncDispatchState(
                x_send=pltpu.HBM(shape=x_packed.shape, dtype=x_packed.dtype),
                x_recv=pltpu.HBM(shape=recv_shape, dtype=x_packed.dtype),
                data_send_sem=pltpu.SemaphoreType.DMA,
                data_recv_sem=pltpu.SemaphoreType.DMA,
            ),
            collective_id=collective_id,
            axis_name=axis_key,
            order_dep=x_packed,
            allow_pin=True,
        )
        live_state, start_tok = call_kernel(
            _start_kernel,
            args=(x_packed,),
            inout=(init_state,),
            output=(jnp.zeros((1,), dtype=jnp.int32),),
            scoped={"local_sem": pltpu.SemaphoreType.DMA},
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
            allow_pin=True,
            name="kimi_ep_dispatch_start",
        )
    return live_state, start_tok


def async_ep_dispatch_wait(
    state: AsyncDispatchState | jax.Array,
    wait_tok: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    rows: int,
    orig_lanes: int,
    collective_id: int = COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    zero_extra_copy: bool = False,
) -> jax.Array:
    """Wait on in-flight async EP dispatch DMAs, drain sends (`0` leak), and unpack."""
    axis_key = _normalize_axis_name(axis_name)
    world_size = jax.lax.axis_size(axis_key)
    if world_size == 1:
        assert isinstance(state, jax.Array)
        return state

    assert isinstance(state, AsyncDispatchState)
    if zero_extra_copy:
        def _wait_kernel_inplace(
            wait_tok_ref: Any,
            state_ref: AsyncDispatchState,
        ) -> None:
            _ = wait_tok_ref[0]
            with jax.named_scope("wait_remote_dispatch_recv"):
                recv_slice = state_ref.x_recv.at[pl.ds(0, world_size - 1)]
                pltpu.make_async_copy(
                    src_ref=recv_slice,
                    dst_ref=recv_slice,
                    sem=state_ref.data_recv_sem,
                ).wait()
            with jax.named_scope("drain_remote_dispatch_sends"):
                for _ in range(world_size - 1):
                    pltpu.make_async_copy(
                        src_ref=state_ref.x_send,
                        dst_ref=state_ref.x_send,
                        sem=state_ref.data_send_sem,
                    ).wait()

        with xla_metadata.set_xla_metadata(
            async_type="collective-done",
            latency_metadata="300000",
        ):
            (state_done,) = call_kernel(
                _wait_kernel_inplace,
                args=(wait_tok,),
                inout=(state,),
                compiler_params=pltpu.CompilerParams(collective_id=collective_id),
                allow_pin=True,
                name="kimi_ep_dispatch_wait_inplace",
            )
        return seam_pieces_unpack(state_done.x_recv, rows, orig_lanes)

    recv_shape = state.x_recv.shape
    recv_dtype = state.x_recv.dtype

    def _wait_kernel(
        wait_tok_ref: Any,
        state_ref: AsyncDispatchState,
        out_ref: Any,
        *,
        out_sem: Any,
    ) -> None:
        _ = wait_tok_ref[0]
        with jax.named_scope("wait_remote_dispatch_recv"):
            recv_slice = state_ref.x_recv.at[pl.ds(0, world_size - 1)]
            pltpu.make_async_copy(
                src_ref=recv_slice,
                dst_ref=recv_slice,
                sem=state_ref.data_recv_sem,
            ).wait()
        with jax.named_scope("drain_remote_dispatch_sends"):
            for _ in range(world_size - 1):
                pltpu.make_async_copy(
                    src_ref=state_ref.x_send,
                    dst_ref=state_ref.x_send,
                    sem=state_ref.data_send_sem,
                ).wait()
        with jax.named_scope("emit_dispatch_rows"):
            pltpu.make_async_copy(
                src_ref=state_ref.x_recv,
                dst_ref=out_ref,
                sem=out_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=state_ref.x_recv,
                dst_ref=out_ref,
                sem=out_sem,
            ).wait()

    with xla_metadata.set_xla_metadata(
        async_type="collective-done",
        latency_metadata="300000",
    ):
        _, out_raw = call_kernel(
            _wait_kernel,
            args=(wait_tok,),
            inout=(state,),
            output=(pltpu.HBM(shape=recv_shape, dtype=recv_dtype),),
            scoped={"out_sem": pltpu.SemaphoreType.DMA},
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
            allow_pin=True,
            name="kimi_ep_dispatch_wait",
        )
    return seam_pieces_unpack(out_raw, rows, orig_lanes)


def _unpack_meta_columns(
    meta_2d: jax.Array,
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Unpack FTZ-safe byte-valued `bf16` `[rows, >= 3 * topk]` -> `(ag_ids, ag_weights)`."""
    ag_meta = meta_2d[:, : 3 * topk]
    low_i32 = ag_meta[:, :topk].astype(jnp.int32)
    high_i32 = ag_meta[:, topk : 2 * topk].astype(jnp.int32)
    ag_ids = (((high_i32 << 8) | low_i32) << 16) >> 16
    ag_weights = ag_meta[:, 2 * topk : 3 * topk]
    return ag_ids, ag_weights


def _unpack_axial_gathered(
    gathered_dp_2d: jax.Array,
    *,
    latent_dim: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unpack `[dp_size * b_dp, latent_dim + 3 * topk]` -> `(ag_latent, ag_ids, ag_weights)` (`0` swapaxes)."""
    ag_latent = gathered_dp_2d[:, :latent_dim]
    ag_ids, ag_weights = _unpack_meta_columns(
        gathered_dp_2d[:, latent_dim:], topk=topk
    )
    return ag_latent, ag_ids, ag_weights


def _unpack_world_gathered(
    gathered_3d: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    b_dp: int,
    d_local: int,
    m_per_tp: int,
    topk: int,
    copy_opt: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unpack `[world_size, b_dp, d_local + m_per_tp]` -> `(ag_latent, ag_ids, ag_weights)`."""
    if copy_opt and b_dp == 1:
        ag_latent = gathered_3d[..., :d_local].reshape(dp_size, tp_size * d_local)
        ag_meta_padded = gathered_3d[..., d_local:].reshape(
            dp_size, tp_size * m_per_tp
        )
    else:
        gathered_4d = gathered_3d.reshape(dp_size, tp_size, b_dp, d_local + m_per_tp)
        lat_4d = gathered_4d[..., :d_local]
        meta_4d = gathered_4d[..., d_local:]
        ag_latent = jnp.swapaxes(lat_4d, 1, 2).reshape(
            dp_size * b_dp, tp_size * d_local
        )
        ag_meta_padded = jnp.swapaxes(meta_4d, 1, 2).reshape(
            dp_size * b_dp, tp_size * m_per_tp
        )
    ag_ids, ag_weights = _unpack_meta_columns(ag_meta_padded, topk=topk)
    return ag_latent, ag_ids, ag_weights


# =============================================================================
# Option `O4-A` (`world` Owner Combine + Local `RMSNorm` + `seam_exit_split`, `cid=6`)
# and Option `O4-B` (`cid=2` + `cid=3` + Fused `RMSNorm` + `[latent_local]` Slice)
# =============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class O4ACombineSeamState:
    """Layer-barrier-born state for `O4-A` owner combine + barrier-free `seam_exit_split`."""

    combine_recv: Any
    combine_recv_sem: Ref
    seam_exit_recv: Any
    seam_exit_recv_sem: Ref


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class O4AOwnerOnlyState:
    combine_recv: Any
    combine_recv_sem: Ref


def o4a_owner_combine_rms_seam_split(
    moe_out: jax.Array,
    local_norm_w: jax.Array,
    local_tp_rank: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    b_dp: int,
    latent_dim: int,
    use_norm: bool,
    rms_eps: float,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    collective_id: int = COLLECTIVE_ID_MOE_OWNER_SEAM_SPLIT,
    vmem_mode: str = "off",
) -> jax.Array:
    """Option `O4-A`: `world` owner combine + exact local `RMSNorm(latent_dim)` + `'tp'` `seam_exit_split`."""
    latent_local = latent_dim // tp_size
    world_size = dp_size * tp_size
    b_global = dp_size * b_dp

    # Prefill fallback (`T_global > max_dma_tokens`) inside JAX — zero Python shape branching!
    if b_global > max_dma_tokens or world_size == 1:
        routed_dp = (
            lax.psum_scatter(moe_out, DP_AXIS_NAME, scatter_dimension=0, tiled=True)
            if dp_size > 1
            else moe_out
        )
        routed_full = (
            lax.psum(routed_dp, TP_AXIS_NAME) if tp_size > 1 else routed_dp
        )
        if use_norm:
            routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)
        if tp_size > 1:
            return lax.dynamic_slice_in_dim(
                routed_full,
                local_tp_rank[0, 0] * latent_local,
                latent_local,
                axis=1,
            )
        return routed_full

    world_axis: str | tuple[str, str] = (
        TP_AXIS_NAME if dp_size == 1 else (DP_AXIS_NAME, TP_AXIS_NAME)
    )

    # Pad each DP group's token count up to a multiple of `tp_size` (`s_dp = tp_size * slots_per_tp`)
    # so every `(d, t)` rank owns `slots_per_tp >= 1` tokens of its own DP group `d`!
    slots_per_tp = (b_dp + tp_size - 1) // tp_size
    s_dp = tp_size * slots_per_tp
    moe_by_dp = moe_out.reshape(dp_size, b_dp, latent_dim)
    if s_dp > b_dp:
        moe_by_dp = jnp.pad(moe_by_dp, ((0, 0), (0, s_dp - b_dp), (0, 0)))
    moe_3d = moe_by_dp.reshape(world_size, slots_per_tp, latent_dim)
    moe_packed = seam_pieces_pack(moe_3d, world_size)

    if tp_size == 1:
        init_only = barrier_and_allocate(
            O4AOwnerOnlyState(
                combine_recv=pltpu.HBM(
                    shape=moe_packed.shape, dtype=moe_packed.dtype
                ),
                combine_recv_sem=pltpu.SemaphoreType.DMA,
            ),
            collective_id=collective_id,
            axis_name=world_axis,
            order_dep=moe_packed,
            allow_pin=True,
        )

        def _owner_only_kernel(
            moe_ref: Any,
            state_ref: O4AOwnerOnlyState,
            *,
            send_sem: Any,
            local_sem: Any,
        ) -> None:
            barrier_sem = pltpu.get_barrier_semaphore()
            pl.semaphore_wait(barrier_sem, world_size)
            my_w = jnp.uint32(jax.lax.axis_index(world_axis))
            for i in range(world_size):

                @pl.when(i != my_w)
                def _(i: int = i) -> None:
                    pltpu.make_async_remote_copy(
                        src_ref=moe_ref.at[pl.ds(i, 1)],
                        dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                        send_sem=send_sem,
                        recv_sem=state_ref.combine_recv_sem,
                        device_id={world_axis: i},
                    ).start()

            pltpu.make_async_copy(
                src_ref=moe_ref.at[pl.ds(my_w, 1)],
                dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                sem=local_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=moe_ref.at[pl.ds(my_w, 1)],
                dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                sem=local_sem,
            ).wait()
            wait_ref = moe_ref.at[pl.ds(0, world_size - 1)]
            pltpu.make_async_copy(
                src_ref=wait_ref, dst_ref=wait_ref, sem=state_ref.combine_recv_sem
            ).wait()
            pltpu.make_async_copy(
                src_ref=wait_ref, dst_ref=wait_ref, sem=send_sem
            ).wait()

        (state_only_out,) = call_kernel(
            _owner_only_kernel,
            args=(moe_packed,),
            inout=(init_only,),
            scoped={
                "send_sem": pltpu.SemaphoreType.DMA,
                "local_sem": pltpu.SemaphoreType.DMA,
            },
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
            allow_pin=True,
            name="kimi_o4a_owner_combine",
        )
        recv_unpacked = seam_pieces_unpack(
            state_only_out.combine_recv, slots_per_tp, latent_dim
        )
        owned_sum = recv_unpacked.astype(jnp.float32).sum(axis=0).astype(moe_out.dtype)
        if use_norm:
            owned_sum = _jax_rms_norm(owned_sum, local_norm_w, rms_eps)
        return owned_sum[:b_dp, :]

    # `tp_size > 1`: allocate BOTH `combine_recv` AND `seam_exit_recv` under ONE
    # `world_axis` barrier so `seam_exit_split` needs ZERO extra barrier wave!
    seam_tile = seam_pieces_tile(tp_size, slots_per_tp, latent_local, moe_out.dtype)
    seam_shape = seam_tile.shape

    use_vmem_src = vmem_mode in ("vmem_chain", "vmem_src")
    use_vmem_recv = vmem_mode == "vmem_chain"
    recv_ctor = pltpu.VMEM if use_vmem_recv else pltpu.HBM
    vmem_fields: tuple[str, ...] = (
        ("combine_recv", "seam_exit_recv") if use_vmem_recv else ()
    )

    init_state = barrier_and_allocate(
        O4ACombineSeamState(
            combine_recv=recv_ctor(shape=moe_packed.shape, dtype=moe_packed.dtype),
            combine_recv_sem=pltpu.SemaphoreType.DMA,
            seam_exit_recv=recv_ctor(shape=seam_shape, dtype=moe_out.dtype),
            seam_exit_recv_sem=pltpu.SemaphoreType.DMA,
        ),
        collective_id=collective_id,
        axis_name=world_axis,
        order_dep=moe_packed,
        allow_pin=True,
        order_dep_vmem=use_vmem_src,
    )

    def _owner_combine_kernel(
        moe_ref: Any,
        state_ref: O4ACombineSeamState,
        *,
        send_sem: Any,
        local_sem: Any,
    ) -> None:
        with jax.named_scope("o4a_combine_barrier_wait"):
            barrier_sem = pltpu.get_barrier_semaphore()
            pl.semaphore_wait(barrier_sem, world_size)
        my_w = jnp.uint32(jax.lax.axis_index(world_axis))
        with jax.named_scope("o4a_send_owner_partials"):
            for i in range(world_size):

                @pl.when(i != my_w)
                def _(i: int = i) -> None:
                    pltpu.make_async_remote_copy(
                        src_ref=moe_ref.at[pl.ds(i, 1)],
                        dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                        send_sem=send_sem,
                        recv_sem=state_ref.combine_recv_sem,
                        device_id={world_axis: i},
                    ).start()

        with jax.named_scope("o4a_local_owner_copy"):
            pltpu.make_async_copy(
                src_ref=moe_ref.at[pl.ds(my_w, 1)],
                dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                sem=local_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=moe_ref.at[pl.ds(my_w, 1)],
                dst_ref=state_ref.combine_recv.at[pl.ds(my_w, 1)],
                sem=local_sem,
            ).wait()

        with jax.named_scope("o4a_wait_owner_partials"):
            wait_ref = moe_ref.at[pl.ds(0, world_size - 1)]
            pltpu.make_async_copy(
                src_ref=wait_ref, dst_ref=wait_ref, sem=state_ref.combine_recv_sem
            ).wait()
            pltpu.make_async_copy(
                src_ref=wait_ref, dst_ref=wait_ref, sem=send_sem
            ).wait()

    owner_scoped: dict[str, Any] = {
        "send_sem": pltpu.SemaphoreType.DMA,
        "local_sem": pltpu.SemaphoreType.DMA,
    }

    (state_after_combine,) = call_kernel(
        _owner_combine_kernel,
        args=(moe_packed,),
        inout=(init_state,),
        scoped=owner_scoped,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        allow_pin=True,
        vmem_args=use_vmem_src,
        vmem_inout_fields=vmem_fields,
        name=(
            f"kimi_o4a_owner_combine_{vmem_mode}"
            if use_vmem_src
            else "kimi_o4a_owner_combine"
        ),
    )

    # Extract only `seam_exit_recv` + `seam_exit_recv_sem` so `combine_recv` dies right after sum!
    combine_raw = state_after_combine.combine_recv
    seam_exit_recv = state_after_combine.seam_exit_recv
    seam_exit_recv_sem = state_after_combine.seam_exit_recv_sem
    del state_after_combine

    received_3d = seam_pieces_unpack(combine_raw, slots_per_tp, latent_dim)
    owned_sum = received_3d.astype(jnp.float32).sum(axis=0).astype(moe_out.dtype)
    owned_norm = (
        _jax_rms_norm(owned_sum, local_norm_w, rms_eps) if use_norm else owned_sum
    )

    pieces = jnp.swapaxes(
        owned_norm.reshape(slots_per_tp, tp_size, latent_local), 0, 1
    )
    pieces_packed = seam_pieces_pack(pieces, tp_size)

    if use_vmem_src:
        def _seam_exit_split_inplace_kernel(
            pieces_ref: Any,
            seam_recv_ref: Any,
            seam_sem_ref: Any,
            *,
            send_sem: Any,
            local_sem: Any,
        ) -> None:
            my_tp = jnp.uint32(jax.lax.axis_index(TP_AXIS_NAME))
            with jax.named_scope("seam_send_pieces"):
                for p in range(tp_size):

                    @pl.when(p != my_tp)
                    def _(p: int = p) -> None:
                        pltpu.make_async_remote_copy(
                            src_ref=pieces_ref.at[pl.ds(p, 1)],
                            dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                            send_sem=send_sem,
                            recv_sem=seam_sem_ref,
                            device_id={TP_AXIS_NAME: p},
                        ).start()

            with jax.named_scope("seam_local_piece_copy"):
                pltpu.make_async_copy(
                    src_ref=pieces_ref.at[pl.ds(my_tp, 1)],
                    dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                    sem=local_sem,
                ).start()
                pltpu.make_async_copy(
                    src_ref=pieces_ref.at[pl.ds(my_tp, 1)],
                    dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                    sem=local_sem,
                ).wait()

            with jax.named_scope("seam_wait_pieces"):
                wait_tp_ref = pieces_ref.at[pl.ds(0, tp_size - 1)]
                pltpu.make_async_copy(
                    src_ref=wait_tp_ref, dst_ref=wait_tp_ref, sem=seam_sem_ref
                ).wait()
                pltpu.make_async_copy(
                    src_ref=wait_tp_ref, dst_ref=wait_tp_ref, sem=send_sem
                ).wait()

        seam_scoped: dict[str, Any] = {
            "send_sem": pltpu.SemaphoreType.DMA,
            "local_sem": pltpu.SemaphoreType.DMA,
        }

        seam_out_raw, _ = call_kernel(
            _seam_exit_split_inplace_kernel,
            args=(pieces_packed,),
            inout=(seam_exit_recv, seam_exit_recv_sem),
            scoped=seam_scoped,
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
            allow_pin=True,
            vmem_args=True,
            vmem_inout_all=use_vmem_recv,
            name=f"kimi_o4a_seam_exit_split_{vmem_mode}",
        )
        recv_pieces = seam_pieces_unpack(seam_out_raw, slots_per_tp, latent_local)
        return recv_pieces.reshape(tp_size * slots_per_tp, latent_local)[:b_dp, :]

    def _seam_exit_split_kernel(
        pieces_ref: Any,
        seam_recv_ref: Any,
        seam_sem_ref: Any,
        out_ref: Any,
        *,
        send_sem: Any,
        local_sem: Any,
        out_sem: Any,
    ) -> None:
        my_tp = jnp.uint32(jax.lax.axis_index(TP_AXIS_NAME))
        with jax.named_scope("seam_send_pieces"):
            for p in range(tp_size):

                @pl.when(p != my_tp)
                def _(p: int = p) -> None:
                    pltpu.make_async_remote_copy(
                        src_ref=pieces_ref.at[pl.ds(p, 1)],
                        dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                        send_sem=send_sem,
                        recv_sem=seam_sem_ref,
                        device_id={TP_AXIS_NAME: p},
                    ).start()

        with jax.named_scope("seam_local_piece_copy"):
            pltpu.make_async_copy(
                src_ref=pieces_ref.at[pl.ds(my_tp, 1)],
                dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                sem=local_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=pieces_ref.at[pl.ds(my_tp, 1)],
                dst_ref=seam_recv_ref.at[pl.ds(my_tp, 1)],
                sem=local_sem,
            ).wait()

        with jax.named_scope("seam_wait_pieces"):
            wait_tp_ref = pieces_ref.at[pl.ds(0, tp_size - 1)]
            pltpu.make_async_copy(
                src_ref=wait_tp_ref, dst_ref=wait_tp_ref, sem=seam_sem_ref
            ).wait()
            pltpu.make_async_copy(
                src_ref=wait_tp_ref, dst_ref=wait_tp_ref, sem=send_sem
            ).wait()

        with jax.named_scope("seam_emit_rows"):
            pltpu.make_async_copy(
                src_ref=seam_recv_ref, dst_ref=out_ref, sem=out_sem
            ).start()
            pltpu.make_async_copy(
                src_ref=seam_recv_ref, dst_ref=out_ref, sem=out_sem
            ).wait()

    _, _, seam_out_raw = call_kernel(
        _seam_exit_split_kernel,
        args=(pieces_packed,),
        inout=(seam_exit_recv, seam_exit_recv_sem),
        output=(pltpu.HBM(shape=seam_shape, dtype=moe_out.dtype),),
        scoped={
            "send_sem": pltpu.SemaphoreType.DMA,
            "local_sem": pltpu.SemaphoreType.DMA,
            "out_sem": pltpu.SemaphoreType.DMA,
        },
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        allow_pin=True,
        name="kimi_o4a_seam_exit_split",
    )
    recv_pieces = seam_pieces_unpack(seam_out_raw, slots_per_tp, latent_local)
    return recv_pieces.reshape(tp_size * slots_per_tp, latent_local)[:b_dp, :]


def o4b_dp_rs_tp_psum_fused_rms_slice(
    moe_out: jax.Array,
    local_norm_w: jax.Array,
    local_tp_rank: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    latent_dim: int,
    use_norm: bool,
    rms_eps: float,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    vmem_mode: str = "off",
) -> jax.Array:
    """Option `O4-B`: 1-shot `'dp'` RS (`cid=2`) + guarded `'tp'` `psum` (`cid=3`) + fused `RMSNorm` + slice."""
    latent_local = latent_dim // tp_size
    world_size = dp_size * tp_size
    if moe_out.shape[0] > max_dma_tokens or world_size == 1:
        routed_dp = (
            lax.psum_scatter(moe_out, DP_AXIS_NAME, scatter_dimension=0, tiled=True)
            if dp_size > 1
            else moe_out
        )
        routed_full = (
            lax.psum(routed_dp, TP_AXIS_NAME) if tp_size > 1 else routed_dp
        )
        if use_norm:
            routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)
        if tp_size > 1:
            return lax.dynamic_slice_in_dim(
                routed_full,
                local_tp_rank[0, 0] * latent_local,
                latent_local,
                axis=1,
            )
        return routed_full

    vmax = (
        max_dma_tokens
        if vmem_mode in ("vmem_chain", "vmem_src")
        else DEFAULT_VMEM_MAX_TOKENS
    )
    if dp_size > 1:
        routed_dp = kimi_guarded_psum_scatter_rows(
            moe_out,
            DP_AXIS_NAME,
            collective_id=COLLECTIVE_ID_MOE_DP_COMBINE_RS,
            recv_memory_space="auto",
            max_dma_tokens=max_dma_tokens,
            vmem_max_tokens=vmax,
        )
    else:
        routed_dp = moe_out

    if tp_size > 1:
        # Guard 1-shot `tp` `latency_psum` (`cid=3`) to `b_dp * tp_size <= 128`
        # (`b_dp <= 32` at `TP=4`, `b_dp <= 16` at `TP=8`, `b_dp <= 8` at `TP=16`),
        # falling back inside JAX to `lax.psum` for wider prefill/large-batch buckets!
        tp_psum_cap = min(max_dma_tokens, max(8, 128 // tp_size))
        routed_full = kimi_guarded_psum(
            routed_dp,
            TP_AXIS_NAME,
            collective_id=COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
            recv_memory_space="auto",
            max_dma_tokens=tp_psum_cap,
            vmem_max_tokens=vmax,
        )
    else:
        routed_full = routed_dp

    if use_norm:
        routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)

    if tp_size > 1:
        return lax.dynamic_slice_in_dim(
            routed_full,
            local_tp_rank[0, 0] * latent_local,
            latent_local,
            axis=1,
        )
    return routed_full


# =============================================================================
# Unified Single-`@jax_op` Factory (`build_kimi_moe_ep_overlap_op`) & HLO Audit
# =============================================================================


def _extract_moe_overlap_config(moe_module: Any) -> dict[str, Any]:
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        resolve_onehot_permute_threshold,
    )

    routed = moe_module.experts.routed_experts
    qm = routed.quant_method
    moe_act_str = getattr(qm, "_tpu_activation_str", None)
    assert moe_act_str is not None, (
        "[O4] process_weights_after_loading must run on routed_experts before building O4 op"
    )
    topk = int(routed.moe_config.experts_per_token)
    has_scales = (
        getattr(routed, "w13_weight_scale", None) is not None
        and getattr(routed, "w2_weight_scale", None) is not None
    )
    has_bias = (
        getattr(routed, "w13_bias", None) is not None
        and getattr(routed, "w2_bias", None) is not None
    )
    use_norm = getattr(moe_module, "routed_expert_norm", None) is not None
    rms_eps = (
        float(moe_module.routed_expert_norm.variance_epsilon)
        if use_norm
        else 1e-5
    )
    act_fn = moe_module.shared_experts.act_fn
    shared_act_kind = "situ" if hasattr(act_fn, "beta") else "silu"
    situ_beta = float(getattr(act_fn, "beta", 1.0))
    situ_linear_beta_raw = getattr(act_fn, "linear_beta", None)
    situ_linear_beta = (
        float(situ_linear_beta_raw) if situ_linear_beta_raw is not None else None
    )
    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot_thresh = int(resolve_onehot_permute_threshold())
    skip_padded = True if has_scales else bool(envs.TPU_MOE_SKIP_PADDED_TOKENS)

    return {
        "moe_act_str": str(moe_act_str),
        "topk": topk,
        "has_scales": bool(has_scales),
        "has_bias": bool(has_bias),
        "use_norm": bool(use_norm),
        "rms_eps": rms_eps,
        "shared_act_kind": shared_act_kind,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
        "use_sparse_core": use_sparse_core,
        "onehot_thresh": onehot_thresh,
        "skip_padded": skip_padded,
    }


def _make_moe_ep_overlap_jax_fn(
    mesh: Mesh,
    *,
    dp_size: int,
    tp_size: int,
    exit_fold_mode: str,
    max_dma_tokens: int,
    cfg: dict[str, Any],
) -> tuple[Callable[..., jax.Array], tuple[PartitionSpec, ...], PartitionSpec]:
    """Construct the pure JAX SPMD function + input/output `PartitionSpec`s."""
    from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

    topk = int(cfg["topk"])
    has_scales = bool(cfg["has_scales"])
    has_bias = bool(cfg["has_bias"])
    use_norm = bool(cfg["use_norm"])
    rms_eps = float(cfg["rms_eps"])
    moe_act_str = str(cfg["moe_act_str"])
    shared_act_kind = str(cfg["shared_act_kind"])
    situ_beta = float(cfg["situ_beta"])
    situ_linear_beta = cfg["situ_linear_beta"]
    use_sparse_core = bool(cfg["use_sparse_core"])
    onehot_thresh = int(cfg["onehot_thresh"])
    skip_padded = bool(cfg["skip_padded"])
    polish_trays = active_trays_mode()
    polish_vmem = vmem_chain_mode()
    polish_copy = use_copy_opt()

    world_axis: str | tuple[str, str] = (
        TP_AXIS_NAME if dp_size == 1 else (DP_AXIS_NAME, TP_AXIS_NAME)
    )

    spec_hidden = PartitionSpec(DP_AXIS_NAME, None)
    spec_rdown = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    spec_meta = PartitionSpec(DP_AXIS_NAME, None)
    spec_w_sh = PartitionSpec(None, TP_AXIS_NAME)
    spec_ep_w = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    spec_norm = PartitionSpec(None)
    spec_scalar_2d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    out_spec = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    out_sharding = NamedSharding(mesh, out_spec)

    in_specs_list: list[PartitionSpec] = [
        spec_hidden,     # 0: hidden_states [dp * B_dp, H]
        spec_rdown,      # 1: routed_down_local [dp * B_dp, tp * d_local]
        spec_meta,       # 2: packed_meta_padded [dp * B_dp, m_padded]
        spec_w_sh,       # 3: w_shared_gu [H, tp * 2 * s_inter_tp]
        spec_ep_w,       # 4: w1_sharded [dp, tp, E_local, ...]
        spec_ep_w,       # 5: w2_sharded [dp, tp, E_local, ...]
    ]
    arg_names: list[str] = [
        "hidden_states",
        "routed_down_local",
        "packed_meta",
        "w_shared_gu",
        "w1_sharded",
        "w2_sharded",
    ]
    if has_scales:
        in_specs_list.extend([spec_ep_w, spec_ep_w])  # w1_scale, w2_scale
        arg_names.extend(["w1_scale_sharded", "w2_scale_sharded"])
    if has_bias:
        in_specs_list.extend([spec_ep_w, spec_ep_w])  # w1_bias, w2_bias
        arg_names.extend(["w1_bias_sharded", "w2_bias_sharded"])
    if use_norm:
        in_specs_list.append(spec_norm)               # norm_weight [latent_dim]
        arg_names.append("norm_weight")
    in_specs_list.extend([spec_scalar_2d, spec_scalar_2d])  # experts_start_2d, tp_rank_2d
    arg_names.extend(["experts_start_2d", "tp_rank_2d"])
    in_specs = tuple(in_specs_list)
    arg_names_t = tuple(arg_names)

    def _jax_fn(*global_args: jax.Array) -> jax.Array:
        def _local(*local_args: jax.Array) -> jax.Array:
            idx = 0
            local_hidden = local_args[idx]; idx += 1
            local_rdown = local_args[idx]; idx += 1
            local_meta = local_args[idx]; idx += 1
            local_w_shared_gu = local_args[idx]; idx += 1
            local_w1 = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
            local_w2 = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
            if has_scales:
                local_w1_scale = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
                local_w2_scale = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
            else:
                local_w1_scale = None
                local_w2_scale = None
            if has_bias:
                local_w1_bias = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
                local_w2_bias = jnp.squeeze(local_args[idx], axis=(0, 1)); idx += 1
            else:
                local_w1_bias = None
                local_w2_bias = None
            if use_norm:
                local_norm_w = local_args[idx]; idx += 1
            else:
                local_norm_w = jnp.ones((1,), dtype=jnp.bfloat16)
            local_experts_start = local_args[idx][0, 0]; idx += 1
            local_tp_rank = local_args[idx]; idx += 1

            b_dp, d_local = local_rdown.shape
            latent_dim = tp_size * d_local
            b_global = dp_size * b_dp
            zero_wait_copy = polish_vmem in ("vmem_chain", "vmem_src")

            combine_kind, async_dispatch, dispatch_kind = (
                resolve_effective_exit_fold_plan(
                    exit_fold_mode,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    b_dp=b_dp,
                )
            )

            # 1. Dispatch + Shared Gate/Up Compute:
            #    - "axial" (`o4d` / `c2` / `o4d_sync`): 2-stage `TP` (`cid=0`) -> async `DP` (`cid=1`)
            #      for decode (`<= max_dma_tokens`), and 1D axial `TP` (`axis=1`) -> `DP` (`axis=0`) XLA
            #      `all_gather` (`0` 4D `swapaxes`!) for prefill (`> max_dma_tokens`).
            #    - "world" (`o4a` / `o4b` / `o4a_sync` / `o4b_sync`): flat `('dp', 'tp')` dispatch
            #      preserved for exact A/B reproducibility (with 1D axial prefill fallback on `o4b`).
            if b_global > max_dma_tokens:
                shared_act = _compute_shared_gu_and_act(
                    local_hidden,
                    local_w_shared_gu,
                    act_kind=shared_act_kind,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
                if combine_kind == "o4a" and dispatch_kind == "world":
                    m_padded = local_meta.shape[1]
                    m_per_tp = m_padded // tp_size
                    col_start = local_tp_rank[0, 0] * m_per_tp
                    local_meta_shard = lax.dynamic_slice_in_dim(
                        local_meta, col_start, m_per_tp, axis=1
                    )
                    send_shard = jnp.concatenate(
                        (local_rdown, local_meta_shard), axis=-1
                    )
                    gathered_3d = (
                        lax.all_gather(send_shard, world_axis, axis=0, tiled=False)
                        if (dp_size * tp_size > 1)
                        else send_shard[None, :, :]
                    )
                    ag_latent, ag_ids, ag_weights = _unpack_world_gathered(
                        gathered_3d,
                        dp_size=dp_size,
                        tp_size=tp_size,
                        b_dp=b_dp,
                        d_local=d_local,
                        m_per_tp=m_per_tp,
                        topk=topk,
                        copy_opt=polish_copy,
                    )
                else:
                    # Fix #3 (`D91` / `OG-4`): 1D axial XLA gathers on contiguous dims (`0` swapaxes!).
                    lat_tp = (
                        lax.all_gather(local_rdown, TP_AXIS_NAME, axis=1, tiled=True)
                        if tp_size > 1
                        else local_rdown
                    )
                    if dp_size > 1:
                        packed_dp = jnp.concatenate(
                            (lat_tp, local_meta[:, : 3 * topk]), axis=-1
                        )
                        gathered_dp = lax.all_gather(
                            packed_dp, DP_AXIS_NAME, axis=0, tiled=True
                        )
                        ag_latent, ag_ids, ag_weights = _unpack_axial_gathered(
                            gathered_dp, latent_dim=latent_dim, topk=topk
                        )
                    else:
                        ag_latent = lat_tp
                        ag_ids, ag_weights = _unpack_meta_columns(
                            local_meta, topk=topk
                        )
            elif dispatch_kind == "axial":
                if dp_size > 1:
                    lat_tp = (
                        kimi_guarded_all_gather_cols(
                            local_rdown,
                            TP_AXIS_NAME,
                            collective_id=COLLECTIVE_ID_MOE_TP_LATENT_AG,
                            recv_memory_space="auto",
                            max_dma_tokens=max_dma_tokens,
                        )
                        if tp_size > 1
                        else local_rdown
                    )
                    dp_orig_lanes = latent_dim + 3 * topk
                    packed_dp = jnp.concatenate(
                        (lat_tp, local_meta[:, : 3 * topk]), axis=-1
                    )
                    if async_dispatch:
                        rx_state, start_tok = async_ep_dispatch_start(
                            packed_dp,
                            DP_AXIS_NAME,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                        )
                        hidden_after_start, start_tok = lax.optimization_barrier(
                            (local_hidden, start_tok)
                        )
                        shared_act = _compute_shared_gu_and_act(
                            hidden_after_start,
                            local_w_shared_gu,
                            act_kind=shared_act_kind,
                            situ_beta=situ_beta,
                            situ_linear_beta=situ_linear_beta,
                        )
                        wait_tok, shared_act = lax.optimization_barrier(
                            (start_tok, shared_act)
                        )
                        gathered_dp_3d = async_ep_dispatch_wait(
                            rx_state,
                            wait_tok,
                            DP_AXIS_NAME,
                            rows=b_dp,
                            orig_lanes=dp_orig_lanes,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                            zero_extra_copy=zero_wait_copy,
                        )
                        gathered_dp = gathered_dp_3d.reshape(
                            dp_size * b_dp, dp_orig_lanes
                        )
                    else:
                        gathered_dp = kimi_guarded_all_gather_rows(
                            packed_dp,
                            DP_AXIS_NAME,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                            recv_memory_space="hbm",
                            max_dma_tokens=max_dma_tokens,
                        )
                        gathered_dp, local_hidden_sync = lax.optimization_barrier(
                            (gathered_dp, local_hidden)
                        )
                        shared_act = _compute_shared_gu_and_act(
                            local_hidden_sync,
                            local_w_shared_gu,
                            act_kind=shared_act_kind,
                            situ_beta=situ_beta,
                            situ_linear_beta=situ_linear_beta,
                        )
                    ag_latent, ag_ids, ag_weights = _unpack_axial_gathered(
                        gathered_dp, latent_dim=latent_dim, topk=topk
                    )
                elif tp_size > 1:
                    if async_dispatch:
                        rx_state, start_tok = async_ep_dispatch_start(
                            local_rdown,
                            TP_AXIS_NAME,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                        )
                        hidden_after_start, start_tok = lax.optimization_barrier(
                            (local_hidden, start_tok)
                        )
                        shared_act = _compute_shared_gu_and_act(
                            hidden_after_start,
                            local_w_shared_gu,
                            act_kind=shared_act_kind,
                            situ_beta=situ_beta,
                            situ_linear_beta=situ_linear_beta,
                        )
                        wait_tok, shared_act = lax.optimization_barrier(
                            (start_tok, shared_act)
                        )
                        lat_tp_3d = async_ep_dispatch_wait(
                            rx_state,
                            wait_tok,
                            TP_AXIS_NAME,
                            rows=b_dp,
                            orig_lanes=d_local,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                            zero_extra_copy=zero_wait_copy,
                        )
                    else:
                        lat_tp_2d = kimi_guarded_all_gather_rows(
                            local_rdown,
                            TP_AXIS_NAME,
                            collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                            recv_memory_space="hbm",
                            max_dma_tokens=max_dma_tokens,
                        )
                        lat_tp_3d = lat_tp_2d.reshape(tp_size, b_dp, d_local)
                        lat_tp_3d, local_hidden_sync = lax.optimization_barrier(
                            (lat_tp_3d, local_hidden)
                        )
                        shared_act = _compute_shared_gu_and_act(
                            local_hidden_sync,
                            local_w_shared_gu,
                            act_kind=shared_act_kind,
                            situ_beta=situ_beta,
                            situ_linear_beta=situ_linear_beta,
                        )
                    ag_latent = (
                        lat_tp_3d.reshape(1, latent_dim)
                        if b_dp == 1
                        else jnp.swapaxes(lat_tp_3d, 0, 1).reshape(b_dp, latent_dim)
                    )
                    ag_ids, ag_weights = _unpack_meta_columns(
                        local_meta, topk=topk
                    )
                else:
                    shared_act = _compute_shared_gu_and_act(
                        local_hidden,
                        local_w_shared_gu,
                        act_kind=shared_act_kind,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                    )
                    ag_latent = local_rdown
                    ag_ids, ag_weights = _unpack_meta_columns(
                        local_meta, topk=topk
                    )
            else:
                m_padded = local_meta.shape[1]
                m_per_tp = m_padded // tp_size
                col_start = local_tp_rank[0, 0] * m_per_tp
                local_meta_shard = lax.dynamic_slice_in_dim(
                    local_meta, col_start, m_per_tp, axis=1
                )
                send_shard = jnp.concatenate(
                    (local_rdown, local_meta_shard), axis=-1
                )
                orig_lanes = d_local + m_per_tp
                if async_dispatch and (dp_size * tp_size > 1):
                    rx_state, start_tok = async_ep_dispatch_start(
                        send_shard,
                        world_axis,
                        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                    )
                    # Topological HLO fence: guarantees XLA schedules
                    # [ep_dispatch_start] -> [w_shared_gu + act] -> [ep_dispatch_wait]!
                    hidden_after_start, start_tok = lax.optimization_barrier(
                        (local_hidden, start_tok)
                    )
                    shared_act = _compute_shared_gu_and_act(
                        hidden_after_start,
                        local_w_shared_gu,
                        act_kind=shared_act_kind,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                    )
                    wait_tok, shared_act = lax.optimization_barrier(
                        (start_tok, shared_act)
                    )
                    gathered_3d = async_ep_dispatch_wait(
                        rx_state,
                        wait_tok,
                        world_axis,
                        rows=b_dp,
                        orig_lanes=orig_lanes,
                        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                        zero_extra_copy=zero_wait_copy,
                    )
                else:
                    # Synchronous dispatch ablation (`o4a_sync` / `o4b_sync`) or 1-rank edge case:
                    gathered_2d = kimi_guarded_all_gather_rows(
                        send_shard,
                        world_axis,
                        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                        recv_memory_space="hbm",
                        max_dma_tokens=max_dma_tokens,
                    )
                    gathered_3d = gathered_2d.reshape(
                        dp_size * tp_size, b_dp, orig_lanes
                    )
                    gathered_3d, local_hidden_sync = lax.optimization_barrier(
                        (gathered_3d, local_hidden)
                    )
                    shared_act = _compute_shared_gu_and_act(
                        local_hidden_sync,
                        local_w_shared_gu,
                        act_kind=shared_act_kind,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                    )

                ag_latent, ag_ids, ag_weights = _unpack_world_gathered(
                    gathered_3d,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    b_dp=b_dp,
                    d_local=d_local,
                    m_per_tp=m_per_tp,
                    topk=topk,
                    copy_opt=polish_copy,
                )

            # 2. Routed MoE Compute (`fused_decode_moe_v6e_planar_t_b11` for <=64, or
            #    <=256 multi-tray under `TPU_KIMI_OPT_SPEC_KDA=o6b`; `moe_gmm` above)
            moe_out = fused_moe_func(
                ag_latent,
                local_w1,
                local_w2,
                local_w1_scale,
                local_w2_scale,
                local_w1_bias,
                local_w2_bias,
                ag_weights,
                ag_ids,
                local_experts_start,
                topk=topk,
                activation=moe_act_str,
                use_ep=True,
                use_sparse_core=use_sparse_core,
                onehot_moe_permute_threshold=onehot_thresh,
                rhs_quant_dtype=None,
                skip_padded_tokens=skip_padded,
                token_trays_mode=polish_trays,
                rank3_scales=polish_copy,
            )

            # 3. Mathematically Exact Combine + `RMSNorm(latent_dim)` + `[latent_local]` Exit-Fold
            if combine_kind == "o4a":
                routed_local = o4a_owner_combine_rms_seam_split(
                    moe_out,
                    local_norm_w,
                    local_tp_rank,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    b_dp=b_dp,
                    latent_dim=latent_dim,
                    use_norm=use_norm,
                    rms_eps=rms_eps,
                    max_dma_tokens=max_dma_tokens,
                    collective_id=COLLECTIVE_ID_MOE_OWNER_SEAM_SPLIT,
                    vmem_mode=polish_vmem,
                )
            elif combine_kind == "o4b":
                routed_local = o4b_dp_rs_tp_psum_fused_rms_slice(
                    moe_out,
                    local_norm_w,
                    local_tp_rank,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    latent_dim=latent_dim,
                    use_norm=use_norm,
                    rms_eps=rms_eps,
                    max_dma_tokens=max_dma_tokens,
                    vmem_mode=polish_vmem,
                )
            else:
                raise ValueError(f"Unknown combine_kind={combine_kind!r}")

            # Concatenate `(shared_act, routed_local)` inside `@jax_op` so `packed_moe_out`
            # receives a single contiguous `[B_dp, s_inter_tp + latent_local]` tensor!
            return jnp.concatenate((shared_act, routed_local), axis=-1)

        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_vma=False,
        )(*global_args)
        return lax.with_sharding_constraint(out, out_sharding)

    _jax_fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                nm, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
            )
            for nm in arg_names_t
        ],
        return_annotation=jax.Array,
    )
    return _jax_fn, in_specs, out_spec, arg_names_t


def build_kimi_moe_ep_overlap_op(
    mesh: Mesh,
    moe_module: Any,
    *,
    exit_fold_mode: str,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "o4",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Any], torch.Tensor]:
    """Build (or reuse cached) unified `kimi_moe_ep_overlap` `@jax_op` across layers."""
    # Imported here: kernels/ must not import models/ at module load (cycle).
    from vllm_torchtpu.models.vllm.kimi_k3.dma_collective_ops import (
        _REGISTERED_TORCH_OPS,
        _mesh_shape_and_ids,
        _sanitize_tag,
        format_pallas_dma_op_name,
    )

    norm_mode = normalize_exit_fold_mode(exit_fold_mode)
    if norm_mode == "off":
        raise ValueError("build_kimi_moe_ep_overlap_op requires an active exit_fold_mode")

    dp_size, tp_size, dev_ids = _mesh_shape_and_ids(mesh)
    cfg = _extract_moe_overlap_config(moe_module)
    cfg_sig = (
        cfg["moe_act_str"],
        cfg["topk"],
        cfg["has_scales"],
        cfg["has_bias"],
        cfg["use_norm"],
        cfg["rms_eps"],
        cfg["shared_act_kind"],
        cfg["situ_beta"],
        cfg["situ_linear_beta"],
        cfg["use_sparse_core"],
        cfg["onehot_thresh"],
        cfg["skip_padded"],
    )

    combine_kind_sig, _, dispatch_kind_sig = resolve_effective_exit_fold_plan(
        norm_mode, dp_size=dp_size, tp_size=tp_size, b_dp=1
    )
    impl_tag = (
        f"_{dispatch_kind_sig}_{combine_kind_sig}"
        if dispatch_kind_sig == "axial"
        else ""
    )
    spec_tag = (
        f"{norm_mode}{impl_tag}_k{cfg['topk']}_{_sanitize_tag(cfg['shared_act_kind'])}"
        f"_q{int(cfg['has_scales'])}_n{int(cfg['use_norm'])}"
    )
    op_name = format_pallas_dma_op_name(
        kind="moe_ep_overlap",
        spec_tag=spec_tag,
        axis_name=("dp", "tp"),
        collective_id=COLLECTIVE_ID_MOE_OWNER_SEAM_SPLIT
        if combine_kind_sig == "o4a"
        else COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
        recv_memory_space="hbm",
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    raw_key = ("moe_ep_overlap_raw", op_name, dev_ids, cfg_sig)

    if raw_key in _REGISTERED_TORCH_OPS:
        raw_op = _REGISTERED_TORCH_OPS[raw_key]
    else:
        from torch.tpu.pallas import jax_op

        jax_fn, in_specs, out_spec, arg_names_t = _make_moe_ep_overlap_jax_fn(
            mesh,
            dp_size=dp_size,
            tp_size=tp_size,
            exit_fold_mode=norm_mode,
            max_dma_tokens=max_dma_tokens,
            cfg=cfg,
        )
        raw_op = jax_op(
            op_name,
            jax_fn,
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=(out_spec,),
        )

        def _fake_fn(*fake_args: torch.Tensor) -> torch.Tensor:
            hidden = fake_args[0]
            rdown = fake_args[1]
            w_sh_gu = fake_args[3]
            s_inter_tp = w_sh_gu.shape[1] // 2
            lat_local = rdown.shape[1]
            return hidden.new_empty((hidden.shape[0], s_inter_tp + lat_local))

        _fake_fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [
                inspect.Parameter(
                    nm, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor
                )
                for nm in arg_names_t
            ],
            return_annotation=torch.Tensor,
        )
        raw_op.register_fake(_fake_fn)
        _REGISTERED_TORCH_OPS[raw_key] = raw_op  # type: ignore[assignment]

    has_scales = bool(cfg["has_scales"])
    has_bias = bool(cfg["has_bias"])
    use_norm = bool(cfg["use_norm"])

    def _call(
        hidden_states: torch.Tensor,
        routed_down_local: torch.Tensor,
        packed_meta: torch.Tensor,
        layer_moe: Any,
    ) -> torch.Tensor:
        meta_cols = int(packed_meta.shape[1])
        m_per_tp = (meta_cols + tp_size - 1) // tp_size
        pad_m = m_per_tp * tp_size - meta_cols
        padded_meta = (
            torch.nn.functional.pad(packed_meta, (0, pad_m))
            if pad_m > 0
            else packed_meta
        )
        routed = layer_moe.experts.routed_experts
        w1_5d = routed.w13_weight.unsqueeze(0).unsqueeze(0)
        w2_5d = routed.w2_weight.unsqueeze(0).unsqueeze(0)

        call_args: list[torch.Tensor] = [
            hidden_states,
            routed_down_local,
            padded_meta,
            layer_moe.w_shared_gu,
            w1_5d,
            w2_5d,
        ]
        if has_scales:
            call_args.append(routed.w13_weight_scale.unsqueeze(0).unsqueeze(0))
            call_args.append(routed.w2_weight_scale.unsqueeze(0).unsqueeze(0))
        if has_bias:
            call_args.append(routed.w13_bias.unsqueeze(0).unsqueeze(0))
            call_args.append(routed.w2_bias.unsqueeze(0).unsqueeze(0))
        if use_norm:
            call_args.append(layer_moe.routed_expert_norm.weight)
        call_args.append(routed._experts_start.reshape(1, 1))
        call_args.append(layer_moe._o2_tp_rank_buf)
        return raw_op(*call_args)

    _call._qualname = op_name  # type: ignore[attr-defined]
    return _call


def _audit_hlo_text(
    hlo_txt: str,
    *,
    mode: str,
    dp_size: int,
    b_dp: int,
    is_post_scheduler_hlo: bool,
) -> dict[str, Any]:
    """Parse computation instructions in scheduled order from HLO text."""
    lines = hlo_txt.splitlines()
    if is_post_scheduler_hlo:
        entry_start = 0
        for idx, ln in enumerate(lines):
            if ln.lstrip().startswith("ENTRY "):
                entry_start = idx
                break
        scan_lines = lines[entry_start:]
    else:
        # Pre-optimization GSPMD HLO emits `%shmap_body...` as a sub-computation called by `ENTRY`:
        scan_lines = lines

    idx_start = -1
    idx_dot = -1
    idx_wait = -1
    ar_count = 0
    ag_count = 0
    rs_count = 0
    has_o4a_combine = False
    has_o4a_seam = False
    has_tp_latent_ag_cid0 = False
    has_dp_combine_rs_cid2 = False
    has_tp_routed_psum_cid3 = False
    transpose_4d_world_count = 0
    o4a_retile_128_count = 0

    world_4d_re = re.compile(rf"\bbf16\[{dp_size},\d+,\d+,\d+\]")

    for i, ln in enumerate(scan_lines):
        s = ln.strip()
        if (
            "kimi_ep_dispatch_start" in s
            and "custom-call(" in s
            and "s32[1]" in s
            and idx_start < 0
        ) or ("kimi_ep_dispatch_start" in s and not is_post_scheduler_hlo and idx_start < 0):
            idx_start = i
        if "kimi_ep_dispatch_wait" in s and idx_wait < 0:
            idx_wait = i
        if "kimi_o4a_owner_combine" in s:
            has_o4a_combine = True
        if "kimi_o4a_seam_exit_split" in s:
            has_o4a_seam = True
        if "latency_all_gather_cols" in s:
            has_tp_latent_ag_cid0 = True
        if "latency_psum_scatter_rows" in s:
            has_dp_combine_rs_cid2 = True
        if "latency_psum/" in s or "latency_psum:" in s:
            has_tp_routed_psum_cid3 = True
        if dp_size > 1 and ("transpose(" in s or " copy(" in s) and world_4d_re.search(s):
            transpose_4d_world_count += 1
        if "bf16[128,1,2,1792]" in s:
            o4a_retile_128_count += 1
        is_dot = (
            ("kimi_shared_gu_dot" in s and (" dot(" in s or "dot_general" in s))
            or ((" = bf16[" in s or " = f32[" in s) and " dot(" in s)
            or ("stablehlo.dot_general" in s)
        )
        if is_dot and idx_start >= 0 and idx_dot < 0:
            idx_dot = i
        if " all-reduce(" in s or " all-reduce-start(" in s or "stablehlo.all_reduce" in s:
            ar_count += 1
        if " all-gather(" in s or " all-gather-start(" in s or "stablehlo.all_gather" in s:
            ag_count += 1
        if (
            " reduce-scatter(" in s
            or " reduce-scatter-start(" in s
            or "stablehlo.reduce_scatter" in s
        ):
            rs_count += 1

    return {
        "mode": mode,
        "is_post_scheduler_hlo": is_post_scheduler_hlo,
        "b_dp": b_dp,
        "b_global": dp_size * b_dp,
        "entry_line_count": len(scan_lines),
        "idx_dispatch_start": idx_start,
        "idx_shared_dot": idx_dot,
        "idx_dispatch_wait": idx_wait,
        "start_before_shared_dot": (idx_start >= 0 and idx_dot > idx_start),
        "shared_dot_before_wait": (idx_dot >= 0 and idx_wait > idx_dot),
        "all_reduce_count": ar_count,
        "all_gather_count": ag_count,
        "reduce_scatter_count": rs_count,
        "has_xla_fallback_collective": (ar_count + ag_count + rs_count) > 0,
        "has_o4a_owner_combine": has_o4a_combine,
        "has_o4a_seam_exit_split": has_o4a_seam,
        "has_tp_latent_ag_cid0": has_tp_latent_ag_cid0,
        "has_dp_combine_rs_cid2": has_dp_combine_rs_cid2,
        "has_tp_routed_psum_cid3": has_tp_routed_psum_cid3,
        "transpose_4d_world_count": transpose_4d_world_count,
        "o4a_retile_128_count": o4a_retile_128_count,
    }


def inspect_kimi_moe_ep_overlap_hlo(
    moe_module: Any,
    mesh: Mesh,
    *,
    b_dp: int,
    exit_fold_mode: str | None = None,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> dict[str, Any]:
    """Lower (or compile when `mesh` holds real `jax.Device`s) SPMD JAX HLO and audit ordering."""
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from vllm_torchtpu.models.vllm.kimi_k3.dma_collective_ops import (
        _mesh_shape_and_ids,
    )

    mode = normalize_exit_fold_mode(
        envs.TPU_KIMI_OPT_EXIT_FOLD if exit_fold_mode is None else exit_fold_mode
    )
    dp_size, tp_size, _ = _mesh_shape_and_ids(mesh)
    cfg = _extract_moe_overlap_config(moe_module)
    jax_fn, in_specs, out_spec, _ = _make_moe_ep_overlap_jax_fn(
        mesh,
        dp_size=dp_size,
        tp_size=tp_size,
        exit_fold_mode=mode,
        max_dma_tokens=max_dma_tokens,
        cfg=cfg,
    )

    routed = moe_module.experts.routed_experts
    h_dim = int(moe_module.w_shared_gu.shape[0])
    d_local = int(moe_module.w_routed_down_tp.shape[1])
    topk = int(cfg["topk"])
    meta_cols = 3 * topk
    m_per_tp = (meta_cols + tp_size - 1) // tp_size
    m_padded = m_per_tp * tp_size
    dev = moe_module.w_shared_gu.device

    sample_torch_args: list[torch.Tensor] = [
        torch.empty((b_dp, h_dim), dtype=torch.bfloat16, device=dev),
        torch.empty((b_dp, d_local), dtype=torch.bfloat16, device=dev),
        torch.empty((b_dp, m_padded), dtype=torch.bfloat16, device=dev),
        moe_module.w_shared_gu,
        routed.w13_weight.unsqueeze(0).unsqueeze(0),
        routed.w2_weight.unsqueeze(0).unsqueeze(0),
    ]
    if cfg["has_scales"]:
        sample_torch_args.append(routed.w13_weight_scale.unsqueeze(0).unsqueeze(0))
        sample_torch_args.append(routed.w2_weight_scale.unsqueeze(0).unsqueeze(0))
    if cfg["has_bias"]:
        sample_torch_args.append(routed.w13_bias.unsqueeze(0).unsqueeze(0))
        sample_torch_args.append(routed.w2_bias.unsqueeze(0).unsqueeze(0))
    if cfg["use_norm"]:
        sample_torch_args.append(moe_module.routed_expert_norm.weight)
    sample_torch_args.append(routed._experts_start.reshape(1, 1))
    sample_torch_args.append(moe_module._o2_tp_rank_buf)

    jax_avals = pallas_impl.jax_placeholders(
        sample_torch_args, mesh=mesh, partition_specs=in_specs
    )
    in_shardings = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sharding = NamedSharding(mesh, out_spec)
    lowered = jax.jit(jax_fn, in_shardings=in_shardings, out_shardings=out_sharding).lower(
        *jax_avals
    )
    has_real_jax_devices = all(isinstance(d, jax.Device) for d in mesh.devices.flat)
    if has_real_jax_devices:
        hlo_txt = lowered.compile().as_text()
    else:
        hlo_txt = lowered.as_text("stablehlo")

    return _audit_hlo_text(
        hlo_txt,
        mode=mode,
        dp_size=dp_size,
        b_dp=b_dp,
        is_post_scheduler_hlo=has_real_jax_devices,
    )


def compile_post_scheduler_entry_hlo_audit(
    *,
    dp_size: int,
    tp_size: int,
    b_dp: int,
    exit_fold_mode: str,
    hidden_size: int = 3584,
    latent_dim: int = 1792,
    shared_inter_size: int = 3584,
    num_experts: int = 32,
    moe_inter_size: int = 256,
    topk: int = 16,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> dict[str, Any]:
    """Compile `kimi_moe_ep_overlap` with real `jax.devices()` and audit post-scheduler ENTRY HLO."""
    import numpy as np

    devs = jax.devices()
    world = dp_size * tp_size
    if len(devs) < world:
        raise RuntimeError(f"Need {world} real JAX devices for post-scheduler HLO compile, got {len(devs)}")
    mesh = Mesh(np.array(devs[:world]).reshape(dp_size, tp_size), (DP_AXIS_NAME, TP_AXIS_NAME))
    mode = normalize_exit_fold_mode(exit_fold_mode)
    cfg: dict[str, Any] = {
        "moe_act_str": "silu",
        "topk": topk,
        "has_scales": False,
        "has_bias": False,
        "use_norm": True,
        "rms_eps": 1e-5,
        "shared_act_kind": "situ",
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "use_sparse_core": False,
        "onehot_thresh": 1024,
        "skip_padded": True,
    }
    jax_fn, in_specs, out_spec, _ = _make_moe_ep_overlap_jax_fn(
        mesh,
        dp_size=dp_size,
        tp_size=tp_size,
        exit_fold_mode=mode,
        max_dma_tokens=max_dma_tokens,
        cfg=cfg,
    )
    in_shardings = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sharding = NamedSharding(mesh, out_spec)

    d_local = latent_dim // tp_size
    meta_cols = 3 * topk
    m_per_tp = (meta_cols + tp_size - 1) // tp_size
    m_padded = m_per_tp * tp_size
    s_gu_global = 2 * shared_inter_size
    e_local = num_experts // world

    global_shapes_dtypes = [
        ((dp_size * b_dp, hidden_size), jnp.bfloat16),
        ((dp_size * b_dp, latent_dim), jnp.bfloat16),
        ((dp_size * b_dp, m_padded), jnp.bfloat16),
        ((hidden_size, s_gu_global), jnp.bfloat16),
        ((dp_size, tp_size, e_local, latent_dim, 2 * moe_inter_size), jnp.bfloat16),
        ((dp_size, tp_size, e_local, moe_inter_size, latent_dim), jnp.bfloat16),
        ((latent_dim,), jnp.bfloat16),
        ((dp_size, tp_size), jnp.int32),
        ((dp_size, tp_size), jnp.int32),
    ]
    jax_avals = [
        jax.ShapeDtypeStruct(shape, dt, sharding=sh)
        for (shape, dt), sh in zip(global_shapes_dtypes, in_shardings, strict=True)
    ]
    compiled = (
        jax.jit(jax_fn, in_shardings=in_shardings, out_shardings=out_sharding)
        .lower(*jax_avals)
        .compile()
    )
    hlo_txt = compiled.as_text()
    return _audit_hlo_text(
        hlo_txt,
        mode=mode,
        dp_size=dp_size,
        b_dp=b_dp,
        is_post_scheduler_hlo=True,
    )
