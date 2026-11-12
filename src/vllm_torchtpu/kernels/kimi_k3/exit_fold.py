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
"""Pallas and JAX pieces of the Kimi K3 EP-overlap MoE layer exit.

`layers/adapter/custom_ops/kimi_moe_ep_overlap_op.py` composes these into one
`jax_op`
(a `jax.shard_map` over the `('dp', 'tp')` mesh) that runs a whole Kimi K3
MoE block for expert parallelism with its Pallas DMA semaphores held
internally:

- `async_ep_dispatch_start` / `async_ep_dispatch_wait`: a non-blocking
  one-shot remote-DMA all-gather split into a start and a wait kernel, and
  `dispatch_overlapping_shared_experts`, which runs the shared-expert
  gate/up matmul and gated activation between the two;
  `lax.optimization_barrier` edges pin that order (start -> shared experts
  -> wait).
- `unpack_meta_columns` / `unpack_axial_gathered`: split the gathered rows
  into the routed latent and the byte-packed routing metadata.
- `hierarchical_combine_rms_slice`: one-shot `'dp'` reduce-scatter, `'tp'`
  psum guarded to small widths (XLA collectives above the caps), the exact
  `RMSNorm(latent_dim)` of the routed output, and the slice of this rank's
  `latent_dim / tp` columns.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax import lax
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.one_shot_dma import (
    DEFAULT_MAX_DMA_TOKENS,
    barrier_and_allocate,
    call_kernel,
    guarded_psum,
    guarded_psum_scatter_rows,
    normalize_axis_name,
    seam_pieces_pack,
    seam_pieces_unpack,
)

Ref = Any

# Barrier `collective_id` of each DMA collective of the layer exit.
# Collectives that can be in flight at the same time use different ids:
#   0 TP gather of the routed latent columns
#   1 DP dispatch gather of the latent and the routing metadata
#   2 DP reduce-scatter of the routed expert outputs
#   3 TP psum of the routed expert outputs
COLLECTIVE_ID_MOE_TP_LATENT_AG = 0
COLLECTIVE_ID_MOE_DP_DISPATCH_AG = 1
COLLECTIVE_ID_MOE_DP_COMBINE_RS = 2
COLLECTIVE_ID_MOE_TP_ROUTED_PSUM = 3

# The one-shot TP psum serves at most this many rows summed over the TP axis,
# and never fewer than `_TP_PSUM_MIN_ROWS` per rank; wider buckets take
# `lax.psum`.
_TP_PSUM_MAX_TOTAL_ROWS = 128
_TP_PSUM_MIN_ROWS = 8


# =============================================================================
# Exact JAX RMSNorm and shared-expert activations (`SituAndMul` / `SiluAndMul`)
# =============================================================================


def _jax_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """RMSNorm matching `vllm.ir.ops.layernorm.rms_norm` on TPU (<= 1 bf16 ULP).

    The normalized value is rounded to bf16 before the weight multiply.
    """
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


def compute_shared_gu_and_act(
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
# Split non-blocking Pallas EP dispatch (`start` + `wait`)
# =============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AsyncDispatchState:
    """Pinned HBM send/recv buffers + DMA semaphores kept inside the `jax_op`."""

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
    """Barrier-allocate buffers and launch a non-blocking one-shot DMA gather.

    Returns the in-flight state and a token for `async_ep_dispatch_wait`. On
    a single-rank axis the state is `send_shard[None]` and no DMA is issued.
    """
    axis_key = normalize_axis_name(axis_name)
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
            stage_send = pltpu.make_async_copy(
                src_ref=x_ref, dst_ref=state_ref.x_send, sem=local_sem
            )
            stage_local_recv = pltpu.make_async_copy(
                src_ref=x_ref,
                dst_ref=state_ref.x_recv.at[pl.ds(my_id, 1)],
                sem=local_sem,
            )
            stage_send.start()
            stage_local_recv.start()
            stage_send.wait()
            stage_local_recv.wait()

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
) -> jax.Array:
    """Wait for an in-flight EP dispatch, drain its sends, and unpack the rows.

    Returns `[world, rows, orig_lanes]`, unpacked straight from the receive
    buffer.
    """
    axis_key = normalize_axis_name(axis_name)
    world_size = jax.lax.axis_size(axis_key)
    if world_size == 1:
        assert isinstance(state, jax.Array)
        return state

    assert isinstance(state, AsyncDispatchState)

    def _wait_and_drain(state_ref: AsyncDispatchState) -> None:
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

    def _wait_kernel_inplace(
        wait_tok_ref: Any,
        state_ref: AsyncDispatchState,
    ) -> None:
        _ = wait_tok_ref[0]
        _wait_and_drain(state_ref)

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


def unpack_meta_columns(
    meta_2d: jax.Array,
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Unpack byte-valued bf16 metadata `[rows, >= 3 * topk]`.

    Columns `[0, topk)` and `[topk, 2 * topk)` hold the low and high bytes of
    the int16 expert ids (byte values survive bf16 flush-to-zero), columns
    `[2 * topk, 3 * topk)` the routing weights. Returns `(ids, weights)`.
    """
    ag_meta = meta_2d[:, : 3 * topk]
    low_i32 = ag_meta[:, :topk].astype(jnp.int32)
    high_i32 = ag_meta[:, topk : 2 * topk].astype(jnp.int32)
    ag_ids = (((high_i32 << 8) | low_i32) << 16) >> 16
    ag_weights = ag_meta[:, 2 * topk : 3 * topk]
    return ag_ids, ag_weights


def unpack_axial_gathered(
    gathered_dp_2d: jax.Array,
    *,
    latent_dim: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split `[dp * B_dp, latent_dim + 3 * topk]` into `(latent, ids, weights)`."""
    ag_latent = gathered_dp_2d[:, :latent_dim]
    ag_ids, ag_weights = unpack_meta_columns(gathered_dp_2d[:, latent_dim:], topk=topk)
    return ag_latent, ag_ids, ag_weights


# =============================================================================
# Hierarchical combine: DP reduce-scatter + TP psum + RMSNorm + column slice
# =============================================================================


def hierarchical_combine_rms_slice(
    moe_out: jax.Array,
    local_norm_w: jax.Array,
    local_tp_rank: jax.Array,
    *,
    dp_axis_name: str,
    tp_axis_name: str,
    dp_size: int,
    tp_size: int,
    latent_dim: int,
    use_norm: bool,
    rms_eps: float,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> jax.Array:
    """Combine the routed expert partial sums of every rank.

    One-shot reduce-scatter of the rows over `dp_axis_name`, psum over
    `tp_axis_name`, exact `RMSNorm(latent_dim)` (when `use_norm`), then this
    rank's `latent_dim / tp_size` columns. Batches above `max_dma_tokens`
    (and a single-rank mesh) use XLA collectives; the DMA collectives receive
    into VMEM.
    """
    latent_local = latent_dim // tp_size
    world_size = dp_size * tp_size
    if moe_out.shape[0] > max_dma_tokens or world_size == 1:
        routed_dp = (
            lax.psum_scatter(moe_out, dp_axis_name, scatter_dimension=0, tiled=True)
            if dp_size > 1
            else moe_out
        )
        routed_full = lax.psum(routed_dp, tp_axis_name) if tp_size > 1 else routed_dp
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

    if dp_size > 1:
        routed_dp = guarded_psum_scatter_rows(
            moe_out,
            dp_axis_name,
            collective_id=COLLECTIVE_ID_MOE_DP_COMBINE_RS,
            recv_memory_space="auto",
            max_dma_tokens=max_dma_tokens,
            vmem_max_tokens=max_dma_tokens,
        )
    else:
        routed_dp = moe_out

    if tp_size > 1:
        tp_psum_cap = min(
            max_dma_tokens, max(_TP_PSUM_MIN_ROWS, _TP_PSUM_MAX_TOTAL_ROWS // tp_size)
        )
        routed_full = guarded_psum(
            routed_dp,
            tp_axis_name,
            collective_id=COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
            recv_memory_space="auto",
            max_dma_tokens=tp_psum_cap,
            vmem_max_tokens=max_dma_tokens,
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
# Dispatch overlapped by the shared experts
# =============================================================================


def dispatch_overlapping_shared_experts(
    send_shard: jax.Array,
    axis_name: str,
    local_hidden: jax.Array,
    local_w_shared_gu: jax.Array,
    *,
    rows: int,
    orig_lanes: int,
    act_kind: str,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Async EP dispatch over `axis_name` with the shared experts in between.

    The optimization barriers make the shared-expert inputs depend on the
    dispatch start token and the wait token depend on the shared-expert
    output, so XLA keeps the order start -> shared experts -> wait.

    Returns `(gathered [world, rows, orig_lanes], shared_act)`.
    """
    rx_state, start_tok = async_ep_dispatch_start(
        send_shard,
        axis_name,
        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    )
    hidden_after_start, start_tok = lax.optimization_barrier((local_hidden, start_tok))
    shared_act = compute_shared_gu_and_act(
        hidden_after_start,
        local_w_shared_gu,
        act_kind=act_kind,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    wait_tok, shared_act = lax.optimization_barrier((start_tok, shared_act))
    gathered = async_ep_dispatch_wait(
        rx_state,
        wait_tok,
        axis_name,
        rows=rows,
        orig_lanes=orig_lanes,
        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    )
    return gathered, shared_act
