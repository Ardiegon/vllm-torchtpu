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
`jax_op` (a `jax.shard_map` over the `('dp', 'tp')` mesh) that runs a whole
Kimi K3 MoE block for expert parallelism with its Pallas DMA semaphores held
internally:

- `compute_shared_gu_and_act`: the shared-expert gate/up matmul and the Kimi
  `SituAndMul` activation.
- `async_ep_dispatch_start` / `async_ep_dispatch_wait`: a non-blocking
  one-shot remote-DMA all-gather split into a start and a wait kernel, and
  `dispatch_overlapping_shared_experts`, which runs the shared-expert
  gate/up matmul and `SituAndMul` activation between the two;
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
    DMA_LATENCY_METADATA,
    barrier_and_allocate,
    call_kernel,
    guarded_psum,
    latency_psum_scatter_rows,
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
# `lax.psum`. Empirical values from TPU v6e (on TP=16, 128 // 16 = 8 keeps
# single-token decode on the one-shot DMA psum while multi-token verify batches
# take XLA's ring psum); not derived from a general cost model.
_TP_PSUM_MAX_TOTAL_ROWS = 128
_TP_PSUM_MIN_ROWS = 8

# =============================================================================
# Exact JAX RMSNorm and the shared-expert activation (`SituAndMul`)
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


def _apply_situ_and_mul(
    x: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    """Apply Kimi `SituAndMul` to `[B_dp, 2 * inter_tp]`."""
    gate, up = jnp.split(x, 2, axis=-1)
    gf = gate.astype(jnp.float32)
    beta = float(situ_beta)
    gate_act = (beta * jnp.tanh(gf / beta) * jax.nn.sigmoid(gf)).astype(jnp.bfloat16)
    if situ_linear_beta is not None:
        uf = up.astype(jnp.float32)
        lbeta = float(situ_linear_beta)
        up_act = (lbeta * jnp.tanh(uf / lbeta)).astype(jnp.bfloat16)
    else:
        up_act = up
    return (gate_act * up_act).astype(jnp.bfloat16)


def compute_shared_gu_and_act(
    local_hidden: jax.Array,
    local_w_shared_gu: jax.Array,
    *,
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
        return _apply_situ_and_mul(
            shared_gu,
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
    collective_id: int,
) -> tuple[AsyncDispatchState, jax.Array]:
    """Barrier-allocate buffers and launch a non-blocking one-shot DMA gather.

    Returns the in-flight state and a token for `async_ep_dispatch_wait`. The
    axis must have several ranks.
    """
    axis_key = normalize_axis_name(axis_name)
    world_size = jax.lax.axis_size(axis_key)
    assert world_size > 1, f"axis {axis_key} has one rank; nothing to dispatch"

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
        latency_metadata=DMA_LATENCY_METADATA,
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
        )
        live_state, start_tok = call_kernel(
            _start_kernel,
            args=(x_packed,),
            inout=(init_state,),
            output=(jnp.zeros((1,), dtype=jnp.int32),),
            scoped={"local_sem": pltpu.SemaphoreType.DMA},
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
            name="kimi_ep_dispatch_start",
        )
    return live_state, start_tok


def async_ep_dispatch_wait(
    state: AsyncDispatchState,
    wait_tok: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    rows: int,
    orig_lanes: int,
    collective_id: int,
) -> jax.Array:
    """Wait for an in-flight EP dispatch, drain its sends, and unpack the rows.

    Returns `[world, rows, orig_lanes]`, unpacked straight from the receive
    buffer.
    """
    axis_key = normalize_axis_name(axis_name)
    world_size = jax.lax.axis_size(axis_key)

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
        latency_metadata=DMA_LATENCY_METADATA,
    ):
        (state_done,) = call_kernel(
            _wait_kernel_inplace,
            args=(wait_tok,),
            inout=(state,),
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
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
    tp_size: int,
    latent_dim: int,
    use_norm: bool,
    rms_eps: float,
    max_dma_tokens: int,
) -> jax.Array:
    """Combine the routed expert partial sums of every rank.

    One-shot reduce-scatter of the rows over `dp_axis_name`, psum over
    `tp_axis_name`, exact `RMSNorm(latent_dim)` (when `use_norm`), then this
    rank's `latent_dim / tp_size` columns. Batches above `max_dma_tokens` use
    XLA collectives; the DMA collectives receive into VMEM when within bounds.
    """
    latent_local = latent_dim // tp_size
    if moe_out.shape[0] > max_dma_tokens:
        routed_dp = lax.psum_scatter(
            moe_out, dp_axis_name, scatter_dimension=0, tiled=True
        )
        routed_full = lax.psum(routed_dp, tp_axis_name)
    else:
        routed_dp = latency_psum_scatter_rows(
            moe_out,
            dp_axis_name,
            collective_id=COLLECTIVE_ID_MOE_DP_COMBINE_RS,
            recv_memory_space="auto",
            vmem_max_tokens=max_dma_tokens,
        )
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
    if use_norm:
        routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)
    return lax.dynamic_slice_in_dim(
        routed_full,
        local_tp_rank[0, 0] * latent_local,
        latent_local,
        axis=1,
    )


# =============================================================================
# Large-batch compact ragged EP all-to-all over DP
# =============================================================================

_COMPACT_MIN_CAPACITY = 4096
_COMPACT_SEND_HEADROOM_FACTOR = 4
_COMPACT_RECV_HEADROOM_FACTOR = 8
_COMPACT_ROW_ALIGN = 512


def _round_up_capacity(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def compact_ep_capacity(
    *,
    b_dp: int,
    dp_size: int,
    tp_size: int,
    e_local: int,
    topk: int,
) -> tuple[int, int]:
    """Compact `(send_cap, recv_cap)` row capacities for ragged DP EP exchange.

    Each `(dp, tp)` shard owns `1 / (dp * tp)` of experts and handles only
    routes targeting its `tp` column (`expected = ceil(b_dp * topk / tp_size)`).
    Sizing the send buffer at `max(4096, 4 * expected)` and the receive buffer
    at `max(4096, 8 * expected)` (each capped by its exact theoretical upper
    bound and aligned to 512 rows for GMM tiling) gives `4x` column headroom
    and `8x` per-shard skew headroom while eliminating `[dp * b_dp * topk,
    latent_dim]` allocations.
    """
    expected_local = -(-(b_dp * topk) // max(tp_size, 1))
    send_target = max(
        _COMPACT_MIN_CAPACITY, _COMPACT_SEND_HEADROOM_FACTOR * expected_local
    )
    recv_target = max(
        _COMPACT_MIN_CAPACITY, _COMPACT_RECV_HEADROOM_FACTOR * expected_local
    )
    send_upper = min(b_dp * topk, b_dp * dp_size * e_local)
    recv_upper = min(dp_size * b_dp * topk, dp_size * b_dp * e_local)

    def _cap(upper: int, target: int) -> int:
        raw = min(upper, target)
        if raw >= _COMPACT_ROW_ALIGN:
            return _round_up_capacity(raw, _COMPACT_ROW_ALIGN)
        if raw >= 128:
            return _round_up_capacity(raw, 128)
        return max(1, raw)

    return _cap(send_upper, send_target), _cap(recv_upper, recv_target)


def ragged_dp_dispatch_combine_rms_slice(
    lat_tp: jax.Array,
    local_meta: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array,
    w2_scale: jax.Array,
    local_norm_w: jax.Array,
    local_tp_rank: jax.Array,
    *,
    dp_axis_name: str,
    tp_axis_name: str,
    dp_size: int,
    tp_size: int,
    topk: int,
    routed_activation: str,
    use_norm: bool,
    rms_eps: float,
    use_sparse_core: bool,
    counting_sort_fn: Any,
    ragged_gather_fn: Any,
    ragged_gather_reduce_fn: Any,
    moe_gmm_fn: Any,
    ragged_all_to_all_fn: Any = lax.ragged_all_to_all,
    send_cap: int | None = None,
    recv_cap: int | None = None,
) -> jax.Array:
    """Compact ragged EP dispatch, local planar_t GMM, and reverse combine.

    Replaces padded `all_gather(dp)` + full-batch `[dp * b_dp * topk, H]` MoE
    buffers + `psum_scatter(dp)` for large-batch prefill (`b_global >
    max_dma_tokens`, `dp_size > 1`).

    Each `(dp, tp)` rank selects only routes whose expert belongs to its `tp`
    column (`owner_tp == tp_rank`), groups them by `(owner_dp, local_exp)`,
    dispatches them into expert-major order on the owning `dp` peer via
    `ragged_all_to_all(dp)`, runs the local planar_t MXFP4 GMM on the compact
    `[recv_cap, H]` buffer, returns outputs via reverse `ragged_all_to_all(dp)`,
    performs weighted local reduction, `tp` `psum`, exact `RMSNorm`, and slices
    this rank's `latent_dim / tp_size` columns.
    """
    b_dp, latent_dim = lat_tp.shape
    latent_local = latent_dim // tp_size
    e_local = w13.shape[0]

    if send_cap is None or recv_cap is None:
        auto_send, auto_recv = compact_ep_capacity(
            b_dp=b_dp,
            dp_size=dp_size,
            tp_size=tp_size,
            e_local=e_local,
            topk=topk,
        )
        send_cap = auto_send if send_cap is None else send_cap
        recv_cap = auto_recv if recv_cap is None else recv_cap

    local_ids, local_weights = unpack_meta_columns(local_meta, topk=topk)
    ids_flat = local_ids.reshape(-1)
    weights_flat = local_weights.reshape(-1)
    tok_flat = jnp.arange(b_dp, dtype=jnp.int32).repeat(topk)

    tp_rank = local_tp_rank[0, 0]
    owner_ep = ids_flat // e_local
    owner_dp = owner_ep // tp_size
    owner_tp = owner_ep % tp_size
    local_exp = ids_flat - owner_ep * e_local

    in_col = (
        (ids_flat >= 0)
        & (weights_flat != 0)
        & (owner_tp == tp_rank)
        & (owner_dp >= 0)
        & (owner_dp < dp_size)
    )
    num_buckets = dp_size * e_local
    bucket = jnp.where(in_col, owner_dp * e_local + local_exp, num_buckets)

    # Counting sort gives stable within-bucket positions + per-(dp, exp) counts.
    revert_idx, bucket_counts = counting_sort_fn(bucket, num_buckets + 1)
    token_bits = max(1, (b_dp - 1).bit_length())
    packable = (num_buckets + 1).bit_length() + token_bits < 31
    if revert_idx is not None and packable:
        send_counts_flat = bucket_counts[:num_buckets]
        packed = lax.sort(
            jnp.left_shift(bucket, token_bits) | tok_flat,
            is_stable=False,
        )
        sorted_tok = jnp.bitwise_and(packed, (1 << token_bits) - 1)
    else:
        sorted_idx = jnp.argsort(bucket, stable=True)
        revert_idx = jnp.argsort(sorted_idx)
        sorted_tok = tok_flat[sorted_idx]
        send_counts_flat = jnp.bincount(bucket, length=num_buckets + 1)[
            :num_buckets
        ].astype(jnp.int32)

    send_tok = sorted_tok[:send_cap]

    # Bounds-safe send/recv offsets for `(dp_size, e_local)` updates.
    send_offsets_flat = jnp.cumsum(send_counts_flat) - send_counts_flat
    send_start = jnp.minimum(send_offsets_flat, send_cap).reshape(dp_size, e_local)
    send_end = jnp.minimum(send_offsets_flat + send_counts_flat, send_cap).reshape(
        dp_size, e_local
    )
    send_counts_clamped = send_end - send_start

    # Exchange (send_counts_clamped, send_start) over 'dp' in one tiny all_to_all:
    # receiver learns both how many rows arrive from each peer and where to write
    # each peer's outputs back in the reverse combine ragged_all_to_all.
    fwd_meta = lax.all_to_all(
        jnp.stack((send_counts_clamped, send_start), axis=-1),
        dp_axis_name,
        split_axis=0,
        concat_axis=0,
    )
    recv_counts_raw = fwd_meta[..., 0]
    rev_output_offsets = fwd_meta[..., 1]
    # Place received rows in expert-major order [e_local, dp_size] so recv_buf
    # feeds directly into moe_gmm without any post-dispatch sort or gather.
    rc_ed = recv_counts_raw.T.reshape(-1)
    ro_ed = jnp.cumsum(rc_ed) - rc_ed
    recv_start_ed = jnp.minimum(ro_ed, recv_cap)
    recv_end_ed = jnp.minimum(ro_ed + rc_ed, recv_cap)
    recv_start = recv_start_ed.reshape(e_local, dp_size).T
    recv_counts = (recv_end_ed - recv_start_ed).reshape(e_local, dp_size).T

    # Single small all_to_all returning (output_offsets, eff_send_counts) to senders.
    back = lax.all_to_all(
        jnp.stack((recv_start, recv_counts), axis=-1),
        dp_axis_name,
        split_axis=0,
        concat_axis=0,
    )
    output_offsets = back[..., 0]
    eff_send_counts = back[..., 1]

    # 1. Gather compact send buffer [send_cap, latent_dim].
    total_sent = send_counts_clamped.sum(dtype=jnp.int32)
    if use_sparse_core:
        send_buf = ragged_gather_fn(
            lat_tp,
            send_tok,
            jnp.array([0], dtype=jnp.int32),
            total_sent[None],
        )
    else:
        send_buf = lat_tp[send_tok]

    # 2. Forward ragged EP dispatch over 'dp' directly into expert-major recv_buf.
    recv_init = lax.empty((recv_cap, latent_dim), dtype=lat_tp.dtype)
    recv_buf = ragged_all_to_all_fn(
        send_buf,
        recv_init,
        send_start.reshape(-1),
        eff_send_counts.reshape(-1),
        output_offsets.reshape(-1),
        recv_counts.reshape(-1),
        axis_name=dp_axis_name,
    )

    # 3. Local planar_t MXFP4 GMM on compact [recv_cap, padded_hidden] buffer.
    group_sizes = recv_counts.sum(axis=0)
    padded_hidden = w13.shape[1]
    gmm_in = (
        jnp.pad(recv_buf, ((0, 0), (0, padded_hidden - latent_dim)))
        if padded_hidden > latent_dim
        else recv_buf
    )
    expected_local = -(-(b_dp * topk) // max(tp_size, 1))
    gmm_ep_shards = max(1, recv_cap // max(expected_local, 1))
    gmm_out = moe_gmm_fn(
        gmm_in,
        w13,
        w13_scale,
        w2,
        w2_scale,
        group_sizes,
        activation=routed_activation,
        preferred_element_type=lat_tp.dtype,
        ep_shards=gmm_ep_shards,
    )
    if padded_hidden > latent_dim:
        gmm_out = gmm_out[:, :latent_dim]

    # 4. Reverse ragged EP combine over 'dp' back into sender's send_buf order.
    comb_init = jnp.zeros((send_cap, latent_dim), dtype=lat_tp.dtype)
    comb_buf = ragged_all_to_all_fn(
        gmm_out,
        comb_init,
        recv_start.reshape(-1),
        recv_counts.reshape(-1),
        rev_output_offsets.reshape(-1),
        eff_send_counts.reshape(-1),
        axis_name=dp_axis_name,
    )

    # 5. Weighted local reduction back to [b_dp, latent_dim].
    eff_valid = in_col & (revert_idx < send_cap)
    safe_revert = jnp.where(eff_valid, revert_idx, 0)
    if use_sparse_core:
        routed_dp = ragged_gather_reduce_fn(
            comb_buf,
            safe_revert,
            weights_flat,
            eff_valid,
            reduce_group_size=topk,
            allow_fallback=False,
        ).astype(lat_tp.dtype)
    else:
        tok_h = comb_buf[safe_revert].reshape(b_dp, topk, latent_dim)
        tok_h = tok_h * weights_flat.reshape(b_dp, topk, 1).astype(jnp.float32)
        tok_h = jnp.where(eff_valid.reshape(b_dp, topk, 1), tok_h, 0)
        routed_dp = tok_h.sum(axis=1).astype(lat_tp.dtype)

    # 6. TP psum + exact RMSNorm(latent_dim) + local column slice.
    routed_full = lax.psum(routed_dp, tp_axis_name)
    if use_norm:
        routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)
    return lax.dynamic_slice_in_dim(
        routed_full,
        tp_rank * latent_local,
        latent_local,
        axis=1,
    )


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
