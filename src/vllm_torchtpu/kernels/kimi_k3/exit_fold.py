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

One `jax_op` (`jax.shard_map` over the `('dp', 'tp')` mesh) runs a whole
Kimi K3 MoE block for expert parallelism, holding its Pallas DMA semaphores
internally:

1. Dispatch. Each rank owns a `[B_dp, latent_dim / tp]` slice of the routed
   latent. Decode batches (at most `max_dma_tokens` tokens in total) gather
   it in two axial stages: a one-shot remote-DMA all-gather over `'tp'`,
   then a non-blocking remote-DMA all-gather over `'dp'` of the latent plus
   the packed routing metadata (`async_ep_dispatch_start` /
   `async_ep_dispatch_wait`). The shared-expert gate/up matmul and gated
   activation run between start and wait; `lax.optimization_barrier` edges
   pin that order (start -> shared experts -> wait). Larger (prefill)
   batches use XLA all-gathers on contiguous axes instead.
2. Routed experts (`fused_moe_func`) on the gathered tokens.
3. Combine. A one-shot `'dp'` reduce-scatter, a `'tp'` psum guarded to
   small widths (XLA collectives above the caps), the exact
   `RMSNorm(latent_dim)` of the routed output, and the slice of this rank's
   `latent_dim / tp` columns.

The op returns `[B_dp, shared_inter_tp + latent_dim / tp]`: the shared
activation and the normalized routed slice, concatenated.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

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
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
    DEFAULT_MAX_DMA_TOKENS,
    DEFAULT_VMEM_MAX_TOKENS,
    DP_AXIS_NAME,
    TP_AXIS_NAME,
    barrier_and_allocate,
    call_kernel,
    kimi_guarded_all_gather_cols,
    kimi_guarded_psum,
    kimi_guarded_psum_scatter_rows,
    normalize_axis_name,
    seam_pieces_pack,
    seam_pieces_unpack,
)
from vllm_torchtpu.kernels.megablox.gmm_v6e_planar_t_b11 import (
    FUSED_DECODE_MOE_MAX_TOKENS,
    FUSED_VERIFY_MOE_MAX_TOKENS,
)
from vllm_torchtpu.kimi_k3_opt_modes import (
    normalize_compute_polish_mode,
    normalize_exit_fold_mode,
    normalize_spec_kda_mode,
)

Ref = Any


class ExitFoldPlan(NamedTuple):
    """How the EP-overlap op dispatches tokens and combines expert outputs."""

    combine_kind: str
    async_dispatch: bool
    dispatch_kind: str


# Every active `TPU_KIMI_OPT_EXIT_FOLD` mode runs this plan: the two-stage
# axial (TP, then async DP) dispatch and the hierarchical combine (DP
# reduce-scatter, then TP psum). The combine kind keeps its historical name
# "o4b" because it is part of the op name, which keys the compile caches.
_EXIT_FOLD_PLAN = ExitFoldPlan(
    combine_kind="o4b", async_dispatch=True, dispatch_kind="axial"
)


def resolve_effective_exit_fold_plan(mode: str) -> ExitFoldPlan:
    """Return the plan of an active `TPU_KIMI_OPT_EXIT_FOLD` mode.

    Raises:
      ValueError: `mode` is invalid or `"off"`.
    """
    if normalize_exit_fold_mode(mode) == "off":
        raise ValueError(
            f"resolve_effective_exit_fold_plan called with inactive mode={mode!r}"
        )
    return _EXIT_FOLD_PLAN


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
    zero_extra_copy: bool = False,
) -> jax.Array:
    """Wait for an in-flight EP dispatch, drain its sends, and unpack the rows.

    Returns `[world, rows, orig_lanes]`. With `zero_extra_copy` the rows are
    unpacked straight from the receive buffer instead of being copied to a
    fresh output buffer first.
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

    if zero_extra_copy:

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
        _wait_and_drain(state_ref)
        with jax.named_scope("emit_dispatch_rows"):
            emit = pltpu.make_async_copy(
                src_ref=state_ref.x_recv,
                dst_ref=out_ref,
                sem=out_sem,
            )
            emit.start()
            emit.wait()

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


def _unpack_axial_gathered(
    gathered_dp_2d: jax.Array,
    *,
    latent_dim: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split `[dp * B_dp, latent_dim + 3 * topk]` into `(latent, ids, weights)`."""
    ag_latent = gathered_dp_2d[:, :latent_dim]
    ag_ids, ag_weights = _unpack_meta_columns(gathered_dp_2d[:, latent_dim:], topk=topk)
    return ag_latent, ag_ids, ag_weights


# =============================================================================
# Hierarchical combine: DP reduce-scatter + TP psum + RMSNorm + column slice
# =============================================================================


def hierarchical_combine_rms_slice(
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
    vmem_chain: bool = False,
) -> jax.Array:
    """Combine the routed expert partial sums of every rank.

    One-shot `'dp'` reduce-scatter of the rows, `'tp'` psum, exact
    `RMSNorm(latent_dim)` (when `use_norm`), then this rank's
    `latent_dim / tp_size` columns. Batches above `max_dma_tokens` (and a
    single-rank mesh) use XLA collectives. With `vmem_chain` the DMA
    collectives receive into VMEM up to `max_dma_tokens` rows.
    """
    latent_local = latent_dim // tp_size
    world_size = dp_size * tp_size
    if moe_out.shape[0] > max_dma_tokens or world_size == 1:
        routed_dp = (
            lax.psum_scatter(moe_out, DP_AXIS_NAME, scatter_dimension=0, tiled=True)
            if dp_size > 1
            else moe_out
        )
        routed_full = lax.psum(routed_dp, TP_AXIS_NAME) if tp_size > 1 else routed_dp
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

    vmax = max_dma_tokens if vmem_chain else DEFAULT_VMEM_MAX_TOKENS
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
        # The one-shot TP psum handles at most `b_dp * tp_size <= 128` rows
        # (b_dp <= 32 at TP=4, 16 at TP=8, 8 at TP=16); wider buckets fall back
        # to `lax.psum` inside the guarded collective.
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
# The EP-overlap `jax_op`
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MoeOverlapConfig:
    """Static configuration of one EP-overlap op, resolved before tracing.

    Hashable, so it also keys the op cache.
    """

    moe_act_str: str
    topk: int
    has_scales: bool
    has_bias: bool
    use_norm: bool
    rms_eps: float
    shared_act_kind: str
    situ_beta: float
    situ_linear_beta: float | None
    use_sparse_core: bool
    onehot_thresh: int
    skip_padded: bool
    # Fused small-batch MXFP4 MoE kernel settings (see `fused_moe_func`).
    token_trays_mode: str
    rank3_scales: bool
    fused_decode_max_tokens: int
    # Leave the gathered rows in the dispatch receive buffer and let the
    # combine collectives receive into VMEM.
    vmem_chain: bool


class MoeOverlapJaxFn(NamedTuple):
    """The SPMD function of the EP-overlap op and its operand layout."""

    fn: Callable[..., jax.Array]
    in_specs: tuple[PartitionSpec, ...]
    out_spec: PartitionSpec
    arg_names: tuple[str, ...]


def _extract_moe_overlap_config(moe_module: Any) -> MoeOverlapConfig:
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        resolve_onehot_permute_threshold,
    )

    routed = moe_module.experts.routed_experts
    qm = routed.quant_method
    moe_act_str = getattr(qm, "_tpu_activation_str", None)
    assert moe_act_str is not None, (
        "[kimi_moe_ep_overlap] process_weights_after_loading must run on "
        "routed_experts before building the EP-overlap op"
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
        float(moe_module.routed_expert_norm.variance_epsilon) if use_norm else 1e-5
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
    _, polish_tokens = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    spec_kda_mode = normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA)

    return MoeOverlapConfig(
        moe_act_str=str(moe_act_str),
        topk=topk,
        has_scales=bool(has_scales),
        has_bias=bool(has_bias),
        use_norm=bool(use_norm),
        rms_eps=rms_eps,
        shared_act_kind=shared_act_kind,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        use_sparse_core=use_sparse_core,
        onehot_thresh=onehot_thresh,
        skip_padded=skip_padded,
        token_trays_mode="trays" if "trays" in polish_tokens else "off",
        rank3_scales="copy_opt" in polish_tokens,
        fused_decode_max_tokens=(
            FUSED_VERIFY_MOE_MAX_TOKENS
            if spec_kda_mode == "o6b"
            else FUSED_DECODE_MOE_MAX_TOKENS
        ),
        vmem_chain="vmem_chain" in polish_tokens,
    )


def _dispatch_overlapping_shared_experts(
    send_shard: jax.Array,
    axis_name: str,
    local_hidden: jax.Array,
    local_w_shared_gu: jax.Array,
    *,
    rows: int,
    orig_lanes: int,
    zero_extra_copy: bool,
    cfg: MoeOverlapConfig,
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
    shared_act = _compute_shared_gu_and_act(
        hidden_after_start,
        local_w_shared_gu,
        act_kind=cfg.shared_act_kind,
        situ_beta=cfg.situ_beta,
        situ_linear_beta=cfg.situ_linear_beta,
    )
    wait_tok, shared_act = lax.optimization_barrier((start_tok, shared_act))
    gathered = async_ep_dispatch_wait(
        rx_state,
        wait_tok,
        axis_name,
        rows=rows,
        orig_lanes=orig_lanes,
        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
        zero_extra_copy=zero_extra_copy,
    )
    return gathered, shared_act


def _make_moe_ep_overlap_jax_fn(
    mesh: Mesh,
    *,
    dp_size: int,
    tp_size: int,
    max_dma_tokens: int,
    cfg: MoeOverlapConfig,
) -> MoeOverlapJaxFn:
    """Construct the SPMD JAX function and its input/output `PartitionSpec`s."""
    from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

    topk = cfg.topk

    spec_hidden = PartitionSpec(DP_AXIS_NAME, None)
    spec_rdown = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    spec_meta = PartitionSpec(DP_AXIS_NAME, None)
    spec_w_sh = PartitionSpec(None, TP_AXIS_NAME)
    spec_ep_w = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    spec_norm = PartitionSpec(None)
    spec_scalar_2d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    out_spec = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    out_sharding = NamedSharding(mesh, out_spec)

    # (name, spec) of every operand, in call order.
    operands: list[tuple[str, PartitionSpec]] = [
        ("hidden_states", spec_hidden),  # [dp * B_dp, H]
        ("routed_down_local", spec_rdown),  # [dp * B_dp, tp * d_local]
        ("packed_meta", spec_meta),  # [dp * B_dp, m_padded]
        ("w_shared_gu", spec_w_sh),  # [H, tp * 2 * s_inter_tp]
        ("w1_sharded", spec_ep_w),  # [dp, tp, E_local, ...]
        ("w2_sharded", spec_ep_w),  # [dp, tp, E_local, ...]
    ]
    if cfg.has_scales:
        operands += [("w1_scale_sharded", spec_ep_w), ("w2_scale_sharded", spec_ep_w)]
    if cfg.has_bias:
        operands += [("w1_bias_sharded", spec_ep_w), ("w2_bias_sharded", spec_ep_w)]
    if cfg.use_norm:
        operands.append(("norm_weight", spec_norm))  # [latent_dim]
    operands += [("experts_start_2d", spec_scalar_2d), ("tp_rank_2d", spec_scalar_2d)]
    in_specs = tuple(spec for _, spec in operands)
    arg_names_t = tuple(name for name, _ in operands)

    def _jax_fn(*global_args: jax.Array) -> jax.Array:
        def _local(*local_args: jax.Array) -> jax.Array:
            named = dict(zip(arg_names_t, local_args, strict=True))
            local_hidden = named["hidden_states"]
            local_rdown = named["routed_down_local"]
            local_meta = named["packed_meta"]
            local_w_shared_gu = named["w_shared_gu"]
            local_w1 = jnp.squeeze(named["w1_sharded"], axis=(0, 1))
            local_w2 = jnp.squeeze(named["w2_sharded"], axis=(0, 1))
            if cfg.has_scales:
                local_w1_scale = jnp.squeeze(named["w1_scale_sharded"], axis=(0, 1))
                local_w2_scale = jnp.squeeze(named["w2_scale_sharded"], axis=(0, 1))
            else:
                local_w1_scale = None
                local_w2_scale = None
            if cfg.has_bias:
                local_w1_bias = jnp.squeeze(named["w1_bias_sharded"], axis=(0, 1))
                local_w2_bias = jnp.squeeze(named["w2_bias_sharded"], axis=(0, 1))
            else:
                local_w1_bias = None
                local_w2_bias = None
            if cfg.use_norm:
                local_norm_w = named["norm_weight"]
            else:
                local_norm_w = jnp.ones((1,), dtype=jnp.bfloat16)
            local_experts_start = named["experts_start_2d"][0, 0]
            local_tp_rank = named["tp_rank_2d"]

            b_dp, d_local = local_rdown.shape
            latent_dim = tp_size * d_local
            b_global = dp_size * b_dp

            # 1. Dispatch + shared gate/up compute.
            if b_global > max_dma_tokens:
                # Prefill: XLA all-gathers on contiguous axes (TP on axis 1,
                # then DP on axis 0), so no 4D transpose is needed.
                shared_act = _compute_shared_gu_and_act(
                    local_hidden,
                    local_w_shared_gu,
                    act_kind=cfg.shared_act_kind,
                    situ_beta=cfg.situ_beta,
                    situ_linear_beta=cfg.situ_linear_beta,
                )
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
                    ag_ids, ag_weights = _unpack_meta_columns(local_meta, topk=topk)
            elif dp_size > 1:
                # Decode: one-shot TP all-gather of the latent, then the async
                # DP dispatch of latent + metadata.
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
                gathered_dp_3d, shared_act = _dispatch_overlapping_shared_experts(
                    packed_dp,
                    DP_AXIS_NAME,
                    local_hidden,
                    local_w_shared_gu,
                    rows=b_dp,
                    orig_lanes=dp_orig_lanes,
                    zero_extra_copy=cfg.vmem_chain,
                    cfg=cfg,
                )
                gathered_dp = gathered_dp_3d.reshape(dp_size * b_dp, dp_orig_lanes)
                ag_latent, ag_ids, ag_weights = _unpack_axial_gathered(
                    gathered_dp, latent_dim=latent_dim, topk=topk
                )
            elif tp_size > 1:
                # Decode, single DP group: async TP dispatch of the latent.
                lat_tp_3d, shared_act = _dispatch_overlapping_shared_experts(
                    local_rdown,
                    TP_AXIS_NAME,
                    local_hidden,
                    local_w_shared_gu,
                    rows=b_dp,
                    orig_lanes=d_local,
                    zero_extra_copy=cfg.vmem_chain,
                    cfg=cfg,
                )
                ag_latent = (
                    lat_tp_3d.reshape(1, latent_dim)
                    if b_dp == 1
                    else jnp.swapaxes(lat_tp_3d, 0, 1).reshape(b_dp, latent_dim)
                )
                ag_ids, ag_weights = _unpack_meta_columns(local_meta, topk=topk)
            else:
                shared_act = _compute_shared_gu_and_act(
                    local_hidden,
                    local_w_shared_gu,
                    act_kind=cfg.shared_act_kind,
                    situ_beta=cfg.situ_beta,
                    situ_linear_beta=cfg.situ_linear_beta,
                )
                ag_latent = local_rdown
                ag_ids, ag_weights = _unpack_meta_columns(local_meta, topk=topk)

            # 2. Routed experts.
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
                activation=cfg.moe_act_str,
                use_ep=True,
                use_sparse_core=cfg.use_sparse_core,
                onehot_moe_permute_threshold=cfg.onehot_thresh,
                rhs_quant_dtype=None,
                skip_padded_tokens=cfg.skip_padded,
                token_trays_mode=cfg.token_trays_mode,
                rank3_scales=cfg.rank3_scales,
                fused_decode_max_tokens=cfg.fused_decode_max_tokens,
            )

            # 3. Exact combine + RMSNorm(latent_dim) + [latent_local] slice.
            routed_local = hierarchical_combine_rms_slice(
                moe_out,
                local_norm_w,
                local_tp_rank,
                dp_size=dp_size,
                tp_size=tp_size,
                latent_dim=latent_dim,
                use_norm=cfg.use_norm,
                rms_eps=cfg.rms_eps,
                max_dma_tokens=max_dma_tokens,
                vmem_chain=cfg.vmem_chain,
            )

            # One contiguous [B_dp, s_inter_tp + latent_local] output.
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
    return MoeOverlapJaxFn(_jax_fn, in_specs, out_spec, arg_names_t)


def build_kimi_moe_ep_overlap_op(
    mesh: Mesh,
    moe_module: Any,
    *,
    exit_fold_mode: str,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "ep_overlap",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Any], torch.Tensor]:
    """Build (or reuse the cached) EP-overlap `jax_op` for one MoE layer shape."""
    # Imported here: kernels/ must not import models/ at module load (cycle).
    from vllm_torchtpu.models.vllm.kimi_k3.dma_collective_ops import (
        REGISTERED_OPS,
        format_pallas_dma_op_name,
        mesh_shape_and_ids,
        sanitize_tag,
    )

    norm_mode = normalize_exit_fold_mode(exit_fold_mode)
    if norm_mode == "off":
        raise ValueError(
            "build_kimi_moe_ep_overlap_op requires an active exit_fold_mode"
        )

    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
    cfg = _extract_moe_overlap_config(moe_module)

    plan = resolve_effective_exit_fold_plan(norm_mode)
    impl_tag = f"_{plan.dispatch_kind}_{plan.combine_kind}"
    spec_tag = (
        f"{norm_mode}{impl_tag}_k{cfg.topk}_{sanitize_tag(cfg.shared_act_kind)}"
        f"_q{int(cfg.has_scales)}_n{int(cfg.use_norm)}"
    )
    op_name = format_pallas_dma_op_name(
        kind="moe_ep_overlap",
        spec_tag=spec_tag,
        axis_name=("dp", "tp"),
        collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
        recv_memory_space="hbm",
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    raw_key = ("moe_ep_overlap_raw", op_name, dev_ids, cfg)

    if raw_key in REGISTERED_OPS:
        raw_op = REGISTERED_OPS[raw_key]
    else:
        from torch.tpu.pallas import jax_op

        jax_fn, in_specs, out_spec, arg_names_t = _make_moe_ep_overlap_jax_fn(
            mesh,
            dp_size=dp_size,
            tp_size=tp_size,
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
        REGISTERED_OPS[raw_key] = raw_op  # type: ignore[assignment]

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
        if cfg.has_scales:
            call_args.append(routed.w13_weight_scale.unsqueeze(0).unsqueeze(0))
            call_args.append(routed.w2_weight_scale.unsqueeze(0).unsqueeze(0))
        if cfg.has_bias:
            call_args.append(routed.w13_bias.unsqueeze(0).unsqueeze(0))
            call_args.append(routed.w2_bias.unsqueeze(0).unsqueeze(0))
        if cfg.use_norm:
            call_args.append(layer_moe.routed_expert_norm.weight)
        call_args.append(routed._experts_start.reshape(1, 1))
        call_args.append(layer_moe._dma_tp_rank_buf)
        return raw_op(*call_args)

    _call._qualname = op_name  # type: ignore[attr-defined]
    return _call
