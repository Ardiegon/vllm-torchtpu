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
         combine + `[latent_local]` exit-fold:
           * `O4-B`: 1-shot `'dp'` `latency_psum_scatter_rows` (`cid=2`) +
             width-guarded `'tp'` `latency_psum` (`cid=3`) + in-`@jax_op`
             fused `RMSNorm(latent_dim)` + `[latent_local]` slice.
"""

from __future__ import annotations

import dataclasses
import inspect
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

DEFAULT_C2_COMBINE_KIND = "o4b"
DEFAULT_C2_DISPATCH_KIND = "axial"


def resolve_effective_exit_fold_plan(mode: str) -> tuple[str, bool, str]:
    """Map canonical ``exit_fold_mode`` + static shape to ``(combine_kind, async_dispatch, dispatch_kind)``."""
    norm = normalize_exit_fold_mode(mode)
    if norm in ("o4d", "c2"):
        return DEFAULT_C2_COMBINE_KIND, True, DEFAULT_C2_DISPATCH_KIND
    raise ValueError(f"resolve_effective_exit_fold_plan called with inactive mode={mode!r}")


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
    axis_key = normalize_axis_name(axis_name)
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


# =============================================================================
# Option `O4-B` (`cid=2` + `cid=3` + Fused `RMSNorm` + `[latent_local]` Slice)
# =============================================================================


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

    vmax = max_dma_tokens if vmem_mode == "vmem_chain" else DEFAULT_VMEM_MAX_TOKENS
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
    _, polish_tokens = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    polish_trays = "trays" if "trays" in polish_tokens else "off"
    polish_vmem = "vmem_chain" if "vmem_chain" in polish_tokens else "off"
    polish_copy = "copy_opt" in polish_tokens
    fused_decode_max_tokens = (
        FUSED_VERIFY_MOE_MAX_TOKENS
        if normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA) == "o6b"
        else FUSED_DECODE_MOE_MAX_TOKENS
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
            zero_wait_copy = polish_vmem == "vmem_chain"

            # 1. Dispatch + Shared Gate/Up Compute: 2-stage `TP` (`cid=0`) ->
            #    async `DP` (`cid=1`) for decode (`<= max_dma_tokens`), and 1D
            #    axial `TP` (`axis=1`) -> `DP` (`axis=0`) XLA `all_gather` for
            #    prefill (`> max_dma_tokens`).
            if b_global > max_dma_tokens:
                shared_act = _compute_shared_gu_and_act(
                    local_hidden,
                    local_w_shared_gu,
                    act_kind=shared_act_kind,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
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
                    ag_ids, ag_weights = _unpack_meta_columns(
                        local_meta, topk=topk
                    )
            elif dp_size > 1:
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
                ag_latent, ag_ids, ag_weights = _unpack_axial_gathered(
                    gathered_dp, latent_dim=latent_dim, topk=topk
                )
            elif tp_size > 1:
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
                fused_decode_max_tokens=fused_decode_max_tokens,
            )

            # 3. Mathematically Exact Combine + `RMSNorm(latent_dim)` + `[latent_local]` Exit-Fold
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
        REGISTERED_OPS,
        format_pallas_dma_op_name,
        mesh_shape_and_ids,
        sanitize_tag,
    )

    norm_mode = normalize_exit_fold_mode(exit_fold_mode)
    if norm_mode == "off":
        raise ValueError("build_kimi_moe_ep_overlap_op requires an active exit_fold_mode")

    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
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
        norm_mode
    )
    impl_tag = f"_{dispatch_kind_sig}_{combine_kind_sig}"
    spec_tag = (
        f"{norm_mode}{impl_tag}_k{cfg['topk']}_{sanitize_tag(cfg['shared_act_kind'])}"
        f"_q{int(cfg['has_scales'])}_n{int(cfg['use_norm'])}"
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
    raw_key = ("moe_ep_overlap_raw", op_name, dev_ids, cfg_sig)

    if raw_key in REGISTERED_OPS:
        raw_op = REGISTERED_OPS[raw_key]
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
        REGISTERED_OPS[raw_key] = raw_op  # type: ignore[assignment]

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
        call_args.append(layer_moe._dma_tp_rank_buf)
        return raw_op(*call_args)

    _call._qualname = op_name  # type: ignore[attr-defined]
    return _call
