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
"""Kimi K3 packed DP dispatch and one-shot Pallas DMA collective bundle.

Provides:
  1. FTZ-safe, exact byte-valued ``bfloat16`` packing of
     ``(latent_bf16, topk_ids_i32, topk_weights_bf16)`` into one
     ``[B_dp, latent_dim + 3 * K]`` ``bfloat16`` tensor, so the MoE dispatch
     needs one DP all-gather per layer instead of three.
  2. A custom-op variant tag that encodes every active Kimi DMA/kernel-variant
     env value in the ``pallas::kimi_*`` op names, so a warm compile cache
     never serves a stale variant.
  3. ``KimiDmaCollectives``: the ``@jax_op`` collectives built once at model
     load and attached to the Kimi modules. Each one takes the one-shot
     Pallas DMA path for small (decode) batches and the ``jax.lax``
     collective otherwise; the choice is made at trace time from static
     shapes. Barrier ids (see ``kernels/kimi_k3/pallas_dma.py``):
       - ``cid=0`` ('tp'): KimiMoE routed_down_local column all-gather
       - ``cid=1`` ('dp'): packed MoE dispatch row all-gather
       - ``cid=2`` ('dp'): post-MoE combine reduce-scatter
       - ``cid=3`` ('tp'): KimiMoE routed_out psum
       - ``cid=4`` ('tp'): KDA/MLA o_proj and layer-0 KimiMLP down_proj psum
       - ``cid=5`` ('tp'): KimiMoE packed_moe_out psum
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_tp_mesh import validate_dp_tp_group_ranks
from vllm_torchtpu.kernels.kimi_k3.pallas_dma import (
    COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
    COLLECTIVE_ID_MOE_DP_COMBINE_RS,
    COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    COLLECTIVE_ID_MOE_TP_EXIT_PSUM,
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
    DEFAULT_MAX_DMA_TOKENS,
    DP_AXIS_NAME,
    TP_AXIS_NAME,
)
from vllm_torchtpu.kimi_k3_opt_modes import (
    normalize_compute_polish_mode,
    normalize_exit_fold_mode,
    normalize_spec_kda_mode,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.kimi_k3.dma_collective_ops import (
    build_pallas_all_gather_cols_op,
    build_pallas_all_gather_rows_op,
    build_pallas_psum_op,
    build_pallas_psum_scatter_rows_op,
    mesh_shape_and_ids,
    sanitize_tag,
)
from vllm_torchtpu.models.vllm.kimi_k3.moe_ep_overlap_op import (
    build_kimi_moe_ep_overlap_op,
)

logger = init_logger(__name__)


def normalize_latency_tp_psum_mode(raw: str | None) -> str:
    """Normalize ``TPU_KIMI_OPT_LATENCY_TP_PSUM`` to a canonical mode token.

    Supported modes:
      - ``"off"`` (aliases: ``""``, ``"0"``, ``"false"``, ``"no"``, ``"none"``)
      - ``"tp16"`` (alias: ``"auto"``): active when ``tp_size >= 16`` with
        ``max_dma_tokens = 8`` (`vmem` 1-shot decode reduction)
      - ``"routed_only"``: wires only ``cid=3`` (`routed_out`) when ``tp_size > 1``
      - ``"all"`` (aliases: ``"1"``, ``"true"``, ``"yes"``): wires ``cid=3, 4, 5``
        whenever ``tp_size > 1`` with ``max_dma_tokens = 64``
      - Positive integer string ``"<N>"`` (e.g. ``"8"``, ``"16"``): wires
        ``cid=3, 4, 5`` whenever ``tp_size > 1`` with ``max_dma_tokens = int(N)``
    """
    val = (raw or "off").strip().lower()
    if val in ("", "0", "false", "no", "off", "none"):
        return "off"
    if val in ("tp16", "auto"):
        return "tp16"
    if val == "routed_only":
        return "routed_only"
    if val in ("1", "true", "yes", "all"):
        return "all"
    if val.isdigit() and int(val) > 0:
        return str(int(val))
    raise ValueError(
        f"Invalid TPU_KIMI_OPT_LATENCY_TP_PSUM={raw!r}; expected 'off', 'tp16'/'auto', "
        "'routed_only', 'all', or a positive integer max_dma_tokens threshold."
    )


def kimi_dma_variant_tag() -> str:
    """Variant tag for the custom-op names, encoding every active Kimi knob.

    The "o2_" prefix and the field layout are part of the cached op names;
    changing them invalidates warm compile caches.
    """
    tp_mode = normalize_latency_tp_psum_mode(envs.TPU_KIMI_OPT_LATENCY_TP_PSUM)
    raw_ef = (envs.TPU_KIMI_OPT_EXIT_FOLD or "off").strip().lower()
    ef_mode = normalize_exit_fold_mode(raw_ef)
    polish_tag, _ = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    spec_kda = normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA)
    base = (
        f"o2_m{int(bool(envs.TPU_KIMI_OPT_2D_EP_MESH))}"
        f"_p{int(bool(envs.TPU_KIMI_OPT_PACKED_DP_AG))}"
        f"_lc{int(bool(envs.TPU_KIMI_OPT_LATENCY_COLLECTIVES))}"
        # The flat world dispatch this field once selected is gone; the field
        # stays as a fixed "_w0" so the custom-op names do not change.
        "_w0"
        f"_tp_{tp_mode}"
    )
    if ef_mode == "off":
        tag = base
    else:
        raw_ef_tok = sanitize_tag(raw_ef)
        if raw_ef_tok == ef_mode:
            tag = f"{base}_ef_{ef_mode}"
        else:
            tag = f"{base}_ef_{ef_mode}_{raw_ef_tok}"
    if polish_tag != "off":
        tag = f"{tag}_cp_{polish_tag}"
    if spec_kda != "off":
        tag = f"{tag}_sk_{spec_kda}"
    return tag


def pack_dispatch_metadata(
    topk_ids_i32: torch.Tensor,
    topk_weights_bf16: torch.Tensor,
) -> torch.Tensor:
    """Pack ``topk_ids[B, K]`` (int32) and ``topk_weights[B, K]`` into ``[B, 3K]`` bf16.

    Why byte-valued normal ``bfloat16`` lanes instead of raw ``.view(torch.bfloat16)``:
    TPU floating-point / padding operations flush ``bfloat16`` subnormals to signed
    zero (FTZ), which corrupts raw-bitcast ``int32`` expert IDs ``1..127``
    (``0x0001..0x007F``) to ``0`` and canonicalizes ``-1`` (``0xFFFFFFFF``) NaN bits.
    Encoding signed 16-bit IDs (`-32768..32767`, covering `-1` and `0..895`) into two
    unsigned byte-valued ``bfloat16`` lanes ``0..255`` (matching the byte
    packing in ``kernels/kimi_k3/collectives.py``) uses the same ``2 * K`` ``bf16``
    lanes with ``0`` ULP error and complete immunity to FTZ and NaN canonicalization.
    """
    ids_i32 = (
        topk_ids_i32
        if topk_ids_i32.dtype == torch.int32
        else topk_ids_i32.to(torch.int32)
    )
    w_bf16 = (
        topk_weights_bf16
        if topk_weights_bf16.dtype == torch.bfloat16
        else topk_weights_bf16.to(torch.bfloat16)
    )
    low_bf16 = (ids_i32 & 0xFF).to(torch.bfloat16)
    high_bf16 = ((ids_i32 >> 8) & 0xFF).to(torch.bfloat16)
    return torch.cat((low_bf16, high_bf16, w_bf16), dim=-1)


def pack_dp_dispatch(
    latent_bf16: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_weights_bf16: torch.Tensor,
) -> torch.Tensor:
    """Pack ``(latent[B, D], topk_ids[B, K], topk_weights[B, K])`` as ``[B, D+3K]``."""
    x_bf16 = (
        latent_bf16
        if latent_bf16.dtype == torch.bfloat16
        else latent_bf16.to(torch.bfloat16)
    )
    meta_bf16 = pack_dispatch_metadata(topk_ids_i32, topk_weights_bf16)
    return torch.cat((x_bf16, meta_bf16), dim=-1)


def unpack_dp_dispatch(
    packed_bf16: torch.Tensor,
    latent_dim: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack ``[B, D + 3K]`` bf16 into ``(latent, topk_ids_i32, topk_weights)``."""
    latent = packed_bf16[:, :latent_dim]
    low_i32 = packed_bf16[:, latent_dim : latent_dim + topk].to(torch.int32)
    high_i32 = packed_bf16[:, latent_dim + topk : latent_dim + 2 * topk].to(torch.int32)
    topk_ids = (((high_i32 << 8) | low_i32) << 16) >> 16
    topk_weights = packed_bf16[:, latent_dim + 2 * topk : latent_dim + 3 * topk]
    return latent, topk_ids, topk_weights


@dataclass(frozen=True)
class KimiDmaCollectives:
    """``@jax_op`` collectives built at model load and attached to Kimi modules."""

    variant_tag: str
    dp_size: int
    tp_size: int
    tp_latent_ag_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    dp_packed_ag_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    dp_combine_rs_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    tp_routed_psum_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    tp_attn_mlp_psum_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    tp_moe_exit_psum_op: Callable[[torch.Tensor], torch.Tensor] | None = None
    exit_fold_mode: str = "off"
    mesh: Mesh | None = None
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS

    @property
    def has_any_op(self) -> bool:
        return self.exit_fold_mode != "off" or any(
            op is not None
            for op in (
                self.tp_latent_ag_op,
                self.dp_packed_ag_op,
                self.dp_combine_rs_op,
                self.tp_routed_psum_op,
                self.tp_attn_mlp_psum_op,
                self.tp_moe_exit_psum_op,
            )
        )

    def attach_to_moe(self, moe_module: Any) -> None:
        moe_module._dma_tp_latent_ag_op = self.tp_latent_ag_op
        moe_module._dma_dp_packed_ag_op = self.dp_packed_ag_op
        moe_module._dma_dp_combine_rs_op = self.dp_combine_rs_op
        moe_module._dma_tp_routed_psum_op = self.tp_routed_psum_op
        moe_module._dma_tp_moe_exit_psum_op = self.tp_moe_exit_psum_op
        w_shared_gu = getattr(moe_module, "w_shared_gu", None)
        can_attach_ep_overlap = (
            self.exit_fold_mode != "off"
            and self.mesh is not None
            and getattr(moe_module, "w_routed_down_tp", None) is not None
            and w_shared_gu is not None
            and getattr(moe_module, "packed_moe_out_weight", None) is not None
            and bool(getattr(moe_module, "use_ep", False))
        )
        if not can_attach_ep_overlap:
            moe_module._dma_tp_rank_buf = None
            moe_module._ep_overlap_op = None
            return

        assert self.mesh is not None and w_shared_gu is not None
        # The fused MoE op reads this rank's TP index from a runtime buffer so
        # every rank traces the same program.
        _, tp_rank = validate_dp_tp_group_ranks(self.dp_size, self.tp_size)
        moe_module.register_buffer(
            "_dma_tp_rank_buf",
            torch.tensor([[tp_rank]], dtype=torch.int32, device=w_shared_gu.device),
            persistent=False,
        )
        moe_module._ep_overlap_op = build_kimi_moe_ep_overlap_op(
            self.mesh,
            moe_module,
            exit_fold_mode=self.exit_fold_mode,
            max_dma_tokens=self.max_dma_tokens,
            variant_tag=self.variant_tag,
        )

    def attach_to_attn(self, attn_module: Any) -> None:
        attn_module._tp_exit_psum_op = self.tp_attn_mlp_psum_op

    def attach_to_mlp(self, mlp_module: Any) -> None:
        # Only attach full-TP cid=4 psum to standard TP-wide row-parallel MLPs
        # (Layer-0 dense KimiMLP where tp_group is None and down_proj reduces).
        if (
            getattr(mlp_module, "tp_group", None) is None
            and getattr(mlp_module.down_proj, "reduce_results", False)
            and getattr(mlp_module.down_proj, "tp_size", 1) == self.tp_size
            and self.tp_size > 1
        ):
            mlp_module._tp_exit_psum_op = self.tp_attn_mlp_psum_op


class _TpPsumPlan(NamedTuple):
    """Which TP psums take the one-shot DMA path, and their row cap."""

    routed: bool
    all_exits: bool
    max_dma_tokens: int


# "tp16" mode: only at TP >= this size, with this row cap (decode batches only).
_TP16_MIN_TP_SIZE = 16
_TP16_MAX_DMA_TOKENS = 8


def _resolve_tp_psum_plan(
    tp_psum_mode: str, tp_size: int, max_dma_tokens: int
) -> _TpPsumPlan:
    """TP psum plan for a normalized ``TPU_KIMI_OPT_LATENCY_TP_PSUM`` mode."""
    if tp_size <= 1 or tp_psum_mode == "off":
        return _TpPsumPlan(False, False, max_dma_tokens)
    if tp_psum_mode == "tp16":
        if tp_size >= _TP16_MIN_TP_SIZE:
            return _TpPsumPlan(True, True, _TP16_MAX_DMA_TOKENS)
        return _TpPsumPlan(False, False, max_dma_tokens)
    if tp_psum_mode == "routed_only":
        return _TpPsumPlan(True, False, max_dma_tokens)
    if tp_psum_mode == "all":
        return _TpPsumPlan(True, True, max_dma_tokens)
    if tp_psum_mode.isdigit():
        return _TpPsumPlan(True, True, int(tp_psum_mode))
    return _TpPsumPlan(False, False, max_dma_tokens)


def build_kimi_dma_collectives(
    mesh: Mesh,
    *,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> KimiDmaCollectives:
    """Build the active Kimi DMA collectives; call outside ``torch.compile``."""
    dp_size, tp_size, _ = mesh_shape_and_ids(mesh)
    vtag = kimi_dma_variant_tag()
    use_lat = bool(envs.TPU_KIMI_OPT_LATENCY_COLLECTIVES)
    tp_psum_mode = normalize_latency_tp_psum_mode(envs.TPU_KIMI_OPT_LATENCY_TP_PSUM)
    ef_mode = normalize_exit_fold_mode(envs.TPU_KIMI_OPT_EXIT_FOLD)
    common = {"recv_memory_space": "auto", "variant_tag": vtag}

    tp_latent_ag_op = None
    dp_packed_ag_op = None
    dp_combine_rs_op = None
    tp_routed_psum_op = None
    tp_attn_mlp_psum_op = None
    tp_moe_exit_psum_op = None

    # With exit fold on, the fused MoE op does its own dispatch and combine.
    if ef_mode == "off" and use_lat:
        if tp_size > 1:
            tp_latent_ag_op = build_pallas_all_gather_cols_op(
                mesh,
                axis_name=TP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_TP_LATENT_AG,
                max_dma_tokens=max_dma_tokens,
                **common,
            )
        if dp_size > 1:
            dp_packed_ag_op = build_pallas_all_gather_rows_op(
                mesh,
                axis_name=DP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                input_sharding="dp_replicated",
                max_dma_tokens=max_dma_tokens,
                **common,
            )
            dp_combine_rs_op = build_pallas_psum_scatter_rows_op(
                mesh,
                axis_name=DP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_DP_COMBINE_RS,
                max_dma_tokens=max_dma_tokens,
                **common,
            )

    plan = _resolve_tp_psum_plan(tp_psum_mode, tp_size, max_dma_tokens)
    if plan.routed and ef_mode == "off":
        tp_routed_psum_op = build_pallas_psum_op(
            mesh,
            axis_name=TP_AXIS_NAME,
            collective_id=COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
            max_dma_tokens=plan.max_dma_tokens,
            **common,
        )
    if plan.all_exits:
        tp_attn_mlp_psum_op = build_pallas_psum_op(
            mesh,
            axis_name=TP_AXIS_NAME,
            collective_id=COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
            max_dma_tokens=plan.max_dma_tokens,
            **common,
        )
        tp_moe_exit_psum_op = build_pallas_psum_op(
            mesh,
            axis_name=TP_AXIS_NAME,
            collective_id=COLLECTIVE_ID_MOE_TP_EXIT_PSUM,
            max_dma_tokens=plan.max_dma_tokens,
            **common,
        )

    bundle = KimiDmaCollectives(
        variant_tag=vtag,
        dp_size=dp_size,
        tp_size=tp_size,
        tp_latent_ag_op=tp_latent_ag_op,
        dp_packed_ag_op=dp_packed_ag_op,
        dp_combine_rs_op=dp_combine_rs_op,
        tp_routed_psum_op=tp_routed_psum_op,
        tp_attn_mlp_psum_op=tp_attn_mlp_psum_op,
        tp_moe_exit_psum_op=tp_moe_exit_psum_op,
        exit_fold_mode=ef_mode,
        mesh=mesh,
        max_dma_tokens=max_dma_tokens,
    )
    if bundle.has_any_op:
        logger.info(
            "Armed Kimi DMA collectives | variant=%s | mesh=(dp=%d, tp=%d) | "
            "exit_fold=%s tp_lat_ag=%s dp_pack_ag=%s dp_rs=%s "
            "tp_psum=(routed=%s, exit=%s)",
            vtag,
            dp_size,
            tp_size,
            ef_mode,
            tp_latent_ag_op is not None,
            dp_packed_ag_op is not None,
            dp_combine_rs_op is not None,
            tp_routed_psum_op is not None,
            tp_moe_exit_psum_op is not None,
        )
    return bundle
