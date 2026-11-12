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
"""Kimi K3 Phase O2 packed DP dispatch & 1-shot Pallas ``latency_*`` collectives.

Provides:
  1. FTZ-safe, zero-ULP byte-valued ``bfloat16`` packing/unpacking of
     ``(latent_bf16, topk_ids_i32, topk_weights_bf16)`` into a single
     ``[B_dp, latent_dim + 3 * K]`` ``bfloat16`` tensor (cutting per-layer
     ``DP`` ``all_gather`` calls ``3 -> 1``: ``276 -> 92`` on ``93L``,
     ``12 -> 4`` on ``4``-MoE ``L5``).
  2. Warm-cache-safe custom-op variant tagging (`D87`) encoding all active
     `O2` environment toggles directly into `pallas::kimi_*` custom-op names.
  3. Pre-built `@jax_op` bundle (`KimiO2Collectives`) with JAX-internal
     ``T_padded <= max_dma_tokens`` width guards (`T_padded <= 64` 1-shot Pallas
     remote-DMA, `T_padded > 64` `jax.lax.*` ring fallback inside JAX — zero
     Python branching on symbolic `num_tokens`) across the non-overlapping
     `collective_id` `0..5` barrier registry:
       - `cid=0` (`'tp'`): `KimiMoE` `routed_down_local` 1-shot `all_gather_cols`
       - `cid=1` (`'dp'` or `('dp', 'tp')`): packed `DP` (or flat `world`)
         EP dispatch `all_gather_rows`
       - `cid=2` (`'dp'`): post-MoE `DP` combine `psum_scatter_rows`
       - `cid=3` (`'tp'`): `KimiMoE` `routed_out` deterministic 1-shot `psum`
       - `cid=4` (`'tp'`): `KDA`/`MLA` `o_proj` + `L0` `KimiMLP.down_proj`
         deterministic 1-shot `psum`
       - `cid=5` (`'tp'`): `KimiMoE` `packed_moe_out` deterministic 1-shot `psum`
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.kimi_dma_mesh import DP_AXIS_NAME, TP_AXIS_NAME
from vllm_torchtpu.kernels.kimi_k3.exit_fold import (
    build_kimi_moe_ep_overlap_op,
    normalize_exit_fold_mode,
)
from vllm_torchtpu.kernels.kimi_k3.polish import normalize_compute_polish_mode
from vllm_torchtpu.kernels.kimi_k3.spec_kda import normalize_spec_kda_mode
from vllm_torchtpu.kernels.kimi_k3.pallas_dma import (
    COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
    COLLECTIVE_ID_MOE_DP_COMBINE_RS,
    COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    COLLECTIVE_ID_MOE_TP_EXIT_PSUM,
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
    DEFAULT_MAX_DMA_TOKENS,
    _mesh_shape_and_ids,
    _sanitize_tag,
    build_pallas_all_gather_cols_op,
    build_pallas_all_gather_rows_op,
    build_pallas_psum_op,
    build_pallas_psum_scatter_rows_op,
)
from vllm_torchtpu.logger import init_logger

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


def kimi_o2_variant_tag() -> str:
    """Return a cache-safe variant tag encoding all active O2/O4/O5 env flags (`D87`)."""
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
        raw_ef_tok = _sanitize_tag(raw_ef)
        if raw_ef_tok == ef_mode:
            tag = f"{base}_ef_{ef_mode}"
        else:
            tag = f"{base}_ef_{ef_mode}_{raw_ef_tok}"
    if polish_tag != "off":
        tag = f"{tag}_cp_{polish_tag}"
    if spec_kda != "off":
        tag = f"{tag}_sk_{spec_kda}"
    return tag


def _pack_dispatch_metadata(
    topk_ids_i32: torch.Tensor,
    topk_weights_bf16: torch.Tensor,
) -> torch.Tensor:
    """Pack ``topk_ids[B, K]`` (int32) + ``topk_weights[B, K]`` (bf16) into ``[B, 3*K]`` bf16.

    Why byte-valued normal ``bfloat16`` lanes instead of raw ``.view(torch.bfloat16)``:
    TPU floating-point / padding operations flush ``bfloat16`` subnormals to signed
    zero (FTZ), which corrupts raw-bitcast ``int32`` expert IDs ``1..127``
    (``0x0001..0x007F``) to ``0`` and canonicalizes ``-1`` (``0xFFFFFFFF``) NaN bits.
    Encoding signed 16-bit IDs (`-32768..32767`, covering `-1` and `0..895`) into two
    unsigned byte-valued ``bfloat16`` lanes ``0..255`` (matching
    ``kernels/kimi_k3/collectives.py:133-153``) uses the exact same ``2 * K`` ``bf16``
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


def _pack_dp_dispatch(
    latent_bf16: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_weights_bf16: torch.Tensor,
) -> torch.Tensor:
    """Pack ``(latent[B, D], topk_ids[B, K], topk_weights[B, K])`` -> ``[B, D + 3*K]`` bf16."""
    x_bf16 = (
        latent_bf16
        if latent_bf16.dtype == torch.bfloat16
        else latent_bf16.to(torch.bfloat16)
    )
    meta_bf16 = _pack_dispatch_metadata(topk_ids_i32, topk_weights_bf16)
    return torch.cat((x_bf16, meta_bf16), dim=-1)


def _unpack_dp_dispatch(
    packed_bf16: torch.Tensor,
    latent_dim: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack ``[B_global, D + 3*K]`` bf16 -> ``(latent_bf16, topk_ids_i32, topk_weights_bf16)``."""
    latent = packed_bf16[:, :latent_dim]
    low_i32 = packed_bf16[:, latent_dim : latent_dim + topk].to(torch.int32)
    high_i32 = packed_bf16[:, latent_dim + topk : latent_dim + 2 * topk].to(torch.int32)
    topk_ids = (((high_i32 << 8) | low_i32) << 16) >> 16
    topk_weights = packed_bf16[:, latent_dim + 2 * topk : latent_dim + 3 * topk]
    return latent, topk_ids, topk_weights


@dataclass(frozen=True)
class KimiO2Collectives:
    """Pre-built Phase O2/O4 `@jax_op` callables attached at model load time."""

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
        moe_module._o2_tp_latent_ag_op = self.tp_latent_ag_op
        moe_module._o2_dp_packed_ag_op = self.dp_packed_ag_op
        moe_module._o2_dp_combine_rs_op = self.dp_combine_rs_op
        moe_module._o2_tp_routed_psum_op = self.tp_routed_psum_op
        moe_module._o2_tp_moe_exit_psum_op = self.tp_moe_exit_psum_op
        can_attach_o4 = (
            self.exit_fold_mode != "off"
            and self.mesh is not None
            and getattr(moe_module, "w_routed_down_tp", None) is not None
            and getattr(moe_module, "w_shared_gu", None) is not None
            and getattr(moe_module, "packed_moe_out_weight", None) is not None
            and bool(getattr(moe_module, "use_ep", False))
        )
        if can_attach_o4:
            tp_rank = (
                int(torch.distributed.get_rank()) % self.tp_size
                if torch.distributed.is_initialized()
                else 0
            )
            dev = torch.device("cpu")
            for buf in (
                getattr(moe_module, "w_shared_gu", None),
                getattr(moe_module, "packed_moe_out_weight", None),
            ):
                if isinstance(buf, torch.Tensor) and buf.numel() > 0:
                    dev = buf.device
                    break
            else:
                for p in moe_module.parameters():
                    if p.numel() > 0:
                        dev = p.device
                        break
            moe_module.register_buffer(
                "_o2_tp_rank_buf",
                torch.tensor([[tp_rank]], dtype=torch.int32, device=dev),
                persistent=False,
            )
        else:
            moe_module._o2_tp_rank_buf = None

        if can_attach_o4:
            assert self.mesh is not None
            moe_module._o4_ep_overlap_op = build_kimi_moe_ep_overlap_op(
                self.mesh,
                moe_module,
                exit_fold_mode=self.exit_fold_mode,
                max_dma_tokens=self.max_dma_tokens,
                variant_tag=self.variant_tag,
            )
        else:
            moe_module._o4_ep_overlap_op = None

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


def build_kimi_o2_collectives(
    mesh: Mesh,
    *,
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> KimiO2Collectives:
    """Pre-build active O2/O4 `@jax_op` callables outside `torch.compile`."""
    dp_size, tp_size, _ = _mesh_shape_and_ids(mesh)
    vtag = kimi_o2_variant_tag()
    use_lat = bool(envs.TPU_KIMI_OPT_LATENCY_COLLECTIVES)
    tp_psum_mode = normalize_latency_tp_psum_mode(envs.TPU_KIMI_OPT_LATENCY_TP_PSUM)
    ef_mode = normalize_exit_fold_mode(envs.TPU_KIMI_OPT_EXIT_FOLD)

    tp_latent_ag_op = None
    dp_packed_ag_op = None
    dp_combine_rs_op = None
    tp_routed_psum_op = None
    tp_attn_mlp_psum_op = None
    tp_moe_exit_psum_op = None

    if ef_mode == "off":
        if use_lat:
            if tp_size > 1:
                tp_latent_ag_op = build_pallas_all_gather_cols_op(
                    mesh,
                    axis_name=TP_AXIS_NAME,
                    collective_id=COLLECTIVE_ID_MOE_TP_LATENT_AG,
                    recv_memory_space="auto",
                    max_dma_tokens=max_dma_tokens,
                    variant_tag=vtag,
                )
            if dp_size > 1:
                dp_packed_ag_op = build_pallas_all_gather_rows_op(
                    mesh,
                    axis_name=DP_AXIS_NAME,
                    collective_id=COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
                    input_sharding="dp_replicated",
                    recv_memory_space="auto",
                    max_dma_tokens=max_dma_tokens,
                    variant_tag=vtag,
                )

        if use_lat and dp_size > 1:
            dp_combine_rs_op = build_pallas_psum_scatter_rows_op(
                mesh,
                axis_name=DP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_DP_COMBINE_RS,
                recv_memory_space="auto",
                max_dma_tokens=max_dma_tokens,
                variant_tag=vtag,
            )

    if tp_size > 1 and tp_psum_mode != "off":
        enable_routed = False
        enable_all_exits = False
        psum_cap = max_dma_tokens
        if tp_psum_mode == "tp16":
            if tp_size >= 16:
                enable_routed = True
                enable_all_exits = True
                psum_cap = 8
        elif tp_psum_mode == "routed_only":
            enable_routed = True
            psum_cap = max_dma_tokens
        elif tp_psum_mode == "all":
            enable_routed = True
            enable_all_exits = True
            psum_cap = max_dma_tokens
        elif tp_psum_mode.isdigit():
            enable_routed = True
            enable_all_exits = True
            psum_cap = int(tp_psum_mode)

        if enable_routed and ef_mode == "off":
            tp_routed_psum_op = build_pallas_psum_op(
                mesh,
                axis_name=TP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_TP_ROUTED_PSUM,
                recv_memory_space="auto",
                max_dma_tokens=psum_cap,
                variant_tag=vtag,
            )
        if enable_all_exits:
            tp_attn_mlp_psum_op = build_pallas_psum_op(
                mesh,
                axis_name=TP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
                recv_memory_space="auto",
                max_dma_tokens=psum_cap,
                variant_tag=vtag,
            )
            tp_moe_exit_psum_op = build_pallas_psum_op(
                mesh,
                axis_name=TP_AXIS_NAME,
                collective_id=COLLECTIVE_ID_MOE_TP_EXIT_PSUM,
                recv_memory_space="auto",
                max_dma_tokens=psum_cap,
                variant_tag=vtag,
            )

    bundle = KimiO2Collectives(
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
            "Armed Kimi O2/O4 collectives | variant=%s | mesh=(dp=%d, tp=%d) | "
            "exit_fold=%s tp_lat_ag=%s dp_pack_ag=%s dp_rs=%s tp_psum=(routed=%s, exit=%s)",
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
