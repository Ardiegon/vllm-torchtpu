# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.distributed.pcp import get_pcp_world_size
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import make_latent_projection
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.layers.adapter.quantization.weight_only_int8 import (
    quantize_weight_w8a16,
    w8a16_matmul,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer

from .collective_ops import FusedPrefillCollectives
from .latency_collective_ops import (
    pack_dispatch_metadata,
    pack_dp_dispatch,
    unpack_dp_dispatch,
)
from .layers import KimiMLP
from .opt_flags import is_exit_fold_active

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    )

logger = init_logger(__name__)

# A collective attached after weight loading: tensor in, tensor out.
CollectiveOp = Callable[[torch.Tensor], torch.Tensor]

# The packed entry projection pads the routed-down columns to a full MXU
# lane tile so the shared and routed column blocks start tile-aligned.
_MXU_LANES = 128


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        # Replicated or intra-chip sharded; both return ``(output, bias)``.
        up_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


class KimiMoE(nn.Module):
    """Standard or latent Kimi MoE backed by the fused TPU GMM custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        self.prefix = prefix
        hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )
        routed_expert_hidden_size = config.routed_expert_hidden_size
        expert_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # Keep the routing correction bias in f32 (checkpoint dtype). Created
        # under the model's default dtype it would be stored as bf16, which
        # rounds the bias (|err| up to ~2.4e-4 on real K3 layer-1 values,
        # comparable to the gaps between neighbouring expert scores) and
        # changes top-k selection relative to the HF reference.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )

        # Replicated by default; sharded across a chip's cores under
        # TPU_LATENT_PROJ_INTRA_CHIP_TP. These two are the model's largest
        # unquantized weights and are read in full by every rank each step.
        self.routed_expert_down_proj = (
            make_latent_projection(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            if routed_expert_hidden_size is not None
            else None
        )
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if self.routed_expert_down_proj is not None and config.latent_moe_use_norm
            else None
        )
        self.routed_expert_up_proj = (
            make_latent_projection(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            if self.routed_expert_down_proj is not None
            else None
        )
        routed_output_transform = (
            KimiRoutedOutputTransform(
                self.routed_expert_norm, self.routed_expert_up_proj
            )
            if self.routed_expert_up_proj is not None
            else None
        )

        self.prefill_group = None
        self.shared_group = None
        self.register_buffer("packed_moe_in_weight", None, persistent=False)
        self.register_buffer("packed_moe_in_scale", None, persistent=False)
        self.register_buffer("packed_moe_out_weight", None, persistent=False)
        self.register_buffer("packed_moe_out_scale", None, persistent=False)
        self.register_buffer("w_routed_down_tp", None, persistent=False)
        self.register_buffer("w_shared_gu", None, persistent=False)
        # Column layout of the packed projections (set by prepare_moe_projections).
        self._moe_in_shared_cols = 0
        self._moe_in_down_cols = 0
        self._moe_in_down_pad_cols = 0
        self._moe_out_latent_start = 0
        self._moe_out_latent_end = 0
        # One-shot DMA collectives and the fused EP-overlap op, attached by
        # KimiDmaCollectives.attach_to_moe after weight loading. None selects
        # the XLA collective (or, for the overlap op, the packed path).
        self._dma_tp_latent_ag_op: CollectiveOp | None = None
        self._dma_dp_packed_ag_op: CollectiveOp | None = None
        self._dma_dp_combine_rs_op: CollectiveOp | None = None
        self._dma_tp_routed_psum_op: CollectiveOp | None = None
        self._dma_tp_moe_exit_psum_op: CollectiveOp | None = None
        self._ep_overlap_op: Callable[..., torch.Tensor] | None = None
        if envs.TPU_K3_SP_PREFILL:
            self.prefill_group = FusedPrefillCollectives(prefix)
            self.shared_group = get_intra_chip_group()
            if self.shared_group is None or self.shared_group.world_size != 2:
                raise ValueError("K3 SP prefill needs a physical two-core chip group")

        self.shared_experts = (
            KimiMLP(
                hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                tp_group=self.shared_group,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            if config.num_shared_experts > 0
            else None
        )

        padded_intermediate_size = config.moe_intermediate_size
        min_per_partition = getattr(config, "min_moe_intermediate_per_partition", 128)
        if self.use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = self.tp_size
        if (
            moe_tp_size > 1
            and padded_intermediate_size // moe_tp_size < min_per_partition
        ):
            padded_intermediate_size = min_per_partition * moe_tp_size

        situ_beta = config.activation_situ_beta if config.hidden_act == "situ" else None
        situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts
        # with expert parallelism between chips and tensor parallelism
        # within one chip, instead of vLLM's flat all-experts-per-device EP.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
                renormalize=config.moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=cast(
                    torch.Tensor, self.gate.e_score_correction_bias
                ),
                routed_scaling_factor=config.routed_scaling_factor,
                routed_input_transform=self.routed_expert_down_proj,
                routed_output_transform=routed_output_transform,
            )
        if self.shared_group is not None and not hasattr(
            self.experts.routed_experts.quant_method, "apply_with_routing"
        ):
            raise ValueError("K3 SP prefill requires the native MXFP4 MoE backend")
        if padded_intermediate_size != config.moe_intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = cast(torch.Tensor, routed_experts.w13_weight_packed)
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = cast(torch.Tensor, routed_experts.w2_weight_packed)
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // moe_tp_size
            )

    def initialize_dummy_router(self) -> None:
        """Use fan-in random weights and fixed, distinct random layer biases.

        The generic dummy loader's +/-0.001 weights make the correction bias
        dominate sigmoid routing scores. Match the prototype's unit logit
        variance for normalized inputs without imposing any route counts.
        """
        config = get_current_vllm_config_or_none()
        if config is None or str(config.load_config.load_format).lower() != "dummy":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            1234 + zlib.crc32(self.prefix.encode())
        )
        bias_param = cast(torch.Tensor, self.gate.e_score_correction_bias)
        bias = (
            torch.randn(
                bias_param.shape,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        )
        weight_param = cast(torch.Tensor, self.gate.weight)
        weight = torch.randn(
            weight_param.shape, generator=generator, dtype=torch.float32
        )
        # The TPU dense loader transposes weights to [input, output]. Use
        # the logical fan-in rather than depending on the stored layout.
        weight *= self.gate.input_size**-0.5
        with torch.no_grad():
            bias_param.copy_(bias)
            weight_param.copy_(weight)

    def prepare_moe_projections(self) -> None:
        """Pack MoE entry (shared gate_up + routed down + router gate) and exit
        (shared down + row-sharded routed up) weights on TPU when eligible.

        Fuses 3 entry matmuls into 1, 2 exit matmuls into 1, and eliminates
        `routed_expert_up_proj`'s `[T, hidden_size]` all-gather by folding its
        row-parallel partial sum into the single exit `all_reduce`.
        """
        self.packed_moe_in_weight = None
        self.packed_moe_in_scale = None
        self.packed_moe_out_weight = None
        self.packed_moe_out_scale = None
        self.w_routed_down_tp = None
        self.w_shared_gu = None
        if (
            self.shared_group is not None
            or self.shared_experts is None
            or self.routed_expert_down_proj is None
            or self.routed_expert_up_proj is None
        ):
            return
        shared_gu = self.shared_experts.gate_up_proj
        shared_down = self.shared_experts.down_proj
        down_proj = self.routed_expert_down_proj
        up_proj = self.routed_expert_up_proj
        layers = (shared_gu, down_proj, shared_down, up_proj)
        # Quantized checkpoints keep their weights under other names (no
        # bf16 `weight`), which the packed bf16 projections cannot fuse.
        if any(
            getattr(layer, "weight", None) is None
            or layer.weight.dtype != torch.bfloat16
            or layer.weight.device.type != "tpu"
            or getattr(layer, "bias", None) is not None
            for layer in layers
        ):
            self._log_unpacked("projection weights are not unbiased bf16 on TPU")
            return

        def _as_kn(layer: nn.Module) -> torch.Tensor:
            w = layer.weight.data
            if getattr(layer, WEIGHT_FLIPPED_ATTR, False) and not getattr(
                layer, KEEP_VLLM_LAYOUT_ATTR, False
            ):
                return w
            return w.t().contiguous()

        with torch.no_grad():
            w_shared_gu = _as_kn(shared_gu)
            w_down = _as_kn(down_proj)
            w_shared_down = _as_kn(shared_down)
            w_up = _as_kn(up_proj)
            hidden_size = w_shared_gu.shape[0]
            if w_down.shape[0] != hidden_size or w_shared_down.shape[1] != hidden_size:
                self._log_unpacked(
                    f"hidden size mismatch: shared gate_up {tuple(w_shared_gu.shape)}, "
                    f"routed down {tuple(w_down.shape)}, "
                    f"shared down {tuple(w_shared_down.shape)}"
                )
                return
            up_group = getattr(up_proj, "_chip_group", None)
            tp_rank = get_tensor_model_parallel_rank()
            latent_dim = w_up.shape[0]
            if self.tp_size > 1 and latent_dim % self.tp_size != 0:
                self._log_unpacked(
                    f"latent dim {latent_dim} not divisible by TP {self.tp_size}"
                )
                return
            latent_local = latent_dim // max(1, self.tp_size)
            if up_group is not None:
                if up_group.world_size != self.tp_size:
                    self._log_unpacked(
                        f"up projection chip group size {up_group.world_size} "
                        f"!= TP {self.tp_size}"
                    )
                    return
                full_w_up = up_group.all_gather(w_up, dim=-1)
                rank = up_group.rank_in_group
            else:
                full_w_up = w_up
                rank = tp_rank
            if full_w_up.shape != (latent_dim, hidden_size):
                self._log_unpacked(
                    f"routed up weight {tuple(full_w_up.shape)} != "
                    f"{(latent_dim, hidden_size)}"
                )
                return
            w_up_row = full_w_up[
                rank * latent_local : (rank + 1) * latent_local, :
            ].contiguous()

            shared_gu_cols = w_shared_gu.shape[1]
            down_cols = w_down.shape[1]
            down_pad_cols = (down_cols + _MXU_LANES - 1) // _MXU_LANES * _MXU_LANES
            if shared_gu_cols % _MXU_LANES != 0:
                self._log_unpacked(
                    f"shared gate_up width {shared_gu_cols} is not a multiple of "
                    f"{_MXU_LANES}"
                )
                return

            self._moe_in_shared_cols = shared_gu_cols
            self._moe_in_down_cols = down_cols
            self._moe_in_down_pad_cols = down_pad_cols

            packed_out = torch.cat((w_shared_down, w_up_row), dim=0)
            self._moe_out_latent_start = rank * latent_local
            self._moe_out_latent_end = (rank + 1) * latent_local
            empty_dtype = w_shared_gu.dtype
            empty_device = w_shared_gu.device

            down_group = getattr(down_proj, "_chip_group", None)
            can_split_down_tp = (
                down_group is not None
                and down_group.world_size == self.tp_size
                and down_cols == latent_local
            ) or (down_group is None and down_cols == latent_dim)
            use_ep_overlap_split = (
                is_exit_fold_active()
                and self.use_ep
                and get_pcp_world_size() == 1
                and not envs.TPU_KIMI_DENSE_W8A16_B11
                and hierarchical_split_or_none() is None
                and can_split_down_tp
            )
            if use_ep_overlap_split:
                w_down_tp = (
                    w_down.contiguous()
                    if down_cols == latent_local
                    else w_down[
                        :, rank * latent_local : (rank + 1) * latent_local
                    ].contiguous()
                )
                self._moe_in_down_cols = latent_local
                self.w_routed_down_tp = fresh_buffer(w_down_tp)
                self.w_shared_gu = fresh_buffer(w_shared_gu.contiguous())
                self.packed_moe_out_weight = fresh_buffer(packed_out)
                del w_down_tp, packed_out, w_up_row, full_w_up
            else:
                packed_in = torch.cat(
                    (
                        w_shared_gu,
                        w_down,
                        w_down.new_zeros((hidden_size, down_pad_cols - down_cols)),
                    ),
                    dim=-1,
                )
                if envs.TPU_KIMI_DENSE_W8A16_B11:
                    in_q, in_s = quantize_weight_w8a16(packed_in)
                    out_q, out_s = quantize_weight_w8a16(packed_out)
                    self.packed_moe_in_weight = in_q
                    self.packed_moe_in_scale = in_s
                    self.packed_moe_out_weight = out_q
                    self.packed_moe_out_scale = out_s
                else:
                    self.packed_moe_in_weight = fresh_buffer(packed_in)
                    self.packed_moe_out_weight = fresh_buffer(packed_out)
                del packed_in, packed_out, w_up_row, full_w_up
            for layer in layers:
                layer.weight.data = torch.empty(
                    0, dtype=empty_dtype, device=empty_device
                )

    def _log_unpacked(self, reason: str) -> None:
        logger.warning_once(
            "Kimi K3 MoE keeps the unpacked shared/latent projections: %s", reason
        )

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        if sequence_parallel or self.shared_group is not None:
            return self._forward_expert_parallel(hidden_states, sequence_parallel)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if (
            self.w_routed_down_tp is not None
            and self.w_shared_gu is not None
            and self.packed_moe_out_weight is not None
            and self._ep_overlap_op is not None
        ):
            output = self._forward_async_ep_unified(hidden_states)
            return output.view(num_tokens, hidden_size)
        if (
            self.packed_moe_in_weight is not None
            and self.packed_moe_out_weight is not None
        ):
            output = self._forward_packed(hidden_states)
            return output.view(num_tokens, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return output.view(num_tokens, hidden_size)

    def _tp_all_reduce(
        self, x: torch.Tensor, dma_op: CollectiveOp | None
    ) -> torch.Tensor:
        """TP psum through the attached one-shot DMA op, else the XLA all-reduce."""
        if dma_op is not None:
            return dma_op(x)
        return tensor_model_parallel_all_reduce(x)

    def _exit_all_reduce(self, out_partial: torch.Tensor) -> torch.Tensor:
        """Sum the row-parallel exit projection's partial outputs over TP."""
        if self.tp_size > 1:
            return self._tp_all_reduce(out_partial, self._dma_tp_moe_exit_psum_op)
        return out_partial

    def _forward_async_ep_unified(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ep_overlap_op = self._ep_overlap_op
        assert ep_overlap_op is not None
        assert self.w_routed_down_tp is not None
        assert self.packed_moe_out_weight is not None
        router_logits, _ = self.gate(hidden_states)
        routed = self.experts.routed_experts
        topk_weights, topk_ids = moe_routing.route(routed, hidden_states, router_logits)
        topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
            topk_ids, topk_weights, is_local_tensor=True
        )
        packed_meta = pack_dispatch_metadata(topk_ids, topk_weights)
        routed_down_local = hidden_states @ self.w_routed_down_tp
        combined_in = ep_overlap_op(
            hidden_states,
            routed_down_local,
            packed_meta,
            self,
        )
        out_partial = combined_in @ self.packed_moe_out_weight
        return self._exit_all_reduce(out_partial)

    def _forward_packed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Packed entry matmul, latent MoE dispatch, packed exit matmul."""
        router_logits, _ = self.gate(hidden_states)
        packed_in = self._packed_matmul(
            hidden_states, self.packed_moe_in_weight, self.packed_moe_in_scale
        )
        shared_gate_up, routed_down_pad = packed_in.split(
            [
                self._moe_in_shared_cols,
                self._moe_in_down_pad_cols,
            ],
            dim=-1,
        )
        routed_down_local = routed_down_pad[:, : self._moe_in_down_cols]
        routed_out = self._dispatch_latent_moe(
            hidden_states, router_logits, routed_down_local
        )
        if self.routed_expert_norm is not None:
            routed_out = self.routed_expert_norm(routed_out)
        routed_local = routed_out[
            :, self._moe_out_latent_start : self._moe_out_latent_end
        ]
        assert self.shared_experts is not None
        shared_act = self.shared_experts.act_fn(shared_gate_up)
        combined_in = torch.cat((shared_act, routed_local), dim=-1)
        out_partial = self._packed_matmul(
            combined_in, self.packed_moe_out_weight, self.packed_moe_out_scale
        )
        return self._exit_all_reduce(out_partial)

    @staticmethod
    def _packed_matmul(
        x: torch.Tensor, weight: torch.Tensor | None, scale: torch.Tensor | None
    ) -> torch.Tensor:
        """``x @ weight``, or the W8A16 matmul when the weight has a scale."""
        assert weight is not None
        if scale is not None:
            return w8a16_matmul(x, weight, scale)
        return x @ weight

    def _dispatch_latent_moe(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        routed_down_local: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the latent across TP, run the routed experts, reduce over TP.

        Returns the full routed-expert latent output on every TP rank. With
        EP across DP ranks the tokens are routed here and dispatched either as
        one packed DP all-gather (`TPU_KIMI_OPT_PACKED_DP_AG`) or by the
        pipelined EP experts; otherwise the experts route monolithically.
        """
        down_group = getattr(self.routed_expert_down_proj, "_chip_group", None)
        routed = self.experts.routed_experts
        dp_size = routed.moe_config.dp_size if self.use_ep else 1
        pcp_size = get_pcp_world_size() if self.use_ep else 1
        use_ep_dispatch = self.use_ep and dp_size > 1
        use_packed_dp = bool(
            use_ep_dispatch and pcp_size == 1 and envs.TPU_KIMI_OPT_PACKED_DP_AG
        )

        # Route before the latent all-gather: route() reads only the dtype of
        # its second tensor argument, so the gather does not have to finish.
        routing = (
            moe_routing.route(routed, hidden_states, router_logits)
            if use_ep_dispatch
            else None
        )
        if down_group is not None:
            tp_latent_ag_op = self._dma_tp_latent_ag_op
            latent = (
                tp_latent_ag_op(routed_down_local)
                if tp_latent_ag_op is not None
                else down_group.all_gather(routed_down_local, dim=-1)
            )
        else:
            latent = routed_down_local

        if use_packed_dp:
            assert routing is not None
            topk_weights, topk_ids = routing
            topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
                topk_ids, topk_weights, is_local_tensor=True
            )
            packed_payload = pack_dp_dispatch(latent, topk_ids, topk_weights)
            dp_packed_ag_op = self._dma_dp_packed_ag_op
            gathered_payload = (
                dp_packed_ag_op(packed_payload)
                if dp_packed_ag_op is not None
                else get_dp_group().all_gather(packed_payload, dim=0)
            )
            ag_latent, ag_ids, ag_weights = unpack_dp_dispatch(
                gathered_payload, latent.shape[-1], topk_ids.shape[-1]
            )
            moe_out = routed.quant_method.apply_with_routing(
                routed,
                ag_latent,
                ag_weights,
                ag_ids,
                pipelined=False,
                mask_padding=False,
            )
            dp_combine_rs_op = self._dma_dp_combine_rs_op
            routed_out = (
                dp_combine_rs_op(moe_out)
                if dp_combine_rs_op is not None
                else get_dp_group().reduce_scatter(moe_out, dim=0)
            )
            if self.tp_size > 1:
                routed_out = self._tp_all_reduce(
                    routed_out, self._dma_tp_routed_psum_op
                )
        elif use_ep_dispatch:
            assert routing is not None
            topk_weights, topk_ids = routing
            routed_out = routed.quant_method.apply_with_routing(
                routed,
                latent,
                topk_weights,
                topk_ids,
                pipelined=True,
            )
            if self.tp_size > 1:
                routed_out = self._tp_all_reduce(
                    routed_out, self._dma_tp_routed_psum_op
                )
        else:
            routed_out = routed.quant_method.apply_monolithic(
                routed, latent, router_logits
            )
            # Under EP with one TP rank the experts' partial sums still need
            # the (XLA) all-reduce over the EP ranks.
            if self.tp_size > 1 or self.use_ep:
                routed_out = self._tp_all_reduce(
                    routed_out,
                    self._dma_tp_routed_psum_op if self.tp_size > 1 else None,
                )
        return routed_out

    def _forward_expert_parallel(
        self, hidden_states: torch.Tensor, sequence_parallel: bool
    ) -> torch.Tensor:
        """Run EP experts with sharded or replicated tokens, then add shared experts."""
        group = self.prefill_group
        assert group is not None
        router_logits, _ = self.gate(hidden_states)
        latent = hidden_states
        if self.routed_expert_down_proj is not None:
            latent, _ = self.routed_expert_down_proj(latent)
        routed = self.experts.routed_experts
        if sequence_parallel:
            weights, ids = moe_routing.route(routed, latent, router_logits)
            latent, weights, ids = group.gather_moe_inputs(
                latent, weights, ids, num_experts=self.gate.output_size
            )
            output = cast(
                "VllmCompressedTensorsW4ANMxfp4MoEMethod", routed.quant_method
            ).apply_with_routing(routed, latent, weights, ids)
        else:
            output = routed.quant_method.apply_monolithic(routed, latent, router_logits)
        output = (
            group.reduce_scatter(output, dim=0)
            if sequence_parallel
            else group.all_reduce(output)
        )
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm(output)
        if self.routed_expert_up_proj is not None:
            output, _ = self.routed_expert_up_proj(output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(
                hidden_states, sequence_parallel=sequence_parallel
            )
        return output
