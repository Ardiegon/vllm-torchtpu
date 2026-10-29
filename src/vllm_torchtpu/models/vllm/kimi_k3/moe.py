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
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
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
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import make_latent_projection
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.utils import fresh_buffer

from .collective_ops import FusedPrefillCollectives
from .layers import KimiMLP

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    )


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
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts)
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
        self.register_buffer("packed_moe_out_weight", None, persistent=False)
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
        self.packed_moe_out_weight = None
        if (
            self.shared_group is not None
            or self.shared_experts is None
            or self.routed_expert_down_proj is None
            or self.routed_expert_up_proj is None
        ):
            return
        shared_gu = getattr(self.shared_experts, "gate_up_proj", None)
        shared_down = getattr(self.shared_experts, "down_proj", None)
        down_proj = self.routed_expert_down_proj
        up_proj = self.routed_expert_up_proj
        gate = self.gate
        layers = (shared_gu, down_proj, gate, shared_down, up_proj)
        if any(
            layer is None
            or getattr(layer, "weight", None) is None
            or layer.weight.dtype != torch.bfloat16
            or layer.weight.device.type != "tpu"
            or getattr(layer, "bias", None) is not None
            or not getattr(layer, WEIGHT_FLIPPED_ATTR, False)
            for layer in layers
        ):
            return
        w_shared_gu = shared_gu.weight
        w_down = down_proj.weight
        w_gate = gate.weight
        w_shared_down = shared_down.weight
        w_up = up_proj.weight
        hidden_size = w_shared_gu.shape[0]
        if w_down.shape[0] != hidden_size or w_gate.shape[0] != hidden_size:
            return
        if w_shared_down.shape[1] != hidden_size:
            return
        up_group = getattr(up_proj, "_chip_group", None)
        tp_rank = get_tensor_model_parallel_rank()
        latent_dim = w_up.shape[0]
        if self.tp_size > 1 and latent_dim % self.tp_size != 0:
            return
        latent_local = latent_dim // max(1, self.tp_size)
        with torch.no_grad():
            if up_group is not None:
                if up_group.world_size != self.tp_size:
                    return
                full_w_up = up_group.all_gather(w_up, dim=-1)
                rank = up_group.rank_in_group
            else:
                full_w_up = w_up
                rank = tp_rank
            if full_w_up.shape != (latent_dim, hidden_size):
                return
            w_up_row = full_w_up[
                rank * latent_local : (rank + 1) * latent_local, :
            ].contiguous()

            shared_gu_cols = w_shared_gu.shape[1]
            down_cols = w_down.shape[1]
            down_pad_cols = ((down_cols + 127) // 128) * 128
            if shared_gu_cols % 128 != 0:
                return

            packed_in = torch.cat(
                (
                    w_shared_gu,
                    w_down,
                    w_down.new_zeros((hidden_size, down_pad_cols - down_cols)),
                ),
                dim=-1,
            )
            self.packed_moe_in_weight = fresh_buffer(packed_in)
            self._moe_in_shared_cols = shared_gu_cols
            self._moe_in_down_cols = down_cols
            self._moe_in_down_pad_cols = down_pad_cols

            packed_out = torch.cat((w_shared_down, w_up_row), dim=0)
            self.packed_moe_out_weight = fresh_buffer(packed_out)
            self._moe_out_latent_start = rank * latent_local
            self._moe_out_latent_end = (rank + 1) * latent_local
            for layer in (shared_gu, down_proj, shared_down, up_proj):
                layer.weight.data = torch.empty(
                    0, dtype=w_shared_gu.dtype, device=w_shared_gu.device
                )

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        if sequence_parallel or self.shared_group is not None:
            return self._forward_expert_parallel(hidden_states, sequence_parallel)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if (
            getattr(self, "packed_moe_in_weight", None) is not None
            and getattr(self, "packed_moe_out_weight", None) is not None
        ):
            output = self._forward_packed(hidden_states)
            return output.view(num_tokens, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return output.view(num_tokens, hidden_size)

    def _forward_packed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        packed_in = hidden_states @ self.packed_moe_in_weight
        shared_gate_up, routed_down_pad = packed_in.split(
            [
                self._moe_in_shared_cols,
                self._moe_in_down_pad_cols,
            ],
            dim=-1,
        )
        routed_down_local = routed_down_pad[:, : self._moe_in_down_cols]
        down_group = getattr(self.routed_expert_down_proj, "_chip_group", None)
        if down_group is not None:
            latent = down_group.all_gather(routed_down_local, dim=-1)
        else:
            latent = routed_down_local
        routed = self.experts.routed_experts
        if self.use_ep and routed.moe_config.dp_size > 1:
            topk_weights, topk_ids = moe_routing.route(routed, latent, router_logits)
            routed_out = routed.quant_method.apply_with_routing(
                routed,
                latent,
                topk_weights,
                topk_ids,
                pipelined=True,
            )
            if self.tp_size > 1:
                routed_out = tensor_model_parallel_all_reduce(routed_out)
        else:
            routed_out = routed.quant_method.apply_monolithic(
                routed, latent, router_logits
            )
            if self.tp_size > 1 or self.use_ep:
                routed_out = tensor_model_parallel_all_reduce(routed_out)
        if self.routed_expert_norm is not None:
            routed_out = self.routed_expert_norm(routed_out)
        routed_local = routed_out[
            :, self._moe_out_latent_start : self._moe_out_latent_end
        ]
        assert self.shared_experts is not None
        shared_act = self.shared_experts.act_fn(shared_gate_up)
        combined_in = torch.cat((shared_act, routed_local), dim=-1)
        out_partial = combined_in @ self.packed_moe_out_weight
        if self.tp_size > 1:
            return tensor_model_parallel_all_reduce(out_partial)
        return out_partial

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
