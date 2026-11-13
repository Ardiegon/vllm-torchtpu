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
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_tp_mesh import (
    build_dp_tp_mesh,
    require_dp_tp_mesh_order,
)
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.distributed.pcp import get_pcp_world_size
from vllm_torchtpu.kernels.megablox.gmm_v6e_planar_t_b11 import (
    FUSED_DECODE_MOE_MAX_TOKENS,
    FUSED_VERIFY_MOE_MAX_TOKENS,
    is_supported_by_planar_t_b11,
)
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.custom_ops.kimi_moe_ep_overlap_op import (
    MoeOverlapConfig,
    SharedExpertActivation,
    build_kimi_moe_ep_overlap_op,
)
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import (
    IntraChipColumnParallelLinear,
    make_latent_projection,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
    VllmCompressedTensorsW4ANMxfp4MoEMethod,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tpu_info import get_chip_version
from vllm_torchtpu.utils import fresh_buffer

from .collective_ops import FusedPrefillCollectives
from .layers import KimiMLP, SituAndMul

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.configs import TpuRoutedExperts

logger = init_logger(__name__)


def pack_dispatch_metadata(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Pack `topk_ids[B, K]` and `topk_weights[B, K]` into `[B, 3K]` bf16.

    The ids travel as two byte-valued bf16 lanes each (low byte, high byte of
    the int16 id, covering -1 and every expert id) rather than as raw int32
    bits viewed as bf16: TPU float ops flush bf16 subnormals to zero, which
    would turn the bit patterns of ids 1..127 into 0, and canonicalize the
    NaN pattern of -1. Byte values 0..255 are exact, normal bf16 numbers.
    """
    ids = topk_ids.to(torch.int32)
    low = (ids & 0xFF).to(torch.bfloat16)
    high = ((ids >> 8) & 0xFF).to(torch.bfloat16)
    return torch.cat((low, high, topk_weights.to(torch.bfloat16)), dim=-1)


def _unbiased_bf16_weight(layer: nn.Module) -> torch.Tensor | None:
    """The weight of an unquantized, unbiased bf16 linear layer, else None."""
    assert isinstance(layer, LinearBase)
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return None
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    if layer.bias is not None or weight.dtype != torch.bfloat16:
        return None
    return weight


def _kn_weight(layer: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """`weight` of `layer` in the `[n_in, n_out]` layout."""
    # Both markers exist on a layer only once they apply (see linear_common).
    if getattr(layer, WEIGHT_FLIPPED_ATTR, False) and not getattr(
        layer, KEEP_VLLM_LAYOUT_ATTR, False
    ):
        return weight
    return weight.t().contiguous()


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
        # Largest batch the fused small-batch routed-expert kernel serves:
        # speculative decoding verifies `1 + K` tokens per request.
        self._fused_decode_max_tokens = (
            FUSED_VERIFY_MOE_MAX_TOKENS
            if vllm_config is not None and vllm_config.speculative_config is not None
            else FUSED_DECODE_MOE_MAX_TOKENS
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
        # TPU_LATENT_PROJ_INTRA_CHIP_TP, or across TP on chips with one core.
        # These two are the model's largest unquantized weights and are read
        # in full by every rank each step.
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
        # The fused EP-overlap layer exit with its weights and TP rank
        # operand, set by process_weights_after_loading where it applies;
        # without the op the forward runs the experts unfused.
        self.packed_moe_out_weight: torch.Tensor | None
        self.w_routed_down_tp: torch.Tensor | None
        self.w_shared_gu: torch.Tensor | None
        self._tp_rank: torch.Tensor | None
        self.register_buffer("packed_moe_out_weight", None, persistent=False)
        self.register_buffer("w_routed_down_tp", None, persistent=False)
        self.register_buffer("w_shared_gu", None, persistent=False)
        self.register_buffer("_tp_rank", None, persistent=False)
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
        self._uses_ep_overlap = (
            self._resolve_ep_overlap_layout()
            and self._routed_experts_load_as_planar_t()
        )
        if self._uses_ep_overlap:
            # Requested at construction, before any layer builds its EP mesh
            # at weight loading: the overlap op's (dp, tp) mesh shares the
            # compiled module with the EP kernels.
            require_dp_tp_mesh_order()

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

    def _resolve_ep_overlap_layout(self) -> bool:
        """Whether the parallel layout fits the fused EP-overlap layer exit.

        It needs flat expert parallelism over a (dp, tp) grid of ranks, shared
        experts on the TP group, and the latent projections sharded over the
        whole TP group, so each rank holds its `latent_dim / tp` columns of
        the routed latent.
        """
        down_proj = self.routed_expert_down_proj
        up_proj = self.routed_expert_up_proj
        return (
            self.use_ep
            and self.tp_size > 1
            and self.shared_group is None
            and self.shared_experts is not None
            and isinstance(down_proj, IntraChipColumnParallelLinear)
            and isinstance(up_proj, IntraChipColumnParallelLinear)
            and down_proj.tp_size == self.tp_size
            and up_proj.tp_size == self.tp_size
            and get_dp_group().world_size > 1
            and get_pcp_world_size() == 1
            and hierarchical_split_or_none() is None
        )

    def _routed_experts_load_as_planar_t(self) -> bool:
        """Whether the routed experts will load in the planar_t MXFP4 layout.

        The EP-overlap exit runs them through the planar_t kernels. This is
        the choice `VllmCompressedTensorsW4ANMxfp4MoEMethod` makes in
        `process_weights_after_loading`, from the same inputs.
        """
        routed = cast("TpuRoutedExperts", self.experts.routed_experts)
        quant_method = routed.quant_method
        return (
            isinstance(quant_method, VllmCompressedTensorsW4ANMxfp4MoEMethod)
            and envs.MOE_REQUANTIZE_BLOCK_SIZE is None
            and is_supported_by_planar_t_b11(
                routed.w13_weight_packed.shape[-1] * 2,
                routed.w2_weight_packed.shape[-1] * 2,
                chip_version=get_chip_version(),
                activation=quant_method._resolve_tpu_activation(routed),
            )
        )

    def _routed_experts_are_planar_t(self) -> bool:
        """Whether the loaded routed experts are in the planar_t MXFP4 layout.

        The torch-side counterpart of the kernels' `is_planar_t_b11_layout`,
        which takes JAX arrays: only the planar_t loader stores uint8 weights
        with uint8 scales.
        """
        routed = cast("TpuRoutedExperts", self.experts.routed_experts)
        return all(
            tensor.dtype == torch.uint8
            for tensor in (
                routed.w13_weight,
                routed.w2_weight,
                routed.w13_weight_scale,
                routed.w2_weight_scale,
            )
        )

    def process_weights_after_loading(self) -> None:
        """Build the fused EP-overlap layer exit where it applies."""
        self.initialize_dummy_router()
        if self._uses_ep_overlap and self._pack_ep_overlap_weights():
            self._build_ep_overlap_op()

    def _pack_ep_overlap_weights(self) -> bool:
        """Set the weights the EP-overlap layer exit reads; False if it cannot.

        The op runs the shared gate/up matmul itself (`w_shared_gu`) on this
        rank's columns of the routed latent (`w_routed_down_tp`). One exit
        matmul with `packed_moe_out_weight` (shared down rows, then this
        rank's rows of the routed up projection) replaces the shared down and
        routed up projections and folds the up projection's
        `[T, hidden_size]` all-gather into the exit all-reduce. The four
        original weights are freed.
        """
        shared_experts = self.shared_experts
        down_proj = self.routed_expert_down_proj
        up_proj = self.routed_expert_up_proj
        assert shared_experts is not None
        assert isinstance(down_proj, IntraChipColumnParallelLinear)
        assert isinstance(up_proj, IntraChipColumnParallelLinear)
        if not self._routed_experts_are_planar_t():
            self._warn_ep_overlap_not_built(
                "the routed experts were not loaded in the planar_t MXFP4 layout"
            )
            return False
        layers = (
            shared_experts.gate_up_proj,
            down_proj,
            shared_experts.down_proj,
            up_proj,
        )
        weights: list[torch.Tensor] = []
        for layer in layers:
            weight = _unbiased_bf16_weight(layer)
            if weight is None:
                self._warn_ep_overlap_not_built(
                    "a shared-expert or latent projection is not an unquantized, "
                    "unbiased bf16 linear layer"
                )
                return False
            weights.append(weight)

        with torch.no_grad():
            w_shared_gu, w_down, w_shared_down, w_up = (
                _kn_weight(layer, weight.data)
                for layer, weight in zip(layers, weights, strict=True)
            )
            hidden_size = w_shared_gu.shape[0]
            latent_dim = w_up.shape[0]
            latent_local = latent_dim // self.tp_size
            assert w_down.shape == (hidden_size, latent_local), (
                f"[kimi_moe] routed down weight {tuple(w_down.shape)} is not this "
                f"rank's {(hidden_size, latent_local)} share of latent dim "
                f"{latent_dim} over TP {self.tp_size}"
            )
            assert latent_local * self.tp_size == latent_dim
            assert w_shared_down.shape[1] == hidden_size
            up_group = up_proj._chip_group
            full_w_up = up_group.all_gather(w_up, dim=-1)
            assert full_w_up.shape == (latent_dim, hidden_size), (
                f"[kimi_moe] gathered routed up weight {tuple(full_w_up.shape)} != "
                f"{(latent_dim, hidden_size)}"
            )
            rank = up_group.rank_in_group
            w_up_row = full_w_up[rank * latent_local : (rank + 1) * latent_local, :]
            self.w_routed_down_tp = fresh_buffer(w_down.contiguous())
            self.w_shared_gu = fresh_buffer(w_shared_gu.contiguous())
            self.packed_moe_out_weight = fresh_buffer(
                torch.cat((w_shared_down, w_up_row), dim=0)
            )
            for weight in weights:
                weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
        return True

    def _build_ep_overlap_op(self) -> None:
        """Build the fused EP-overlap op for the split entry weights."""
        assert self.shared_experts is not None and self.w_shared_gu is not None
        routed = cast("TpuRoutedExperts", self.experts.routed_experts)
        act_fn = self.shared_experts.act_fn
        if isinstance(act_fn, SituAndMul):
            shared_activation = SharedExpertActivation.SITU
            situ_beta = float(act_fn.beta)
            situ_linear_beta = (
                None if act_fn.linear_beta is None else float(act_fn.linear_beta)
            )
        elif isinstance(act_fn, SiluAndMul):
            shared_activation = SharedExpertActivation.SILU
            situ_beta = 1.0
            situ_linear_beta = None
        else:
            raise NotImplementedError(
                f"shared_experts activation {type(act_fn).__name__} is not supported "
                "by the EP-overlap op"
            )
        routed_activation = cast(
            VllmCompressedTensorsW4ANMxfp4MoEMethod, routed.quant_method
        )._tpu_activation_str
        assert routed_activation is not None, (
            "[kimi_moe] the routed experts must finish weight loading before "
            "the EP-overlap op is built"
        )
        norm = self.routed_expert_norm
        cfg = MoeOverlapConfig(
            routed_activation=routed_activation,
            topk=int(routed.moe_config.experts_per_token),
            use_norm=norm is not None,
            # Unused without the norm.
            rms_eps=float(norm.variance_epsilon) if norm is not None else 0.0,
            shared_activation=shared_activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            fused_decode_max_tokens=self._fused_decode_max_tokens,
        )
        # The op reads this rank's TP index from a runtime buffer so every
        # rank traces the same program.
        self._tp_rank = torch.tensor(
            [[get_tensor_model_parallel_rank()]],
            dtype=torch.int32,
            device=self.w_shared_gu.device,
        )
        self._ep_overlap_op = build_kimi_moe_ep_overlap_op(build_dp_tp_mesh(), cfg)

    def _warn_ep_overlap_not_built(self, reason: str) -> None:
        logger.warning_once(
            "[kimi_moe] The parallel layout selects the fused EP-overlap MoE "
            "layer exit, but it cannot be built: %s. The MoE layers run the "
            "shared and routed experts unfused instead (slower decode), on the "
            "rank-ordered EP mesh the layer exit requested.",
            reason,
        )

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        if sequence_parallel or self.shared_group is not None:
            return self._forward_expert_parallel(hidden_states, sequence_parallel)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if self._ep_overlap_op is not None:
            output = self._forward_ep_overlap(hidden_states)
            return output.view(num_tokens, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        assert isinstance(output, torch.Tensor)
        return output.view(num_tokens, hidden_size)

    def _forward_ep_overlap(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route, run the fused EP-overlap op, then the packed exit matmul."""
        ep_overlap_op = self._ep_overlap_op
        assert ep_overlap_op is not None
        assert self.w_routed_down_tp is not None
        assert self.w_shared_gu is not None
        assert self.packed_moe_out_weight is not None
        assert self._tp_rank is not None
        router_logits, _ = self.gate(hidden_states)
        routed = cast("TpuRoutedExperts", self.experts.routed_experts)
        topk_weights, topk_ids = moe_routing.route(routed, hidden_states, router_logits)
        topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
            topk_ids, topk_weights, is_local_tensor=True
        )
        packed_meta = pack_dispatch_metadata(topk_ids, topk_weights)
        routed_down_local = hidden_states @ self.w_routed_down_tp
        operands = [
            hidden_states,
            routed_down_local,
            packed_meta,
            self.w_shared_gu,
            routed.w13_weight[None, None],
            routed.w2_weight[None, None],
            routed.w13_weight_scale[None, None],
            routed.w2_weight_scale[None, None],
        ]
        if self.routed_expert_norm is not None:
            operands.append(self.routed_expert_norm.weight)
        assert routed._experts_start is not None
        operands += [routed._experts_start.reshape(1, 1), self._tp_rank]
        combined_in = ep_overlap_op(*operands)
        out_partial = combined_in @ self.packed_moe_out_weight
        # Sum the row-parallel exit projection's partial outputs over TP.
        return tensor_model_parallel_all_reduce(out_partial)

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
