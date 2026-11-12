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
"""Weight-only int8 (W8A16) linear layers.

Weights are quantized per output channel (symmetric int8, one f32 scale per
column of the `[K, N]` weight); activations stay in their dtype. On TPU the
matmul runs through `xla_quantized_matmul`; elsewhere a float reference.
Opt-in for Kimi K3's dense projections (`TPU_KIMI_DENSE_W8A16_B11`).
"""

from __future__ import annotations

from typing import cast

import jax
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.core.quantization import quantize_tensor
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

from .compressed_tensors.compressed_tensors_moe.utils import release_memory_to_os


def _w8a16_matmul_jax(x: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
    return xla_quantized_matmul(x, w_q, w_s, quantize_activation=False)


_w8a16_matmul_op = pallas.jax_op("pallas::kimi_w8a16_matmul", _w8a16_matmul_jax)
_w8a16_matmul_op.register_fake(
    lambda x, w_q, w_s: torch.empty(
        (x.shape[0], w_q.shape[1]), dtype=x.dtype, device=x.device
    )
)


def w8a16_matmul(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute weight-only W8A16 matmul (quantize_activation=False)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    if x_2d.device.type == "tpu":
        out_2d = _w8a16_matmul_op(x_2d, w_q, w_scale)
    else:
        out_2d = (torch.matmul(x_2d.float(), w_q.float()) * w_scale.unsqueeze(0)).to(
            x.dtype
        )
    if bias is not None:
        out_2d = out_2d + bias
    return out_2d.reshape(*orig_shape[:-1], w_q.shape[1])


def quantize_weight_w8a16(
    w_kn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [K, N] weight tensor per-output-channel (axis=0) for W8A16."""
    w_q, w_scale = quantize_tensor(w_kn, torch.int8, axis=0)
    return fresh_buffer(w_q), fresh_buffer(w_scale)


class W8A16LinearMethod(QuantizeMethodBase):
    """Weight-only W8A16 method installed on an already-loaded linear layer.

    `quantize_linear_layer_w8a16` swaps it in after the bf16 checkpoint
    weights are loaded, so it never creates weights itself.
    """

    supports_pre_processed_weights = True

    def create_weights(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q = cast(torch.Tensor, layer.weight)
        w_scale = cast(torch.Tensor, layer.weight_scale)
        return w8a16_matmul(x, w_q, w_scale, bias)


def quantize_linear_layer_w8a16(layer: nn.Module | None) -> list[torch.Tensor]:
    """Convert a loaded LinearBase or ParallelLMHead layer to W8A16 in-place.

    Returns the tensors to synchronize: the int8 weight and its scales, or
    nothing when the layer is absent or its weight was already released.
    """
    if layer is None:
        return []
    weight = getattr(layer, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor) or weight.numel() == 0:
        return []
    if weight.dtype == torch.int8:
        scale = getattr(layer, "weight_scale", None)
        return [weight, scale] if isinstance(scale, torch.Tensor) else [weight]
    w_data = weight.data
    is_flipped = getattr(layer, WEIGHT_FLIPPED_ATTR, False) and not getattr(
        layer, KEEP_VLLM_LAYOUT_ATTR, False
    )
    w_kn = w_data if is_flipped else w_data.t().contiguous()
    w_q, w_scale = quantize_weight_w8a16(w_kn)
    weight.data = torch.empty(0, dtype=w_data.dtype, device=w_data.device)
    del w_data, w_kn
    q_weight = nn.Parameter(w_q, requires_grad=False)
    q_scale = nn.Parameter(w_scale, requires_grad=False)
    layer.weight = q_weight
    layer.register_parameter("weight_scale", q_scale)
    setattr(layer, WEIGHT_FLIPPED_ATTR, True)
    layer.quant_method = W8A16LinearMethod()
    return [q_weight, q_scale]


def synchronize_quantized_weights(tensors: list[torch.Tensor]) -> None:
    """Materialize freshly quantized TPU weights, then return freed host memory."""
    tpu_tensors = [
        t for t in tensors if isinstance(t, torch.Tensor) and t.device.type == "tpu"
    ]
    if tpu_tensors:
        synchronize_tensors(tpu_tensors)
        release_memory_to_os()
