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
"""Weight-only int8 (W8A16) linear helpers, CPU reference path."""

import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.weight_only_int8 import (
    W8A16LinearMethod,
    quantize_linear_layer_w8a16,
    quantize_weight_w8a16,
    synchronize_quantized_weights,
    w8a16_matmul,
)


def _weight(k, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, n, generator=g, dtype=torch.float32).to(torch.bfloat16)


def test_quantize_weight_is_per_output_channel_int8():
    w_f32 = _weight(64, 32).float()
    w_q32, w_s32 = quantize_weight_w8a16(w_f32)
    assert torch.equal(w_q32.abs().amax(0), torch.full((32,), 127, dtype=torch.int8))
    assert w_s32.shape == (1, 32) and w_s32.dtype == torch.float32

    w = _weight(64, 32)
    w_q, w_s = quantize_weight_w8a16(w)
    assert w_q.dtype == torch.int8 and w_q.shape == (64, 32)
    assert w_s.shape == (1, 32) and w_s.dtype == torch.float32
    recon = w_q.float() * w_s
    assert torch.all((recon - w.float()).abs().amax(0) <= w_s.squeeze(0) * 1.5)


@pytest.mark.parametrize("lead_shape", [(5,), (2, 3)])
def test_matmul_matches_bf16_within_int8_error(lead_shape):
    w = _weight(64, 32, seed=1)
    w_q, w_s = quantize_weight_w8a16(w)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(*lead_shape, 64, generator=g).to(torch.bfloat16)
    out = w8a16_matmul(x, w_q, w_s)
    ref = (x.float() @ w.float()).to(torch.bfloat16)
    assert out.shape == (*lead_shape, 32) and out.dtype == torch.bfloat16
    # Per-channel step is max|w|/127; the error grows with sqrt(K) * |x|.
    bound = (w.float().abs().amax(0) / 127 * 64**0.5 * 4).max().item()
    assert (out.float() - ref.float()).abs().max().item() < bound


def test_matmul_adds_bias():
    w = _weight(16, 8, seed=3)
    w_q, w_s = quantize_weight_w8a16(w)
    x = torch.ones(2, 16, dtype=torch.bfloat16)
    bias = torch.arange(8, dtype=torch.bfloat16)
    assert torch.equal(
        w8a16_matmul(x, w_q, w_s, bias), w8a16_matmul(x, w_q, w_s) + bias
    )


class _Linear(nn.Module):
    """Minimal loaded linear layer: weight stored [N, K] like vLLM's."""

    def __init__(self, k, n):
        super().__init__()
        self.weight = nn.Parameter(_weight(n, k, seed=4), requires_grad=False)
        self.quant_method = None


def test_quantize_linear_layer_swaps_in_the_w8a16_method():
    layer = _Linear(16, 8)
    w_kn = layer.weight.data.t().contiguous()
    tensors = quantize_linear_layer_w8a16(layer)
    assert isinstance(layer.quant_method, W8A16LinearMethod)
    assert layer.weight.dtype == torch.int8 and layer.weight.shape == (16, 8)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR)
    assert tensors == [layer.weight, layer.weight_scale]
    expected_q, expected_s = quantize_weight_w8a16(w_kn)
    assert torch.equal(layer.weight, expected_q)
    assert torch.equal(layer.weight_scale, expected_s)
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    assert torch.equal(
        layer.quant_method.apply(layer, x), w8a16_matmul(x, expected_q, expected_s)
    )


def test_quantize_linear_layer_is_idempotent_and_skips_absent_layers():
    layer = _Linear(16, 8)
    first = quantize_linear_layer_w8a16(layer)
    assert quantize_linear_layer_w8a16(layer) == first
    assert quantize_linear_layer_w8a16(None) == []
    released = _Linear(16, 8)
    released.weight.data = torch.empty(0, dtype=torch.bfloat16)
    assert quantize_linear_layer_w8a16(released) == []


def test_synchronize_ignores_host_tensors():
    synchronize_quantized_weights([torch.zeros(2)])


def test_kimi_k3_modules_import_the_moved_helpers():
    from vllm_torchtpu.models.vllm.kimi_k3 import attention, model, moe

    assert moe.w8a16_matmul is w8a16_matmul
    assert attention.quantize_weight_w8a16 is quantize_weight_w8a16
    assert model.quantize_linear_layer_w8a16 is quantize_linear_layer_w8a16
