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
"""Units for the Kimi K3 DSpark draft model: TP-sharded context projections,
the rank-major context-KV reorder, the position-major opt-in hook and the
fused context-KV weight duplication.
"""

import pytest
import torch

import vllm_torchtpu.models.vllm.kimi_k3.dspark as k3_dspark
from vllm_torchtpu.models.vllm.kimi_k3.dspark import (
    K3DSparkForCausalLM,
    _context_tp_shards,
    _duplicate_context_kv_weights,
    _layer_major_context_kv,
)


@pytest.mark.parametrize(
    "enabled,tp,hidden,kv_width,expected",
    [
        (True, 1, 64, 32, 1),
        (True, 2, 64, 32, 2),
        (True, 4, 64, 32, 4),
        (True, 4, 66, 32, 1),  # hidden size not divisible by TP
        (True, 4, 64, 30, 1),  # KV width not divisible by TP
        (False, 4, 64, 32, 1),
    ],
)
def test_context_tp_shards(monkeypatch, enabled, tp, hidden, kv_width, expected):
    monkeypatch.setattr(k3_dspark, "use_spec_o6b", lambda: enabled)
    assert _context_tp_shards(tp, hidden, kv_width) == expected


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_layer_major_context_kv_matches_replicated_layout(shards):
    tokens, layers, kv_width = 3, 5, 8
    layer_major = torch.randn(tokens, layers, kv_width)
    # What a gather_output=True projection sharded by output rows returns:
    # each rank holds a kv_width/shards slice of every layer, ranks in order.
    width = kv_width // shards
    gathered = torch.cat(
        [
            layer_major[:, :, r * width : (r + 1) * width].reshape(tokens, -1)
            for r in range(shards)
        ],
        dim=-1,
    )
    out = _layer_major_context_kv(gathered, shards, layers, kv_width)
    assert torch.equal(out, layer_major)


@pytest.mark.parametrize("enabled", [True, False])
def test_position_major_draft_logits_hook(monkeypatch, enabled):
    monkeypatch.setattr(k3_dspark, "use_spec_o6b", lambda: enabled)
    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    assert model.position_major_draft_logits() is enabled


def test_duplicate_context_kv_weights():
    names = [
        "layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "layers.0.self_attn.q_a_proj.weight",
        "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
        "layers.2.self_attn.kv_a_proj_with_mqa.weight",  # >= num_layers
        "markov_head.markov_w1.weight",
    ]
    weights = [(n, torch.full((2, 3), float(i))) for i, n in enumerate(names)]
    out = list(_duplicate_context_kv_weights(iter(weights), num_layers=2))
    assert [n for n, _ in out] == [
        names[0],
        "context_kv_proj.weight",
        names[1],
        names[2],
        "context_kv_proj.weight_scale_inv",
        names[3],
        names[4],
    ]
    assert out[1][1].shard_id == 0 and out[4][1].shard_id == 1
    assert torch.equal(out[1][1], weights[0][1])
    assert torch.equal(out[4][1], weights[2][1])
