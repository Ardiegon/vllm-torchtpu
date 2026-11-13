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
the rank-major context-KV reorder, the fused context-KV weight duplication
and the vocab-sharded Markov head.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu.models.vllm.kimi_k3 import dspark as k3_dspark
from vllm_torchtpu.models.vllm.kimi_k3.dspark import (
    K3DSparkForCausalLM,
    _context_tp_shards,
    _duplicate_context_kv_weights,
    _layer_major_context_kv,
)
from vllm_torchtpu.spec_decode.dspark import VocabParallelMarkovDraft


@pytest.mark.parametrize(
    "tp,hidden,kv_width,expected",
    [
        (1, 64, 32, 1),
        (2, 64, 32, 2),
        (4, 64, 32, 4),
        (4, 66, 32, 1),  # hidden size not divisible by TP
        (4, 64, 30, 1),  # KV width not divisible by TP
    ],
)
def test_context_tp_shards(tp, hidden, kv_width, expected):
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


_VOCAB, _RANK, _TP = 24, 3, 4


def _draft_with_markov_head(
    tp=_TP, rank=2, padding=0, soft_cap=None, scale=1.0, org_vocab=_VOCAB
):
    """A K3 draft holding only what the Markov-head sharding reads."""
    shard = _VOCAB // tp
    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(lm_head)
    lm_head.tp_size = tp
    lm_head.shard_indices = SimpleNamespace(
        org_vocab_start_index=rank * shard,
        org_vocab_end_index=(rank + 1) * shard,
        num_org_vocab_padding=padding,
    )
    draft = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.lm_head = lm_head
    draft.logits_processor = SimpleNamespace(
        soft_cap=soft_cap, scale=scale, org_vocab_size=org_vocab
    )
    w2 = torch.arange(_VOCAB * _RANK, dtype=torch.float32).reshape(_VOCAB, _RANK)
    draft.model = SimpleNamespace(
        markov_head=SimpleNamespace(markov_w2=SimpleNamespace(weight=w2))
    )
    draft._markov_head_vocab_sharded = False
    return draft


def test_kimi_k3_draft_opts_in_to_vocab_parallel_drafting():
    draft = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    assert isinstance(draft, VocabParallelMarkovDraft)


def test_shard_markov_head_keeps_the_local_vocab_rows():
    draft = _draft_with_markov_head()
    full = draft.model.markov_head.markov_w2.weight.clone()

    assert draft.shard_markov_head() == 12

    assert torch.equal(draft.model.markov_head.markov_w2.weight, full[12:18])
    markov_embed = torch.randn(5, _RANK)
    assert torch.equal(
        draft.markov_bias_local(markov_embed), markov_embed @ full[12:18].T
    )
    # The full-vocab bias no longer exists on this rank.
    with pytest.raises(RuntimeError, match="markov_bias_local"):
        draft.markov_bias(markov_embed)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(padding=4),
        dict(soft_cap=30.0),
        dict(scale=0.5),
        dict(org_vocab=_VOCAB - 1),
    ],
    ids=["vocab_padding", "soft_cap", "scale", "vocab_mismatch"],
)
def test_shard_markov_head_warns_and_stays_replicated(overrides):
    draft = _draft_with_markov_head(**overrides)
    full = draft.model.markov_head.markov_w2.weight
    with mock.patch.object(k3_dspark, "logger") as log:
        assert draft.shard_markov_head() is None
    log.warning_once.assert_called_once()
    assert draft.model.markov_head.markov_w2.weight is full
    assert draft._markov_head_vocab_sharded is False


def test_shard_markov_head_is_a_no_op_without_tensor_parallelism():
    draft = _draft_with_markov_head(tp=1, rank=0)
    with mock.patch.object(k3_dspark, "logger") as log:
        assert draft.shard_markov_head() is None
    log.warning_once.assert_not_called()


@pytest.mark.parametrize(
    ("spec_tokens", "max_model_len", "expected_rows"),
    [
        (None, 8192, 8320),
        (7, 8192, 8320),
        (200, 8192, 8448),
    ],
)
def test_mla_deepseek_yarn_applies_mscale_squared_and_trims_rope_cache(
    monkeypatch, spec_tokens, max_model_len, expected_rows
):
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        yarn_get_mscale,
    )
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    from vllm_torchtpu.models.vllm.kimi_k3 import attention as kimi_attention

    class FakeLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, *args, **kwargs):
            super().__init__()
            del args, kwargs
            out = sum(output_size) if isinstance(output_size, list) else output_size
            self.weight = torch.nn.Parameter(torch.empty(out, input_size))

    class FakeNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.variance_epsilon = eps
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    captured = {}

    class FakeMLAWrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["softmax_scale"] = args[2]
            captured["mla_modules"] = args[8]

    full_table = torch.arange(32768 * 64, dtype=torch.float32).reshape(32768, 64)
    rope_obj = SimpleNamespace(cos_sin_cache=full_table.clone())

    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RMSNorm", FakeNorm)
    monkeypatch.setattr(
        kimi_attention, "MultiHeadLatentAttentionWrapper", FakeMLAWrapper
    )
    monkeypatch.setattr(kimi_attention, "get_rope", lambda *a, **k: rope_obj)
    monkeypatch.setattr(kimi_attention.envs, "TPU_K3_SP_PREFILL", False)

    config = KimiLinearConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=1,
        q_lora_rank=64,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        max_position_embeddings=1048576,
        rope_parameters={
            "rope_type": "yarn",
            "apply_yarn_scaling": True,
            "factor": 32.0,
            "mscale_all_dim": 1.0,
        },
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        cache_config=SimpleNamespace(),
        quant_config=None,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_tokens)
            if spec_tokens is not None
            else None
        ),
    )

    kimi_attention.MultiHeadLatentAttention(
        config=config,
        vllm_config=vllm_config,
        prefix="model.layers.0.self_attn",
        use_rope=True,
    )

    mscale = float(yarn_get_mscale(32.0, 1.0))
    expected_scale = ((64 + 64) ** -0.5) * mscale * mscale
    assert captured["softmax_scale"] == pytest.approx(expected_scale)
    trimmed = captured["mla_modules"].rotary_emb.cos_sin_cache
    assert trimmed.shape == (expected_rows, 64)
    assert torch.equal(trimmed, full_table[:expected_rows])
