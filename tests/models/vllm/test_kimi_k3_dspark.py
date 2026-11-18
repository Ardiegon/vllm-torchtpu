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
"""Unit tests for TP-sharded context projections in the Kimi K3 DSpark drafter."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.kimi_k3 import dspark as dspark_mod
from vllm_torchtpu.models.vllm.kimi_k3.dspark import (
    _context_tp_shards,
    _layer_major_context_kv,
)


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


def _build_dspark_model_for_tp_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int,
    hidden_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    full_w_ctx: torch.Tensor,
    full_w_kv: torch.Tensor,
) -> tuple[dspark_mod.K3DSparkModel, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Construct `K3DSparkModel` with lightweight linear/MLA stand-ins."""
    num_layers = 2
    mla_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeReplicated(nn.Module):
        def __init__(self, input_size, output_size, *, bias, quant_config, prefix):
            super().__init__()
            del input_size, output_size, bias, quant_config, prefix

        def forward(self, x: torch.Tensor):
            return x @ full_w_ctx.t(), None

    class FakeColumnParallel(nn.Module):
        def __init__(
            self,
            input_size,
            output_size,
            *,
            bias,
            gather_output,
            quant_config,
            prefix,
        ):
            super().__init__()
            del input_size, bias, quant_config, prefix
            self.gather_output = gather_output
            self.shard = output_size // tp_size

        def forward(self, x: torch.Tensor):
            assert self.gather_output is True
            shards = [
                x @ full_w_ctx[r * self.shard : (r + 1) * self.shard].t()
                for r in range(tp_size)
            ]
            return torch.cat(shards, dim=-1), None

    class FakeMergedColumnParallel(nn.Module):
        def __init__(
            self,
            input_size,
            output_sizes,
            *,
            bias,
            gather_output,
            quant_config,
            prefix,
            disable_tp,
        ):
            super().__init__()
            del input_size, bias, quant_config, prefix
            self.output_sizes = list(output_sizes)
            self.gather_output = gather_output
            self.disable_tp = disable_tp

        def forward(self, x: torch.Tensor):
            if self.disable_tp:
                assert self.gather_output is False
                return x @ full_w_kv.t(), None
            assert self.gather_output is True
            # Each rank owns `kv_width // tp_size` rows of each layer's slice;
            # `gather_output=True` concatenates ranks along the output axis.
            kv_w = self.output_sizes[0]
            part = kv_w // tp_size
            rank_outputs = []
            for r in range(tp_size):
                layer_slices = [
                    full_w_kv[
                        layer_idx * kv_w + r * part : layer_idx * kv_w + (r + 1) * part
                    ]
                    for layer_idx in range(len(self.output_sizes))
                ]
                rank_outputs.append(x @ torch.cat(layer_slices, dim=0).t())
            return torch.cat(rank_outputs, dim=-1), None

    class FakeDecoderLayer(nn.Module):
        def __init__(self, *, config, vllm_config, layer_idx, start_layer_id, prefix):
            super().__init__()
            del config, vllm_config, layer_idx, start_layer_id, prefix
            core = SimpleNamespace(
                kv_cache=torch.empty(1, dtype=torch.float32),
                kv_cache_quantized_dtype=None,
                mla_op=lambda _cache, _qn, _qp, kv_c, k_pe, *_meta: mla_calls.append(
                    (kv_c, k_pe)
                ),
            )
            wrapper = SimpleNamespace(
                mla_attn=core,
                kv_a_layernorm=nn.Identity(),
                num_heads=2,
                rotary_emb=lambda _pos, q, k: (q, k),
            )
            self.self_attn = SimpleNamespace(mla_attn=wrapper)

    monkeypatch.setattr(
        dspark_mod, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(dspark_mod, "get_draft_quant_config", lambda _cfg: None)
    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", FakeReplicated)
    monkeypatch.setattr(dspark_mod, "ColumnParallelLinear", FakeColumnParallel)
    monkeypatch.setattr(
        dspark_mod, "MergedColumnParallelLinear", FakeMergedColumnParallel
    )
    monkeypatch.setattr(dspark_mod, "RMSNorm", lambda _dim, _eps: nn.Identity())
    monkeypatch.setattr(dspark_mod, "K3DSparkDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(
        dspark_mod, "DSparkMarkovHead", lambda *_args, **_kwargs: nn.Identity()
    )

    hf_config = SimpleNamespace(
        hidden_size=hidden_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        target_hidden_size=8,
        num_target_layers=2,
        rms_norm_eps=1e-5,
        num_hidden_layers=num_layers,
        vocab_size=32,
        draft_vocab_size=16,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        )
    )
    model = dspark_mod.K3DSparkModel(
        vllm_config=vllm_config, start_layer_id=0, prefix="draft"
    )
    return model, mla_calls


def test_k3_dspark_model_tp_sharded_context_matches_replicated(monkeypatch):
    """`K3DSparkModel` TP-shards divisible context projections and preserves KV."""
    torch.manual_seed(7)
    hidden_size, kv_lora_rank, qk_rope_head_dim = 16, 8, 4
    kv_width = kv_lora_rank + qk_rope_head_dim
    context_in = 8 * 2
    full_w_ctx = torch.randn(hidden_size, context_in, dtype=torch.float32)
    full_w_kv = torch.randn(2 * kv_width, hidden_size, dtype=torch.float32)
    target_hs = torch.randn(3, context_in, dtype=torch.float32)
    positions = torch.arange(3)
    meta_entry = SimpleNamespace(
        seq_lens=torch.tensor([3]),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3]),
        request_distribution=torch.tensor([0, 0, 1]),
    )
    metadata = (meta_entry, meta_entry)

    rep_model, rep_calls = _build_dspark_model_for_tp_test(
        monkeypatch,
        tp_size=1,
        hidden_size=hidden_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        full_w_ctx=full_w_ctx,
        full_w_kv=full_w_kv,
    )
    assert rep_model.context_tp_shards == 1
    assert rep_model.context_kv_proj.disable_tp is True
    rep_ctx = rep_model.precompute_and_store_context_kv(target_hs, positions, metadata)

    tp_model, tp_calls = _build_dspark_model_for_tp_test(
        monkeypatch,
        tp_size=2,
        hidden_size=hidden_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        full_w_ctx=full_w_ctx,
        full_w_kv=full_w_kv,
    )
    assert tp_model.context_tp_shards == 2
    assert getattr(tp_model.context_proj, "gather_output", False) is True
    assert tp_model.context_kv_proj.disable_tp is False
    assert tp_model.context_kv_proj.gather_output is True
    tp_ctx = tp_model.precompute_and_store_context_kv(target_hs, positions, metadata)

    torch.testing.assert_close(tp_ctx, rep_ctx, rtol=0, atol=0)
    assert len(tp_calls) == len(rep_calls) == 2
    for (tp_kvc, tp_kpe), (rep_kvc, rep_kpe) in zip(tp_calls, rep_calls, strict=True):
        torch.testing.assert_close(tp_kvc, rep_kvc, rtol=0, atol=0)
        torch.testing.assert_close(tp_kpe, rep_kpe, rtol=0, atol=0)

    # Indivisible KV width (10 % 4 != 0) falls back cleanly to replicated.
    indiv_model, _ = _build_dspark_model_for_tp_test(
        monkeypatch,
        tp_size=4,
        hidden_size=16,
        kv_lora_rank=6,
        qk_rope_head_dim=4,
        full_w_ctx=full_w_ctx,
        full_w_kv=torch.randn(2 * 10, 16),
    )
    assert indiv_model.context_tp_shards == 1
    assert indiv_model.context_kv_proj.disable_tp is True
