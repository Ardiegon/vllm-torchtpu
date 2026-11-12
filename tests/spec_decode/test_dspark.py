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
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import vllm.distributed

from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict, num_speculative_tokens=4, **spec_fields):
    # The vLLM 0.29 drafting fields at their defaults; tests override them.
    fields = dict(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )
    fields.update(spec_fields)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
        **fields,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DSparkProposer(runner=runner, vllm_config=vllm_config)


_LAYER_IDS = {"dspark_target_layer_ids": [1, 15, 30]}


def test_noise_resolution_and_dense_block():
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 4,
            **_LAYER_IDS,
        }
    )
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer(
        {
            "mask_token_id": 5,
            "sample_from_anchor": False,
            **_LAYER_IDS,
        }
    )
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("draft_sample_method", "probabilistic"),
        ("enable_adaptive_verification", True),
        ("dspark_draft_topk", 8),
    ],
)
def test_warns_on_unsupported_drafting_options(field, value):
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 4,
                **_LAYER_IDS,
            },
            **{field: value},
        )
    # Drafting falls back to greedy over the fixed K-token block.
    assert p.block_size == 4
    log.warning.assert_called_once()
    assert "does not support" in log.warning.call_args.args[0]


@pytest.mark.parametrize(
    "sample_from_anchor,trained_block,K,query_block",
    [
        (True, 5, 6, 6),  # DeepSeek-V4-Pro-0813 served at K=6
        (True, 7, 4, 4),
        (False, 7, 4, 5),  # bonus-anchor layout keeps 1 + K
    ],
)
def test_k_need_not_match_checkpoint_block(
    sample_from_anchor, trained_block, K, query_block
):
    # The query block follows K whatever block the checkpoint was trained
    # at; the mismatch is allowed but warned about.
    hf = {
        "dspark_noise_token_id": 7,
        "dspark_block_size": trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(hf, num_speculative_tokens=K)
    assert p.block_size == query_block
    log.warning.assert_called_once()
    assert "trained with block size" in log.warning.call_args.args[0]


@pytest.mark.parametrize(
    "block_key,sample_from_anchor,trained_block,K",
    [
        ("dspark_block_size", True, 4, 4),
        # Speculators-format checkpoints count the bonus anchor in block_size
        # and draft block_size - 1 tokens (e.g. dspark_qwen3_8b_block7, K=6).
        ("block_size", False, 7, 6),
    ],
)
def test_matching_checkpoint_block_does_not_warn(
    block_key, sample_from_anchor, trained_block, K
):
    hf = {
        "dspark_noise_token_id": 7,
        block_key: trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        _make_proposer(hf, num_speculative_tokens=K)
    log.warning.assert_not_called()


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
            "dspark_target_layer_ids": [9, 9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "target_layer_ids": [3, 7],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dflash_config": {"target_layer_ids": [4, 8]},
            "dspark_target_layer_ids": [9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False


# --- Position-major logits, vocab-parallel Markov head ----------------------


class _ToyLogitsProcessor:
    """Per-shard lm_head matmul, like LogitsProcessor with skip_gather=True."""

    def __call__(self, lm_head, hidden_states, embedding_bias=None, skip_gather=False):
        assert embedding_bias is None and skip_gather
        return hidden_states @ lm_head.weight.T


class _ToyDraft:
    """CPU DSpark draft model: tiny vocab, optional vocab-sharded lm_head."""

    def __init__(self, vocab=32, hidden=16, rank=4, tp=1, tp_rank=0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(vocab, hidden, generator=g)
        self.lm_w = torch.randn(vocab, hidden, generator=g)
        self.w1 = torch.randn(vocab, rank, generator=g)
        self.w2 = torch.randn(vocab, rank, generator=g)
        shard = vocab // tp
        self.vocab_start = tp_rank * shard
        self.w2_local = self.w2[self.vocab_start : self.vocab_start + shard]
        self.lm_head = SimpleNamespace(
            weight=self.lm_w[self.vocab_start : self.vocab_start + shard]
        )
        self.logits_processor = _ToyLogitsProcessor()

    def model(self, input_ids, positions):
        return torch.tanh(self.embed[input_ids] + 0.01 * positions[:, None].float())

    def compute_draft_logits(self, hidden_states):
        return hidden_states @ self.lm_w.T

    def markov_embed(self, token_ids):
        return self.w1[token_ids]

    def markov_bias(self, markov_embed):
        return markov_embed @ self.w2.T

    def map_draft_to_target(self, draft_ids):
        return draft_ids


def _bare_proposer(**attrs):
    # Only the attributes the drafting helpers read; no runner or config.
    p = DSparkProposer.__new__(DSparkProposer)
    for name, value in attrs.items():
        setattr(p, name, value)
    return p


def _draft_tokens(draft, *, reqs, block, sample_from_anchor, position_major, tp=1):
    p = _bare_proposer(
        draft_model=draft,
        sample_from_anchor=sample_from_anchor,
        _position_major_logits=position_major,
        _markov_w2_local=draft.w2_local if tp > 1 else None,
        _markov_vocab_start=draft.vocab_start,
        _markov_tp=tp,
    )
    g = torch.Generator().manual_seed(1)
    # A few trailing rows past the last full block, as in a padded batch.
    input_ids = torch.randint(0, draft.embed.shape[0], (reqs * block + 3,), generator=g)
    positions = torch.arange(input_ids.shape[0])
    # Trace the plain Python body; the compiled wrapper needs the TPU backend.
    forward = DSparkProposer._dflash_forward_and_sample._torchdynamo_orig_callable
    tokens, _ = forward(p, input_ids, positions, block)
    return tokens


class _ThreadedAllGather:
    """Fakes tensor_model_parallel_all_gather for `tp` ranks run as threads."""

    def __init__(self, tp):
        self.tp = tp
        self.barrier = threading.Barrier(tp)
        self.inputs = {}
        self.rank = threading.local()

    def __call__(self, tensor, dim=-1):
        self.inputs[self.rank.value] = tensor
        self.barrier.wait()
        out = torch.cat([self.inputs[r] for r in range(self.tp)], dim=dim)
        self.barrier.wait()
        return out

    def run(self, fn):
        results, errors = [None] * self.tp, []

        def work(rank):
            self.rank.value = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:
                errors.append(e)
                self.barrier.abort()

        threads = [threading.Thread(target=work, args=(r,)) for r in range(self.tp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


@pytest.mark.parametrize(
    "sample_from_anchor,block,reqs",
    [(True, 4, 3), (True, 4, 8), (False, 5, 5)],
)
def test_position_major_matches_request_major(sample_from_anchor, block, reqs):
    # Request counts that are and are not a multiple of the 8-row tile.
    kwargs = dict(reqs=reqs, block=block, sample_from_anchor=sample_from_anchor)
    pm = _draft_tokens(_ToyDraft(), position_major=True, **kwargs)
    rm = _draft_tokens(_ToyDraft(), position_major=False, **kwargs)
    assert pm.shape == (reqs, block - (0 if sample_from_anchor else 1))
    assert torch.equal(pm, rm)


@pytest.mark.parametrize("sample_from_anchor,block,reqs", [(True, 4, 3), (False, 5, 9)])
def test_vocab_parallel_forward_matches_replicated(
    monkeypatch, sample_from_anchor, block, reqs
):
    tp = 2
    gather = _ThreadedAllGather(tp)
    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", gather)
    kwargs = dict(reqs=reqs, block=block, sample_from_anchor=sample_from_anchor)
    per_rank = gather.run(
        lambda rank: _draft_tokens(
            _ToyDraft(tp=tp, tp_rank=rank), position_major=True, tp=tp, **kwargs
        )
    )
    replicated = _draft_tokens(_ToyDraft(), position_major=False, **kwargs)
    for tokens in per_rank:
        assert torch.equal(tokens, replicated)


def test_vocab_parallel_argmax_matches_global_argmax_with_ties(monkeypatch):
    tp, rows, shard = 2, 4, 5
    scores = torch.randn(rows, tp * shard, generator=torch.Generator().manual_seed(0))
    scores[1, 3] = scores[1, 7] = 9.0  # tie across ranks: lower id wins
    scores[2, 6] = scores[2, 8] = 9.0  # tie inside one rank
    gather = _ThreadedAllGather(tp)
    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", gather)

    def rank_argmax(rank):
        p = _bare_proposer(_markov_vocab_start=rank * shard, _markov_tp=tp)
        local = scores[:, rank * shard : (rank + 1) * shard]
        return p._vocab_parallel_argmax(local, rows)

    expected = scores.argmax(dim=-1)
    assert expected[1] == 3 and expected[2] == 6
    for ids in gather.run(rank_argmax):
        assert torch.equal(ids, expected)


_VOCAB, _RANK, _TP = 24, 3, 4


def _sharding_draft(
    tp=_TP, rank=2, padding=0, soft_cap=None, scale=1.0, d2t=None, org=_VOCAB
):
    shard = _VOCAB // tp
    w2 = torch.arange(_VOCAB * _RANK, dtype=torch.float32).reshape(_VOCAB, _RANK)
    return SimpleNamespace(
        lm_head=SimpleNamespace(
            tp_size=tp,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=rank * shard,
                org_vocab_end_index=(rank + 1) * shard,
                num_org_vocab_padding=padding,
            ),
        ),
        logits_processor=SimpleNamespace(
            soft_cap=soft_cap, scale=scale, org_vocab_size=org
        ),
        model=SimpleNamespace(
            markov_head=SimpleNamespace(markov_w2=SimpleNamespace(weight=w2))
        ),
        draft_id_to_target_id=d2t,
    )


def test_shard_markov_head_slices_local_vocab_rows():
    draft = _sharding_draft()
    full = draft.model.markov_head.markov_w2.weight.clone()
    p = _bare_proposer(draft_model=draft)
    local = p._shard_markov_head()
    assert torch.equal(local, full[12:18])
    assert p._markov_vocab_start == 12 and p._markov_tp == _TP
    # Only the local shard stays resident on the draft model.
    assert torch.equal(draft.model.markov_head.markov_w2.weight, full[12:18])


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tp=1, rank=0),
        dict(padding=4),
        dict(soft_cap=30.0),
        dict(scale=0.5),
        dict(d2t=torch.arange(_VOCAB)),
        dict(org=_VOCAB - 1),
    ],
    ids=["tp1", "vocab_padding", "soft_cap", "scale", "d2t", "vocab_mismatch"],
)
def test_shard_markov_head_falls_back_to_replicated(overrides):
    draft = _sharding_draft(**overrides)
    full = draft.model.markov_head.markov_w2.weight
    p = _bare_proposer(draft_model=draft)
    assert p._shard_markov_head() is None
    assert draft.model.markov_head.markov_w2.weight is full


def _loadable_draft(**hooks):
    draft = _ToyDraft()
    draft.has_own_lm_head = False
    draft.has_own_embed_tokens = False
    # tp_size 1: _shard_markov_head keeps the replicated head.
    draft.lm_head.tp_size = 1
    for name, value in hooks.items():
        setattr(draft, name, value)
    return draft


@pytest.mark.parametrize("hook,expected", [(None, False), (False, False), (True, True)])
def test_position_major_logits_follow_draft_model_hook(hook, expected):
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._position_major_logits is False
    hooks = {} if hook is None else {"position_major_draft_logits": lambda: hook}
    p.draft_model = _loadable_draft(**hooks)
    with mock.patch.object(DFlashProposer, "load_model"):
        p.load_model(target_model=None)
    assert p._position_major_logits is expected
    assert p._markov_w2_local is None
