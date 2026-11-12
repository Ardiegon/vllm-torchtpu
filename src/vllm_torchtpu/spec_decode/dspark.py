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
"""DSpark speculative decoding for TPU.

DSpark is semi-autoregressive parallel drafting: one DFlash-style parallel
block forward (anchor + noise tokens, non-causal within the block, context
KV precomputed from the target's aux hidden states), followed by a
lightweight sequential Markov head that biases each position's draft logits
on the previously sampled token. Upstream implements DSparkSpeculator as a
small DFlashSpeculator subclass; this proposer mirrors that on top of the
TPU DFlashProposer.

Dense-checkpoint layout (``sample_from_anchor`` absent or true, as in
upstream DSparkSpeculator): the query block is K tokens — the anchor is the
FIRST prediction position and every slot predicts the NEXT token — versus
DFlash's 1 + K (anchor is bonus only). Speculators-format checkpoints set
``sample_from_anchor=False`` and keep the DFlash layout.

PoC scope: greedy draft sampling only (argmax + d2t remap); probabilistic
Gumbel-coupled sampling is not implemented.
"""

from __future__ import annotations

import torch
import vllm.distributed
from vllm.config import VllmConfig
from vllm.utils.math_utils import round_up

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)

# Rows per aligned TPU tile (the vreg sublane count).
_SUBLANE_ROWS = 8
# Largest integer up to which every integer is exact in float32.
_MAX_EXACT_FLOAT32_INT = 2**24


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        hf_config = speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        if not self.dflash_config.get("target_layer_ids"):
            raise ValueError(
                "DSpark requires target layer ids in the draft config (one of "
                "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
                "target_layer_ids, or dflash_config.target_layer_ids)."
            )

        # Same key/default as upstream DSparkSpeculator: dense checkpoints
        # omit it (-> True); the speculators converter writes it explicitly.
        self.sample_from_anchor = bool(hf_dict.get("sample_from_anchor", True))
        K = self.speculative_config.num_speculative_tokens
        # vLLM 0.29 drafting options this proposer does not implement yet. They
        # are ignored, not rejected: drafting stays greedy over a fixed block.
        spec = self.speculative_config
        if (
            spec.draft_sample_method != "greedy"
            or spec.enable_adaptive_verification
            or spec.dspark_draft_topk is not None
        ):
            logger.warning(
                "DSpark on TPU does not support draft_sample_method=%r, "
                "enable_adaptive_verification=%s, dspark_draft_topk=%s; "
                "ignoring them and using greedy drafting of a fixed block.",
                spec.draft_sample_method,
                spec.enable_adaptive_verification,
                spec.dspark_draft_topk,
            )
        # The query block follows K, not the block length the checkpoint was
        # trained at; upstream vLLM likewise only enforces K == block for
        # Qwen3-Omni DSpark. The checkpoint's block is a query-block length in
        # both layouts (K for dense, 1 + K for speculators-format), so compare
        # it to block_size rather than K. In-block attention is non-causal, so
        # a mismatch shifts every slot's hidden state, not just the slots past
        # the trained length: outputs stay exact (the target verifies), but
        # acceptance can drop at every position.
        trained_block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        if trained_block is not None and int(trained_block) != self.block_size:
            logger.warning(
                "DSpark: num_speculative_tokens=%d gives a %d-token query "
                "block, but the checkpoint was trained with block size %s; "
                "every draft slot is out of the training distribution, so "
                "check the acceptance rate.",
                K,
                self.block_size,
                trained_block,
            )
        # Position-major draft logits (see `_dflash_forward_and_sample`) are
        # opt-in per draft model; `load_model` reads the draft's choice.
        self._position_major_logits = False
        # Vocab-parallel Markov head state, set by `_shard_markov_head`.
        self._markov_w2_local: torch.Tensor | None = None
        self._markov_vocab_start = 0
        self._markov_tp = 1
        logger.info(
            "DSpark: sample_from_anchor=%s, K=%d.",
            self.sample_from_anchor,
            K,
        )

    # A reduced-vocab DSpark checkpoint carries its own lm_head
    # ([draft_vocab, hidden] plus a d2t remap) and possibly its own
    # embed_tokens; force-sharing the target's would clobber them. Honor the
    # has_own_embed_tokens / has_own_lm_head flags the loader sets (upstream
    # load_dspark_model does the same via _should_share).
    _force_share_target_embeddings = False

    def _query_block_size(self, K: int) -> int:
        # Dense layout: K query tokens (anchor + K-1 noise), every slot
        # predicts the next token. Speculators-format checkpoints keep the
        # DFlash 1 + K layout.
        return K if self.sample_from_anchor else K + 1

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        for attr in (
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
            "has_own_lm_head",
            "has_own_embed_tokens",
        ):
            if not hasattr(self.draft_model, attr):
                raise RuntimeError(
                    f"DSpark draft model lacks {attr}(); expected a "
                    "Qwen3DSpark/Gemma4DSpark-family model."
                )
        # Optional draft-model hook: a draft model opts in to position-major
        # draft logits by defining `position_major_draft_logits()`. Draft
        # models without it keep the request-major path.
        if hasattr(self.draft_model, "position_major_draft_logits"):
            self._position_major_logits = bool(
                self.draft_model.position_major_draft_logits()
            )
        if self._position_major_logits:
            self._markov_w2_local = self._shard_markov_head()
        logger.info(
            "DSpark: position_major_logits=%s, vocab-parallel Markov sampling=%s.",
            self._position_major_logits,
            self._markov_w2_local is not None,
        )

    def _shard_markov_head(self) -> torch.Tensor | None:
        """Keep only this rank's vocab rows of the Markov head.

        The draft logits come from the target's vocab-parallel lm_head; with
        the Markov bias sliced to the same vocab shard, every draft position
        takes a local max plus a tiny cross-rank (value, index) reduction
        instead of all-gathering the full-vocab logits and reading the whole
        replicated ``markov_w2`` (V x r) once per position. Each logit is the
        same full-hidden dot product either way. Returns None (replicated
        path) when the layout does not allow it.
        """
        reason = self._markov_head_replication_reason()
        if reason is not None:
            logger.info("DSpark: keeping the Markov head replicated: %s.", reason)
            return None
        draft = self.draft_model
        assert draft is not None
        head = draft.model.markov_head
        shard = draft.lm_head.shard_indices
        start = int(shard.org_vocab_start_index)
        end = int(shard.org_vocab_end_index)
        local = head.markov_w2.weight[start:end].contiguous().clone()
        # Drop the replicated copy: only the shard is read from now on. The
        # draft's own `markov_bias` would now see a partial vocab, so the
        # vocab-parallel drafting path never calls it.
        head.markov_w2.weight = torch.nn.Parameter(local, requires_grad=False)
        self._markov_vocab_start = start
        self._markov_tp = draft.lm_head.tp_size
        return local

    def _markov_head_replication_reason(self) -> str | None:
        """Why the Markov head cannot be vocab-sharded, or None if it can."""
        draft = self.draft_model
        assert draft is not None
        lm_head = draft.lm_head
        if lm_head is None or lm_head.tp_size <= 1:
            return "the lm_head is not vocab-parallel"
        if lm_head.shard_indices.num_org_vocab_padding != 0:
            return "the lm_head vocab shards are padded"
        lp = draft.logits_processor
        if lp.soft_cap is not None or lp.scale != 1.0:
            return "the logits are soft-capped or scaled"
        if draft.draft_id_to_target_id is not None:
            return "draft token ids are remapped to the target vocab"
        w2 = draft.model.markov_head.markov_w2.weight
        if not isinstance(w2, torch.Tensor) or w2.dim() != 2:
            return "the Markov head weight is not a 2-D tensor"
        if w2.shape[0] != lp.org_vocab_size:
            return "the Markov head rows do not match the lm_head vocab"
        if lp.org_vocab_size > _MAX_EXACT_FLOAT32_INT:
            return "token ids are not exact in float32 for this vocab size"
        return None

    def _vocab_parallel_argmax(
        self, local_scores: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """Global argmax of vocab-sharded ``[rows, V/tp]`` scores (first max
        wins on ties, like ``torch.argmax`` over the gathered vocab)."""
        vals, idx = local_scores.max(dim=-1)
        # One float32 all-gather carries (value, token id) pairs; the ids are
        # exact because `_shard_markov_head` caps the vocab at 2**24.
        pair = torch.stack(
            [vals.float(), (idx + self._markov_vocab_start).float()], dim=-1
        )
        gathered = vllm.distributed.tensor_model_parallel_all_gather(pair, dim=-1).view(
            rows, self._markov_tp, 2
        )
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=best).squeeze(-1).long()

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential greedy Markov sampling, in-graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + argmax.
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]
        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        sample_start = 0 if self.sample_from_anchor else 1

        if self._position_major_logits:
            # Phase O6b: viewing the [rows, vocab] logits as
            # [reqs, block, vocab] relayouts the whole full-vocab tensor
            # (a ~1.3 ms while/dynamic-update-slice loop per draft on v6e
            # when `reqs` is not a multiple of the 8-row tile). Reorder the
            # (small) hidden rows position-major instead, padding requests
            # to the tile, so each position's logits are one aligned slab.
            reqs_tile = round_up(padded_num_reqs, _SUBLANE_ROWS)
            hidden_pm = valid_hidden.view(
                padded_num_reqs, block_size, valid_hidden.shape[-1]
            ).transpose(0, 1)
            if reqs_tile != padded_num_reqs:
                hidden_pm = torch.nn.functional.pad(
                    hidden_pm, (0, 0, 0, reqs_tile - padded_num_reqs)
                )
            flat_pm = hidden_pm.reshape(block_size * reqs_tile, valid_hidden.shape[-1])
            prev = ids_3d[:, 0]
            if reqs_tile != padded_num_reqs:
                prev = torch.nn.functional.pad(prev, (0, reqs_tile - padded_num_reqs))
            draft_tokens = []
            w2_local = self._markov_w2_local
            if w2_local is not None:
                # This rank's vocab slice of the logits (no TP gather), like
                # upstream LlamaForCausalLM.compute_logits_local.
                local_logits = self.draft_model.logits_processor(
                    self.draft_model.lm_head, flat_pm, skip_gather=True
                )
                local_logits = local_logits.view(
                    block_size, reqs_tile, local_logits.shape[-1]
                )
                for i in range(sample_start, block_size):
                    bias = torch.nn.functional.linear(
                        self.draft_model.markov_embed(prev), w2_local
                    )
                    step = self._vocab_parallel_argmax(
                        local_logits[i] + bias, reqs_tile
                    )
                    draft_tokens.append(step)
                    prev = step
                draft_tokens_chunk = torch.stack(draft_tokens, dim=1)
                return draft_tokens_chunk[:padded_num_reqs], hidden
            logits_pm = self.draft_model.compute_draft_logits(flat_pm)
            logits_pm = logits_pm.view(block_size, reqs_tile, logits_pm.shape[-1])
            for i in range(sample_start, block_size):
                bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
                step = self.draft_model.map_draft_to_target(
                    (logits_pm[i] + bias).argmax(dim=-1)
                )
                draft_tokens.append(step)
                prev = step
            draft_tokens_chunk = torch.stack(draft_tokens, dim=1)[:padded_num_reqs]
            return draft_tokens_chunk, hidden

        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after argmax.
        logits = self.draft_model.compute_draft_logits(valid_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])

        prev = ids_3d[:, 0]
        draft_tokens = []
        for i in range(sample_start, block_size):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            step = self.draft_model.map_draft_to_target(
                (logits_3d[:, i] + bias).argmax(dim=-1)
            )
            draft_tokens.append(step)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden
