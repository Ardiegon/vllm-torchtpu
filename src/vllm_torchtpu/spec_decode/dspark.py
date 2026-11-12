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

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import vllm.distributed
from jax.experimental.pallas import tpu as pltpu
from vllm.config import VllmConfig
from vllm.utils.math_utils import round_up

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.tpu_info import get_chip_version

logger = init_logger(__name__)


@runtime_checkable
class VocabParallelMarkovDraft(Protocol):
    """DSpark draft model that drafts position-major from its vocab shard.

    A draft model opts in by implementing these methods; the proposer then
    computes the draft logits position-major (see `_position_major_hidden`)
    and, when the model could shard its Markov head, samples from this
    rank's vocab slice. Draft models that do not implement them keep the
    request-major path through `compute_draft_logits` and `markov_bias`.
    """

    def shard_markov_head(self) -> int | None:
        """Keep only this rank's vocab rows of the Markov head.

        Called once, after the lm_head is shared with the target. The
        proposer reduces (value, token id) pairs across ranks in float32, so
        a model shards only a vocab whose ids are exact in float32.

        Returns:
            The first vocab id of this rank's shard, or None when the head
            stays replicated.
        """
        ...

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, vocab / tp]`` slice of the draft logits."""
        ...

    def markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """The Markov bias over this rank's vocab rows."""
        ...


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
        # Set by `load_model` for a `VocabParallelMarkovDraft` draft model.
        self._position_major_logits = False
        # Rows of one aligned tile of position-major requests. Resolved here
        # because the drafting forward is traced.
        self._request_tile_rows = pltpu.get_tpu_info_for_chip(
            get_chip_version(), 1
        ).num_sublanes
        # First vocab id of this rank's shard when the draft model sharded its
        # Markov head (`VocabParallelMarkovDraft.shard_markov_head`).
        self._markov_vocab_start: int | None = None
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
        draft = self.draft_model
        if isinstance(draft, VocabParallelMarkovDraft):
            self._position_major_logits = True
            self._markov_vocab_start = draft.shard_markov_head()
        logger.info(
            "DSpark: position_major_logits=%s, vocab-parallel Markov sampling=%s.",
            self._position_major_logits,
            self._markov_vocab_start is not None,
        )

    def _vocab_parallel_argmax(
        self, local_scores: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """Global argmax of vocab-sharded ``[rows, V/tp]`` scores (first max
        wins on ties, like ``torch.argmax`` over the gathered vocab)."""
        vals, idx = local_scores.max(dim=-1)
        vocab_start = self._markov_vocab_start
        assert vocab_start is not None
        # One float32 all-gather carries (value, token id) pairs; the draft
        # model only shards a vocab whose ids are exact in float32.
        pair = torch.stack([vals.float(), (idx + vocab_start).float()], dim=-1)
        tp_size = vllm.distributed.get_tensor_model_parallel_world_size()
        gathered = vllm.distributed.tensor_model_parallel_all_gather(pair, dim=-1).view(
            rows, tp_size, 2
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
            flat_pm, prev, reqs_tile = self._position_major_hidden(
                valid_hidden, ids_3d[:, 0], block_size, self._request_tile_rows
            )
            if self._markov_vocab_start is not None:
                draft = self.draft_model
                assert isinstance(draft, VocabParallelMarkovDraft)
                local_logits = draft.compute_draft_logits_local(flat_pm)
                local_logits = local_logits.view(
                    block_size, reqs_tile, local_logits.shape[-1]
                )
                draft_tokens_chunk = self._greedy_markov_chain(
                    lambda i: local_logits[i],
                    prev,
                    sample_start,
                    block_size,
                    bias_fn=draft.markov_bias_local,
                    select_fn=lambda s: self._vocab_parallel_argmax(s, reqs_tile),
                )
                return draft_tokens_chunk[:padded_num_reqs], hidden
            logits_pm = self.draft_model.compute_draft_logits(flat_pm)
            logits_pm = logits_pm.view(block_size, reqs_tile, logits_pm.shape[-1])
            draft_tokens_chunk = self._greedy_markov_chain(
                lambda i: logits_pm[i],
                prev,
                sample_start,
                block_size,
                bias_fn=self.draft_model.markov_bias,
                select_fn=self._greedy_draft_to_target,
            )
            return draft_tokens_chunk[:padded_num_reqs], hidden

        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after argmax.
        logits = self.draft_model.compute_draft_logits(valid_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])
        draft_tokens_chunk = self._greedy_markov_chain(
            lambda i: logits_3d[:, i],
            ids_3d[:, 0],
            sample_start,
            block_size,
            bias_fn=self.draft_model.markov_bias,
            select_fn=self._greedy_draft_to_target,
        )
        return draft_tokens_chunk, hidden

    @staticmethod
    def _position_major_hidden(
        valid_hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        block_size: int,
        tile_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Reorder ``[reqs * block, H]`` hidden rows position-major.

        Viewing the ``[rows, vocab]`` logits as ``[reqs, block, vocab]``
        relayouts the whole full-vocab tensor (a ~1.3 ms
        while/dynamic-update-slice loop per draft on v6e when ``reqs`` is
        not a multiple of the ``tile_rows``-row sublane tile). Reordering the
        small hidden rows instead, with requests padded to the tile, makes
        each position's logits one aligned ``[reqs_tile, vocab]`` slab.
        Returns the flat ``[block * reqs_tile, H]`` hidden rows, the anchor
        ids padded to ``reqs_tile``, and ``reqs_tile``.
        """
        padded_num_reqs = valid_hidden.shape[0] // block_size
        hidden_size = valid_hidden.shape[-1]
        reqs_tile = round_up(padded_num_reqs, tile_rows)
        hidden_pm = valid_hidden.view(
            padded_num_reqs, block_size, hidden_size
        ).transpose(0, 1)
        if reqs_tile != padded_num_reqs:
            hidden_pm = torch.nn.functional.pad(
                hidden_pm, (0, 0, 0, reqs_tile - padded_num_reqs)
            )
        flat_pm = hidden_pm.reshape(block_size * reqs_tile, hidden_size)
        prev = anchor_ids
        if reqs_tile != padded_num_reqs:
            prev = torch.nn.functional.pad(prev, (0, reqs_tile - padded_num_reqs))
        return flat_pm, prev, reqs_tile

    def _greedy_draft_to_target(self, scores: torch.Tensor) -> torch.Tensor:
        """Greedy pick over replicated draft-vocab scores, as target ids."""
        assert self.draft_model is not None
        return self.draft_model.map_draft_to_target(scores.argmax(dim=-1))

    def _greedy_markov_chain(
        self,
        position_logits: Callable[[int], torch.Tensor],
        prev: torch.Tensor,
        sample_start: int,
        block_size: int,
        *,
        bias_fn: Callable[[torch.Tensor], torch.Tensor],
        select_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Sequential greedy Markov sampling of positions
        ``sample_start..block_size-1``.

        Each step adds the Markov bias of the previously sampled token to
        that position's logits and picks the next token with ``select_fn``.
        The loop is static, so it unrolls when traced. Returns
        ``[rows, block_size - sample_start]`` token ids.
        """
        assert self.draft_model is not None
        draft_tokens = []
        for i in range(sample_start, block_size):
            bias = bias_fn(self.draft_model.markov_embed(prev))
            step = select_fn(position_logits(i) + bias)
            draft_tokens.append(step)
            prev = step
        return torch.stack(draft_tokens, dim=1)
