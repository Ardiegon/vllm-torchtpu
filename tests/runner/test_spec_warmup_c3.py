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

"""CPU regression tests for C3-L93 speculative decoding startup & KDA commit bounds:
1. `_spec_warmup_num_output_tokens` preserves in-flight `1 + K` async placeholders
   on synthetic warmup requests (matching `vllm/v1/core/sched/scheduler.py`).
2. `TPUModelRunner._spec_warmup_bucket_grid` and `_warmup_spec_batch` bound active
   synthetic request counts by `scheduler_config.max_num_seqs` when
   `max_num_seqs < MIN_NUM_SEQS` (e.g. `max_num_seqs = 4, max_num_reqs = 8`).
3. `kimi_pooled_kda_commit` clamps `active_reqs` by `cache_qkv.shape[0] // (1 + K)`
   (requests) rather than `cache_qkv.shape[0]` (token rows).
"""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp

from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kop
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _spec_warmup_num_output_tokens,
)


def test_spec_warmup_num_output_tokens_includes_async_placeholders():
    req_state = SimpleNamespace(output_token_ids=[42] + [0] * 8)  # 1 + (1 + K=7) = 9
    async_runner = SimpleNamespace(
        use_async_scheduling=True,
        requests={"__spec_warmup_0__": req_state},
    )
    assert _spec_warmup_num_output_tokens(
        async_runner, ["__spec_warmup_0__"], [2]
    ) == [9]

    sync_runner = SimpleNamespace(
        use_async_scheduling=False,
        requests={"__spec_warmup_0__": req_state},
    )
    assert _spec_warmup_num_output_tokens(
        sync_runner, ["__spec_warmup_0__"], [2]
    ) == [2]


def test_spec_warmup_bucket_grid_respects_max_num_seqs_below_min_num_seqs():
    runner = object.__new__(TPUModelRunner)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=7)
    runner.drafter = SimpleNamespace(block_size=7)
    runner.max_num_reqs = 8  # MIN_NUM_SEQS = 8
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4)
    runner.max_num_tokens = 1024
    runner.num_tokens_paddings = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    grid = runner._spec_warmup_bucket_grid(mixed=True)
    assert grid == [1, 2, 3, 4]
    assert max(grid) <= runner.scheduler_config.max_num_seqs

    # Guard in _warmup_spec_batch must no-op immediately if R > max_num_seqs:
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[], num_blocks=582)
    assert runner._warmup_spec_batch(5, 4, greedy=True, short=True, mixed=True) is None


def test_kimi_pooled_kda_commit_clamps_active_reqs_by_window_rows():
    captured: dict = {}

    def fake_run_kernel(
        mixed_qkv,
        raw_gate,
        beta,
        pool,
        conv_weight,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        seq_lens,
        distribution,
        **kwargs,
    ):
        captured["n_state_indices"] = int(state_indices.shape[0])
        captured["n_query_start_loc"] = int(query_start_loc.shape[0])
        captured["n_accepted_lens"] = int(kwargs["window_lens"].shape[0])
        captured["window_distribution"] = [int(x) for x in distribution]
        return pool, mixed_qkv

    # max_num_seqs=4, num_spec_tokens=7 -> cache_qkv has 32 token rows (= 4 requests),
    # while runner metadata arrays are padded to MIN_NUM_SEQS=8.
    pool = jnp.zeros((10, 162, 2, 640), dtype=jnp.bfloat16)
    cache_qkv = jnp.zeros((32, 3 * 4 * 128), dtype=jnp.bfloat16)
    cache_gate = jnp.zeros((32, 4 * 128), dtype=jnp.bfloat16)
    cache_beta = jnp.zeros((32, 4), dtype=jnp.bfloat16)
    conv_w = jnp.zeros((4, 3, 4, 128), dtype=jnp.bfloat16)
    a_log = jnp.zeros((4,), dtype=jnp.float32)
    dt_bias = jnp.zeros((4 * 128,), dtype=jnp.float32)
    qsl = jnp.arange(9, dtype=jnp.int32) * 8
    si = jnp.arange(1, 9, dtype=jnp.int32)
    sl = jnp.full((8,), 16, dtype=jnp.int32)
    wd = jnp.array([0, 5, 5], dtype=jnp.int32)
    al = jnp.ones((8,), dtype=jnp.int32)

    with patch.object(kop, "_o6a_run_kernel", side_effect=fake_run_kernel):
        kop.kimi_pooled_kda_commit(
            pool,
            cache_qkv,
            cache_gate,
            cache_beta,
            conv_w,
            a_log,
            dt_bias,
            qsl,
            si,
            sl,
            wd,
            al,
            lower_bound=-5.0,
            pool_block_tokens=324,
            num_spec_tokens=7,
        )

    assert captured == {
        "n_state_indices": 4,
        "n_query_start_loc": 5,
        "n_accepted_lens": 4,
        "window_distribution": [0, 4, 4],
    }
