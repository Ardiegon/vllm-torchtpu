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

"""CPU test for the request clamp of `kimi_pooled_kda_commit`.

The commit clamps the active requests to the window rows the verify input
caches hold (`cache_qkv.shape[0] // (1 + K)` requests), not to the token rows,
so runner metadata padded to `MIN_NUM_SEQS` never indexes past the caches.
"""

from unittest.mock import patch

import jax.numpy as jnp

from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op


def test_pooled_kda_commit_clamps_active_requests_to_window_rows():
    captured: dict[str, object] = {}

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
        captured["num_state_indices"] = int(state_indices.shape[0])
        captured["num_query_start_loc"] = int(query_start_loc.shape[0])
        captured["num_accepted_lens"] = int(kwargs["window_lens"].shape[0])
        captured["window_distribution"] = [int(x) for x in distribution]
        return pool, mixed_qkv

    # max_num_seqs=4 and num_spec_tokens=7 give caches of 32 token rows
    # (4 requests), while the runner pads its metadata to MIN_NUM_SEQS=8.
    pool = jnp.zeros((10, 162, 2, 640), dtype=jnp.bfloat16)
    cache_qkv = jnp.zeros((32, 3 * 4 * 128), dtype=jnp.bfloat16)
    cache_gate = jnp.zeros((32, 4 * 128), dtype=jnp.bfloat16)
    cache_beta = jnp.zeros((32, 4), dtype=jnp.bfloat16)
    conv_weight = jnp.zeros((4, 3, 4, 128), dtype=jnp.bfloat16)
    a_log = jnp.zeros((4,), dtype=jnp.float32)
    dt_bias = jnp.zeros((4 * 128,), dtype=jnp.float32)
    query_start_loc = jnp.arange(9, dtype=jnp.int32) * 8
    state_indices = jnp.arange(1, 9, dtype=jnp.int32)
    seq_lens = jnp.full((8,), 16, dtype=jnp.int32)
    window_distribution = jnp.array([0, 5, 5], dtype=jnp.int32)
    accepted_lens = jnp.ones((8,), dtype=jnp.int32)

    with patch.object(
        kda_attention_op, "_run_kda_window_kernel", side_effect=fake_run_kernel
    ):
        kda_attention_op.kimi_pooled_kda_commit(
            pool,
            cache_qkv,
            cache_gate,
            cache_beta,
            conv_weight,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            seq_lens,
            window_distribution,
            accepted_lens,
            lower_bound=-5.0,
            pool_block_tokens=324,
            num_spec_tokens=7,
        )

    assert captured == {
        "num_state_indices": 4,
        "num_query_start_loc": 5,
        "num_accepted_lens": 4,
        "window_distribution": [0, 4, 4],
    }
