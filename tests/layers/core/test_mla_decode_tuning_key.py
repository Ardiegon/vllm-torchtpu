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
"""The MLA `batched_decode` tuning key must describe the kernel's real inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import TuningKey, get_tuned_params
from vllm_torchtpu.layers.core import attention_interface
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

# MLA dims of a DeepSeek-style model: kv_lora_rank (absorbed q / latent dim)
# 512, rope 64, qk_nope_head_dim 128 (what the model layer passes as
# `qk_nope_head_dim`).
_LKV_DIM = 512
_R_DIM = 64
_QK_NOPE_HEAD_DIM = 128
_NUM_TOKENS = 64
_MAX_NUM_SEQS = 64
_PAGES_PER_SEQ = 13
_KV_PACKING = 2
# The mesh axis attention_interface shards the heads over.
_MODEL_AXIS = "model"


def _expected_decode_key(num_heads: int, page_size_per_kv_packing: int) -> TuningKey:
    return TuningKey(
        case="batched_decode",
        max_num_tokens=_NUM_TOKENS,
        actual_num_q_heads=num_heads,
        actual_lkv_dim=_LKV_DIM,
        actual_r_dim=_R_DIM,
        kv_dtype="bfloat16",
        q_dtype="bfloat16",
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=_KV_PACKING,
        max_num_seqs=_MAX_NUM_SEQS,
        pages_per_seq=_PAGES_PER_SEQ,
    )


# (local MLA heads, bf16 page size per kv packing): 12 heads with 324-token
# pages per packing (648 total), 6 heads with 162 (324 total).
_LOCAL_HEAD_PAGE_GEOMETRIES = [
    pytest.param(12, 324, id="h12_page324"),
    pytest.param(6, 162, id="h6_page162"),
]


def _decode_metadata() -> AttentionMetadata:
    return AttentionMetadata(
        input_positions=None,
        seq_lens=jnp.ones((_MAX_NUM_SEQS,), dtype=jnp.int32),
        block_tables=jnp.zeros((_MAX_NUM_SEQS * _PAGES_PER_SEQ,), dtype=jnp.int32),
        query_start_loc=jnp.arange(_MAX_NUM_SEQS + 1, dtype=jnp.int32),
        request_distribution=jnp.full((3,), _MAX_NUM_SEQS, dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    "num_heads,page_size_per_kv_packing", _LOCAL_HEAD_PAGE_GEOMETRIES
)
def test_mla_attention_decode_key_uses_kernel_geometry(
    monkeypatch: pytest.MonkeyPatch, num_heads: int, page_size_per_kv_packing: int
) -> None:
    cache = jnp.zeros(
        (8, page_size_per_kv_packing, _KV_PACKING, _LKV_DIM + 128), dtype=jnp.bfloat16
    )
    keys: list[TuningKey] = []

    def _recording_get_tuned_params(key: TuningKey) -> tuned_params.TunableParams:
        keys.append(key)
        return get_tuned_params(key)

    monkeypatch.setattr(
        attention_interface, "get_tuned_params", _recording_get_tuned_params
    )
    kernel_calls: list[dict] = []

    def _fake_kernel(*args, **kwargs):
        kernel_calls.append(kwargs)
        return (
            jnp.zeros((num_heads, _NUM_TOKENS, _LKV_DIM), dtype=jnp.bfloat16),
            cache,
        )

    monkeypatch.setattr(attention_interface, "mla_ragged_paged_attention", _fake_kernel)
    monkeypatch.setattr(envs, "MIXED_Q_SPLIT", None)
    monkeypatch.setattr(envs, "MIXED_NUM_QUERIES_PER_BLOCK", None)
    monkeypatch.setattr(envs, "MIXED_NUM_KV_PAGES_PER_BLOCK", None)

    attention_interface.mla_attention(
        q_TNA=jnp.ones((_NUM_TOKENS, num_heads, _LKV_DIM), dtype=jnp.bfloat16),
        q_rope_TNH=jnp.ones((_NUM_TOKENS, num_heads, _R_DIM), dtype=jnp.bfloat16),
        k_SA=jnp.ones((_NUM_TOKENS, _LKV_DIM), dtype=jnp.bfloat16),
        k_rope_SH=jnp.ones((_NUM_TOKENS, _R_DIM), dtype=jnp.bfloat16),
        kv_cache=cache,
        md=_decode_metadata(),
        mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), (_MODEL_AXIS,)),
        num_attention_heads=num_heads,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        sm_scale=0.1,
    )

    decode_keys = [key for key in keys if key.case == "batched_decode"]
    assert decode_keys == [_expected_decode_key(num_heads, page_size_per_kv_packing)]
    # The decode stage keeps the parameters it ran with before the key fix.
    kwargs = kernel_calls[-1]
    assert kwargs["num_kv_pages_per_block"][0] == 3
    assert kwargs["num_queries_per_block"][0] == 1
    assert kwargs["decode_batch_size"] == 4
