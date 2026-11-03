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

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import TuningKey, get_tuned_params
from vllm_torchtpu.layers.core import attention_interface

# Kimi K3 MLA: kv_lora_rank (absorbed q / latent dim) 512, rope 64,
# qk_nope_head_dim 128 (what the model layer passes as `qk_nope_head_dim`).
_LKV_DIM = 512
_R_DIM = 64
_QK_NOPE_HEAD_DIM = 128


def _kimi_k3_decode_key(num_heads: int, page_size_per_kv_packing: int) -> TuningKey:
    return TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=num_heads,
        actual_lkv_dim=_LKV_DIM,
        actual_r_dim=_R_DIM,
        kv_dtype="bfloat16",
        q_dtype="bfloat16",
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=2,
        max_num_seqs=64,
        pages_per_seq=13,
    )


# (local MLA heads, bf16 unified KDA+MLA pool page size per kv packing):
# TP8 -> 12 heads, 648-token pages; TP16 -> 6 heads, 324-token pages.
_KIMI_K3_GEOMETRIES = [
    pytest.param(12, 324, id="tp8_h12"),
    pytest.param(6, 162, id="tp16_h6"),
]


@pytest.mark.parametrize("num_heads,page_size_per_kv_packing", _KIMI_K3_GEOMETRIES)
def test_mla_attention_decode_key_uses_kernel_geometry(
    monkeypatch: pytest.MonkeyPatch, num_heads: int, page_size_per_kv_packing: int
) -> None:
    num_tokens = 64
    max_num_seqs = 64
    pages_per_seq = 13
    kv_packing = 2
    q = jnp.ones((num_tokens, num_heads, _LKV_DIM), dtype=jnp.bfloat16)
    q_rope = jnp.ones((num_tokens, num_heads, _R_DIM), dtype=jnp.bfloat16)
    k = jnp.ones((num_tokens, _LKV_DIM), dtype=jnp.bfloat16)
    k_rope = jnp.ones((num_tokens, _R_DIM), dtype=jnp.bfloat16)
    cache = jnp.zeros(
        (8, page_size_per_kv_packing, kv_packing, _LKV_DIM + 128), dtype=jnp.bfloat16
    )
    md = SimpleNamespace(
        seq_lens=jnp.ones((max_num_seqs,), dtype=jnp.int32),
        block_tables=jnp.zeros((max_num_seqs * pages_per_seq,), dtype=jnp.int32),
        query_start_loc=jnp.arange(max_num_seqs + 1, dtype=jnp.int32),
        request_distribution=jnp.array(
            [max_num_seqs, max_num_seqs, max_num_seqs], dtype=jnp.int32
        ),
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))

    keys: list[TuningKey] = []

    def _recording_get_tuned_params(key: TuningKey):
        keys.append(key)
        return get_tuned_params(key)

    monkeypatch.setattr(
        attention_interface, "get_tuned_params", _recording_get_tuned_params
    )
    mock_logger = MagicMock()
    monkeypatch.setattr(tuned_params, "logger", mock_logger)
    mock_kernel = MagicMock(
        return_value=(
            jnp.zeros((num_heads, num_tokens, _LKV_DIM), dtype=jnp.bfloat16),
            cache,
        )
    )
    monkeypatch.setattr(attention_interface, "mla_ragged_paged_attention", mock_kernel)
    monkeypatch.setattr(envs, "MIXED_Q_SPLIT", None)
    monkeypatch.setattr(envs, "MIXED_NUM_QUERIES_PER_BLOCK", None)
    monkeypatch.setattr(envs, "MIXED_NUM_KV_PAGES_PER_BLOCK", None)

    attention_interface.mla_attention(
        q_TNA=q,
        q_rope_TNH=q_rope,
        k_SA=k,
        k_rope_SH=k_rope,
        kv_cache=cache,
        md=md,
        mesh=mesh,
        num_attention_heads=num_heads,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        sm_scale=0.1,
    )

    decode_keys = [key for key in keys if key.case == "batched_decode"]
    assert decode_keys == [_kimi_k3_decode_key(num_heads, page_size_per_kv_packing)]
    missed = [
        call.args[-1]
        for call in mock_logger.warning.call_args_list
        if getattr(call.args[-1], "case", None) == "batched_decode"
    ]
    assert missed == []

    # The decode stage keeps the parameters it ran with before the key fix.
    _, kwargs = mock_kernel.call_args
    assert kwargs["num_kv_pages_per_block"][0] == 3
    assert kwargs["num_queries_per_block"][0] == 1
    assert kwargs["decode_batch_size"] == 4


@pytest.mark.parametrize("num_heads,page_size_per_kv_packing", _KIMI_K3_GEOMETRIES)
@pytest.mark.parametrize("max_num_tokens", [4, 64, 256, 1024])
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "float8_e4m3fn"])
def test_kimi_k3_batched_decode_tuned_params(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    page_size_per_kv_packing: int,
    max_num_tokens: int,
    kv_dtype: str,
) -> None:
    mock_logger = MagicMock()
    monkeypatch.setattr(tuned_params, "logger", mock_logger)
    key = TuningKey(
        case="batched_decode",
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=num_heads,
        actual_lkv_dim=_LKV_DIM,
        actual_r_dim=_R_DIM,
        kv_dtype=kv_dtype,
        q_dtype="bfloat16",
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=2 if kv_dtype == "bfloat16" else 4,
        max_num_seqs=128,
        pages_per_seq=7,
    )

    tuned = get_tuned_params(key)

    mock_logger.warning.assert_not_called()
    assert tuned.num_kv_pages_per_block == 3
    assert tuned.num_queries_per_block == 1
    assert tuned.decode_batch_size == 4
    assert tuned.vmem_limit_bytes == 62914560


def test_non_kimi_batched_decode_still_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_logger = MagicMock()
    monkeypatch.setattr(tuned_params, "logger", mock_logger)
    key = TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=_LKV_DIM,
        actual_r_dim=_R_DIM,
        q_dtype="bfloat16",
        page_size_per_kv_packing=256,
        kv_packing=4,
        max_num_seqs=2,
        pages_per_seq=9,
    )

    tuned = get_tuned_params(key)

    mock_logger.warning.assert_called_once()
    assert tuned.num_kv_pages_per_block == 3
    assert tuned.decode_batch_size == 4
