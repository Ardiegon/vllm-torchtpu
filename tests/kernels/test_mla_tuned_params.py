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

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import TuningKey, get_tuned_params


def _k3_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=8,
        pages_per_seq=128,
    )


def _k3_tp32_fp8_8k_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=520,
    )


def _k3_tp32_fp8_8k_page256_mixed_key() -> TuningKey:
    return replace(
        _k3_tp32_fp8_8k_mixed_key(),
        page_size_per_kv_packing=64,
        pages_per_seq=40,
    )


def test_k3_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_mixed_key())

    assert tuned.num_kv_pages_per_block == 4
    assert tuned.num_queries_per_block == 256
    assert tuned.q_split == 1


def test_k3_tp32_fp8_8k_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_mixed_key())

    assert tuned.num_kv_pages_per_block == 256
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 4
    assert tuned.vmem_limit_bytes == 56 * 1024 * 1024


def test_k3_tp32_fp8_8k_page256_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_page256_mixed_key())

    assert tuned.num_kv_pages_per_block == 16
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 8
    assert tuned.vmem_limit_bytes == 52 * 1024 * 1024


def test_k3_mixed_tuned_params_require_matching_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mismatched_key = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    tuned = get_tuned_params(mismatched_key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1


def test_untuned_batched_decode_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_logger = MagicMock()
    monkeypatch.setattr(tuned_params, "logger", mock_logger)
    key = TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        page_size_per_kv_packing=256,
        kv_packing=4,
        max_num_seqs=2,
        pages_per_seq=9,
    )

    tuned = get_tuned_params(key)

    mock_logger.warning_once.assert_called_once()
    assert mock_logger.warning_once.call_args.args[-1] == key
    assert tuned.num_kv_pages_per_block == 3
    assert tuned.num_queries_per_block == 1
    assert tuned.decode_batch_size == 4
    assert tuned.vmem_limit_bytes == 60 * 1024 * 1024


def _large_page_mixed_key(
    heads: int, page_size_per_kv_packing: int, max_num_tokens: int
) -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=heads,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=2,
        max_num_seqs=32,
        pages_per_seq=26,
    )


@pytest.mark.parametrize(
    ("heads", "page_size_per_kv_packing", "max_num_tokens", "bkvp", "bq"),
    [
        # One 1294-token page per block; 16 heads hit the score-tile cap.
        (24, 647, 4096, 1, 128),
        (16, 647, 4096, 1, 128),
        (16, 647, 64, 1, 64),
        # 648-token pages.
        (12, 324, 2048, 1, 256),
        (8, 324, 8192, 1, 512),
        # 324-token pages are grouped; 8 per block hits the score-tile cap.
        (6, 162, 1024, 4, 512),
        (6, 162, 8192, 8, 256),
        (6, 81, 512, 4, 512),
    ],
)
def test_untuned_large_page_mixed_uses_cost_model(
    heads: int, page_size_per_kv_packing: int, max_num_tokens: int, bkvp: int, bq: int
) -> None:
    key = _large_page_mixed_key(heads, page_size_per_kv_packing, max_num_tokens)
    assert key not in tuned_params.tuned_params_mapping

    tuned = get_tuned_params(key)

    assert tuned.num_kv_pages_per_block == bkvp
    assert tuned.num_queries_per_block == bq
    assert tuned.q_split == 2
    assert tuned.decode_batch_size == 1


def test_large_page_mixed_below_min_tokens_keeps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())

    tuned = get_tuned_params(_large_page_mixed_key(12, 324, 32))

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1
