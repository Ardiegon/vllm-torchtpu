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
from vllm_torchtpu.kernels.mla.v2.tuned_params import (
    TunableParams,
    TuningKey,
    get_tuned_params,
)


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


def _state_pool_mixed_key(**overrides) -> TuningKey:
    fields = dict(
        case="mixed",
        max_num_tokens=256,
        actual_num_q_heads=6,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="bfloat16",
        q_dtype="bfloat16",
        page_size_per_kv_packing=162,
        kv_packing=2,
        max_num_seqs=32,
        pages_per_seq=26,
    )
    fields.update(overrides)
    return TuningKey(**fields)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(actual_num_q_heads=4),
        dict(actual_num_q_heads=8, page_size_per_kv_packing=324),
        dict(actual_num_q_heads=12, page_size_per_kv_packing=324),
        dict(max_num_tokens=64, max_num_seqs=8, pages_per_seq=101),
        dict(max_num_tokens=8192, max_num_seqs=64, pages_per_seq=13),
    ],
)
def test_state_pool_mixed_row_serves_every_batch_shape(overrides) -> None:
    key = _state_pool_mixed_key(**overrides)
    assert key not in tuned_params.tuned_params_mapping

    tuned = get_tuned_params(key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 256
    assert tuned.q_split == 2
    assert tuned.vmem_limit_bytes == 60 * 1024 * 1024


@pytest.mark.parametrize(
    "overrides",
    [
        dict(max_num_tokens=32),
        dict(case="batched_decode"),
        dict(kv_dtype="float8_e4m3fn"),
        dict(q_dtype="float8_e4m3fn"),
        dict(kv_packing=4),
        dict(actual_lkv_dim=256),
        dict(actual_r_dim=32),
        dict(actual_num_q_heads=8),
        dict(page_size_per_kv_packing=324),
        dict(soft_cap=30.0),
    ],
)
def test_keys_outside_the_state_pool_rows_keep_the_default(
    monkeypatch: pytest.MonkeyPatch, overrides
) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    key = _state_pool_mixed_key(**overrides)

    tuned = get_tuned_params(key)

    assert tuned.q_split == 1
    if key.case == "mixed":
        assert tuned.num_queries_per_block == 16
    else:
        assert tuned.decode_batch_size == 4


def _off_table_neighbours(row: TuningKey) -> list[TuningKey]:
    """Keys of the row's model on another batch shape or token bucket."""
    return [
        replace(row, max_num_tokens=row.max_num_tokens * 2 + 1),
        replace(row, max_num_seqs=row.max_num_seqs + 1),
        replace(row, pages_per_seq=row.pages_per_seq + 1),
    ]


def test_off_table_keys_of_the_tuned_rows_keep_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No row added for one workload may capture the off-table keys of the
    # models the static table was tuned for.
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mixed_default = TunableParams(
        num_kv_pages_per_block=1, num_queries_per_block=16, vmem_limit_bytes=62914560
    )
    decode_default = TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    )
    checked = 0
    for row in tuned_params.tuned_params_mapping:
        for key in _off_table_neighbours(row):
            if key in tuned_params.tuned_params_mapping:
                continue
            expected = mixed_default if key.case == "mixed" else decode_default
            assert get_tuned_params(key) == expected, key
            checked += 1
    assert checked > len(tuned_params.tuned_params_mapping)
