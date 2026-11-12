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

import pytest

from vllm_torchtpu.distributed import dp_tp_mesh


def _groups(rank, dp, tp):
    d, t = divmod(rank, tp)
    return [x * tp + t for x in range(dp)], list(range(d * tp, (d + 1) * tp))


@pytest.mark.parametrize(("dp", "tp"), [(2, 4), (1, 8), (8, 1), (8, 16)])
def test_validate_accepts_rank_ordered_grid(dp, tp):
    world = dp * tp
    for rank in sorted({0, 5 % world, world // 2, world - 1}):
        dp_ranks, tp_ranks = _groups(rank, dp, tp)
        got = dp_tp_mesh.validate_dp_tp_group_ranks(
            dp,
            tp,
            dp_ranks=dp_ranks,
            tp_ranks=tp_ranks,
            global_rank=rank,
            world_size=world,
        )
        assert got == divmod(rank, tp)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp_ranks": [4, 5], "tp_ranks": [4, 5, 6, 7]},  # dp column scrambled
        {"dp_ranks": [1, 5], "tp_ranks": [5, 4, 6, 7]},  # tp row permuted
        {"dp_ranks": [1, 5], "tp_ranks": [4, 5, 6]},  # tp row short
    ],
)
def test_validate_rejects_mismatched_groups(kwargs):
    with pytest.raises(ValueError):
        dp_tp_mesh.validate_dp_tp_group_ranks(
            2, 4, global_rank=5, world_size=8, **kwargs
        )


@pytest.mark.parametrize(
    ("dp", "tp", "rank", "world"),
    [(2, 4, 5, 16), (2, 4, 8, 8), (2, 4, -1, 8), (0, 4, 0, 0), (2, 0, 0, 0)],
)
def test_validate_rejects_bad_sizes(dp, tp, rank, world):
    with pytest.raises(ValueError):
        dp_tp_mesh.validate_dp_tp_group_ranks(
            dp, tp, global_rank=rank, world_size=world
        )


def test_validate_reads_torch_distributed(monkeypatch):
    monkeypatch.setattr(dp_tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_rank", lambda: 37)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_world_size", lambda: 128)
    assert dp_tp_mesh.validate_dp_tp_group_ranks(8, 16) == (2, 5)


def test_validate_needs_torch_distributed_without_explicit_rank(monkeypatch):
    monkeypatch.setattr(dp_tp_mesh.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError):
        dp_tp_mesh.validate_dp_tp_group_ranks(2, 4, global_rank=1)
