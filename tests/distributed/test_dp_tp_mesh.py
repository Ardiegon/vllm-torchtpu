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

from vllm_torchtpu.distributed import dp_tp_mesh, ep_mesh


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


@pytest.fixture
def ep_state(monkeypatch):
    """A fresh EP mesh state whose devices sort differently from the ranks."""
    monkeypatch.setattr(ep_mesh, "_rank_order_required", False)
    monkeypatch.setattr(ep_mesh, "_MESH_CACHE", {})
    monkeypatch.setattr(ep_mesh, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(ep_mesh, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(ep_mesh.envs, "MOE_FUSED_EP_SORTED_GATHER", False)
    monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda: (40, 20, 10, 30))


def test_ep_mesh_stays_device_id_ordered_until_required(ep_state):
    assert not dp_tp_mesh.dp_tp_mesh_order_required()
    assert ep_mesh.ep_rank_order() == (2, 1, 3, 0)

    dp_tp_mesh.require_dp_tp_mesh_order()

    assert dp_tp_mesh.dp_tp_mesh_order_required()
    assert ep_mesh.ep_rank_order() == (0, 1, 2, 3)


def test_require_rejects_a_cached_device_id_ordered_ep_mesh(ep_state):
    ep_mesh._MESH_CACHE[(ep_mesh.EP_AXIS_NAME, (10, 20, 30, 40))] = object()

    with pytest.raises(RuntimeError, match="device-id order"):
        dp_tp_mesh.require_dp_tp_mesh_order()

    assert not dp_tp_mesh.dp_tp_mesh_order_required()


def test_require_accepts_a_cached_rank_ordered_ep_mesh(ep_state):
    ep_mesh._MESH_CACHE[(ep_mesh.EP_AXIS_NAME, (40, 20, 10, 30))] = object()

    dp_tp_mesh.require_dp_tp_mesh_order()
    dp_tp_mesh.require_dp_tp_mesh_order()

    assert dp_tp_mesh.dp_tp_mesh_order_required()


def _fake_2d_mesh(ids, shape, axis_names=("dp", "tp")):
    from types import SimpleNamespace

    import numpy as np

    devs = np.array([SimpleNamespace(id=d) for d in ids], dtype=object).reshape(shape)
    return SimpleNamespace(axis_names=axis_names, devices=devs)


def test_build_dp_tp_mesh_validates_groups_and_ep_assignment(monkeypatch, ep_state):
    from types import SimpleNamespace

    from torch_tpu._internal import pallas as pallas_mod

    monkeypatch.setattr(dp_tp_mesh, "_MESH_2D_CACHE", {})
    monkeypatch.setattr(dp_tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_world_size", lambda: 4)
    dp_ranks, tp_ranks = _groups(1, 2, 2)
    monkeypatch.setattr(dp_tp_mesh.ps, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        dp_tp_mesh.ps, "get_dp_group", lambda: SimpleNamespace(ranks=dp_ranks)
    )
    monkeypatch.setattr(
        dp_tp_mesh.ps, "get_tp_group", lambda: SimpleNamespace(ranks=tp_ranks)
    )
    rank_ids = (40, 20, 10, 30)
    calls = []

    def fake_get_pallas_mesh(*, axis_names, mesh_shape):
        calls.append((axis_names, mesh_shape))
        return _fake_2d_mesh(rank_ids, mesh_shape, axis_names)

    monkeypatch.setattr(pallas_mod, "get_pallas_mesh", fake_get_pallas_mesh)
    monkeypatch.setattr(
        ep_mesh,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]),
    )
    monkeypatch.setattr(
        ep_mesh, "build_ep_mesh", lambda: _fake_2d_mesh(rank_ids, (4,), ("d",))
    )

    mesh = dp_tp_mesh.build_dp_tp_mesh()
    assert dp_tp_mesh.dp_tp_mesh_order_required()
    assert calls[0] == (("dp", "tp"), (2, 2))
    assert dp_tp_mesh.build_dp_tp_mesh() is mesh


def test_build_dp_tp_mesh_rejects_conflicting_ep_device_assignment(
    monkeypatch, ep_state
):
    from types import SimpleNamespace

    from torch_tpu._internal import pallas as pallas_mod

    monkeypatch.setattr(dp_tp_mesh, "_MESH_2D_CACHE", {})
    monkeypatch.setattr(dp_tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dp_tp_mesh.ps, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(
        pallas_mod,
        "get_pallas_mesh",
        lambda *, axis_names, mesh_shape: _fake_2d_mesh(
            (40, 20, 10, 30), mesh_shape, axis_names
        ),
    )
    monkeypatch.setattr(
        ep_mesh,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]),
    )
    monkeypatch.setattr(
        ep_mesh, "build_ep_mesh", lambda: _fake_2d_mesh((10, 20, 30, 40), (4,), ("d",))
    )

    with pytest.raises(RuntimeError, match="Conflicting DeviceAssignment"):
        dp_tp_mesh.build_dp_tp_mesh(2, 2)


def test_build_dp_tp_mesh_inside_compile_requires_cached_entry(monkeypatch, ep_state):
    monkeypatch.setattr(dp_tp_mesh, "_MESH_2D_CACHE", {})
    monkeypatch.setattr(dp_tp_mesh.ps, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(dp_tp_mesh.torch.compiler, "is_compiling", lambda: True)

    with pytest.raises(RuntimeError, match="before torch.compile tracing"):
        dp_tp_mesh.build_dp_tp_mesh(2, 2)
    cached = _fake_2d_mesh((0, 1, 2, 3), (2, 2))
    dp_tp_mesh._MESH_2D_CACHE[(2, 2, (0, 1, 2, 3))] = cached
    assert dp_tp_mesh.build_dp_tp_mesh(2, 2) is cached
