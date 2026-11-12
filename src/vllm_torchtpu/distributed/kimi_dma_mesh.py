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
"""Rank-ordered 2D ``('dp', 'tp')`` JAX mesh for Kimi K3 Pallas remote-DMA ops.

``ep_mesh.build_ep_mesh`` sorts devices by hardware ``d.id`` by default, which
does not match ``global_rank = dp_rank * tp_size + tp_rank`` on multi-host TPU
slices. Reshaping a hardware-sorted 1D mesh to ``(dp_size, tp_size)`` would
scramble ``('dp', 'tp')`` coordinates relative to vLLM's ``dp_group`` and
``tp_group``.

Instead, ``build_ep_2d_mesh`` constructs a PyTorch-rank-ordered 2D mesh via
``torch_tpu._internal.pallas.get_pallas_mesh(axis_names=('dp', 'tp'),
mesh_shape=(dp_size, tp_size))`` (mirroring ``dcp.get_or_create_dcp_mesh``),
asserts on every worker that ``dp_group.ranks == mesh_ranks[:, tp_rank]`` and
``tp_group.ranks == mesh_ranks[dp_rank, :]``, and verifies that its flattened
``DeviceAssignment`` matches ``ep_mesh``'s rank-ordered assignment when
``TPU_KIMI_OPT_2D_EP_MESH=1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from jax.sharding import Mesh

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"
KIMI_2D_MESH_AXES: tuple[str, str] = (DP_AXIS_NAME, TP_AXIS_NAME)

_MESH_2D_CACHE: dict[tuple[int, int, tuple[int, ...]], Mesh] = {}


def reset_ep_2d_mesh() -> None:
    """Clear cached 2D ``('dp', 'tp')`` meshes."""
    _MESH_2D_CACHE.clear()


def kimi_use_2d_ep_mesh() -> bool:
    """True when any Kimi K3 2D ``('dp', 'tp')`` mesh / Pallas DMA opt-in flag is active."""
    from vllm_torchtpu import envs

    tp_psum_raw = (envs.TPU_KIMI_OPT_LATENCY_TP_PSUM or "off").strip().lower()
    tp_psum_active = tp_psum_raw not in ("", "0", "false", "no", "off", "none")
    exit_fold_raw = (envs.TPU_KIMI_OPT_EXIT_FOLD or "off").strip().lower()
    exit_fold_active = exit_fold_raw not in ("", "0", "false", "no", "off", "none")
    return bool(
        envs.TPU_KIMI_OPT_2D_EP_MESH
        or envs.TPU_KIMI_OPT_LATENCY_COLLECTIVES
        or tp_psum_active
        or exit_fold_active
    )


def _resolve_vllm_dp_tp_groups() -> tuple[
    int, int, tuple[int, ...] | None, tuple[int, ...] | None
]:
    """Resolve ``(dp_size, tp_size, dp_ranks, tp_ranks)`` from vLLM parallel state."""
    from vllm.distributed import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        raise RuntimeError(
            "vLLM parallel_state is not initialized; pass explicit dp_size/tp_size "
            "(and dp_ranks/tp_ranks) when calling outside vLLM."
        )
    dp_group = ps.get_dp_group()
    tp_group = ps.get_tp_group()
    dp_ranks = tuple(int(r) for r in dp_group.ranks)
    tp_ranks = tuple(int(r) for r in tp_group.ranks)
    return len(dp_ranks), len(tp_ranks), dp_ranks, tp_ranks


def validate_ep_2d_group_ranks(
    dp_size: int,
    tp_size: int,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> tuple[int, int]:
    """Validate that ``dp_ranks`` and ``tp_ranks`` match a ``(dp_size, tp_size)`` grid.

    Position ``(d, t)`` of the 2D mesh corresponds 1:1 to PyTorch global rank
    ``d * tp_size + t``. Consequently:
      - ``dp_group.ranks`` must equal column ``[d * tp_size + tp_rank for d in range(dp_size)]``
      - ``tp_group.ranks`` must equal row ``[dp_rank * tp_size + t for t in range(tp_size)]``

    Returns:
        ``(dp_rank, tp_rank)`` of ``global_rank`` in the 2D mesh.
    """
    if dp_size < 1 or tp_size < 1:
        raise ValueError(
            f"dp_size and tp_size must be >= 1, got dp_size={dp_size}, tp_size={tp_size}"
        )

    if global_rank is None or world_size is None:
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized; pass explicit global_rank and world_size."
            )
        if global_rank is None:
            global_rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()

    expected_world = dp_size * tp_size
    if world_size != expected_world:
        raise ValueError(
            f"2D ('dp', 'tp') mesh requires world_size == dp_size * tp_size "
            f"({dp_size} * {tp_size} = {expected_world}), got world_size={world_size}."
        )
    if not (0 <= global_rank < world_size):
        raise ValueError(
            f"global_rank={global_rank} out of bounds for world_size={world_size}."
        )

    dp_rank, tp_rank = divmod(global_rank, tp_size)
    expected_dp_ranks = tuple(d * tp_size + tp_rank for d in range(dp_size))
    expected_tp_ranks = tuple(
        range(dp_rank * tp_size, (dp_rank + 1) * tp_size)
    )

    if dp_ranks is not None:
        got_dp = tuple(int(r) for r in dp_ranks)
        if got_dp != expected_dp_ranks:
            raise ValueError(
                f"dp_group.ranks mismatch on global_rank={global_rank} "
                f"(dp_rank={dp_rank}, tp_rank={tp_rank}): "
                f"got {got_dp}, expected mesh_ranks[:, {tp_rank}]={expected_dp_ranks}."
            )

    if tp_ranks is not None:
        got_tp = tuple(int(r) for r in tp_ranks)
        if got_tp != expected_tp_ranks:
            raise ValueError(
                f"tp_group.ranks mismatch on global_rank={global_rank} "
                f"(dp_rank={dp_rank}, tp_rank={tp_rank}): "
                f"got {got_tp}, expected mesh_ranks[{dp_rank}, :]={expected_tp_ranks}."
            )

    return dp_rank, tp_rank


def build_ep_2d_mesh(
    dp_size: int | None = None,
    tp_size: int | None = None,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
    validate_ep_device_assignment: bool = True,
) -> Mesh:
    """Build (or return cached) rank-ordered 2D ``('dp', 'tp')`` JAX mesh.

    Must be pre-built outside ``torch.compile`` (e.g., during model/weight
    initialization) so ``jax.sharding.Mesh`` construction never enters a
    Dynamo-traced frame.
    """
    if dp_size is None or tp_size is None:
        v_dp, v_tp, v_dp_ranks, v_tp_ranks = _resolve_vllm_dp_tp_groups()
        dp_size = v_dp if dp_size is None else dp_size
        tp_size = v_tp if tp_size is None else tp_size
        if dp_ranks is None:
            dp_ranks = v_dp_ranks
        if tp_ranks is None:
            tp_ranks = v_tp_ranks
    elif dp_ranks is None and tp_ranks is None:
        try:
            _, _, v_dp_ranks, v_tp_ranks = _resolve_vllm_dp_tp_groups()
            dp_ranks = v_dp_ranks
            tp_ranks = v_tp_ranks
        except RuntimeError:
            pass

    if torch.compiler.is_compiling():
        for (cached_dp, cached_tp, _), cached_mesh in _MESH_2D_CACHE.items():
            if cached_dp == dp_size and cached_tp == tp_size:
                return cached_mesh
        raise RuntimeError(
            "build_ep_2d_mesh must be called before torch.compile tracing "
            f"(no cached mesh found for dp_size={dp_size}, tp_size={tp_size})."
        )

    dp_rank, tp_rank = validate_ep_2d_group_ranks(
        dp_size,
        tp_size,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )

    from torch_tpu._internal.pallas.pallas import _get_pallas_mesh

    mesh = _get_pallas_mesh(
        axis_names=KIMI_2D_MESH_AXES,
        mesh_shape=(dp_size, tp_size),
    )
    if tuple(mesh.axis_names) != KIMI_2D_MESH_AXES:
        raise RuntimeError(
            f"Unexpected mesh axis_names={mesh.axis_names}, expected {KIMI_2D_MESH_AXES}."
        )
    if tuple(mesh.devices.shape) != (dp_size, tp_size):
        raise RuntimeError(
            f"Unexpected 2D mesh shape={mesh.devices.shape}, expected {(dp_size, tp_size)}."
        )

    flat_device_ids = tuple(int(d.id) for d in mesh.devices.flat)
    if len(set(flat_device_ids)) != len(flat_device_ids):
        raise RuntimeError(
            f"Duplicate TPU device IDs in 2D mesh: {flat_device_ids}"
        )

    if validate_ep_device_assignment:
        from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, get_ep_group

        if get_ep_group() is not None:
            ep_1d_mesh = build_ep_mesh()
            if ep_1d_mesh is not None:
                ep_1d_ids = tuple(int(d.id) for d in ep_1d_mesh.devices.flat)
                if flat_device_ids != ep_1d_ids:
                    raise RuntimeError(
                        "Conflicting DeviceAssignment between 2D ('dp', 'tp') mesh "
                        f"{flat_device_ids} and 1D EP mesh {ep_1d_ids}. "
                        "Enable TPU_KIMI_OPT_2D_EP_MESH=1 so all multi-device "
                        "@jax_op kernels share one rank-ordered DeviceAssignment."
                    )

    cache_key = (dp_size, tp_size, flat_device_ids)
    if cache_key not in _MESH_2D_CACHE:
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        my_dev = mesh.devices[dp_rank, tp_rank]
        my_coords = getattr(my_dev, "coords", None)
        if my_coords is None:
            try:
                from torch_tpu._internal.distributed import tpu_distributed

                my_coords = [int(c) for c in tpu_distributed.local_device_coords()]
            except Exception:
                my_coords = None
        logger.info(
            "Built Kimi 2D EP mesh | shape=(dp=%d, tp=%d) | rank=%d -> (dp=%d, tp=%d) "
            "| dev_id=%d coords=%s | flat_device_ids=%s",
            dp_size,
            tp_size,
            global_rank,
            dp_rank,
            tp_rank,
            int(my_dev.id),
            my_coords,
            list(flat_device_ids),
        )
        _MESH_2D_CACHE[cache_key] = mesh

    return _MESH_2D_CACHE[cache_key]
