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
"""Torch ``@jax_op`` builders for the Kimi K3 Pallas remote-DMA collectives.

Each builder wraps a width-guarded collective from
``vllm_torchtpu.kernels.kimi_k3.pallas_dma`` in a ``jax.shard_map`` over the
2D ``('dp', 'tp')`` mesh and registers it as a torch custom op with a fake
(shape-only) implementation for ``torch.compile``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu.kernels.kimi_k3.pallas_dma import (
    COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
    COLLECTIVE_ID_MOE_DP_COMBINE_RS,
    COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    COLLECTIVE_ID_MOE_TP_LATENT_AG,
    DEFAULT_MAX_DMA_TOKENS,
    DP_AXIS_NAME,
    TP_AXIS_NAME,
    axis_tuple,
    kimi_guarded_all_gather_cols,
    kimi_guarded_all_gather_rows,
    kimi_guarded_psum,
    kimi_guarded_psum_scatter_rows,
    normalize_axis_name,
)

REGISTERED_OPS: dict[tuple[Any, ...], Callable[[torch.Tensor], torch.Tensor]] = {}


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", tag)
    return cleaned or "v1"


def format_pallas_dma_op_name(
    *,
    kind: str,
    spec_tag: str,
    axis_name: str | Sequence[str],
    collective_id: int,
    recv_memory_space: str,
    max_dma_tokens: int,
    dp_size: int,
    tp_size: int,
    variant_tag: str,
) -> str:
    """Construct a unique ``pallas::...`` custom-op name encoding all variant knobs (D87)."""
    ax_str = "_".join(axis_tuple(axis_name))
    v_str = sanitize_tag(variant_tag)
    rmem_str = sanitize_tag(recv_memory_space)
    return (
        f"pallas::kimi_{sanitize_tag(kind)}_{sanitize_tag(spec_tag)}"
        f"_ax_{ax_str}_cid{int(collective_id)}_rmem_{rmem_str}"
        f"_cap{int(max_dma_tokens)}_dp{int(dp_size)}_tp{int(tp_size)}_{v_str}"
    )


def mesh_shape_and_ids(mesh: Mesh) -> tuple[int, int, tuple[int, ...]]:
    if tuple(mesh.axis_names) != (DP_AXIS_NAME, TP_AXIS_NAME):
        raise ValueError(
            f"Expected mesh.axis_names == ('dp', 'tp'), got {mesh.axis_names}"
        )
    dp_size, tp_size = (int(mesh.devices.shape[0]), int(mesh.devices.shape[1]))
    dev_ids = tuple(int(d.id) for d in mesh.devices.flat)
    return dp_size, tp_size, dev_ids


def build_pallas_all_gather_cols_op(
    mesh: Mesh,
    *,
    axis_name: str = TP_AXIS_NAME,
    collective_id: int = COLLECTIVE_ID_MOE_TP_LATENT_AG,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "opt2d_1",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build ``PartitionSpec('dp', 'tp') -> PartitionSpec('dp', None)`` column all-gather."""
    if axis_name != TP_AXIS_NAME:
        raise ValueError(f"all_gather_cols expects axis_name='tp', got {axis_name!r}")
    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
    in_spec = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)
    out_spec = PartitionSpec(DP_AXIS_NAME, None)
    out_sharding = NamedSharding(mesh, out_spec)

    op_name = format_pallas_dma_op_name(
        kind="ag_cols",
        spec_tag="dp_tp",
        axis_name=axis_name,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    cache_key = ("ag_cols", op_name, dev_ids)
    if cache_key in REGISTERED_OPS:
        return REGISTERED_OPS[cache_key]

    def _ag_cols_jax(x: jax.Array) -> jax.Array:
        def _local(part: jax.Array) -> jax.Array:
            return kimi_guarded_all_gather_cols(
                part,
                TP_AXIS_NAME,
                collective_id=collective_id,
                recv_memory_space=recv_memory_space,
                max_dma_tokens=max_dma_tokens,
            )

        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_spec,
            out_specs=out_spec,
            check_vma=False,
        )(x)
        return jax.lax.with_sharding_constraint(out, out_sharding)

    raw_op = jax_op(
        op_name,
        _ag_cols_jax,
        mesh=mesh,
        input_partition_specs=(in_spec,),
        output_partition_specs=(out_spec,),
    )
    raw_op.register_fake(lambda x: x.new_empty((x.shape[0], x.shape[1] * tp_size)))

    def _call(x: torch.Tensor) -> torch.Tensor:
        return raw_op(x)

    _call._qualname = op_name  # type: ignore[attr-defined]
    REGISTERED_OPS[cache_key] = _call
    return _call


def build_pallas_all_gather_rows_op(
    mesh: Mesh,
    *,
    axis_name: str | Sequence[str] = DP_AXIS_NAME,
    collective_id: int = COLLECTIVE_ID_MOE_DP_DISPATCH_AG,
    input_sharding: str = "dp_replicated",
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "opt2d_1",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build 1-shot Pallas row all-gather over ``'dp'``, ``'tp'``, or ``('dp', 'tp')``.

    ``input_sharding``:
      - ``'dp_replicated'`` (for ``axis_name='dp'``): input is ``[B_dp, D]``
        replicated across ``'tp'`` (``PartitionSpec('dp', None)``), output is
        ``[dp_size * B_dp, D]`` replicated across ``('dp', 'tp')``
        (``PartitionSpec(None, None)``).
      - ``'rank_unshared'`` (for ``'tp'``, ``'dp'``, or ``('dp', 'tp')``): every
        ``(dp, tp)`` rank holds its own independent ``[B_in, D]`` rows, passed
        via zero-copy ``x.unsqueeze(1)`` with ``PartitionSpec('dp', 'tp', None)``.
    """
    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
    axis_key = normalize_axis_name(axis_name)
    axes_t = axis_tuple(axis_key)
    if axes_t == (DP_AXIS_NAME,):
        gather_factor = dp_size
    elif axes_t == (TP_AXIS_NAME,):
        gather_factor = tp_size
    elif axes_t == (DP_AXIS_NAME, TP_AXIS_NAME):
        gather_factor = dp_size * tp_size
    else:
        raise ValueError(f"Unsupported axis_name={axis_name!r} for 2D ('dp', 'tp') mesh")

    if input_sharding == "dp_replicated" and axes_t != (DP_AXIS_NAME,):
        raise ValueError(
            "input_sharding='dp_replicated' is only valid when gathering across 'dp'; "
            "use input_sharding='rank_unshared' when gathering distinct rows across 'tp'."
        )
    if input_sharding not in ("dp_replicated", "rank_unshared"):
        raise ValueError(
            f"input_sharding must be 'dp_replicated' or 'rank_unshared', got {input_sharding!r}"
        )

    op_name = format_pallas_dma_op_name(
        kind="ag_rows",
        spec_tag=input_sharding,
        axis_name=axis_key,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    cache_key = ("ag_rows", op_name, dev_ids)
    if cache_key in REGISTERED_OPS:
        return REGISTERED_OPS[cache_key]

    if input_sharding == "dp_replicated":
        in_spec = PartitionSpec(DP_AXIS_NAME, None)
        out_spec = PartitionSpec(None, None)
        out_sharding = NamedSharding(mesh, out_spec)

        def _ag_dp_rep_jax(x: jax.Array) -> jax.Array:
            def _local(part: jax.Array) -> jax.Array:
                return kimi_guarded_all_gather_rows(
                    part,
                    axis_key,
                    collective_id=collective_id,
                    recv_memory_space=recv_memory_space,
                    max_dma_tokens=max_dma_tokens,
                )

            out = jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=in_spec,
                out_specs=out_spec,
                check_vma=False,
            )(x)
            return jax.lax.with_sharding_constraint(out, out_sharding)

        raw_op = jax_op(
            op_name,
            _ag_dp_rep_jax,
            mesh=mesh,
            input_partition_specs=(in_spec,),
            output_partition_specs=(out_spec,),
        )
        raw_op.register_fake(
            lambda x: x.new_empty((x.shape[0] * gather_factor, x.shape[1]))
        )

        def _call_dp_rep(x: torch.Tensor) -> torch.Tensor:
            return raw_op(x)

        _call_dp_rep._qualname = op_name  # type: ignore[attr-defined]
        REGISTERED_OPS[cache_key] = _call_dp_rep
        return _call_dp_rep

    # rank_unshared: 3D view [B_in, 1, D] partitioned across ('dp', 'tp', None).
    in_spec_3d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME, None)
    if axes_t == (TP_AXIS_NAME,):
        out_spec_unshared = PartitionSpec(DP_AXIS_NAME, None)
        squeeze_out = False
    elif axes_t == (DP_AXIS_NAME,):
        out_spec_unshared = PartitionSpec(None, TP_AXIS_NAME, None)
        squeeze_out = True
    else:
        out_spec_unshared = PartitionSpec(None, None)
        squeeze_out = False
    out_sharding_u = NamedSharding(mesh, out_spec_unshared)

    def _ag_unshared_jax(x3d: jax.Array) -> jax.Array:
        def _local(part3d: jax.Array) -> jax.Array:
            res2d = kimi_guarded_all_gather_rows(
                jnp.squeeze(part3d, axis=1),
                axis_key,
                collective_id=collective_id,
                recv_memory_space=recv_memory_space,
                max_dma_tokens=max_dma_tokens,
            )
            return res2d[:, None, :] if squeeze_out else res2d

        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_spec_3d,
            out_specs=out_spec_unshared,
            check_vma=False,
        )(x3d)
        return jax.lax.with_sharding_constraint(out, out_sharding_u)

    raw_op_u = jax_op(
        op_name,
        _ag_unshared_jax,
        mesh=mesh,
        input_partition_specs=(in_spec_3d,),
        output_partition_specs=(out_spec_unshared,),
    )
    if squeeze_out:
        raw_op_u.register_fake(
            lambda x3d: x3d.new_empty((x3d.shape[0] * gather_factor, 1, x3d.shape[2]))
        )
    else:
        raw_op_u.register_fake(
            lambda x3d: x3d.new_empty((x3d.shape[0] * gather_factor, x3d.shape[2]))
        )

    def _call_unshared(x: torch.Tensor) -> torch.Tensor:
        y = raw_op_u(x.unsqueeze(1))
        return y.squeeze(1) if squeeze_out else y

    _call_unshared._qualname = op_name  # type: ignore[attr-defined]
    REGISTERED_OPS[cache_key] = _call_unshared
    return _call_unshared


def build_pallas_psum_op(
    mesh: Mesh,
    *,
    axis_name: str | Sequence[str] = TP_AXIS_NAME,
    collective_id: int = COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "opt2d_1",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build deterministic 1-shot Pallas ``psum`` over ``'tp'``, ``'dp'``, or ``('dp', 'tp')``."""
    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
    axis_key = normalize_axis_name(axis_name)
    axes_t = axis_tuple(axis_key)

    in_spec_3d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME, None)
    if axes_t == (TP_AXIS_NAME,):
        out_spec = PartitionSpec(DP_AXIS_NAME, None)
        squeeze_out = False
    elif axes_t == (DP_AXIS_NAME,):
        out_spec = PartitionSpec(None, TP_AXIS_NAME, None)
        squeeze_out = True
    elif axes_t == (DP_AXIS_NAME, TP_AXIS_NAME):
        out_spec = PartitionSpec(None, None)
        squeeze_out = False
    else:
        raise ValueError(f"Unsupported axis_name={axis_name!r} for 2D ('dp', 'tp') mesh")

    op_name = format_pallas_dma_op_name(
        kind="psum",
        spec_tag="rank_unshared",
        axis_name=axis_key,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    cache_key = ("psum", op_name, dev_ids)
    if cache_key in REGISTERED_OPS:
        return REGISTERED_OPS[cache_key]

    out_sharding = NamedSharding(mesh, out_spec)

    def _psum_jax(x3d: jax.Array) -> jax.Array:
        def _local(part3d: jax.Array) -> jax.Array:
            res2d = kimi_guarded_psum(
                jnp.squeeze(part3d, axis=1),
                axis_key,
                collective_id=collective_id,
                recv_memory_space=recv_memory_space,
                max_dma_tokens=max_dma_tokens,
            )
            return res2d[:, None, :] if squeeze_out else res2d

        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_spec_3d,
            out_specs=out_spec,
            check_vma=False,
        )(x3d)
        return jax.lax.with_sharding_constraint(out, out_sharding)

    raw_op = jax_op(
        op_name,
        _psum_jax,
        mesh=mesh,
        input_partition_specs=(in_spec_3d,),
        output_partition_specs=(out_spec,),
    )
    if squeeze_out:
        raw_op.register_fake(lambda x3d: x3d.new_empty(x3d.shape))
    else:
        raw_op.register_fake(
            lambda x3d: x3d.new_empty((x3d.shape[0], x3d.shape[2]))
        )

    def _call(x: torch.Tensor) -> torch.Tensor:
        y = raw_op(x.unsqueeze(1))
        return y.squeeze(1) if squeeze_out else y

    _call._qualname = op_name  # type: ignore[attr-defined]
    REGISTERED_OPS[cache_key] = _call
    return _call


def build_pallas_psum_scatter_rows_op(
    mesh: Mesh,
    *,
    axis_name: str | Sequence[str] = DP_AXIS_NAME,
    collective_id: int = COLLECTIVE_ID_MOE_DP_COMBINE_RS,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    variant_tag: str = "opt2d_1",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build deterministic 1-shot Pallas ``psum_scatter_rows`` over ``'dp'``, ``'tp'``, or ``('dp', 'tp')``."""
    dp_size, tp_size, dev_ids = mesh_shape_and_ids(mesh)
    axis_key = normalize_axis_name(axis_name)
    axes_t = axis_tuple(axis_key)
    if axes_t == (DP_AXIS_NAME,):
        scatter_factor = dp_size
    elif axes_t == (TP_AXIS_NAME,):
        scatter_factor = tp_size
    elif axes_t == (DP_AXIS_NAME, TP_AXIS_NAME):
        scatter_factor = dp_size * tp_size
    else:
        raise ValueError(f"Unsupported axis_name={axis_name!r} for 2D ('dp', 'tp') mesh")

    in_spec_3d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME, None)
    out_spec_3d = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME, None)
    out_sharding = NamedSharding(mesh, out_spec_3d)

    op_name = format_pallas_dma_op_name(
        kind="rs_rows",
        spec_tag="rank_unshared",
        axis_name=axis_key,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
        max_dma_tokens=max_dma_tokens,
        dp_size=dp_size,
        tp_size=tp_size,
        variant_tag=variant_tag,
    )
    cache_key = ("rs_rows", op_name, dev_ids)
    if cache_key in REGISTERED_OPS:
        return REGISTERED_OPS[cache_key]

    def _rs_jax(x3d: jax.Array) -> jax.Array:
        def _local(part3d: jax.Array) -> jax.Array:
            res2d = kimi_guarded_psum_scatter_rows(
                jnp.squeeze(part3d, axis=1),
                axis_key,
                collective_id=collective_id,
                recv_memory_space=recv_memory_space,
                max_dma_tokens=max_dma_tokens,
            )
            return res2d[:, None, :]

        out = jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_spec_3d,
            out_specs=out_spec_3d,
            check_vma=False,
        )(x3d)
        return jax.lax.with_sharding_constraint(out, out_sharding)

    raw_op = jax_op(
        op_name,
        _rs_jax,
        mesh=mesh,
        input_partition_specs=(in_spec_3d,),
        output_partition_specs=(out_spec_3d,),
    )
    raw_op.register_fake(
        lambda x3d: x3d.new_empty(
            (x3d.shape[0] // scatter_factor, 1, x3d.shape[2])
        )
    )

    def _call(x: torch.Tensor) -> torch.Tensor:
        return raw_op(x.unsqueeze(1)).squeeze(1)

    _call._qualname = op_name  # type: ignore[attr-defined]
    REGISTERED_OPS[cache_key] = _call
    return _call
