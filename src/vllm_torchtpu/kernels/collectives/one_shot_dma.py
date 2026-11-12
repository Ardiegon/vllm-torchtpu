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
"""Pallas one-shot remote-DMA collectives for small batches.

A one-shot collective sends each peer its piece with a single point-to-point
remote DMA instead of going through XLA's collective ops, which is faster
for the few rows of a decode step.

1. ``barrier_and_allocate``, ``_color_dma_operands`` and ``call_kernel`` pin
   the memory space (``pltpu.HBM`` / ``pltpu.VMEM``) a buffer was created in
   across kernel boundaries, so XLA's memory-space pass never relocates a
   buffer that an in-flight remote DMA targets.
2. ``dense_all_to_all`` sends one ``make_async_remote_copy`` to each of the
   ``N - 1`` peers in parallel with a local VMEM copy. It pads the lane
   dimension to a lane multiple and retiles or pads the sublane dimension.
3. ``latency_all_gather_cols``, ``latency_psum`` and
   ``latency_psum_scatter_rows`` call ``dense_all_to_all`` on a 3D
   ``(axis_size, rows, lanes)`` view with ``split_axis=0, concat_axis=0``, so
   small batches with odd lane widths tile cleanly.
4. ``guarded_*`` pick the one-shot DMA collective for small inputs and the
   ``jax.lax`` collective otherwise; the choice is made at trace time from
   static shapes.

Collectives that can be in flight at the same time must use different
barrier ``collective_id`` values; callers assign them.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

Ref = Any

# Inputs with more rows than this take the jax.lax collective (prefill sizes).
DEFAULT_MAX_DMA_TOKENS = 64
# "auto" receive buffers go to VMEM only up to this many rows ...
DEFAULT_VMEM_MAX_TOKENS = 8
# ... and up to this many bytes.
_VMEM_RECV_MAX_BYTES = 2 * 1024 * 1024
# The one-shot DMA path is also limited to this many rows in total over the axis.
_ONE_SHOT_MAX_TOTAL_ROWS = 1024
# VMEM window of the local-copy pipeline in run_dense_a2a.
_LOCAL_COPY_VMEM_WINDOW_BYTES = 3 * 1024 * 1024
# XLA latency-hiding-scheduler estimate (cycles) for the barrier and the DMA.
_DMA_LATENCY_METADATA = "300000"


def normalize_axis_name(axis_name: str | Sequence[str]) -> str | tuple[str, ...]:
    """Hashable axis identifier for ``jax.lax`` and Pallas ``device_id``."""
    if isinstance(axis_name, str):
        return axis_name
    axes = tuple(axis_name)
    if not axes:
        raise ValueError("axis_name must not be empty")
    return axes[0] if len(axes) == 1 else axes


# =============================================================================
# Memory-Space Coloring, Barrier Allocation & Kernel Invocation Primitives
# =============================================================================


def block_spec_from_array(x: Any, force_any: bool = False) -> pl.BlockSpec:
    if (
        isinstance(x, pltpu.SemaphoreType)
        or x.dtype == pltpu.SemaphoreType.DMA(()).dtype
    ):
        return pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)
    if not x.shape:
        return pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)
    if force_any:
        return pl.BlockSpec(memory_space=pl.ANY)
    if x.dtype in (jnp.uint32, jnp.int32, jnp.uint16, jnp.int16):
        return pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)
    return pl.BlockSpec(memory_space=pl.ANY)


def _color_dma_operands(
    tree: Any,
    *,
    vmem_dense_recv: bool = False,
    vmem_all_data: bool = False,
    vmem_field_names: Sequence[str] = (),
) -> Any:
    """Pin HBM/VMEM operands explicitly on ``pl.ANY`` DMA kernel inputs/inout.

    Without explicit ``with_memory_space_constraint``, XLA's
    ``TpuCustomCallMemorySpacePolicy`` can stage ``pl.ANY`` operands through
    temporary VMEM copies while remote DMAs are still in flight.
    """
    vmem_names_set = frozenset(vmem_field_names)

    def _pin(path: tuple[Any, ...], x: Any) -> Any:
        if not isinstance(x, jax.Array) or not x.shape:
            return x
        if x.dtype == pltpu.SemaphoreType.DMA(()).dtype:
            return x
        if x.dtype in (jnp.uint32, jnp.int32, jnp.uint16, jnp.int16):
            return x
        is_dense_recv = any(getattr(entry, "name", None) == "x_recv" for entry in path)
        is_named_vmem = bool(vmem_names_set) and any(
            getattr(entry, "name", None) in vmem_names_set for entry in path
        )
        if vmem_all_data or (vmem_dense_recv and is_dense_recv) or is_named_vmem:
            space = pltpu.MemorySpace.VMEM
        else:
            space = pltpu.MemorySpace.HBM
        return pltpu.with_memory_space_constraint(x, space)

    return jax.tree_util.tree_map_with_path(_pin, tree)


def call_kernel(
    kernel: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    inout: Any = (),
    output: Any = (),
    any_args: tuple[Any, ...] = (),
    scoped: dict[str, Any] | None = None,
    compiler_params: Any = None,
    allow_pin: bool = True,
    vmem_dense_recv: bool = False,
    vmem_args: bool = False,
    vmem_inout_all: bool = False,
    vmem_inout_fields: Sequence[str] = (),
    **kwargs: Any,
) -> Any:
    """Invoke a Pallas kernel with explicit memory-space coloring and I/O aliasing."""
    scoped = {} if scoped is None else dict(scoped)
    if compiler_params is None:
        compiler_params = pltpu.CompilerParams()
    compiler_params = dataclasses.replace(
        compiler_params,
        allow_collective_id_without_custom_barrier=True,
    )
    if allow_pin:
        args = _color_dma_operands(
            args,
            vmem_dense_recv=vmem_dense_recv,
            vmem_all_data=vmem_args,
        )
        inout = _color_dma_operands(
            inout,
            vmem_dense_recv=vmem_dense_recv,
            vmem_all_data=vmem_inout_all,
            vmem_field_names=vmem_inout_fields,
        )
        any_args = _color_dma_operands(any_args, vmem_dense_recv=vmem_dense_recv)

    in_specs = jax.tree.map(
        functools.partial(block_spec_from_array, force_any=not allow_pin), args
    )
    any_specs = jax.tree.map(
        functools.partial(block_spec_from_array, force_any=True), any_args
    )
    inout_specs = jax.tree.map(
        functools.partial(block_spec_from_array, force_any=not allow_pin), inout
    )
    out_specs = jax.tree.map(
        functools.partial(block_spec_from_array, force_any=not allow_pin), output
    )
    io_aliasing = {i: i for i in range(len(jax.tree.leaves(inout_specs)))}

    def wrapped_kernel(
        inout_in_args: Any,
        in_args: Any,
        any_in_args: Any,
        inout_out_args: Any,
        out_args: Any,
    ) -> None:
        del inout_out_args
        pl.run_scoped(
            functools.partial(
                kernel, *in_args, *inout_in_args, *out_args, *any_in_args
            ),
            **scoped,
        )

    aliased_args, out_args = pl.pallas_call(
        wrapped_kernel,
        in_specs=(inout_specs, in_specs, any_specs),
        out_specs=(inout_specs, out_specs),
        out_shape=(
            inout,
            jax.tree.map(
                lambda x: (
                    pltpu.SMEM(x.shape, dtype=x.dtype)
                    if block_spec_from_array(x).memory_space == pltpu.MemorySpace.SMEM
                    else x
                ),
                output,
            ),
        ),
        input_output_aliases=io_aliasing,
        compiler_params=compiler_params,
        **kwargs,
    )(inout, args, any_args)

    return (*aliased_args, *out_args)


def barrier_and_allocate(
    to_alloc: Any,
    *,
    collective_id: int,
    axis_name: str | Sequence[str],
    order_dep: jax.Array | None = None,
    allow_pin: bool = True,
    order_dep_vmem: bool = False,
) -> Any:
    """Signal the hardware barrier semaphore across ``axis_name`` and allocate buffers.

    Under ``torch_tpu``'s ``jax_op`` SPMD export bridge, XLA's SPMD partitioner
    rejects zero-operand custom calls marked ``has_side_effects=True``
    (``Side-effect HLO must have sharding``). Passing ``order_dep`` (colored
    ``pltpu.HBM`` or ``pltpu.VMEM`` on ``pl.ANY``, unread inside the kernel) keeps
    ``has_side_effects=False`` safe against both HLO CSE and out-of-order
    scheduling across consecutive collectives sharing ``collective_id``, while
    ``to_alloc``'s dataflow into ``run_dense_a2a`` prevents DCE.
    """
    axis_key = normalize_axis_name(axis_name)
    in_args: tuple[Any, ...] = ()
    in_specs: Any = ()
    if order_dep is not None:
        dep_space = pltpu.MemorySpace.VMEM if order_dep_vmem else pltpu.MemorySpace.HBM
        dep = (
            pltpu.with_memory_space_constraint(order_dep, dep_space)
            if allow_pin
            else order_dep
        )
        in_args = (dep,)
        in_specs = (pl.BlockSpec(memory_space=pl.ANY),)

    def kernel(*_: Any) -> None:
        with jax.named_scope("main_barrier"):
            barrier_sem = pltpu.get_barrier_semaphore()
            for i in range(jax.lax.axis_size(axis_key)):
                pl.semaphore_signal(
                    barrier_sem,
                    1,
                    device_id={axis_key: i},
                    core_index=0,
                )

    tpu_gen = pltpu.get_tpu_info().generation
    compiler_flags = (
        {
            "XLA_TPU_STORE_TO_LOAD_FORWARDING_WINDOW": 99999,
            "XLA_TPU_SCHEDULE_ACROSS_SFLAG_OPS": True,
        }
        if tpu_gen >= 7
        else {}
    )
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=jax.tree.map(
            lambda x: block_spec_from_array(x, force_any=not allow_pin),
            to_alloc,
        ),
        out_shape=jax.tree.map(
            lambda x: (
                pltpu.SMEM(x.shape, dtype=x.dtype)
                if block_spec_from_array(x, force_any=not allow_pin).memory_space
                == pltpu.MemorySpace.SMEM
                else x
            ),
            to_alloc,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=False,
            allow_collective_id_without_custom_barrier=True,
            flags=compiler_flags,
        ),
    )(*in_args)


# =============================================================================
# 1-Shot Dense Remote-DMA All-to-All (`dense_all_to_all`)
# =============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DenseReceiveState:
    x_recv: Any
    data_recv_sem: Ref


class BadTilingException(ValueError):
    """The shape cannot be tiled for a one-shot DMA all-to-all."""


def _get_vmem_tiling(
    shape: Sequence[int],
    dtype: Any,
    window_size: int = _LOCAL_COPY_VMEM_WINDOW_BYTES,
    num_chunks: int = 1,
) -> tuple[int, ...]:
    tpu_info = pltpu.get_tpu_info()
    block_shape = list(shape)
    while math.prod(block_shape) * jnp.dtype(dtype).itemsize * num_chunks > window_size:
        candidates = []
        for i, val in enumerate(block_shape):
            if i == len(block_shape) - 1:
                if val % (2 * tpu_info.num_lanes) == 0:
                    candidates.append(i)
            elif i == len(block_shape) - 2:
                if val % (2 * tpu_info.num_sublanes) == 0:
                    candidates.append(i)
            elif val % 2 == 0:
                candidates.append(i)
        if not candidates:
            break
        idx = max(candidates, key=lambda k: block_shape[k])
        block_shape[idx] //= 2
    return tuple(block_shape)


def run_dense_a2a(
    x: jax.Array,
    recv_state: DenseReceiveState,
    *,
    collective_id: int,
    axis_name: str | Sequence[str],
    allow_pin: bool,
    split_axis: int,
    concat_axis: int,
    num_chunks: int,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """Complete the barrier, push ``N - 1`` remote DMAs + local VMEM copy, and wait."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)

    def kernel(
        x_ref: Any,
        recv_state_ref: DenseReceiveState,
        *,
        send_sem: Any,
    ) -> None:
        with jax.named_scope("main_barrier_wait"):
            barrier_sem = pltpu.get_barrier_semaphore()
            pl.semaphore_wait(barrier_sem, axis_size)

        my_id = jnp.uint32(jax.lax.axis_index(axis_key))
        num_packets = axis_size * num_chunks

        with jax.named_scope("send_remote_data"):
            for chunk_id in range(num_chunks):
                for i in range(axis_size):
                    src_slices = [pl.ds(None) for _ in range(len(x_ref.shape))]
                    src_slices[split_axis] = pl.ds(
                        (chunk_id * axis_size + i)
                        * x_ref.shape[split_axis]
                        // num_packets,
                        x_ref.shape[split_axis] // num_packets,
                    )
                    dest_slices = [pl.ds(None) for _ in range(len(x_ref.shape))]
                    dest_slices[concat_axis] = pl.ds(
                        (chunk_id * axis_size + my_id)
                        * (recv_state_ref.x_recv.shape[concat_axis] // num_packets),
                        recv_state_ref.x_recv.shape[concat_axis] // num_packets,
                    )

                    @pl.when(i != my_id)
                    def _(
                        i: int = i,
                        src_slices: list[Any] = src_slices,
                        dest_slices: list[Any] = dest_slices,
                    ) -> None:
                        pltpu.make_async_remote_copy(
                            src_ref=x_ref.at[*src_slices],
                            dst_ref=recv_state_ref.x_recv.at[*dest_slices],
                            send_sem=send_sem,
                            recv_sem=recv_state_ref.data_recv_sem,
                            device_id={axis_key: i},
                        ).start()

        with jax.named_scope("local_copy_via_vmem"):
            all_src_slices = [
                [pl.ds(None) for _ in range(len(x_ref.shape))]
                for _ in range(num_chunks)
            ]
            all_dest_slices = [
                [pl.ds(None) for _ in range(len(x_ref.shape))]
                for _ in range(num_chunks)
            ]
            for chunk_id in range(num_chunks):
                all_src_slices[chunk_id][split_axis] = pl.ds(
                    (chunk_id * axis_size + my_id)
                    * (x_ref.shape[split_axis] // num_packets),
                    x_ref.shape[split_axis] // num_packets,
                )
                all_dest_slices[chunk_id][concat_axis] = pl.ds(
                    (chunk_id * axis_size + my_id)
                    * (recv_state_ref.x_recv.shape[concat_axis] // num_packets),
                    recv_state_ref.x_recv.shape[concat_axis] // num_packets,
                )
            all_local_src = [x_ref.at[*s] for s in all_src_slices]
            all_local_dst = [recv_state_ref.x_recv.at[*d] for d in all_dest_slices]

            def copy_kernel(local_srcs: Any, local_dsts: Any) -> None:
                for s_ref, d_ref in zip(local_srcs, local_dsts, strict=True):
                    d_ref[...] = s_ref[...]

            block_shape = _get_vmem_tiling(
                all_local_src[0].shape,
                x_ref.dtype,
                num_chunks=num_chunks,
            )
            grid = tuple(
                pl.cdiv(s, b)
                for s, b in zip(all_local_src[0].shape, block_shape, strict=True)
            )
            pltpu.emit_pipeline(
                copy_kernel,
                grid=grid,
                in_specs=[
                    [
                        pl.BlockSpec(block_shape, lambda *indices: indices)
                        for _ in range(num_chunks)
                    ]
                ],
                out_specs=[
                    [
                        pl.BlockSpec(block_shape, lambda *indices: indices)
                        for _ in range(num_chunks)
                    ]
                ],
            )(all_local_src, all_local_dst)

        with jax.named_scope("wait_recv_data"):
            chunk = x_ref.shape[split_axis] // axis_size
            wait_slices = [pl.ds(None) for _ in range(len(x_ref.shape))]
            wait_slices[split_axis] = pl.ds(0, chunk * (axis_size - 1))
            wait_ref = x_ref.at[*wait_slices]
            pltpu.make_async_copy(
                src_ref=wait_ref,
                dst_ref=wait_ref,
                sem=recv_state_ref.data_recv_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=wait_ref,
                dst_ref=wait_ref,
                sem=send_sem,
            ).wait()

    return call_kernel(
        kernel,
        args=(x,),
        inout=(recv_state,),
        scoped={"send_sem": pltpu.SemaphoreType.DMA},
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        allow_pin=allow_pin,
        vmem_dense_recv=(recv_memory_space == "vmem"),
    )[0].x_recv


def _tiling_problem(
    x_shape: tuple[int, ...],
    axis_size: int,
    split_axis: int,
    concat_axis: int,
    num_chunks: int,
) -> str | None:
    """Why ``x_shape`` cannot be sent as-is by a one-shot all-to-all, or None."""
    if axis_size == 1:
        return None
    out_shape = list(x_shape)
    out_shape[split_axis] //= axis_size * num_chunks
    out_shape[concat_axis] *= axis_size * num_chunks
    tpu_info = pltpu.get_tpu_info()
    num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
    if x_shape[-1] % num_lanes != 0 or out_shape[-1] % num_lanes != 0:
        return (
            f"lane dim of input {x_shape} or output {tuple(out_shape)} "
            f"is not a multiple of {num_lanes}"
        )
    if len(x_shape) > 1 and min(x_shape[-2], num_sublanes) != min(
        out_shape[-2], num_sublanes
    ):
        return (
            f"sublane dim changes from {x_shape} to {tuple(out_shape)} "
            f"(axis_size={axis_size}, num_chunks={num_chunks})"
        )
    return None


def _check_tiling(
    x_shape: tuple[int, ...],
    axis_size: int,
    split_axis: int,
    concat_axis: int,
    num_chunks: int,
) -> None:
    problem = _tiling_problem(x_shape, axis_size, split_axis, concat_axis, num_chunks)
    if problem is not None:
        raise BadTilingException(problem)


def _unpad_lanes_pallas(
    x: jax.Array,
    *,
    num_pieces: int,
    orig_lanes: int,
    padded_lane_size: int,
) -> jax.Array:
    grid = x.shape[:-2]
    out_lanes = num_pieces * orig_lanes
    in_block_shape = (1,) * len(grid) + (x.shape[-2], x.shape[-1])
    out_block_shape = (1,) * len(grid) + (x.shape[-2], out_lanes)

    def unpad_kernel(x_ref: Any, out_ref: Any) -> None:
        x_val = x_ref.get()
        base_indices = (0,) * len(grid)
        x_val_flat = x_val[base_indices + (slice(None), slice(None))]
        for p in range(num_pieces):
            src_start = p * padded_lane_size
            dst_start = p * orig_lanes
            out_ref[
                base_indices + (slice(None), slice(dst_start, dst_start + orig_lanes))
            ] = x_val_flat[:, src_start : src_start + orig_lanes]

    in_spec = pl.BlockSpec(
        block_shape=in_block_shape,
        index_map=lambda *indices: indices + (0, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=out_block_shape,
        index_map=lambda *indices: indices + (0, 0),
    )
    out_tuple = pl.pallas_call(
        unpad_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[in_spec],
            out_specs=[out_spec],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(shape=x.shape[:-1] + (out_lanes,), dtype=x.dtype)
        ],
        name="unpad_lanes_pallas",
    )(x)
    return out_tuple[0]


def _sublane_packing_params(x: jax.Array) -> tuple[int, int]:
    """``(rows per 32-bit word, sublane tile rows)`` for ``x``'s dtype and rows."""
    dtype_width_bits = jax.dtypes.itemsize_bits(x.dtype)
    max_sublane_packing = pltpu.get_tpu_info().get_sublane_tiling(x.dtype)
    min_sublane_packing = 32 // dtype_width_bits
    next_pow2_sublane = 1 << (x.shape[-2] - 1).bit_length()
    sublane_packing = max(
        min(next_pow2_sublane, max_sublane_packing), min_sublane_packing
    )
    return min_sublane_packing, sublane_packing


def needs_retile(
    x: jax.Array,
    axis_size: int | None = None,
    split_axis: int = 0,
    concat_axis: int = 0,
    num_chunks: int = 1,
) -> bool:
    if len(x.shape) <= 1:
        return False
    _, sublane_packing = _sublane_packing_params(x)
    if x.shape[-2] % sublane_packing != 0:
        return True
    return (
        axis_size is not None
        and _tiling_problem(x.shape, axis_size, split_axis, concat_axis, num_chunks)
        is not None
    )


def can_retile(x: jax.Array) -> bool:
    min_sublane_packing, _ = _sublane_packing_params(x)
    return x.shape[-1] % (pltpu.get_tpu_info().num_lanes * min_sublane_packing) == 0


def retile(x: jax.Array) -> jax.Array:
    min_sublane_packing, _ = _sublane_packing_params(x)
    return x.reshape(
        *x.shape[:-1], min_sublane_packing, x.shape[-1] // min_sublane_packing
    )


def unretile(x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-2], -1)


def seam_pieces_pack(x: jax.Array, axis_size: int) -> jax.Array:
    assert x.ndim == 3 and x.shape[0] == axis_size, (x.shape, axis_size)
    lane_pad = -x.shape[-1] % pltpu.get_tpu_info().num_lanes
    if lane_pad:
        x = jnp.pad(x, [(0, 0), (0, 0), (0, lane_pad)])
    if needs_retile(x, axis_size=axis_size, split_axis=0, concat_axis=0):
        if can_retile(x):
            x = retile(x)
        else:
            _, sublane_packing = _sublane_packing_params(x)
            pad_to = sublane_packing
            pad_amount = (pad_to - (x.shape[-2] % pad_to)) % pad_to
            if pad_amount > 0:
                x = jnp.pad(x, [(0, 0), (0, pad_amount), (0, 0)])
    return x


def seam_pieces_unpack(out: jax.Array, rows: int, orig_lanes: int) -> jax.Array:
    if out.ndim == 4:
        out = unretile(out)
    return out[:, :rows, :orig_lanes]


@functools.partial(
    jax.jit,
    static_argnames=[
        "axis_name",
        "allow_pin",
        "collective_id",
        "split_axis",
        "concat_axis",
        "num_chunks",
        "recv_memory_space",
    ],
)
def _dense_all_to_all(
    x: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
    allow_pin: bool,
    collective_id: int,
    split_axis: int,
    concat_axis: int,
    num_chunks: int,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    if recv_memory_space not in ("hbm", "vmem"):
        raise ValueError(
            f"recv_memory_space must be 'hbm' or 'vmem', got {recv_memory_space!r}"
        )
    axis_size = jax.lax.axis_size(axis_name)
    if axis_size == 1:
        return x

    out_shape = list(x.shape)
    out_shape[split_axis] //= axis_size * num_chunks
    out_shape[concat_axis] *= axis_size * num_chunks
    out_shape_t = tuple(out_shape)

    _check_tiling(x.shape, axis_size, split_axis, concat_axis, num_chunks)

    recv_space = pltpu.VMEM if recv_memory_space == "vmem" else pltpu.HBM
    with xla_metadata.set_xla_metadata(
        async_type="collective-start",
        latency_metadata=_DMA_LATENCY_METADATA,
    ):
        recv_state = barrier_and_allocate(
            DenseReceiveState(
                x_recv=recv_space(shape=out_shape_t, dtype=x.dtype),
                data_recv_sem=pltpu.SemaphoreType.DMA,
            ),
            collective_id=collective_id,
            axis_name=axis_name,
            order_dep=x,
            allow_pin=allow_pin,
        )

    with xla_metadata.set_xla_metadata(
        async_type="collective-done",
        latency_metadata=_DMA_LATENCY_METADATA,
    ):
        return run_dense_a2a(
            x,
            recv_state,
            collective_id=collective_id,
            axis_name=axis_name,
            allow_pin=allow_pin,
            split_axis=split_axis,
            concat_axis=concat_axis,
            num_chunks=num_chunks,
            recv_memory_space=recv_memory_space,
        )


@jax.named_scope("all_to_all_kernel")
def dense_all_to_all(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    allow_pin: bool = True,
    collective_id: int,
    split_axis: int = 0,
    concat_axis: int = 0,
    num_chunks: int = 1,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """One-shot Pallas remote-DMA all-to-all over ``axis_name``."""
    axis_key = normalize_axis_name(axis_name)
    split_axis = split_axis if split_axis >= 0 else split_axis + len(x.shape)
    concat_axis = concat_axis if concat_axis >= 0 else concat_axis + len(x.shape)

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    tile_sublanes = (
        tpu_info.num_sublanes * _sublane_packing_params(x)[0]
        if len(x.shape) == 2
        else 1
    )
    if (
        len(x.shape) == 2
        and split_axis == 0
        and concat_axis == 0
        and (x.shape[0] // (jax.lax.axis_size(axis_key) * num_chunks)) % tile_sublanes
        != 0
    ):
        raise BadTilingException(
            f"2D dense_all_to_all piece rows {x.shape[0]} / "
            f"{jax.lax.axis_size(axis_key) * num_chunks} not tile-aligned "
            f"(needs multiple of {tile_sublanes}). Reshape to 3D first."
        )

    assert x.shape[split_axis] % (jax.lax.axis_size(axis_key) * num_chunks) == 0, (
        f"Split axis {split_axis} of shape {x.shape} not divisible by "
        f"axis_size={jax.lax.axis_size(axis_key)} * num_chunks={num_chunks}"
    )

    retiled = False
    sublane_padded = False
    orig_second_minor = x.shape[-2] if len(x.shape) > 1 else 0
    orig_lanes = x.shape[-1]
    lanes_padded = False

    if len(x.shape) > 1:
        _, sublane_packing = _sublane_packing_params(x)
        if x.shape[-1] % num_lanes != 0 and split_axis < len(x.shape) - 1:
            lanes_pad_amount = num_lanes - (x.shape[-1] % num_lanes)
            pad_widths = [(0, 0)] * len(x.shape)
            pad_widths[-1] = (0, lanes_pad_amount)
            x = jnp.pad(x, pad_widths)
            lanes_padded = True

        retiled = needs_retile(
            x,
            axis_size=jax.lax.axis_size(axis_key),
            split_axis=split_axis,
            concat_axis=concat_axis,
            num_chunks=num_chunks,
        )
        if retiled:
            if (
                split_axis < len(x.shape) - 1
                and concat_axis < len(x.shape) - 1
                and can_retile(x)
            ):
                x = retile(x)
            else:
                sublane_padded = True
                retiled = False
                pad_to = sublane_packing
                pad_amount = (pad_to - (x.shape[-2] % pad_to)) % pad_to
                if pad_amount > 0:
                    pad_widths = [(0, 0)] * len(x.shape)
                    pad_widths[-2] = (0, pad_amount)
                    x = jnp.pad(x, pad_widths)

    @jax.custom_gradient
    def _fwd_and_grad(inp: jax.Array):
        return (
            _dense_all_to_all(
                inp,
                axis_name=axis_key,
                allow_pin=allow_pin,
                recv_memory_space=recv_memory_space,
                collective_id=collective_id,
                split_axis=split_axis,
                concat_axis=concat_axis,
                num_chunks=num_chunks,
            ),
            functools.partial(
                dense_all_to_all,
                axis_name=axis_key,
                allow_pin=allow_pin,
                recv_memory_space=recv_memory_space,
                collective_id=collective_id,
                split_axis=concat_axis,
                concat_axis=split_axis,
                num_chunks=num_chunks,
            ),
        )

    out: jax.Array = _fwd_and_grad(x)

    if retiled:
        out = unretile(out)
    if sublane_padded:
        slices = [slice(None)] * len(out.shape)
        slices[-2] = slice(0, orig_second_minor)
        out = out[tuple(slices)]
    if lanes_padded:
        if concat_axis == len(x.shape) - 1 or (
            retiled and concat_axis == len(x.shape) - 2
        ):
            padded_lane_size = orig_lanes + (num_lanes - orig_lanes % num_lanes)
            out = _unpad_lanes_pallas(
                out,
                num_pieces=out.shape[-1] // padded_lane_size,
                orig_lanes=orig_lanes,
                padded_lane_size=padded_lane_size,
            )
        else:
            out = out[..., :orig_lanes]
    return out


# =============================================================================
# 1-Shot Latency Collectives & JAX-Internal Width-Guarded Wrappers
# =============================================================================


def _resolve_recv_memory_space(
    recv_memory_space: str,
    rows: int,
    lanes: int,
    axis_size: int,
    dtype: Any,
    *,
    allow_vmem_auto: bool = False,
    vmem_max_tokens: int = DEFAULT_VMEM_MAX_TOKENS,
) -> str:
    if recv_memory_space in ("hbm", "vmem"):
        return recv_memory_space
    if recv_memory_space != "auto":
        raise ValueError(
            "recv_memory_space must be 'auto', 'hbm', or 'vmem', "
            f"got {recv_memory_space!r}"
        )
    tpu_info = pltpu.get_tpu_info()
    if not allow_vmem_auto or lanes % tpu_info.num_lanes != 0:
        return "hbm"
    recv_rows = max(rows, tpu_info.num_sublanes)
    recv_bytes = axis_size * recv_rows * lanes * jnp.dtype(dtype).itemsize
    if rows <= vmem_max_tokens and recv_bytes <= _VMEM_RECV_MAX_BYTES:
        return "vmem"
    return "hbm"


@jax.named_scope("latency_all_gather_cols")
def latency_all_gather_cols(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """1-shot Pallas remote-DMA trailing-dim gather: ``[rows, D] -> [rows, Ax * D]``."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    if axis_size == 1:
        return x
    x_3d = jax.lax.broadcast(x, [axis_size])
    gathered = dense_all_to_all(
        x_3d,
        axis_key,
        split_axis=0,
        concat_axis=0,
        num_chunks=1,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
    )
    # gathered has shape [Ax, rows, D]; transpose to [rows, Ax, D] -> [rows, Ax * D].
    return jnp.swapaxes(gathered, 0, 1).reshape(x.shape[0], axis_size * x.shape[1])


@jax.named_scope("latency_psum")
def latency_psum(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """Deterministic 1-shot Pallas remote-DMA all-reduce over ``axis_name``."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    if axis_size == 1:
        return x
    x_3d = jax.lax.broadcast(x, [axis_size])
    gathered = dense_all_to_all(
        x_3d,
        axis_key,
        split_axis=0,
        concat_axis=0,
        num_chunks=1,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
    )
    return gathered.astype(jnp.float32).sum(axis=0).astype(x.dtype)


@jax.named_scope("latency_psum_scatter_rows")
def latency_psum_scatter_rows(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """Deterministic one-shot DMA row reduce-scatter.

    ``[axis_size * rows, ...] -> [rows, ...]``, summed in float32.
    """
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    if axis_size == 1:
        return x
    assert x.shape[0] % axis_size == 0, (x.shape, axis_size)
    rows_out = x.shape[0] // axis_size
    x_3d = x.reshape(axis_size, rows_out, *x.shape[1:])
    received = dense_all_to_all(
        x_3d,
        axis_key,
        split_axis=0,
        concat_axis=0,
        num_chunks=1,
        collective_id=collective_id,
        recv_memory_space=recv_memory_space,
    )
    return received.astype(jnp.float32).sum(axis=0).astype(x.dtype)


def _use_one_shot(rows: int, total_rows: int, max_dma_tokens: int) -> bool:
    """Whether a guarded collective takes the one-shot DMA path.

    ``rows`` is the per-shard row count compared with ``max_dma_tokens``;
    ``total_rows`` is the row count summed over the axis.
    """
    return rows <= max_dma_tokens and total_rows <= _ONE_SHOT_MAX_TOTAL_ROWS


def guarded_all_gather_cols(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> jax.Array:
    """Column all-gather: one-shot DMA for ``<= max_dma_tokens`` rows, else lax."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows = x.shape[0]
    if _use_one_shot(rows, rows * axis_size, max_dma_tokens):
        rmem = _resolve_recv_memory_space(
            recv_memory_space, rows, x.shape[-1], axis_size, x.dtype
        )
        return latency_all_gather_cols(
            x,
            axis_key,
            collective_id=collective_id,
            recv_memory_space=rmem,
        )
    return jax.lax.all_gather(x, axis_key, axis=1, tiled=True)


def guarded_psum(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    vmem_max_tokens: int = DEFAULT_VMEM_MAX_TOKENS,
) -> jax.Array:
    """All-reduce: one-shot DMA for ``<= max_dma_tokens`` rows, else lax."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows = x.shape[0]
    if _use_one_shot(rows, rows * axis_size, max_dma_tokens):
        rmem = _resolve_recv_memory_space(
            recv_memory_space,
            rows,
            x.shape[-1],
            axis_size,
            x.dtype,
            allow_vmem_auto=True,
            vmem_max_tokens=vmem_max_tokens,
        )
        return latency_psum(
            x,
            axis_key,
            collective_id=collective_id,
            recv_memory_space=rmem,
        )
    return jax.lax.psum(x, axis_key)


def guarded_psum_scatter_rows(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    vmem_max_tokens: int = DEFAULT_VMEM_MAX_TOKENS,
) -> jax.Array:
    """Row reduce-scatter: one-shot DMA for ``<= max_dma_tokens`` output rows."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows_out = x.shape[0] // axis_size
    if _use_one_shot(rows_out, x.shape[0], max_dma_tokens):
        rmem = _resolve_recv_memory_space(
            recv_memory_space,
            rows_out,
            x.shape[-1],
            axis_size,
            x.dtype,
            allow_vmem_auto=True,
            vmem_max_tokens=vmem_max_tokens,
        )
        return latency_psum_scatter_rows(
            x,
            axis_key,
            collective_id=collective_id,
            recv_memory_space=rmem,
        )
    return jax.lax.psum_scatter(x, axis_key, scatter_dimension=0, tiled=True)
