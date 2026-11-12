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
"""Pallas 1-shot remote-DMA primitives and ``@jax_op`` wrappers for Kimi K3.

Ported from Google-internal GemAX / Kimi K3 Pallas transport modules
(``kiloblox_a2a.py``, ``dense_a2a.py``, and ``latency_collectives.py``) with
zero external dependencies:

1. ``barrier_and_allocate``, ``_color_dma_operands`` and ``call_kernel``
   preserve birth memory spaces (``pltpu.HBM`` / ``pltpu.VMEM``)
   across kernel boundaries so XLA's memory-space pass never relocates buffers
   targeted by in-flight remote DMAs.
2. ``dense_all_to_all`` performs 1-shot point-to-point ``make_async_remote_copy``
   transfers to ``N - 1`` peers in parallel with a local VMEM bounce copy,
   automatically padding lane dimensions to a multiple of ``128`` (e.g.
   ``224 -> 256``, ``3632 -> 3712``) and retiling or padding sublane dims.
3. ``latency_all_gather_rows``, ``latency_all_gather_cols``, ``latency_psum``,
   and ``latency_psum_scatter_rows`` wrap ``dense_all_to_all`` with 3D
   ``(axis_size, rows, lanes)`` ``split_axis=0, concat_axis=0`` geometry so
   arbitrary small decode batch sizes (including ``B=4`` on odd-128 widths like
   ``3632 -> 3712``) never trigger sublane tiling mismatches.
4. Pre-built ``@jax_op`` + ``register_fake`` factories expose 2D
   ``('dp', 'tp')`` ``jax.shard_map`` collectives to PyTorch / ``torch.compile``
   with JAX-internal ``T_padded <= max_dma_tokens`` width guards (falling back
   inside JAX to ``lax.all_gather`` / ``lax.psum_scatter`` / ``lax.psum`` for
   chunked prefill buckets ``> 64``) and explicit variant tags in every
   ``pallas::...`` custom-op name (``D87`` warm-cache safety).
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

# Axis names of the rank-ordered 2D ('dp', 'tp') mesh the DMA collectives run on.
DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"

# =============================================================================
# Unified Non-Overlapping Kimi K3 `collective_id` Barrier Registry (0..6)
# =============================================================================
COLLECTIVE_ID_MOE_TP_LATENT_AG = 0       # 'tp': KimiMoE routed_down_local [B_dp, 224->256]
COLLECTIVE_ID_MOE_DP_DISPATCH_AG = 1     # 'dp' / ('dp','tp'): pack_dp_dispatch [B_dp, 3632->3712]
COLLECTIVE_ID_MOE_DP_COMBINE_RS = 2      # 'dp': post-MoE combine psum_scatter_rows [B_glob, 3584]
COLLECTIVE_ID_MOE_TP_ROUTED_PSUM = 3     # 'tp': KimiMoE routed_out psum [B_dp, 3584]
COLLECTIVE_ID_ATTN_MLP_TP_EXIT_PSUM = 4  # 'tp': KDA/MLA o_proj + L0 KimiMLP.down_proj [B_dp, 7168]
COLLECTIVE_ID_MOE_TP_EXIT_PSUM = 5       # 'tp': KimiMoE packed_moe_out [B_dp, 7168]
COLLECTIVE_ID_MOE_OWNER_SEAM_SPLIT = 6   # ('dp','tp') + 'tp': O4-A owner combine + seam_exit_split

DENSE_COLLECTIVE_ID = COLLECTIVE_ID_MOE_TP_LATENT_AG
DENSE_TRANSPOSE_COLLECTIVE_ID = COLLECTIVE_ID_MOE_DP_COMBINE_RS

DEFAULT_MAX_DMA_TOKENS = 64
DEFAULT_VMEM_MAX_TOKENS = 8


def normalize_axis_name(axis_name: str | Sequence[str]) -> str | tuple[str, ...]:
    """Return a hashable axis identifier suitable for ``lax`` and Pallas ``device_id``."""
    if isinstance(axis_name, str):
        return axis_name
    axes = tuple(axis_name)
    if not axes:
        raise ValueError("axis_name must not be empty")
    return axes[0] if len(axes) == 1 else axes


def axis_tuple(axis_name: str | Sequence[str]) -> tuple[str, ...]:
    return (axis_name,) if isinstance(axis_name, str) else tuple(axis_name)


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
        is_dense_recv = any(
            getattr(entry, "name", None) == "x_recv" for entry in path
        )
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
    *extra_pos: Any,
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
    assert not extra_pos, "Pass operands via `args=...`"
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
                    if block_spec_from_array(x).memory_space
                    == pltpu.MemorySpace.SMEM
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
        dep_space = (
            pltpu.MemorySpace.VMEM if order_dep_vmem else pltpu.MemorySpace.HBM
        )
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


class BadTilingException(Exception):
    pass


def _get_vmem_tiling(
    shape: Sequence[int],
    dtype: Any,
    window_size: int = 3 * 1024 * 1024,
    num_chunks: int = 1,
) -> tuple[int, ...]:
    block_shape = list(shape)
    while (
        math.prod(block_shape) * jnp.dtype(dtype).itemsize * num_chunks
        > window_size
    ):
        candidates = []
        for i, val in enumerate(block_shape):
            if i == len(block_shape) - 1:
                if val % 256 == 0:
                    candidates.append(i)
            elif i == len(block_shape) - 2:
                if val % 16 == 0:
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
                    def _() -> None:
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


def _check_tiling(
    x_shape: tuple[int, ...],
    axis_size: int,
    split_axis: int,
    concat_axis: int,
    num_chunks: int,
) -> None:
    if axis_size == 1:
        return

    def _assert(cond: bool, msg: Any) -> None:
        if not cond:
            raise BadTilingException(msg)

    out_shape = list(x_shape)
    out_shape[split_axis] //= axis_size * num_chunks
    out_shape[concat_axis] *= axis_size * num_chunks
    out_shape_t = tuple(out_shape)

    _assert(x_shape[-1] % 128 == 0, (x_shape, out_shape_t))
    _assert(out_shape_t[-1] % 128 == 0, (x_shape, out_shape_t))
    if len(x_shape) > 1:
        _assert(
            min(x_shape[-2], 8) == min(out_shape_t[-2], 8),
            (x_shape, out_shape_t, axis_size, num_chunks),
        )


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
                base_indices
                + (slice(None), slice(dst_start, dst_start + orig_lanes))
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
    if x.dtype == jnp.dtype("int4"):
        dtype_width_bits = 4
    else:
        dtype_width_bits = x.dtype.itemsize * 8

    if dtype_width_bits == 4:
        max_sublane_packing = 64
    elif dtype_width_bits == 8:
        max_sublane_packing = 32
    else:
        max_sublane_packing = 8

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
    if axis_size is not None:
        try:
            _check_tiling(x.shape, axis_size, split_axis, concat_axis, num_chunks)
        except BadTilingException:
            return True
    return False


def can_retile(x: jax.Array) -> bool:
    min_sublane_packing, _ = _sublane_packing_params(x)
    return x.shape[-1] % (128 * min_sublane_packing) == 0


def retile(x: jax.Array) -> jax.Array:
    min_sublane_packing, _ = _sublane_packing_params(x)
    return x.reshape(
        *x.shape[:-1], min_sublane_packing, x.shape[-1] // min_sublane_packing
    )


def unretile(x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-2], -1)


def seam_pieces_pack(x: jax.Array, axis_size: int) -> jax.Array:
    assert x.ndim == 3 and x.shape[0] == axis_size, (x.shape, axis_size)
    if x.shape[-1] % 128 != 0:
        x = jnp.pad(x, [(0, 0), (0, 0), (0, 128 - (x.shape[-1] % 128))])
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


def seam_pieces_tile(
    axis_size: int, rows: int, lanes: int, dtype: Any
) -> jax.ShapeDtypeStruct:
    """Return the static packed shape produced by ``seam_pieces_pack``."""
    packed = jax.eval_shape(
        functools.partial(seam_pieces_pack, axis_size=axis_size),
        jax.ShapeDtypeStruct((axis_size, rows, lanes), dtype),
    )
    return jax.ShapeDtypeStruct(packed.shape, packed.dtype)


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
        latency_metadata="300000",
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
        latency_metadata="300000",
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
    collective_id: int = DENSE_COLLECTIVE_ID,
    split_axis: int = 0,
    concat_axis: int = 0,
    num_chunks: int = 1,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """One-shot Pallas remote-DMA all-to-all over ``axis_name``."""
    axis_key = normalize_axis_name(axis_name)
    split_axis = split_axis if split_axis >= 0 else split_axis + len(x.shape)
    concat_axis = concat_axis if concat_axis >= 0 else concat_axis + len(x.shape)

    tile_sublanes = 8 * _sublane_packing_params(x)[0] if len(x.shape) == 2 else 1
    if (
        len(x.shape) == 2
        and split_axis == 0
        and concat_axis == 0
        and (x.shape[0] // (jax.lax.axis_size(axis_key) * num_chunks))
        % tile_sublanes
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
        if x.shape[-1] % 128 != 0 and split_axis < len(x.shape) - 1:
            lanes_pad_amount = 128 - (x.shape[-1] % 128)
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
            padded_lane_size = orig_lanes + (128 - orig_lanes % 128)
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
            f"recv_memory_space must be 'auto', 'hbm', or 'vmem', got {recv_memory_space!r}"
        )
    if not allow_vmem_auto or lanes % 128 != 0:
        return "hbm"
    recv_bytes = axis_size * max(rows, 8) * lanes * jnp.dtype(dtype).itemsize
    if rows <= vmem_max_tokens and recv_bytes <= 2 * 1024 * 1024:
        return "vmem"
    return "hbm"


@jax.named_scope("latency_all_gather_rows")
def latency_all_gather_rows(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int = DENSE_COLLECTIVE_ID,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """1-shot Pallas remote-DMA row all-gather: ``[rows, ...] -> [Ax * rows, ...]``."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    if axis_size == 1:
        return x
    # Broadcast [rows, ...] -> [Ax, rows, ...] and run split_axis=0, concat_axis=0
    # so x.shape[-2] == out.shape[-2] == rows for every small batch size (incl. B=4
    # with odd-128 widths such as 3632 -> 3712).
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
    return gathered.reshape(axis_size * x.shape[0], *x.shape[1:])


@jax.named_scope("latency_all_gather_cols")
def latency_all_gather_cols(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int = DENSE_COLLECTIVE_ID,
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
    collective_id: int = DENSE_COLLECTIVE_ID,
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
    collective_id: int = DENSE_TRANSPOSE_COLLECTIVE_ID,
    recv_memory_space: str = "hbm",
) -> jax.Array:
    """Deterministic 1-shot Pallas remote-DMA row reduce-scatter: ``[Ax*rows, ...] -> [rows, ...]``."""
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


def kimi_guarded_all_gather_rows(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> jax.Array:
    """JAX-internal width-guarded row all-gather (1-shot Pallas for ``<= max_dma_tokens``)."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows = x.shape[0]
    if rows <= max_dma_tokens and rows * axis_size <= 1024:
        rmem = _resolve_recv_memory_space(
            recv_memory_space, rows, x.shape[-1], axis_size, x.dtype
        )
        return latency_all_gather_rows(
            x,
            axis_key,
            collective_id=collective_id,
            recv_memory_space=rmem,
        )
    return jax.lax.all_gather(x, axis_key, axis=0, tiled=True)


def kimi_guarded_all_gather_cols(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
) -> jax.Array:
    """JAX-internal width-guarded column all-gather (1-shot Pallas for ``<= max_dma_tokens``)."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows = x.shape[0]
    if rows <= max_dma_tokens and rows * axis_size <= 1024:
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


def kimi_guarded_psum(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    vmem_max_tokens: int = DEFAULT_VMEM_MAX_TOKENS,
) -> jax.Array:
    """JAX-internal width-guarded all-reduce (1-shot Pallas for ``<= max_dma_tokens``)."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows = x.shape[0]
    if rows <= max_dma_tokens and rows * axis_size <= 1024:
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


def kimi_guarded_psum_scatter_rows(
    x: jax.Array,
    axis_name: str | Sequence[str],
    *,
    collective_id: int,
    recv_memory_space: str = "auto",
    max_dma_tokens: int = DEFAULT_MAX_DMA_TOKENS,
    vmem_max_tokens: int = DEFAULT_VMEM_MAX_TOKENS,
) -> jax.Array:
    """JAX-internal width-guarded row reduce-scatter (1-shot Pallas for ``<= max_dma_tokens``)."""
    axis_key = normalize_axis_name(axis_name)
    axis_size = jax.lax.axis_size(axis_key)
    rows_out = x.shape[0] // axis_size
    if rows_out <= max_dma_tokens and x.shape[0] <= 1024:
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
