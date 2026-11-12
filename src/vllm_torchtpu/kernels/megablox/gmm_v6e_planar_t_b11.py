# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped matmul over MXFP4 expert stacks, decoded inside the kernel.

MXFP4 expert weights (uint8 nibble pairs plus one e8m0 exponent per 32
contraction elements) stream HBM->VMEM as they are and each tile is
decoded in VMEM right before the MXU consumes it, so no dense bf16 copy
exists and weight read traffic drops about 4x.

Layout: at load time the weights are repacked into the planar_t layout
(`repack_w13_planar_t` / `repack_w2_planar_t`). Within each
PLANAR_WINDOW (512) element contraction window, byte j holds elements j
and j + 256, and the stack is transposed to `[E, K/2, N]`; the e8m0
scales are transposed to `[E, K/32, N]`.

Decode (`planar_t_b11`): each e2m1 nibble is relabelled to its exactly
equal float8_e4m3b11fnuz byte with the group's offset from the window's
largest exponent folded into the byte, the codes meet bf16 activations on
the MXU, and the window's largest scale is applied once to the f32
accumulator. This is exact as long as every window's e8m0 spread is at
most B11_FOLD_WINDOW octaves (`planar_t_b11_window_ok`).

Entry points: `moe_gmm_v6e_planar_t_b11` (sorted routed tokens) and
`fused_decode_moe_v6e_planar_t_b11` (one kernel for small batches).

The grid metadata is adapted from `make_group_metadata` in
`jax.experimental.pallas.ops.tpu.megablox.gmm` (JAX, Apache-2.0).
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Nibble/scale stacks are handled on device (jax arrays and tracers) and on
# the host (numpy) by the load-time layout helpers.
_NibbleArray = jax.Array | np.ndarray

# e2m1 decode table, index = sign<<3 | exp<<1 | mantissa.
_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# One e8m0 exponent per this many contraction elements (the MX block size).
SCALE_GROUP = 32

# The contraction window the PLANAR weight layout is packed per (see
# `repack_planar_stack`). Fixed at the k chunk `default_tiling` picks
# for every real serving shape (k = 3584 and 3072 both fit chunk 512 at
# every tm/tn/rhs_pairs combination the model runs, under BOTH chips'
# budgets and scoped-stack calibrations), because a planar kernel's k
# chunk MUST equal the repack window: each chunk step's nibble planes
# are self-contained only when the chunk IS a window.
PLANAR_WINDOW = 512

# TPU lane count: a Pallas block's minor dim must be a multiple of it
# (or the whole array dim).
_LANES = 128

# Per grid step VMEM to stay under. Half of a 128 MiB budget, leaving
# the rest for Mosaic's own scratch and any spill the scheduler needs.
_VMEM_BUDGET = 64 * 1024 * 1024

# Mosaic's default per-kernel stack ("scoped vmem") limit, from
# --xla_tpu_scoped_vmem_limit_kib=32768. A kernel whose live
# temporaries exceed it fails the whole compile with
# CompileTimeScopedVmemOom. Raising it is a PROGRAM-WIDE flag flip
# that takes the same bytes away from MSA for every other op in the
# serving step, so the tiling search fits under the default instead.
_SCOPED_VMEM_LIMIT = 32 * 1024 * 1024


def _vmem_bytes(
    tm: int,
    k: int,
    tn: int,
    chunk: int,
    lhs_bytes: int,
    rhs_pairs: int = 1,
) -> int:
    """VMEM a single grid step needs, the terms that actually scale.

    Pallas double-buffers every block, the accumulator is a f32 scratch,
    and the dequant holds an i32 code plane plus f32 value and scale
    planes for the chunk in flight. The fused gate+up kernel streams TWO
    `(packed, scales)` pairs against one lhs block (`rhs_pairs=2`), so
    the rhs blocks, the accumulators and the dequant working set double
    while the lhs and out terms do not.
    """
    packed_rows, scale_rows = k // 2, k // SCALE_GROUP
    blocks = 2 * (
        tm * k * lhs_bytes  # lhs, whole contraction
        + rhs_pairs * tn * packed_rows  # packed u8 nibbles
        + rhs_pairs * tn * scale_rows  # scales u8
        + tm * tn * 2  # bf16 out
    )
    accumulator = rhs_pairs * tm * tn * 4
    # Codes i32, values f32, scaled f32 per pair: the fused kernel
    # interleaves the two dequants inside one chunk iteration, so count
    # both in flight (Mosaic may overlap them).
    dequant = rhs_pairs * tn * chunk * (4 + 4 + 4)
    # The rolled kernel's `[k/chunk, tn, chunk/32]` u8 scratch per pair
    # (see `mxfp4_gmm`), whose sub-lane minor pads out to a full lane of
    # VMEM.
    scale_codes = rhs_pairs * max(k // chunk, 1) * tn * _LANES
    return blocks + accumulator + dequant + scale_codes


# Bytes the Mosaic register allocator keeps live per `[tn, chunk]` element
# of the dequant chain, calibrated against measured compiler output rather
# than derived:
#   * TPU v6e (`SCOPED_ELEM_BYTES_V6E`, the default): a two-pair
#     (512, 512, 512) tile reported a 33.01M scoped allocation against the
#     32M limit, i.e. ~56 bytes per element (~14 live 4-byte planes);
#     rounded up so the model rejects everything the allocator would.
#   * TPU7x (`SCOPED_ELEM_BYTES_TPU7X`): its Mosaic pipeline lowers the same
#     chain thinner, measured at <= ~26.6 bytes per element; rounded up.
# Both were measured on a wider dequant chain than the b11 fold.
# `moe_gmm_v6e_planar_t_b11` passes the TPU7x value; that choice has not
# been re-measured for the b11 chain on v6e.
SCOPED_ELEM_BYTES_V6E = 56
SCOPED_ELEM_BYTES_TPU7X = 28


def _scoped_vmem_bytes(
    tm: int, tn: int, chunk: int, rhs_pairs: int, scoped_elem_bytes: int
) -> int:
    """Kernel stack bytes Mosaic allocates, against `_SCOPED_VMEM_LIMIT`.

    A model of the register allocator's live set: the dominant term is the
    per-pair dequant chain at `scoped_elem_bytes` per `[tn, chunk]` element
    (per-chip calibration above). The accumulators are VMEM scratch, which
    shares the same budget, and the fused epilogue holds a few f32 SiTU
    planes on the `[tm, tn]` tile.
    """
    dequant = rhs_pairs * tn * chunk * scoped_elem_bytes
    accumulators = rhs_pairs * tm * tn * 4
    situ = 3 * tm * tn * 4 if rhs_pairs > 1 else 0
    return dequant + accumulators + situ


# The smallest row tile the expected-rows heuristic will pick. Below
# this the window DMA is latency-bound anyway, and a few MXU sublane
# blocks of headroom keep mild routing skew from spilling into a
# second tile immediately.
_MIN_EXPECTED_TM = 32


def default_tiling(
    m: int,
    k: int,
    n: int,
    lhs_bytes: int = 2,
    rhs_pairs: int = 1,
    expected_rows: int | None = None,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_V6E,
    force_chunk: int | None = None,
) -> tuple[int, int, int]:
    """The largest valid `(tm, k_chunk, tn)` that fits the VMEM budget.

    Chosen explicitly rather than left to chance: an oversized choice does
    not degrade, it fails the whole compile late, in Mosaic.

    Validity, in the order the search relaxes it: `tm` divides m and is
    8-divisible (or is all of m), `tn` is lane-divisible (or all of n), and
    `chunk` divides k and is scale-group-divisible.

    `expected_rows` opts into the sparse expert-parallel regime: each EP
    shard sorts all M assignment rows but owns only about M/shards of them,
    so a large default row tile would make every group visit move a mostly
    padding lhs block and output tile. When set, `tm` becomes the smallest
    candidate covering the expected rows (floored at `_MIN_EXPECTED_TM`).
    Groups larger than expected (routing skew) stay correct; they just take
    more visits, each re-decoding its weight tile.

    Args:
      m: lhs rows.
      k: contraction.
      n: output columns.
      lhs_bytes: activation element size.
      rhs_pairs: how many `(packed, scales)` stacks stream against the lhs (1 for
        the plain kernel, 2 for the fused gate+up kernel).
      expected_rows: rows this caller expects to actually own (the EP shard's
        share of M), or None for the dense default.
      vmem_budget: per-grid-step VMEM bound in bytes, or None for `_VMEM_BUDGET`.
      scoped_elem_bytes: per-chip scoped-stack bytes per dequant element
        (`SCOPED_ELEM_BYTES_V6E` / `SCOPED_ELEM_BYTES_TPU7X`).
      force_chunk: pin the k chunk to this value when it is a valid candidate
        (planar weights require chunk == PLANAR_WINDOW). When the shape cannot
        serve it the normal search runs instead and `can_use_kernel` rejects the
        result, so this never errors on toy shapes.

    Returns:
      `(tm, k_chunk, tn)`.

    Raises:
      ValueError: if even the smallest valid tiling exceeds the budget.
    """
    budget = _VMEM_BUDGET if vmem_budget is None else vmem_budget
    tms = [t for t in (512, 256, 128, 64, 32, 16, 8) if m % t == 0] or [m]
    tns = [t for t in (512, 256, 128) if n % t == 0] or [n]
    chunks = [c for c in (512, 256, 128, 64, SCALE_GROUP) if c <= k and k % c == 0] or [
        k
    ]
    if force_chunk is not None and force_chunk in chunks:
        chunks = [force_chunk]
    if expected_rows is not None and expected_rows < tms[0]:
        fitting = [t for t in tms if t >= max(expected_rows, _MIN_EXPECTED_TM)]
        if fitting:
            pick = min(fitting)
            # Keep the smaller candidates as budget fallbacks (they only
            # ever need LESS VMEM, so in practice the pick wins).
            tms = [pick] + [t for t in tms if t < pick]
    for tm in tms:
        for tn in tns:
            for chunk in chunks:
                if (
                    _vmem_bytes(tm, k, tn, chunk, lhs_bytes, rhs_pairs) <= budget
                    and _scoped_vmem_bytes(tm, tn, chunk, rhs_pairs, scoped_elem_bytes)
                    <= _SCOPED_VMEM_LIMIT
                ):
                    return (tm, chunk, tn)
    smallest = _vmem_bytes(tms[-1], k, tns[-1], chunks[-1], lhs_bytes, rhs_pairs)
    smallest_scoped = _scoped_vmem_bytes(
        tms[-1], tns[-1], chunks[-1], rhs_pairs, scoped_elem_bytes
    )
    raise ValueError(
        f"no valid tiling for (m, k, n) = ({m}, {k}, {n}) fits"
        f" {budget} bytes of VMEM (or {_SCOPED_VMEM_LIMIT} scoped); the"
        f" smallest candidate needs {smallest} / {smallest_scoped}."
    )


# `_make_group_metadata` builds. Only the loop-free closed form is kept:
# no while loop, no scatter and no gather, so it is cheap to build once per
# layer and share between the gate+up and down kernels.
METADATA_IMPLS = ("closed_form",)


def _closed_form_group_metadata(
    group_sizes: jax.Array,
    *,
    m: int,
    tm: int,
    num_groups: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """`_make_group_metadata` as compare-and-sum arithmetic, loop free.

    The megablox build relies on three ragged primitives: `jnp.histogram`
    (a binary-search while loop) and two `jnp.repeat` chains (scatter,
    cumsum, gather). Each of them is a prefix count over a vector of at most
    `num_groups` or `tiles_m` entries, and a prefix count is one compare
    against an iota plus one reduce, which XLA fuses:

      * `group_ends[g]` = sum of sizes over g' <= g (the cumsum);
      * `repeat(arange(n), r, total)[j]` = min(#{i : cumsum(r)[i] <= j},
        n - 1) (past sum(r) the entry is the last source index);
      * `histogram(partial_tile_ids)[t] + 1`, accumulated to the visit
        prefix sum, = (t + 1) + #{g : partial_tile_ids[g] <= t} (the parked
        value `tiles_m` is above every real tile, so it drops out exactly as
        the histogram's out-of-range rule drops it).

    Integer in, integer out, the same values as the megablox build on every
    entry including the grid padding.

    Args:
      group_sizes: `[num_groups]` int32 rows per group.
      m: total rows in the lhs.
      tm: m-dimension tile size.
      num_groups: number of groups.

    Returns:
      As `_make_group_metadata`.
    """
    sizes = group_sizes.astype(jnp.int32)
    tiles_m = m // tm
    grid_len = tiles_m + num_groups - 1
    gid = jnp.arange(num_groups, dtype=jnp.int32)
    # [G, G]: column g' belongs to row g's inclusive prefix.
    in_prefix = gid[None, :] <= gid[:, None]
    group_ends = jnp.sum(jnp.where(in_prefix, sizes[None, :], 0), axis=1)
    group_starts = group_ends - sizes
    group_offsets = jnp.concatenate([jnp.zeros((1,), jnp.int32), group_ends])
    # Round each group's span out to whole tiles (empty groups own none).
    rounded_ends = (group_ends + tm - 1) // tm * tm
    rounded_starts = group_starts // tm * tm
    group_tiles = jnp.where(sizes == 0, 0, (rounded_ends - rounded_starts) // tm)
    tiles_through = jnp.sum(jnp.where(in_prefix, group_tiles[None, :], 0), axis=1)
    grid = jnp.arange(grid_len, dtype=jnp.int32)
    group_ids = jnp.minimum(
        jnp.sum((tiles_through[None, :] <= grid[:, None]).astype(jnp.int32), 1),
        num_groups - 1,
    )
    # A tile is visited once by its owner plus once per group STARTING
    # inside it; tile-aligned (or empty) starts park at tiles_m.
    parked = jnp.logical_or(group_starts % tm == 0, sizes == 0)
    partial_tile_ids = jnp.where(parked, tiles_m, group_starts // tm)
    tiles = jnp.arange(tiles_m, dtype=jnp.int32)
    visits_through = (
        tiles
        + 1
        + jnp.sum((partial_tile_ids[None, :] <= tiles[:, None]).astype(jnp.int32), 1)
    )
    m_tile_ids = jnp.minimum(
        jnp.sum((visits_through[None, :] <= grid[:, None]).astype(jnp.int32), 1),
        tiles_m - 1,
    )
    return (group_offsets, group_ids, m_tile_ids), tiles_through[-1]


def _make_group_metadata(
    group_sizes: jax.Array,
    m: int,
    tm: int,
    num_groups: int,
    impl: str = "closed_form",
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Grid metadata for the grouped matmul.

    Adapted from `make_group_metadata` in
    `jax.experimental.pallas.ops.tpu.megablox.gmm` (JAX, Apache-2.0),
    specialized to a single unsharded group range and
    `visit_empty_groups=False`, which is all this caller needs.

    Args:
      group_sizes: `[num_groups]` int32 rows per group.
      m: total rows in the lhs.
      tm: m-dimension tile size.
      num_groups: number of groups.
      impl: one of `METADATA_IMPLS`.

    Returns:
      `((group_offsets, group_ids, m_tile_ids), num_active_tiles)` where
      `group_offsets[i]` is the first row of group i, and `group_ids[j]` /
      `m_tile_ids[j]` say which group and which m tile grid step j works
      on. A tile straddling a group boundary appears once per group it
      touches, which is why the grid is `tiles_m + num_groups - 1` long.
    """
    if impl not in METADATA_IMPLS:
        raise ValueError(f"unknown metadata impl {impl!r} (want {METADATA_IMPLS}).")
    return _closed_form_group_metadata(group_sizes, m=m, tm=tm, num_groups=num_groups)


#: A precomputed `_make_group_metadata` result, as the kernels take it.
GroupMetadata = tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]


def _check_group_metadata(
    group_metadata: GroupMetadata, *, m: int, tm: int, num_groups: int
) -> GroupMetadata:
    """Refuses a precomputed build made for a different `(m, tm, num_groups)`.

    The grid length `m // tm + num_groups - 1` identifies the row tile
    (distinct dividing tiles give distinct lengths), so a triple built
    for another tiling cannot pass.
    """
    (group_offsets, group_ids, m_tile_ids), num_active_tiles = group_metadata
    grid_len = m // tm + num_groups - 1
    want = ((num_groups + 1,), (grid_len,), (grid_len,), ())
    got = (
        tuple(group_offsets.shape),
        tuple(group_ids.shape),
        tuple(m_tile_ids.shape),
        tuple(jnp.shape(num_active_tiles)),
    )
    if got != want:
        raise ValueError(
            "group_metadata was built for a different tiling: shapes"
            f" {got}, this kernel (m={m}, tm={tm}, num_groups={num_groups})"
            f" wants {want}. Build it with shared_group_metadata over the"
            " tilings of every consuming kernel."
        )
    for name, arr in (
        ("group_offsets", group_offsets),
        ("group_ids", group_ids),
        ("m_tile_ids", m_tile_ids),
        ("num_active_tiles", num_active_tiles),
    ):
        if jnp.asarray(arr).dtype != jnp.int32:
            raise ValueError(
                f"group_metadata.{name} must be int32, got {jnp.asarray(arr).dtype}."
            )
    return group_metadata


def _planar_repack_windows(packed: _NibbleArray, window: int) -> _NibbleArray:
    """`[..., K/2]` interleaved nibble pairs -> planar window bytes.

    Shared by `repack_planar_stack` (whole array or one slab). Works on
    numpy arrays, jax arrays, AND tracers alike (only `& >> | reshape
    stack` are used, all trace-safe on the jnp side).
    """
    xp = jnp if isinstance(packed, (jax.Array, jax.core.Tracer)) else np
    half_k = packed.shape[-1]
    k = 2 * half_k
    hw = window // 2
    lead = packed.shape[:-1]
    pairs = packed.reshape(lead + (k // window, hw))
    # Nibbles back into original contraction order, per window.
    codes = xp.stack([pairs & 0x0F, pairs >> 4], axis=-1).reshape(
        lead + (k // window, window)
    )
    lo = codes[..., :hw]
    hi = codes[..., hw:]
    return (lo | (hi << 4)).astype(packed.dtype).reshape(lead + (half_k,))


def repack_planar_stack(
    packed: npt.ArrayLike | jax.Array, window: int = PLANAR_WINDOW
) -> _NibbleArray:
    """Reorders MXFP4 nibble pairs into the planar window layout.

    Interleaved storage puts contraction elements (2j, 2j+1) in byte j, so a
    decode that reads the two nibble planes whole emits even-then-odd order
    and the activation would have to be permuted to match. Planar storage
    puts elements (w*window + j, w*window + window/2 + j) in window w's byte
    j instead: the same two whole-plane nibble decodes then emit each
    window's elements in the original contraction order, so no operand is
    permuted at runtime.

    Scales are not repacked: group g still owns contraction elements
    [32g, 32g + 32), which the planar decode consumes as the first half of a
    window's groups against the low plane and the second half against the
    high plane.

    Must stay trace-safe: weight loading can apply this under `jax.jit`, so
    jax values and tracers take the pure-jnp path (a deterministic byte
    permutation) and never force a numpy conversion; only host-resident
    arrays take the bounded numpy slab walk. Branches on shape/dtype only,
    never on values.

    Args:
      packed: `[..., K/2]` uint8 nibble pairs in checkpoint (interleaved) order.
        K must divide by `window`. jax array or tracer in -> jnp out; numpy (or
        lazy host array-like) in -> numpy out (slab-wise, bounded transients).
      window: the planar window in contraction elements; must equal the kernel k
        chunk (`PLANAR_WINDOW`).

    Returns:
      The repacked array, same shape and dtype.
    """
    half_k = packed.shape[-1]
    k = 2 * half_k
    if window % (2 * SCALE_GROUP):
        raise ValueError(
            f"planar window {window} must be a multiple of {2 * SCALE_GROUP} so"
            " each nibble plane covers whole scale groups."
        )
    if k % window:
        raise ValueError(
            f"contraction {k} must be a multiple of the planar window {window}."
        )
    if isinstance(packed, (jax.Array, jax.core.Tracer)):
        # jit/trace path: pure jnp; numpy here would raise
        # TracerArrayConversionError under a trace.
        return _planar_repack_windows(packed, window)
    if not isinstance(packed, np.ndarray):
        # Lazy host array-likes (tensorstore / memmap views): materialize
        # once, then the bounded slab walk below.
        packed = np.asarray(packed)
    if packed.ndim < 2:
        return _planar_repack_windows(packed, window)
    # Host arrays are checkpoint-scale ([E, N, K/2] is ~5 GiB per real
    # leaf and the repack holds ~4x that in transients); walk a flat row
    # view in bounded slabs so the load projection's peak stays small.
    flat = packed.reshape(-1, half_k)
    out = np.empty_like(flat)
    step = max(1, (256 * 1024 * 1024) // max(half_k, 1))
    for s in range(0, flat.shape[0], step):
        out[s : s + step] = _planar_repack_windows(flat[s : s + step], window)
    return out.reshape(packed.shape)


def unpack_planar_stack(
    packed: _NibbleArray, window: int = PLANAR_WINDOW
) -> _NibbleArray:
    """The inverse of `repack_planar_stack` (back to interleaved pairs).

    Round trips bitwise by construction; used by tests and reference
    checks. Same trace-safety dispatch as the repack: jax values and tracers
    stay on jnp, host arrays stay on numpy.
    """
    xp = jnp if isinstance(packed, (jax.Array, jax.core.Tracer)) else np
    half_k = packed.shape[-1]
    k = 2 * half_k
    hw = window // 2
    if k % window:
        raise ValueError(
            f"contraction {k} must be a multiple of the planar window {window}."
        )
    lead = packed.shape[:-1]
    planar = packed.reshape(lead + (k // window, hw))
    # Original contraction order per window, then re-pair (even, odd).
    codes = xp.concatenate([planar & 0x0F, planar >> 4], axis=-1)
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).astype(packed.dtype).reshape(lead + (half_k,))


def repack_planar_t_stack(
    packed: npt.ArrayLike | jax.Array, window: int = PLANAR_WINDOW
) -> _NibbleArray:
    """Planar window bytes, transposed: `[..., N, K/2]` -> `[..., K/2, N]`.

    The planar_t layout is `repack_planar_stack` followed by a swap of the
    last two dims, so byte `[..., w*window/2 + j, n]` holds the nibbles of
    contraction elements `w*window + j` (low) and `w*window + window/2 + j`
    (high) of weight row `n`. Same weight values bit for bit; only placement
    changes. The decoded tile then comes out `[chunk, tn]` with the weight
    row on the lane dim, so the per-group scale varies along lanes and the
    decode table can be built per lane (`_dequantize_chunk_planar_t_b11`).

    Trace-safe exactly like `repack_planar_stack` (jax values and tracers
    stay on jnp; host arrays take the bounded slab walk, then a numpy
    transpose materialized once).
    """
    planar = repack_planar_stack(packed, window)
    if isinstance(planar, (jax.Array, jax.core.Tracer)):
        return jnp.swapaxes(planar, -1, -2)
    return np.ascontiguousarray(np.swapaxes(planar, -1, -2))


def unpack_planar_t_stack(
    packed: _NibbleArray, window: int = PLANAR_WINDOW
) -> _NibbleArray:
    """The inverse of `repack_planar_t_stack` (back to interleaved pairs).

    Round trips bitwise by construction (a transpose then the planar
    unpack); used by tests and reference checks.
    """
    xp = jnp if isinstance(packed, (jax.Array, jax.core.Tracer)) else np
    return unpack_planar_stack(xp.swapaxes(packed, -1, -2), window)


def transpose_scales_planar_t(scales: npt.ArrayLike | jax.Array) -> _NibbleArray:
    """`[..., N, K/32]` e8m0 codes -> `[..., K/32, N]`, the planar_t order.

    Scales carry no nibble structure, so their planar_t form is a plain
    transpose of the last two dims (group g still owns contraction
    elements [32g, 32g + 32)). Self-inverse modulo the axis swap: apply
    it again to get back. Trace-safe like the packed repack.
    """
    if isinstance(scales, (jax.Array, jax.core.Tracer)):
        return jnp.swapaxes(scales, -1, -2)
    if not isinstance(scales, np.ndarray):
        scales = np.asarray(scales)
    return np.ascontiguousarray(np.swapaxes(scales, -1, -2))


# The e2m1 nibble -> float8_e4m3b11fnuz byte relabel, as the (sign,
# magnitude) arithmetic the kernels build from an iota (Pallas cannot
# capture array constants). e4m3b11fnuz is bias-11 with no negative zero:
# magnitude m in [2, 7] lands at byte 4 * m + 80, magnitude 1 (the value
# 0.5, which renormalizes) at 0x50, magnitude 0 at 0x00 for both signs (the
# fnuz 0x80 slot is NaN, so e2m1's -0 folds onto +0; -0 * scale contributes
# nothing to any dot).
_B11_MAG1_BYTE = 0x50
_B11_MAG_BASE = 80


def _b11_relabel_table(tn: int) -> jax.Array:
    """`[16, tn]` uint8: each e2m1 nibble's exactly-equal e4m3b11fnuz byte."""
    c = lax.broadcasted_iota(jnp.int32, (16, tn), 0)
    mag = c & 7
    body = jnp.where(mag == 1, _B11_MAG1_BYTE, 4 * mag + _B11_MAG_BASE)
    return jnp.where(mag == 0, 0, ((c >> 3) << 7) | body).astype(jnp.uint8)


# The exact fold window for `planar_t_b11`: within one repack window the
# per-group e8m0 codes may span at most this many octaves. The fold is an
# exponent-field subtraction on the e4m3b11 byte, so it is exact while the
# folded code stays normal; e2m1's smallest magnitude (0.5) encodes at
# E = 10, leaving 9 octaves of headroom. `planar_t_b11_window_ok` checks
# it on concrete scales (the Kimi K3 checkpoint's 99th-percentile spread
# is 2).
B11_FOLD_WINDOW = 9


def planar_t_scale_spread(
    scales: npt.ArrayLike | jax.Array, chunk: int = PLANAR_WINDOW
) -> int:
    """Widest per-window e8m0 octave spread in `scales` (the fold demand).

    The b11 fold rewrites a group's code as `w * 2^-delta` with
    `delta = s_max - s` taken over the window, so the largest `delta` any
    window asks for is exactly this spread. Runs on host values (numpy).

    Args:
      scales: e8m0 codes in any of the layouts this module uses; only the
        contraction axis grouping matters, so `[..., N, K/32]` (checkpoint or
        planar order) and `[..., K/32, N]` (planar_t) are both accepted.
      chunk: the repack window in contraction elements.

    Returns:
      The maximum spread over every (column, window) pair.
    """
    arr = np.asarray(scales)
    groups = chunk // SCALE_GROUP
    # Find the axis that counts scale groups: the planar_t layouts put it
    # second-minor, the checkpoint layouts minor.
    axis = -1 if arr.shape[-1] % groups == 0 else -2
    moved = np.moveaxis(arr, axis, -1)
    windows = moved.reshape(*moved.shape[:-1], moved.shape[-1] // groups, groups)
    spread = windows.max(axis=-1).astype(np.int32) - windows.min(axis=-1).astype(
        np.int32
    )
    return int(spread.max())


def planar_t_b11_window_ok(
    scales: npt.ArrayLike | jax.Array, chunk: int = PLANAR_WINDOW
) -> bool:
    """True when every window column's e8m0 spread fits `B11_FOLD_WINDOW`.

    Args:
      scales: e8m0 codes, any layout `planar_t_scale_spread` accepts.
      chunk: the repack window in contraction elements.

    Returns:
      Whether `planar_t_b11` decodes these weights exactly. A False here means
      the wide columns would mis-decode. The vLLM weight-loading path does not
      run this check.
    """
    return planar_t_scale_spread(scales, chunk) <= B11_FOLD_WINDOW


def _dequantize_chunk_planar_t_b11(
    packed: jax.Array,
    scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Transposed-planar chunk -> f8e4m3b11 codes + one post-dot column scale.

    Never materializes bf16 weights: each e2m1 nibble is relabelled into its
    exactly equal float8_e4m3b11fnuz byte and the codes go to the MXU, which
    on TPU v6e expands float8_e4m3b11fnuz to bf16 in hardware (TPU7x has no
    such hardware path, so there the widen would be software). Per chunk
    step that halves the gather output (u8, not bf16), uses the native byte
    gather, and takes the scale multiply off the per-element path.

    Exactness: all 16 e2m1 values embed exactly in e4m3b11fnuz. The
    per-group e8m0 scale splits into a per-window maximum and a per-group
    delta: `w * 2^(s - 127) = (w * 2^-delta) * 2^(s_max - 127)` with
    `delta = s_max - s`. The folded code `w * 2^-delta` is exactly
    representable while it stays normal, the expansion to bf16 is exact, and
    the residual `2^(s_max - 127)` is a power of two applied once to the f32
    accumulator, where it is exact. So the result equals the dot against the
    dequantized bf16 weights on any window inside the fold bound.

    That bound is `B11_FOLD_WINDOW` = 9 and is set by the smallest
    magnitude: the fold is an integer subtraction on the exponent field, so
    it is exact while `E - delta >= 1`. e2m1's largest magnitude
    (6 = 1.5 x 2^2) sits at E = 13 and tolerates delta <= 12, but its
    smallest (0.5 = 2^-1) sits at E = 10 and tolerates only delta <= 9; a
    window is decoded correctly only if every code in it is, so 9 binds.
    Past it the field underflows into the sign bit and the wide columns
    mis-decode entirely.

    Args:
      packed: `[chunk/2, tn]` uint8 planar_t window bytes.
      scales: `[chunk/32, tn]` uint8 e8m0 exponents (group g = row g).

    Returns:
      `(codes, wscale)`: `[chunk, tn]` f8e4m3b11fnuz weights folded by the
      per-column window delta, and the `[1, tn]` float32 window scale the
      caller multiplies into the f32 accumulator after the dot.
    """
    half_k, tn = packed.shape
    groups = scales.shape[0]
    if groups % 2:
        raise ValueError(
            f"planar_t windows split {groups} scale groups across two nibble"
            " planes; the count must be even."
        )
    codes32 = scales.astype(jnp.int32)
    s_max = jnp.max(codes32, axis=0, keepdims=True)  # [1, tn]
    delta = s_max - codes32  # [groups, tn], 0 at the window max
    wscale = jnp.where(
        s_max == 0,
        jnp.float32(2.0**-127),
        lax.bitcast_convert_type(s_max.astype(jnp.uint32) << 23, jnp.float32),
    )
    # The folded table, built with integer arithmetic on the e4m3b11 byte.
    # The v6e VPU has no f8 convert instruction (its only f8 hardware is
    # the MXU expansion), so spelling this as `.astype(f8e4m3b11fnuz)`
    # lowers to a ~20-op software conversion per vreg (measured at +900
    # bundles on this loop). Scaling by 2^-delta is instead exactly an
    # exponent-field subtraction: the field sits at bits 3..6, so the
    # folded byte is `base - (delta << 3)`, with the zero entries pinned
    # at the all-zero byte. That is exact while the result stays normal,
    # i.e. for delta <= B11_FOLD_WINDOW.
    base = _b11_relabel_table(tn).astype(jnp.int32)  # [16, tn]
    folded = base[None, :, :] - (delta[:, None, :] << 3)
    table_u8 = jnp.where(base[None, :, :] == 0, 0, folded).astype(jnp.uint8)

    # Raw-nibble u8 indices: the NATIVE byte gather path (no 16-bit index
    # emulation), one lookup per output byte.
    nib = packed
    gh = groups // 2
    rows = half_k // gh
    idx = jnp.concatenate(
        [
            (nib & 0x0F).reshape(gh, rows, tn),
            (nib >> 4).reshape(gh, rows, tn),
        ],
        axis=0,
    ).astype(jnp.uint8)
    out = jnp.take_along_axis(
        table_u8,
        idx,
        axis=1,
        mode="promise_in_bounds",
        wrap_negative_indices=False,
    )
    codes = lax.bitcast_convert_type(
        out.reshape(groups * rows, tn), jnp.float8_e4m3b11fnuz
    )
    return codes, wscale


# Rolled schedule: one BlockSpec-pipelined grid step per (column tile,
# row-tile visit), the k walk a rolled loop over PLANAR_WINDOW chunks.
DEQUANT_PLANAR_T_B11 = "planar_t_b11"
# Deep schedule: the same chunk math inside a hand-rolled weight
# pipeline (see `_deep_planar_t_b11_call`). Shapes the deep kernel cannot
# take fall back to the rolled kernel; the two are bitwise identical.
DEQUANT_PLANAR_T_B11_DEEP = "planar_t_b11_deep"
# Both modes read the transposed planar layout (`[E, K/2, N]` nibbles
# plus `[E, K/32, N]` scales from `repack_planar_t_stack` /
# `transpose_scales_planar_t`) with the kernel k chunk pinned to
# PLANAR_WINDOW.
DEQUANT_MODES = (DEQUANT_PLANAR_T_B11, DEQUANT_PLANAR_T_B11_DEEP)


def _planar_t_fold_chunk(
    dequant: str,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """The `(packed, scales) -> (codes, wscale)` step for a fold mode."""
    if dequant == DEQUANT_PLANAR_T_B11:
        return _dequantize_chunk_planar_t_b11
    raise ValueError(f"{dequant!r} is not a planar_t fold mode.")


# Deep mode -> the rolled mode whose chunk step it wraps (and falls back
# to for shapes the deep kernel cannot take).
DEEP_DEQUANT_BODIES = {DEQUANT_PLANAR_T_B11_DEEP: DEQUANT_PLANAR_T_B11}

# Deep-schedule pipeline depth: outstanding `[K, tn]` weight slab copies
# per stack. Decode is bound by the weight stream serialized behind the
# chunk decode, not by VPU work or bytes, so keeping several slab copies
# in flight (instead of the BlockSpec pipeline's one) overlaps the
# stream with the decode.
DEEP_STREAM_DEPTH = 4
# Per-tile f32 accumulator bound of the deep kernel, per stack.
DEEP_MAX_ACC_BYTES = 2 * 1024 * 1024
# Row tile slots of the deep kernel: the visited `[tm, K]` lhs tile is
# double buffered so visit j + 1's rows land while visit j computes.
DEEP_ROW_TILE_SLOTS = 2


def _e8m0_to_f32(scales: jax.Array) -> jax.Array:
    """2^(u8-127) by exponent-bit construction, exact for all 256 codes.

    Not a libm exp2: code 0 is the f32 subnormal 2^-127 (whose bit pattern
    is not e<<23) and 255 is inf, so building the exponent field directly
    matches np.exp2 bit for bit.
    """
    exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
    return jnp.where(scales == 0, jnp.float32(2.0**-127), exps)


def _use_k_loop(k: int, chunk: int) -> bool:
    """Whether the k walk runs as a real (rolled) loop for this tiling.

    The rolled walk needs every per-iteration ref slice lane-aligned: the
    packed nibble plane advances `chunk/2` u8 lanes per step, so `chunk`
    must be a multiple of 256. Below that (toy shapes) the walk stays a
    static python unroll, whose text is small at those sizes anyway. Rolling
    the real serving shapes keeps one kernel's program text small, so every
    per-layer call site shares one copy and Mosaic compiles faster.
    """
    return chunk % 256 == 0 and k > chunk


def _situ_f32(
    gate: jax.Array,
    up: jax.Array,
    *,
    beta: float,
    linear_beta: float | None,
) -> jax.Array:
    """SiTU-GLU in f32.

    `[beta * tanh(gate / beta) * sigmoid(gate)] * [linear_beta * tanh(up /
    linear_beta)]` (up stays linear when `linear_beta` is None). `beta == 0.0`
    selects plain SiLU-and-multiply (`parse_situ_activation` maps 'silu' to
    it). Shared by the kernels and the fused reference so on-TPU and off-TPU
    paths run the same arithmetic.
    """
    g = gate.astype(jnp.float32)
    u = up.astype(jnp.float32)
    if beta == 0.0:
        return jax.nn.silu(g) * u
    a = beta * jnp.tanh(g / beta) * jax.nn.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * jnp.tanh(u / linear_beta)
    return a * u


def _validate(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, int, int],
) -> None:
    """Rejects shapes the kernel cannot serve.

    The weights are in the planar_t orientation: `[E, K/2, N]` packed
    nibbles and `[E, K/32, N]` e8m0 scales.
    """
    if lhs.ndim != 2:
        raise ValueError(f"lhs must be 2-D, got {lhs.ndim}-D.")
    if packed.ndim != 3 or scales.ndim != 3:
        raise ValueError(
            "packed and scales must be 3-D [E, K/2, N] and [E, K/32, N], got"
            f" {packed.shape} and {scales.shape}."
        )
    if packed.dtype != jnp.uint8 or scales.dtype != jnp.uint8:
        raise ValueError(
            f"packed/scales must be uint8, got {packed.dtype}/{scales.dtype}."
        )
    if group_sizes.dtype != jnp.int32:
        raise ValueError(f"group_sizes must be int32, got {group_sizes.dtype}.")
    k = lhs.shape[1]
    if packed.shape[1] * 2 != k:
        raise ValueError(
            f"packed contraction {packed.shape[1] * 2} != lhs contraction {k}."
        )
    if scales.shape[1] * SCALE_GROUP != k:
        raise ValueError(
            f"scales cover {scales.shape[1] * SCALE_GROUP} elements, expected {k}."
        )
    if packed.shape[0] != scales.shape[0] or packed.shape[2] != scales.shape[2]:
        raise ValueError(
            f"packed {packed.shape} and scales {scales.shape} disagree on"
            " (experts, out)."
        )
    tm, chunk, tn = tiling
    # A scale group must never straddle a dequant chunk, or the chunk
    # would need a neighbouring chunk's exponent.
    if chunk % SCALE_GROUP:
        raise ValueError(f"k chunk {chunk} must be a multiple of {SCALE_GROUP}.")
    if k % chunk:
        raise ValueError(f"contraction {k} must be a multiple of k chunk {chunk}.")
    if lhs.shape[0] % tm:
        raise ValueError(f"rows {lhs.shape[0]} must be a multiple of m tile {tm}.")
    # Pallas TPU wants a block's second-minor dim 8-divisible and its
    # minor dim 128-divisible, unless the block covers the whole array
    # dim. Every block here spans the whole contraction, so only the row
    # and column tiles need checking. Interpret mode does NOT enforce
    # this, so it is checked here or it only shows up at TPU lowering.
    n = packed.shape[2]
    if tm % 8 and tm != lhs.shape[0]:
        raise ValueError(
            f"m tile {tm} must be a multiple of 8 or all {lhs.shape[0]} rows."
        )
    if tn % _LANES and tn != n:
        raise ValueError(
            f"n tile {tn} must be a multiple of {_LANES} or all {n} columns."
        )


def _deep_eligible(m: int, k: int, n: int, tm: int, chunk: int, tn: int) -> bool:
    """Whether the deep-stream kernel serves these static shapes.

    Divisibility plus the per-tile f32 accumulator bound
    (DEEP_MAX_ACC_BYTES per stack). There is deliberately no bound on `m`:
    choosing the deep schedule for decode-sized batches is the caller's
    job. Ineligible shapes fall back to the bitwise-identical rolled
    `planar_t_b11` kernel; only the schedule differs.
    """
    return (
        m % tm == 0
        and k % chunk == 0
        and n % tn == 0
        and tm * tn * 4 <= DEEP_MAX_ACC_BYTES
    )


def _deep_fit_tn(
    *,
    k: int,
    n: int,
    tm: int,
    chunk: int,
    tn: int,
    num_stacks: int,
    budget: int,
    lhs_bytes: int,
    out_bytes: int,
) -> int | None:
    """Largest column tile whose deep-kernel VMEM footprint fits `budget`.

    The deep kernel's VMEM working set is not the rolled kernel's: each
    stack keeps DEEP_STREAM_DEPTH outstanding `[k, tn']` slab copies of
    nibbles plus scale codes, and the operands stream through
    DEEP_ROW_TILE_SLOTS `[tm, k]` lhs tiles and one `[tm, n]` out tile.
    `default_tiling` models the rolled kernel, so a tiling it admits can
    overflow the deep kernel. Shrinking only the deep kernel's column tile
    is value-exact: tn' regroups the output columns and the DMA slabs, and
    every element's chunk-ordered accumulation and per-column window scale
    are unchanged. Returns the largest lane-divisible tn' <= tn that
    divides n and fits, or None (the caller falls back to the rolled
    kernel).
    """
    steps = max(k // chunk, 1)
    slab_bytes = steps * (chunk // 2 + chunk // SCALE_GROUP)
    operands = DEEP_ROW_TILE_SLOTS * tm * k * lhs_bytes + tm * n * out_bytes
    cand = tn
    while cand >= _LANES:
        if n % cand == 0:
            scratch = num_stacks * DEEP_STREAM_DEPTH * slab_bytes * cand
            accs = num_stacks * tm * cand * 4
            total = operands + scratch + accs
            if total <= budget:
                return cand
        cand //= 2
    return None


def _deep_planar_t_b11_call(
    lhs: jax.Array,
    stacks: tuple[tuple[jax.Array, jax.Array], ...],
    group_metadata: tuple[jax.Array, jax.Array, jax.Array],
    num_active_tiles: jax.Array,
    *,
    tm: int,
    chunk: int,
    tn: int,
    situ: tuple[float, float | None] | None,
    preferred_element_type: jax.typing.DTypeLike,
    interpret: bool,
) -> jax.Array:
    """Deep-stream grouped matmul shared by the plain and fused entry points.

    Runs `planar_t_b11`'s chunk math (same codes, same window scale, same
    accumulation order, so bitwise the same outputs) inside a hand-rolled
    weight pipeline:

      * Weights: per active (row tile, group) visit, the expert's `[k, tn]`
        slab tiles stream HBM->VMEM through DEEP_STREAM_DEPTH outstanding
        copies per stack while the unrolled k walk decodes and multiplies
        the previously landed tile. The (visit, column tile) sequence is
        one flat stream: the copy for global tile q + depth is issued after
        tile q computes, so the last `depth` tiles of visit j already pull
        visit j + 1's first tiles and the pipeline is primed once per call.
      * Rows: the visit's `[tm, k]` lhs tile is copied from HBM into one of
        DEEP_ROW_TILE_SLOTS VMEM slots (visit j + 1's tile lands while visit
        j computes). The visit's `[tm, n]` out tile is assembled in one VMEM
        buffer across its column tiles and copied out when the next visit
        moves to another row tile (or at the last visit), so a tile
        straddling two groups is merged in VMEM. Rows the kernel never
        visits, and rows of a visited tile outside every group, are
        unspecified (the callers' contract).

    Args:
      lhs: `[m, k]` activations, rows grouped by expert.
      stacks: the chunk-major planar_t weights: one `(packed, scales)` pair of
        `[E, steps, chunk/2, n]` / `[E, steps, chunk/32, n]` arrays for the plain
        matmul, or the fused gate+up form where both pairs are the same
        `[E, 2, steps, chunk/2, n]` / `[E, 2, steps, chunk/32, n]` arrays and
        stack `s` reads index `s` of axis 1.
      group_metadata: `(group_offsets, group_ids, m_tile_ids)` from
        `_make_group_metadata`.
      num_active_tiles: number of valid entries of the metadata grid.
      tm: row tile.
      chunk: k chunk (PLANAR_WINDOW).
      tn: column tile.
      situ: `(beta, linear_beta)` for the fused gate+up epilogue (`_situ_f32` on
        the two accumulators), or None for the plain matmul.
      preferred_element_type: output dtype.
      interpret: run the Pallas kernel in interpret mode.

    Returns:
      `[m, n]` products (SiTU-gated when `situ` is set).
    """
    m, k = lhs.shape
    num_stacks = len(stacks)
    if stacks[0][0].ndim == 5:
        _, _, steps, plane_rows, n = stacks[0][0].shape
        scale_rows = stacks[0][1].shape[3]
    else:
        _, steps, plane_rows, n = stacks[0][0].shape
        scale_rows = stacks[0][1].shape[2]
    tiles_n = n // tn
    depth = min(DEEP_STREAM_DEPTH, tiles_n)
    group_offsets, group_ids, m_tile_ids = group_metadata
    grid_len = m_tile_ids.shape[0]
    fold_chunk = _dequantize_chunk_planar_t_b11

    def kernel(off_ref, ids_ref, mtid_ref, nat_ref, lhs_ref, *rest):
        stack_refs = rest[: 2 * num_stacks]
        out_ref = rest[2 * num_stacks]
        scratch = rest[2 * num_stacks + 1 :]
        bufs = scratch[: 2 * num_stacks]
        acc_refs = scratch[2 * num_stacks : 3 * num_stacks]
        sems = scratch[3 * num_stacks : 5 * num_stacks]
        lhs_slots, out_buf, lhs_sems, out_sem = scratch[5 * num_stacks :]

        def lhs_copy(row0, slot):
            return pltpu.make_async_copy(
                lhs_ref.at[pl.ds(row0, tm), :],
                lhs_slots.at[slot],
                lhs_sems.at[slot],
            )

        def copies(g, t, slot):
            out = []
            for s in range(num_stacks):
                packed_ref, scales_ref = stack_refs[2 * s], stack_refs[2 * s + 1]
                pbuf, sbuf = bufs[2 * s], bufs[2 * s + 1]
                psem, ssem = sems[2 * s], sems[2 * s + 1]
                if packed_ref.ndim == 5:
                    packed_src = packed_ref.at[g, s, :, :, pl.ds(t * tn, tn)]
                    scales_src = scales_ref.at[g, s, :, :, pl.ds(t * tn, tn)]
                else:
                    packed_src = packed_ref.at[g, :, :, pl.ds(t * tn, tn)]
                    scales_src = scales_ref.at[g, :, :, pl.ds(t * tn, tn)]
                out.append(
                    pltpu.make_async_copy(
                        packed_src,
                        pbuf.at[slot],
                        psem.at[slot],
                    )
                )
                out.append(
                    pltpu.make_async_copy(scales_src, sbuf.at[slot], ssem.at[slot])
                )
            return out

        def visit(j, carry):
            g = ids_ref[j]
            mt = mtid_ref[j]
            row0 = mt * tm
            start = off_ref[g]
            end = off_ref[g + 1]
            lslot = lax.rem(j, DEEP_ROW_TILE_SLOTS)
            # This visit's rows were issued by the previous visit (or the
            # prime below for visit 0); the other slot's reader, visit
            # j - 1, is done, so visit j + 1's rows can land in it now.
            lhs_copy(row0, lslot).wait()

            @pl.when(j + 1 < nat_ref[0])
            def _():
                lhs_copy(
                    mtid_ref[j + 1] * tm, lax.rem(j + 1, DEEP_ROW_TILE_SLOTS)
                ).start()

            # The out tile is flushed by the last visit of its row tile;
            # visits of one row tile are consecutive (m_tile_ids is
            # nondecreasing), so "last" is "the next visit is on another
            # tile, or there is none". The clamp keeps the lookahead read in
            # bounds at the padded tail.
            next_mt = mtid_ref[jnp.minimum(j + 1, grid_len - 1)]
            last_of_tile = jnp.logical_or(j + 1 >= nat_ref[0], next_mt != mt)
            # Slot of tile t in the flattened (visit, tile) stream; the
            # copies for this visit's first `depth` tiles were issued by
            # the previous visit (or the prime below for visit 0).
            base = lax.rem(j * tiles_n, depth)
            for t in range(tiles_n):
                slot = lax.rem(base + t, depth)
                for c in copies(g, t, slot):
                    c.wait()
                # Accumulate through a scratch ref, not a carried value: each
                # `+=` then materializes exactly like the rolled kernel's, so
                # no backend can fuse the window-scale multiply into the add
                # (an FMA there is a 1-ulp reassociation and breaks bitwise
                # equality with the rolled kernel).
                accs = []
                for s in range(num_stacks):
                    pbuf, sbuf = bufs[2 * s], bufs[2 * s + 1]
                    acc_ref = acc_refs[s]
                    acc_ref[...] = jnp.zeros_like(acc_ref)
                    for cs in range(steps):
                        lhs_chunk = lhs_slots[lslot, :, cs * chunk : (cs + 1) * chunk]
                        dims = (((1,), (0,)), ((), ()))
                        codes, wscale = fold_chunk(pbuf[slot, cs], sbuf[slot, cs])
                        acc_ref[...] += (
                            lax.dot_general(
                                lhs_chunk,
                                codes,
                                dimension_numbers=dims,
                                preferred_element_type=jnp.float32,
                            )
                            * wscale
                        )
                    accs.append(acc_ref[...])
                if situ is not None:
                    value = _situ_f32(
                        accs[0].astype(preferred_element_type),
                        accs[1].astype(preferred_element_type),
                        beta=situ[0],
                        linear_beta=situ[1],
                    )
                else:
                    value = accs[0]
                row = lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + row0
                mask = jnp.logical_and(row >= start, row < end)
                first_of_tile = start <= row0
                old = jnp.where(
                    first_of_tile,
                    0.0,
                    out_buf[:, pl.ds(t * tn, tn)].astype(jnp.float32),
                )
                out_buf[:, pl.ds(t * tn, tn)] = jnp.where(mask, value, old).astype(
                    preferred_element_type
                )
                nxt = t + depth
                if nxt < tiles_n:
                    for c in copies(g, nxt, slot):
                        c.start()
                else:
                    # This slot is free and, in the flattened stream, next holds
                    # tile nxt - tiles_n of the NEXT visit: pull it now so the
                    # visit boundary never drains the pipeline. Every copy issued
                    # here is waited by visit j + 1, which exists by the guard.
                    @pl.when(j + 1 < nat_ref[0])
                    def _(nxt=nxt, slot=slot):
                        g_next = ids_ref[j + 1]
                        for c in copies(g_next, nxt - tiles_n, slot):
                            c.start()

            @pl.when(last_of_tile)
            def _():
                flush = pltpu.make_async_copy(
                    out_buf, out_ref.at[pl.ds(row0, tm), :], out_sem
                )
                flush.start()
                flush.wait()

            return carry

        @pl.when(nat_ref[0] > 0)
        def _():
            lhs_copy(mtid_ref[0] * tm, 0).start()

        @pl.when(nat_ref[0] > 0)
        def _():
            g_first = ids_ref[0]
            for s in range(depth):
                for c in copies(g_first, s, s):
                    c.start()

        lax.fori_loop(0, nat_ref[0], visit, 0)

    flops = 2 * m * k * n * num_stacks
    bytes_accessed = (
        lhs.size * lhs.itemsize
        + sum(p.size + s.size for p, s in stacks)
        + m * n * jnp.dtype(preferred_element_type).itemsize
    )
    scratch_shapes = []
    for _ in range(num_stacks):
        scratch_shapes.append(pltpu.VMEM((depth, steps, plane_rows, tn), jnp.uint8))
        scratch_shapes.append(pltpu.VMEM((depth, steps, scale_rows, tn), jnp.uint8))
    for _ in range(num_stacks):
        scratch_shapes.append(pltpu.VMEM((tm, tn), jnp.float32))
    for _ in range(num_stacks):
        scratch_shapes.append(pltpu.SemaphoreType.DMA((depth,)))
        scratch_shapes.append(pltpu.SemaphoreType.DMA((depth,)))
    scratch_shapes.append(pltpu.VMEM((DEEP_ROW_TILE_SLOTS, tm, k), lhs.dtype))
    scratch_shapes.append(pltpu.VMEM((tm, n), preferred_element_type))
    scratch_shapes.append(pltpu.SemaphoreType.DMA((DEEP_ROW_TILE_SLOTS,)))
    scratch_shapes.append(pltpu.SemaphoreType.DMA(()))
    # The operands stay wherever XLA puts them; the kernel moves row tiles
    # itself. No `vmem_limit_bytes`: the kernel fits the default scoped
    # VMEM limit.
    call = pl.pallas_call(
        kernel,
        grid=(),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 4
        + [pl.BlockSpec(memory_space=pl.ANY)]
        + [pl.BlockSpec(memory_space=pl.ANY)] * (2 * num_stacks),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        scratch_shapes=scratch_shapes,
        interpret=interpret,
        cost_estimate=pl.CostEstimate(
            flops=flops, bytes_accessed=bytes_accessed, transcendentals=0
        ),
    )
    stack_args = []
    for p, s in stacks:
        stack_args.extend((p, s))
    return call(
        group_offsets.astype(jnp.int32),
        group_ids.astype(jnp.int32),
        m_tile_ids.astype(jnp.int32),
        jnp.reshape(num_active_tiles, (1,)).astype(jnp.int32),
        lhs,
        *stack_args,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "interpret",
        "dequant",
        "deep_vmem_budget",
    ],
)
def mxfp4_gmm(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jax.typing.DTypeLike = jnp.bfloat16,
    tiling: tuple[int, int, int] | None = None,
    interpret: bool = False,
    dequant: str = DEQUANT_PLANAR_T_B11,
    deep_vmem_budget: int | None = None,
    group_metadata: GroupMetadata | None = None,
) -> jax.Array:
    """`lhs[rows of group i, :] @ dequant(packed[i], scales[i])` per group.

    Args:
      lhs: `[M, K]` activations, rows grouped by expert.
      packed: `[E, K/2, N]` uint8 MXFP4 nibbles in the planar_t layout
        (`repack_planar_t_stack`).
      scales: `[E, K/SCALE_GROUP, N]` uint8 e8m0 exponents
        (`transpose_scales_planar_t`).
      group_sizes: `[E]` int32 rows per expert; entries may be 0 and need not sum
        to M (rows past the total are left unspecified, which is how the
        expert-parallel caller parks assignments it does not own).
      preferred_element_type: output dtype.
      tiling: `(tm, k_chunk, tn)`. Blocks always span the whole contraction;
        `k_chunk` is how much of it the kernel decodes at a time and must equal
        PLANAR_WINDOW. `tm` must divide M. None derives them from the problem.
      interpret: run the Pallas kernel in interpret mode (used off TPU).
      dequant: `DEQUANT_PLANAR_T_B11` (rolled schedule) or
        `DEQUANT_PLANAR_T_B11_DEEP` (deep schedule, falling back to the rolled
        kernel for shapes it cannot take). Both are bitwise identical.
      deep_vmem_budget: VMEM bound for the deep kernel's column tile search
        (`_deep_fit_tn`); None keeps the rolled tiling's column tile.
      group_metadata: optional precomputed grid metadata (`_make_group_metadata`
        with the same `(m, tm, num_groups)`), so the gate+up and down kernels of
        one layer share one build. Checked by shape; built here when None.

    Returns:
      `[M, N]` products.
    """
    m, k = lhs.shape
    if dequant not in DEQUANT_MODES:
        raise ValueError(
            f"unknown dequant mode {dequant!r}, want one of {DEQUANT_MODES}"
        )
    num_groups = packed.shape[0]
    n = packed.shape[2]
    # Clamp before validating: a tile larger than the array is just the
    # whole array, which toy shapes rely on.
    tm, chunk, tn = tiling or default_tiling(m, k, n, force_chunk=PLANAR_WINDOW)
    tm, chunk, tn = min(tm, m), min(chunk, k), min(tn, n)
    if chunk != PLANAR_WINDOW:
        raise ValueError(
            f"planar weights are repacked per {PLANAR_WINDOW}-element window,"
            f" so the kernel k chunk must equal it; got chunk {chunk} for k"
            f" {k}. Callers gate on can_use_kernel(...) for shapes that cannot"
            " serve the window."
        )
    _validate(lhs, packed, scales, group_sizes, (tm, chunk, tn))
    if jnp.dtype(lhs.dtype).itemsize == 1:
        raise ValueError(
            f"{lhs.dtype} activations are not supported: the kernel multiplies"
            " the activations against float8_e4m3b11fnuz weight codes and"
            " expects bf16 or f32 activations."
        )
    # Chunk-major restage at zero cost: splitting the k-major dim of the
    # planar_t layout is a row-major-preserving reshape (same bytes, same
    # order), and it lets the rolled k walk index the LEADING dim of the
    # in-kernel ref, which needs no lane or sublane alignment.
    steps = max(k // chunk, 1)
    packed = packed.reshape(num_groups, steps, chunk // 2, n)
    scales = scales.reshape(num_groups, steps, chunk // SCALE_GROUP, n)
    tiles_n = (n + tn - 1) // tn

    if group_metadata is not None:
        grid_metadata, num_active_tiles = _check_group_metadata(
            group_metadata, m=m, tm=tm, num_groups=num_groups
        )
    else:
        grid_metadata, num_active_tiles = _make_group_metadata(
            group_sizes, m=m, tm=tm, num_groups=num_groups
        )

    if dequant in DEEP_DEQUANT_BODIES:
        deep_body = DEEP_DEQUANT_BODIES[dequant]
        deep_tn = (
            _deep_fit_tn(
                k=k,
                n=n,
                tm=tm,
                chunk=chunk,
                tn=tn,
                num_stacks=1,
                budget=deep_vmem_budget,
                lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
                out_bytes=jnp.dtype(preferred_element_type).itemsize,
            )
            if deep_vmem_budget is not None
            else tn
        )
        if deep_tn is not None and _deep_eligible(
            m=m, k=k, n=n, tm=tm, chunk=chunk, tn=deep_tn
        ):
            return _deep_planar_t_b11_call(
                lhs,
                ((packed, scales),),
                grid_metadata,
                num_active_tiles,
                tm=tm,
                chunk=chunk,
                tn=deep_tn,
                situ=None,
                preferred_element_type=preferred_element_type,
                interpret=interpret,
            )
        # Shapes the deep pipeline does not serve (prefill widths) take the
        # bitwise-identical rolled kernel; only the schedule differs.
        dequant = deep_body

    use_k_loop = _use_k_loop(k, chunk)
    fold_chunk = _planar_t_fold_chunk(dequant)

    def kernel(
        group_metadata,
        group_offset,
        lhs_ref,
        packed_ref,
        scales_ref,
        out_ref,
        acc,
        scale_codes_ref,
    ):
        del group_offset, scale_codes_ref
        group_offsets, group_ids, m_tile_ids = group_metadata
        grid_id = pl.program_id(1)

        acc[...] = jnp.zeros_like(acc)
        # Walk the contraction in chunks. The block covers the whole of K,
        # but decoding all of it at once puts a [K, tn] chain in flight and
        # Mosaic spills. Chunking bounds that working set while keeping every
        # decoded value in VMEM for exactly one MXU consumption. The walk is
        # a real loop, not a python unroll, so one kernel's ISA stays small
        # and every per-layer call site shares one copy of the program text.
        # The planar_t chunk-major reshape put the chunk index on the refs'
        # leading dim, so the walk indexes it directly.

        def chunk_body_t(i, carry):
            lhs_chunk = lhs_ref[:, pl.ds(i * chunk, chunk)]
            # Hand f8 codes to the MXU and apply the window scale once, on the
            # f32 accumulator.
            codes, wscale = fold_chunk(packed_ref[i], scales_ref[i])
            acc[...] += (
                lax.dot_general(
                    lhs_chunk,
                    codes,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                * wscale
            )
            return carry

        if use_k_loop:
            lax.fori_loop(0, k // chunk, chunk_body_t, 0)
        else:
            for c in range(0, k, chunk):
                chunk_body_t(c // chunk, 0)

        # Only the rows of the tile that belong to THIS group may be
        # written; a tile straddling a group boundary is visited once per
        # group it touches, and rows outside every group are left alone.
        group_id = group_ids[grid_id]
        start = group_offsets[group_id]
        end = group_offsets[group_id + 1]
        row0 = m_tile_ids[grid_id] * tm
        row = lax.broadcasted_iota(jnp.int32, acc.shape, 0) + row0
        mask = jnp.logical_and(row >= start, row < end)
        old_f32 = jnp.where(start <= row0, 0.0, out_ref[...].astype(jnp.float32))
        out_ref[...] = jnp.where(mask, acc[...], old_f32).astype(preferred_element_type)

    def lhs_indices(n_i, grid_id, group_metadata, group_offset):
        del n_i, group_offset
        return group_metadata[2][grid_id], 0

    def rhs_indices_t(n_i, grid_id, group_metadata, group_offset):
        del group_offset
        return group_metadata[1][grid_id], 0, 0, n_i

    def out_indices(n_i, grid_id, group_metadata, group_offset):
        del group_offset
        return group_metadata[2][grid_id], n_i

    weight_specs = [
        pl.BlockSpec((None, steps, chunk // 2, tn), rhs_indices_t),
        pl.BlockSpec((None, steps, chunk // SCALE_GROUP, tn), rhs_indices_t),
    ]
    cost = pl.CostEstimate(
        flops=2 * m * k * n,
        bytes_accessed=lhs.size * lhs.itemsize
        + packed.size
        + scales.size
        + m * n * jnp.dtype(preferred_element_type).itemsize,
        transcendentals=0,
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec((tm, k), lhs_indices)] + weight_specs,
            out_specs=pl.BlockSpec((tm, tn), out_indices),
            grid=(tiles_n, num_active_tiles),
            scratch_shapes=[
                pltpu.VMEM((tm, tn), jnp.float32),
                # `[k/chunk, tn, chunk/32]` u8 scratch. The planar_t walk
                # does not read it; it is kept so the compiled kernel and
                # its VMEM footprint (which `_vmem_bytes` models) stay as
                # they are.
                pltpu.VMEM((max(k // chunk, 1), tn, chunk // SCALE_GROUP), jnp.uint8),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        cost_estimate=cost,
    )(
        grid_metadata,
        jnp.array([0], dtype=jnp.int32),
        lhs,
        packed,
        scales,
    )


def mxfp4_gmm_reference(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    """Grouped MXFP4 matmul without Pallas, for CPU and equality tests.

    Dequantizes the whole stack and runs `jax.lax.ragged_dot`. Unlike
    `mxfp4_gmm` it takes the checkpoint layout.

    Args:
      lhs: `[M, K]` activations, rows grouped by expert.
      packed: `[E, N, K/2]` uint8 MXFP4 nibble pairs (checkpoint order).
      scales: `[E, N, K/32]` uint8 e8m0 exponents.
      group_sizes: `[E]` int32 rows per expert.
      preferred_element_type: output dtype; None keeps the activation dtype.

    Returns:
      `[M, N]` products; rows outside every group are zero.
    """
    e, n, half_k = packed.shape
    k = half_k * 2
    codes = jnp.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(e, n, k)
    values = jnp.take(jnp.asarray(_E2M1_LUT, jnp.float32), codes.astype(jnp.int32))
    exps = _e8m0_to_f32(scales)
    groups = scales.shape[-1]
    values = values.reshape(e, n, groups, k // groups) * exps[..., None]
    # [E, N, K] -> [E, K, N]: ragged_dot contracts lhs's minor with the
    # rhs's SECOND-minor, while the kernel contracts both minors.
    dense = values.reshape(e, n, k).transpose(0, 2, 1).astype(jnp.bfloat16)
    out = jax.lax.ragged_dot(lhs, dense.astype(lhs.dtype), group_sizes)
    return out if preferred_element_type is None else out.astype(preferred_element_type)


def mxfp4_fused_gate_up_reference(
    lhs: jax.Array,
    gate_packed: jax.Array,
    gate_scales: jax.Array,
    up_packed: jax.Array,
    up_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    """Fused gate+up MXFP4 matmul with the SiTU epilogue, without Pallas.

    Two `mxfp4_gmm_reference` matmuls followed by `_situ_f32`, so rows
    outside every group come back zero where the kernels leave them
    unspecified. Output keeps the activation dtype.

    Args:
      lhs: `[M, K]` activations, rows grouped by expert.
      gate_packed: `[E, N, K/2]` uint8 gate nibble pairs (checkpoint order).
      gate_scales: `[E, N, K/32]` uint8 gate e8m0 exponents.
      up_packed: `[E, N, K/2]` uint8 up nibble pairs.
      up_scales: `[E, N, K/32]` uint8 up e8m0 exponents.
      group_sizes: `[E]` int32 rows per expert.
      situ_beta: SiTU gate clamp (0.0 for SiLU).
      situ_linear_beta: SiTU linear clamp, or None.

    Returns:
      `[M, N]` SiTU-gated activations.
    """
    gate = mxfp4_gmm_reference(lhs, gate_packed, gate_scales, group_sizes)
    up = mxfp4_gmm_reference(lhs, up_packed, up_scales, group_sizes)
    return _situ_f32(gate, up, beta=situ_beta, linear_beta=situ_linear_beta).astype(
        gate.dtype
    )


def can_use_kernel(
    lhs_rows: int,
    k: int,
    tiling: tuple[int, int, int] | None = None,
    rhs_pairs: int = 1,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_V6E,
    dequant: str = DEQUANT_PLANAR_T_B11,
) -> bool:
    """Whether `mxfp4_gmm`'s tiling constraints hold for these shapes.

    Toy configs (a 12-wide latent, say) cannot tile a 32-element scale group
    or the 512-element planar window. `rhs_pairs=2` sizes the default tiling
    for the fused gate+up kernel (same constraint set, doubled rhs working
    set); `vmem_budget` and `scoped_elem_bytes` as `default_tiling`.

    Args:
      lhs_rows: lhs rows (M).
      k: contraction.
      tiling: explicit `(tm, k_chunk, tn)`, or None for `default_tiling`.
      rhs_pairs: weight stacks streamed per lhs block.
      vmem_budget: as `default_tiling`.
      scoped_elem_bytes: as `default_tiling`.
      dequant: one of `DEQUANT_MODES`.

    Returns:
      Whether the kernel can serve the shape.
    """
    if dequant not in DEQUANT_MODES:
        raise ValueError(
            f"unknown dequant mode {dequant!r}, want one of {DEQUANT_MODES}"
        )
    tm, chunk, _ = tiling or default_tiling(
        lhs_rows,
        k,
        1,
        rhs_pairs=rhs_pairs,
        vmem_budget=vmem_budget,
        scoped_elem_bytes=scoped_elem_bytes,
        force_chunk=PLANAR_WINDOW,
    )
    tm, chunk = min(tm, lhs_rows), min(chunk, k)
    return (
        k % SCALE_GROUP == 0
        and chunk % SCALE_GROUP == 0
        and k % chunk == 0
        and lhs_rows % tm == 0
        and (tm % 8 == 0 or tm == lhs_rows)
        and chunk == PLANAR_WINDOW
    )


# ============================================================================
# Served entry points (TPU v6e)
# ============================================================================

# Per-grid-step VMEM budgets of the served entry point, sized under the
# 32 MiB default scoped VMEM limit: half of it for the rolled kernel and
# three quarters for the deep pipeline.
V6E_ROLLED_VMEM_BUDGET = _SCOPED_VMEM_LIMIT // 2
V6E_DEEP_VMEM_BUDGET = 3 * _SCOPED_VMEM_LIMIT // 4
# Routed row count up to which `moe_gmm_v6e_planar_t_b11` uses the deep
# schedule and 32-row alignment; larger (prefill) batches use the rolled
# schedule and 128-row alignment.
_MOE_GMM_DEEP_MAX_ROWS = 2048
# Expert-parallel shard count assumed when sizing the row tile: each
# shard expects to own about `padded_m / _MOE_GMM_EP_SHARDS` of the routed
# rows (`default_tiling(expected_rows=...)`).
_MOE_GMM_EP_SHARDS = 8


def _as_rank3_scales(scale: jax.Array) -> jax.Array:
    """Drops the singleton axis 2 of `[E, rows, 1, N]` scales (rank-3 as is)."""
    return jnp.squeeze(scale, axis=2) if scale.ndim == 4 else scale


def repack_w13_planar_t(
    w13_packed: jax.Array, w13_scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Repacks a checkpoint gate+up MXFP4 stack into the planar_t layout.

    Args:
      w13_packed: `[E, 2*N, K/2]` uint8 MXFP4 nibble pairs, gate rows first.
      w13_scale: `[E, 2*N, K/32]` uint8 e8m0 scales.

    Returns:
      `(w13_pt, w13_sc)`: `[E, K, N]` uint8 planar_t nibbles (gate and up
      stacked on axis 1, i.e. `[E, 2, K/2, N]` flattened) and
      `[E, 2*(K/32), 1, N]` uint8 transposed scales (`[E, 2, K/32, 1, N]`
      flattened).
    """
    n = w13_packed.shape[1] // 2
    gate_pt = repack_planar_t_stack(w13_packed[:, :n, :])
    up_pt = repack_planar_t_stack(w13_packed[:, n:, :])
    w13_pt = jnp.concatenate([gate_pt, up_pt], axis=1)

    gate_sc = transpose_scales_planar_t(w13_scale[:, :n, :])
    up_sc = transpose_scales_planar_t(w13_scale[:, n:, :])
    w13_sc = jnp.expand_dims(jnp.concatenate([gate_sc, up_sc], axis=1), axis=2)
    return w13_pt, w13_sc


def repack_w2_planar_t(
    w2_packed: jax.Array, w2_scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Repacks a checkpoint down-projection MXFP4 stack into the planar_t layout.

    Args:
      w2_packed: `[E, L, N/2]` uint8 MXFP4 nibble pairs.
      w2_scale: `[E, L, N/32]` uint8 e8m0 scales.

    Returns:
      `(w2_pt, w2_sc)`: `[E, N/2, L]` uint8 planar_t nibbles and
      `[E, N/32, 1, L]` uint8 transposed scales.
    """
    w2_pt = repack_planar_t_stack(w2_packed)
    w2_sc = jnp.expand_dims(transpose_scales_planar_t(w2_scale), axis=2)
    return w2_pt, w2_sc


def parse_situ_activation(activation: str) -> tuple[float, float | None]:
    """Parses the expert activation string into SiTU clamp parameters.

    Args:
      activation: `'situ:<beta>:<linear_beta>'` (`linear_beta` may be `none`) or
        `'silu'`.

    Returns:
      `(beta, linear_beta)`; `'silu'` maps to `(0.0, None)`, which `_situ_f32`
      evaluates as plain SiLU-and-multiply.

    Raises:
      NotImplementedError: for any other activation.
    """
    if activation.startswith("situ:"):
        _, beta_str, lin_str = activation.split(":")
        beta = float(beta_str)
        linear_beta = None if lin_str == "none" else float(lin_str)
        return beta, linear_beta
    if activation == "silu":
        return 0.0, None
    raise NotImplementedError(
        f"Unsupported activation {activation!r} for v6e planar_t_b11 MoE kernel."
    )


def _run_fused_gate_up_5d(
    lhs: jax.Array,
    w13_pt: jax.Array,
    w13_sc_3d: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    tiling: tuple[int, int, int],
    dequant: str,
    group_metadata: GroupMetadata,
    deep_vmem_budget: int = V6E_DEEP_VMEM_BUDGET,
) -> jax.Array:
    """Fused gate+up grouped matmul with the SiTU epilogue.

    Reads the combined planar_t stack in place: `w13_pt` `[E, K, N]` is viewed
    as `[E, 2, steps, chunk/2, N]` and `w13_sc_3d` `[E, 2*(K/32), N]` as
    `[E, 2, steps, chunk/32, N]` (index 0 gate, 1 up), so no HBM copy splits
    the two halves.

    Args:
      lhs: `[M, K]` sorted routed activations.
      w13_pt: `[E, K, N]` uint8 from `repack_w13_planar_t`.
      w13_sc_3d: `[E, 2*(K/32), N]` uint8 scales.
      situ_beta: SiTU gate clamp (`parse_situ_activation`).
      situ_linear_beta: SiTU linear clamp, or None.
      tiling: `(tm, k_chunk, tn)`.
      dequant: one of `DEQUANT_MODES`.
      group_metadata: `_make_group_metadata` result for `(M, tm, E)`.
      deep_vmem_budget: VMEM bound for the deep kernel's column tile.

    Returns:
      `[M, N]` SiTU-gated activations in `lhs.dtype`.
    """
    m, k = lhs.shape
    num_groups, _, n = w13_pt.shape
    tm, chunk, tn = tiling
    tm, chunk, tn = min(tm, m), min(chunk, k), min(tn, n)
    steps = max(k // chunk, 1)
    plane_rows = chunk // 2
    scale_rows = chunk // SCALE_GROUP

    w13_packed_5d = w13_pt.reshape(num_groups, 2, steps, plane_rows, n)
    w13_scales_5d = w13_sc_3d.reshape(num_groups, 2, steps, scale_rows, n)

    grid_metadata, num_active_tiles = _check_group_metadata(
        group_metadata, m=m, tm=tm, num_groups=num_groups
    )
    preferred_element_type = lhs.dtype
    interpret = jax.default_backend() != "tpu"

    if dequant in DEEP_DEQUANT_BODIES:
        deep_body = DEEP_DEQUANT_BODIES[dequant]
        deep_tn = _deep_fit_tn(
            k=k,
            n=n,
            tm=tm,
            chunk=chunk,
            tn=tn,
            num_stacks=2,
            budget=deep_vmem_budget,
            lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
            out_bytes=jnp.dtype(preferred_element_type).itemsize,
        )
        if deep_tn is not None and _deep_eligible(
            m=m, k=k, n=n, tm=tm, chunk=chunk, tn=deep_tn
        ):
            return _deep_planar_t_b11_call(
                lhs,
                ((w13_packed_5d, w13_scales_5d), (w13_packed_5d, w13_scales_5d)),
                grid_metadata,
                num_active_tiles,
                tm=tm,
                chunk=chunk,
                tn=deep_tn,
                situ=(situ_beta, situ_linear_beta),
                preferred_element_type=preferred_element_type,
                interpret=interpret,
            )
        dequant = deep_body

    # Rolled planar_t_b11 kernel with 5D BlockSpec indexing (0 for gate, 1 for up)
    tiles_n = (n + tn - 1) // tn
    use_k_loop = _use_k_loop(k, chunk)
    fold_chunk = _planar_t_fold_chunk(dequant)
    dims_t = (((1,), (0,)), ((), ()))

    def kernel(
        group_metadata_ref,
        group_offset,
        lhs_ref,
        gate_packed_ref,
        gate_scales_ref,
        up_packed_ref,
        up_scales_ref,
        out_ref,
        gate_acc,
        up_acc,
    ):
        del group_offset
        group_offsets, group_ids, m_tile_ids = group_metadata_ref
        grid_id = pl.program_id(1)
        gate_acc[...] = jnp.zeros_like(gate_acc)
        up_acc[...] = jnp.zeros_like(up_acc)

        def chunk_body_t(i, carry):
            lhs_chunk = lhs_ref[:, pl.ds(i * chunk, chunk)]
            for packed_c, scales_c, target in (
                (gate_packed_ref[i], gate_scales_ref[i], gate_acc),
                (up_packed_ref[i], up_scales_ref[i], up_acc),
            ):
                codes, wscale = fold_chunk(packed_c, scales_c)
                target[...] += (
                    lax.dot_general(
                        lhs_chunk,
                        codes,
                        dimension_numbers=dims_t,
                        preferred_element_type=jnp.float32,
                    )
                    * wscale
                )
            return carry

        if use_k_loop:
            lax.fori_loop(0, k // chunk, chunk_body_t, 0)
        else:
            for c in range(0, k, chunk):
                chunk_body_t(c // chunk, 0)

        h = _situ_f32(
            gate_acc[...].astype(preferred_element_type),
            up_acc[...].astype(preferred_element_type),
            beta=situ_beta,
            linear_beta=situ_linear_beta,
        )
        group_id = group_ids[grid_id]
        start = group_offsets[group_id]
        end = group_offsets[group_id + 1]
        row0 = m_tile_ids[grid_id] * tm
        row = lax.broadcasted_iota(jnp.int32, h.shape, 0) + row0
        mask = jnp.logical_and(row >= start, row < end)
        old_f32 = jnp.where(start <= row0, 0.0, out_ref[...].astype(jnp.float32))
        out_ref[...] = jnp.where(mask, h, old_f32).astype(preferred_element_type)

    def lhs_indices(n_i, grid_id, gm, go):
        del n_i, go
        return gm[2][grid_id], 0

    def gate_indices_t5(n_i, grid_id, gm, go):
        del go
        return gm[1][grid_id], 0, 0, 0, n_i

    def up_indices_t5(n_i, grid_id, gm, go):
        del go
        return gm[1][grid_id], 1, 0, 0, n_i

    def out_indices(n_i, grid_id, gm, go):
        del go
        return gm[2][grid_id], n_i

    gate_specs = [
        pl.BlockSpec((None, None, steps, plane_rows, tn), gate_indices_t5),
        pl.BlockSpec((None, None, steps, scale_rows, tn), gate_indices_t5),
    ]
    up_specs = [
        pl.BlockSpec((None, None, steps, plane_rows, tn), up_indices_t5),
        pl.BlockSpec((None, None, steps, scale_rows, tn), up_indices_t5),
    ]
    cost = pl.CostEstimate(
        flops=2 * 2 * m * k * n,
        bytes_accessed=lhs.size * lhs.itemsize
        + w13_pt.size
        + w13_sc_3d.size
        + m * n * jnp.dtype(preferred_element_type).itemsize,
        transcendentals=3 * m * n,
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec((tm, k), lhs_indices)] + gate_specs + up_specs,
            out_specs=pl.BlockSpec((tm, tn), out_indices),
            grid=(tiles_n, num_active_tiles),
            scratch_shapes=[
                pltpu.VMEM((tm, tn), jnp.float32),
                pltpu.VMEM((tm, tn), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        cost_estimate=cost,
    )(
        grid_metadata,
        jnp.array([0], dtype=jnp.int32),
        lhs,
        w13_packed_5d,
        w13_scales_5d,
        w13_packed_5d,
        w13_scales_5d,
    )


def moe_gmm_v6e_planar_t_b11(
    lhs: jax.Array,
    w13_pt: jax.Array,
    w13_scale_pt: jax.Array,
    w2_pt: jax.Array,
    w2_scale_pt: jax.Array,
    group_sizes: jax.Array,
    *,
    activation: str,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    """Routed MoE expert MLP over planar_t MXFP4 weights on TPU v6e.

    Runs the fused gate+up grouped matmul with the SiTU epilogue, then the
    down grouped matmul, decoding the weights in VMEM to float8_e4m3b11fnuz
    (see the module docstring). Both kernels share one row tile and one
    group-metadata build. Routed batches of up to `_MOE_GMM_DEEP_MAX_ROWS`
    rows use the deep schedule; larger (prefill) batches use the rolled one
    (bitwise identical).

    Args:
      lhs: `[M, K]` sorted routed activations (bf16).
      w13_pt: `[E, K, N]` uint8 from `repack_w13_planar_t`.
      w13_scale_pt: `[E, 2*(K/32), 1, N]` or `[E, 2*(K/32), N]` uint8.
      w2_pt: `[E, N/2, K]` uint8 from `repack_w2_planar_t`.
      w2_scale_pt: `[E, N/32, 1, K]` or `[E, N/32, K]` uint8.
      group_sizes: `[E]` int32 local expert row counts.
      activation: expert activation (`'situ:<beta>:<linear_beta>'` or `'silu'`).
      preferred_element_type: output dtype (defaults to `lhs.dtype`).

    Returns:
      `[M, K]` expert outputs in `preferred_element_type`.
    """
    orig_m, k = lhs.shape
    num_groups, _, n = w13_pt.shape
    out_dtype = (
        lhs.dtype
        if preferred_element_type is None
        else jnp.dtype(preferred_element_type)
    )

    w13_sc_3d = _as_rank3_scales(w13_scale_pt)
    w2_sc_3d = _as_rank3_scales(w2_scale_pt)

    situ_beta, situ_linear_beta = parse_situ_activation(activation)

    # Pad M to a multiple of 32 (or 128 for large prefill) so tm divides it.
    align_m = 32 if orig_m <= _MOE_GMM_DEEP_MAX_ROWS else 128
    padded_m = ((orig_m + align_m - 1) // align_m) * align_m
    if padded_m != orig_m:
        lhs = jnp.pad(lhs, ((0, padded_m - orig_m), (0, 0)))

    expected_rows = max(32, -(-padded_m // _MOE_GMM_EP_SHARDS))
    dequant = (
        DEQUANT_PLANAR_T_B11_DEEP
        if padded_m <= _MOE_GMM_DEEP_MAX_ROWS
        else DEQUANT_PLANAR_T_B11
    )

    tiling_13 = default_tiling(
        padded_m,
        k,
        n,
        lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
        rhs_pairs=2,
        expected_rows=expected_rows,
        vmem_budget=V6E_ROLLED_VMEM_BUDGET,
        scoped_elem_bytes=SCOPED_ELEM_BYTES_TPU7X,
        force_chunk=PLANAR_WINDOW,
    )
    tiling_2 = default_tiling(
        padded_m,
        n,
        k,
        lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
        rhs_pairs=1,
        expected_rows=expected_rows,
        vmem_budget=V6E_ROLLED_VMEM_BUDGET,
        scoped_elem_bytes=SCOPED_ELEM_BYTES_TPU7X,
        force_chunk=PLANAR_WINDOW,
    )
    # Ensure both kernels share the same row tile tm so group_metadata is built once.
    tm_common = min(tiling_13[0], tiling_2[0], padded_m)
    tiling_13 = (tm_common, tiling_13[1], tiling_13[2])
    tiling_2 = (tm_common, tiling_2[1], tiling_2[2])

    group_metadata = _make_group_metadata(
        group_sizes,
        m=padded_m,
        tm=tm_common,
        num_groups=num_groups,
        impl="closed_form",
    )

    h = _run_fused_gate_up_5d(
        lhs,
        w13_pt,
        w13_sc_3d,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        tiling=tiling_13,
        dequant=dequant,
        group_metadata=group_metadata,
        deep_vmem_budget=V6E_DEEP_VMEM_BUDGET,
    )

    out = mxfp4_gmm(
        h,
        w2_pt,
        w2_sc_3d,
        group_sizes,
        preferred_element_type=out_dtype,
        tiling=tiling_2,
        interpret=jax.default_backend() != "tpu",
        dequant=dequant,
        deep_vmem_budget=V6E_DEEP_VMEM_BUDGET,
        group_metadata=group_metadata,
    )
    return out[:orig_m]


FUSED_DECODE_MOE_MAX_TOKENS = 64
# Speculative-decoding verify batches of 65..256 tokens run the fused
# decode kernel with a multi-tray row loop (the weight unpack stays
# outside the tray loop, so its cost is paid once per chunk regardless of
# the batch size). Callers opt in by passing
# `max_tokens=FUSED_VERIFY_MOE_MAX_TOKENS` to the eligibility check.
FUSED_VERIFY_MOE_MAX_TOKENS = 256
# Measured on v6e (7 local experts of 3584x3072, no VREG spills for
# 32/64/128/256-row trays): 128-row trays are within 4% of the best tray
# size at every M in 65..256 (M=96/128: 165/167 us vs 175/176 at 64 rows
# and 179 at 32; M=192: 203 vs 196 at 64; M=256: 204 vs 215 at 64, 325 at
# 32, 573 at 16).
FUSED_VERIFY_MOE_TRAY_M = 128
_FUSED_VERIFY_MOE_VMEM_BUDGET = 46 * 1024 * 1024
# Per-kernel VMEM limit requested by the fused decode/verify kernel.
_FUSED_DECODE_MOE_VMEM_LIMIT = 48 * 1024 * 1024


def _fused_verify_moe_vmem_bytes(m: int, k: int, n: int, num_groups: int) -> int:
    """Upper-bound VMEM estimate of the multi-tray kernel at `m` padded rows."""
    depth_w2 = 1 if n >= 3072 else 2
    weights = (
        2 * (k * n + 2 * (k // SCALE_GROUP) * n)
        + depth_w2 * (n // 2 + n // SCALE_GROUP) * k
    )
    accs = 3 * m * PLANAR_WINDOW * 4 + m * n * 2 + m * k * 4
    io = m * k * 2 + num_groups * m * _LANES * 4 + m * k * 2
    return weights + accs + io


def can_use_fused_decode_moe_v6e_planar_t_b11(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    *,
    rhs_quant_dtype: jnp.dtype | None = None,
    max_tokens: int = FUSED_DECODE_MOE_MAX_TOKENS,
) -> bool:
    """Whether `fused_decode_moe_v6e_planar_t_b11` serves these operands.

    Requires planar_t uint8 weights and scales (512-aligned K and N), no
    biases, no logical quant dtype, and an expert slab that fits VMEM.
    `max_tokens` above `FUSED_DECODE_MOE_MAX_TOKENS` (at most
    `FUSED_VERIFY_MOE_MAX_TOKENS`) admits the multi-tray verify variant,
    subject to a VMEM estimate at the padded row count.

    Args:
      hidden_states: `[M, K]` activations.
      w1: `[E, K, N]` gate+up weights.
      w2: `[E, N/2, K]` down weights.
      w1_scale: gate+up scales, or None.
      w2_scale: down scales, or None.
      w1_bias: gate+up bias, or None.
      w2_bias: down bias, or None.
      rhs_quant_dtype: logical weight dtype of packed carriers, or None.
      max_tokens: largest batch the caller wants served by this kernel.

    Returns:
      Whether the fused kernel is eligible.
    """
    max_tokens = min(int(max_tokens), FUSED_VERIFY_MOE_MAX_TOKENS)
    if not (0 < hidden_states.shape[0] <= max_tokens):
        return False
    if (
        w1.dtype != jnp.uint8
        or w1_scale is None
        or w1_scale.dtype != jnp.uint8
        or w2.dtype != jnp.uint8
        or w2_scale is None
        or w2_scale.dtype != jnp.uint8
        or rhs_quant_dtype is not None
        or w1_bias is not None
        or w2_bias is not None
    ):
        return False
    if w1.ndim != 3 or w2.ndim != 3:
        return False
    num_groups, k, n = w1.shape
    if hidden_states.shape[1] != k or k % PLANAR_WINDOW != 0 or n % PLANAR_WINDOW != 0:
        return False
    if w2.shape != (num_groups, n // 2, k):
        return False
    slab_bytes = 2 * (k + 2 * (k // SCALE_GROUP)) * n + (n // 2 + n // SCALE_GROUP) * k
    if slab_bytes > 30 * 1024 * 1024:
        return False
    m = hidden_states.shape[0]
    if m > FUSED_DECODE_MOE_MAX_TOKENS:
        tray_m = FUSED_VERIFY_MOE_TRAY_M
        padded_m = ((m + tray_m - 1) // tray_m) * tray_m
        return (
            _fused_verify_moe_vmem_bytes(padded_m, k, n, num_groups)
            <= _FUSED_VERIFY_MOE_VMEM_BUDGET
        )
    return True


def fused_decode_moe_v6e_planar_t_b11(
    hidden_states: jax.Array,
    w13_pt: jax.Array,
    w13_scale_pt: jax.Array,
    w2_pt: jax.Array,
    w2_scale_pt: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    *,
    activation: str,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    token_trays_mode: str = "off",
    rank3_scales: bool = False,
) -> jax.Array:
    """Single-kernel routed MoE on TPU v6e for small token batches.

    Replaces token sorting, the SparseCore gather/reduce and the
    intermediate `[M*topk, N]` HBM round trips: whole contiguous expert slabs
    stream into VMEM (`w13[g]` double-buffered, `w2[g]` streamed during the
    `w13[g]` compute) and one Pallas kernel computes gate+up, SiTU, down and
    the routing-weighted combine for every active expert.

    For speculative-decoding verify batches
    (`FUSED_DECODE_MOE_MAX_TOKENS < M <= FUSED_VERIFY_MOE_MAX_TOKENS`) the
    row loop always runs in `FUSED_VERIFY_MOE_TRAY_M`-row trays and
    `token_trays_mode` is ignored.

    Args:
      hidden_states: `[M, K]` activations.
      w13_pt: `[E, K, N]` uint8 from `repack_w13_planar_t`.
      w13_scale_pt: `[E, 2*(K/32), 1, N]` or `[E, 2*(K/32), N]` uint8.
      w2_pt: `[E, N/2, K]` uint8 from `repack_w2_planar_t`.
      w2_scale_pt: `[E, N/32, 1, K]` or `[E, N/32, K]` uint8.
      topk_weights: `[M, topk]` routing weights.
      topk_ids: `[M, topk]` int32 local expert ids (-1 or out of range for
        assignments this shard does not own).
      activation: expert activation (`'situ:<beta>:<linear_beta>'` or `'silu'`).
      preferred_element_type: output dtype (defaults to the activation dtype).
      token_trays_mode: row loop for `M <= FUSED_DECODE_MOE_MAX_TOKENS`: 'off'
        (one tile), 'trays' (16-row trays), 'trays2' (32-row trays from 64
        rows) or 'trays32' (16-row trays above 16 rows).
      rank3_scales: read the scales through rank-3 VMEM buffers when their row
        count is a multiple of 32.

    Returns:
      `[M, K]` combined expert outputs.

    Raises:
      ValueError: for more than `FUSED_VERIFY_MOE_MAX_TOKENS` tokens or an
        unknown `token_trays_mode`.
    """
    orig_m, k = hidden_states.shape
    num_groups, _, n = w13_pt.shape
    out_dtype = (
        hidden_states.dtype
        if preferred_element_type is None
        else jnp.dtype(preferred_element_type)
    )

    w13_sc_3d = _as_rank3_scales(w13_scale_pt)
    w2_sc_3d = _as_rank3_scales(w2_scale_pt)

    situ_beta, situ_linear_beta = parse_situ_activation(activation)

    trays_norm = (token_trays_mode or "off").strip().lower()
    if orig_m > FUSED_DECODE_MOE_MAX_TOKENS:
        if orig_m > FUSED_VERIFY_MOE_MAX_TOKENS:
            raise ValueError(
                f"fused_decode_moe_v6e_planar_t_b11 supports at most "
                f"{FUSED_VERIFY_MOE_MAX_TOKENS} tokens, got {orig_m}."
            )
        tray_m = FUSED_VERIFY_MOE_TRAY_M
        use_trays = True
        padded_m = ((orig_m + tray_m - 1) // tray_m) * tray_m
        num_trays = padded_m // tray_m
    elif trays_norm == "off":
        use_trays = False
        padded_m = max(16, ((orig_m + 7) // 8) * 8)
        tray_m = padded_m
        num_trays = 1
    elif trays_norm == "trays":
        use_trays = True
        padded_m = max(16, ((orig_m + 15) // 16) * 16)
        tray_m = 16
        num_trays = padded_m // tray_m
    elif trays_norm == "trays2":
        use_trays = True
        padded_m = max(16, ((orig_m + 15) // 16) * 16)
        tray_m = 32 if padded_m >= 64 else 16
        num_trays = padded_m // tray_m
    elif trays_norm == "trays32":
        if orig_m <= 16:
            use_trays = False
            padded_m = 16
            tray_m = 16
            num_trays = 1
        else:
            use_trays = True
            padded_m = ((orig_m + 15) // 16) * 16
            tray_m = 16
            num_trays = padded_m // tray_m
    else:
        raise ValueError(
            f"Invalid token_trays_mode={token_trays_mode!r}; expected 'off', "
            "'trays', 'trays2', or 'trays32'."
        )
    if padded_m != orig_m:
        lhs = jnp.pad(hidden_states, ((0, padded_m - orig_m), (0, 0)))
        topk_w = jnp.pad(topk_weights, ((0, padded_m - orig_m), (0, 0)))
        topk_i = jnp.pad(topk_ids, ((0, padded_m - orig_m), (0, 0)), constant_values=-1)
    else:
        lhs = hidden_states
        topk_w = topk_weights
        topk_i = topk_ids

    topk_w_f32 = jnp.where(
        jnp.isfinite(topk_w), topk_w.astype(lhs.dtype).astype(jnp.float32), 0.0
    )
    row_valid = jnp.any(
        (topk_i >= 0) & (topk_i < num_groups) & (topk_w_f32 != 0.0),
        axis=-1,
        keepdims=True,
    )
    lhs = jnp.where(row_valid & jnp.isfinite(lhs), lhs, 0)

    e_ids = jnp.arange(num_groups, dtype=jnp.int32)[:, None, None]
    match = (topk_i[None, :, :] == e_ids) & (topk_w_f32[None, :, :] != 0.0)
    w_em = jnp.sum(jnp.where(match, topk_w_f32[None, :, :], 0.0), axis=2)
    active_mask = jnp.any(w_em != 0.0, axis=1)
    num_active = jnp.sum(active_mask.astype(jnp.int32))[None]
    active_ids = jnp.argsort(~active_mask).astype(jnp.int32)
    w_expert = jnp.broadcast_to(w_em[:, :, None], (num_groups, padded_m, _LANES))

    chunk = PLANAR_WINDOW
    tn13 = PLANAR_WINDOW
    tn2 = PLANAR_WINDOW
    steps_13 = k // chunk
    steps_2 = n // chunk
    tiles_n13 = n // tn13
    tiles_n2 = k // tn2
    plane_rows = chunk // 2
    scale_rows = chunk // SCALE_GROUP
    depth_w13 = 2
    depth_w2 = 1 if n >= 3072 else 2

    use_rank3_w13 = bool(rank3_scales and (w13_sc_3d.shape[1] % 32 == 0))
    use_rank3_w2 = bool(rank3_scales and (w2_sc_3d.shape[1] % 32 == 0))

    w13_packed_5d = w13_pt.reshape(num_groups, 2, steps_13, plane_rows, n)
    w13_scales_in = (
        w13_sc_3d
        if use_rank3_w13
        else w13_sc_3d.reshape(num_groups, 2, steps_13, scale_rows, n)
    )
    w2_packed_4d = w2_pt.reshape(num_groups, steps_2, plane_rows, k)
    w2_scales_in = (
        w2_sc_3d
        if use_rank3_w2
        else w2_sc_3d.reshape(num_groups, steps_2, scale_rows, k)
    )
    w13_sbuf_shape = (
        (depth_w13, w13_sc_3d.shape[1], n)
        if use_rank3_w13
        else (depth_w13, 2, steps_13, scale_rows, n)
    )
    w2_sbuf_shape = (
        (depth_w2, w2_sc_3d.shape[1], k)
        if use_rank3_w2
        else (depth_w2, steps_2, scale_rows, k)
    )

    fold_chunk = _planar_t_fold_chunk(DEQUANT_PLANAR_T_B11)
    dims = (((1,), (0,)), ((), ()))

    def kernel(
        active_ids_ref,
        nat_ref,
        lhs_ref,
        w_expert_ref,
        w13_p_ref,
        w13_s_ref,
        w2_p_ref,
        w2_s_ref,
        out_ref,
        w13_pbuf,
        w13_sbuf,
        w2_pbuf,
        w2_sbuf,
        gate_acc,
        up_acc,
        w2_acc,
        h_buf,
        out_acc,
        w13_psem,
        w13_ssem,
        w2_psem,
        w2_ssem,
    ):
        if use_trays:
            for tr in range(num_trays):
                r_sl = pl.ds(tr * tray_m, tray_m)
                for t2_init in range(tiles_n2):
                    out_acc[r_sl, pl.ds(t2_init * tn2, tn2)] = jnp.zeros(
                        (tray_m, tn2), jnp.float32
                    )
            lhs_tile = None
        else:
            out_acc[...] = jnp.zeros_like(out_acc)
            lhs_tile = lhs_ref[...]

        def get_w13_scale(slot, gate_or_up, cs, sl):
            if use_rank3_w13:
                r0 = (gate_or_up * steps_13 + cs) * scale_rows
                tile_r0 = (r0 // 32) * 32
                sub_r0 = r0 - tile_r0
                tile32 = w13_sbuf[slot, pl.ds(tile_r0, 32), sl]
                return tile32[sub_r0 : sub_r0 + scale_rows, :]
            return w13_sbuf[slot, gate_or_up, cs, :, sl]

        def get_w2_scale(slot2, cs, sl2):
            if use_rank3_w2:
                r0 = cs * scale_rows
                tile_r0 = (r0 // 32) * 32
                sub_r0 = r0 - tile_r0
                tile32 = w2_sbuf[slot2, pl.ds(tile_r0, 32), sl2]
                return tile32[sub_r0 : sub_r0 + scale_rows, :]
            return w2_sbuf[slot2, cs, :, sl2]

        def copy_w13_expert(g, slot):
            return (
                pltpu.make_async_copy(
                    w13_p_ref.at[g], w13_pbuf.at[slot], w13_psem.at[slot]
                ),
                pltpu.make_async_copy(
                    w13_s_ref.at[g], w13_sbuf.at[slot], w13_ssem.at[slot]
                ),
            )

        def copy_w2_expert(g, slot=0):
            return (
                pltpu.make_async_copy(
                    w2_p_ref.at[g], w2_pbuf.at[slot], w2_psem.at[slot]
                ),
                pltpu.make_async_copy(
                    w2_s_ref.at[g], w2_sbuf.at[slot], w2_ssem.at[slot]
                ),
            )

        @pl.when(nat_ref[0] > 0)
        def _prime_first():
            g0 = active_ids_ref[0]
            for c in copy_w13_expert(g0, 0):
                c.start()
            if depth_w2 >= 2:
                for c2 in copy_w2_expert(g0, 0):
                    c2.start()

        def visit_expert(j, carry):
            g = active_ids_ref[j]
            slot = lax.rem(j, depth_w13)
            next_slot = lax.rem(j + 1, depth_w13)
            slot2 = lax.rem(j, depth_w2)
            next_slot2 = lax.rem(j + 1, depth_w2)

            if depth_w2 == 1:
                for c2 in copy_w2_expert(g, 0):
                    c2.start()
            else:

                @pl.when(j + 1 < nat_ref[0])
                def _prefetch_next_w2():
                    g_next = active_ids_ref[j + 1]
                    for c2 in copy_w2_expert(g_next, next_slot2):
                        c2.start()

            @pl.when(j + 1 < nat_ref[0])
            def _prefetch_next():
                g_next = active_ids_ref[j + 1]
                for c in copy_w13_expert(g_next, next_slot):
                    c.start()

            for c in copy_w13_expert(g, slot):
                c.wait()

            if use_trays:
                for t in range(tiles_n13):
                    sl = pl.ds(t * tn13, tn13)
                    for tr in range(num_trays):
                        r_sl = pl.ds(tr * tray_m, tray_m)
                        gate_acc[r_sl, :] = jnp.zeros((tray_m, tn13), jnp.float32)
                        up_acc[r_sl, :] = jnp.zeros((tray_m, tn13), jnp.float32)
                    for cs in range(steps_13):
                        k_sl = pl.ds(cs * chunk, chunk)
                        # One weight unpack per chunk, shared by all trays.
                        g_codes, g_wscale = fold_chunk(
                            w13_pbuf[slot, 0, cs, :, sl], get_w13_scale(slot, 0, cs, sl)
                        )
                        for tr in range(num_trays):
                            r_sl = pl.ds(tr * tray_m, tray_m)
                            lhs_tr = lhs_ref[r_sl, k_sl]
                            gate_acc[r_sl, :] += (
                                lax.dot_general(
                                    lhs_tr,
                                    g_codes,
                                    dimension_numbers=dims,
                                    preferred_element_type=jnp.float32,
                                )
                                * g_wscale
                            )
                        u_codes, u_wscale = fold_chunk(
                            w13_pbuf[slot, 1, cs, :, sl], get_w13_scale(slot, 1, cs, sl)
                        )
                        for tr in range(num_trays):
                            r_sl = pl.ds(tr * tray_m, tray_m)
                            lhs_tr = lhs_ref[r_sl, k_sl]
                            up_acc[r_sl, :] += (
                                lax.dot_general(
                                    lhs_tr,
                                    u_codes,
                                    dimension_numbers=dims,
                                    preferred_element_type=jnp.float32,
                                )
                                * u_wscale
                            )
                    for tr in range(num_trays):
                        r_sl = pl.ds(tr * tray_m, tray_m)
                        h_buf[r_sl, sl] = _situ_f32(
                            gate_acc[r_sl, :],
                            up_acc[r_sl, :],
                            beta=situ_beta,
                            linear_beta=situ_linear_beta,
                        ).astype(out_dtype)

                for c2 in copy_w2_expert(g, slot2):
                    c2.wait()

                for t2 in range(tiles_n2):
                    sl2 = pl.ds(t2 * tn2, tn2)
                    for tr in range(num_trays):
                        w2_acc[pl.ds(tr * tray_m, tray_m), :] = jnp.zeros(
                            (tray_m, tn2), jnp.float32
                        )
                    for cs in range(steps_2):
                        n_sl = pl.ds(cs * chunk, chunk)
                        # One weight unpack per chunk, shared by all trays.
                        codes2, wscale2 = fold_chunk(
                            w2_pbuf[slot2, cs, :, sl2], get_w2_scale(slot2, cs, sl2)
                        )
                        for tr in range(num_trays):
                            r_sl = pl.ds(tr * tray_m, tray_m)
                            h_tr = h_buf[r_sl, n_sl]
                            w2_acc[r_sl, :] += (
                                lax.dot_general(
                                    h_tr,
                                    codes2,
                                    dimension_numbers=dims,
                                    preferred_element_type=jnp.float32,
                                )
                                * wscale2
                            )
                    for tr in range(num_trays):
                        r_sl = pl.ds(tr * tray_m, tray_m)
                        down_bf16 = (
                            w2_acc[r_sl, :].astype(out_dtype).astype(jnp.float32)
                        )
                        w_g_tr = jnp.tile(w_expert_ref[g, r_sl, :], (1, tn2 // _LANES))
                        out_acc[r_sl, sl2] += down_bf16 * w_g_tr
            else:
                # Narrow the Optional for the type checker; the non-tray
                # prologue always loads it.
                tile = lhs_tile
                assert tile is not None
                w_g = w_expert_ref[g]
                for t in range(tiles_n13):
                    sl = pl.ds(t * tn13, tn13)
                    gate_acc[...] = jnp.zeros_like(gate_acc)
                    up_acc[...] = jnp.zeros_like(up_acc)
                    for cs in range(steps_13):
                        lhs_chunk = tile[:, cs * chunk : (cs + 1) * chunk]
                        g_codes, g_wscale = fold_chunk(
                            w13_pbuf[slot, 0, cs, :, sl], get_w13_scale(slot, 0, cs, sl)
                        )
                        gate_acc[...] += (
                            lax.dot_general(
                                lhs_chunk,
                                g_codes,
                                dimension_numbers=dims,
                                preferred_element_type=jnp.float32,
                            )
                            * g_wscale
                        )
                        u_codes, u_wscale = fold_chunk(
                            w13_pbuf[slot, 1, cs, :, sl], get_w13_scale(slot, 1, cs, sl)
                        )
                        up_acc[...] += (
                            lax.dot_general(
                                lhs_chunk,
                                u_codes,
                                dimension_numbers=dims,
                                preferred_element_type=jnp.float32,
                            )
                            * u_wscale
                        )
                    h_tile = _situ_f32(
                        gate_acc[...],
                        up_acc[...],
                        beta=situ_beta,
                        linear_beta=situ_linear_beta,
                    ).astype(out_dtype)
                    h_buf[:, sl] = h_tile

                for c2 in copy_w2_expert(g, slot2):
                    c2.wait()

                h_full = h_buf[...]
                w_g_tile = jnp.tile(w_g, (1, tn2 // _LANES))
                for t2 in range(tiles_n2):
                    sl2 = pl.ds(t2 * tn2, tn2)
                    w2_acc[...] = jnp.zeros_like(w2_acc)
                    for cs in range(steps_2):
                        h_chunk = h_full[:, cs * chunk : (cs + 1) * chunk]
                        codes2, wscale2 = fold_chunk(
                            w2_pbuf[slot2, cs, :, sl2], get_w2_scale(slot2, cs, sl2)
                        )
                        w2_acc[...] += (
                            lax.dot_general(
                                h_chunk,
                                codes2,
                                dimension_numbers=dims,
                                preferred_element_type=jnp.float32,
                            )
                            * wscale2
                        )
                    down_bf16 = w2_acc[...].astype(out_dtype).astype(jnp.float32)
                    out_acc[:, sl2] += down_bf16 * w_g_tile

            return carry

        lax.fori_loop(0, nat_ref[0], visit_expert, 0)
        if use_trays:
            for tr in range(num_trays):
                r_sl = pl.ds(tr * tray_m, tray_m)
                for t2_ep in range(tiles_n2):
                    sl2_ep = pl.ds(t2_ep * tn2, tn2)
                    out_ref[r_sl, sl2_ep] = out_acc[r_sl, sl2_ep].astype(out_dtype)
        else:
            out_ref[...] = out_acc[...].astype(out_dtype)

    scratch_shapes = [
        pltpu.VMEM((depth_w13, 2, steps_13, plane_rows, n), jnp.uint8),
        pltpu.VMEM(w13_sbuf_shape, jnp.uint8),
        pltpu.VMEM((depth_w2, steps_2, plane_rows, k), jnp.uint8),
        pltpu.VMEM(w2_sbuf_shape, jnp.uint8),
        pltpu.VMEM((padded_m, tn13), jnp.float32),
        pltpu.VMEM((padded_m, tn13), jnp.float32),
        pltpu.VMEM((padded_m, tn2), jnp.float32),
        pltpu.VMEM((padded_m, n), out_dtype),
        pltpu.VMEM((padded_m, k), jnp.float32),
        pltpu.SemaphoreType.DMA((depth_w13,)),
        pltpu.SemaphoreType.DMA((depth_w13,)),
        pltpu.SemaphoreType.DMA((depth_w2,)),
        pltpu.SemaphoreType.DMA((depth_w2,)),
    ]

    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=_FUSED_DECODE_MOE_VMEM_LIMIT
    )

    kname = f"fused_decode_moe_v6e_planar_t_b11_m{padded_m}"
    if use_trays:
        kname += f"_trays{num_trays}x{tray_m}"
    if use_rank3_w13 or use_rank3_w2:
        kname += "_s3d"

    out = pl.pallas_call(
        kernel,
        grid=(),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((padded_m, k), out_dtype),
        scratch_shapes=scratch_shapes,
        compiler_params=compiler_params,
        name=kname,
    )(
        active_ids,
        num_active,
        lhs,
        w_expert,
        w13_packed_5d,
        w13_scales_in,
        w2_packed_4d,
        w2_scales_in,
    )
    return out[:orig_m]
