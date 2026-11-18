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
most B11_FOLD_WINDOW octaves.

Entry points:
  * `moe_gmm_v6e_planar_t_b11`: sorted routed tokens.

The grid metadata is adapted from `make_group_metadata` in
`jax.experimental.pallas.ops.tpu.megablox.gmm` (JAX, Apache-2.0).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import situ_and_mul

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

# These kernels serve TPU v6e only (`is_supported_by_planar_t_b11`). Their
# tiling and VMEM models read the v6e table rather than
# `pltpu.get_tpu_info()` because they also run off TPU (interpret mode, CPU
# tests), where the device query is unavailable.
_V6E_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)

# TPU lane count: a Pallas block's minor dim must be a multiple of it
# (or the whole array dim).
_LANES = _V6E_INFO.num_lanes


def _round_up(x: int, multiple: int) -> int:
    """Smallest multiple of `multiple` that is >= `x`."""
    return ((x + multiple - 1) // multiple) * multiple


# Mosaic's default per-kernel stack ("scoped vmem") limit, from
# --xla_tpu_scoped_vmem_limit_kib=32768 (1/4 of v6e's 128 MiB VMEM). A kernel
# whose live temporaries exceed it fails the whole compile with
# CompileTimeScopedVmemOom. Raising it is a PROGRAM-WIDE flag flip
# that takes the same bytes away from MSA for every other op in the
# serving step, so the tiling search fits under the default instead.
_SCOPED_VMEM_LIMIT = _V6E_INFO.vmem_capacity_bytes // 4


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
# of the dequant chain against `_SCOPED_VMEM_LIMIT`, calibrated against
# measured compiler output (~26.6 bytes per element, rounded up to 28).
SCOPED_ELEM_BYTES_B11 = 28


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
    *,
    lhs_bytes: int,
    rhs_pairs: int,
    expected_rows: int | None,
    vmem_budget: int,
    scoped_elem_bytes: int,
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
      vmem_budget: per-grid-step VMEM bound in bytes.
      scoped_elem_bytes: scoped-stack bytes per dequant element
        (`SCOPED_ELEM_BYTES_B11`).
      force_chunk: pin the k chunk to this value when it is a valid candidate
        (planar weights require chunk == PLANAR_WINDOW). When the shape cannot
        serve it the normal search runs instead and `mxfp4_gmm` rejects the
        result.

    Returns:
      `(tm, k_chunk, tn)`.

    Raises:
      ValueError: if even the smallest valid tiling exceeds the budget.
    """
    budget = vmem_budget
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


def _make_group_metadata(
    group_sizes: jax.Array,
    *,
    m: int,
    tm: int,
    num_groups: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Grid metadata for the grouped matmul, as loop-free arithmetic.

    Adapted from `make_group_metadata` in
    `jax.experimental.pallas.ops.tpu.megablox.gmm` (JAX, Apache-2.0),
    specialized to a single unsharded group range and
    `visit_empty_groups=False`, which is all this caller needs. No while
    loop, no scatter and no gather, so it is cheap to build once per layer
    and share between the gate+up and down kernels.

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
      `((group_offsets, group_ids, m_tile_ids), num_active_tiles)` where
      `group_offsets[i]` is the first row of group i, and `group_ids[j]` /
      `m_tile_ids[j]` say which group and which m tile grid step j works
      on. A tile straddling a group boundary appears once per group it
      touches, which is why the grid is `tiles_m + num_groups - 1` long.
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


#: A precomputed `_make_group_metadata` result, as the kernels take it.
GroupMetadata = tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]


def _planar_repack_windows(packed: jax.Array, window: int) -> jax.Array:
    """`[..., K/2]` interleaved nibble pairs -> planar window bytes."""
    half_k = packed.shape[-1]
    k = 2 * half_k
    hw = window // 2
    lead = packed.shape[:-1]
    pairs = packed.reshape(lead + (k // window, hw))
    # Nibbles back into original contraction order, per window.
    codes = jnp.stack([pairs & 0x0F, pairs >> 4], axis=-1).reshape(
        lead + (k // window, window)
    )
    lo = codes[..., :hw]
    hi = codes[..., hw:]
    return (lo | (hi << 4)).astype(packed.dtype).reshape(lead + (half_k,))


def repack_planar_stack(packed: jax.Array, window: int = PLANAR_WINDOW) -> jax.Array:
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

    Trace-safe (weight loading applies this to TPU tensors through
    `pallas.jax_op`): a deterministic byte permutation that branches on
    shape/dtype only, never on values.

    Args:
      packed: `[..., K/2]` uint8 nibble pairs in checkpoint (interleaved) order.
        K must divide by `window`.
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
    return _planar_repack_windows(packed, window)


def repack_planar_t_stack(packed: jax.Array, window: int = PLANAR_WINDOW) -> jax.Array:
    """Planar window bytes, transposed: `[..., N, K/2]` -> `[..., K/2, N]`.

    The planar_t layout is `repack_planar_stack` followed by a swap of the
    last two dims, so byte `[..., w*window/2 + j, n]` holds the nibbles of
    contraction elements `w*window + j` (low) and `w*window + window/2 + j`
    (high) of weight row `n`. Same weight values bit for bit; only placement
    changes. The decoded tile then comes out `[chunk, tn]` with the weight
    row on the lane dim, so the per-group scale varies along lanes and the
    decode table can be built per lane (`_dequantize_chunk_planar_t_b11`).
    """
    return jnp.swapaxes(repack_planar_stack(packed, window), -1, -2)


def transpose_scales_planar_t(scales: jax.Array) -> jax.Array:
    """`[..., N, K/32]` e8m0 codes -> `[..., K/32, N]`, the planar_t order.

    Scales carry no nibble structure, so their planar_t form is a plain
    transpose of the last two dims (group g still owns contraction
    elements [32g, 32g + 32)). Self-inverse modulo the axis swap: apply
    it again to get back. Trace-safe like the packed repack.
    """
    return jnp.swapaxes(scales, -1, -2)


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
# E = 10, leaving 9 octaves of headroom. The weight loader does not check
# the spread (a real checkpoint's 99th-percentile spread is 2).
B11_FOLD_WINDOW = 9


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


def _situ_f32(
    gate: jax.Array,
    up: jax.Array,
    *,
    beta: float,
    linear_beta: float | None,
) -> jax.Array:
    """SiTU-GLU (or SwiGLU when `beta == 0.0`) in f32.

    `[beta * tanh(gate / beta) * sigmoid(gate)] * [linear_beta * tanh(up /
    linear_beta)]` (up stays linear when `linear_beta` is None). `beta == 0.0`
    selects plain SiLU-and-multiply (`parse_situ_activation` maps `'silu'` to
    it). Shared by the kernels and the fused reference so on-TPU and off-TPU
    paths run the same arithmetic.
    """
    g = gate.astype(jnp.float32)
    u = up.astype(jnp.float32)
    if beta == 0.0:
        return jax.nn.silu(g) * u
    return situ_and_mul(g, u, beta, linear_beta)


def _group_row_mask(
    shape: tuple[int, ...], *, row0: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Rows of a `[tm, tn]` output tile that belong to the visited group.

    A row tile straddling a group boundary is visited once per group it
    touches; each visit may write only the rows in `[start, end)` and must
    keep what earlier visits of the same tile wrote (or start from zero on
    the tile's first visit, `start <= row0`). Rows outside every group are
    left unspecified.
    """
    row = lax.broadcasted_iota(jnp.int32, shape, 0) + row0
    return jnp.logical_and(row >= start, row < end)


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


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "interpret",
    ],
)
def mxfp4_gmm(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, int, int],
    group_metadata: GroupMetadata,
    preferred_element_type: jax.typing.DTypeLike = jnp.bfloat16,
    interpret: bool = False,
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
        PLANAR_WINDOW. `tm` must divide M.
      group_metadata: grid metadata (`_make_group_metadata` with the same
        `(m, tm, num_groups)`), so the gate+up and down kernels of one layer
        share one build.
      interpret: run the Pallas kernel in interpret mode (used off TPU).

    Returns:
      `[M, N]` products.
    """
    m, k = lhs.shape
    num_groups = packed.shape[0]
    n = packed.shape[2]
    # Clamp before validating: a tile larger than the array is just the
    # whole array, which toy shapes rely on.
    tm, chunk, tn = tiling
    tm, chunk, tn = min(tm, m), min(chunk, k), min(tn, n)
    if chunk != PLANAR_WINDOW:
        raise ValueError(
            f"planar weights are repacked per {PLANAR_WINDOW}-element window,"
            f" so the kernel k chunk must equal it; got chunk {chunk} for k"
            f" {k}."
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

    grid_metadata, num_active_tiles = group_metadata

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
            codes, wscale = _dequantize_chunk_planar_t_b11(packed_ref[i], scales_ref[i])
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

        lax.fori_loop(0, k // chunk, chunk_body_t, 0)

        # Only the rows of the tile that belong to THIS group may be
        # written; a tile straddling a group boundary is visited once per
        # group it touches, and rows outside every group are left alone.
        group_id = group_ids[grid_id]
        start = group_offsets[group_id]
        end = group_offsets[group_id + 1]
        row0 = m_tile_ids[grid_id] * tm
        mask = _group_row_mask(acc.shape, row0=row0, start=start, end=end)
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


# ============================================================================
# Served entry points (TPU v6e)
# ============================================================================

# Per-grid-step VMEM budget of the served entry point: half of the 32 MiB
# default scoped VMEM limit.
V6E_ROLLED_VMEM_BUDGET = _SCOPED_VMEM_LIMIT // 2
# Routed-row padding of `moe_gmm_v6e_planar_t_b11`.
_MOE_GMM_ROW_ALIGN = 128
# Expert-parallel shard count assumed when sizing the row tile: each
# shard expects to own about `padded_m / _MOE_GMM_EP_SHARDS` of the routed
# rows (`default_tiling(expected_rows=...)`). A fixed assumption, not read
# from the deployment: it only steers the row tile, every value is correct.
_MOE_GMM_EP_SHARDS = 8


def _as_rank3_scales(scale: jax.Array) -> jax.Array:
    """Drops the singleton axis 2 of `[E, rows, 1, N]` scales (rank-3 as is)."""
    return jnp.squeeze(scale, axis=2) if scale.ndim == 4 else scale


def is_planar_t_b11_layout(
    w1: jax.Array, w1_scale: jax.Array | None, *, rhs_quant_dtype: jnp.dtype | None
) -> bool:
    """Whether MoE weights are in the planar_t MXFP4 layout this module serves.

    The planar_t loader (`repack_w13_planar_t` / `repack_w2_planar_t`) is the
    only one that stores uint8 weights with uint8 e8m0 scales and no logical
    quant dtype. Such weights cannot be read by any other MoE kernel, so a
    caller that sees this layout must use `moe_gmm_v6e_planar_t_b11`.
    """
    return (
        w1.dtype == jnp.uint8
        and w1_scale is not None
        and w1_scale.dtype == jnp.uint8
        and rhs_quant_dtype is None
    )


# The smallest row and column tiles `default_tiling` falls back to for the
# row counts `moe_gmm_v6e_planar_t_b11` pads to and PLANAR_WINDOW-aligned
# widths.
_SMALLEST_ROW_TILE = 8
_SMALLEST_COL_TILE = 128


def _smallest_tiling_fits(k: int, rhs_pairs: int) -> bool:
    """Whether `moe_gmm_v6e_planar_t_b11` finds a tiling for contraction `k`.

    `default_tiling` raises when its smallest candidate exceeds the budget;
    this is that candidate for bf16 activations.
    """
    return (
        _vmem_bytes(
            _SMALLEST_ROW_TILE, k, _SMALLEST_COL_TILE, PLANAR_WINDOW, 2, rhs_pairs
        )
        <= V6E_ROLLED_VMEM_BUDGET
    )


def is_supported_by_planar_t_b11(
    hidden: int, inter: int, *, chip_version: pltpu.ChipVersion, activation: str
) -> bool:
    """Whether an MXFP4 MoE layer can be served by this module's kernels.

    A layer this rejects keeps the loader's default MXFP4 layout and kernels.

    Envelope: the kernels have been compiled and run on TPU v6e for
    `hidden` 3584 x `inter` 3072 (7 experts/shard) only. Other admitted widths rest
    on the VMEM models, not on a measurement:
      * the grouped kernel needs a `default_tiling` under
        `V6E_ROLLED_VMEM_BUDGET`, which bounds both widths (checked here, so
        a wider layer is rejected instead of failing to trace);
      * the fused decode kernel has its own gate
        (`can_use_fused_decode_moe_v6e_planar_t_b11`: slab bytes and a VMEM
        estimate) and falls back to the grouped kernel.

    Args:
      hidden: model (contraction) width of the gate+up projection.
      inter: expert intermediate width.
      chip_version: TPU generation. Compared directly because the decode
        relies on TPU v6e's hardware float8_e4m3b11fnuz expansion and all
        tables here are the v6e ones; chips that decode FP4 on the MXU keep
        the default layout.
      activation: expert activation string (see `parse_situ_activation`).

    Returns:
      True for TPU v6e, PLANAR_WINDOW-aligned widths that fit the grouped
      kernel's smallest tiling, and a SiTU or SiLU activation.
    """
    return (
        chip_version == pltpu.ChipVersion.TPU_V6E
        and hidden % PLANAR_WINDOW == 0
        and inter % PLANAR_WINDOW == 0
        and _smallest_tiling_fits(hidden, rhs_pairs=2)
        and _smallest_tiling_fits(inter, rhs_pairs=1)
        and (activation == "silu" or activation.startswith("situ:"))
    )


def _check_planar_t_moe_operands(
    lhs: jax.Array,
    w13_pt: jax.Array,
    w13_sc_3d: jax.Array,
    w2_pt: jax.Array,
    w2_sc_3d: jax.Array,
) -> None:
    """Rejects operands outside the planar_t MoE envelope, with the shapes."""
    if w13_pt.ndim != 3 or w2_pt.ndim != 3:
        raise ValueError(
            f"planar_t weights must be rank 3, got w13 {w13_pt.shape} and"
            f" w2 {w2_pt.shape}."
        )
    num_groups, k, n = w13_pt.shape
    if (
        lhs.shape[1] != k
        or k % PLANAR_WINDOW
        or n % PLANAR_WINDOW
        or w2_pt.shape != (num_groups, n // 2, k)
        or w13_sc_3d.shape != (num_groups, 2 * (k // SCALE_GROUP), n)
        or w2_sc_3d.shape != (num_groups, n // SCALE_GROUP, k)
    ):
        raise ValueError(
            "planar_t MoE operands outside the supported envelope (K and N"
            f" multiples of {PLANAR_WINDOW}, w13 [E, K, N], w2 [E, N/2, K]):"
            f" lhs {lhs.shape}, w13 {w13_pt.shape}, w13 scales"
            f" {w13_sc_3d.shape}, w2 {w2_pt.shape}, w2 scales {w2_sc_3d.shape}."
        )


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
    assert isinstance(w2_pt, jax.Array)
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
        _, beta, linear_beta = activation.split(":")
        return float(beta), None if linear_beta == "none" else float(linear_beta)
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
    group_metadata: GroupMetadata,
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
      group_metadata: `_make_group_metadata` result for `(M, tm, E)`.

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

    grid_metadata, num_active_tiles = group_metadata
    preferred_element_type = lhs.dtype
    interpret = jax.default_backend() != "tpu"

    # Rolled planar_t_b11 kernel with 5D BlockSpec indexing (0 for gate, 1 for up)
    tiles_n = (n + tn - 1) // tn
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
                codes, wscale = _dequantize_chunk_planar_t_b11(packed_c, scales_c)
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

        lax.fori_loop(0, k // chunk, chunk_body_t, 0)

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
        mask = _group_row_mask(h.shape, row0=row0, start=start, end=end)
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
    ep_shards: int = _MOE_GMM_EP_SHARDS,
) -> jax.Array:
    """Routed MoE expert MLP over planar_t MXFP4 weights on TPU v6e.

    Runs the fused gate+up grouped matmul with the SiTU/SiLU epilogue, then the
    down grouped matmul, decoding the weights in VMEM to float8_e4m3b11fnuz
    (see the module docstring). Both kernels share one row tile and one
    group-metadata build.

    Args:
      lhs: `[M, K]` sorted routed activations (bf16).
      w13_pt: `[E, K, N]` uint8 from `repack_w13_planar_t`.
      w13_scale_pt: `[E, 2*(K/32), 1, N]` or `[E, 2*(K/32), N]` uint8.
      w2_pt: `[E, N/2, K]` uint8 from `repack_w2_planar_t`.
      w2_scale_pt: `[E, N/32, 1, K]` or `[E, N/32, K]` uint8.
      group_sizes: `[E]` int32 local expert row counts.
      activation: expert activation (`'situ:<beta>:<linear_beta>'` or `'silu'`).
      preferred_element_type: output dtype (defaults to `lhs.dtype`).
      ep_shards: expected expert-parallel shard count for row-tile sizing
        (`expected_rows = ceil(padded_m / ep_shards)`; steers tile size only,
        every positive value is numerically valid).

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
    _check_planar_t_moe_operands(lhs, w13_pt, w13_sc_3d, w2_pt, w2_sc_3d)

    situ_beta, situ_linear_beta = parse_situ_activation(activation)

    # Pad M to the row alignment so the row tile divides it.
    padded_m = _round_up(orig_m, _MOE_GMM_ROW_ALIGN)
    if padded_m != orig_m:
        lhs = jnp.pad(lhs, ((0, padded_m - orig_m), (0, 0)))

    expected_rows = max(_MIN_EXPECTED_TM, -(-padded_m // max(ep_shards, 1)))

    tiling_13 = default_tiling(
        padded_m,
        k,
        n,
        lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
        rhs_pairs=2,
        expected_rows=expected_rows,
        vmem_budget=V6E_ROLLED_VMEM_BUDGET,
        scoped_elem_bytes=SCOPED_ELEM_BYTES_B11,
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
        scoped_elem_bytes=SCOPED_ELEM_BYTES_B11,
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
    )

    h = _run_fused_gate_up_5d(
        lhs,
        w13_pt,
        w13_sc_3d,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        tiling=tiling_13,
        group_metadata=group_metadata,
    )

    out = mxfp4_gmm(
        h,
        w2_pt,
        w2_sc_3d,
        group_sizes,
        preferred_element_type=out_dtype,
        tiling=tiling_2,
        interpret=jax.default_backend() != "tpu",
        group_metadata=group_metadata,
    )
    return out[:orig_m]
