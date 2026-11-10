from __future__ import annotations
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

"""Grouped matmul over MXFP4 expert stacks, dequantized inside the kernel.

`ffw._ragged_expert_combine` runs three grouped matmuls per MoE layer
against the routed expert weights, which the checkpoint stores as
MXFP4 (u8 nibble pairs plus one e8m0 exponent per 32 contraction
elements). Dequantizing them into a dense bf16 stack before the matmul
is what makes prefill not fit: at the 8x16 geometry the chain
`u8[7, N, K/2] -> s32[7, N, K/2, 2] -> s32[7, N, K] -> f32[7, N, K] ->
bf16[7, K, N]` leaves about five 154-294 MiB transients per weight per
layer, and the compile report showed dozens of them coexisting (36.54 G
of temporaries against 31.24 G of HBM, after expert parallelism had
already removed the whole-stack all-gather).

So the dequant moves INTO the matmul: the u8 pairs stream HBM to VMEM
and each rhs TILE is decoded in registers right before the MXU
consumes it. No dense view of the weights exists at any point, which
removes that liveness by construction and cuts expert-weight read
traffic about 4x (bf16 2 bytes/weight -> mxfp4 0.5 + 1/32).

The tiling, group metadata and pipelining are megablox's
(`jax.experimental.pallas.ops.tpu.megablox.gmm`, whose
`make_group_metadata` is reproduced below because that directory has no
BUILD file and so cannot be depended on from google3); only the rhs
load stage differs. That kernel supports bf16/f32 operands ONLY, and gemax's
subchannel-quantized gmm covers int8/int4/float8_e4m3fn/float8_e5m2
with no e2m1 or e8m0 support, so neither takes MXFP4 as-is. MXFP4 also
cannot be re-expressed as int4 with a scale: e2m1's magnitudes
(0, .5, 1, 1.5, 2, 3, 4, 6) are not uniformly spaced.

No transpose is needed anywhere. The checkpoint's `[E, N, K/2]` layout
already has the contraction dim minor, so a tile is `[tn, tk/2]` and
the contraction is `dot_general` on the last dim of both operands, the
native weight-stationary MXU orientation.

The dequant arithmetic is bitwise the same as
`ffw.dequantize_mxfp4_stack_jnp` (e2m1 LUT decode, exponent-bit
constructed 2^(u8-127) scale, f32 multiply, bf16 cast), so a
checkpoint served through this kernel computes on EXACTLY the weights
the dense path would have loaded; `moe_gmm_test` pins that equality.

The routed experts' gate (w1) and up (w3) projections additionally
fuse into ONE kernel (`mxfp4_fused_gate_up_gmm`): both stacks stream
against the same lhs tiles and the SiTU-GLU combine runs on the
accumulators in registers, so only the activated `[M, ffn]` product is
ever written to HBM and the w1/w3 side pays one launch instead of two.

A separate mxfp4-NATIVE (fp8) path exists behind `native_f8=True`
(threaded from `K3MoeConfig.gmm_fp8_native`, Ghostfish only): the e2m1
nibbles RELABEL to float8_e4m3fn in registers (lossless - every e2m1
value is exactly representable in e4m3, all 16 codes bit exact) and
feed the MXU directly, with the k32 e8m0 scales applied POST-dot on
the f32 accumulator (`_subchannel_dot_f8`; `fold_scales` is the
one-deep-pass variant). HOW A bf16 LHS AND AN f8 WEIGHT MEET ON THE
MXU IS OPEN as of 2026-08-08, and this sentence previously answered it
wrongly. It cited a Mosaic canonicalisation that is the REVERSE rule:
the `tpu::ExtFOp` insert that fires on an f8 LHS against a bf16/f32
RHS and rewrites operand 0, which never matches this pair. Anchor on
that construct, not on a line number. The full contradiction, with its
three sourced sides, lives at `_subchannel_dot_f8`; do not restate it
here, and do not write a mechanism in either place until a hardware
gate answers. Two corrections this lane owns, neither recorded there:
  - THE LHS IS NOT NARROWED, at either end, so "the activations are
    secretly e4m3" is excluded and must not be re-derived. `lhs` is
    rebound 14 times in this file (AST, not grep: two are tuple
    unpackings a `lhs =` regex cannot see). Only FOUR change dtype and
    all four sit under the fp8-activation opt-in, `use_mxfp8` or
    `elif lhs_f8:`. The other ten preserve it: two `_prepare_lhs`
    k-order permutes, four `with_memory_space_constraint`, and four
    reference-fallback round trips that end in `.astype(lhs.dtype)`.
    Mosaic does not narrow it either.
  - SCOPE THE LATCH CITATION BY CALL SITE, not by function name.
    `GetGainLatchModeAndScalingFactor` has three callers and they do
    NOT agree: two lower a `tpu.matmul`, while the explicit
    `tpu::MatmulPushRhsOp` pattern REJECTS a scaling factor outright,
    so the b11 branch is a compile error there rather than a scaled
    latch; and only one of the two matmul callers forwards
    `is_rhs_sparse`, where a sparse RHS picks `...SparseE4M3Fn`
    instead. Only the DENSE gf answer is call-site-invariant.
f8e4m3fn activations
(`K3MoeConfig.gmm_fp8_activations`) run the fp8 x fp8 mode at 2x push
rate at the cost of e4m3 activation rounding; `lhs_f8_scales`
(`K3MoeConfig.gmm_fp8_act_scales`) upgrades that cast to the MXFP8
grid the model was QAT-TRAINED with - dynamic per-32-block e8m0
activation scales (`quantize_mxfp8`) folded per subchannel next to
the weight e8m0s, aligning the serve numerics with the training
forward pass and the GPU kMxfp8Dynamic path.

MEASURED VERDICT of the FIRST fp8 round
(perf/gf_fp8_gmm_bench_forge_test, real gf hardware, pre-gfdq2 base):
every fp8 arm LOST to the split_mxu dequant kernel at serving shapes,
this kernel class being VPU/elementwise bound so that the fp8 MXU
rate never binds. READ IT WITH TWO CAVEATS, both established
2026-08-08. The mechanism survives; the baseline and the direction
do not.

(1) IT NAMES A BASELINE THE SERVED GF STACK CANNOT RUN. split_mxu is
the INTERLEAVED-layout default (`ffw.dequant_mode_for`); every gf
serving preset pins `mxfp4_weight_layout='planar_t'` with
`gmm_dequant_impl='planar_t_gather'`, and `K3MoeConfig` RAISES on a
cross-family impl, so split_mxu is unreachable from those presets by
construction. And it is SLOWER than the served kernel, 1.3x-2.3x at
every shape, measured and confirmed by a POD READ of 278041926 rather
than by preset name. So a ratio quoted against split_mxu UNDERSTATES
how far an arm sits from what gf actually runs. An earlier version of
this line said "2.10x planar_t_gather", i.e. the opposite direction,
from an unverified recollection used as if it were a receipt; that was
wrong and it made the dead fp8 family look closer to parity than it
is. PRICE GF ARMS AGAINST planar_t_gather.

(2) IT PREDATES THE ROUND-2 f8-TABLE KERNEL (planar_t bytes into a
static 16-entry e4m3 gather, no in-register relabel chain), which on
GF silicon measured FASTER than planar_t_gather, not slower. So round
2 inverts the DIRECTION while the mechanism above still explains the
MAGNITUDE: single-digit percent, not the MXU-rate tier.
DO NOT QUOTE A ROUND-2 RATIO FROM HERE. The two banked pairs
(459.4 us/pair against planar_t_gather's 491.0; and 432.6 for
int4-sc512 against f8table's 478.3 in a different session) are
recorded WITHOUT their operating point: no batch, no tokens per data
shard, no m. The f8table figure differs between those two sessions
(459.4 vs 478.3), so the shapes demonstrably are not the same. That
matters more than usual here because the decode operating point is
load-bearing on gf: with

    tokens_per_data_shard = batch_size / (data_dim * data2_dim)

on the post-transform mesh, a value of 1 costs a narrow format the
attention projections (bf16 is immune on gf, not on ghostlite), and
the whole EP-64 sub-family sits at 1 by default (B32 over 32 data
shards). Do NOT divide by `devices / model_dim`: the ici mesh has
five axes, so that form over-counts the denominator by
seq * model2 * stage and reads 2.0 where the truth is 16.0 on a
seq-sharded preset.

The MLA G=1 collapse does NOT also land here, and an earlier version
of this note claiming both penalties stack is WITHDRAWN.
`mla_pallas.decode_blocks` is reached only under
`K3MlaConfig.kernel_impl` of 'auto' or 'flash'; the default is 'jax'
and no pod queried so far runs anything else, so these arms pay the
projection cliff alone. A ratio taken at an unrecorded operating
point cannot be transferred to another one. Re-establish it, and
state tokens per shard with it, before pricing.
An earlier 0.25-0.34x figure for this arm is WITHDRAWN (denominator
unnamed) and must not be re-derived from.

The knobs stay default-OFF pending the 4-way bench at ep64ol serving
shapes (split_mxu vs f8xf8 vs int4+Hadamard vs fp8 activations x
int4 subchannel-512), which is the open follow-up. Anyone updating
this verdict: name the denominator and the round, because both are
what went stale here.
"""

from collections.abc import Sequence
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

# e2m1 decode table, index = sign<<3 | exp<<1 | mantissa. Mirrors
# `ffw._E2M1_LUT` and `checkpoint_converter._E2M1_LUT`.
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

# One e8m0 exponent per this many contraction elements (the MX spec's
# block size, and what the K3 checkpoint was written with).
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
    lhs_scale_planes: bool = False,
    f8_planes: bool = False,
) -> int:
  """VMEM a single grid step needs, the terms that actually scale.

  Pallas double-buffers every block, the accumulator is a f32 scratch,
  and the dequant holds an i32 code plane plus f32 value and scale
  planes for the chunk in flight. The fused gate+up kernel streams TWO
  `(packed, scales)` pairs against one lhs block (`rhs_pairs=2`), so
  the rhs blocks, the accumulators and the dequant working set double
  while the lhs and out terms do not. `lhs_scale_planes` adds the
  MXFP8-activation e8m0 stream (`lhs_scales`): a `[tm, k/32]` u8 block
  plus its chunk-major restage scratch, the lhs twin of the weight
  scale terms. `f8_planes` (the `F8_PLANE_MODES` operands) bills one
  byte per weight plus one e8m0 per `PLANAR_WINDOW` and no dequant
  working set: the chunk step is a bitcast and a dot.
  """
  if f8_planes:
    packed_rows = k
    scale_rows = k // PLANAR_WINDOW * PLANE_SCALE_ROWS_PER_WINDOW
  else:
    packed_rows, scale_rows = k // 2, k // SCALE_GROUP
  blocks = 2 * (
      tm * k * lhs_bytes  # lhs, whole contraction
      + rhs_pairs * tn * packed_rows  # packed u8 (nibbles or f8 bytes)
      + rhs_pairs * tn * scale_rows  # scales u8
      + tm * tn * 2  # bf16 out
  )
  accumulator = rhs_pairs * tm * tn * 4
  # Codes i32, values f32, scaled f32 per pair: the fused kernel
  # interleaves the two dequants inside one chunk iteration, so count
  # both in flight (Mosaic may overlap them). Plane operands are the
  # MXU input as stored; nothing is materialized.
  dequant = 0 if f8_planes else rhs_pairs * tn * chunk * (4 + 4 + 4)
  # Chunk-major e8m0 code scratch for the rolled k walk
  # (`_stage_scale_codes`): [k/chunk, tn, chunk/32] u8 per pair, whose
  # sub-lane minor pads out to a full lane of VMEM.
  scale_codes = rhs_pairs * max(k // chunk, 1) * tn * _LANES
  # Plane modes: the window maxima widened to f32 rows per stack
  # (`_stage_plane_scales`), [k/chunk, tn] f32.
  plane_exps = (
      rhs_pairs * max(k // chunk, 1) * PLANE_SCALE_ROWS_PER_WINDOW * tn * 4
      if f8_planes
      else 0
  )
  lhs_scales = 0
  if lhs_scale_planes:
    lhs_scales = 2 * tm * (k // SCALE_GROUP) + max(k // chunk, 1) * tm * _LANES
  return blocks + accumulator + dequant + scale_codes + plane_exps + lhs_scales


# Per-chip bytes the register allocator holds live per `[tn, chunk]`
# dequant-chain element, both calibrated against MEASURED allocator
# output rather than derived:
#   * GhostLite (the default, `SCOPED_ELEM_BYTES_GHOSTLITE`): at
#     `(tm, chunk, tn) = (512, 512, 512)` with two rhs pairs Mosaic
#     reported a 33.01M scoped allocation against the 32.00M limit
#     (the ghostlite bench compile), pinning the chain at ~56 bytes
#     per element (~14 live 4-byte planes: the nibble split, i32
#     codes, mantissa/exponent/sign planes, the scale broadcast, and
#     the f32 products before the bf16 cast). 56 rounds that point UP
#     so the model rejects everything the allocator would.
#   * Ghostfish (`SCOPED_ELEM_BYTES_GHOSTFISH`): the gen-7 Mosaic
#     pipeline lowers the same split_mxu chain THINNER: the silicon
#     compile matrix (perf/gf_tile_cap_bench_forge_test, 2026-08-06)
#     compiles the two-pair (512, 512, 512) tile inside prod
#     fst_tput_gf's 19,456 KiB generate ceiling, bounding the chain
#     at <= ~26.6 B/elem; 28 rounds that bound UP the same way. The
#     P34 bench's (256, 256, 256) fused cap was forced by the
#     NON-default arms (legacy dequant / native relabel / fold
#     chains), not by this kernel: the serving split_mxu kernel has
#     no gf tile cap, and the (512, 512, 512) prefill tile the gf
#     constant admits measured 15.5% faster than the GLP-model pick
#     on gf silicon (same test, prefill bench).
SCOPED_ELEM_BYTES_GHOSTLITE = 56
SCOPED_ELEM_BYTES_GHOSTFISH = 28


def _scoped_vmem_bytes(
    tm: int, tn: int, chunk: int, rhs_pairs: int, scoped_elem_bytes: int
) -> int:
  """Kernel STACK bytes Mosaic allocates, against `_SCOPED_VMEM_LIMIT`.

  A model of the register allocator's live set: the dominant term is
  the per-pair dequant chain at `scoped_elem_bytes` per `[tn, chunk]`
  element (per-chip calibration above; the single-pair kernel lands at
  ~15.7M under the GhostLite constant and compiles today, consistent).
  The accumulators are VMEM scratch, which shares the same budget, and
  the fused epilogue holds a few f32 SiTU planes on the `[tm, tn]`
  tile.
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
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_GHOSTLITE,
    force_chunk: int | None = None,
    lhs_scale_planes: bool = False,
    f8_planes: bool = False,
) -> tuple[int, int, int]:
  """The largest valid `(tm, k_chunk, tn)` that fits the VMEM budget.

  Chosen explicitly rather than left to chance: an oversized choice
  does not degrade, it fails the whole compile late, in Mosaic (the
  first version of this kernel dequantized the whole contraction at
  once and the 2048-bucket compile died with 127.49M of
  register-allocator spill slots against a 127.94M VMEM budget).

  `vmem_budget` (None = the module's 64 MiB GhostLite constant)
  bounds the per-grid-step working set; Ghostfish configs pass 32 MiB
  (half of the 64 MiB core, the same one-half discipline;
  `K3MoeConfig.gmm_vmem_budget_bytes`). The scoped-stack CEILING is
  chip-independent (both chips default --xla_tpu_scoped_vmem_limit_kib
  to 32 MiB; target_ghostlite.h and target_ghostfish.h agree) but the
  bytes the allocator actually holds per dequant element are NOT:
  `scoped_elem_bytes` carries the per-chip calibration
  (`SCOPED_ELEM_BYTES_GHOSTLITE` = 56, the default;
  `SCOPED_ELEM_BYTES_GHOSTFISH` = 28, the gen-7 Mosaic measurement -
  `K3MoeConfig.gmm_scoped_elem_bytes` threads the chip value). Under
  the gf constant the search admits the (512, 512, 512) two-pair
  prefill tile that measured 15.5% faster on gf silicon
  (perf/gf_tile_cap_bench_forge_test); GhostLite behavior is byte
  identical to before the seam existed.

  Validity, in the order the search relaxes it: `tm` divides m and is
  8-divisible (or is all of m), `tn` is lane-divisible (or all of n),
  and `chunk` divides k and is scale-group-divisible.

  `expected_rows` opts into the sparse EXPERT-PARALLEL regime: each
  shard sorts ALL M assignment rows but owns only ~M/shards of them
  (7 of 896 experts at the 8x16 mesh), so at the spec round's width-7
  shape (M = 7168, ~56 owned rows) the default 512-row tile made
  every group visit move a ~90%-padding lhs block and output tile
  (profiled at 4.17 ms per grouped-matmul event against plain
  decode's 0.25). When set, `tm` becomes the smallest candidate
  covering the expected rows (floored at `_MIN_EXPECTED_TM`). Groups
  larger than expected (routing skew) stay CORRECT, they just take
  more visits, each re-dequantizing its weight tile; that trade is
  deliberate and the reason the dense regimes (single-device,
  `expected_rows=None`) keep the large default. A whole-N tile (one
  contiguous weight fetch per group visit) was measured back when the
  k walk was a compile-time unroll and REJECTED: tn = 3072 meant a
  56-iteration two-pair dequant chain that Mosaic chewed on for over
  9.5 minutes per shape without finishing. The walk is a real loop now
  for 256-divisible chunks (`_use_k_loop`, via per-element scale codes
  that make the chunk slices lane-aligned), which removes that compile
  blowup; whole-N remains unadopted pending a fresh measurement.

  Args:
    m: lhs rows.
    k: contraction.
    n: output columns.
    lhs_bytes: activation element size.
    rhs_pairs: how many `(packed, scales)` stacks stream against the lhs (1 for
      the plain kernel, 2 for the fused gate+up kernel).
    expected_rows: rows this caller expects to actually own (the EP shard's
      share of M), or None for the dense default.
    vmem_budget: per-grid-step VMEM bound in bytes, or None for the module's
      GhostLite constant (see the docstring note).
    scoped_elem_bytes: per-chip scoped-stack bytes per dequant element
      (`SCOPED_ELEM_BYTES_GHOSTLITE` / `SCOPED_ELEM_BYTES_GHOSTFISH`;
      `K3MoeConfig.gmm_scoped_elem_bytes` threads the chip value).
    force_chunk: pin the k chunk to this value when it is a valid candidate
      (planar weights require chunk == PLANAR_WINDOW). When the shape cannot
      serve it the normal search runs instead and `can_use_kernel` rejects the
      result for planar modes (reference fallback), so this never errors on toy
      shapes.
    lhs_scale_planes: account for the MXFP8-activation e8m0 stream (`lhs_scales`
      block + restage scratch, `_vmem_bytes`); set by the local seams when
      `lhs_f8_scales` is on. Sub-percent of every real budget, carried so the
      model rejects exactly what the compiler would.
    f8_planes: size the weight blocks as LOAD-TIME f8 planes (`F8_PLANE_MODES`:
      one byte per weight, one e8m0 per window, no dequant working set) instead
      of mxfp4 nibble pairs; the seams derive it from `dequant`.

  Returns:
    `(tm, k_chunk, tn)`.

  Raises:
    ValueError: if even the smallest valid tiling exceeds the budget.
  """
  budget = _VMEM_BUDGET if vmem_budget is None else vmem_budget
  tms = [t for t in (512, 256, 128, 64, 32, 16, 8) if m % t == 0] or [m]
  tns = [t for t in (512, 256, 128) if n % t == 0] or [n]
  chunks = [
      c for c in (512, 256, 128, 64, SCALE_GROUP) if c <= k and k % c == 0
  ] or [k]
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
            _vmem_bytes(
                tm,
                k,
                tn,
                chunk,
                lhs_bytes,
                rhs_pairs,
                lhs_scale_planes=lhs_scale_planes,
                f8_planes=f8_planes,
            )
            <= budget
            and _scoped_vmem_bytes(tm, tn, chunk, rhs_pairs, scoped_elem_bytes)
            <= _SCOPED_VMEM_LIMIT
        ):
          return (tm, chunk, tn)
  raise ValueError(
      f'no valid tiling for (m, k, n) = ({m}, {k}, {n}) fits'
      f' {budget} bytes of VMEM (or {_SCOPED_VMEM_LIMIT} scoped); the'
      ' smallest candidate needs'
      f' {_vmem_bytes(tms[-1], k, tns[-1], chunks[-1], lhs_bytes, rhs_pairs, lhs_scale_planes=lhs_scale_planes, f8_planes=f8_planes)}'
      ' /'
      f' {_scoped_vmem_bytes(tms[-1], tns[-1], chunks[-1], rhs_pairs, scoped_elem_bytes)}.'
  )


#: `_make_group_metadata` spellings (`K3MoeConfig.gmm_metadata_impl`).
METADATA_IMPLS = ('repeat', 'searchsorted', 'closed_form')
#: The spellings whose build is loop free (no `while`, no scatter, no
#: gather): safe to build once per layer and share across the kernel
#: calls of one expert visit (`shared_group_metadata`).
METADATA_IMPLS_LOOP_FREE = ('closed_form',)


def _repeat_arange_searchsorted(
    repeats: jax.Array, n: int, total: int
) -> jax.Array:
  """`jnp.repeat(jnp.arange(n), repeats, total_repeat_length=total)`, scatter-free.

  `jnp.repeat`'s ragged lowering scatters ones into a zero buffer and
  cumsums twice (lax_numpy.py `_repeat`), and on GhostLite/Viperfish
  the XLA offloader ships that s32 element scatter to SparseCore BY
  DEFAULT (offloader_util.cc `ShouldOffloadScatter`, TristateProto::
  AUTO). In the served 128-wide chunk step that is TWO async SparseCore
  round trips PER MoE LAYER (one per grouped matmul's metadata build),
  each with a ~2 ms completion span on the kb4 trace - a per-chip
  jitter source the whole mesh then resynchronizes on at the combine
  collectives. The closed form is the identity the scatter+cumsum
  chain computes: entry j belongs to the source index whose exclusive
  prefix sum covers j, i.e. `searchsorted(cumsum(repeats), j, 'right')`;
  entries past `sum(repeats)` repeat the final value (jnp.repeat's
  documented pad), which the clamp reproduces because the source is
  `arange(n)`. Integer in, integer out: bit-identical to the repeat
  spelling (pinned exhaustively in moe_gmm_test).
  """
  cum = jnp.cumsum(repeats.astype(jnp.int32))
  grid = jnp.arange(total, dtype=jnp.int32)
  idx = jnp.searchsorted(cum, grid, side='right')
  return jnp.minimum(idx, n - 1).astype(jnp.int32)


def _closed_form_group_metadata(
    group_sizes: jax.Array,
    *,
    m: int,
    tm: int,
    num_groups: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
  """`_make_group_metadata` as compare-and-sum arithmetic, loop free.

  The megablox build spends its ops on three ragged primitives:
  `jnp.histogram` (a `searchsorted` binary search = a 6-iteration
  `while` loop of ~11 ops per iteration at 56 tiles, its bin edges a
  runtime f32 linspace) and two `jnp.repeat` chains (roll, scatter-add,
  cumsum, gather each). On the GF rung-4 decode program that is 184
  while loops, 460 `scatter_custom_fusion` and 1104 in-loop
  `gather_fusion` executions per step, ~77 instructions per layer on
  the recipient path between `mp_broadcast_packed` and the gmm pair
  (~12 us per layer, ~1.1 ms/step). Every one of those primitives is a
  prefix count over a vector of at most `num_groups` (7) or `tiles_m`
  (56) entries, and a prefix count is one compare against an iota plus
  one reduce, which XLA fuses:

    * `group_ends[g]` = sum of sizes over g' <= g (the cumsum);
    * `repeat(arange(n), r, total)[j]` = min(#{i : cumsum(r)[i] <= j},
      n - 1) (`_repeat_arange_searchsorted`'s identity, incl. the pad
      rule: past sum(r) the entry is the LAST source index);
    * `histogram(partial_tile_ids)[t] + 1`, accumulated to the visit
      prefix sum, = (t + 1) + #{g : partial_tile_ids[g] <= t} (the
      parked value `tiles_m` is above every real tile, so it drops out
      exactly as the histogram's out-of-range rule drops it).

  Integer in, integer out, the same values on every entry including
  the grid pad (moe_gmm_test pins whole-array equality against both
  the repeat and the searchsorted spellings over random, empty and
  full-tray group sizes).

  Args:
    group_sizes: `[num_groups]` int32 rows per group.
    m: total rows in the lhs.
    tm: m-dimension tile size.
    num_groups: number of groups.

  Returns:
    `((group_offsets, group_ids, m_tile_ids), num_active_tiles)`, the
    `_make_group_metadata` contract.
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
      + jnp.sum(
          (partial_tile_ids[None, :] <= tiles[:, None]).astype(jnp.int32), 1
      )
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
    impl: str = 'repeat',
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
  """Grid metadata for the grouped matmul.

  Reproduced from `jax.experimental.pallas.ops.tpu.megablox.gmm`'s
  `make_group_metadata` (which google3 cannot depend on: that directory
  ships no BUILD file), specialized to a single unsharded group range
  and `visit_empty_groups=False`, which is all this caller needs.

  Args:
    group_sizes: `[num_groups]` int32 rows per group.
    m: total rows in the lhs.
    tm: m-dimension tile size.
    num_groups: number of groups.
    impl: 'repeat' (the `jnp.repeat` spelling, whose ragged lowering carries an
      s32 scatter that GhostLite offloads to SparseCore per call),
      'searchsorted' (the scatter-free closed form,
      `_repeat_arange_searchsorted`; identical integers; keeps the histogram's
      binary-search loop) or 'closed_form' (the whole build as compare-and-sum
      arithmetic, `_closed_form_group_metadata`: no while loop, no scatter, no
      gather; identical integers).

  Returns:
    `((group_offsets, group_ids, m_tile_ids), num_active_tiles)` where
    `group_offsets[i]` is the first row of group i, and `group_ids[j]` /
    `m_tile_ids[j]` say which group and which m tile grid step j works
    on. A tile straddling a group boundary appears once per group it
    touches, which is why the grid is `tiles_m + num_groups - 1` long.
  """
  if impl not in METADATA_IMPLS:
    raise ValueError(f'unknown metadata impl {impl!r} (want {METADATA_IMPLS}).')
  if impl == 'closed_form':
    return _closed_form_group_metadata(
        group_sizes, m=m, tm=tm, num_groups=num_groups
    )
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

  # Round each group's span out to whole tiles: a group that starts or
  # ends mid tile still has to compute that whole tile.
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
  group_tiles = rounded_group_sizes // tm

  tiles_m = m // tm
  grid_len = tiles_m + num_groups - 1
  if impl == 'searchsorted':
    group_ids = _repeat_arange_searchsorted(group_tiles, num_groups, grid_len)
  else:
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=grid_len,
    )

  # An m tile is visited once by the group that owns it, plus once per
  # additional group that starts inside it. Tiles whose owner starts
  # tile-aligned (or is empty) are parked at `tiles_m` so the histogram
  # does not double count them.
  partial_tile_mask = jnp.logical_or(
      (group_offsets[:-1] % tm) == 0, group_sizes == 0
  )
  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )
  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )
  if impl == 'searchsorted':
    m_tile_ids = _repeat_arange_searchsorted(
        tile_visits.astype(jnp.int32), tiles_m, grid_len
    )
  else:
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=grid_len,
    )
  return (group_offsets, group_ids, m_tile_ids), group_tiles.sum()


def one_hot_group_metadata(
    group_id: jax.Array,
    size: jax.Array,
    *,
    m: int,
    tm: int,
    num_groups: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
  """`_make_group_metadata` in closed form for ONE group at rows [0, size).

  The kiloblox chunk loop's shape: every chunk holds one expert's rows,
  contiguous from row 0. The general metadata build costs a
  `jnp.histogram` (which lowers to a binary-search while loop; the kb4
  v2 trace bills its gather at 2.0% of the decode step, the gf trace at
  4.6% - gather_fusion.310/.203) plus two `jnp.repeat` chains, ALL
  recomputed per chunk per expert visit. For a single group starting at
  row 0 the answer is arithmetic:

    * `group_offsets[g]` = 0 for g <= id, `size` after;
    * every tile is visited exactly once by the owning group (the group
      starts tile-aligned at row 0), so `m_tile_ids` is the iota and
      `group_ids` is the constant id;
    * `num_active_tiles` = ceil(size / tm).

  Entries past `num_active_tiles` are padding the kernel's grid never
  reads, exactly like the general build's `total_repeat_length` tail
  (moe_gmm_test pins the read prefix equal to `_make_group_metadata`
  on a one-hot grid).

  Args:
    group_id: scalar int32, the single non-empty group.
    size: scalar int32 rows in that group (0 is legal: zero active tiles).
    m: total lhs rows.
    tm: m-dimension tile size (the kernel's, so callers pass the hint through
      `single_group=` and let the kernel derive tm rather than guessing it).
    num_groups: number of groups.

  Returns:
    The `_make_group_metadata` tuple shape:
    `((group_offsets, group_ids, m_tile_ids), num_active_tiles)`.
  """
  group_id = jnp.asarray(group_id, jnp.int32)
  size = jnp.asarray(size, jnp.int32)
  ids = jnp.arange(num_groups + 1, dtype=jnp.int32)
  group_offsets = jnp.where(ids > group_id, size, jnp.int32(0))
  n = m // tm + num_groups - 1
  group_ids = jnp.full((n,), group_id, jnp.int32)
  m_tile_ids = jnp.arange(n, dtype=jnp.int32)
  num_active_tiles = (size + tm - 1) // tm
  return (group_offsets, group_ids, m_tile_ids), num_active_tiles


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
        'group_metadata was built for a different tiling: shapes'
        f' {got}, this kernel (m={m}, tm={tm}, num_groups={num_groups})'
        f' wants {want}. Build it with shared_group_metadata over the'
        ' tilings of every consuming kernel.'
    )
  for name, arr in (
      ('group_offsets', group_offsets),
      ('group_ids', group_ids),
      ('m_tile_ids', m_tile_ids),
      ('num_active_tiles', num_active_tiles),
  ):
    if jnp.asarray(arr).dtype != jnp.int32:
      raise ValueError(
          f'group_metadata.{name} must be int32, got {jnp.asarray(arr).dtype}.'
      )
  return group_metadata


def _make_paired_group_metadata(
    group_sizes: jax.Array,
    m: int,
    tm: int,
    num_groups: int,
    impl: str = 'repeat',
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
  """`_make_group_metadata` with same-tile visit PAIRS merged.

  The grouped grid visits an m tile once per group that touches it; at
  decode (tm covers the whole row block, several activated experts)
  that is one full k walk PER EXPERT over the same lhs tile, and the
  single-stream walk runs at ~42% of its own V_ALU slot bound on GF
  (in-body dependence-chain interlocks). Merging two visits into one
  grid step with TWO group ids lets the paired kernel dequantize both
  experts' strips interleaved, each stream filling the other's
  interlock windows: measured wall ratio 1.257 for 2x the work vs 2.0
  serial (rung 3, pre-registered, sponge 829e304d).

  Args:
    group_sizes: `[num_groups]` int32 rows per group, `_make_group_metadata`
      semantics.
    m: total rows in the lhs.
    tm: m-dimension tile size.
    num_groups: number of groups.
    impl: metadata spelling, forwarded to `_make_group_metadata`. The pairing
      itself is scatter-free (searchsorted / cumsum only), so it adds no
      SparseCore offload exposure beyond the base build's.

  Returns:
    `((group_offsets, group_ids_a, group_ids_b, m_tile_ids),
    num_paired_tiles)`. Grid step k works on m tile `m_tile_ids[k]` for
    group `group_ids_a[k]`, and ALSO for group `group_ids_b[k]` unless
    `group_ids_b[k] == group_ids_a[k]`, the NO-PARTNER sentinel (sound
    because a tile is visited at most once per group, so two visits of
    one tile always carry distinct groups). Entries past
    `num_paired_tiles` are padding the kernel's grid never reads,
    exactly like the base build's tail.
  """
  (group_offsets, group_ids, m_tile_ids), num_active_tiles = (
      _make_group_metadata(
          group_sizes, m=m, tm=tm, num_groups=num_groups, impl=impl
      )
  )
  grid_len = m_tile_ids.shape[0]
  j = jnp.arange(grid_len, dtype=jnp.int32)
  # Rank of visit j within its tile's run of consecutive visits:
  # m_tile_ids is nondecreasing, so each run's start is a searchsorted.
  run_start = jnp.searchsorted(m_tile_ids, m_tile_ids, side='left').astype(
      jnp.int32
  )
  rank = j - run_start
  # Even-rank ACTIVE visits lead a pair (or stand alone at a run's odd
  # tail); odd-rank visits are absorbed as some leader's partner.
  leader = jnp.logical_and((rank % 2) == 0, j < num_active_tiles)
  # Compact leaders to the grid prefix, scatter-free: paired step k
  # reads the k-th leader (same searchsorted-over-cumsum form as
  # `_repeat_arange_searchsorted`).
  leader_count = jnp.cumsum(leader.astype(jnp.int32))
  src = jnp.searchsorted(leader_count, j + 1, side='left').astype(jnp.int32)
  src = jnp.minimum(src, grid_len - 1)
  nxt = jnp.minimum(src + 1, grid_len - 1)
  group_ids_a = group_ids[src]
  has_b = jnp.logical_and(
      src + 1 < num_active_tiles, m_tile_ids[nxt] == m_tile_ids[src]
  )
  group_ids_b = jnp.where(has_b, group_ids[nxt], group_ids_a)
  paired_m_tile_ids = m_tile_ids[src]
  num_paired_tiles = leader_count[-1]
  return (
      group_offsets,
      group_ids_a,
      group_ids_b,
      paired_m_tile_ids,
  ), num_paired_tiles


def _decode_e2m1(codes: jax.Array) -> jax.Array:
  """e2m1 nibbles -> f32 magnitudes with sign, arithmetically.

  Bitwise equal to indexing `_E2M1_LUT`, but expressed as arithmetic so
  the kernel does not need a gather in VMEM: with exp = bits 2:1 and
  mant = bit 0, the table is `0.5 * mant` at exp 0 and
  `(1 + 0.5 * mant) * 2^(exp - 1)` above it, which reproduces
  (0, .5, 1, 1.5, 2, 3, 4, 6) exactly. Every value is exact in f32, so
  this is not an approximation of the table, it IS the table.
  """
  codes = codes.astype(jnp.int32)
  mant = (codes & 1).astype(jnp.float32)
  exp = (codes >> 1) & 3
  # 2^(exp-1) for exp in 1..3, i.e. 1, 2, 4.
  pow2 = jnp.exp2((exp - 1).astype(jnp.float32))
  magnitude = jnp.where(exp == 0, 0.5 * mant, (1.0 + 0.5 * mant) * pow2)
  return jnp.where((codes >> 3) & 1 == 1, -magnitude, magnitude)


def _dequantize_tile(packed: jax.Array, scales: jax.Array) -> jax.Array:
  """Decodes one `[tn, tk/2]` u8 tile into `[tn, tk]` bf16.

  Args:
    packed: `[tn, tk/2]` uint8, two e2m1 nibbles per byte, low nibble first
      (even contraction index).
    scales: `[tn, tk/SCALE_GROUP]` uint8 e8m0 shared exponents.

  Returns:
    `[tn, tk]` bfloat16 weights.
  """
  tn, half_k = packed.shape
  tk = half_k * 2
  groups = scales.shape[-1]

  # Interleave the nibbles so the reshape yields (even0, odd0, ...),
  # the same stack-then-reshape `dequantize_mxfp4_stack_jnp` uses.
  codes = jnp.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(tn, tk)
  values = _decode_e2m1(codes)

  # 2^(u8-127) by exponent-bit construction, not a libm exp2: exact for
  # every code including 0 (subnormal 2^-127, whose bit pattern is not
  # e<<23) and 255 (inf), so it matches np.exp2 bit for bit.
  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)

  values = values.reshape(tn, groups, tk // groups) * exps[..., None]
  return values.reshape(tn, tk).astype(jnp.bfloat16)


# e8m0 codes whose fused exponent add stays inside the f32 normal band.
# The fast decode writes exponent field `e + (code - 1)` with e in 0..3,
# which must land in [1, 254], so code must be in [2, 252]. Outside that
# band `_scale_planes` hands the leftover factor to the residual.
_SCALE_CODE_LO = 2
_SCALE_CODE_HI = 252


def _scale_planes(scales: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Splits e8m0 codes into an exponent bias and a residual multiplier.

  The fast decode folds the block scale into the f32 exponent field as
  an integer add, which is only exact while the sum stays normal. This
  runs on the `[tn, k/32]` scale plane, 1/32 the elements of the value
  plane, so the clamp and the residual construction are effectively
  free per weight element.

  Returns:
    `(bias, residual)`. `bias` is int32 `clamp(code, 2, 252) - 1`, the
    value added into the exponent field. `residual` is the f32 leftover
    `2^(code - clamped)`, exactly 1.0 for every code in [2, 252] (every
    code the real checkpoint carries). It is built from the same
    bit-constructed `2^(code-127)` `_dequantize_tile` uses, so code 0
    keeps its subnormal and code 255 keeps its inf, including the
    `0 * inf` NaN the residual multiply then reproduces.
  """
  codes = scales.astype(jnp.int32)
  clamped = jnp.clip(codes, _SCALE_CODE_LO, _SCALE_CODE_HI)

  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)
  # 2^(127 - clamped): field 254 - clamped lands in [2, 252], normal.
  inv = lax.bitcast_convert_type(
      (254 - clamped).astype(jnp.uint32) << 23, jnp.float32
  )
  return clamped - 1, exps * inv


def _decode_e2m1_scaled(
    codes: jax.Array, bias: jax.Array, residual: jax.Array
) -> jax.Array:
  """`e2m1(code) * 2^(scale-127)` as f32, built in the exponent field.

  e2m1 is `sign<<3 | exp<<1 | mant` and, for exp > 0, denotes
  `(1 + 0.5*mant) * 2^(exp-1)`. f32 shares bf16's bias of 127, so that
  value scaled by `2^(code-127)` has exponent field
  `126 + exp + code - 127 = exp + (code - 1) = exp + bias` and mantissa
  field `mant << 22`. The block scale is therefore an integer add into
  the exponent, not a multiply: no `exp2`, no separate broadcast
  multiply over the value plane.

  Two codes fall outside that closed form and cost one select each:
  exp == 0 with mant == 1 is 0.5, i.e. `(1 + 0)*2^-1`, so its mantissa
  field must be cleared (its exponent already matches, `0 + bias`); and
  code 0 / 8 are +-0.0, so the whole exponent-and-mantissa field clears
  while the sign survives.

  `codes` is `[..., width]` and `bias` / `residual` are `[..., groups]`,
  one entry per `width // groups` codes. The regroup to 3D happens on
  COMPUTED values, never straight off the caller's nibble split: Mosaic
  cannot infer a layout for a reshape chained onto the split itself
  (`vector<512x128x2xi8> -> vector<512x8x32xi8>` fails infer-vector-
  layout), and folding the arithmetic in first is what breaks the chain.

  Args:
    codes: `[..., width]` e2m1 nibbles in the low 4 bits.
    bias: `[..., groups]` int32 exponent addend from `_scale_planes`.
    residual: `[..., groups]` f32 leftover factor, 1.0 in the real band.

  Returns:
    `[..., width]` f32 scaled weights.
  """
  n = codes.astype(jnp.uint32)
  e = (n >> 1) & 3
  # Sign is placed BEFORE the bias add and survives the zero mask, the
  # order the exhaustive grid test pins: the exponent field maxes at
  # 254 so the add never carries into bit 31, and clearing bits 0..30
  # for +-0.0 leaves a signed zero whose `0 * inf` still yields the NaN
  # the baseline produces at scale code 255.
  partial = ((n & 8) << 28) | (e << 23) | jnp.where(e == 0, 0, (n & 1) << 22)
  # +-0.0 keeps ONLY its sign bit; every other code keeps everything.
  keep = jnp.where((n & 7) == 0, jnp.uint32(0x80000000), jnp.uint32(0xFFFFFFFF))

  groups = bias.shape[-1]
  shape3 = n.shape[:-1] + (groups, n.shape[-1] // groups)
  bits = (
      partial.reshape(shape3) + (bias.astype(jnp.uint32) << 23)[..., None]
  ) & keep.reshape(shape3)
  scaled = lax.bitcast_convert_type(bits, jnp.float32) * residual[..., None]
  return scaled.reshape(n.shape)


def _dequantize_tile_fast(packed: jax.Array, scales: jax.Array) -> jax.Array:
  """Bitwise-identical drop-in for `_dequantize_tile`, fewer VPU ops.

  Same `[tn, tk/2]` u8 in, `[tn, tk]` bf16 out. `moe_gmm_test` pins the
  two exhaustively equal over the whole 16 x 256 e2m1 x e8m0 grid,
  compared as raw bits so NaN and signed-zero cases count.
  """
  tn, half_k = packed.shape
  tk = half_k * 2

  codes = jnp.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(tn, tk)
  bias, residual = _scale_planes(scales)
  return _decode_e2m1_scaled(codes, bias, residual).astype(jnp.bfloat16)


@functools.cache
def _split_k_indices(k: int, chunk: int) -> tuple[int, ...]:
  """The k permutation `_dequantize_tile_split` implies, per chunk.

  The kernel walks the contraction `chunk` elements at a time and each
  step's decode emits that chunk's evens then its odds, so the whole
  permutation is that pattern repeated per chunk window. Materialized
  as static indices because it must be applied OUTSIDE the kernel (see
  `split_k_order`).
  """
  order: list[int] = []
  for start in range(0, k, chunk):
    width = min(chunk, k - start)
    order.extend(range(start, start + width, 2))
    order.extend(range(start + 1, start + width, 2))
  return tuple(order)


def split_k_order(x: jax.Array, chunk: int | None = None) -> jax.Array:
  """Reorders a `[..., k]` dot operand into per chunk even-then-odd k.

  The pair to `_dequantize_tile_split`. Both dots in this module
  contract the minor dim of both operands, so permuting k on BOTH sides
  leaves the product unchanged; that is what lets the weight side skip
  the nibble interleave entirely.

  This runs OUTSIDE the Pallas kernel, on the `[M, K]` activation, once
  per call. It has to: a stride 2 slice lowers to
  `vector.extract_strided_slice` with `strides = [1, 2]`, which Mosaic
  rejects (`expected strides to be confined to [1, 2)`). Hoisting it
  out is also strictly better than an in kernel shuffle would have
  been, since the activation is touched once instead of once per k
  chunk per expert visit, and it is `M x K` against the weight side's
  `E x N x K`.

  Args:
    x: `[..., k]` operand.
    chunk: the kernel's contraction chunk. None means the whole k.

  Returns:
    `x` with its minor dim permuted.
  """
  k = x.shape[-1]
  return jnp.take(
      x, jnp.asarray(_split_k_indices(k, chunk or k), jnp.int32), axis=-1
  )


def _dequantize_tile_split(packed: jax.Array, scales: jax.Array) -> jax.Array:
  """`[tn, tk/2]` u8 -> `[tn, tk]` bf16 with k in even-then-odd order.

  `_dequantize_tile` spends a cross-lane shuffle per weight element
  putting the two nibbles of a byte back into contraction order
  (`jnp.stack(..., axis=-1).reshape` is a lane interleave, and it runs
  on the BIG plane: `tn` output features by `chunk` contraction). That
  ordering is not load-bearing. Decoding the low nibbles and the high
  nibbles as two whole planes and concatenating them is pure vreg
  placement, no lane movement, and the dot is identical as long as the
  activation gets the same permutation via `split_k_order` (a shuffle
  on a `tm`-row plane, and `tm` is the token tile, far smaller than
  `tn`).

  Scale groups survive the split intact: group g owns contraction
  indices [32g, 32g+32), whose evens are exactly low-nibble columns
  [16g, 16g+16) and whose odds are exactly high-nibble columns
  [16g, 16g+16). So each half broadcasts its scale over 16 lanes
  instead of 32, against the same `[tn, k/32]` plane.

  Output column p < tk/2 holds contraction index 2p; column tk/2 + p
  holds 2p + 1.
  """
  bias, residual = _scale_planes(scales)
  halves = [
      _decode_e2m1_scaled(codes, bias, residual)
      for codes in (packed & 0x0F, packed >> 4)
  ]
  return jnp.concatenate(halves, axis=-1).astype(jnp.bfloat16)


def _expand_scale_codes(scales: jax.Array, width: int) -> jax.Array:
  """`[tn, g]` e8m0 codes -> `[tn, width]`, each repeated `width // g`.

  Done as a matmul against a constant 0/1 expansion matrix, i.e. on the
  MXU, for a measured reason: the obvious VPU route reshapes the value
  plane to `[tn, g, width // g]`, whose minor dim is 16 lanes out of
  128, and that narrow relayout prices at 3.4 ms of a 4.5 ms kernel
  (75%) at the serving geometry. The MXU is ~95% idle here (the bf16
  floor is 0.25 ms), so moving the broadcast onto it is nearly free and
  it lands the result already at full lane width.

  Exact, not approximate. Every code is a small integer (0..255), which
  bf16 represents exactly, and each output is a sum with exactly ONE
  nonzero term, so no MXU pass count can round it. CODES are expanded
  rather than the f32 scale factors precisely so that code 255 (inf)
  never meets a zero of the expansion matrix, which would make
  `0 * inf` a NaN and poison the sum.
  """
  groups = scales.shape[-1]
  per = width // groups
  onehot = (
      jnp.arange(width, dtype=jnp.int32)[None, :] // per
      == jnp.arange(groups, dtype=jnp.int32)[:, None]
  ).astype(jnp.float32)
  wide = lax.dot_general(
      scales.astype(jnp.float32),
      onehot,
      dimension_numbers=(((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return wide.astype(jnp.uint32)


def _dequantize_tile_split_mxu(
    packed: jax.Array, scales: jax.Array
) -> jax.Array:
  """Split k layout with the block scale broadcast on the MXU.

  Same arithmetic as `_dequantize_tile_split_lut`, but the per group
  scale is expanded to full lane width by `_expand_scale_codes` first,
  so the value plane is never reshaped to a 16 lane minor dim. Bitwise
  equal to it: the expansion is exact, and the surviving per element
  work is the same `_decode_e2m1` and the same f32 multiply.
  """
  codes = _expand_scale_codes(scales, packed.shape[1])
  exps = lax.bitcast_convert_type(codes << 23, jnp.float32)
  exps = jnp.where(codes == 0, jnp.float32(2.0**-127), exps)
  halves = [
      _decode_e2m1(nibbles) * exps for nibbles in (packed & 0x0F, packed >> 4)
  ]
  return jnp.concatenate(halves, axis=-1).astype(jnp.bfloat16)


def _dequantize_tile_split_lut(
    packed: jax.Array, scales: jax.Array
) -> jax.Array:
  """Split k layout with the SHIPPED per element arithmetic.

  The fourth corner of the A/B, and the one that separates the two
  changes: it keeps `_decode_e2m1` and the f32 block scale multiply
  exactly as `_dequantize_tile` has them and changes ONLY where the two
  nibbles land. Bitwise equal to `split_k_order(_dequantize_tile(...))`,
  because the scale multiply is elementwise, so regrouping it 16 wide
  per half instead of 32 wide across the pair cannot move a bit.
  """
  tn, half_k = packed.shape
  groups = scales.shape[-1]
  per_half = half_k // groups

  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)

  halves = []
  for codes in (packed & 0x0F, packed >> 4):
    values = _decode_e2m1(codes).reshape(tn, groups, per_half)
    halves.append((values * exps[..., None]).reshape(tn, half_k))
  return jnp.concatenate(halves, axis=-1).astype(jnp.bfloat16)


def _planar_repack_windows(packed, window: int):
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


def repack_planar_stack(packed, window: int = PLANAR_WINDOW):
  """Reorders MXFP4 nibble pairs into the PLANAR window layout.

  The split idea (`_dequantize_tile_split`) moved OFFLINE. Interleaved
  storage puts contraction elements (2j, 2j+1) in byte j, so a decode
  that reads the two nibble planes whole emits even-then-odd order and
  the ACTIVATION must be permuted to match (`split_k_order`), which the
  kiloblox while body pays as an XLA gather + two layout copies per
  chunk per expert visit (profiled at ~14 ms of an 89 ms decode step at
  ~225 GiB/s on the kb4 v2 trace; ~78 ms/step on gf). Planar storage
  puts elements (w*window + j, w*window + window/2 + j) in window w's
  byte j instead: the same two whole-plane nibble decodes then emit
  each window's columns in ORIGINAL contraction order, so no operand is
  ever permuted at runtime, on either side.

  Scales are NOT repacked: group g still owns contraction elements
  [32g, 32g + 32), which planar decode consumes as the first half of a
  window's groups against the low plane and the second half against
  the high plane (32-wide broadcasts either way).

  TRACE SAFETY (the P44 export outage, xid/277673658): the serving
  exporter applies the checkpoint projection UNDER A JIT TRACE (roc
  loading.py `transformation_returning_parts`), so this function must
  never force a numpy conversion on its input. jax values and TRACERS
  take the pure-jnp path (a deterministic byte permutation: reshape /
  slice / shift / or, all traceable); only genuinely host-resident
  arrays take the bounded numpy slab walk. Shape/dtype branching only,
  never value branching.

  Args:
    packed: `[..., K/2]` uint8 nibble pairs, checkpoint (interleaved) order. K
      must divide by `window`. jax array or tracer in -> jnp out (trace-safe);
      numpy (or lazy host array-like) in -> numpy out (slab-wise, bounded
      transients for the host-side tools).
    window: The planar window in contraction elements; must equal the kernel k
      chunk (`PLANAR_WINDOW`).

  Returns:
    The repacked array, same shape and dtype.
  """
  half_k = packed.shape[-1]
  k = 2 * half_k
  if window % (2 * SCALE_GROUP):
    raise ValueError(
        f'planar window {window} must be a multiple of {2 * SCALE_GROUP} so'
        ' each nibble plane covers whole scale groups.'
    )
  if k % window:
    raise ValueError(
        f'contraction {k} must be a multiple of the planar window {window}.'
    )
  if isinstance(packed, (jax.Array, jax.core.Tracer)):
    # jit/trace path: pure jnp. numpy anywhere here is the
    # TracerArrayConversionError that killed every export since P44.
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


def unpack_planar_stack(packed, window: int = PLANAR_WINDOW):
  """The inverse of `repack_planar_stack` (back to interleaved pairs).

  Used by the reference fallback (shapes the kernel cannot serve), the
  dense-path stack dequant, and tests; round trips bitwise by
  construction. Same trace-safety dispatch as the repack: jax values
  and tracers stay on jnp (this already runs inside jit on the dense
  dequant path), everything host-side stays on numpy.
  """
  xp = jnp if isinstance(packed, (jax.Array, jax.core.Tracer)) else np
  half_k = packed.shape[-1]
  k = 2 * half_k
  hw = window // 2
  if k % window:
    raise ValueError(
        f'contraction {k} must be a multiple of the planar window {window}.'
    )
  lead = packed.shape[:-1]
  planar = packed.reshape(lead + (k // window, hw))
  # Original contraction order per window, then re-pair (even, odd).
  codes = xp.concatenate([planar & 0x0F, planar >> 4], axis=-1)
  even = codes[..., 0::2]
  odd = codes[..., 1::2]
  return (even | (odd << 4)).astype(packed.dtype).reshape(lead + (half_k,))


def repack_planar_t_stack(packed, window: int = PLANAR_WINDOW):
  """Planar window bytes, TRANSPOSED: `[..., N, K/2]` -> `[..., K/2, N]`.

  The planar_t layout is `repack_planar_stack` followed by a swap of
  the last two dims, so byte `[..., w*window/2 + j, n]` holds the
  nibbles of contraction elements `w*window + j` (low) and
  `w*window + window/2 + j` (high) of weight row `n`. Same weight
  VALUES bit for bit; only placement changes. What the transpose buys:
  the dequantized tile comes out `[chunk, tn]` with the weight ROW on
  the lane dim, which is the axis the vperm.slane gather table varies
  over, so the per-group e8m0 scale can be multiplied into the 16-entry
  table instead of into every weight element
  (`_dequantize_chunk_planar_t_gather`).

  Trace-safe exactly like `repack_planar_stack` (jax values and tracers
  stay on jnp; host arrays take the bounded slab walk, then a numpy
  transpose materialized once).
  """
  planar = repack_planar_stack(packed, window)
  if isinstance(planar, (jax.Array, jax.core.Tracer)):
    return jnp.swapaxes(planar, -1, -2)
  return np.ascontiguousarray(np.swapaxes(planar, -1, -2))


def unpack_planar_t_stack(packed, window: int = PLANAR_WINDOW):
  """The inverse of `repack_planar_t_stack` (back to interleaved pairs).

  Used by the reference fallback and tests; round trips bitwise by
  construction (a transpose then the planar unpack).
  """
  xp = jnp if isinstance(packed, (jax.Array, jax.core.Tracer)) else np
  return unpack_planar_stack(xp.swapaxes(packed, -1, -2), window)


def transpose_scales_planar_t(scales):
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


#: Scale rows per planar window: the second-minor dim of the chunk-major
#: scale form, and of the in-kernel scale ref every planar_t dequant
#: body reads (one e8m0 row per SCALE_GROUP slice of the window).
CHUNK_MAJOR_SCALE_ROWS = PLANAR_WINDOW // SCALE_GROUP


def restage_scales_chunk_major(scales):
  """k-major planar_t scales `[..., K/32, N]` -> `[..., K/512, 16, N]`.

  The exact chunk-major split `mxfp4_gmm` applies to its scale operand
  before the kernel call (a row-major reshape: same bytes, same order),
  moved to LOAD time so the params can be DECLARED in the operand shape
  the kernels read. Why that matters: with k-major scale params under
  relayout-optimally exports, XLA assigns the u8 `[E, 96, N]` down-stack
  param the T(32,128) sublane tiling (96 divides 32) while the in-step
  16-row split is a bitcast only from T(8,128), so every MoE layer's
  down scales paid a per-decode-step HBM->HBM relayout copy (4.8 MiB
  x 92 layers = 0.59 ms/step on the settled GF flagship trace); the
  gate/up stacks dodged it only because 112 does not divide 32.
  Chunk-major params ARE the kernel operand, so no in-step transform
  exists to mis-lay out. Trace-safe like the packed repack. mxfp4
  planes only (f8_load_time window scales already hold one row per
  window and take the `[E, S, 1, N]` restage in `mxfp4_gmm` itself).
  """
  if scales.shape[-2] % CHUNK_MAJOR_SCALE_ROWS:
    raise ValueError(
        'k-major planar_t scales must hold whole'
        f' {PLANAR_WINDOW}-element windows ({CHUNK_MAJOR_SCALE_ROWS}'
        f' rows each); got {scales.shape}.'
    )
  lead = tuple(scales.shape[:-2])
  steps = scales.shape[-2] // CHUNK_MAJOR_SCALE_ROWS
  return scales.reshape(
      lead + (steps, CHUNK_MAJOR_SCALE_ROWS, scales.shape[-1])
  )


def mxfp4_pair_specs(stack, matrix_spec):
  """Shard specs for an mxfp4 `(packed, scales)` pair from ONE matrix spec.

  `matrix_spec` is the 3-entry `(experts, rows, cols)` placement of the
  stack's matrix dims (the packed leaf's rank). K-major scales share
  that rank, so the one spec serves both leaves. Chunk-major scales
  (`restage_scales_chunk_major`, `[..., S, 16, N]`) insert the 16-row
  window slab between rows and cols, where a positional 3-entry spec
  would land the rows-slot axis name on the SLAB and shard windows in
  half: the dual-TC F split then delivers 8-row slabs that
  `flatten_chunk_major_scales` refuses by design. Lift the placement
  through the restage's reshape instead: the window axis S inherits
  the k-major rows placement (whole windows of the same bytes per
  shard), the slab stays unsharded, and N keeps its placement.

  Args:
    stack: The stack value: an mxfp4 `(packed, scales)` tuple or a dense array.
    matrix_spec: The 3-entry `PartitionSpec` for the matrix dims.

  Returns:
    `matrix_spec` unchanged for dense stacks and k-major pairs, or the
    `(packed_spec, scales_spec)` pair matching the tuple's structure
    for chunk-major pairs (a shard_map in_specs tree).
  """
  if not isinstance(stack, tuple):
    return matrix_spec
  packed, scales = stack
  if scales.ndim != packed.ndim + 1:
    return matrix_spec
  if len(matrix_spec) != 3:
    raise ValueError(
        f'mxfp4 stack specs carry (experts, rows, cols); got {matrix_spec}.'
    )
  scales_spec = jax.sharding.PartitionSpec(
      matrix_spec[0], matrix_spec[1], None, matrix_spec[2]
  )
  return (matrix_spec, scales_spec)


def flatten_chunk_major_scales(scales):
  """Undo `restage_scales_chunk_major`: `[..., S, 16, N]` -> `[..., S*16, N]`.

  The kernels' internal contract stays k-major 3-D; chunk-major params
  normalize through here at the wrapper tops, and the merge cancels
  against the kernels' own chunk-major restage (reshape of reshape back
  to the param's shape), so the custom call consumes the param
  directly. Raises on a second-minor dim that is not the window's
  16-row slab, so a mis-wired stack fails loudly instead of decoding
  garbage.
  """
  if scales.shape[-2] != CHUNK_MAJOR_SCALE_ROWS:
    raise ValueError(
        'chunk-major planar_t scales carry'
        f' {CHUNK_MAJOR_SCALE_ROWS} rows per window slab, got'
        f' {scales.shape}.'
    )
  lead = tuple(scales.shape[:-3])
  merged = scales.shape[-3] * scales.shape[-2]
  return scales.reshape(lead + (merged, scales.shape[-1]))


def _dequantize_tile_planar_lut(
    packed: jax.Array, scales: jax.Array
) -> jax.Array:
  """PLANAR-packed `[tn, tk/2]` u8 -> `[tn, tk]` bf16 in ORIGINAL k order.

  The shipped per-element arithmetic of `_dequantize_tile_split_lut` on
  planar-repacked bytes: the low nibble plane is the window's first
  half of contraction columns and the high plane its second half, so
  the concat lands in original order and the lhs needs no permutation.
  Each half broadcasts its own half of the scale groups 32 wide (the
  original group geometry, just split at the window midpoint).
  """
  tn, half_k = packed.shape
  groups = scales.shape[-1]
  gh = groups // 2

  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)

  halves = []
  for i, codes in enumerate((packed & 0x0F, packed >> 4)):
    values = _decode_e2m1(codes).reshape(tn, gh, half_k // gh)
    e = exps[:, i * gh : (i + 1) * gh]
    halves.append((values * e[..., None]).reshape(tn, half_k))
  return jnp.concatenate(halves, axis=-1).astype(jnp.bfloat16)


def _dequantize_tile_planar_mxu(
    packed: jax.Array, scales: jax.Array
) -> jax.Array:
  """Planar layout with the block scale broadcast on the MXU.

  `_dequantize_tile_split_mxu`'s arithmetic on planar-repacked bytes
  (see `_dequantize_tile_planar_lut` for the layout). Planar halves own
  DISJOINT scale groups (interleaved halves share theirs), so the exps
  plane is chunk wide instead of chunk/2: built as ONE
  `_expand_scale_codes` matmul at 32 codes per group and sliced
  statically per half (lane-aligned at every legal window), which keeps
  one expansion dispatch and one where/bitcast pass instead of two of
  each. Bitwise equal to `_dequantize_tile_planar_lut`.
  """
  half_k = packed.shape[1]
  codes = _expand_scale_codes(scales, 2 * half_k)
  exps = lax.bitcast_convert_type(codes << 23, jnp.float32)
  exps = jnp.where(codes == 0, jnp.float32(2.0**-127), exps)
  halves = [
      _decode_e2m1(packed & 0x0F) * exps[:, :half_k],
      _decode_e2m1(packed >> 4) * exps[:, half_k:],
  ]
  return jnp.concatenate(halves, axis=-1).astype(jnp.bfloat16)


def _e2m1_gather_table(half_k: int) -> jax.Array:
  """The 16-entry e2m1 value table, one copy per lane: `[16, half_k]` bf16.

  Built from an iota through the SHIPPED `_decode_e2m1` arithmetic (not
  a dense literal, which Mosaic cannot materialize as a vreg constant,
  and not a `broadcast_to`, whose lane-replicated layout the gather
  lowering rejects), so the table VALUES are `_decode_e2m1`'s by
  construction and the build is ~2 vregs of loop-invariant work per
  grid step. bf16 is exact for every e2m1 value (2 significant bits
  against bf16's 8), so widening a gathered entry back to f32
  reproduces `_decode_e2m1`'s f32 output bit for bit.
  """
  codes = lax.broadcasted_iota(jnp.int32, (16, half_k), 0)
  return _decode_e2m1(codes).astype(jnp.bfloat16)


def _dequantize_tile_planar_gather(
    packed: jax.Array,
    scales: jax.Array,
    table: jax.Array | None = None,
) -> jax.Array:
  """Planar layout with the e2m1 decode as a hardware table lookup.

  `_dequantize_tile_planar_mxu` spends ~15 VPU ops per weight element
  turning a nibble into its f32 value (`_decode_e2m1`: widen, masks,
  shifts, an exp2, two selects, a sign select), and at decode widths
  that arithmetic IS the grouped-matmul wall (the decode-floor-2 map:
  ~176 Gelem/s of expansion, m-independent, 30% of the step). The
  decode is a 16-entry lookup, and the VPU has a 32-byte in-register
  table-lookup instruction (`vperm.slane`, one ALU slot per vreg,
  go/glc-tc-isa#permute-between-sublanes; Ghostfish keeps byte
  granularity too): `jnp.take_along_axis` over a `[16, half_k]` bf16
  table lowers to `tpu.dynamic_gather` along the second-minor dim,
  which Mosaic emits as ONE byte-granularity sublane permute per
  output vreg (apply_vector_layout.cc `tpu_dynamic_gather_rule`; the
  16-bit form rides the 8-bit hardware path with interleaved 2x/2x+1
  byte indices). The per-element decode chain collapses to the nibble
  split plus one lookup.

  The indices must be int16: the gather lowering requires index and
  result bitwidths to match (infer_vector_layout.cc).
  `wrap_negative_indices=False` keeps `take_along_axis` from widening
  them to int32 (with wrapping on, jax normalizes every index to
  `int_dtype_for_dim` = int32 regardless of the incoming dtype), and
  `mode='promise_in_bounds'` keeps the bounds clamp out; nibbles are
  in [0, 16) by construction, so both promises hold.

  Everything around the lookup is `_dequantize_tile_planar_mxu`
  verbatim: the same `_expand_scale_codes` MXU expansion, the same
  bitcast+where exps construction, the same f32 multiply and bf16
  cast. The gathered bf16 entry widened to f32 equals `_decode_e2m1`'s
  f32 output exactly (every e2m1 value is bf16-exact), so this variant
  is BITWISE equal to `planar_mxu` (pinned exhaustively in
  moe_gmm_test and on-chip in the bench).

  Args:
    packed: `[tn, half_k]` planar-repacked uint8 nibble pairs.
    scales: `[tn, 2 * half_k / SCALE_GROUP]` uint8 e8m0 exponents.
    table: optional precomputed `_e2m1_gather_table(half_k)`. The kernels build
      it once per grid step, outside the rolled k walk, so the loop body does
      not lean on LICM to hoist the iota chain.

  Returns:
    `[tn, 2 * half_k]` bfloat16 weights in original contraction order.
  """
  half_k = packed.shape[1]
  if table is None:
    table = _e2m1_gather_table(half_k)
  codes = _expand_scale_codes(scales, 2 * half_k)
  exps = lax.bitcast_convert_type(codes << 23, jnp.float32)
  exps = jnp.where(codes == 0, jnp.float32(2.0**-127), exps)
  nibbles = packed.astype(jnp.int16)
  halves = [
      jnp.take_along_axis(
          table,
          nib,
          axis=0,
          mode='promise_in_bounds',
          wrap_negative_indices=False,
      ).astype(jnp.float32)
      for nib in (nibbles & 0x0F, nibbles >> 4)
  ]
  return jnp.concatenate(
      [halves[0] * exps[:, :half_k], halves[1] * exps[:, half_k:]], axis=-1
  ).astype(jnp.bfloat16)


def _dequantize_chunk_planar_t_gather(
    packed: jax.Array,
    scales: jax.Array,
    values: jax.Array | None = None,
) -> jax.Array:
  """Transposed-planar chunk decode with the SCALE inside the table.

  What is left of the dequant wall after `planar_gather`, per the P66
  LLO, is not the e2m1 decode (one vperm.slane per vreg) but the work
  AROUND it: the scale side's wide-plane cvt/shift/where feed (~29% of
  the loop body), the f32 multiply, and the f32->bf16 cast tail
  (~16%) - all per WEIGHT ELEMENT, m-independent. None of it can move
  to the accumulator (the e8m0 scale varies per (row, 32-group) inside
  a 512-deep chunk dot, and per-group MXU passes are gain-latch bound,
  the measured fp8-native failure), but ALL of it can move into the
  gather table: `vperm.slane`'s 32-byte table is PER LANE COLUMN, so
  once the tile is transposed (weight rows on lanes,
  `repack_planar_t_stack`), each group's 16-entry lane table can hold
  `e2m1(c) * 2^(s[n, g] - 127)` directly. The gather then emits the
  FINAL scaled bf16 weight - zero per-element post-ops - and the scale
  arithmetic runs on `16 * tn` table entries per group instead of
  `32 * tn` weight elements (half the multiplies, and the exps
  construction shrinks from the `[tn, chunk]` wide plane to the
  `[chunk/32, tn]` code plane, 1/32 the elements).

  BITWISE planar_gather by construction: each table entry is the same
  f32 multiply of the same two operands (the bf16-exact e2m1 value
  widened to f32, the same bitcast+where exps), rounded by the same
  f32->bf16 cast - done once per (group, row) instead of once per
  weight element - and the gather moves bits without touching them.
  Signed zeros, the code-0 subnormal scale and the code-255 `0 * inf`
  NaN all happen inside the table exactly as they happened on the wide
  plane (pinned over the full e2m1 x e8m0 grid in moe_gmm_test and on
  chip in the bench).

  Args:
    packed: `[chunk/2, tn]` uint8, one planar_t window (byte row j holds
      contraction elements j (low nibble) and j + chunk/2 (high) of each lane's
      weight row).
    scales: `[chunk/32, tn]` uint8 e8m0 exponents (group g = row g).
    values: optional `[16, tn]` f32 e2m1 value plane
      (`_e2m1_gather_table(tn).astype(f32)`); the kernels build it once per grid
      step, outside the rolled k walk. None builds it here.

  Returns:
    `[chunk, tn]` bfloat16 scaled weights in original contraction order,
    k on the second-minor dim (the `lhs [tm, chunk] @ rhs [chunk, tn]`
    dot orientation).
  """
  half_k, tn = packed.shape
  groups = scales.shape[0]
  if groups % 2:
    raise ValueError(
        f'planar_t windows split {groups} scale groups across two nibble'
        ' planes; the count must be even (the repack guarantees it for'
        ' every legal window).'
    )
  if values is None:
    values = _e2m1_gather_table(tn).astype(jnp.float32)

  # The scaled table, [groups * 16, tn]: entry [16g + c, n] holds
  # e2m1(c) * 2^(scales[g, n] - 127) as bf16. Built with the exact ops
  # the per-element path used (bitcast+where exps, f32 multiply, bf16
  # cast), so every entry's bits match the value the wide plane would
  # have produced for that (code, scale) pair.
  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)
  table = (
      (values[None, :, :] * exps[:, None, :])
      .astype(jnp.bfloat16)
      .reshape(groups * 16, tn)
  )

  nib = packed.astype(jnp.int16)
  lo = nib & 0x0F  # Window rows [0, chunk/2): groups [0, groups/2).
  hi = nib >> 4  # Window rows [chunk/2, chunk): groups [groups/2, groups).
  gh = groups // 2
  rows_per_group = half_k // gh  # 32: one scale group per nibble plane slab.
  pieces = []
  for g in range(groups):
    plane = lo if g < gh else hi
    r0 = (g % gh) * rows_per_group
    piece = jnp.take_along_axis(
        table[g * 16 : (g + 1) * 16, :],
        plane[r0 : r0 + rows_per_group, :],
        axis=0,
        mode='promise_in_bounds',
        wrap_negative_indices=False,
    )
    pieces.append(piece)
  return jnp.concatenate(pieces, axis=0)


# The e2m1 nibble -> f8e4m3b11fnuz byte relabel, as the (sign, magnitude)
# arithmetic the kernels build from an iota (Pallas cannot capture array
# constants). e4m3b11fnuz is bias-11 with no negative zero: magnitude m
# in [2, 7] lands at byte 4 * m + 80, magnitude 1 (the value 0.5, which
# renormalizes) at 0x50, magnitude 0 at 0x00 for BOTH signs (the fnuz
# 0x80 slot is NaN, so e2m1's -0 folds onto +0 - the only value the
# relabel does not carry, and one the dequant path never distinguishes
# because -0 * scale = -0 contributes nothing to any dot).
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
# folded code stays normal, and e2m1's smallest magnitude (0.5) encodes at
# E = 10, leaving 9 octaves of headroom. The real checkpoint's p99 spread
# is 2; `planar_t_b11_window_ok` is how a caller proves it rather than
# assuming it.
B11_FOLD_WINDOW = 9


def has_materializable_values(x) -> bool:
  """True when `x` holds real values a host-side gate can actually read.

  The complement of the jit-trace path, and the correct guard for any
  load-time check. `repack_planar_stack` distinguishes THREE input
  classes - tracers, `np.ndarray`, and lazy host array-likes
  (tensorstore / memmap views) - so a gate written as
  `isinstance(x, np.ndarray)` silently skips the third, which is
  precisely what a streaming Orbax load hands the projection. Only a
  tracer genuinely has no values.
  """
  return not isinstance(x, jax.core.Tracer)


def planar_t_scale_spread(scales, chunk: int = PLANAR_WINDOW) -> int:
  """Widest per-window e8m0 octave spread in `scales` (the fold demand).

  Shared by every folded-f8 planar_t arm: the fold rewrites a group's
  code as `w * 2^-delta` with `delta = s_max - s` taken over the window,
  so the largest `delta` any window asks for is exactly this spread. A
  mode is exact iff its own fold bound covers it.

  Args:
    scales: e8m0 codes in ANY of the layouts this module uses; only the
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


def planar_t_b11_window_ok(scales, chunk: int = PLANAR_WINDOW) -> bool:
  """True when every window column's e8m0 spread fits `B11_FOLD_WINDOW`.

  Args:
    scales: e8m0 codes, any layout `planar_t_scale_spread` accepts.
    chunk: the repack window in contraction elements.

  Returns:
    Whether `planar_t_b11` decodes these weights exactly. A False here
    means the arm would silently mis-decode the wide columns, so callers
    gate on it before selecting the mode.
  """
  return planar_t_scale_spread(scales, chunk) <= B11_FOLD_WINDOW


def _dequantize_chunk_planar_t_b11(
    packed: jax.Array,
    scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Transposed-planar chunk -> f8e4m3b11 codes + one post-dot column scale.

  SILICON STATUS: bitwise planar_t_gather on real ghostlite (the perf
  A/B pins it per covered row at decode sizes 1, 2, 4 and on the full
  buffer) PROVIDED the fold window holds - `planar_t_b11_window_ok`,
  enforced at conversion. A violated window is not a graceful
  degradation: the wide columns mis-decode entirely, which is exactly
  how it first showed up here (a bench fixture drawing 14 octaves of
  e8m0 against a 9 octave window looked like a compiler bug until the
  unit bisects came back exact). MEASURED VALUE: the loop body drops
  38.6% but the prefill wall only 4.4%, i.e. after the round-2 arm this
  kernel is no longer VPU-issue-bound; the decode-shape win cannot be
  resolved by the bench (dispatch floor) and needs a live A/B.

  The dequant-DELETION arm. planar_t_gather still materializes bf16
  weights with the VPU (its remaining 43 us/slab wall is per-element
  work at the VALU issue bound); this variant never materializes them
  at all: it relabels each e2m1 nibble into its EXACTLY EQUAL
  f8e4m3b11fnuz byte and hands the codes to the MXU, whose GLC gain
  latch expands f8e4m3b11 -> bf16 in HARDWARE
  (`vmatprep.subr.f8e4m3b11.bf16`, the format prod's GLP serving uses on
  its own weight path).

  CHIP SCOPE - THE HARDWARE EXPANSION ABOVE IS A GHOSTLITE FACT, NOT A
  PORTABLE ONE. The two serving chips take OPPOSITE fp8 formats.
  GhostLite's native spelling is b11 and it has no e4m3fn support;
  GHOSTFISH IS THE MIRROR and has NO GAIN LATCH FOR b11 AT ALL
  (`SupportsGainLatchMode` returns FALSE for every
  `k*XposeF8E4M3B11ToBf16` case in target_ghostfish.h, read by RETURN
  ARM rather than by case label, which is where this reads wrong;
  target_ghostfish.h also carries the e4m3fn VpackFormats that
  target_ghostlite.h has zero of). So on gf the widen is SOFTWARE and
  the whole "dequant-DELETION" argument above does not apply: this
  variant's value proposition is GLP-only as written.
  Measured on GF at batch 1, same dot spelling: b11 loses 672
  convolutions against a GF bf16 control that loses NONE, and loses
  MORE than e4m3fn does (-465), so on gf at that batch the dtype is
  genuinely the cause and b11 is the worst of the three. GF at batch
  >= 2 is UNVERIFIED - nobody has run it - so this is deliberately NOT
  a claim that b11 loses the MXU on gf generally.
  A gf lane wanting this mechanism should spell the planes e4m3fn
  instead; `_codes_to_f8` (:1911) already does exactly that relabel in
  this file. No
  chip guard is asserted here on purpose: a deliberate b11-on-GF
  reproduction arm exists and a refusal would break it.

  What that buys per chunk step: the gather output
  halves (u8, not bf16), the index path becomes the NATIVE byte gather
  (8-bit indices need none of the 16-bit emulation arithmetic), the
  f32->bf16 cast tail disappears, and the scale multiply leaves the
  per-element path entirely.

  EXACTNESS (why this is a bitwise-class arm, not a quality arm). All 16
  e2m1 values embed exactly in e4m3b11fnuz (bias 11 covers 2^-1..2^2, and
  a 3-bit mantissa covers e2m1's 1 bit; verified over the whole grid in
  moe_gmm_test). The per-group e8m0 scale splits into a per-window
  maximum and a per-group DELTA: `w * 2^(s - 127) = (w * 2^-delta) *
  2^(s_max - 127)` with `delta = s_max - s`. The folded code `w *
  2^-delta` stays exactly representable in e4m3b11fnuz while the folded
  code stays NORMAL, the expansion to bf16 is exact (3-bit mantissa
  into 7, by the latch on GLP or in software on gf - the EXACTNESS
  argument is arithmetic and holds either way; only the COST differs by
  chip), and the residual `2^(s_max - 127)` is a power of two applied
  ONCE to the f32 accumulator, where it is exact. So every
  product, every partial sum and the result are bit-identical to
  planar_t_gather's, on any window inside the fold bound.

  THAT BOUND IS `B11_FOLD_WINDOW` = 9, AND IT IS PER-MAGNITUDE-MINIMUM,
  not per-magnitude-maximum. The fold is an integer subtraction on the
  exponent FIELD, so it is exact exactly while `E - delta >= 1`. e2m1's
  LARGEST magnitude (6 = 1.5 x 2^2) sits at E = 13 and tolerates delta
  <= 12; its SMALLEST (0.5 = 2^-1) sits at E = 10 and tolerates only
  delta <= 9, and a window is decoded correctly only if EVERY code in
  it is, so 9 binds. (An earlier revision of this docstring quoted the
  12: that is the loosest per-magnitude bound, and taking it as the
  window criterion would silently mis-decode every window with spread
  10..12 - the field underflows into the sign bit rather than rolling
  into the subnormal encoding that would represent those values.)
  `moe_gmm_test` pins both sides of the boundary. The property is
  checkable on the loaded weights (the checkpoint's p99 spread is 2),
  and `planar_t_b11_window_ok` is how a caller proves it.

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
        f'planar_t windows split {groups} scale groups across two nibble'
        ' planes; the count must be even.'
    )
  codes32 = scales.astype(jnp.int32)
  s_max = jnp.max(codes32, axis=0, keepdims=True)  # [1, tn]
  delta = s_max - codes32  # [groups, tn], 0 at the window max
  wscale = jnp.where(
      s_max == 0,
      jnp.float32(2.0**-127),
      lax.bitcast_convert_type(s_max.astype(jnp.uint32) << 23, jnp.float32),
  )
  # The folded table, built with INTEGER arithmetic on the e4m3b11 byte.
  # GLC has no f8 convert instruction (its only f8 hardware is the MXU
  # expand latch), so spelling this as `.astype(f8e4m3b11fnuz)` lowers
  # to a ~20-op software conversion per vreg: measured at +900 bundles
  # of vsel/vcmp on this very loop, which ate the whole arm. Scaling by
  # 2^-delta is instead EXACTLY an exponent-field subtraction - the
  # field sits at bits 3..6, so the folded byte is `base - (delta << 3)`
  # - with the zero entries pinned at the all-zero byte. That is exact
  # while the result stays normal, i.e. for delta <= 9 (e2m1's smallest
  # magnitude 0.5 encodes at E = 10), the window precondition the caller
  # pins.
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
      mode='promise_in_bounds',
      wrap_negative_indices=False,
  )
  codes = lax.bitcast_convert_type(
      out.reshape(groups * rows, tn), jnp.float8_e4m3b11fnuz
  )
  return codes, wscale


# The e2m1 nibble -> float8_e4m3fn byte relabel: the GHOSTFISH mirror of
# `_b11_relabel_table`, and the spelling `_dequantize_chunk_planar_t_b11`
# names for a gf lane ("should spell the planes e4m3fn instead"). Same
# (sign, magnitude) arithmetic, two different constants, because e4m3fn
# is bias 7 where e4m3b11fnuz is bias 11: magnitude m in [2, 7] lands at
# byte 4 * m + 48, magnitude 1 (the value 0.5, which renormalizes to
# mantissa 0) at 0x30, magnitude 0 at 0x00.
#
# e4m3fn DOES have a -0 encoding (0x80) where fnuz spends that slot on
# NaN, but this table still folds e2m1's -0 onto +0, deliberately: the
# fold below is an integer subtraction on the exponent field and 0x80
# is not exponent-field data, so a signed zero would have to be masked
# out of the subtraction anyway. Dropping it is free - the kernel's f32
# accumulator starts at +0.0 and IEEE round-to-nearest gives
# `+0.0 + -0.0 == +0.0`, so a -0 weight and a +0 weight contribute the
# identical bits to every partial sum.
_F8FN_MAG1_BYTE = 0x30
_F8FN_MAG_BASE = 48


def _f8fn_relabel_table(tn: int) -> jax.Array:
  """`[16, tn]` uint8: each e2m1 nibble's exactly-equal float8_e4m3fn byte."""
  c = lax.broadcasted_iota(jnp.int32, (16, tn), 0)
  mag = c & 7
  body = jnp.where(mag == 1, _F8FN_MAG1_BYTE, 4 * mag + _F8FN_MAG_BASE)
  return jnp.where(mag == 0, 0, ((c >> 3) << 7) | body).astype(jnp.uint8)


# The exact fold window for `planar_t_f8fn`, derived the same way
# `B11_FOLD_WINDOW` is and NARROWER for the same reason it is: the bound
# is PER-MAGNITUDE-MINIMUM, because a window decodes correctly only if
# EVERY code in it does. The fold is an integer subtraction on the
# exponent FIELD, exact while `E - delta >= 1`. e2m1's largest magnitude
# (6 = 1.5 x 2^2) sits at E = 9 in e4m3fn and would tolerate delta <= 8;
# its smallest (0.5 = 2^-1) sits at E = 6 and tolerates only delta <= 5,
# so 5 binds. That is 4 octaves tighter than b11's 9 - the whole
# difference being the exponent bias, 7 against 11 - and it is why this
# arm carries its own gate rather than reusing the b11 one. The real
# checkpoint's p99 window spread is 2, so 5 still covers it with room;
# `planar_t_f8fn_window_ok` is how a caller proves that rather than
# assuming it. A violated window is NOT graceful: the field underflows
# into the sign bit and the wide columns mis-decode entirely.
#
# 5 IS THE BARE-SUBTRACTION BOUND, NOT A CEILING, so "the window is too
# tight" must never be the reason this arm is refused. Two reliefs, both
# already understood, neither needed at a p99 spread of 2. (1) DIRECTION:
# the table occupies E in [6, 9] and e4m3fn's NaN is only S.1111.111,
# which the table's mantissa field (0 or 4, never 7) cannot reach, so
# E may legally run to 15 and anchoring the post-dot scale at the window
# MINIMUM instead - folding UP by `s - s_min` - buys 6. The foldable
# span is `(E_lo - 1) + (15 - E_hi)` = 11 for either bias; the bias only
# decides how the 11 splits (b11 is 9 down / 2 up, e4m3fn 5 down / 6 up).
# (2) SUBNORMAL LANDING: `_codes_to_f8_folded` already re-encodes
# `1.MMM * 2^(ep-7)` as `MMM' * 2^-9` when the field would underflow,
# which holds both binding magnitudes exact out to delta 8. It is far
# cheaper here than where it lives today - a `[16, tn]` table per group
# rather than a select chain per weight element.
F8FN_FOLD_WINDOW = 5


def planar_t_f8fn_window_ok(scales, chunk: int = PLANAR_WINDOW) -> bool:
  """True when every window column's e8m0 spread fits `F8FN_FOLD_WINDOW`.

  Args:
    scales: e8m0 codes, any layout `planar_t_scale_spread` accepts.
    chunk: the repack window in contraction elements.

  Returns:
    Whether `planar_t_f8fn` decodes these weights exactly.
  """
  return planar_t_scale_spread(scales, chunk) <= F8FN_FOLD_WINDOW


def _dequantize_chunk_planar_t_f8fn(
    packed: jax.Array,
    scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Transposed-planar chunk -> float8_e4m3fn codes + post-dot column scale.

  The GHOSTFISH twin of `_dequantize_chunk_planar_t_b11`: identical
  structure, identical exactness argument, e4m3fn in place of
  e4m3b11fnuz because the two serving chips take OPPOSITE fp8 formats.
  Ghostfish's `SupportsMatmulDataFormat` admits {f32, bf16, f8e4m3fn,
  f8e5m2} and has no b11 gain latch at all; GhostLite is the mirror. So
  b11 is the GLP dequant-deletion arm and this is the GF one, and
  neither is portable to the other chip.

  WHY THIS EXISTS AS A SEPARATE MODE rather than as a flag on the
  existing gf fp8 machinery: `gmm_fp8_native` relabels e2m1 to e4m3fn
  too, but it consumes the CHECKPOINT (interleaved) nibble order, so
  `mxfp4_gmm` refuses it against planar or planar_t weights and the
  config layer refuses the pairing outright. Every gf serving preset
  pins `mxfp4_weight_layout='planar_t'`, which made the whole native-f8
  family - and with it prod's fp8-activation posture - unreachable from
  the shipped presets BY CONSTRUCTION. This mode closes that: it reads
  the planar_t bytes the presets already store, so nothing about the
  checkpoint, the stack layout or the HBM residency changes.

  WHAT IT UNLOCKS. On its own it is the gf dequant-deletion arm (the
  weights never materialize as bf16; the MXU takes the codes). Its
  other purpose is that it puts an f8 operand on the WEIGHT side, which
  is the PRECONDITION for quantizing the activation side: an f8 lhs
  against a bf16 rhs is not a narrow matmul at all - canonicalize_
  mosaic's f8-lhs rule inserts a `tpu::ExtFOp` widening operand 0,
  comment included ("no hardware support for f8 @ bf16/f32") - so lhs
  quant without an f8 weight is pure rounding loss at zero rate
  benefit. That e4m3 x e4m3 pair is prod's GF FFW posture, reached from
  the other side: prod stores int4 and megablox widens it to e4m3fn
  in-kernel at generation 7 (Ghostfish), so prod's gf dot is already an
  e4m3 x e4m3 one.

  DO NOT PRICE THIS AT THE PAPER RATE. target_ghostfish.h FlopsPerSecond
  does say f8 1992e12 against bf16 996e12, but the one banked GF
  silicon receipt for a native f8 x f8 gmm has it landing WITHIN 1% of
  native bf16 at every shape and both phases (sponge2/1a084ed2, quoted
  at `_subchannel_dot_f8`). The 2x is a gain-path property this kernel
  class has never been shown to reach, and the kb14 receipt (a 38.6%
  body cut bought 4.4% of the wall) says the loop is no longer
  VPU-issue-bound either. So the honest claim is "exact, native, and
  plausibly single-digit percent", and the open measurement is a
  PREFILL-shape A/B against planar_t_gather at a named operating point:
  m >= 512 is the regime where the MXU is the binding resource and it
  is exactly the regime nobody has run a narrow weight dtype in on gf
  (the b11 batch-1 control is the only gf dtype receipt and it says
  batch >= 2 is unverified).

  EXACTNESS. All 16 e2m1 values embed exactly in e4m3fn (bias 7 covers
  2^-1..2^2 and a 3-bit mantissa covers e2m1's 1 bit). The per-group
  e8m0 scale splits into a per-window maximum and a per-group DELTA:
  `w * 2^(s - 127) = (w * 2^-delta) * 2^(s_max - 127)` with
  `delta = s_max - s`. The folded code stays exactly representable
  while it stays NORMAL (the `F8FN_FOLD_WINDOW` precondition), the
  widen to the MXU's internal format is exact (3 mantissa bits into 7 or
  more), and the residual `2^(s_max - 127)` is a power of two applied
  ONCE to the f32 accumulator, where it is exact. So on any window
  inside the fold bound every product, every partial sum and the result
  are bit-identical to `planar_t_gather`'s. The mode is a SPEED arm,
  not a quality arm; all of this lane's quality risk sits on the
  activation side, not here.

  Args:
    packed: `[chunk/2, tn]` uint8 planar_t window bytes.
    scales: `[chunk/32, tn]` uint8 e8m0 exponents (group g = row g).

  Returns:
    `(codes, wscale)`: `[chunk, tn]` float8_e4m3fn weights folded by the
    per-column window delta, and the `[1, tn]` float32 window scale the
    caller multiplies into the f32 accumulator after the dot.
  """
  half_k, tn = packed.shape
  groups = scales.shape[0]
  if groups % 2:
    raise ValueError(
        f'planar_t windows split {groups} scale groups across two nibble'
        ' planes; the count must be even.'
    )
  codes32 = scales.astype(jnp.int32)
  s_max = jnp.max(codes32, axis=0, keepdims=True)  # [1, tn]
  delta = s_max - codes32  # [groups, tn], 0 at the window max
  wscale = jnp.where(
      s_max == 0,
      jnp.float32(2.0**-127),
      lax.bitcast_convert_type(s_max.astype(jnp.uint32) << 23, jnp.float32),
  )
  # Integer arithmetic on the e4m3fn byte, for the same reason the b11
  # twin uses it: an `.astype(float8_e4m3fn)` spelling lowers to a
  # software convert chain per vreg, which on that arm ate the whole
  # win. The exponent field sits at bits 3..6, so scaling by 2^-delta is
  # exactly `base - (delta << 3)`, with the zero entries pinned.
  base = _f8fn_relabel_table(tn).astype(jnp.int32)  # [16, tn]
  folded = base[None, :, :] - (delta[:, None, :] << 3)
  table_u8 = jnp.where(base[None, :, :] == 0, 0, folded).astype(jnp.uint8)

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
      mode='promise_in_bounds',
      wrap_negative_indices=False,
  )
  codes = lax.bitcast_convert_type(
      out.reshape(groups * rows, tn), jnp.float8_e4m3fn
  )
  return codes, wscale


# ---------------------------------------------------------------------
# LOAD-TIME f8 EXPERT WEIGHT PLANES.
#
# `planar_t_b11` computes its f8e4m3b11 operand IN THE KERNEL, once per
# expert visit: build the folded 16-entry table, gather the nibbles
# through it (the 384 surviving vperms), bitcast. The bytes it produces
# are a pure function of the loaded weights, so they can equally be
# produced ONCE at checkpoint conversion and stored - which is what
# `EXPERT_PLANES_F8_LOAD_TIME` does. The kernel then loads the operand
# and dots it, and the whole relabel/gather chain leaves the loop.
#
# THE PRICE, stated here because it decides the arm (see the byte
# ledger in config.expert_plane_ledger and the 2026-08-08 PROGRESS
# entry): an f8 plane is 8 bits/weight against mxfp4's 4.25 (nibble +
# e8m0/32), so HBM residency and - the part that matters - the PER
# VISIT WEIGHT STREAM both go up 1.886x. The kb14 receipt (a 38.6%
# kernel-body cut bought 4.4% of the wall) says this gmm is no longer
# VPU-issue-bound, so the body deletion this buys is small while the
# extra stream lands on the bucket that IS binding. Default OFF, and
# the config layer REFUSES the lanes where it does not even fit.
EXPERT_PLANES_MXFP4 = 'mxfp4'
EXPERT_PLANES_F8_LOAD_TIME = 'f8_load_time'
# The GHOSTFISH twin: the same load-time fold spelled float8_e4m3fn (the
# `planar_t_f8fn` byte, `_f8fn_relabel_table` minus `delta << 3`), for
# the chip whose MXU takes e4m3fn and has no b11 latch. Same 1.886x
# bytes; what changes is the trade on the OTHER side, measured by the
# W44 slot census on the streamed rung 6c kernel: its visit body is
# VALU bound at 97 pct with 95 pct of the VALU ops in the nibble -> f8
# fold chain, so deleting the chain flips the kernel to HBM bound at
# roughly 0.6x the visit (W44_GMM_SLOT_CENSUS.md section 5).
EXPERT_PLANES_F8FN_LOAD_TIME = 'f8fn_load_time'
EXPERT_PLANE_MODES = (
    EXPERT_PLANES_MXFP4,
    EXPERT_PLANES_F8_LOAD_TIME,
    EXPERT_PLANES_F8FN_LOAD_TIME,
)
# The f8 dtype a load-time plane mode's bytes spell.
EXPERT_PLANE_DTYPES = {
    EXPERT_PLANES_F8_LOAD_TIME: jnp.float8_e4m3b11fnuz,
    EXPERT_PLANES_F8FN_LOAD_TIME: jnp.float8_e4m3fn,
}


def _b11_fold_bytes(codes, delta, xp):
  """e2m1 nibble codes -> folded e4m3b11fnuz bytes (the kernel's math).

  Mirrors `_b11_relabel_table` followed by the `base - (delta << 3)`
  exponent-field subtraction inside `_dequantize_chunk_planar_t_b11`,
  INCLUDING the uint8 wrap an out-of-window delta produces, so an
  offline plane and the in-kernel table agree bit for bit on any input
  rather than only on in-window ones.

  Args:
    codes: integer array of e2m1 nibble codes (0..15).
    delta: `s_max - s` per scale group, broadcastable against `codes`.
    xp: `np` or `jnp`, the array module to build with.

  Returns:
    uint8 array of e4m3b11fnuz bytes, shaped by the broadcast.
  """
  c = codes.astype(xp.int32)
  mag = c & 7
  body = xp.where(mag == 1, _B11_MAG1_BYTE, 4 * mag + _B11_MAG_BASE)
  base = xp.where(mag == 0, 0, ((c >> 3) << 7) | body)
  folded = base - (delta.astype(xp.int32) << 3)
  return xp.where(base == 0, 0, folded).astype(xp.uint8)


def _f8fn_fold_bytes(codes, delta, xp):
  """e2m1 nibble codes -> folded float8_e4m3fn bytes (the kernel's math).

  `_b11_fold_bytes` with the e4m3fn constants: mirrors
  `_f8fn_relabel_table` followed by the `base - (delta << 3)` subtraction
  inside `_dequantize_chunk_planar_t_f8fn`, uint8 wrap included, so an
  offline plane and the in-kernel table agree bit for bit on any input.
  Exact only inside `F8FN_FOLD_WINDOW` (the converter gates on it).

  Args:
    codes: integer array of e2m1 nibble codes (0..15).
    delta: `s_max - s` per scale group, broadcastable against `codes`.
    xp: `np` or `jnp`, the array module to build with.

  Returns:
    uint8 array of float8_e4m3fn bytes, shaped by the broadcast.
  """
  c = codes.astype(xp.int32)
  mag = c & 7
  body = xp.where(mag == 1, _F8FN_MAG1_BYTE, 4 * mag + _F8FN_MAG_BASE)
  base = xp.where(mag == 0, 0, ((c >> 3) << 7) | body)
  folded = base - (delta.astype(xp.int32) << 3)
  return xp.where(base == 0, 0, folded).astype(xp.uint8)


# The offline fold per plane dtype (`EXPERT_PLANE_DTYPES` values).
_PLANE_FOLD_BYTES = {
    jnp.dtype(jnp.float8_e4m3b11fnuz): _b11_fold_bytes,
    jnp.dtype(jnp.float8_e4m3fn): _f8fn_fold_bytes,
}


def _plane_fold_bytes(plane_dtype):
  key = jnp.dtype(plane_dtype)
  if key not in _PLANE_FOLD_BYTES:
    raise ValueError(
        'load-time planes spell float8_e4m3b11fnuz or float8_e4m3fn, got'
        f' {key}.'
    )
  return _PLANE_FOLD_BYTES[key]


def _planar_t_f8_windows(
    pt_packed, pt_scales, window: int, plane_dtype=jnp.float8_e4m3b11fnuz
):
  """One slab of `repack_planar_t_f8_stack` (planar_t bytes in)."""
  fold_bytes = _plane_fold_bytes(plane_dtype)
  xp = jnp if isinstance(pt_packed, (jax.Array, jax.core.Tracer)) else np
  half_k, n = pt_packed.shape[-2], pt_packed.shape[-1]
  k = 2 * half_k
  groups = window // SCALE_GROUP
  gh = groups // 2
  rows = (window // 2) // gh
  steps = k // window
  lead = pt_packed.shape[:-2]
  # Per window: the e8m0 maximum leaves the plane (it rides the f32
  # accumulator once, as `wscale`), the per-group residual becomes the
  # byte's exponent-field delta.
  sg = pt_scales.reshape(lead + (steps, groups, n)).astype(xp.int32)
  s_max = xp.max(sg, axis=-2, keepdims=True)
  delta = s_max - sg
  # Nibble planes in the kernel's output order: group g < gh reads the
  # LOW nibble of byte row g*rows + r, g >= gh the HIGH nibble of byte
  # row (g - gh)*rows + r, and output row g*rows + r owns contraction
  # element g*SCALE_GROUP + r of the window - the same correspondence
  # `_dequantize_chunk_planar_t_b11` builds with its concatenated idx.
  pw = pt_packed.reshape(lead + (steps, gh, rows, n))
  nib = xp.concatenate([pw & 0x0F, pw >> 4], axis=-3)
  planes = fold_bytes(nib, delta[..., None, :], xp)
  return (
      planes.reshape(lead + (k, n)),
      s_max.astype(xp.uint8).reshape(lead + (steps, n)),
  )


def repack_planar_t_f8_stack(
    packed,
    scales,
    window: int = PLANAR_WINDOW,
    plane_dtype=jnp.float8_e4m3b11fnuz,
):
  """Checkpoint MXFP4 pair -> load-time f8 planes + window scales.

  The offline twin of `_dequantize_chunk_planar_t_b11` (`plane_dtype`
  float8_e4m3b11fnuz, the default) or `_dequantize_chunk_planar_t_f8fn`
  (float8_e4m3fn): every output byte is the byte that kernel's folded
  table would have gathered, and every output scale is that kernel's
  `wscale` in its e8m0 code form. `moe_gmm_test` pins the equality
  window by window for both dtypes, so the paths cannot drift.

  PRECONDITION: the fold is exact only while a window's e8m0 spread
  fits the dtype's fold window (`B11_FOLD_WINDOW`, `F8FN_FOLD_WINDOW`).
  Unlike the runtime arm - where a violation merely mis-decodes at every
  visit - here the loss is BAKED INTO THE CHECKPOINT, so
  `checkpoint_converter` gates on `planar_t_b11_window_ok` /
  `planar_t_f8fn_window_ok` before calling this and refuses by module
  name rather than writing a silently wrong plane.

  Trace safety: same contract as `repack_planar_stack` (the serving
  exporter runs the projection under a jit trace). jax values and
  tracers take the pure-jnp path; host arrays take a bounded slab walk,
  which matters more here than for the nibble repack because the f8
  output is 2x the input and the fold runs in int32.

  Args:
    packed: `[..., N, K/2]` uint8 checkpoint-order nibble pairs.
    scales: `[..., N, K/32]` uint8 e8m0 exponents.
    window: the planar window in contraction elements; must equal the kernel k
      chunk (`PLANAR_WINDOW`).
    plane_dtype: the f8 format the bytes spell (`EXPERT_PLANE_DTYPES`).

  Returns:
    `(planes, window_scales)`: `[..., K, N]` uint8 bytes of `plane_dtype`
    in planar_t (k-major) order, and the `[..., 8 * K/window, N]` uint8
    window maxima leaf (row 8w = window w's e8m0 maximum, the other rows
    zero; `PLANE_SCALE_ROWS_PER_WINDOW`, `plane_window_maxima`).
  """
  _plane_fold_bytes(plane_dtype)  # Validates the dtype up front.
  if window % (2 * SCALE_GROUP):
    raise ValueError(
        f'planar window {window} must be a multiple of {2 * SCALE_GROUP} so'
        ' each nibble plane covers whole scale groups.'
    )
  k = 2 * packed.shape[-1]
  if k % window:
    raise ValueError(
        f'contraction {k} must be a multiple of the planar window {window}.'
    )
  if scales.shape[-1] * SCALE_GROUP != k:
    raise ValueError(
        f'scales cover {scales.shape[-1] * SCALE_GROUP} contraction elements,'
        f' expected {k}.'
    )
  if packed.shape[:-1] != scales.shape[:-1]:
    raise ValueError(
        f'packed {packed.shape} and scales {scales.shape} disagree outside'
        ' the contraction axis.'
    )
  if isinstance(packed, (jax.Array, jax.core.Tracer)):
    planes, maxima = _planar_t_f8_windows(
        repack_planar_t_stack(packed, window),
        transpose_scales_planar_t(scales),
        window,
        plane_dtype,
    )
    return planes, pad_plane_window_rows(maxima)
  packed = np.asarray(packed)
  scales = np.asarray(scales)
  if packed.ndim < 3:
    planes, maxima = _planar_t_f8_windows(
        repack_planar_t_stack(packed, window),
        transpose_scales_planar_t(scales),
        window,
        plane_dtype,
    )
    return planes, pad_plane_window_rows(maxima)
  # Host arrays are checkpoint-scale: a real gate leaf is u8[896, 3072,
  # 1792] (4.9 GiB in, 9.9 GiB out) and the int32 fold holds 4x the
  # slab. Walk the leading (expert) dim in bounded slabs and write into
  # the preallocated outputs so the projection's peak stays small.
  lead = packed.shape[:-2]
  n, half_k = packed.shape[-2], packed.shape[-1]
  flat_p = packed.reshape(-1, n, half_k)
  flat_s = scales.reshape(-1, n, scales.shape[-1])
  steps = k // window
  rows = steps * PLANE_SCALE_ROWS_PER_WINDOW
  out_p = np.empty((flat_p.shape[0], k, n), dtype=np.uint8)
  out_s = np.zeros((flat_p.shape[0], rows, n), dtype=np.uint8)
  per_slab = max(1, k * n)
  step = max(1, (256 * 1024 * 1024) // per_slab)
  for s in range(0, flat_p.shape[0], step):
    out_p[s : s + step], maxima = _planar_t_f8_windows(
        repack_planar_t_stack(flat_p[s : s + step], window),
        transpose_scales_planar_t(flat_s[s : s + step]),
        window,
        plane_dtype,
    )
    out_s[s : s + step, ::PLANE_SCALE_ROWS_PER_WINDOW, :] = maxima
  return out_p.reshape(lead + (k, n)), out_s.reshape(lead + (rows, n))


def dequantize_planar_t_f8_stack(
    planes,
    window_scales,
    window: int = PLANAR_WINDOW,
    plane_dtype=jnp.float8_e4m3b11fnuz,
) -> jax.Array:
  """f8 planes + window scales -> the dense weights they decode to.

  The gain latch's job done in software: the reference every non-kernel
  path (tests, the CPU fallback) uses to read a load-time plane. Value
  exact against the kernel - the widen f8 -> bf16 loses nothing (3
  mantissa bits into 7) and the window scale is a power of two applied
  once on the f32 accumulator.

  Args:
    planes: `[..., K, N]` uint8 bytes of `plane_dtype` (planar_t, k-major).
    window_scales: the `[..., 8 * K/window, N]` uint8 window maxima leaf
      (`repack_planar_t_f8_stack`).
    window: the planar window in contraction elements.
    plane_dtype: the f8 format the bytes spell (`EXPERT_PLANE_DTYPES`).

  Returns:
    `[..., K, N]` float32 dense weights in planar_t (k-major) order.
  """
  _plane_fold_bytes(plane_dtype)  # Validates the dtype.
  planes = jnp.asarray(planes)
  window_scales = jnp.asarray(window_scales)
  k, n = planes.shape[-2], planes.shape[-1]
  steps = k // window
  lead = planes.shape[:-2]
  wide = lax.bitcast_convert_type(planes, plane_dtype).astype(jnp.float32)
  s = plane_window_maxima(window_scales).reshape(lead + (steps, 1, n))
  exps = jnp.where(
      s == 0,
      jnp.float32(2.0**-127),
      lax.bitcast_convert_type(s.astype(jnp.uint32) << 23, jnp.float32),
  )
  return (wide.reshape(lead + (steps, window, n)) * exps).reshape(lead + (k, n))


# Dequant inner-loop variants. All decode a bitwise identical set
# of weights (`moe_gmm_test.FastDequantLosslessTest` pins that over the
# whole e2m1 x e8m0 grid); `perf/moe_gmm_bench_forge_test` A/Bs them.
# The `planar_*` variants read `repack_planar_stack`ed bytes (same
# weight VALUES, offline byte order) and emit ORIGINAL contraction
# order, so they are the only variants that need no lhs preparation.
DEQUANT_BASELINE = 'baseline'
DEQUANT_FAST = 'fast'
DEQUANT_SPLIT = 'split'
DEQUANT_SPLIT_LUT = 'split_lut'
DEQUANT_SPLIT_MXU = 'split_mxu'
DEQUANT_PLANAR_LUT = 'planar_lut'
DEQUANT_PLANAR_MXU = 'planar_mxu'
DEQUANT_PLANAR_GATHER = 'planar_gather'
DEQUANT_PLANAR_T_GATHER = 'planar_t_gather'
DEQUANT_PLANAR_T_B11 = 'planar_t_b11'
# The GHOSTFISH twin of `planar_t_b11`: same in-register relabel of the
# planar_t nibbles into an exactly-equal f8 byte with the per-group e8m0
# delta folded into the exponent field, spelled in float8_e4m3fn (the
# only narrow matmul format gf has) instead of e4m3b11fnuz (the only one
# glp has). Bitwise `planar_t_gather` inside `F8FN_FOLD_WINDOW`. Unlike
# `planar_t_f8` this changes NOTHING about the stored stack - it reads
# the same mxfp4 planar_t bytes the gf presets already ship.
DEQUANT_PLANAR_T_F8FN = 'planar_t_f8fn'
DEQUANT_PLANAR_T_F8 = 'planar_t_f8'
DEQUANT_PLANAR_T_B11_DEEP = 'planar_t_b11_deep'
# The deep pipeline with the GATHER body: same DEEP_STREAM_DEPTH
# outstanding weight-tile copies and the same unrolled k walk, but the
# chunk decode is `planar_t_gather`, which every chip runs. The deep
# win measured on GLP splits into a FORMAT half (the b11 fold) and a
# PIPELINE half (the DMA depth); only the first needs the b11 widen,
# so this mode carries the second to targets without it - Ghostfish
# above all, where `F8E4M3B11ToBf16` does not exist.
#
# QUALITY CLASS, measured not assumed: exact in interpret (the unit
# pin) and structurally the rolled gather's chunk step, but on
# GHOSTFISH SILICON the (bf16, bf16) dot rounds differently between
# the rolled and deep kernel structures - 1 bf16 ulp on ~0.02% of
# covered outputs per call, deterministic (the bench's single-call
# diagnostic; the f8fn fold pair on the same runs is bit-exact, which
# acquits the DMA schedule itself). So on gf this arm's serving claim
# is "bf16 rounding class", not "bitwise", and the fold-bodied deep
# modes are the bit-exact spellings.
DEQUANT_PLANAR_T_GATHER_DEEP = 'planar_t_gather_deep'
# The deep pipeline with the F8FN fold body: BOTH halves of the deep
# win on the chip the gather-bodied mode only half serves. planar_t_f8fn
# is Ghostfish's dequant-deletion arm (the e4m3fn relabel, bitwise
# planar_t_gather inside F8FN_FOLD_WINDOW) and this runs that exact
# chunk decode inside the DEEP_STREAM_DEPTH weight-copy pipeline, the
# way planar_t_b11_deep pairs the two halves on GhostLite. Chip scope
# is the CALLER's business exactly as for the rolled fold modes: the
# codes go straight to the MXU, so the config layer gates it with
# planar_t_f8fn (GMM_E4M3FN_MXU_OPERAND_IMPLS).
DEQUANT_PLANAR_T_F8FN_DEEP = 'planar_t_f8fn_deep'
# LOAD-TIME e4m3fn planes (`EXPERT_PLANES_F8FN_LOAD_TIME`): the Ghostfish
# twin of `planar_t_f8` - `packed` is `[E, K, N]` uint8 float8_e4m3fn
# bytes and `scales` the `[E, 8 * K/PLANAR_WINDOW, N]` window maxima leaf, so the
# chunk step is a bitcast, the dot and one post-dot column scale: the
# whole nibble -> e4m3fn fold chain (95 pct of the streamed 6c body's
# VALU ops, W44 census) leaves the visit. Bitwise `planar_t_f8fn` by
# construction (same bytes, same wscale, same accumulation order) at
# 1.886x the weight bytes per visit. The `_deep` spelling runs that
# chunk step inside the deep weight pipeline (the served GF form).
DEQUANT_PLANAR_T_F8FN_PLANE = 'planar_t_f8fn_plane'
DEQUANT_PLANAR_T_F8FN_PLANE_DEEP = 'planar_t_f8fn_plane_deep'
DEQUANT_MODES = (
    DEQUANT_BASELINE,
    DEQUANT_FAST,
    DEQUANT_SPLIT,
    DEQUANT_SPLIT_LUT,
    DEQUANT_SPLIT_MXU,
    DEQUANT_PLANAR_LUT,
    DEQUANT_PLANAR_MXU,
    DEQUANT_PLANAR_GATHER,
    DEQUANT_PLANAR_T_GATHER,
    DEQUANT_PLANAR_T_B11,
    DEQUANT_PLANAR_T_F8FN,
    DEQUANT_PLANAR_T_F8,
    DEQUANT_PLANAR_T_B11_DEEP,
    DEQUANT_PLANAR_T_GATHER_DEEP,
    DEQUANT_PLANAR_T_F8FN_DEEP,
    DEQUANT_PLANAR_T_F8FN_PLANE,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP,
)
# The variants whose weights land in permuted k order.
_SPLIT_MODES = (DEQUANT_SPLIT, DEQUANT_SPLIT_LUT, DEQUANT_SPLIT_MXU)
# The variants that expect planar-repacked weights (identity k order,
# no lhs permutation anywhere, kernel k chunk pinned to PLANAR_WINDOW).
PLANAR_MODES = (DEQUANT_PLANAR_LUT, DEQUANT_PLANAR_MXU, DEQUANT_PLANAR_GATHER)
# The variant that expects TRANSPOSED planar weights (`[E, K/2, N]`
# planar bytes + `[E, K/32, N]` scales, `repack_planar_t_stack` /
# `transpose_scales_planar_t`): the k chunk is still pinned to
# PLANAR_WINDOW and the lhs is still untouched, but the dequantized
# tile is emitted `[chunk, tn]` (k-major) so the e8m0 block scale can
# live INSIDE the vperm.slane gather table (see
# `_dequantize_chunk_planar_t_gather`).
PLANAR_T_MODES = (
    DEQUANT_PLANAR_T_GATHER,
    DEQUANT_PLANAR_T_B11,
    DEQUANT_PLANAR_T_F8,
    DEQUANT_PLANAR_T_B11_DEEP,
    DEQUANT_PLANAR_T_GATHER_DEEP,
    # Order matches `config.GMM_PLANAR_T_DEQUANT_IMPLS`, which
    # `ffw_test` pins tuple-for-tuple.
    DEQUANT_PLANAR_T_F8FN,
    DEQUANT_PLANAR_T_F8FN_DEEP,
    DEQUANT_PLANAR_T_F8FN_PLANE,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP,
)
# The planar_t variants whose chunk step emits f8 CODES plus a post-dot
# column scale instead of a materialized bf16 tile (the "fold" family).
# They share a kernel body: dequant -> (codes, wscale), dot, scale the
# f32 accumulator once. Chip scope differs per member and is the
# CALLER's business - `planar_t_b11` needs GhostLite's b11 gain latch,
# `planar_t_f8fn` needs Ghostfish's e4m3fn matmul format.
PLANAR_T_FOLD_MODES = (
    DEQUANT_PLANAR_T_B11,
    DEQUANT_PLANAR_T_F8FN,
)


def _planar_t_fold_chunk(dequant: str):
  """The `(packed, scales) -> (codes, wscale)` step for a fold mode."""
  if dequant == DEQUANT_PLANAR_T_B11:
    return _dequantize_chunk_planar_t_b11
  if dequant == DEQUANT_PLANAR_T_F8FN:
    return _dequantize_chunk_planar_t_f8fn
  raise ValueError(f'{dequant!r} is not a planar_t fold mode.')


# The deep-schedule family, each member mapped to its CHUNK-DECODE body
# (the rolled kernel whose exact chunk step the deep pipeline wraps).
# Both entry points dispatch on this map, so a deep mode that one entry
# served and the other silently rolled cannot exist by construction
# (the gather-bodied mode briefly had exactly that hole: mxfp4_gmm ran
# the deep call while the fused kernel fell through to a rolled gather
# rebuilding its value table per chunk, wearing the deep arm's name on
# half the pair). Ineligible shapes fall back to the body's rolled
# kernel, bitwise; only the DMA schedule differs.
DEEP_DEQUANT_BODIES = {
    DEQUANT_PLANAR_T_B11_DEEP: DEQUANT_PLANAR_T_B11,
    DEQUANT_PLANAR_T_GATHER_DEEP: DEQUANT_PLANAR_T_GATHER,
    DEQUANT_PLANAR_T_F8FN_DEEP: DEQUANT_PLANAR_T_F8FN,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP: DEQUANT_PLANAR_T_F8FN_PLANE,
}


# The deep-stream b11 variant: `planar_t_b11`'s exact chunk math (same
# codes, same wscale, same accumulation order, BITWISE the same
# outputs - `moe_gmm_test.DeepStreamParityTest` pins it against
# planar_t_gather cross-family) inside a hand-rolled weight pipeline:
# the [K, tn] slab tiles stream HBM->VMEM via DEEP_STREAM_DEPTH
# outstanding `make_async_copy`s per stack and the k walk is unrolled,
# instead of the BlockSpec pipeline's single outstanding block copy.
# Mechanism receipts (perf/moe_gmm_bench_forge_test, real glc): the
# decode wall is the weight stream x body serialization, not VALU and
# not bytes (pure-stream 6.6 us vs kernel 15.3 for the same slab;
# manual depth curve 537 GB/s serial -> 1081 at D4, knee at 4; overlap
# law wall ~ body + 0.65 x stream). Prototype-measured per-visit on
# silicon: w2 17.2 -> 10.0 us, fused 29.2 -> 15.5, pair 0.55x vs the
# served planar_t_gather. That ratio was measured on the PROTOTYPE
# structure (single visit per call); this production spelling adds the
# general/one-hot metadata walk and must be re-measured before its
# number is quoted. It has been, AT DECODE SHAPE (m=8, tn=512, the
# chained per-visit harness; a gmm receipt inherits the PHASE it was
# taken in and a prefill one would not price this): the bench's in-tree
# arms put the PRODUCTION dispatch at 10.7 us per w2 visit against 14.6
# for the rolled planar_t_b11 and 18.4 for the served planar_t_gather,
# and 15.7 / 21.4 / 29.2 on the fused arm, so the PAIR is 26.4 vs 47.6
# = 0.555x on real glc. The deep-vs-rolled separation on both arms is
# also the engagement receipt: a fallback would have read the rolled
# row. Prefill widths are a different machine (MXU bound at m >= 512)
# and are deliberately not served by this kernel, which is what the
# caller-side band split exists to guarantee.
#
# Band scoping lives in the CALLER (the width-keyed
# `gmm_dequant_impl_decode` config knob), NOT in a row bound here: the
# kiloblox layer hands BOTH bands a padded assignment tray (the served
# kb15 generate call is m=65,536 rows at tm=512, measured by a
# raise-with-values control after the whole-program code-size
# discriminator showed the deep kernels missing), so any m bound
# either starves the served decode call or admits prefill. The only
# hard limit the kernel itself owns is the per-tile f32 accumulator.
DEEP_STREAM_DEPTH = 4
DEEP_MAX_ACC_BYTES = 2 * 1024 * 1024
# Row tile slots of the deep kernel's STREAMED form (`stream_rows`): the
# visited `[tm, K]` lhs tile is double buffered so visit j + 1's rows
# land while visit j computes. The resident form holds the whole tray
# in VMEM instead (see `_deep_planar_t_b11_call`).
DEEP_ROW_TILE_SLOTS = 2
# Rows per PLANAR_WINDOW in a load-time plane's window maxima leaf
# (`[E, PLANE_SCALE_ROWS_PER_WINDOW * K/PLANAR_WINDOW, N]`): row 8w is
# window w's e8m0 maximum, rows 8w+1..8w+7 are zero and never read. The
# padding is the layout contract, not a convenience: XLA lays out an
# entry parameter by the (sublane, lane) pair with the least padded
# bytes (transfer_size_util.cc ChooseCompactOrSparseCoreLayoutForShape;
# rows below 8 pad to the next power of two, above to a multiple of 8)
# and Mosaic tiles a kernel operand by its second minor alone
# (infer_memref_layout.cc getTilingFactor: 4 rows below 5, 8 up to 31,
# 32 at multiples of 32). One row per window ([E, 3, N] for the down
# stack) put the EXPERT dim on the sublanes at rest and a 4-row tile on
# the kernel side, and XLA relayouted every scale operand per step (92
# pinned copies in the prefill chunk step, 276 in decode before the rank
# 3 form; w27's planes16s gate). With a multiple of 8 rows per expert the
# chooser keeps the rows on the sublanes for every expert count (S'E <=
# pad(E)S'), both tilings agree (8 with 8, 32k with 32k), and a window
# aligned F split (kiloblox_mp_axis) still lands on whole window groups,
# which padding the STEPS to 8 would break for the 3 + 3 down split.
# Cost: 8x the scale bytes per tile (1.6 pct of the plane slab), at most
# one extra u8 row tile per stack at rest.
PLANE_SCALE_ROWS_PER_WINDOW = 8
# The packed u8 row tile the deep kernel's VMEM model rounds a plane
# body's window maxima slab up to (8 sublanes x 4 bytes).
_PLANE_SCALE_TILE_ROWS = 32


def pad_plane_window_rows(window_maxima):
  """`[..., S, N]` window maxima -> the at rest `[..., 8S, N]` leaf.

  The traced path spells the pad as `lax.pad` (a reshape, a pad, a
  reshape): the serving exporter does not run the projection, roc's
  export planner abstract transposes its jaxpr per equation and only
  the primitives in its rule tables survive (`.at[].set` is a scatter,
  which has no rule and killed export xid 286020629 at planning;
  checkpoint_converter_test pins the plane arms' primitive set).
  """
  lead = window_maxima.shape[:-2]
  steps, n = window_maxima.shape[-2], window_maxima.shape[-1]
  flat = lead + (steps * PLANE_SCALE_ROWS_PER_WINDOW, n)
  row_pad = (0, PLANE_SCALE_ROWS_PER_WINDOW - 1)
  if isinstance(window_maxima, (jax.Array, jax.core.Tracer)):
    rows = window_maxima.reshape(lead + (steps, 1, n))
    pad = ((0, 0),) * len(lead) + ((0, 0), row_pad, (0, 0))
    return jnp.pad(rows, pad).reshape(flat)
  host = np.asarray(window_maxima)
  rows = np.zeros(lead + (steps, PLANE_SCALE_ROWS_PER_WINDOW, n), host.dtype)
  rows[..., 0, :] = host
  return rows.reshape(flat)


def plane_window_maxima(window_scales):
  """The at rest `[..., 8S, N]` leaf -> its `[..., S, N]` window maxima."""
  rows = window_scales.shape[-2]
  if rows % PLANE_SCALE_ROWS_PER_WINDOW:
    raise ValueError(
        f'plane window scales carry {PLANE_SCALE_ROWS_PER_WINDOW} rows per'
        f' window, got {rows} rows.'
    )
  return window_scales[..., ::PLANE_SCALE_ROWS_PER_WINDOW, :]


# The deep kernel never overrides Mosaic's VMEM limit. A per kernel
# `vmem_limit_bytes` above the program's `xla_tpu_scoped_vmem_limit_kib`
# compiles (the compiler honors the request) and then CORRUPTS VMEM at run
# time: the kernel's scratch beyond the program ceiling overlaps program
# buffers (Pallas documents the contract, jax/_src/pallas/mosaic/core.py
# `vmem_limit_bytes`; measured on gf at the m=1792 tray as a core halt in
# the down kernel under the served single expert form and garbage under
# the multi expert form, moe_deep_stream_pin_gf_forge_test). The kernel
# fits the ceiling by construction instead: `_deep_fit_tn` shrinks the
# column tile, and the streamed row form takes the tray out of VMEM.
# The planar_t variants whose weight refs are LOAD-TIME f8 planes rather
# than mxfp4 nibbles: `packed` is `[E, K, N]` uint8 f8 bytes
# (`repack_planar_t_f8_stack`) and `scales` is `[E, 8 * K/PLANAR_WINDOW, N]`
# e8m0 window maxima, so the chunk step is a bitcast and a dot with one
# post-dot column scale. `planar_t_f8` spells e4m3b11fnuz (bitwise
# `planar_t_b11`), the `planar_t_f8fn_plane` pair e4m3fn (bitwise
# `planar_t_f8fn`); each is the same bytes, wscale and accumulation
# order as its in-kernel fold at 1.886x the weight bytes per expert
# visit - the trade the config-layer ledger prices.
F8_PLANE_MODES = (
    DEQUANT_PLANAR_T_F8,
    DEQUANT_PLANAR_T_F8FN_PLANE,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP,
)
# The f8 dtype each plane mode's bytes spell (deep spelling included).
F8_PLANE_DTYPES = {
    DEQUANT_PLANAR_T_F8: jnp.float8_e4m3b11fnuz,
    DEQUANT_PLANAR_T_F8FN_PLANE: jnp.float8_e4m3fn,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP: jnp.float8_e4m3fn,
}

# DIAGNOSTIC ONLY, and deliberately NOT in `DEQUANT_MODES`: the split
# layout with the block scale never applied. It computes the WRONG
# answer and exists solely so the bench can price the per-group scale
# broadcast (the `[tn, chunk] <-> [tn, groups, 32]` regroup) by
# difference against `split_lut`. Nothing but the bench may pass it.
# The modes that put a float8_e4m3fn WEIGHT on the MXU (the fold spelled
# in registers or read from a load-time plane): the only ones an f8 lhs
# may meet without being widened back to bf16.
_E4M3FN_WEIGHT_MODES = (
    DEQUANT_PLANAR_T_F8FN,
    DEQUANT_PLANAR_T_F8FN_DEEP,
    DEQUANT_PLANAR_T_F8FN_PLANE,
    DEQUANT_PLANAR_T_F8FN_PLANE_DEEP,
)
DEQUANT_PROBE_NOSCALE = 'probe_noscale'
# DIAGNOSTIC ONLY, same caveat: split layout with the scale applied as
# a single per row lane broadcast instead of a per group one. Splits
# `probe_noscale`'s gap into 'the multiply' versus 'the regroup'.
DEQUANT_PROBE_SCALAR = 'probe_scalar'

# What the kernels use unless a caller overrides.
#
# Measured on ghostlite (`test_ab_dequant_inner_loop_variants`), fused
# gate/up at the real 3584 x 3072 geometry, versus `baseline`:
#
#   fast        +13.9%   SLOWER. Folding the block scale into the
#                        exponent field is provably lossless but is a
#                        measured REGRESSION: the shipped `_decode_e2m1`
#                        plus one f32 multiply is cheaper than the
#                        integer bit assembly that replaces it.
#   split       -38.5%   Interleave dropped, but paying `fast`'s
#                        arithmetic on top of it.
#   split_lut   -61.2%   Interleave dropped, shipped arithmetic kept.
#   split_mxu   -89.3%   ... and the per group scale broadcast moved
#                        off the VPU onto the idle MXU. 9.35x.
#
# Two things were costing everything, and neither was the arithmetic.
# First the cross-lane nibble interleave. Then, once that was gone, the
# per group scale regroup: reshaping the value plane to a 16 lane minor
# dim priced at 3.4 ms of the remaining 4.5 ms. Expanding the scale
# codes on the MXU instead lands them at full lane width and brings the
# kernel to 1.25 ms, within 24% of a probe that skips the scale
# entirely (1.01 ms) and 5x the pure bf16 matmul floor (0.25 ms).
#
# `split_mxu` is the default. It is bitwise equal to `split_lut`, and
# all the split variants decode a bitwise identical set of weights to
# `baseline`; they differ from it only in MXU reduction ORDER, the same
# perturbation changing `chunk` already causes, which moves 0.0085% of
# bf16 outputs by one ulp. `fast` remains the strictly bit-identical
# escape hatch if a caller ever needs the original reduction order.
#
# `planar_gather` (2026-08-07, the decode-floor lever) attacks what is
# left after split_mxu/planar_mxu: the per-element e2m1 ARITHMETIC.
# The decode-floor-2 map measured the decode-band gmm pair
# dequant-bound at ~176 Gelem/s of expansion (34.4 ms = 30% of the kb6
# step, m-independent), and the ghostlite LLO shows why: the
# planar_mxu inner k-loop body is 4,561 bundles per chunk step at
# ~87% vector-ALU slot occupancy (per chunk step: 2,306 vand, 2,049
# vsel, 1,546 vcmp, 1,152+1,024 vcvt, 1,024 vshra, 516 vpow2 on the
# EUP, ...) - the ALU-slot roofline, 8.9 bundles per 1,024 weights =
# 206 Gelem/s static, 176 live. planar_gather replaces `_decode_e2m1`
# with the VPU's in-register 32-byte table lookup (`vperm.slane`,
# go/glc-tc-isa#permute-between-sublanes, one ALU slot per vreg;
# reached as jnp.take_along_axis -> tpu.dynamic_gather -> byte
# sublane permute, with the 16-bit form riding the 8-bit hardware
# path): the same loop body drops to 2,807 bundles (-38.5%; 257
# vperm.slane + 768 index-prep ops replace ~9,000 decode-arithmetic
# slots). Measured on ghostlite (test_ab_decode_band_planar_gather,
# wall times incl. ~0.1 ms dispatch): gate_up decode visit 0.265 ->
# 0.205 ms (-23%), w2 0.226 -> 0.165 (-27%), prefill T=512 fused
# 2.036 -> 1.576 (-23%), planar_gather == planar_mxu BITWISE on-chip.
# Remaining wall after the lookup, per the same LLO: the scale side's
# wide-plane cvt/shift/where feed (~29%), the f32->bf16 cast tail
# (~16%), and the MXU feed/accumulate floor (~18%).
DEQUANT_DEFAULT = DEQUANT_SPLIT_MXU


def _dequant_rhs(
    packed_chunk: jax.Array,
    scales_chunk: jax.Array,
    mode: str,
    gather_table: jax.Array | None = None,
) -> jax.Array:
  """Decodes one weight chunk with the requested variant.

  Under `split` the result carries a permuted k, so the lhs must go
  through `_prepare_lhs` with the same mode before contracting.
  `gather_table` is the precomputed `_e2m1_gather_table` for the
  `planar_gather` mode (built once per grid step, outside the rolled k
  walk); every other mode ignores it.
  """
  if mode == DEQUANT_SPLIT:
    return _dequantize_tile_split(packed_chunk, scales_chunk)
  if mode == DEQUANT_SPLIT_LUT:
    return _dequantize_tile_split_lut(packed_chunk, scales_chunk)
  if mode == DEQUANT_SPLIT_MXU:
    return _dequantize_tile_split_mxu(packed_chunk, scales_chunk)
  if mode == DEQUANT_PLANAR_LUT:
    return _dequantize_tile_planar_lut(packed_chunk, scales_chunk)
  if mode == DEQUANT_PLANAR_MXU:
    return _dequantize_tile_planar_mxu(packed_chunk, scales_chunk)
  if mode == DEQUANT_PLANAR_GATHER:
    return _dequantize_tile_planar_gather(
        packed_chunk, scales_chunk, table=gather_table
    )
  if mode == DEQUANT_PROBE_SCALAR:
    # Diagnostic: one scale per row, lane broadcast. WRONG NUMBERS.
    flat = lax.bitcast_convert_type(
        scales_chunk[:, :1].astype(jnp.uint32) << 23, jnp.float32
    )
    return jnp.concatenate(
        [
            _decode_e2m1(packed_chunk & 0x0F) * flat,
            _decode_e2m1(packed_chunk >> 4) * flat,
        ],
        axis=-1,
    ).astype(jnp.bfloat16)
  if mode == DEQUANT_PROBE_NOSCALE:
    # Diagnostic: split layout, scale never applied. WRONG NUMBERS.
    return jnp.concatenate(
        [_decode_e2m1(packed_chunk & 0x0F), _decode_e2m1(packed_chunk >> 4)],
        axis=-1,
    ).astype(jnp.bfloat16)
  if mode == DEQUANT_FAST:
    return _dequantize_tile_fast(packed_chunk, scales_chunk)
  if mode == DEQUANT_BASELINE:
    return _dequantize_tile(packed_chunk, scales_chunk)
  if mode in PLANAR_T_MODES:
    raise ValueError(
        f'{mode!r} decodes transposed tiles; the kernels dispatch it to'
        ' _dequantize_chunk_planar_t_gather, never through _dequant_rhs.'
    )
  raise ValueError(
      f'unknown dequant mode {mode!r}, want one of {DEQUANT_MODES}'
  )


def _prepare_lhs(lhs: jax.Array, mode: str, chunk: int) -> jax.Array:
  """Puts the whole activation into the k order `_dequant_rhs` expects.

  Applied before the `pallas_call`, not inside the kernel: see
  `split_k_order` for why the permutation cannot lower in Mosaic, and
  why doing it once per call is the cheaper placement regardless.

  The planar modes need NOTHING here: their weights were repacked
  offline so the decode already emits original contraction order. That
  no-op is the point of the layout: in the kiloblox while body this
  permutation ran per chunk per expert visit as an XLA gather plus two
  layout copies (the profiled gather_fusion ~16%-of-step bucket on the
  kb4 v2 trace, ~20% on gf), and in the ag_psum path once per full
  dispatch.
  """
  if mode in PLANAR_MODES or mode in PLANAR_T_MODES:
    return lhs
  split = mode in _SPLIT_MODES or mode in (
      DEQUANT_PROBE_NOSCALE,
      DEQUANT_PROBE_SCALAR,
  )
  return split_k_order(lhs, chunk) if split else lhs


def _e8m0_to_f32(scales: jax.Array) -> jax.Array:
  """2^(u8-127) by exponent-bit construction, exact for all 256 codes.

  Not a libm exp2: code 0 is the f32 SUBNORMAL 2^-127 (whose bit
  pattern is not e<<23) and 255 is inf, so building the exponent field
  directly matches np.exp2 bit for bit. (The dequant variants above
  carry the same two lines inline; this helper serves the fp8-native
  path below and the tests that pin the mapping.)
  """
  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  return jnp.where(scales == 0, jnp.float32(2.0**-127), exps)


def unpack_mxfp4_codes(packed: jax.Array) -> jax.Array:
  """The packed u8 plane reinterpreted as its e2m1 values, for tooling.

  `lax.bitcast_convert_type(u8, float4_e2m1fn)` splits every byte into
  two f4 sub-elements whose order MATCHES the checkpoint (element 2i is
  the LOW nibble), so the released bytes already ARE a valid f4 buffer
  and this "unpack" is a free relabelling, no arithmetic
  (moe_gmm_fp8_test pins both the order and the values against the
  LUT).

  TEST/TOOLING ONLY - never call it inside (or feed its result into)
  the kernels: Mosaic rejects bitwidth-changing bitcast_convert_type
  outright, and hoisted ABOVE a kernel it makes XLA emit a real HBM
  unpack pass over every weight byte (shift-right + and 0x0F + u8->u4
  convert + bitcast + a T(8,128)E(4) -> T(64,128)E(4)S(1) relayout
  copy; verified in the gf dump). The kernels relabel nibbles in
  registers instead (`_codes_to_f8`).
  """
  f4 = lax.bitcast_convert_type(packed, jnp.float4_e2m1fn)
  return f4.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _codes_to_f8(packed: jax.Array) -> jax.Array:
  """e2m1 nibble pairs -> float8_e4m3fn, a pure bit relabelling.

  Every e2m1 value is exactly representable in e4m3, so the map from
  nibble `S EE M` to byte `S EEEE MMM` is LOSSLESS (all 16 codes pinned
  bit exact in moe_gmm_fp8_test):

    * EE > 0 (normal `1.M * 2^(EE-1)`): EEEE = EE + 6 (bias 1 -> 7),
      MMM = M << 2 (the single mantissa bit keeps its 0.5 weight).
    * EE == 0, M == 1 (the e2m1 subnormal 0.5 = `1.0 * 2^-1`, a NORMAL
      in e4m3): EEEE = 6, MMM = 0.
    * EE == 0, M == 0: all magnitude bits clear (signed zero).
    * The sign moves bit 3 -> bit 7.

  Closed form below: EEEE = where(EE>0, EE+6, M*6); MMM =
  where(EE>0, M<<2, 0). The arithmetic runs in int16, NOT int8:
  Mosaic's vector integer canonicalization accepts only i16/i32
  (canonicalize_mosaic.cc canonicalize_arith_int_binary hard-errors
  "Only vector<i16> and vector<i32> are supported, but got 'i8'"; i16
  is legal on gen >= 6). Every intermediate is < 256 so i16 is exact,
  and the final u8 bitcast is width-preserving, which Mosaic allows.

  Args:
    packed: `[tn, half]` uint8, two e2m1 nibbles per byte, low nibble first
      (even contraction index).

  Returns:
    `[tn, 2 * half]` float8_e4m3fn holding exactly the e2m1 values.
  """
  tn, half = packed.shape
  byte = packed.astype(jnp.int16)
  nib = jnp.stack([byte & 0x0F, byte >> 4], axis=-1).reshape(tn, half * 2)
  exp2 = (nib >> 1) & 3
  mant = nib & 1
  normal = exp2 > 0
  eeee = jnp.where(normal, exp2 + 6, mant * 6)
  mmm = jnp.where(normal, mant << 2, jnp.int16(0))
  bits = ((nib & 8) << 4) | (eeee << 3) | mmm
  return lax.bitcast_convert_type(bits.astype(jnp.uint8), jnp.float8_e4m3fn)


def _codes_to_f8_folded(
    packed: jax.Array, scales: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Relabel a whole chunk with per-subchannel exponent deltas FOLDED in.

  The deep-dot enabler. `_subchannel_dot_f8` must run one MXU pass PER
  32-wide subchannel because each has its own e8m0 scale, and 32-deep
  passes are gain-latch bound: measured on gf hardware they cost ~1.5x
  the legacy dequant kernel per group visit. This variant instead
  normalizes every subchannel to the CHUNK's per-row max exponent by
  folding the delta `d = s_max - s` into the e4m3 codes themselves, so
  the whole chunk feeds ONE deep MXU pass and only the shared max
  exponent remains for the post-dot multiply.

  The fold `value * 2^-d` is EXACT while the result stays representable
  (the >= 8-wide window moe_gmm_fp8_test pins per magnitude: e2m1
  mantissas are 1 bit, e4m3 subnormals keep 3), and rounds half-up /
  flushes below it - i.e. a subchannel whose scale sits > ~10 octaves
  under its chunk-mates loses precision it could only have contributed
  below 2^-9 of the chunk's largest terms. Realistic checkpoints
  (absmax-derived e8m0 over gaussian-class rows) keep within-chunk
  spreads well inside the window, where this path is bit-identical to
  the per-subchannel fold; the adversarial-spread bound is pinned in
  moe_gmm_fp8_test.

  Args:
    packed: `[tn, half]` uint8 nibble pairs (a whole k chunk).
    scales: `[tn, 2 * half / SCALE_GROUP]` uint8 e8m0 exponents.

  Returns:
    `([tn, 2 * half]` f8e4m3fn folded codes, `[tn, 1]` i16 max codes).
  """
  tn, half = packed.shape
  k = half * 2
  groups = scales.shape[-1]
  per = k // groups

  # The max reduces in i32 (Mosaic's vector reductions do not cover
  # i16); the delta then drops back to the i16 the fold chain runs in.
  s32 = scales.astype(jnp.int32)
  s_max32 = jnp.max(s32, axis=-1, keepdims=True)  # [tn, 1]
  s_max = s_max32.astype(jnp.int16)
  d = (s_max32 - s32).astype(jnp.int16)  # [tn, groups] >= 0

  def relabel_block(byte, d_block, block_k):
    """The fold chain over one <= 256-column block, all 2-D planes.

    Blocked ON PURPOSE, twice over: (a) a whole-chunk chain at
    (tn, chunk) = (512, 512) holds enough live i16 planes to blow the
    32 MiB scoped-VMEM stack (measured CompileTimeScopedVmemOom on
    gf; per-block temporaries die at each block boundary instead),
    and (b) the d expansion's one 3-D hop (minor dim `per` = 32
    lanes, quarter vreg utilization) stays block-sized.
    """
    nib = jnp.stack([byte & 0x0F, byte >> 4], axis=-1).reshape(tn, block_k)
    exp2 = (nib >> 1) & 3
    mant = nib & 1
    normal = exp2 > 0
    eeee = jnp.where(normal, exp2 + 6, mant * 6)
    mmm = jnp.where(normal, mant << 2, jnp.int16(0))
    d_full = jnp.broadcast_to(
        d_block[:, :, None], (tn, block_k // per, per)
    ).reshape(tn, block_k)
    zero = eeee == 0  # eeee == 0 only for true (signed) zeros.
    ep = eeee - d_full  # Folded exponent field.
    # Subnormal landing (ep <= 0): value = 1.MMM * 2^(ep - 7)
    # re-encoded as MMM' * 2^-9, MMM' = round_half_up(1MMM >> (1-ep));
    # zero past shift 4 ((12 + 16) >> 5 == 0). Select chain because
    # vector-varying shift amounts are not a Mosaic primitive.
    mant4 = (mmm >> 2) | 8  # 1MMM as an integer: 8 | M.
    shift = 1 - ep
    sub = jnp.where(
        shift == 1,
        (mant4 + 1) >> 1,
        jnp.where(
            shift == 2,
            (mant4 + 2) >> 2,
            jnp.where(
                shift == 3,
                (mant4 + 4) >> 3,
                jnp.where(shift == 4, (mant4 + 8) >> 4, jnp.int16(0)),
            ),
        ),
    )
    bits = jnp.where(
        zero, jnp.int16(0), jnp.where(ep >= 1, (ep << 3) | mmm, sub)
    )
    bits = ((nib & 8) << 4) | bits
    return lax.bitcast_convert_type(bits.astype(jnp.uint8), jnp.float8_e4m3fn)

  block = min(k, 256)
  groups_per_block = block // per
  pieces = [
      relabel_block(
          packed[:, c // 2 : (c + block) // 2].astype(jnp.int16),
          d[:, c // per : c // per + groups_per_block],
          block,
      )
      for c in range(0, k, block)
  ]
  codes = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=-1)
  return codes, s_max


def _subchannel_dot_f8_folded(
    lhs_chunk: jax.Array,
    packed_chunk: jax.Array,
    scales_chunk: jax.Array,
) -> jax.Array:
  """One DEEP MXU pass per chunk: folded codes + one post-dot scale.

  The `fold_scales` arm of the native path (see `_codes_to_f8_folded`):
  y = 2^(s_max - 127) * (lhs @ folded_codes.T), per row. One matmul per
  (chunk, n-tile) like the dequant kernel - the structure that makes
  the fp8 push rate reachable - against `_subchannel_dot_f8`'s
  groups-per-chunk latch-bound passes.
  """
  codes, s_max = _codes_to_f8_folded(packed_chunk, scales_chunk)
  part = lax.dot_general(
      lhs_chunk,
      codes,
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  scale = _e8m0_to_f32(s_max.astype(jnp.uint8))
  return part * scale[:, 0][None, :]


def _subchannel_dot_f8(
    lhs_chunk: jax.Array,
    packed_chunk: jax.Array,
    scales_chunk: jax.Array,
    *,
    lhs_scale: jax.Array | None = None,
    lhs_group_scales: jax.Array | None = None,
) -> jax.Array:
  """One chunk's contraction with the k32 scales applied POST-dot.

  The mxfp4-NATIVE alternative to the dequant variants: the u8 nibbles
  relabel to f8e4m3fn (`_codes_to_f8`, lossless) and feed the MXU
  directly. An f8e4m3fn lhs runs the fp8 x fp8 mode at 2x push rate.
  THAT RATE DID NOT MAKE THE ARM FASTER, and the sentence above is a
  mechanism rather than a recommendation: native f8xf8 measured DEAD
  on GF, within 1% of native bf16 at every shape and both phases, so
  a doubled push rate does not survive contact with the rest of the
  chain (sponge2/1a084ed2). A
  bf16 lhs pairs a WIDE lhs with an 8-bit weight, and HOW THOSE TWO ARE
  RECONCILED IS UNSETTLED AND UNDER ACTIVE MEASUREMENT. Do not write a
  mechanism here until it settles. Three sourced facts contradict one
  another right now:
    SOURCE   GF supports the e4m3fn data format, so the f8e4m3fn arm of
             `GetGainLatchModeAndScalingFactor` takes the PACKED latch
             `kNoXposePackedE4M3Fn` (llo.td 50, prints `f8e4m3b`), NOT
             the conversion latch `kNoXposeF8E4M3B11ToBf16` (llo.td 16,
             `f8e4m3b_to_bf16`) that GF has no data format for. On that
             reading nothing widens the weight at all.
    ISA      the GMR mode bit width must match the lefthand side, so
             bf16 against e4m3 should be refused outright.
    SILICON  it is not refused, and it agrees with the dequant
             reference at 2e-2 on real GF across decode m=8/32/64 and
             prefill m=512/2048, both kernel orientations,
             sponge2/1a084ed2.
  One candidate under measurement is that the LHS is NARROWED rather
  than the weight widened, which a 2e-2 bar would hide. native bf16 and
  native f8xf8 agreeing to within 1% is consistent with that and is not
  decisive. Whatever the answer turns out to be, it is NOT
  canonicalize_mosaic.cc:857, which fires only on an f8 lhs against a
  wide rhs and rewrites operand 0.
  TOKEN TRAP behind much of this confusion: `e4m3b` (MatmulDataFormat
  10) is a SUBSTRING of `e4m3b_to_bf16` (4), of `f8e4m3b` (GainLatchMode
  50) and of `f8e4m3b_to_bf16` (16). Four tokens, two enums. Match them
  exactly, never by substring.
  SCOPE: Mosaic gmm path. The XLA attention path on the same chip
  behaves differently; do not transfer across kernel paths.
  Each 32-wide subchannel's e8m0
  scale multiplies the f32 PARTIAL SUM instead of every weight
  element:

    y[m, n] = sum_b 2^(e[n, b] - 127) * (sum_{k in b} lhs[m, k] f4[n, k])

  the same post-dot fold prod's quantized gmm uses (gemax megablox
  quantization/subchannel.py subchannel_dot_impl). The per-weight VPU
  dequant chain disappears, and scale multiplies drop from K*N
  (pre-dot) to M*N*K/32 (post-dot).

  With MXFP8 activations (`lhs_group_scales`) the SAME identity folds
  the activation side: both scale sets are e8m0 over the SAME 32-wide
  blocks (the MX spec block the checkpoint was written with), so

    y[m, n] = sum_b 2^(ea[m, b] - 127) 2^(ew[n, b] - 127)
              * (sum_{k in b} f8a[m, k] f4[n, k])

  and the per-subchannel structure the weight fold already pays gains
  ONE extra vector multiply per (row tile, block), nothing else. Both
  post-dot factors are exact powers of two, so their f32 product and
  fold introduce no rounding beyond the accumulation itself.

  Converts PER SUBCHANNEL, not per chunk: converting the whole chunk
  up front puts a `[tn, chunk]` f8 tile plus its i16 temporaries in
  flight and blows Mosaic's 32 MiB scoped-VMEM stack at serving tn
  (CompileTimeScopedVmemOom; verified both ways).

  Args:
    lhs_chunk: `[tm, chunk]` activations (bf16, or f8e4m3fn opt-in).
    packed_chunk: `[tn, chunk/2]` uint8 nibble pairs.
    scales_chunk: `[tn, chunk/SCALE_GROUP]` uint8 e8m0 exponents.
    lhs_scale: optional `[tm, 1]` f32 activation scales folded once after the
      accumulation (the hook for per-token-scaled fp8 activations; unused by the
      plain-cast arm).
    lhs_group_scales: optional `[tm, chunk/SCALE_GROUP]` uint8 e8m0 activation
      exponents (the MXFP8 arm, `quantize_mxfp8`); folded per subchannel like
      the weight scales.

  Returns:
    `[tm, tn]` f32 partial products.
  """
  groups = scales_chunk.shape[-1]
  scale_f32 = _e8m0_to_f32(scales_chunk)
  lhs_scale_f32 = (
      _e8m0_to_f32(lhs_group_scales) if lhs_group_scales is not None else None
  )
  half = SCALE_GROUP // 2  # 16 packed bytes per 32-wide subchannel.
  acc = None
  for b in range(groups):
    lo = b * SCALE_GROUP
    codes = _codes_to_f8(packed_chunk[:, b * half : (b + 1) * half])
    part = lax.dot_general(
        lhs_chunk[:, lo : lo + SCALE_GROUP],
        codes,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    part = part * scale_f32[:, b][None, :]
    if lhs_scale_f32 is not None:
      part = part * lhs_scale_f32[:, b][:, None]
    acc = part if acc is None else acc + part
  assert acc is not None  # groups >= 1: a chunk holds >= 1 subchannel.
  if lhs_scale is not None:
    acc = acc * lhs_scale
  return acc


def _use_k_loop(k: int, chunk: int) -> bool:
  """Whether the k walk runs as a real (rolled) loop for this tiling.

  The rolled walk needs every per-iteration ref slice lane-aligned: the
  packed nibble plane advances `chunk/2` u8 lanes per step, so `chunk`
  must be a multiple of 256 (the scales side is handled separately, see
  `_stage_scale_codes`). Below that (toy shapes, tiny contractions) the
  walk stays a static python unroll, whose text is small at those sizes
  anyway. Rolling the real serving shapes (chunk 256/512) is what keeps
  one kernel's ISA under the single-overlay HLO-function ceiling, so
  the 92 identical per-layer calls of the unrolled serving program
  share ONE copy of program text instead of duplicating ~28 MiB per
  layer (and it cuts Mosaic compile time with it).
  """
  return chunk % 256 == 0 and k > chunk


def _stage_scale_codes(scales_ref, codes_ref, chunk: int) -> None:
  """Restages `[tn, K/32]` e8m0 codes as `[K/chunk, tn, chunk/32]`.

  The rolled k walk cannot dynamically slice the raw scale plane: a
  chunk's codes are `chunk/32` lanes wide (8 or 16), far below the
  128-lane alignment Mosaic needs for a loop-variant minor-dim slice.
  Restaged with the chunk index MAJOR, the loop body indexes the
  leading dim (no alignment constraint) and hands `_dequantize_tile`
  exactly the same `[tn, chunk/32]` codes the unrolled walk sliced
  statically. The restage itself is `K/chunk` static sub-lane slices,
  done once per grid step outside the loop.
  """
  tn, groups = scales_ref.shape
  per_chunk = chunk // SCALE_GROUP
  for j in range(groups // per_chunk):
    codes_ref[j] = scales_ref[:, j * per_chunk : (j + 1) * per_chunk]


def _stage_plane_scales(scales_ref, exps_ref) -> None:
  """Widens a `[8 steps, tn]` u8 window maxima block to f32 rows, once.

  The plane kernels' k walk needs one `[1, tn]` window scale per step.
  One row of a packed u8 block cannot be loaded from an arbitrary
  sublane (Mosaic proves 8-bit loads tile aligned); one 32-bit row can
  be loaded from any sublane, static or dynamic. So the block is read
  whole in its at rest tiling (`PLANE_SCALE_ROWS_PER_WINDOW` rows per
  window), `_e8m0_to_f32` runs once per tile, and the walk reads
  `exps_ref[pl.ds(8 * i, 1), :]`. Same exponents, same post-dot
  multiply as the row-at-a-time form: bitwise.
  """
  exps_ref[...] = _e8m0_to_f32(scales_ref[...])


def _situ_f32(
    gate: jax.Array,
    up: jax.Array,
    *,
    beta: float,
    linear_beta: float | None,
) -> jax.Array:
  """SiTU-GLU in f32, the op sequence of `ffw.situ` minus its final cast.

  `[beta * tanh(gate / beta) * sigmoid(gate)] * [linear_beta * tanh(up /
  linear_beta)]` (up stays linear when `linear_beta` is None). Duplicated
  here rather than imported because ffw.py imports this module (the
  same one-way-dependency reason the e2m1 LUT is duplicated);
  `moe_gmm_test` pins the fused reference bitwise equal to composing
  `ffw.situ` over the unfused references, so the two copies cannot
  drift silently. Shared by the fused kernel body and the fused
  reference so on-TPU and off-TPU paths run the same arithmetic.
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
    transposed: bool = False,
    f8_planes: bool = False,
) -> None:
  """Rejects shapes the kernel cannot serve (see module docstring).

  `transposed` declares the planar_t weight orientation
  (`[E, K/2, N]` packed, `[E, K/32, N]` scales); False is the
  checkpoint/planar orientation (`[E, N, K/2]`, `[E, N, K/32]`).
  `f8_planes` declares the LOAD-TIME f8 spelling of planar_t
  (`[E, K, N]` bytes, `[E, 8 * K/PLANAR_WINDOW, N]` window scale rows).
  """
  if lhs.ndim != 2:
    raise ValueError(f'lhs must be 2-D, got {lhs.ndim}-D.')
  if packed.ndim != 3 or scales.ndim != 3:
    if f8_planes:
      want = (
          f'[E, K, N] and [E, {PLANE_SCALE_ROWS_PER_WINDOW} *'
          f' K/{PLANAR_WINDOW}, N]'
      )
    elif transposed:
      want = '[E, K/2, N] and [E, K/32, N]'
    else:
      want = '[E, N, K/2] and [E, N, K/32]'
    raise ValueError(
        f'packed and scales must be 3-D {want}, got'
        f' {packed.shape} and {scales.shape}.'
    )
  if packed.dtype != jnp.uint8 or scales.dtype != jnp.uint8:
    raise ValueError(
        f'packed/scales must be uint8, got {packed.dtype}/{scales.dtype}.'
    )
  if group_sizes.dtype != jnp.int32:
    raise ValueError(f'group_sizes must be int32, got {group_sizes.dtype}.')
  k = lhs.shape[1]
  k_axis, n_axis = (1, 2) if transposed else (2, 1)
  packed_per_row, scale_span = (
      (1, PLANAR_WINDOW // PLANE_SCALE_ROWS_PER_WINDOW)
      if f8_planes
      else (2, SCALE_GROUP)
  )
  if packed.shape[k_axis] * packed_per_row != k:
    raise ValueError(
        f'packed contraction {packed.shape[k_axis] * packed_per_row} !='
        f' lhs contraction {k}.'
    )
  if scales.shape[k_axis] * scale_span != k:
    raise ValueError(
        'scales cover'
        f' {scales.shape[k_axis] * scale_span} elements, expected {k}.'
    )
  if (
      packed.shape[0] != scales.shape[0]
      or packed.shape[n_axis] != scales.shape[n_axis]
  ):
    raise ValueError(
        f'packed {packed.shape} and scales {scales.shape} disagree on'
        ' (experts, out).'
    )
  tm, chunk, tn = tiling
  # A scale group must never straddle a dequant chunk, or the chunk
  # would need a neighbouring chunk's exponent.
  if chunk % SCALE_GROUP:
    raise ValueError(f'k chunk {chunk} must be a multiple of {SCALE_GROUP}.')
  if k % chunk:
    raise ValueError(f'contraction {k} must be a multiple of k chunk {chunk}.')
  if lhs.shape[0] % tm:
    raise ValueError(f'rows {lhs.shape[0]} must be a multiple of m tile {tm}.')
  # Pallas TPU wants a block's second-minor dim 8-divisible and its
  # minor dim 128-divisible, unless the block covers the whole array
  # dim. Every block here spans the whole contraction, so only the row
  # and column tiles need checking. Interpret mode does NOT enforce
  # this, so it is checked here or it only shows up at TPU lowering.
  n = packed.shape[n_axis]
  if tm % 8 and tm != lhs.shape[0]:
    raise ValueError(
        f'm tile {tm} must be a multiple of 8 or all {lhs.shape[0]} rows.'
    )
  if tn % _LANES and tn != n:
    raise ValueError(
        f'n tile {tn} must be a multiple of {_LANES} or all {n} columns.'
    )


def _deep_eligible(
    m: int, k: int, n: int, tm: int, chunk: int, tn: int
) -> bool:
  """Whether the deep-stream kernel serves these static shapes.

  Divisibility plus the accumulator bound. There is deliberately NO
  bound on `m`: the composed kiloblox call hands BOTH bands a padded
  assignment tray (the served kb15 generate program passes m=65,536
  with tm=512, the raise-with-values control receipt), so a row bound
  cannot separate decode from prefill and an early revision keyed on
  one silently served the rolled fallback while every kernel-level
  gate stayed green. Band scoping is the CALLER's job (the width-keyed
  `gmm_dequant_impl_decode` knob); this predicate only refuses shapes
  the kernel structurally cannot take: non-dividing tiles, and a
  per-tile f32 accumulator past DEEP_MAX_ACC_BYTES per stack (the
  served tiling's 512 x 256 x 4 = 512 KiB sits at a quarter of it).
  Ineligible callers fall back to the bitwise-identical rolled
  `planar_t_b11`; only the schedule differs.
  """
  return (
      m % tm == 0
      and k % chunk == 0
      and n % tn == 0
      and tm * tn * 4 <= DEEP_MAX_ACC_BYTES
  )


def _deep_fit_tn(
    *,
    m: int,
    k: int,
    n: int,
    tm: int,
    chunk: int,
    tn: int,
    num_stacks: int,
    budget: int,
    lhs_bytes: int,
    out_bytes: int,
    stream_rows: bool = False,
    f8_planes: bool = False,
) -> int | None:
  """Largest column tile whose DEEP-kernel VMEM footprint fits `budget`.

  The deep kernel's VMEM working set is NOT the rolled kernel's: each
  stack streams DEEP_STREAM_DEPTH outstanding [k, tn'] slab copies of
  nibbles + scale codes, and the operands ride in one of two forms
  (`_deep_planar_t_b11_call(stream_rows=)`):

    * resident (`stream_rows=False`): the whole lhs and out live as full
      VMEM blocks (the visit loop addresses arbitrary row tiles, so no
      BlockSpec pipeline can slice them). At the gf fslp merged tray
      (m=1792 rows per TensorCore) that is 12.25 + 5.25 MiB before a
      single slab, which no gf generate ceiling admits;
    * streamed (`stream_rows=True`): DEEP_ROW_TILE_SLOTS `[tm, k]` lhs
      tiles and one `[tm, n]` out tile, whatever the tray size.

  `default_tiling`'s search models the ROLLED kernel, so a tiling it
  admits can OOM the deep kernel's scoped stack: measured at the gf
  fslp decode tray (m=896 merged rows, 3584->3072 two-stack fused,
  tiling 32/512/512) the resident form bills 18.81 MiB against the prod
  parity bundle's 16 MiB generate scoped floor. Shrinking ONLY the deep
  kernel's column tile is value-exact: tn' regroups the output columns
  and the DMA slabs, and every element's chunk-ordered accumulation
  (and the fold modes' per-column window scale) is untouched, so the
  bitwise pins hold at any tn'. Returns the largest lane-divisible
  tn' <= tn that divides n and fits, or None (caller falls back to the
  rolled kernel, bitwise).

  `f8_planes` (the `F8_PLANE_MODES` bodies) sizes the slab as one byte
  per weight plus one u8 row tile of window maxima per slot (and the
  f32 rows `_stage_plane_scales` widens them to, per stack) instead of
  the nibble + e8m0 pair: 2x the stream per tile, the reason the plane
  arm's fused call needs ~11 MiB at tn 512 where the nibble form
  needs ~6.
  """
  steps = max(k // chunk, 1)
  if f8_planes:
    rows = steps * PLANE_SCALE_ROWS_PER_WINDOW
    tile = _PLANE_SCALE_TILE_ROWS
    slab_bytes = steps * chunk + -(-rows // tile) * tile
    exps_rows = -(-rows // 8) * 8
  else:
    slab_bytes = steps * (chunk // 2 + chunk // SCALE_GROUP)
    exps_rows = 0
  if stream_rows:
    operands = DEEP_ROW_TILE_SLOTS * tm * k * lhs_bytes + tm * n * out_bytes
  else:
    operands = m * k * lhs_bytes + m * n * out_bytes
  cand = tn
  while cand >= _LANES:
    if n % cand == 0:
      scratch = num_stacks * DEEP_STREAM_DEPTH * slab_bytes * cand
      exps = num_stacks * exps_rows * cand * 4
      accs = num_stacks * tm * cand * 4
      total = operands + scratch + exps + accs
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
    out_hbm: bool,
    body: str = DEQUANT_PLANAR_T_B11,
    continuous: bool = False,
    stream_rows: bool = False,
) -> jax.Array:
  """The deep-stream kernel shared by the plain and fused entry points.

  `body` selects the CHUNK DECODE only; the DMA schedule around it is
  identical either way, which is the point. Both bodies consume the
  same chunk-major planar_t staging (`[chunk/2, tn]` nibbles plus
  `[chunk/32, tn]` e8m0), so the depth-`DEEP_STREAM_DEPTH` pipeline is
  orthogonal to the decode:

    `planar_t_b11`     folds the window to f8 codes plus ONE post-dot
                       column scale. Needs the b11 widen, which
                       Ghostfish does not have.
    `planar_t_gather`  folds the scale INTO the e2m1 value table and
                       returns bf16 ready to dot. Runs anywhere the
                       served kernel runs, Ghostfish included.
    `planar_t_f8fn`    the b11 fold spelled float8_e4m3fn, the one
                       narrow matmul format Ghostfish has. Bitwise
                       `planar_t_gather` inside `F8FN_FOLD_WINDOW`.
    `planar_t_f8fn_plane`  the f8fn fold done ONCE at checkpoint load
                       (`repack_planar_t_f8_stack`, e4m3fn): the slab is
                       `[chunk, tn]` bytes per step plus the tile's
                       `[8 steps, tn]` window maxima block (the rank 3
                       leaf in its at rest tiling, widened to f32 rows
                       once per tile by `_stage_plane_scales`), the
                       chunk step a bitcast, the dot and the window
                       scale row. Bitwise `planar_t_f8fn`; 2x the stream.

  The second exists because the pipeline half of the deep win is a DMA
  DEPTH effect, not a format effect: it is the outstanding weight-tile
  copies that pay, so gating it behind a format the chip cannot decode
  left the whole rung unreachable on GF for no structural reason.

  `stacks` holds the chunk-major restaged planar_t arrays
  (`[E, steps, chunk/2, n]` nibbles, `[E, steps, chunk/32, n]` e8m0
  codes): one pair for `mxfp4_gmm`, the (gate, up) pairs for
  `mxfp4_fused_gate_up_gmm` (then `situ` carries the SiTU clamps and
  the epilogue is `_situ_f32` on the two accumulators, the fused
  kernel's exact op order). Per active tile the expert's slab tiles
  stream through DEEP_STREAM_DEPTH outstanding copies per array while
  the unrolled k walk runs the body's chunk decode + dot (+ window
  scale for the fold bodies) on the previously landed tile; the group-interval
  row
  mask and the read-modify-write epilogue mirror the rolled kernel
  exactly, so a tile straddling groups composes identically.

  `continuous` (`K3MoeConfig.gmm_deep_stream_continuous`) changes WHEN
  the slab copies are issued and nothing else. Default False primes
  `depth` copies at the top of every visit and drains them at the
  bottom, so each visit pays one pipeline fill plus the one-shot DMA
  rate (measured on glc: 1.45 us per stack visit and 964 vs 1390 GB/s
  continuous; perf/moe_gmm_bench_forge_test DMA geometry probe). True
  flattens the (visit, tile) sequence into one stream: the copy for
  global tile q + depth is issued after tile q computes, so the last
  `depth` tiles of visit j pull visit j + 1's first tiles
  (`ids_ref[j + 1]`, guarded by the active count) and the pipeline is
  primed once per kernel call. Tile indices stay static; only the
  expert id and the buffer slot ((j * tiles_n + t) mod depth) are
  dynamic. Accumulation order, window scale, epilogue and scratch
  shapes are identical, so the two schedules are bitwise the same
  (moe_gmm_test.DeepStreamParityTest pins knob on == knob off ==
  rolled).

  `stream_rows` (`K3MoeConfig.gmm_deep_stream_rows`) changes WHERE the
  lhs and out live and nothing else. Default False is the resident
  form: the whole `[m, k]` lhs and `[m, n]` out are VMEM blocks of the
  call (the GLP served program, byte for byte). True streams only the
  visited rows: the visit's `[tm, k]` lhs tile is DMA'd from wherever
  XLA left the tray (`pl.ANY`) into one of DEEP_ROW_TILE_SLOTS VMEM
  slots (visit j + 1's tile lands while visit j computes), the visit's
  `[tm, n]` out tile is assembled in one VMEM buffer across its column
  tiles and DMA'd out when the next visit moves to another row tile (or
  at the last visit), so a tile straddling two groups is merged in VMEM
  exactly as the resident form's read-modify-write merges it. Rows the
  kernel never visits, and rows of a visited tile outside every group,
  are unspecified in both forms (the callers' contract). The k walk
  reads the same `[tm, chunk]` lhs values, the dots, the accumulation
  order, the window scale and the epilogue are the same code, so the
  two forms are bitwise the same on every routed row (pinned by
  moe_gmm_test.DeepStreamParityTest and on chip by
  moe_deep_stream_model_pin_forge_test). Why it exists: at the gf
  merged tray (m=1792 rows per TensorCore) the resident form needs
  12.25 + 5.25 MiB of VMEM before its first slab and no gf generate
  scoped ceiling admits it; the streamed form needs ~0.6 MiB for the
  same operands.
  """
  m, k = lhs.shape
  num_stacks = len(stacks)
  if stacks[0][0].ndim == 5:
    num_groups, _, steps, plane_rows, n = stacks[0][0].shape
    scale_rows = stacks[0][1].shape[3]
    plane_dtype = None
  else:
    num_groups, steps, plane_rows, n = stacks[0][0].shape
    plane_dtype = F8_PLANE_DTYPES.get(body)
    if plane_dtype is not None:
      # Plane bodies stream the rank 3 window maxima leaf `[E, 8 steps, N]`
      # as one `[8 steps, tn]` block per tile (see `_stage_plane_scales`).
      scale_rows = steps * PLANE_SCALE_ROWS_PER_WINDOW
      for _, s in stacks:
        if s.shape != (num_groups, scale_rows, n):
          raise ValueError(
              f'plane window scales must be {(num_groups, scale_rows, n)}, got'
              f' {s.shape}.'
          )
    else:
      scale_rows = stacks[0][1].shape[2]
  tiles_n = n // tn
  depth = min(DEEP_STREAM_DEPTH, tiles_n)
  group_offsets, group_ids, m_tile_ids = group_metadata
  grid_len = m_tile_ids.shape[0]

  def kernel(off_ref, ids_ref, mtid_ref, nat_ref, lhs_ref, *rest):
    stack_refs = rest[: 2 * num_stacks]
    out_ref = rest[2 * num_stacks]
    scratch = rest[2 * num_stacks + 1 :]
    bufs = scratch[: 2 * num_stacks]
    acc_refs = scratch[2 * num_stacks : 3 * num_stacks]
    sems = scratch[3 * num_stacks : 5 * num_stacks]
    scratch = scratch[5 * num_stacks :]
    # Plane bodies: per stack f32 window scale rows of the current tile.
    exps_refs = ()
    if plane_dtype is not None:
      exps_refs, scratch = scratch[:num_stacks], scratch[num_stacks:]
    # Streamed row form only: (lhs slots, out tile, lhs sems, out sem).
    row_scratch = scratch

    def lhs_copy(row0, slot):
      return pltpu.make_async_copy(
          lhs_ref.at[pl.ds(row0, tm), :],
          row_scratch[0].at[slot],
          row_scratch[2].at[slot],
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
          scales_src = (
              scales_ref.at[g, :, pl.ds(t * tn, tn)]
              if plane_dtype is not None
              else scales_ref.at[g, :, :, pl.ds(t * tn, tn)]
          )
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

    # Built ONCE per kernel invocation, not per chunk: the rolled
    # gather kernel hoists it the same way, so the two differ only in
    # the DMA schedule and a benched delta is the pipeline alone.
    values = (
        _e2m1_gather_table(tn).astype(jnp.float32)
        if body == DEQUANT_PLANAR_T_GATHER
        else None
    )
    # Fold bodies (b11, f8fn) share one chunk-step shape; resolved
    # statically here so the k walk below carries no mode branch. Plane
    # bodies carry only the dtype their bytes spell.
    fold_chunk = (
        _planar_t_fold_chunk(body) if body in PLANAR_T_FOLD_MODES else None
    )
    if (
        body != DEQUANT_PLANAR_T_GATHER
        and fold_chunk is None
        and (plane_dtype is None)
    ):
      raise ValueError(f'{body!r} is not a deep chunk-decode body.')

    def visit(j, carry):
      g = ids_ref[j]
      mt = mtid_ref[j]
      row0 = mt * tm
      start = off_ref[g]
      end = off_ref[g + 1]
      if stream_rows:
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
        # nondecreasing in both metadata builds), so "last" is "the
        # next visit is on another tile, or there is none". The clamp
        # keeps the lookahead read in bounds at the padded tail.
        next_mt = mtid_ref[jnp.minimum(j + 1, grid_len - 1)]
        last_of_tile = jnp.logical_or(j + 1 >= nat_ref[0], next_mt != mt)
      else:
        lslot = jnp.int32(0)
        last_of_tile = jnp.bool_(False)
      if continuous:
        # Slot of tile t in the flattened (visit, tile) stream; the
        # copies for this visit's first `depth` tiles were issued by
        # the previous visit (or the prime below for visit 0).
        base = lax.rem(j * tiles_n, depth)
      else:
        base = 0
        for s in range(depth):
          for c in copies(g, s, s):
            c.start()
      for t in range(tiles_n):
        slot = lax.rem(base + t, depth) if continuous else t % depth
        for c in copies(g, t, slot):
          c.wait()
        # Accumulate through a scratch ref, not a carried value: each
        # `+=` then materializes exactly like the rolled kernel's, so
        # no backend can fuse the window-scale multiply into the add
        # (an FMA there is a 1-ulp reassociation and breaks the
        # bitwise contract; the interpret-mode pin caught it).
        accs = []
        for s in range(num_stacks):
          pbuf, sbuf = bufs[2 * s], bufs[2 * s + 1]
          acc_ref = acc_refs[s]
          acc_ref[...] = jnp.zeros_like(acc_ref)
          if plane_dtype is not None:
            _stage_plane_scales(sbuf.at[slot], exps_refs[s])
          for cs in range(steps):
            if stream_rows:
              lhs_chunk = row_scratch[0][
                  lslot, :, cs * chunk : (cs + 1) * chunk
              ]
            else:
              lhs_chunk = lhs_ref[
                  pl.ds(row0, tm), cs * chunk : (cs + 1) * chunk
              ]
            dims = (((1,), (0,)), ((), ()))
            if plane_dtype is not None:
              # LOAD-TIME planes: the operand IS the slab (same-width
              # bitcast), the window maximum the post-dot scale; no
              # table, no gather, no fold in the visit.
              acc_ref[...] += (
                  lax.dot_general(
                      lhs_chunk,
                      lax.bitcast_convert_type(pbuf[slot, cs], plane_dtype),
                      dimension_numbers=dims,
                      preferred_element_type=jnp.float32,
                  )
                  * exps_refs[s][pl.ds(cs * PLANE_SCALE_ROWS_PER_WINDOW, 1), :]
              )
            elif fold_chunk is None:
              # The gather body: scale is already inside the table, so
              # the dot lands straight on the accumulator - no post-dot
              # multiply, and `values` is hoisted out of the k walk
              # exactly as the rolled gather kernel hoists it.
              acc_ref[...] += lax.dot_general(
                  lhs_chunk,
                  _dequantize_chunk_planar_t_gather(
                      pbuf[slot, cs], sbuf[slot, cs], values
                  ),
                  dimension_numbers=dims,
                  preferred_element_type=jnp.float32,
              )
            else:
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
        if stream_rows:
          out_buf = row_scratch[1]
          old = jnp.where(
              first_of_tile, 0.0, out_buf[:, pl.ds(t * tn, tn)].astype(jnp.float32)
          )
          out_buf[:, pl.ds(t * tn, tn)] = jnp.where(
              mask, value, old
          ).astype(preferred_element_type)
        else:
          old = jnp.where(
              first_of_tile,
              0.0,
              out_ref[pl.ds(row0, tm), pl.ds(t * tn, tn)].astype(jnp.float32),
          )
          out_ref[pl.ds(row0, tm), pl.ds(t * tn, tn)] = jnp.where(
              mask, value, old
          ).astype(preferred_element_type)
        nxt = t + depth
        if nxt < tiles_n:
          for c in copies(g, nxt, slot):
            c.start()
        elif continuous:
          # This slot is free and, in the flattened stream, next holds
          # tile nxt - tiles_n of the NEXT visit: pull it now so the
          # visit boundary never drains the pipeline. Every copy issued
          # here is waited by visit j + 1, which exists by the guard.
          @pl.when(j + 1 < nat_ref[0])
          def _():
            g_next = ids_ref[j + 1]
            for c in copies(g_next, nxt - tiles_n, slot):
              c.start()

      if stream_rows:

        @pl.when(last_of_tile)
        def _():
          flush = pltpu.make_async_copy(
              row_scratch[1], out_ref.at[pl.ds(row0, tm), :], row_scratch[3]
          )
          flush.start()
          flush.wait()

      return carry

    if stream_rows:

      @pl.when(nat_ref[0] > 0)
      def _():
        lhs_copy(mtid_ref[0] * tm, 0).start()

    if continuous:

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
    if plane_dtype is not None:
      scratch_shapes.append(pltpu.VMEM((depth, scale_rows, tn), jnp.uint8))
    else:
      scratch_shapes.append(
          pltpu.VMEM((depth, steps, scale_rows, tn), jnp.uint8)
      )
  for _ in range(num_stacks):
    scratch_shapes.append(pltpu.VMEM((tm, tn), jnp.float32))
  for _ in range(num_stacks):
    scratch_shapes.append(pltpu.SemaphoreType.DMA((depth,)))
    scratch_shapes.append(pltpu.SemaphoreType.DMA((depth,)))
  if plane_dtype is not None:
    for _ in range(num_stacks):
      scratch_shapes.append(pltpu.VMEM((scale_rows, tn), jnp.float32))
  if stream_rows:
    scratch_shapes.append(pltpu.VMEM((DEEP_ROW_TILE_SLOTS, tm, k), lhs.dtype))
    scratch_shapes.append(pltpu.VMEM((tm, n), preferred_element_type))
    scratch_shapes.append(pltpu.SemaphoreType.DMA((DEEP_ROW_TILE_SLOTS,)))
    scratch_shapes.append(pltpu.SemaphoreType.DMA(()))
  # Operand placement: the streamed form leaves the tray and the output
  # wherever XLA puts them and moves row tiles itself; the resident form
  # asks for whole VMEM blocks. No `vmem_limit_bytes` in either form
  # (see DEEP_ROW_TILE_SLOTS): the kernel fits the program ceiling.
  operand_space = pl.ANY if stream_rows else pltpu.VMEM
  call = pl.pallas_call(
      kernel,
      grid=(),
      in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 4
      + [pl.BlockSpec(memory_space=operand_space)]
      + [pl.BlockSpec(memory_space=pl.ANY)] * (2 * num_stacks),
      out_specs=pl.BlockSpec(memory_space=operand_space),
      out_shape=pltpu.HBM((m, n), preferred_element_type)
      if out_hbm and not interpret
      else jax.ShapeDtypeStruct((m, n), preferred_element_type),
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


def _paired_planar_t_gather_call(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_metadata: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    num_paired_tiles: jax.Array,
    m: int,
    n: int,
    k: int,
    tm: int,
    chunk: int,
    tn: int,
    preferred_element_type: jax.typing.DTypeLike,
    interpret: bool,
    out_hbm: bool,
) -> jax.Array:
  """One grid step per SAME-TILE VISIT PAIR: two expert streams, one out tile.

  The paired arm of `mxfp4_gmm` (`paired_expert_visits=True`): grid
  metadata comes from `_make_paired_group_metadata`, and each grid step
  runs the planar_t_gather chunk walk for group `group_ids_a[step]` AND
  (when the no-partner sentinel does not fire) group `group_ids_b[step]`
  INTERLEAVED per chunk, sharing one lhs block fetch. The two streams'
  dependence chains fill each other's interlock windows (measured wall
  ratio 1.257 for 2x the work vs 2.0 serial; rung 3, pre-registered,
  sponge 829e304d). Partnerless visits run the single-stream walk under
  a predicate, so their compute cost is exactly the rolled kernel's;
  the pipeline still prefetches the (duplicate) stream-b block for
  them, a strip of wasted HBM read the depth-1 schedule hides behind
  the visit it precedes.

  `packed` / `scales` are the planar_t CHUNK-MAJOR restaged stacks
  (`[E, steps, chunk/2, n]` / `[E, steps, chunk/SCALE_GROUP, n]`), the
  same values `mxfp4_gmm`'s transposed branch feeds the rolled kernel;
  they are passed as BOTH the a and b operands with per-stream index
  maps, so no weight bytes are duplicated in HBM.
  """
  steps = max(k // chunk, 1)
  tiles_n = (n + tn - 1) // tn

  def kernel(
      group_metadata,
      group_offset,
      lhs_ref,
      a_packed_ref,
      a_scales_ref,
      b_packed_ref,
      b_scales_ref,
      out_ref,
      acc_a,
      acc_b,
  ):
    del group_offset
    group_offsets, group_ids_a, group_ids_b, m_tile_ids = group_metadata
    grid_id = pl.program_id(1)
    gid_a = group_ids_a[grid_id]
    gid_b = group_ids_b[grid_id]
    has_b = gid_b != gid_a
    acc_a[...] = jnp.zeros_like(acc_a)
    acc_b[...] = jnp.zeros_like(acc_b)
    # Loop-invariant f32 value plane, as the rolled kernel builds it.
    gather_values = _e2m1_gather_table(tn).astype(jnp.float32)
    dims_t = (((1,), (0,)), ((), ()))

    def dual_body(i, carry):
      lhs_chunk = lhs_ref[:, pl.ds(i * chunk, chunk)]
      a_rhs = _dequantize_chunk_planar_t_gather(
          a_packed_ref[i], a_scales_ref[i], values=gather_values
      ).astype(lhs_chunk.dtype)
      acc_a[...] += lax.dot_general(
          lhs_chunk,
          a_rhs,
          dimension_numbers=dims_t,
          preferred_element_type=jnp.float32,
      )
      b_rhs = _dequantize_chunk_planar_t_gather(
          b_packed_ref[i], b_scales_ref[i], values=gather_values
      ).astype(lhs_chunk.dtype)
      acc_b[...] += lax.dot_general(
          lhs_chunk,
          b_rhs,
          dimension_numbers=dims_t,
          preferred_element_type=jnp.float32,
      )
      return carry

    def single_body(i, carry):
      lhs_chunk = lhs_ref[:, pl.ds(i * chunk, chunk)]
      a_rhs = _dequantize_chunk_planar_t_gather(
          a_packed_ref[i], a_scales_ref[i], values=gather_values
      ).astype(lhs_chunk.dtype)
      acc_a[...] += lax.dot_general(
          lhs_chunk,
          a_rhs,
          dimension_numbers=dims_t,
          preferred_element_type=jnp.float32,
      )
      return carry

    # Top-level branch, TWO rolled walks: the dual branch must stay ONE
    # interleaved loop (the interleave IS the fill mechanism), and the
    # single branch must cost exactly the rolled walk. A per-chunk
    # predicate on stream b would put a region boundary between the two
    # streams inside the loop body.
    @pl.when(has_b)
    def _():
      lax.fori_loop(0, steps, dual_body, 0)

    @pl.when(jnp.logical_not(has_b))
    def _():
      lax.fori_loop(0, steps, single_body, 0)

    # Masked combine, `mxfp4_gmm` epilogue semantics per stream: the two
    # groups' row ranges are disjoint, and rows outside both keep their
    # prior contents.
    row = (
        lax.broadcasted_iota(jnp.int32, acc_a.shape, 0)
        + m_tile_ids[grid_id] * tm
    )
    mask_a = jnp.logical_and(
        row >= group_offsets[gid_a], row < group_offsets[gid_a + 1]
    )
    out_ref[...] = jnp.where(
        mask_a, acc_a[...], out_ref[...].astype(jnp.float32)
    ).astype(preferred_element_type)

    @pl.when(has_b)
    def _():
      mask_b = jnp.logical_and(
          row >= group_offsets[gid_b], row < group_offsets[gid_b + 1]
      )
      out_ref[...] = jnp.where(
          mask_b, acc_b[...], out_ref[...].astype(jnp.float32)
      ).astype(preferred_element_type)

  def lhs_indices(n_i, grid_id, group_metadata, group_offset):
    del n_i, group_offset
    return group_metadata[3][grid_id], 0

  def a_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[1][grid_id], 0, 0, n_i

  def b_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[2][grid_id], 0, 0, n_i

  def out_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[3][grid_id], n_i

  weight_specs = [
      pl.BlockSpec((None, steps, chunk // 2, tn), a_indices),
      pl.BlockSpec((None, steps, chunk // SCALE_GROUP, tn), a_indices),
      pl.BlockSpec((None, steps, chunk // 2, tn), b_indices),
      pl.BlockSpec((None, steps, chunk // SCALE_GROUP, tn), b_indices),
  ]
  cost = pl.CostEstimate(
      flops=2 * 2 * m * k * n,
      # Both streams' stacks, the existing full-stack accounting
      # convention (see the rolled call's CostEstimate).
      bytes_accessed=lhs.size * lhs.itemsize
      + 2 * (packed.size + scales.size)
      + m * n * jnp.dtype(preferred_element_type).itemsize,
      transcendentals=0,
  )
  call = pl.pallas_call(
      kernel,
      out_shape=pltpu.HBM((m, n), preferred_element_type)
      if out_hbm and not interpret
      else jax.ShapeDtypeStruct((m, n), preferred_element_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[pl.BlockSpec((tm, k), lhs_indices)] + weight_specs,
          out_specs=pl.BlockSpec((tm, tn), out_indices),
          grid=(tiles_n, num_paired_tiles),
          scratch_shapes=[
              pltpu.VMEM((tm, tn), jnp.float32),
              pltpu.VMEM((tm, tn), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'arbitrary')
      ),
      interpret=interpret,
      cost_estimate=cost,
  )
  return call(
      group_metadata,
      jnp.array([0], dtype=jnp.int32),
      lhs,
      packed,
      scales,
      packed,
      scales,
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        'preferred_element_type',
        'tiling',
        'interpret',
        'dequant',
        'native_f8',
        'fold_scales',
        'metadata_impl',
        'paired_expert_visits',
        'lhs_hbm',
        'out_hbm',
        'deep_vmem_budget',
        'deep_continuous',
        'deep_stream_rows',
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
    dequant: str = DEQUANT_DEFAULT,
    native_f8: bool = False,
    fold_scales: bool = False,
    single_group: tuple[jax.Array, jax.Array] | None = None,
    metadata_impl: str = 'repeat',
    paired_expert_visits: bool = False,
    lhs_scales: jax.Array | None = None,
    lhs_hbm: bool = False,
    out_hbm: bool = False,
    deep_vmem_budget: int | None = None,
    deep_continuous: bool = False,
    group_metadata: GroupMetadata | None = None,
    deep_stream_rows: bool = False,
) -> jax.Array:
  """`lhs[rows of group i, :] @ dequant(packed[i], scales[i]).T` per group.

  Args:
    lhs: `[M, K]` activations, rows grouped by expert. May be float8_e4m3fn (the
      fp8 x fp8 opt-in) only with `native_f8=True`.
    packed: `[E, N, K/2]` uint8 MXFP4 nibble pairs (checkpoint layout, or the
      `repack_planar_stack` order under planar `dequant` modes).
    scales: `[E, N, K/SCALE_GROUP]` uint8 e8m0 exponents.
    group_sizes: `[E]` int32 rows per expert; entries may be 0 and need not sum
      to M (rows past the total are left as zeros, which is how the
      expert-parallel caller parks assignments it does not own).
    lhs_scales: optional `[M, K/SCALE_GROUP]` uint8 e8m0 ACTIVATION exponents
      (the MXFP8 arm, `quantize_mxfp8`): requires `native_f8` with an f8e4m3fn
      `lhs` and folds per subchannel next to the weight scales
      (`_subchannel_dot_f8`). Streams as its own `[tm, K/32]` block on the lhs
      index map and restages chunk-major for the rolled k walk exactly like the
      weight scale codes.
    preferred_element_type: output dtype.
    tiling: `(tm, k_chunk, tn)`. Blocks always span the whole contraction;
      `k_chunk` is how much of it the kernel dequantizes at a time (multiple of
      SCALE_GROUP, divides K). `tm` must divide M. None derives them from the
      problem.
    interpret: run the Pallas kernel in interpret mode (used off TPU).
    dequant: which dequant inner loop to emit, one of `DEQUANT_MODES`. All are
      bitwise equivalent; this exists so the bench can A/B them. Ignored under
      `native_f8` (that path never materializes dequantized weights).
    native_f8: mxfp4-NATIVE path (Ghostfish): relabel the nibbles to f8e4m3fn in
      registers and apply the k32 scales post-dot on the f32 accumulator
      (`_subchannel_dot_f8`) instead of dequantizing weight elements on the VPU.
      Chip-gated at the config layer; measured SLOWER than the split_mxu dequant
      on gf (module docstring), kept as the research/insurance arm.
    fold_scales: with `native_f8`, run ONE deep MXU pass per chunk by folding
      each subchannel's exponent delta into the codes
      (`_subchannel_dot_f8_folded`). Exact within the fold window; rounds or
      flushes subchannels sitting far below their chunk-mates (bounds pinned in
      moe_gmm_fp8_test).
    single_group: optional `(group_id, size)` scalars declaring that
      `group_sizes` is one-hot with the group's rows at [0, size) - the kiloblox
      chunk case. The grid metadata then comes from `one_hot_group_metadata`
      (pure arithmetic) instead of the histogram / searchsorted / repeat build,
      whose binary-search loop the decode trace bills per chunk iteration. Never
      changes results; the two builds are pinned equal on the read prefix in
      moe_gmm_test.
    metadata_impl: `_make_group_metadata` spelling
      (`K3MoeConfig.gmm_metadata_impl`): 'repeat' (today's program) or
      'searchsorted' (scatter-free; kills the per-call SparseCore scatter
      offload round trip on GhostLite). Identical integers either way; ignored
      under `single_group`.
    paired_expert_visits: merge SAME-TILE visit pairs into one dual-stream grid
      step (`_make_paired_group_metadata` + `_paired_planar_t_gather_call`): at
        decode, two activated experts' chunk walks over one lhs tile run
        interleaved instead of as two visits, filling each other's in-body
        interlock windows (measured wall ratio 1.257 for 2x the work; rung 3,
        pre-registered). Partnerless visits run the single-stream walk under a
        predicate. planar_t_gather only; does not compose with the deep modes,
        native_f8, f8 planes, MXFP8 activations or `single_group` (the config
        layer forbids those combinations). Default False
        (`K3MoeConfig.gmm_paired_expert_visits`); never changes results, pinned
        bitwise against the unpaired kernel in moe_gmm_test.
    lhs_hbm: pre-color the lhs OPERAND HBM, `mxfp4_fused_gate_up_gmm` semantics
      (address space only, values bit for bit; no-op under `interpret`).
    out_hbm: required-HBM OUTPUT birth, `mxfp4_fused_gate_up_gmm` semantics (the
      chain_hbm arm; no-op under `interpret`).
    deep_continuous: deep modes only (`K3MoeConfig.gmm_deep_stream_continuous`):
      stream the slab tiles continuously ACROSS expert visits instead of priming
      and draining the DEEP_STREAM_DEPTH pipeline inside every visit
      (`_deep_planar_t_b11_call`). Schedule only by construction (accumulation
      order, window scale, epilogue, scratch identical); exact pins:
      moe_gmm_test.DeepStreamParityTest (Pallas interpret on CPU) and the GF
      served tray ladder (on chip, kernel level); on chip model program pin:
      moe_deep_stream_model_pin_forge_test. Ignored by non-deep modes and by the
      rolled fallback.
    deep_stream_rows: deep modes only (`K3MoeConfig.gmm_deep_stream_rows`):
      stream the visited `[tm, K]` lhs and `[tm, N]` out row tiles through VMEM
      instead of holding the whole tray resident (`_deep_planar_t_b11_call`).
      Placement only by construction, bitwise on every routed row;
      `deep_vmem_budget` then models the streamed footprint. Ignored by non-deep
      modes and by the rolled fallback.
    group_metadata: optional PRECOMPUTED grid metadata (`shared_group_metadata`:
      the layer's gate+up and down kernels share one build). Checked by shape
      against this kernel's `(m, tm, num_groups)`; the same integers the kernel
      would build itself, so values never change. Not combinable with
      `single_group` or `paired_expert_visits` (their metadata is different by
      construction; the config layer never sets both).

  Returns:
    `[M, N]` products.
  """
  m, k = lhs.shape
  transposed = dequant in PLANAR_T_MODES
  f8_planes = dequant in F8_PLANE_MODES
  if transposed and not f8_planes and scales.ndim == 4:
    # Chunk-major scale params ([E, K/512, 16, N]): normalize to the
    # k-major 3-D contract; the merge cancels against the chunk-major
    # restage the kernel applies, so the pallas call reads the param
    # with no in-step transform (restage_scales_chunk_major).
    scales = flatten_chunk_major_scales(scales)
  num_groups = packed.shape[0]
  n = packed.shape[2] if transposed else packed.shape[1]
  # Clamp before validating: a tile larger than the array is just the
  # whole array, which toy shapes rely on.
  tm, chunk, tn = tiling or default_tiling(
      m,
      k,
      n,
      force_chunk=PLANAR_WINDOW
      if (dequant in PLANAR_MODES or transposed)
      else None,
      f8_planes=f8_planes,
  )
  tm, chunk, tn = min(tm, m), min(chunk, k), min(tn, n)
  if dequant in PLANAR_MODES or transposed:
    if native_f8:
      raise ValueError(
          'native_f8 consumes the checkpoint (interleaved) nibble order;'
          ' planar-repacked weights would be misread. The config layer'
          ' forbids the combination; reaching here is a wiring bug.'
      )
    if chunk != PLANAR_WINDOW:
      raise ValueError(
          f'planar weights are repacked per {PLANAR_WINDOW}-element window,'
          f' so the kernel k chunk must equal it; got chunk {chunk} for k'
          f' {k}. Callers gate on can_use_kernel(..., dequant=...) and fall'
          ' back to the reference for shapes that cannot serve the window.'
      )
  _validate(
      lhs,
      packed,
      scales,
      group_sizes,
      (tm, chunk, tn),
      transposed=transposed,
      f8_planes=f8_planes,
  )
  if (
      lhs.dtype == jnp.float8_e4m3fn
      and not native_f8
      and dequant not in _E4M3FN_WEIGHT_MODES
  ):
    raise ValueError(
        'f8e4m3fn activations require native_f8=True or'
        f" dequant='{DEQUANT_PLANAR_T_F8FN}': any other path holds bf16"
        ' weight values, and Mosaic answers an f8 lhs against a bf16 rhs'
        ' by widening operand 0 back - so the cast would be pure'
        ' activation rounding at zero MXU benefit.'
    )
  if fold_scales and not native_f8:
    raise ValueError('fold_scales is a native_f8 variant; set both.')
  if lhs_scales is not None:
    if not native_f8 or lhs.dtype != jnp.float8_e4m3fn:
      raise ValueError(
          'lhs_scales (MXFP8 activations) require native_f8 with an'
          ' f8e4m3fn lhs: the scales are the e8m0 half of the'
          ' quantize_mxfp8 pair. This does NOT extend to the planar_t'
          ' FOLD modes, and not merely because nobody wired it: those'
          ' consume the weight e8m0 BEFORE the dot (folded into the'
          ' byte, plus one post-dot window scale), so a per-32 lhs'
          ' scale has nowhere to land - the k32 groups are already'
          ' summed inside the dot by the time the post-dot scale'
          ' applies. Composing them needs a per-subchannel dot and'
          ' rescale, i.e. a different kernel structure, and the'
          ' component gate has already CONFIRMED-REJECTED the MXFP8'
          ' rounding it would serve.'
      )
    if fold_scales:
      raise ValueError(
          'lhs_scales vary per 32-wide block, which the fold_scales'
          ' one-deep-pass structure cannot fold; use the unfolded'
          ' native path.'
      )
    if lhs_scales.shape != (m, k // SCALE_GROUP):
      raise ValueError(
          f'lhs_scales must be [{m}, {k // SCALE_GROUP}], got'
          f' {lhs_scales.shape}.'
      )
    if lhs_scales.dtype != jnp.uint8:
      raise ValueError(f'lhs_scales must be uint8, got {lhs_scales.dtype}.')
  if not native_f8:
    # The split modes' k permutation applies to the DEQUANT dots only;
    # the native path consumes the natural checkpoint order, and the
    # planar modes' weights were repacked offline (identity, no-op).
    lhs = _prepare_lhs(lhs, dequant, chunk)
  if transposed:
    # Chunk-major restage, at ZERO cost: splitting the k-major dim of
    # the planar_t layout is a row-major-preserving reshape (same
    # bytes, same order), and it lets the rolled k walk index the
    # LEADING dim of the in-kernel ref, which needs no lane or sublane
    # alignment (the constraint `_stage_scale_codes` exists to dodge).
    steps = max(k // chunk, 1)
    packed = packed.reshape(
        num_groups, steps, chunk if f8_planes else chunk // 2, n
    )
    if f8_planes:
      # The window maxima stay the rank 3 leaf `[E, 8 steps, N]`
      # (`PLANE_SCALE_ROWS_PER_WINDOW`): the kernels read the whole
      # `[8 steps, tn]` block in the leaf's at rest tiling. A
      # `[E, steps, 1, N]` view made Mosaic demand a singleton second
      # minor tiling no rank 3 parameter layout can express, and one row
      # per window put the expert dim on the sublanes at rest; either way
      # XLA relayouted every scale operand per step (the copy.22918
      # class; w27's planes16s gate).
      rows = steps * PLANE_SCALE_ROWS_PER_WINDOW
      if scales.shape != (num_groups, rows, n):
        raise ValueError(
            f'window scales must be [E, {PLANE_SCALE_ROWS_PER_WINDOW} *'
            f' K/{PLANAR_WINDOW}, N] = {(num_groups, rows, n)}, got'
            f' {scales.shape}.'
        )
    else:
      scales = scales.reshape(num_groups, steps, chunk // SCALE_GROUP, n)
  tiles_n = (n + tn - 1) // tn

  if paired_expert_visits:
    # Every unsupported combination RAISES rather than silently running
    # unpaired: a flag that quietly degrades reads as "pairing does
    # nothing" instead of "pairing never ran" (the deep bench's
    # negative-control lesson).
    if dequant in DEEP_DEQUANT_BODIES:
      raise ValueError(
          'paired_expert_visits composes with the ROLLED schedule; the deep'
          ' modes carry their own pipeline. Set one or the other.'
      )
    if dequant != DEQUANT_PLANAR_T_GATHER:
      raise ValueError(
          'paired_expert_visits is implemented for the planar_t_gather body'
          f' only; got dequant={dequant!r}. The config layer forbids other'
          ' combinations; reaching here is a wiring bug.'
      )
    if native_f8 or f8_planes or lhs_scales is not None:
      raise ValueError(
          'paired_expert_visits does not compose with native_f8, f8 planes'
          ' or MXFP8 activations.'
      )
    if single_group is not None:
      raise ValueError(
          'paired_expert_visits pairs multiple groups sharing an m tile;'
          ' the single_group one-hot declaration has exactly one group and'
          ' nothing to pair. The kiloblox chunk caller must not set both.'
      )
    if group_metadata is not None:
      raise ValueError(
          'paired_expert_visits builds its own PAIRED metadata; a'
          ' precomputed single-visit group_metadata cannot feed it.'
      )
    paired_metadata, num_paired_tiles = _make_paired_group_metadata(
        group_sizes, m=m, tm=tm, num_groups=num_groups, impl=metadata_impl
    )
    if lhs_hbm and not interpret:
      lhs = pltpu.with_memory_space_constraint(lhs, pltpu.MemorySpace.HBM)
    return _paired_planar_t_gather_call(
        lhs,
        packed,
        scales,
        paired_metadata,
        num_paired_tiles=num_paired_tiles,
        m=m,
        n=n,
        k=k,
        tm=tm,
        chunk=chunk,
        tn=tn,
        preferred_element_type=preferred_element_type,
        interpret=interpret,
        out_hbm=out_hbm,
    )

  if group_metadata is not None:
    if single_group is not None:
      raise ValueError(
          'group_metadata (a precomputed general build) and single_group'
          ' (the one-hot closed form) are two metadata sources; pass one.'
      )
    grid_metadata, num_active_tiles = _check_group_metadata(
        group_metadata, m=m, tm=tm, num_groups=num_groups
    )
  elif single_group is not None:
    grid_metadata, num_active_tiles = one_hot_group_metadata(
        single_group[0], single_group[1], m=m, tm=tm, num_groups=num_groups
    )
  else:
    grid_metadata, num_active_tiles = _make_group_metadata(
        group_sizes, m=m, tm=tm, num_groups=num_groups, impl=metadata_impl
    )

  if dequant in DEEP_DEQUANT_BODIES:
    deep_body = DEEP_DEQUANT_BODIES[dequant]
    deep_tn = (
        _deep_fit_tn(
            m=m,
            k=k,
            n=n,
            tm=tm,
            chunk=chunk,
            tn=tn,
            num_stacks=1,
            budget=deep_vmem_budget,
            lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
            out_bytes=jnp.dtype(preferred_element_type).itemsize,
            stream_rows=deep_stream_rows,
            f8_planes=f8_planes,
        )
        if deep_vmem_budget is not None
        else tn
    )
    if deep_tn is not None and _deep_eligible(
        m=m, k=k, n=n, tm=tm, chunk=chunk, tn=deep_tn
    ):
      if lhs_hbm and not interpret:
        lhs = pltpu.with_memory_space_constraint(lhs, pltpu.MemorySpace.HBM)
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
          out_hbm=out_hbm,
          body=deep_body,
          continuous=deep_continuous,
          stream_rows=deep_stream_rows,
      )
    # Buffers the deep pipeline does not serve (prefill widths) take
    # the bitwise-identical rolled kernel; only the schedule differs.
    dequant = deep_body

  use_k_loop = _use_k_loop(k, chunk)
  has_lhs_scales = lhs_scales is not None

  def kernel(
      group_metadata,
      group_offset,
      lhs_ref,
      packed_ref,
      scales_ref,
      *rest,
  ):
    # The MXFP8 arm adds one input ref (after the weight pair, matching
    # in_specs order) and one restage scratch (after the weight scale
    # codes); unpack by the static flag so the plain path's signature
    # is byte for byte what it always was.
    rest = list(rest)
    lhs_scales_ref = rest.pop(0) if has_lhs_scales else None
    out_ref, acc, scale_codes_ref = rest[:3]
    rest = rest[3:]
    lhs_scale_codes_ref = rest.pop(0) if has_lhs_scales else None
    # Plane modes only: this grid step's window maxima as f32 rows
    # (`_stage_plane_scales`).
    plane_exps_ref = rest.pop(0) if f8_planes else None
    if rest:
      raise ValueError(f'{len(rest)} unexpected trailing kernel refs.')
    del group_offset
    group_offsets, group_ids, m_tile_ids = group_metadata
    grid_id = pl.program_id(1)

    acc[...] = jnp.zeros_like(acc)
    # Walk the contraction in chunks. The BLOCK covers the whole of K
    # (the scales' minor dim, k/32, has to be lane-aligned or whole,
    # and 16 is neither), but dequantizing all of it at once puts a
    # [tn, K] chain in flight and Mosaic spills: the 2048-bucket
    # compile reported 127.49M of register-allocator spill slots
    # against a 127.94M VMEM budget. Chunking bounds that working set
    # while keeping every dequantized value in VMEM for exactly one
    # MXU consumption.
    #
    # The walk is a REAL loop, not a python unroll: unrolled, one
    # kernel instance is ~200k+ ISA bundles, over the single-overlay
    # HLO-function ceiling, so the compiler force-inlines a full copy
    # of it at every one of the unrolled program's 92 per-layer call
    # sites (the ~3.7 GiB program-text bloat). Rolled, the kernel fits
    # the ceiling and every call site shares one copy. The scale codes
    # are restaged chunk-major once per grid step so the loop body's
    # accesses are all legal (`_stage_scale_codes`).
    # The planar_gather decode's 16-entry value table: loop-invariant,
    # built once per grid step so the rolled k walk's body does not
    # depend on LICM to hoist the iota chain. planar_t_gather builds
    # the f32 VALUE plane (its per-chunk scaled tables multiply it by
    # the chunk's exps). None for every other mode.
    gather_table = (
        _e2m1_gather_table(chunk // 2)
        if dequant == DEQUANT_PLANAR_GATHER and not native_f8
        else None
    )
    gather_values = (
        _e2m1_gather_table(tn).astype(jnp.float32)
        if dequant == DEQUANT_PLANAR_T_GATHER
        else None
    )

    def chunk_step(
        lhs_chunk, packed_chunk, scales_chunk, lhs_scales_chunk=None
    ):
      if native_f8:
        # mxfp4-NATIVE: nibbles relabel to f8 in registers, scales
        # fold post-dot on the f32 partials; folded = one deep MXU
        # pass per chunk, unfolded = one 32-deep pass per subchannel.
        # MXFP8 activations ride the unfolded form (validated above)
        # with their e8m0 exponents folded per subchannel too.
        if fold_scales:
          acc[...] += _subchannel_dot_f8_folded(
              lhs_chunk, packed_chunk, scales_chunk
          )
        else:
          acc[...] += _subchannel_dot_f8(
              lhs_chunk,
              packed_chunk,
              scales_chunk,
              lhs_group_scales=lhs_scales_chunk,
          )
        return
      if transposed:
        # Scaled-table gather: the rhs chunk arrives [chunk, tn]
        # (k-major), so the contraction is lhs-minor x rhs-second-minor,
        # the canonical [m, k] x [k, n] matmul orientation. Same
        # products, same k accumulation order as the [tn, chunk] form.
        if f8_planes:
          # LOAD-TIME f8: the operand IS the ref. No table, no gather,
          # no fold - `repack_planar_t_f8_stack` did all of it once at
          # conversion - so the chunk step is a same-width bitcast (the
          # mode's dtype: b11 for planar_t_f8, e4m3fn for the f8fn
          # planes), the dot, and the window scale row (already f32,
          # `_stage_plane_scales`) on the f32 accumulator. Bitwise the
          # matching in-kernel fold (identical bytes and wscale) at 2x
          # the weight bytes streamed per visit.
          acc[...] += (
              lax.dot_general(
                  lhs_chunk,
                  lax.bitcast_convert_type(
                      packed_chunk, F8_PLANE_DTYPES[dequant]
                  ),
                  dimension_numbers=(((1,), (0,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
              * scales_chunk
          )
          return
        if dequant in PLANAR_T_FOLD_MODES:
          # Dequant DELETION: hand f8 codes to the MXU (e4m3b11 through
          # the GLC gain latch on GhostLite, e4m3fn natively on
          # Ghostfish) and apply the window scale once, on the f32
          # accumulator.
          codes, wscale = _planar_t_fold_chunk(dequant)(
              packed_chunk, scales_chunk
          )
          acc[...] += (
              lax.dot_general(
                  lhs_chunk,
                  codes,
                  dimension_numbers=(((1,), (0,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
              * wscale
          )
          return
        rhs_chunk = _dequantize_chunk_planar_t_gather(
            packed_chunk, scales_chunk, values=gather_values
        ).astype(lhs_chunk.dtype)
        acc[...] += lax.dot_general(
            lhs_chunk,
            rhs_chunk,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return
      rhs_chunk = _dequant_rhs(
          packed_chunk, scales_chunk, dequant, gather_table=gather_table
      ).astype(lhs_chunk.dtype)
      acc[...] += lax.dot_general(
          lhs_chunk,
          rhs_chunk,
          # Contract the minor dim of both: the weights are stored
          # [out, in], so this is the native weight-stationary form.
          dimension_numbers=(((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    if transposed:
      # The planar_t restage put the chunk index on the refs' LEADING
      # dim outside the kernel, so both the rolled walk and the toy
      # unroll index it directly: no lane/sublane alignment constraint,
      # no scale-code staging pass.
      if f8_planes:
        assert plane_exps_ref is not None
        exps_ref = plane_exps_ref
        _stage_plane_scales(scales_ref, exps_ref)

        def scale_row(i):
          return exps_ref[pl.ds(i * PLANE_SCALE_ROWS_PER_WINDOW, 1), :]

      else:

        def scale_row(i):
          return scales_ref[i]

      def chunk_body_t(i, carry):
        chunk_step(
            lhs_ref[:, pl.ds(i * chunk, chunk)], packed_ref[i], scale_row(i)
        )
        return carry

      if use_k_loop:
        lax.fori_loop(0, k // chunk, chunk_body_t, 0)
      else:
        for c in range(0, k, chunk):
          chunk_body_t(c // chunk, 0)
    elif use_k_loop:
      _stage_scale_codes(scales_ref, scale_codes_ref, chunk)
      if has_lhs_scales:
        # The lhs e8m0 plane has the same sub-lane minor (k/32) as the
        # weight scales, so the rolled walk restages it the same way.
        _stage_scale_codes(lhs_scales_ref, lhs_scale_codes_ref, chunk)

      def chunk_body(i, carry):
        chunk_step(
            lhs_ref[:, pl.ds(i * chunk, chunk)],
            packed_ref[:, pl.ds(i * (chunk // 2), chunk // 2)],
            scale_codes_ref[i],
            None if lhs_scale_codes_ref is None else lhs_scale_codes_ref[i],
        )
        return carry

      lax.fori_loop(0, k // chunk, chunk_body, 0)
    else:
      # Toy contractions (chunk not 256-divisible): static unroll, the
      # per-chunk slices need no alignment and the text is small.
      for c in range(0, k, chunk):
        chunk_step(
            lhs_ref[:, c : c + chunk],
            packed_ref[:, c // 2 : (c + chunk) // 2],
            scales_ref[:, c // SCALE_GROUP : (c + chunk) // SCALE_GROUP],
            None
            if lhs_scales_ref is None
            else lhs_scales_ref[
                :, c // SCALE_GROUP : (c + chunk) // SCALE_GROUP
            ],
        )

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
    out_ref[...] = jnp.where(
        mask, acc[...], old_f32
    ).astype(preferred_element_type)

  def lhs_indices(n_i, grid_id, group_metadata, group_offset):
    del n_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[1][grid_id], n_i, 0

  def rhs_indices_t(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[1][grid_id], 0, 0, n_i

  def scales_indices_t(n_i, grid_id, group_metadata, group_offset):
    # Plane modes: the rank 3 window maxima leaf `[E, steps, N]`.
    del group_offset
    return group_metadata[1][grid_id], 0, n_i

  def out_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[2][grid_id], n_i

  if transposed:
    steps = max(k // chunk, 1)
    weight_specs = [
        pl.BlockSpec(
            (None, steps, chunk if f8_planes else chunk // 2, tn),
            rhs_indices_t,
        ),
        pl.BlockSpec(
            (None, steps * PLANE_SCALE_ROWS_PER_WINDOW, tn), scales_indices_t
        )
        if f8_planes
        else pl.BlockSpec(
            (None, steps, chunk // SCALE_GROUP, tn), rhs_indices_t
        ),
    ]
  else:
    weight_specs = [
        pl.BlockSpec((None, tn, k // 2), rhs_indices),
        pl.BlockSpec((None, tn, k // SCALE_GROUP), rhs_indices),
    ]

  lhs_scale_specs = (
      [pl.BlockSpec((tm, k // SCALE_GROUP), lhs_indices)]
      if has_lhs_scales
      else []
  )
  lhs_scale_scratch = (
      [
          # Chunk-major restage of the ACTIVATION e8m0 plane for the
          # rolled k walk, the lhs twin of the weight scale codes.
          pltpu.VMEM((max(k // chunk, 1), tm, chunk // SCALE_GROUP), jnp.uint8)
      ]
      if has_lhs_scales
      else []
  )
  # Plane modes: this grid step's window maxima widened to f32 rows
  # (`_stage_plane_scales`), one per weight stack.
  plane_scale_scratch = (
      [
          pltpu.VMEM(
              (max(k // chunk, 1) * PLANE_SCALE_ROWS_PER_WINDOW, tn),
              jnp.float32,
          )
      ]
      if f8_planes
      else []
  )
  cost = pl.CostEstimate(
      flops=2 * m * k * n,
      bytes_accessed=lhs.size * lhs.itemsize
      + packed.size
      + scales.size
      + (lhs_scales.size if has_lhs_scales else 0)
      + m * n * jnp.dtype(preferred_element_type).itemsize,
      transcendentals=0,
  )
  call = pl.pallas_call(
      kernel,
      # out_hbm (the chain_hbm arm) births the output with a required
      # HBM assignment, so the next consumer streams it instead of the
      # policy round-tripping it through VMEM (see lhs_hbm).
      out_shape=pltpu.HBM((m, n), preferred_element_type)
      if out_hbm and not interpret
      else jax.ShapeDtypeStruct((m, n), preferred_element_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[pl.BlockSpec((tm, k), lhs_indices)]
          + weight_specs
          + lhs_scale_specs,
          out_specs=pl.BlockSpec((tm, tn), out_indices),
          grid=(tiles_n, num_active_tiles),
          scratch_shapes=[
              pltpu.VMEM((tm, tn), jnp.float32),
              # Chunk-major e8m0 codes for the rolled k walk (see
              # `_stage_scale_codes`); dead when the walk is unrolled
              # and under planar_t (whose restage happened outside).
              pltpu.VMEM(
                  (max(k // chunk, 1), tn, chunk // SCALE_GROUP), jnp.uint8
              ),
          ]
          + lhs_scale_scratch
          + plane_scale_scratch,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'arbitrary')
      ),
      interpret=interpret,
      cost_estimate=cost,
  )
  if lhs_hbm and not interpret:
    # Applied after every lhs transform, like the fused pair (see
    # mxfp4_fused_gate_up_gmm's lhs_hbm doc).
    lhs = pltpu.with_memory_space_constraint(lhs, pltpu.MemorySpace.HBM)
  args = [
      grid_metadata,
      jnp.array([0], dtype=jnp.int32),
      lhs,
      packed,
      scales,
  ]
  if has_lhs_scales:
    args.append(lhs_scales)
  return call(*args)


@functools.partial(
    jax.jit,
    static_argnames=[
        'situ_beta',
        'situ_linear_beta',
        'preferred_element_type',
        'out_dtype',
        'tiling',
        'interpret',
        'dequant',
        'native_f8',
        'fold_scales',
        'metadata_impl',
        'lhs_hbm',
        'out_hbm',
        'deep_vmem_budget',
        'deep_continuous',
        'deep_stream_rows',
    ],
)
def mxfp4_fused_gate_up_gmm(
    lhs: jax.Array,
    gate_packed: jax.Array,
    gate_scales: jax.Array,
    up_packed: jax.Array,
    up_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    preferred_element_type: jax.typing.DTypeLike = jnp.bfloat16,
    out_dtype: jax.typing.DTypeLike | None = None,
    tiling: tuple[int, int, int] | None = None,
    interpret: bool = False,
    dequant: str = DEQUANT_DEFAULT,
    native_f8: bool = False,
    fold_scales: bool = False,
    single_group: tuple[jax.Array, jax.Array] | None = None,
    metadata_impl: str = 'repeat',
    lhs_scales: jax.Array | None = None,
    lhs_hbm: bool = False,
    out_hbm: bool = False,
    deep_vmem_budget: int | None = None,
    deep_continuous: bool = False,
    group_metadata: GroupMetadata | None = None,
    deep_stream_rows: bool = False,
) -> jax.Array:
  """`situ(lhs @ gate_i.T, lhs @ up_i.T)` per group, in ONE kernel.

  The routed experts' gate (w1) and up (w3) projections consume the
  same lhs rows against identically shaped stacks and feed a pointwise
  SiTU-GLU. Run separately that is two kernel launches, two reads of
  every lhs tile, and two `[M, N]` bf16 activations written to and
  read back from HBM just to combine them. Fused, each grid step
  accumulates BOTH products for its `[tm, tn]` tile (the lhs block is
  fetched once and feeds two MXU contractions), applies SiTU on the
  accumulators in registers, and streams out only the activated
  product - the separate gate/up activations never exist anywhere.

  Numerics match composing `mxfp4_gmm` per projection with `ffw.situ`:
  the accumulators are rounded to `preferred_element_type` before the
  f32 SiTU math, exactly the rounding the unfused path performs when
  it materializes gate/up (the equality is pinned at the tolerances
  `moe_gmm_test` already uses for kernel-vs-reference).

  Args:
    lhs: `[M, K]` activations, rows grouped by expert.
    gate_packed: `[E, N, K/2]` uint8 MXFP4 nibble pairs of the gate (w1) stack.
    gate_scales: `[E, N, K/SCALE_GROUP]` uint8 e8m0 exponents of the gate stack.
    up_packed: `[E, N, K/2]` uint8 nibble pairs of the up (w3) stack.
    up_scales: `[E, N, K/SCALE_GROUP]` uint8 exponents of the up stack.
    group_sizes: `[E]` int32 rows per expert, `mxfp4_gmm` semantics (rows
      outside every group are left unspecified).
    situ_beta: SiTU gate clamp.
    situ_linear_beta: SiTU up clamp (None = linear).
    preferred_element_type: output dtype (unless `out_dtype` narrows the STORE),
      and the dtype the gate/up partial products round through before the f32
      SiTU math (parity with the unfused path's materialized intermediates).
    out_dtype: None (default) stores `preferred_element_type` - today's behavior
      byte for byte. `float8_e4m3fn` narrows ONLY the final
      store: the SiTU result still rounds through `preferred_element_type` FIRST
        and then to f8, deliberately double-rounded so the stored bytes are
        BIT-IDENTICAL to running the bf16 kernel and casting afterwards
        (`_quantize_lhs_f8`), and an A/B between the two spellings therefore
        measures ONLY the deleted materialization (the down gmm's lhs read
        halves, and the bf16 intermediate write + f8 rewrite collapse to one f8
        write). This is prod's GMM2-epilogue placement of the lhs quant: the
        producing kernel emits the quantized transfer type (ag_token.py
        output_quantization_fusion; the effective-fusion finding, NOT the
        config-level fuse_lhs=False input-side knob,
      mcu:1541-1553 - the two are different seams and only the input one is the
        clone posture). The partial-product rounding dtype is NOT changed by
        this arg; an out_dtype equal to `preferred_element_type` is the
        identity. Anything else is
      refused: narrowing the partials would be a different numerics class and no
        caller wants it silently.
    tiling: `(tm, k_chunk, tn)`, as `mxfp4_gmm`. None derives them with the
      fused kernel's doubled rhs working set accounted for.
    interpret: run the Pallas kernel in interpret mode (used off TPU).
    dequant: the dequant inner-loop variant, `mxfp4_gmm` semantics (ignored
      under `native_f8`).
    native_f8: the mxfp4-NATIVE path, `mxfp4_gmm` semantics, applied to BOTH
      contractions (the SiTU epilogue above them is untouched).
    fold_scales: the deep-pass native variant, `mxfp4_gmm` semantics.
    single_group: optional `(group_id, size)` one-hot declaration, as
      `mxfp4_gmm` (the closed-form grid metadata; never changes results).
    metadata_impl: `_make_group_metadata` spelling, as `mxfp4_gmm` ('repeat' |
      'searchsorted' | 'closed_form'; identical integers, ignored under
      `single_group`).
    group_metadata: optional precomputed grid metadata, as `mxfp4_gmm` (the
      build-once path; shape-checked, values identical).
    lhs_scales: optional `[M, K/SCALE_GROUP]` uint8 e8m0 activation exponents
      (the MXFP8 arm), `mxfp4_gmm` semantics; ONE stream folds into BOTH
      contractions (the gate and up stacks share the lhs rows and their
      quantization).
    out_hbm: birth the kernel's OUTPUT with a required HBM assignment
      (`pltpu.HBM` out_shape, the chain_hbm arm). The output aval then carries
      the HBM space, so the NEXT pallas consumer's operand auto-colors HBM too
      (the `ChunkedA2AState` ring precedent) and the generation >= 7 policy can
      never round-trip the intermediate through VMEM sync copies. Address space
      only, values bit for bit. No-op under `interpret`.
    lhs_hbm: pre-color the kernel's lhs OPERAND HBM
      (`pltpu.with_memory_space_constraint` on the value the pallas_call
      actually consumes, i.e. after any k-order permute or fp8 quantize). On
      generation >= 7 chips the custom-call memory-space policy VMEM-colors
      small operands, and when the lhs cannot stay VMEM-resident up to this call
      (the kiloblox merged tray: the wlayout slice trains + the kernel's own
      scoped VMEM crowd it out) buffer assignment materializes it through a
      synchronous HBM->VMEM copy of the FULL buffer right before the kernel. The
      explicit color makes the kernel stream its `(tm, k)` lhs tiles straight
      from HBM instead - an address-space annotation only, values bit for bit.
      No-op under `interpret`.
    deep_continuous: deep modes only: continuous cross-visit slab stream in
      `_deep_planar_t_b11_call` (`K3MoeConfig.gmm_deep_stream_continuous`).
      Schedule only by construction; pins on `mxfp4_gmm`.
    deep_stream_rows: deep modes only: stream the visited lhs / out row tiles
      instead of the whole tray (`K3MoeConfig.gmm_deep_stream_rows`). Placement
      only by construction; pins on `mxfp4_gmm`.

  Returns:
    `[M, N]` SiTU-activated products.
  """
  m, k = lhs.shape
  transposed = dequant in PLANAR_T_MODES
  f8_planes = dequant in F8_PLANE_MODES
  if transposed and not f8_planes:
    # Chunk-major scale params: normalize to the k-major 3-D contract
    # (see mxfp4_gmm; the merge cancels against the restage below).
    if gate_scales.ndim == 4:
      gate_scales = flatten_chunk_major_scales(gate_scales)
    if up_scales.ndim == 4:
      up_scales = flatten_chunk_major_scales(up_scales)
  num_groups = gate_packed.shape[0]
  n = gate_packed.shape[2] if transposed else gate_packed.shape[1]
  tm, chunk, tn = tiling or default_tiling(
      m,
      k,
      n,
      rhs_pairs=2,
      force_chunk=PLANAR_WINDOW
      if (dequant in PLANAR_MODES or transposed)
      else None,
      f8_planes=f8_planes,
  )
  tm, chunk, tn = min(tm, m), min(chunk, k), min(tn, n)
  if dequant in PLANAR_MODES or transposed:
    if native_f8:
      raise ValueError(
          'native_f8 consumes the checkpoint (interleaved) nibble order;'
          ' planar-repacked weights would be misread. The config layer'
          ' forbids the combination; reaching here is a wiring bug.'
      )
    if chunk != PLANAR_WINDOW:
      raise ValueError(
          f'planar weights are repacked per {PLANAR_WINDOW}-element window,'
          f' so the kernel k chunk must equal it; got chunk {chunk} for k'
          f' {k}. Callers gate on can_use_kernel(..., dequant=...) and fall'
          ' back to the reference for shapes that cannot serve the window.'
      )
  _validate(
      lhs,
      gate_packed,
      gate_scales,
      group_sizes,
      (tm, chunk, tn),
      transposed=transposed,
      f8_planes=f8_planes,
  )
  if (
      lhs.dtype == jnp.float8_e4m3fn
      and not native_f8
      and dequant not in _E4M3FN_WEIGHT_MODES
  ):
    raise ValueError(
        'f8e4m3fn activations require native_f8=True or'
        f" dequant='{DEQUANT_PLANAR_T_F8FN}': any other path holds bf16"
        ' weight values, and Mosaic answers an f8 lhs against a bf16 rhs'
        ' by widening operand 0 back - so the cast would be pure'
        ' activation rounding at zero MXU benefit.'
    )
  if out_dtype is None:
    out_dtype = jnp.dtype(preferred_element_type)
  else:
    out_dtype = jnp.dtype(out_dtype)
    if out_dtype not in (
        jnp.dtype(preferred_element_type),
        jnp.dtype(jnp.float8_e4m3fn),
    ):
      raise ValueError(
          'out_dtype narrows only the final STORE, and the only narrow'
          ' store this kernel serves is the f8 down-lhs handoff; got'
          f' out_dtype={out_dtype} against'
          f' preferred_element_type={jnp.dtype(preferred_element_type)}.'
          ' Any other pairing would be a silent numerics-class change.'
      )
  if fold_scales and not native_f8:
    raise ValueError('fold_scales is a native_f8 variant; set both.')
  if lhs_scales is not None:
    if not native_f8 or lhs.dtype != jnp.float8_e4m3fn:
      raise ValueError(
          'lhs_scales (MXFP8 activations) require native_f8 with an'
          ' f8e4m3fn lhs: the scales are the e8m0 half of the'
          ' quantize_mxfp8 pair. This does NOT extend to the planar_t'
          ' FOLD modes, and not merely because nobody wired it: those'
          ' consume the weight e8m0 BEFORE the dot (folded into the'
          ' byte, plus one post-dot window scale), so a per-32 lhs'
          ' scale has nowhere to land - the k32 groups are already'
          ' summed inside the dot by the time the post-dot scale'
          ' applies. Composing them needs a per-subchannel dot and'
          ' rescale, i.e. a different kernel structure, and the'
          ' component gate has already CONFIRMED-REJECTED the MXFP8'
          ' rounding it would serve.'
      )
    if fold_scales:
      raise ValueError(
          'lhs_scales vary per 32-wide block, which the fold_scales'
          ' one-deep-pass structure cannot fold; use the unfolded'
          ' native path.'
      )
    if lhs_scales.shape != (m, k // SCALE_GROUP):
      raise ValueError(
          f'lhs_scales must be [{m}, {k // SCALE_GROUP}], got'
          f' {lhs_scales.shape}.'
      )
    if lhs_scales.dtype != jnp.uint8:
      raise ValueError(f'lhs_scales must be uint8, got {lhs_scales.dtype}.')
  if not native_f8:
    # The split modes' k permutation applies to the DEQUANT dots only.
    lhs = _prepare_lhs(lhs, dequant, chunk)
  if up_packed.shape != gate_packed.shape:
    raise ValueError(
        f'up packed {up_packed.shape} != gate packed {gate_packed.shape}.'
    )
  if up_scales.shape != gate_scales.shape:
    raise ValueError(
        f'up scales {up_scales.shape} != gate scales {gate_scales.shape}.'
    )
  if up_packed.dtype != jnp.uint8 or up_scales.dtype != jnp.uint8:
    raise ValueError(
        'up packed/scales must be uint8, got'
        f' {up_packed.dtype}/{up_scales.dtype}.'
    )
  if transposed:
    # Chunk-major restage outside the kernel (see `mxfp4_gmm`): a free
    # row-major split that lets the k walk index the leading ref dim.
    steps = max(k // chunk, 1)
    plane_rows = chunk if f8_planes else chunk // 2
    gate_packed = gate_packed.reshape(num_groups, steps, plane_rows, n)
    up_packed = up_packed.reshape(num_groups, steps, plane_rows, n)
    if f8_planes:
      # Window maxima stay the rank 3 leaf (see `mxfp4_gmm`).
      rows = steps * PLANE_SCALE_ROWS_PER_WINDOW
      for name, s in (('gate', gate_scales), ('up', up_scales)):
        if s.shape != (num_groups, rows, n):
          raise ValueError(
              f'{name} window scales must be {(num_groups, rows, n)}, got'
              f' {s.shape}.'
          )
    else:
      scale_rows = chunk // SCALE_GROUP
      gate_scales = gate_scales.reshape(num_groups, steps, scale_rows, n)
      up_scales = up_scales.reshape(num_groups, steps, scale_rows, n)
  tiles_n = (n + tn - 1) // tn

  if group_metadata is not None:
    if single_group is not None:
      raise ValueError(
          'group_metadata (a precomputed general build) and single_group'
          ' (the one-hot closed form) are two metadata sources; pass one.'
      )
    grid_metadata, num_active_tiles = _check_group_metadata(
        group_metadata, m=m, tm=tm, num_groups=num_groups
    )
  elif single_group is not None:
    grid_metadata, num_active_tiles = one_hot_group_metadata(
        single_group[0], single_group[1], m=m, tm=tm, num_groups=num_groups
    )
  else:
    grid_metadata, num_active_tiles = _make_group_metadata(
        group_sizes, m=m, tm=tm, num_groups=num_groups, impl=metadata_impl
    )

  if dequant in DEEP_DEQUANT_BODIES:
    deep_body = DEEP_DEQUANT_BODIES[dequant]
    deep_tn = (
        _deep_fit_tn(
            m=m,
            k=k,
            n=n,
            tm=tm,
            chunk=chunk,
            tn=tn,
            num_stacks=2,
            budget=deep_vmem_budget,
            lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
            out_bytes=jnp.dtype(preferred_element_type).itemsize,
            stream_rows=deep_stream_rows,
            f8_planes=f8_planes,
        )
        if deep_vmem_budget is not None
        else tn
    )
    if deep_tn is not None and _deep_eligible(
        m=m, k=k, n=n, tm=tm, chunk=chunk, tn=deep_tn
    ):
      if lhs_hbm and not interpret:
        lhs = pltpu.with_memory_space_constraint(lhs, pltpu.MemorySpace.HBM)
      return _deep_planar_t_b11_call(
          lhs,
          ((gate_packed, gate_scales), (up_packed, up_scales)),
          grid_metadata,
          num_active_tiles,
          tm=tm,
          chunk=chunk,
          tn=deep_tn,
          situ=(situ_beta, situ_linear_beta),
          preferred_element_type=preferred_element_type,
          interpret=interpret,
          out_hbm=out_hbm,
          body=deep_body,
          continuous=deep_continuous,
          stream_rows=deep_stream_rows,
      )
    # Buffers the deep pipeline does not serve (prefill widths) take
    # the bitwise-identical rolled kernel; only the schedule differs.
    dequant = deep_body

  use_k_loop = _use_k_loop(k, chunk)
  has_lhs_scales = lhs_scales is not None

  def kernel(
      group_metadata,
      group_offset,
      lhs_ref,
      gate_packed_ref,
      gate_scales_ref,
      up_packed_ref,
      up_scales_ref,
      *rest,
  ):
    # MXFP8 unpack, `mxfp4_gmm` semantics: one extra input ref (after
    # both weight pairs) and one restage scratch (last).
    rest = list(rest)
    lhs_scales_ref = rest.pop(0) if has_lhs_scales else None
    out_ref, gate_acc, up_acc, gate_codes_ref, up_codes_ref = rest[:5]
    rest = rest[5:]
    lhs_scale_codes_ref = rest.pop(0) if has_lhs_scales else None
    # Plane modes only: this grid step's window maxima as f32 rows, one
    # scratch per stack (`_stage_plane_scales`).
    gate_exps_ref = rest.pop(0) if f8_planes else None
    up_exps_ref = rest.pop(0) if f8_planes else None
    if rest:
      raise ValueError(f'{len(rest)} unexpected trailing kernel refs.')
    del group_offset
    group_offsets, group_ids, m_tile_ids = group_metadata
    grid_id = pl.program_id(1)

    gate_acc[...] = jnp.zeros_like(gate_acc)
    up_acc[...] = jnp.zeros_like(up_acc)
    # Walk the contraction in chunks (see `mxfp4_gmm` for why the block
    # spans K but the dequant does not, and why the walk is a REAL loop
    # rather than a python unroll), interleaved so one lhs chunk load
    # feeds both contractions.
    dims = (((1,), (1,)), ((), ()))

    # One planar_gather value table shared by BOTH stacks (the decode
    # is weight-independent), built outside the rolled k walk; see
    # `mxfp4_gmm`. planar_t_gather builds the f32 value plane its
    # per-chunk scaled tables multiply. None for every other mode.
    gather_table = (
        _e2m1_gather_table(chunk // 2)
        if dequant == DEQUANT_PLANAR_GATHER and not native_f8
        else None
    )
    gather_values = (
        _e2m1_gather_table(tn).astype(jnp.float32)
        if dequant == DEQUANT_PLANAR_T_GATHER
        else None
    )
    dims_t = (((1,), (0,)), ((), ()))

    def chunk_step(
        lhs_chunk,
        gate_packed_c,
        gate_scales_c,
        up_packed_c,
        up_scales_c,
        lhs_scales_chunk=None,
    ):
      if native_f8:
        # mxfp4-NATIVE, both contractions (see `mxfp4_gmm`); the MXFP8
        # activation exponents fold into both, from ONE stream.
        if fold_scales:
          gate_acc[...] += _subchannel_dot_f8_folded(
              lhs_chunk, gate_packed_c, gate_scales_c
          )
          up_acc[...] += _subchannel_dot_f8_folded(
              lhs_chunk, up_packed_c, up_scales_c
          )
        else:
          gate_acc[...] += _subchannel_dot_f8(
              lhs_chunk,
              gate_packed_c,
              gate_scales_c,
              lhs_group_scales=lhs_scales_chunk,
          )
          up_acc[...] += _subchannel_dot_f8(
              lhs_chunk,
              up_packed_c,
              up_scales_c,
              lhs_group_scales=lhs_scales_chunk,
          )
        return
      if transposed:
        # Scaled-table gather, both stacks (see `mxfp4_gmm`): the rhs
        # chunks arrive [chunk, tn], the canonical matmul orientation.
        if f8_planes:
          # LOAD-TIME f8, both stacks (see `mxfp4_gmm`).
          for packed_c, scales_c, target in (
              (gate_packed_c, gate_scales_c, gate_acc),
              (up_packed_c, up_scales_c, up_acc),
          ):
            target[...] += (
                lax.dot_general(
                    lhs_chunk,
                    lax.bitcast_convert_type(
                        packed_c, F8_PLANE_DTYPES[dequant]
                    ),
                    dimension_numbers=dims_t,
                    preferred_element_type=jnp.float32,
                )
                * scales_c
            )
          return
        if dequant in PLANAR_T_FOLD_MODES:
          # Dequant DELETION, both stacks (see `mxfp4_gmm`).
          fold_chunk = _planar_t_fold_chunk(dequant)
          for packed_c, scales_c, target in (
              (gate_packed_c, gate_scales_c, gate_acc),
              (up_packed_c, up_scales_c, up_acc),
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
          return
        gate_rhs = _dequantize_chunk_planar_t_gather(
            gate_packed_c, gate_scales_c, values=gather_values
        ).astype(lhs_chunk.dtype)
        gate_acc[...] += lax.dot_general(
            lhs_chunk,
            gate_rhs,
            dimension_numbers=dims_t,
            preferred_element_type=jnp.float32,
        )
        up_rhs = _dequantize_chunk_planar_t_gather(
            up_packed_c, up_scales_c, values=gather_values
        ).astype(lhs_chunk.dtype)
        up_acc[...] += lax.dot_general(
            lhs_chunk,
            up_rhs,
            dimension_numbers=dims_t,
            preferred_element_type=jnp.float32,
        )
        return
      # One lhs, two dots: under `split` the k permutation is paid once
      # and amortized across both the gate and the up contraction.
      gate_rhs = _dequant_rhs(
          gate_packed_c, gate_scales_c, dequant, gather_table=gather_table
      ).astype(lhs_chunk.dtype)
      gate_acc[...] += lax.dot_general(
          lhs_chunk,
          gate_rhs,
          dimension_numbers=dims,
          preferred_element_type=jnp.float32,
      )
      up_rhs = _dequant_rhs(
          up_packed_c, up_scales_c, dequant, gather_table=gather_table
      ).astype(lhs_chunk.dtype)
      up_acc[...] += lax.dot_general(
          lhs_chunk,
          up_rhs,
          dimension_numbers=dims,
          preferred_element_type=jnp.float32,
      )

    if transposed:
      # Leading-dim chunk indexing (the outside restage), rolled or not.
      if f8_planes:
        assert gate_exps_ref is not None and up_exps_ref is not None
        gate_exps, up_exps = gate_exps_ref, up_exps_ref
        _stage_plane_scales(gate_scales_ref, gate_exps)
        _stage_plane_scales(up_scales_ref, up_exps)

        def chunk_body_t(i, carry):
          row = pl.ds(i * PLANE_SCALE_ROWS_PER_WINDOW, 1)
          chunk_step(
              lhs_ref[:, pl.ds(i * chunk, chunk)],
              gate_packed_ref[i],
              gate_exps[row, :],
              up_packed_ref[i],
              up_exps[row, :],
          )
          return carry

      else:
        # The operand order below is the traced order of the mxfp4
        # kernels' loads; keep it (the 6c programs are pinned byte for
        # byte on it).

        def chunk_body_t(i, carry):
          chunk_step(
              lhs_ref[:, pl.ds(i * chunk, chunk)],
              gate_packed_ref[i],
              gate_scales_ref[i],
              up_packed_ref[i],
              up_scales_ref[i],
          )
          return carry

      if use_k_loop:
        lax.fori_loop(0, k // chunk, chunk_body_t, 0)
      else:
        for c in range(0, k, chunk):
          chunk_body_t(c // chunk, 0)
    elif use_k_loop:
      _stage_scale_codes(gate_scales_ref, gate_codes_ref, chunk)
      _stage_scale_codes(up_scales_ref, up_codes_ref, chunk)
      if has_lhs_scales:
        _stage_scale_codes(lhs_scales_ref, lhs_scale_codes_ref, chunk)

      def chunk_body(i, carry):
        half = pl.ds(i * (chunk // 2), chunk // 2)
        chunk_step(
            lhs_ref[:, pl.ds(i * chunk, chunk)],
            gate_packed_ref[:, half],
            gate_codes_ref[i],
            up_packed_ref[:, half],
            up_codes_ref[i],
            None if lhs_scale_codes_ref is None else lhs_scale_codes_ref[i],
        )
        return carry

      lax.fori_loop(0, k // chunk, chunk_body, 0)
    else:
      for c in range(0, k, chunk):
        packed_slice = slice(c // 2, (c + chunk) // 2)
        scale_slice = slice(c // SCALE_GROUP, (c + chunk) // SCALE_GROUP)
        chunk_step(
            lhs_ref[:, c : c + chunk],
            gate_packed_ref[:, packed_slice],
            gate_scales_ref[:, scale_slice],
            up_packed_ref[:, packed_slice],
            up_scales_ref[:, scale_slice],
            None if lhs_scales_ref is None else lhs_scales_ref[:, scale_slice],
        )

    # Round the partial products to the dtype the unfused path would
    # have materialized them in, then SiTU in f32 (the ffw.situ
    # contract), all in registers.
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
    stored = jnp.where(mask, h, old_f32).astype(
        preferred_element_type
    )
    if out_dtype != jnp.dtype(preferred_element_type):
      # Deliberate DOUBLE rounding (f32 -> pet -> f8): bit-identical to
      # the unfused cast-after-kernel spelling, so the out_dtype A/B
      # measures traffic only. Preserved rows round-trip exactly (every
      # e4m3fn value is representable in bf16 and f32).
      stored = stored.astype(out_dtype)
    out_ref[...] = stored

  def lhs_indices(n_i, grid_id, group_metadata, group_offset):
    del n_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[1][grid_id], n_i, 0

  def rhs_indices_t(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[1][grid_id], 0, 0, n_i

  def scales_indices_t(n_i, grid_id, group_metadata, group_offset):
    # Plane modes: the rank 3 window maxima leaf `[E, steps, N]`.
    del group_offset
    return group_metadata[1][grid_id], 0, n_i

  def out_indices(n_i, grid_id, group_metadata, group_offset):
    del group_offset
    return group_metadata[2][grid_id], n_i

  if transposed:
    steps = max(k // chunk, 1)
    pair_specs = [
        pl.BlockSpec(
            (None, steps, chunk if f8_planes else chunk // 2, tn),
            rhs_indices_t,
        ),
        pl.BlockSpec(
            (None, steps * PLANE_SCALE_ROWS_PER_WINDOW, tn), scales_indices_t
        )
        if f8_planes
        else pl.BlockSpec(
            (None, steps, chunk // SCALE_GROUP, tn), rhs_indices_t
        ),
    ]
  else:
    pair_specs = [
        pl.BlockSpec((None, tn, k // 2), rhs_indices),
        pl.BlockSpec((None, tn, k // SCALE_GROUP), rhs_indices),
    ]

  lhs_scale_specs = (
      [pl.BlockSpec((tm, k // SCALE_GROUP), lhs_indices)]
      if has_lhs_scales
      else []
  )
  lhs_scale_scratch = (
      [
          # Chunk-major restage of the ACTIVATION e8m0 plane for the
          # rolled k walk, the lhs twin of the weight scale codes.
          pltpu.VMEM((max(k // chunk, 1), tm, chunk // SCALE_GROUP), jnp.uint8)
      ]
      if has_lhs_scales
      else []
  )
  # Plane modes: per stack window maxima widened to f32 rows
  # (`_stage_plane_scales`).
  plane_scale_scratch = (
      [
          pltpu.VMEM(
              (max(k // chunk, 1) * PLANE_SCALE_ROWS_PER_WINDOW, tn),
              jnp.float32,
          )
      ]
      * 2
      if f8_planes
      else []
  )
  cost = pl.CostEstimate(
      flops=2 * 2 * m * k * n,
      bytes_accessed=lhs.size * lhs.itemsize
      + gate_packed.size
      + gate_scales.size
      + up_packed.size
      + up_scales.size
      + (lhs_scales.size if has_lhs_scales else 0)
      + m * n * jnp.dtype(out_dtype).itemsize,
      # tanh(gate), sigmoid(gate), tanh(up) per output element.
      transcendentals=3 * m * n,
  )
  call = pl.pallas_call(
      kernel,
      # out_hbm (the chain_hbm arm) births the output with a required
      # HBM assignment, so the next consumer streams it instead of the
      # policy round-tripping it through VMEM (see lhs_hbm).
      out_shape=pltpu.HBM((m, n), out_dtype)
      if out_hbm and not interpret
      else jax.ShapeDtypeStruct((m, n), out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[pl.BlockSpec((tm, k), lhs_indices)]
          + pair_specs
          + pair_specs
          + lhs_scale_specs,
          out_specs=pl.BlockSpec((tm, tn), out_indices),
          grid=(tiles_n, num_active_tiles),
          scratch_shapes=[
              pltpu.VMEM((tm, tn), jnp.float32),
              pltpu.VMEM((tm, tn), jnp.float32),
              # Chunk-major e8m0 codes for the rolled k walk (see
              # `_stage_scale_codes`); dead when the walk is unrolled
              # and under planar_t (whose restage happened outside).
              pltpu.VMEM(
                  (max(k // chunk, 1), tn, chunk // SCALE_GROUP), jnp.uint8
              ),
              pltpu.VMEM(
                  (max(k // chunk, 1), tn, chunk // SCALE_GROUP), jnp.uint8
              ),
          ]
          + lhs_scale_scratch
          + plane_scale_scratch,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'arbitrary')
      ),
      interpret=interpret,
      cost_estimate=cost,
  )
  if lhs_hbm and not interpret:
    # Applied HERE - after every lhs transform - so the color lands on
    # the exact value the pallas_call consumes (see the arg doc).
    lhs = pltpu.with_memory_space_constraint(lhs, pltpu.MemorySpace.HBM)
  args = [
      grid_metadata,
      jnp.array([0], dtype=jnp.int32),
      lhs,
      gate_packed,
      gate_scales,
      up_packed,
      up_scales,
  ]
  if has_lhs_scales:
    args.append(lhs_scales)
  return call(*args)


def mxfp4_gmm_reference(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """`mxfp4_gmm` semantics without Pallas, for CPU and equality tests.

  Dequantizes the whole stack and runs `jax.lax.ragged_dot`, i.e. what
  the model did before the kernel existed. Arguments as `mxfp4_gmm`;
  `preferred_element_type=None` keeps the activation dtype.
  """
  e, n, half_k = packed.shape
  k = half_k * 2
  codes = jnp.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(e, n, k)
  values = jnp.take(
      jnp.asarray(_E2M1_LUT, jnp.float32), codes.astype(jnp.int32)
  )
  exps = lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
  exps = jnp.where(scales == 0, jnp.float32(2.0**-127), exps)
  groups = scales.shape[-1]
  values = values.reshape(e, n, groups, k // groups) * exps[..., None]
  # [E, N, K] -> [E, K, N]: ragged_dot contracts lhs's minor with the
  # rhs's SECOND-minor, while the kernel contracts both minors.
  dense = values.reshape(e, n, k).transpose(0, 2, 1).astype(jnp.bfloat16)
  out = jax.lax.ragged_dot(lhs, dense.astype(lhs.dtype), group_sizes)
  return (
      out
      if preferred_element_type is None
      else out.astype(preferred_element_type)
  )


def f8_plane_gmm_reference(
    lhs: jax.Array,
    planes: jax.Array,
    window_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    plane_dtype,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """`mxfp4_gmm` semantics for LOAD-TIME f8 planes, without Pallas.

  `mxfp4_gmm_reference`'s twin for the `F8_PLANE_MODES` operands:
  decodes the planes through `dequantize_planar_t_f8_stack` (already
  k-major `[E, K, N]`, the `ragged_dot` orientation) and rounds the
  dense stack through bf16 exactly as the mxfp4 reference does, so the
  two references agree bit for bit on the same weights.

  Args:
    lhs: `[M, K]` activations, rows grouped by expert.
    planes: `[E, K, N]` uint8 bytes of `plane_dtype`.
    window_scales: the `[E, 8 * K/PLANAR_WINDOW, N]` uint8 window maxima leaf
      (`repack_planar_t_f8_stack`).
    group_sizes: `[E]` int32 rows per expert.
    plane_dtype: the f8 format the bytes spell (`F8_PLANE_DTYPES`).
    preferred_element_type: output dtype; None keeps the activation dtype.

  Returns:
    `[M, N]` products.
  """
  dense = dequantize_planar_t_f8_stack(
      planes, window_scales, plane_dtype=plane_dtype
  ).astype(jnp.bfloat16)
  out = jax.lax.ragged_dot(lhs, dense.astype(lhs.dtype), group_sizes)
  return (
      out
      if preferred_element_type is None
      else out.astype(preferred_element_type)
  )


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
  """`mxfp4_fused_gate_up_gmm` semantics without Pallas.

  Composes the existing pieces: two `mxfp4_gmm_reference` matmuls and
  the SiTU math, i.e. exactly what `ffw._ragged_expert_combine` did
  before the fusion (so CPU/interpret parity with the unfused model is
  by construction, and rows outside every group come back ZERO from
  `ragged_dot` where the kernel leaves them unspecified). Output keeps
  the activation dtype, like the unfused intermediates did.
  """
  gate = mxfp4_gmm_reference(lhs, gate_packed, gate_scales, group_sizes)
  up = mxfp4_gmm_reference(lhs, up_packed, up_scales, group_sizes)
  return _situ_f32(
      gate, up, beta=situ_beta, linear_beta=situ_linear_beta
  ).astype(gate.dtype)


def can_use_kernel(
    lhs_rows: int,
    k: int,
    tiling: tuple[int, int, int] | None = None,
    rhs_pairs: int = 1,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_GHOSTLITE,
    dequant: str = DEQUANT_DEFAULT,
) -> bool:
  """Whether `mxfp4_gmm`'s tiling constraints hold for these shapes.

  Toy configs (a 12-wide latent, say) cannot tile a 32-element scale
  group, so the model falls back to the reference there; the real
  serving shapes (K = 3584 latent, 3072 ffn) all satisfy this.
  `rhs_pairs=2` sizes the default tiling for the fused gate+up kernel
  (same constraint set, doubled rhs working set); `vmem_budget` and
  `scoped_elem_bytes` as `default_tiling`. Planar and planar_t
  `dequant` modes additionally require the k chunk to equal the repack
  window (toy contractions fall back to the reference, which
  un-repacks first).
  """
  windowed = dequant in PLANAR_MODES or dequant in PLANAR_T_MODES
  tm, chunk, _ = tiling or default_tiling(
      lhs_rows,
      k,
      1,
      rhs_pairs=rhs_pairs,
      vmem_budget=vmem_budget,
      scoped_elem_bytes=scoped_elem_bytes,
      force_chunk=PLANAR_WINDOW if windowed else None,
      f8_planes=dequant in F8_PLANE_MODES,
  )
  tm, chunk = min(tm, lhs_rows), min(chunk, k)
  return (
      k % SCALE_GROUP == 0
      and chunk % SCALE_GROUP == 0
      and k % chunk == 0
      and lhs_rows % tm == 0
      and (tm % 8 == 0 or tm == lhs_rows)
      and (not windowed or chunk == PLANAR_WINDOW)
  )


def _quantize_lhs_f8(lhs: jax.Array) -> jax.Array:
  """The fp8-activation opt-in's cast: bf16 rows -> e4m3, unscaled.

  A plain rounding cast (RTNE), no per-token scale: the fp8-activation
  quality study measured e4m3's 2^-4 relative rounding to be
  scale-invariant (granularity sweeps were flat), so a scale buys
  nothing until per-token k32 scaling ships end to end; `lhs_f8_scales`
  IS that scaling (`quantize_mxfp8`, the QAT-matching MXFP8 grid).
  Values beyond e4m3's +-448 range saturate under XLA's TPU convert;
  SiTU-bounded activations (|gate arm| <= beta, |up arm| <=
  linear_beta) sit far inside it.
  """
  return lhs.astype(jnp.float8_e4m3fn)


# `lhs_f8_scales` spellings (`K3MoeConfig.gmm_fp8_act_scales`): the
# activation-scale companion of the fp8 x fp8 opt-in. 'none' is the
# unscaled RTNE cast above (today's arm, byte for byte). The MXFP8
# arms quantize each 32-wide contraction block of every row against a
# dynamic e8m0 (power-of-two) shared scale - the SAME block size and
# scale format the checkpoint's MXFP4 weights carry, and the numerics
# class Kimi K3's QAT ran forward from the SFT stage on ("MXFP4
# weights with MXFP8 activations", HF README; e4m3 elements +
# per-(1, 32) u8 e8m0 dynamic scales is also the vLLM/TRT-LLM
# kMxfp8Dynamic serve path on GPUs). Two scale roundings exist in the
# wild and both are implemented so quality evals can A/B them:
#   * 'e8m0' (RECOMMENDED): descale = 2^ceil(log2(amax / 448)), the
#     no-clip convention (the block max lands in (224, 448], nothing
#     saturates; the NVIDIA MXFP8 training-recipe rounding).
#   * 'e8m0_floor': descale = 2^floor(log2(amax)), the OCP-v1 /
#     vLLM-torch-reference convention (block max lands in [1, 2); no
#     clip either, but elements >= 2^9 below their block max fall
#     into e4m3's subnormal grid ~8 octaves sooner).
ACT_SCALES_NONE = 'none'
ACT_SCALES_E8M0 = 'e8m0'
ACT_SCALES_E8M0_FLOOR = 'e8m0_floor'
ACT_SCALE_MODES = (ACT_SCALES_NONE, ACT_SCALES_E8M0, ACT_SCALES_E8M0_FLOOR)

# 448 = 1.75 * 2^8: e4m3fn's largest finite value, as its exponent
# offset (8) and the f32 mantissa-field threshold for 1.75 (the ceil
# rounding's bump test works on raw exponent/mantissa bits, no libm).
_E4M3_MAX_EXP = 8
_F32_MANTISSA_1P75 = 0x600000


def quantize_mxfp8(
    x: jax.Array, *, rounding: str = ACT_SCALES_E8M0
) -> tuple[jax.Array, jax.Array]:
  """`[M, K]` rows -> MXFP8: e4m3 codes + per-32-block e8m0 scales.

  The activation twin of the checkpoint's MXFP4 weight format and the
  quantization Kimi K3's QAT trained forward with: each row's 32-wide
  contraction blocks share one dynamic power-of-two scale (u8 biased
  exponent), elements store float8_e4m3fn. Runs INSIDE the jitted
  step (dynamic activations), so everything is integer/exponent-bit
  arithmetic on the VPU: the scale comes from the f32 amax's exponent
  field (plus the mantissa bump for the ceil rounding), the descale
  multiply is by an exact power of two (NO rounding: the one and only
  lossy step is the RNE cast to e4m3), and the e8m0 -> f32 decode is
  the `_e8m0_to_f32` bit construction the weight scales already use.

  `rounding` picks the scale convention (see `ACT_SCALE_MODES`): both
  keep the block max inside e4m3's finite range by construction, so
  the cast never saturates.

  Args:
    x: `[M, K]` activations (any float dtype; K % SCALE_GROUP == 0).
    rounding: 'e8m0' (ceil, recommended) or 'e8m0_floor' (OCP-v1 / vLLM-torch
      parity).

  Returns:
    `(codes float8_e4m3fn [M, K], scales uint8 [M, K/SCALE_GROUP])`.
  """
  if rounding not in (ACT_SCALES_E8M0, ACT_SCALES_E8M0_FLOOR):
    raise ValueError(
        f'rounding must be one of {(ACT_SCALES_E8M0, ACT_SCALES_E8M0_FLOOR)},'
        f' got {rounding!r}.'
    )
  m, k = x.shape
  if k % SCALE_GROUP:
    raise ValueError(f'K {k} must be a multiple of {SCALE_GROUP}.')
  xf = x.astype(jnp.float32).reshape(m, k // SCALE_GROUP, SCALE_GROUP)
  amax = jnp.max(jnp.abs(xf), axis=-1)
  bits = lax.bitcast_convert_type(amax, jnp.uint32)  # sign bit is 0.
  exp_field = (bits >> 23).astype(jnp.int32)
  if rounding == ACT_SCALES_E8M0:
    # ceil(log2(amax / 448)): floor exponent - 8, bumped by one when
    # amax's mantissa exceeds 1.75 (448 = 1.75 * 2^8), so the scaled
    # block max lands in (224, 448].
    bump = (bits & jnp.uint32(0x7FFFFF)) > jnp.uint32(_F32_MANTISSA_1P75)
    scale_exp = exp_field - _E4M3_MAX_EXP + bump.astype(jnp.int32)
  else:
    # floor(log2(amax)): the scaled block max lands in [1, 2).
    scale_exp = exp_field
  # Clamp to e8m0's representable exponents (255 is the format's NaN;
  # 0 decodes as 2^-127). amax == 0 or subnormal has exp_field 0 and
  # lands at the bottom either way: codes come out 0 and the decode
  # side multiplies zeros, never a NaN.
  scales = jnp.clip(scale_exp, 0, 254).astype(jnp.uint8)
  # 2^(127 - s) by exponent-bit construction (the `_e8m0_to_f32`
  # spelling, inverted): a bit-exact power of two, NOT a divide - a
  # VPU reciprocal's last-ulp behavior would otherwise sit inside the
  # only multiply this quantizer performs. s == 254 inverts to the
  # subnormal 2^-127, whose pattern is not an exponent field.
  inv_bits = (jnp.uint32(254) - scales.astype(jnp.uint32)) << 23
  inv = jnp.where(
      scales == 254,
      jnp.float32(2.0**-127),
      lax.bitcast_convert_type(inv_bits, jnp.float32),
  )
  codes = (xf * inv[..., None]).reshape(m, k).astype(jnp.float8_e4m3fn)
  return codes, scales


def dequantize_mxfp8(codes: jax.Array, scales: jax.Array) -> jax.Array:
  """`quantize_mxfp8`'s inverse: the f32 values the kernel computes on.

  `code * 2^(scale - 127)` per 32-wide block, exact (power-of-two
  multiply of a 4-significand-bit value). The reference fallback runs
  the grouped matmul on these values, so a toy-shape model under the
  MXFP8 knob measures the real activation rounding, and the kernel's
  post-dot scale factoring is algebraically this dequantized dot.

  Args:
    codes: `[M, K]` float8_e4m3fn.
    scales: `[M, K/SCALE_GROUP]` uint8 e8m0 exponents.

  Returns:
    `[M, K]` float32.
  """
  m, k = codes.shape
  vals = codes.astype(jnp.float32).reshape(m, k // SCALE_GROUP, SCALE_GROUP)
  return (vals * _e8m0_to_f32(scales)[..., None]).reshape(m, k)


def seam_tiling(
    lhs_shape: tuple[int, ...],
    packed_shape: tuple[int, ...],
    *,
    rhs_pairs: int = 1,
    expected_rows: int | None = None,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_GHOSTLITE,
    dequant: str = DEQUANT_DEFAULT,
    lhs_scale_planes: bool = False,
) -> tuple[int, int, int]:
  """The `(tm, k_chunk, tn)` a local seam derives when `tiling` is None.

  One function for `local_grouped_matmul` (`rhs_pairs=1`),
  `local_fused_gate_up_matmul` (`rhs_pairs=2`) and the callers that
  want to know a seam's row tile BEFORE calling it (the build-once
  metadata path, `shared_group_metadata`), so the two can never drift.

  Args:
    lhs_shape: `(m, k)` of the activation rows.
    packed_shape: the packed stack's shape (`[E, N, K/2]`, or the planar_t `[E,
      K/2, N]` order the `dequant` mode declares).
    rhs_pairs: `default_tiling` semantics.
    expected_rows: `default_tiling` semantics.
    vmem_budget: `default_tiling` semantics.
    scoped_elem_bytes: `default_tiling` semantics.
    dequant: the inner-loop mode; planar modes pin the k chunk to
      `PLANAR_WINDOW` and planar_t modes read N off the transposed stack.
    lhs_scale_planes: `default_tiling` semantics (the MXFP8 activation arm).

  Returns:
    `(tm, k_chunk, tn)`.
  """
  m, k = lhs_shape[0], lhs_shape[1]
  n = packed_shape[2] if dequant in PLANAR_T_MODES else packed_shape[1]
  return default_tiling(
      m,
      k,
      n,
      rhs_pairs=rhs_pairs,
      expected_rows=expected_rows,
      vmem_budget=vmem_budget,
      scoped_elem_bytes=scoped_elem_bytes,
      force_chunk=PLANAR_WINDOW
      if (dequant in PLANAR_MODES or dequant in PLANAR_T_MODES)
      else None,
      lhs_scale_planes=lhs_scale_planes,
      f8_planes=dequant in F8_PLANE_MODES,
  )


def shared_group_metadata(
    group_sizes: jax.Array,
    *,
    m: int,
    num_groups: int,
    tilings: Sequence[tuple[int, int, int]],
    impl: str,
) -> GroupMetadata | None:
  """ONE grid-metadata build for every kernel call of an expert visit.

  The gate+up and the down grouped matmuls of one layer read the same
  `group_sizes` and, at every served geometry, the same row tile, so
  their metadata is the same array triple. Built inside each kernel it
  is traced twice per layer; XLA's CSE merges the loop-free parts of
  that pair but never a `while` (the two histogram searches of the
  repeat/searchsorted spellings stay duplicated in the rung-4 program).
  Under a loop-free spelling the caller builds it here once and hands
  it to both kernels (`group_metadata=`); the kernels check the triple
  against their own `(m, tm, num_groups)` by shape, so a tiling
  mismatch is a loud wiring error, never a silent misread.

  Args:
    group_sizes: `[num_groups]` int32 rows per group.
    m: lhs rows.
    num_groups: groups in the stacks.
    tilings: the `(tm, k_chunk, tn)` of every kernel that will consume the build
      (`seam_tiling`, or the explicit tiling passed to the seam).
    impl: `_make_group_metadata` spelling; only `METADATA_IMPLS_LOOP_FREE`
      spellings are built here.

  Returns:
    The metadata when every tiling shares the (m-clamped) row tile and
    `impl` is loop free, else None: each kernel then builds its own,
    exactly as before this path existed.
  """
  if impl not in METADATA_IMPLS_LOOP_FREE:
    return None
  tms = {min(tiling[0], m) for tiling in tilings}
  if len(tms) != 1:
    return None
  (tm,) = tms
  return _make_group_metadata(
      group_sizes, m=m, tm=tm, num_groups=num_groups, impl=impl
  )


def local_grouped_matmul(
    lhs: jax.Array,
    packed: jax.Array,
    scales: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, int, int] | None = None,
    expected_rows: int | None = None,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_GHOSTLITE,
    native_f8: bool = False,
    lhs_f8: bool = False,
    lhs_f8_scales: str = ACT_SCALES_NONE,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    dequant: str = DEQUANT_DEFAULT,
    single_group: tuple[jax.Array, jax.Array] | None = None,
    metadata_impl: str = 'repeat',
    paired_expert_visits: bool = False,
    lhs_hbm: bool = False,
    out_hbm: bool = False,
    deep_vmem_budget: int | None = None,
    deep_continuous: bool = False,
    group_metadata: GroupMetadata | None = None,
    deep_stream_rows: bool = False,
) -> jax.Array:
  """The seam: fused-dequant kernel on TPU, dense reference elsewhere.

  Both sides compute the same arithmetic on the same weights, so this
  choice is a performance and memory decision, not a numerical one
  (`moe_gmm_test` pins the equality, `ffw_test` and `sharding_test` pin
  the model built on it). The reference also covers shapes the kernel's
  tiling cannot serve, which is what toy configs hit; for planar
  `dequant` modes it un-repacks the bytes first
  (`unpack_planar_stack`, bitwise), so the fallback computes on
  EXACTLY the interleaved model's weights.

  Arguments and returns as `mxfp4_gmm`, plus `expected_rows`: the rows
  this caller expects to own (`default_tiling`'s sparse-EP hint; tm
  and tn never change results, pinned bitwise in moe_gmm_test),
  `vmem_budget` (the per-grid-step bound; None = the GhostLite
  constant, `K3MoeConfig.gmm_vmem_budget_bytes` threads the chip
  value), `native_f8` (the mxfp4-native path,
  `K3MoeConfig.gmm_fp8_native`; the reference fallback ignores it - it
  computes the same values) and `lhs_f8` (the fp8 x fp8 opt-in,
  `K3MoeConfig.gmm_fp8_activations`, requires `native_f8`): the rows
  quantize to e4m3 HERE, once, so the kernel's lhs HBM traffic halves
  and the MXU pass runs 8-bit on both sides; the output keeps the
  incoming activation dtype. `lhs_f8_scales` upgrades that cast to the
  MXFP8 grid (`ACT_SCALE_MODES`; requires `lhs_f8`): the rows quantize
  through `quantize_mxfp8` (dynamic per-32-block e8m0 scales, the QAT
  numerics class) and the kernel folds the scales per subchannel next
  to the weight e8m0s. The reference fallback round-trips the rows
  through the SAME grid instead (`quantize -> widen`), so a toy-shape
  model under either knob still measures the real activation rounding
  (what the 0.40B end-to-end quality gate runs on CPU). `dequant`
  selects the inner-loop variant AND, for planar modes, declares that
  `packed` is `repack_planar_stack`ed; `single_group` is the one-hot
  metadata declaration (`mxfp4_gmm`), which the reference fallback
  ignores (its `ragged_dot` needs no grid metadata and `group_sizes`
  carries the same information). `paired_expert_visits` merges
  same-tile visit pairs into dual-stream grid steps (`mxfp4_gmm`;
  bitwise-pinned in moe_gmm_test); the reference fallback ignores it
  too - pairing is schedule-only, values identical. `group_metadata` is
  the precomputed build-once grid metadata (`shared_group_metadata`),
  forwarded to the kernel and ignored by the reference fallback (the
  kernel derives its tiling exactly as `seam_tiling` does, which is
  what the caller built the metadata over).
  """
  if lhs_f8 and not native_f8 and dequant not in _E4M3FN_WEIGHT_MODES:
    raise ValueError(
        'lhs_f8 (fp8 activations) requires native_f8=True or'
        f" dequant='{DEQUANT_PLANAR_T_F8FN}' (rolled or deep): both are"
        ' ways to put an f8 WEIGHT on the MXU, and without one the f8'
        ' lhs is widened back to bf16 (rounding loss, no rate win).'
    )
  if lhs_f8_scales not in ACT_SCALE_MODES:
    raise ValueError(
        f'lhs_f8_scales must be one of {ACT_SCALE_MODES}, got'
        f' {lhs_f8_scales!r}.'
    )
  if lhs_f8_scales != ACT_SCALES_NONE and not lhs_f8:
    raise ValueError('lhs_f8_scales (MXFP8 activations) requires lhs_f8=True.')
  use_mxfp8 = lhs_f8 and lhs_f8_scales != ACT_SCALES_NONE
  if (
      dequant in PLANAR_T_MODES
      and dequant not in F8_PLANE_MODES
      and scales.ndim == 4
  ):
    # Chunk-major scale params: normalize once here so the kernel and
    # the dense reference fallback below both see the k-major 3-D form
    # (see mxfp4_gmm / restage_scales_chunk_major).
    scales = flatten_chunk_major_scales(scales)
  if tiling is None:
    tiling = seam_tiling(
        lhs.shape,
        packed.shape,
        expected_rows=expected_rows,
        vmem_budget=vmem_budget,
        scoped_elem_bytes=scoped_elem_bytes,
        dequant=dequant,
        lhs_scale_planes=use_mxfp8,
    )
  if not can_use_kernel(lhs.shape[0], lhs.shape[1], tiling, dequant=dequant):
    if use_mxfp8:
      codes, act_scales = quantize_mxfp8(lhs, rounding=lhs_f8_scales)
      lhs = dequantize_mxfp8(codes, act_scales).astype(lhs.dtype)
    elif lhs_f8:
      lhs = _quantize_lhs_f8(lhs).astype(lhs.dtype)
    if lhs.dtype == jnp.float8_e4m3fn and preferred_element_type is not None:
      # Pre-quantized handoff (the fused gate_up's f8_out): the
      # reference computes in the ACTIVATION dtype like every other
      # fallback - quantize-then-WIDEN, so the weights stay exact and
      # only the lhs carries the f8 rounding, matching the kernel arm's
      # value class.
      lhs = lhs.astype(preferred_element_type)
    if dequant in F8_PLANE_MODES:
      # Load-time planes are bytes of an f8 format, not nibble pairs:
      # the un-repack above would read each byte as two e2m1 codes.
      return f8_plane_gmm_reference(
          lhs,
          packed,
          scales,
          group_sizes,
          plane_dtype=F8_PLANE_DTYPES[dequant],
          preferred_element_type=preferred_element_type,
      )
    if dequant in PLANAR_MODES:
      packed = unpack_planar_stack(packed)
    if dequant in PLANAR_T_MODES:
      packed = unpack_planar_t_stack(packed)
      scales = transpose_scales_planar_t(scales)
    return mxfp4_gmm_reference(
        lhs,
        packed,
        scales,
        group_sizes,
        preferred_element_type=preferred_element_type,
    )
  # None keeps lhs.dtype - today's contract. An EXPLICIT dtype exists
  # for the pre-quantized handoff: when the caller feeds an f8 hidden
  # (the fused gate_up's f8_out), lhs.dtype would silently make the
  # DOWN output f8 too, and the combine wants the activation dtype.
  out_dtype = (
      lhs.dtype
      if preferred_element_type is None
      else jnp.dtype(preferred_element_type)
  )
  act_scales = None
  if use_mxfp8:
    lhs, act_scales = quantize_mxfp8(lhs, rounding=lhs_f8_scales)
  elif lhs_f8:
    lhs = _quantize_lhs_f8(lhs)
  return mxfp4_gmm(
      lhs,
      packed,
      scales,
      group_sizes,
      preferred_element_type=out_dtype,
      tiling=tiling,
      # Interpret off TPU so CPU tests run the same dispatch as chips.
      interpret=jax.default_backend() != 'tpu',
      native_f8=native_f8,
      dequant=dequant,
      single_group=single_group,
      metadata_impl=metadata_impl,
      paired_expert_visits=paired_expert_visits,
      lhs_scales=act_scales,
      lhs_hbm=lhs_hbm and jax.default_backend() == 'tpu',
      out_hbm=out_hbm and jax.default_backend() == 'tpu',
      deep_vmem_budget=deep_vmem_budget,
      deep_continuous=deep_continuous,
      group_metadata=group_metadata,
      deep_stream_rows=deep_stream_rows,
  )


def local_fused_gate_up_matmul(
    lhs: jax.Array,
    gate_packed: jax.Array,
    gate_scales: jax.Array,
    up_packed: jax.Array,
    up_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    tiling: tuple[int, int, int] | None = None,
    expected_rows: int | None = None,
    vmem_budget: int | None = None,
    scoped_elem_bytes: int = SCOPED_ELEM_BYTES_GHOSTLITE,
    native_f8: bool = False,
    lhs_f8: bool = False,
    lhs_f8_scales: str = ACT_SCALES_NONE,
    f8_out: bool = False,
    dequant: str = DEQUANT_DEFAULT,
    single_group: tuple[jax.Array, jax.Array] | None = None,
    metadata_impl: str = 'repeat',
    lhs_hbm: bool = False,
    out_hbm: bool = False,
    deep_vmem_budget: int | None = None,
    deep_continuous: bool = False,
    group_metadata: GroupMetadata | None = None,
    deep_stream_rows: bool = False,
) -> jax.Array:
  """The gate+up seam: fused SiTU kernel on TPU, composed reference off it.

  The fused counterpart of `local_grouped_matmul` for the w1/w3 pair:
  one call returns the SiTU-activated product, so the separate
  `[M, ffn]` gate and up activations never materialize and the w1/w3
  side pays one kernel launch instead of two. Same fallback rule as
  the unfused seam (the kernel's tiling constraints), same
  unspecified-rows contract (the caller masks); the reference side
  composes the unfused references, so shapes the kernel cannot serve
  compute exactly what the unfused model computed.

  Arguments as `mxfp4_fused_gate_up_gmm`, plus `expected_rows` (the
  sparse-EP row hint, see `default_tiling`) and the
  `native_f8`/`lhs_f8`/`lhs_f8_scales` trio (`local_grouped_matmul`
  semantics: the fp8-activation cast or MXFP8 quantization happens
  here, once, feeding BOTH contractions, and the SiTU epilogue still
  rounds its accumulators through the ORIGINAL activation dtype); the
  output keeps the activation dtype. `dequant` and `single_group` as
  `local_grouped_matmul` (planar modes declare both packed stacks
  planar-repacked; the reference fallback un-repacks them first,
  bitwise). `lhs_hbm` / `out_hbm` as `mxfp4_fused_gate_up_gmm` (the
  HBM operand pre-color and the required-HBM output birth; the
  reference fallback has no memory spaces and ignores both - values
  are identical either way, only the kernel path's address spaces
  change). `group_metadata` as `local_grouped_matmul` (the build-once
  grid metadata; kernel-only, values identical).
  """
  if lhs_f8 and not native_f8 and dequant not in _E4M3FN_WEIGHT_MODES:
    raise ValueError(
        'lhs_f8 (fp8 activations) requires native_f8=True or'
        f" dequant='{DEQUANT_PLANAR_T_F8FN}' (rolled or deep): both are"
        ' ways to put an f8 WEIGHT on the MXU, and without one the f8'
        ' lhs is widened back to bf16 (rounding loss, no rate win).'
    )
  if lhs_f8_scales not in ACT_SCALE_MODES:
    raise ValueError(
        f'lhs_f8_scales must be one of {ACT_SCALE_MODES}, got'
        f' {lhs_f8_scales!r}.'
    )
  if lhs_f8_scales != ACT_SCALES_NONE and not lhs_f8:
    raise ValueError('lhs_f8_scales (MXFP8 activations) requires lhs_f8=True.')
  if f8_out and not lhs_f8:
    raise ValueError(
        'f8_out narrows the SiTU hidden handoff to the down gmm; outside'
        ' the fp8-activation regime (lhs_f8) that would silently change'
        " the down gmm's lhs numerics class in a bf16 program."
    )
  if f8_out and lhs_f8_scales != ACT_SCALES_NONE:
    raise ValueError(
        'f8_out does not compose with MXFP8 activation scales: the'
        ' MXFP8 arm re-quantizes the SiTU hidden on the e8m0 grid at'
        ' the down seam (quantize_mxfp8 of the bf16 hidden), and an f8'
        ' hidden would feed that quantizer the wrong dtype. The plain'
        ' f8 cast is the only hidden this store serves.'
    )
  use_mxfp8 = lhs_f8 and lhs_f8_scales != ACT_SCALES_NONE
  if dequant in PLANAR_T_MODES and dequant not in F8_PLANE_MODES:
    # Chunk-major scale params: normalize once here so the kernel and
    # the reference fallback below both see the k-major 3-D form.
    if gate_scales.ndim == 4:
      gate_scales = flatten_chunk_major_scales(gate_scales)
    if up_scales.ndim == 4:
      up_scales = flatten_chunk_major_scales(up_scales)
  if tiling is None:
    tiling = seam_tiling(
        lhs.shape,
        gate_packed.shape,
        rhs_pairs=2,
        expected_rows=expected_rows,
        vmem_budget=vmem_budget,
        scoped_elem_bytes=scoped_elem_bytes,
        dequant=dequant,
        lhs_scale_planes=use_mxfp8,
    )
  if not can_use_kernel(
      lhs.shape[0], lhs.shape[1], tiling, rhs_pairs=2, dequant=dequant
  ):
    if use_mxfp8:
      codes, act_scales = quantize_mxfp8(lhs, rounding=lhs_f8_scales)
      lhs = dequantize_mxfp8(codes, act_scales).astype(lhs.dtype)
    elif lhs_f8:
      lhs = _quantize_lhs_f8(lhs).astype(lhs.dtype)
    if dequant in F8_PLANE_MODES:
      plane_dtype = F8_PLANE_DTYPES[dequant]
      gate = f8_plane_gmm_reference(
          lhs, gate_packed, gate_scales, group_sizes, plane_dtype=plane_dtype
      )
      up = f8_plane_gmm_reference(
          lhs, up_packed, up_scales, group_sizes, plane_dtype=plane_dtype
      )
      ref = _situ_f32(
          gate, up, beta=situ_beta, linear_beta=situ_linear_beta
      ).astype(gate.dtype)
      return ref.astype(jnp.float8_e4m3fn) if f8_out else ref
    if dequant in PLANAR_MODES:
      gate_packed = unpack_planar_stack(gate_packed)
      up_packed = unpack_planar_stack(up_packed)
    if dequant in PLANAR_T_MODES:
      gate_packed = unpack_planar_t_stack(gate_packed)
      gate_scales = transpose_scales_planar_t(gate_scales)
      up_packed = unpack_planar_t_stack(up_packed)
      up_scales = transpose_scales_planar_t(up_scales)
    ref = mxfp4_fused_gate_up_reference(
        lhs,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        group_sizes,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    # Same contract as the kernel arm: the f8 store is the ONE extra
    # rounding after the activation-dtype rounding, bit for bit.
    return ref.astype(jnp.float8_e4m3fn) if f8_out else ref
  out_dtype = lhs.dtype
  act_scales = None
  if use_mxfp8:
    lhs, act_scales = quantize_mxfp8(lhs, rounding=lhs_f8_scales)
  elif lhs_f8:
    lhs = _quantize_lhs_f8(lhs)
  return mxfp4_fused_gate_up_gmm(
      lhs,
      gate_packed,
      gate_scales,
      up_packed,
      up_scales,
      group_sizes,
      situ_beta=situ_beta,
      situ_linear_beta=situ_linear_beta,
      preferred_element_type=out_dtype,
      out_dtype=jnp.float8_e4m3fn if f8_out else None,
      tiling=tiling,
      # Interpret off TPU so CPU tests run the same dispatch as chips.
      interpret=jax.default_backend() != 'tpu',
      native_f8=native_f8,
      dequant=dequant,
      single_group=single_group,
      metadata_impl=metadata_impl,
      lhs_scales=act_scales,
      lhs_hbm=lhs_hbm,
      out_hbm=out_hbm and jax.default_backend() == 'tpu',
      deep_vmem_budget=deep_vmem_budget,
      deep_continuous=deep_continuous,
      group_metadata=group_metadata,
      deep_stream_rows=deep_stream_rows,
  )


# Weight-plane memory placement (`K3MoeConfig.gmm_weight_plane_memory`).
# 'auto' leaves the packed expert stacks where the compiler put them
# (HBM params; XLA MSA MAY cross-prefetch them to VMEM, and does for
# the plain width-1 decode program). 'pin_vmem' forces the gate and
# down planes (+ their scales) into VMEM-resident buffers at the top
# of each kiloblox layer body, restoring the plain program's proven
# placement in programs whose live set crowds MSA out of it (the spec
# width-T round: 2026-08-06, gmm events 5x slower on @hbm planes).
WEIGHT_PLANES_AUTO = 'auto'
WEIGHT_PLANES_PIN_VMEM = 'pin_vmem'
WEIGHT_PLANE_MEMORY_MODES = (WEIGHT_PLANES_AUTO, WEIGHT_PLANES_PIN_VMEM)


def _pin_vmem_kernel(in_ref, out_ref):
  out_ref[...] = in_ref[...]


def pin_plane_to_vmem(x: jax.Array) -> jax.Array:
  """Copies one array into a VMEM-resident XLA buffer (memory space S(1)).

  The `kiloblox_pin_chunks` mechanism (kiloblox_a2a.reshape_slice's
  `memory_space=pltpu.VMEM` out_shape, prod v2/decode_routed_layer.py's
  shipped flavor) applied to a whole expert stack: a Pallas copy whose
  OUTPUT buffer is annotated VMEM, so buffer assignment allocates it in
  memory space 1 and every downstream kernel's tile DMAs source from
  VMEM instead of HBM. The grid walks the leading (expert) dim so the
  copy's own staging windows stay one-expert sized (~5.25 MiB per plane
  at the serving shapes, far under the scoped-VMEM ceiling) while the
  RESULT is the whole stack, resident.

  There is deliberately NO size guard here: if the pinned bytes do not
  fit VMEM alongside the program's other residents, the XLA compile
  fails loudly (MSA/buffer-assignment OOM), which is the contract -
  never a silent fallback to HBM. The caller decides WHICH stacks fit
  (the serving policy pins the gate + down PLANES and streams up,
  copying the plain program's MSA-proven placement: 2 x 36.75 MiB
  against the 96 MiB MSA pool on GhostLite).

  Bytes only move, values never change: the kernel is an identity copy
  and the consumer kernels read the same bytes from a faster space
  (moe_kiloblox_forge_test pins pinned == unpinned BITWISE on real
  chips).
  """
  if x.ndim < 2:
    raise ValueError(
        f'pin_vmem expects a stacked [E, ...] array, got rank {x.ndim}.'
    )
  spec = pl.BlockSpec(
      (None,) + x.shape[1:], lambda i: (i,) + (0,) * (x.ndim - 1)
  )
  return pl.pallas_call(
      _pin_vmem_kernel,
      out_shape=pltpu.VMEM(x.shape, x.dtype),
      grid=(x.shape[0],),
      in_specs=[spec],
      out_specs=spec,
      # Interpret off TPU so CPU tests run the same dispatch as chips
      # (interpret mode has no memory spaces; the copy is still a copy).
      interpret=jax.default_backend() != 'tpu',
  )(x)


def pin_stack_to_vmem(
    stack: jax.Array | tuple[jax.Array, jax.Array],
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Pins an expert stack (dense array or MXFP4 pair) to VMEM, whole.

  MXFP4 `(packed, scales)` pairs pin both members; dense stacks pin
  the one array. Returns the same structure with every leaf
  VMEM-resident; values are bitwise unchanged. NOTE the serving
  policy (moe_kiloblox._pin_weight_planes) pins PLANES ONLY via
  `pin_plane_to_vmem` - the xid_277694124 postmortem showed scale
  pinning is redundant (MSA keeps scales resident even in the demoted
  spec program) and just lengthens the serial pin chain; this
  whole-stack form remains for tests and tooling.
  """
  if isinstance(stack, tuple):
    packed, scales = stack
    return (pin_plane_to_vmem(packed), pin_plane_to_vmem(scales))
  return pin_plane_to_vmem(stack)


# ============================================================================
# vLLM-torchtpu TPU v6e (Trillium) MXFP4 -> b11 Integration Entrypoints
# ============================================================================

# TPU v6e has 32 MiB VMEM per TensorCore; reserve half (16 MiB) for the
# rolled kernel's per-step budget and 24 MiB for the deep-stream pipeline.
V6E_ROLLED_VMEM_BUDGET = 16 * 1024 * 1024
V6E_DEEP_VMEM_BUDGET = 24 * 1024 * 1024


def repack_w13_planar_t(
    w13_packed: jax.Array, w13_scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Repacks checkpoint `w13` (`[E, 2*N, K/2]` u8) and `w13_scale` (`[E, 2*N, K/32]` u8)
  into a single zero-copy 3D/4D `planar_t` stack pair:
    - `w13_pt`: `[E, K, N]` uint8 (equivalent to `[E, 2, K/2, N]` flattened on axes 1..2)
    - `w13_sc`: `[E, 2*(K/32), 1, N]` uint8 (equivalent to `[E, 2, K/32, 1, N]`)
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
  """Repacks checkpoint `w2` (`[E, L, N/2]` u8) and `w2_scale` (`[E, L, N/32]` u8)
  into `planar_t` layout:
    - `w2_pt`: `[E, N/2, L]` uint8
    - `w2_sc`: `[E, N/32, 1, L]` uint8
  """
  w2_pt = repack_planar_t_stack(w2_packed)
  w2_sc = jnp.expand_dims(transpose_scales_planar_t(w2_scale), axis=2)
  return w2_pt, w2_sc


def parse_situ_activation(activation: str) -> tuple[float, float | None]:
  """Parses vLLM activation string ('situ:4.0:25.0' or 'silu') into (beta, linear_beta)."""
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
    group_sizes: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    tiling: tuple[int, int, int],
    dequant: str,
    group_metadata: GroupMetadata,
    deep_vmem_budget: int = V6E_DEEP_VMEM_BUDGET,
) -> jax.Array:
  """Runs fused gate+up GMM directly on combined `w13_pt` `[E, K, N]` (`[E, 2, steps, chunk/2, N]`)
  and `w13_sc_3d` `[E, 2*(K/32), N]` (`[E, 2, steps, chunk/32, N]`) with zero HBM copy.
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
        m=m,
        k=k,
        n=n,
        tm=tm,
        chunk=chunk,
        tn=tn,
        num_stacks=2,
        budget=deep_vmem_budget,
        lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
        out_bytes=jnp.dtype(preferred_element_type).itemsize,
        stream_rows=True,
        f8_planes=False,
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
          out_hbm=False,
          body=deep_body,
          continuous=True,
          stream_rows=True,
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
    ep_shards: int = 8,
    decode_max_m: int = 2048,
) -> jax.Array:
  """Executes fused `w13` (gate+up+SiTU) + `w2` (down) grouped matmul on TPU v6e
  using `planar_t` MXFP4 weights unpacked in VMEM to `float8_e4m3b11fnuz` (`b11`).

  Args:
    lhs: `[M, K]` bfloat16 sorted routed activations (`K = 3584`).
    w13_pt: `[E, K, N]` uint8 (`[7, 3584, 3072]`) from `repack_w13_planar_t`.
    w13_scale_pt: `[E, 2*(K/32), 1, N]` or `[E, 2*(K/32), N]` uint8.
    w2_pt: `[E, N/2, K]` uint8 (`[7, 1536, 3584]`) from `repack_w2_planar_t`.
    w2_scale_pt: `[E, N/32, 1, K]` or `[E, N/32, K]` uint8.
    group_sizes: `[E]` int32 local expert token counts.
    activation: vLLM activation string (e.g. `'situ:4.0:25.0'` or `'silu'`).
    preferred_element_type: Output dtype (defaults to `lhs.dtype`).
    ep_shards: Number of EP shards (used to compute `expected_rows = ceil(M / ep_shards)`).
    decode_max_m: Maximum routed rows `M` to use the deep-stream pipeline
      (`planar_t_b11_deep`); larger prefill trays use the rolled `planar_t_b11`
      pipeline (bitwise identical).

  Returns:
    `[M, K]` output activations in `preferred_element_type`.
  """
  orig_m, k = lhs.shape
  num_groups, _, n = w13_pt.shape
  out_dtype = lhs.dtype if preferred_element_type is None else jnp.dtype(preferred_element_type)

  if w13_scale_pt.ndim == 4:
    w13_sc_3d = jnp.squeeze(w13_scale_pt, axis=2)
  else:
    w13_sc_3d = w13_scale_pt
  if w2_scale_pt.ndim == 4:
    w2_sc_3d = jnp.squeeze(w2_scale_pt, axis=2)
  else:
    w2_sc_3d = w2_scale_pt

  situ_beta, situ_linear_beta = parse_situ_activation(activation)

  # Pad M to a multiple of 32 (or 128 for large prefill) so tm always divides m cleanly.
  align_m = 32 if orig_m <= decode_max_m else 128
  padded_m = ((orig_m + align_m - 1) // align_m) * align_m
  if padded_m != orig_m:
    lhs = jnp.pad(lhs, ((0, padded_m - orig_m), (0, 0)))

  expected_rows = max(32, -(-padded_m // max(1, ep_shards)))
  dequant = DEQUANT_PLANAR_T_B11_DEEP if padded_m <= decode_max_m else DEQUANT_PLANAR_T_B11

  tiling_13 = default_tiling(
      padded_m,
      k,
      n,
      lhs_bytes=jnp.dtype(lhs.dtype).itemsize,
      rhs_pairs=2,
      expected_rows=expected_rows,
      vmem_budget=V6E_ROLLED_VMEM_BUDGET,
      scoped_elem_bytes=SCOPED_ELEM_BYTES_GHOSTFISH,
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
      scoped_elem_bytes=SCOPED_ELEM_BYTES_GHOSTFISH,
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
      group_sizes,
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
      metadata_impl="closed_form",
      deep_vmem_budget=V6E_DEEP_VMEM_BUDGET,
      deep_continuous=True,
      group_metadata=group_metadata,
      deep_stream_rows=True,
  )
  return out[:orig_m]


FUSED_DECODE_MOE_MAX_TOKENS = 64
# Phase O6b (`TPU_KIMI_OPT_SPEC_KDA=o6b`): speculative verify batches of
# 65..256 tokens run the same kernel with a forced multi-tray row loop (the
# b11 `fold_chunk` weight unpack stays hoisted outside the tray loop, so the
# weight-side VPU cost is paid once per chunk regardless of M). Callers opt in
# by passing `max_tokens=FUSED_VERIFY_MOE_MAX_TOKENS` to the eligibility check.
FUSED_VERIFY_MOE_MAX_TOKENS = 256
# Measured on v6e (K3 routed expert 7x3584x3072, 0 VREG spills for 32/64/128/
# 256-row trays): 128-row trays are within 4% of the best tray size at every
# M in 65..256 (M=96/128: 165/167 us vs 175/176 at 64 rows and 179 at 32;
# M=192: 203 vs 196 at 64; M=256: 204 vs 215 at 64, 325 at 32, 573 at 16).
FUSED_VERIFY_MOE_TRAY_M = 128
_FUSED_VERIFY_MOE_VMEM_BUDGET = 46 * 1024 * 1024


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
  """Returns whether `fused_decode_moe_v6e_planar_t_b11` is eligible.

  `max_tokens` above `FUSED_DECODE_MOE_MAX_TOKENS` (O6b verify, at most
  `FUSED_VERIFY_MOE_MAX_TOKENS`) admits the multi-tray variant, subject to a
  VMEM estimate at the padded row count.
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
  slab_bytes = (
      2 * (k + 2 * (k // SCALE_GROUP)) * n + (n // 2 + n // SCALE_GROUP) * k
  )
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
    vmem_limit_bytes: int | None = 48 * 1024 * 1024,
    token_trays_mode: str = "off",
    rank3_scales: bool = False,
    verify_tray_m: int | None = None,
) -> jax.Array:
  """Single-kernel fused decode MoE on TPU v6e for small token batches (M <= 64).

  Eliminates SparseCore `ragged_gather_v2`, `ragged_gather_reduce_v2`, token
  sorting, and intermediate `[M*topk, N]` HBM round-trips by streaming whole
  contiguous expert slabs (`w13[g]` double-buffered, `w2[g]` streamed during
  `w13[g]` compute) into VMEM and computing `w13 + SiTU + w2 + weighted_combine`
  in one Pallas kernel.

  For O6b speculative verify (`64 < M <= FUSED_VERIFY_MOE_MAX_TOKENS`) the row
  loop is always multi-tray (`verify_tray_m` rows per tray, default
  `FUSED_VERIFY_MOE_TRAY_M`) and `token_trays_mode` is ignored; `M <= 64` is
  unaffected by `verify_tray_m`.
  """
  orig_m, k = hidden_states.shape
  num_groups, _, n = w13_pt.shape
  out_dtype = (
      hidden_states.dtype
      if preferred_element_type is None
      else jnp.dtype(preferred_element_type)
  )

  if w13_scale_pt.ndim == 4:
    w13_sc_3d = jnp.squeeze(w13_scale_pt, axis=2)
  else:
    w13_sc_3d = w13_scale_pt
  if w2_scale_pt.ndim == 4:
    w2_sc_3d = jnp.squeeze(w2_scale_pt, axis=2)
  else:
    w2_sc_3d = w2_scale_pt

  situ_beta, situ_linear_beta = parse_situ_activation(activation)

  trays_norm = (token_trays_mode or "off").strip().lower()
  if orig_m > FUSED_DECODE_MOE_MAX_TOKENS:
    if orig_m > FUSED_VERIFY_MOE_MAX_TOKENS:
      raise ValueError(
          f"fused_decode_moe_v6e_planar_t_b11 supports at most "
          f"{FUSED_VERIFY_MOE_MAX_TOKENS} tokens, got {orig_m}."
      )
    tray_m = int(
        FUSED_VERIFY_MOE_TRAY_M if verify_tray_m is None else verify_tray_m
    )
    if tray_m <= 0 or tray_m % 16 != 0:
      raise ValueError(f"verify_tray_m must be a positive multiple of 16, got {tray_m}.")
    use_trays = True
    padded_m = ((orig_m + tray_m - 1) // tray_m) * tray_m
    num_trays = padded_m // tray_m
  elif trays_norm in ("", "0", "off", "false", "none"):
    use_trays = False
    padded_m = max(16, ((orig_m + 7) // 8) * 8)
    tray_m = padded_m
    num_trays = 1
  elif trays_norm in ("trays", "trays4"):
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
        "'trays'/'trays4', 'trays2', or 'trays32'."
    )
  if padded_m != orig_m:
    lhs = jnp.pad(hidden_states, ((0, padded_m - orig_m), (0, 0)))
    topk_w = jnp.pad(topk_weights, ((0, padded_m - orig_m), (0, 0)))
    topk_i = jnp.pad(
        topk_ids, ((0, padded_m - orig_m), (0, 0)), constant_values=-1
    )
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
            # Hoist b11 weight unpack outside token_trays loop (1 unpack/chunk):
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
            # Hoist w2 b11 weight unpack outside token_trays loop (1 unpack/chunk):
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
            down_bf16 = w2_acc[r_sl, :].astype(out_dtype).astype(jnp.float32)
            w_g_tr = jnp.tile(w_expert_ref[g, r_sl, :], (1, tn2 // _LANES))
            out_acc[r_sl, sl2] += down_bf16 * w_g_tr
      else:
        w_g = w_expert_ref[g]
        for t in range(tiles_n13):
          sl = pl.ds(t * tn13, tn13)
          gate_acc[...] = jnp.zeros_like(gate_acc)
          up_acc[...] = jnp.zeros_like(up_acc)
          for cs in range(steps_13):
            lhs_chunk = lhs_tile[:, cs * chunk : (cs + 1) * chunk]
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

  compiler_params = (
      pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes)
      if vmem_limit_bytes is not None
      else pltpu.CompilerParams()
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

