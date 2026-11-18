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
"""CPU tests for the planar_t MXFP4 MoE kernels (Pallas interpret mode)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v6e_planar_t_b11 as gmm

E = 3
K = 512
INTER = 512
ACT = "situ:4.0:25.0"
BETA, LINEAR_BETA = 4.0, 25.0


def _ckpt_stack(rng, e, n, k, lo=124, hi=131):
    """Checkpoint-layout MXFP4 stack: `[e, n, k/2]` nibbles, `[e, n, k/32]` e8m0."""
    packed = rng.integers(0, 256, (e, n, k // 2), dtype=np.uint8)
    scales = rng.integers(lo, hi, (e, n, k // 32), dtype=np.uint8)
    return packed, scales


# e2m1 decode table, index = sign<<3 | exp<<1 | mantissa.
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_LUT += tuple(-v for v in _E2M1_LUT)


def _dequant(packed, scales):
    """Dense f32 `[..., n, k]` weights of a checkpoint-layout MXFP4 stack."""
    lut = np.asarray(_E2M1_LUT, np.float32)
    codes = np.stack([packed & 0x0F, packed >> 4], axis=-1)
    values = lut[codes.reshape(*packed.shape[:-1], -1)]
    exps = np.exp2(scales.astype(np.float32) - 127.0)
    k = values.shape[-1]
    values = values.reshape(*values.shape[:-1], k // 32, 32) * exps[..., None]
    return values.reshape(*packed.shape[:-1], k)


def _unpack_planar_t_stack(packed, window=gmm.PLANAR_WINDOW):
    """The inverse of `repack_planar_t_stack` (back to interleaved pairs)."""
    planar = np.swapaxes(np.asarray(packed), -1, -2)
    lead, half_k = planar.shape[:-1], planar.shape[-1]
    planar = planar.reshape(lead + (2 * half_k // window, window // 2))
    # Original contraction order per window, then re-pair (even, odd).
    codes = np.concatenate([planar & 0x0F, planar >> 4], axis=-1)
    even, odd = codes[..., 0::2], codes[..., 1::2]
    return (even | (odd << 4)).astype(np.uint8).reshape(lead + (half_k,))


def _scale_spread(scales, chunk=gmm.PLANAR_WINDOW):
    """Widest e8m0 octave spread inside one window of `[..., N, K/32]` scales."""
    arr = np.asarray(scales).astype(np.int32)
    groups = chunk // gmm.SCALE_GROUP
    windows = arr.reshape(*arr.shape[:-1], arr.shape[-1] // groups, groups)
    return int((windows.max(axis=-1) - windows.min(axis=-1)).max())


def _gmm_reference(lhs, packed, scales, group_sizes, preferred_element_type=None):
    """Grouped MXFP4 matmul without Pallas, on checkpoint-layout weights.

    `[M, K]` rows grouped by expert against `[E, N, K/2]` nibbles and
    `[E, N, K/32]` scales; rows outside every group are zero.
    """
    dense = jnp.asarray(_dequant(np.asarray(packed), np.asarray(scales)))
    # [E, N, K] -> [E, K, N]: ragged_dot contracts lhs's minor with the rhs's
    # second-minor.
    dense = dense.transpose(0, 2, 1).astype(jnp.bfloat16)
    out = jax.lax.ragged_dot(lhs, dense.astype(lhs.dtype), group_sizes)
    return out if preferred_element_type is None else out.astype(preferred_element_type)


def _fused_gate_up_reference(
    lhs, gate_packed, gate_scales, up_packed, up_scales, group_sizes, activation=ACT
):
    """Gate and up `_gmm_reference` matmuls followed by the SiTU/SiLU epilogue."""
    beta, linear_beta = gmm.parse_situ_activation(activation)
    gate = _gmm_reference(lhs, gate_packed, gate_scales, group_sizes)
    up = _gmm_reference(lhs, up_packed, up_scales, group_sizes)
    return gmm._situ_f32(gate, up, beta=beta, linear_beta=linear_beta).astype(
        gate.dtype
    )


def _assert_close(actual, expected, rel=2e-2):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=rel * scale)


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(0)
    w13, s13 = _ckpt_stack(rng, E, 2 * INTER, K)
    w2, s2 = _ckpt_stack(rng, E, K, INTER)
    w13_pt, w13_sc = gmm.repack_w13_planar_t(jnp.asarray(w13), jnp.asarray(s13))
    w2_pt, w2_sc = gmm.repack_w2_planar_t(jnp.asarray(w2), jnp.asarray(s2))
    return {
        "ckpt": (w13, s13, w2, s2),
        "planar_t": (w13_pt, w13_sc, w2_pt, w2_sc),
    }


def _lhs(m, seed, k=K):
    x = np.random.default_rng(seed).standard_normal((m, k)) * 0.5
    return jnp.asarray(x, jnp.bfloat16)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def test_planar_t_repack_round_trips():
    packed = np.random.default_rng(1).integers(0, 256, (2, 64, 512), np.uint8)
    repacked = gmm.repack_planar_t_stack(jnp.asarray(packed))
    assert repacked.shape == (2, 512, 64)
    np.testing.assert_array_equal(_unpack_planar_t_stack(repacked), packed)


def test_planar_repack_puts_window_halves_in_one_byte():
    # One 512-element window: byte j holds elements j (low) and j + 256 (high).
    codes = np.arange(512, dtype=np.int64) % 16
    interleaved = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    planar = np.asarray(gmm.repack_planar_stack(jnp.asarray(interleaved)))
    np.testing.assert_array_equal(planar & 0x0F, codes[:256])
    np.testing.assert_array_equal(planar >> 4, codes[256:])


def test_transpose_scales_planar_t_is_an_involution():
    scales = np.random.default_rng(2).integers(0, 256, (2, 40, 32), np.uint8)
    out = gmm.transpose_scales_planar_t(jnp.asarray(scales))
    assert out.shape == (2, 32, 40)
    np.testing.assert_array_equal(
        np.asarray(gmm.transpose_scales_planar_t(out)), scales
    )


# ---------------------------------------------------------------------------
# b11 chunk decode exactness and the fold window
# ---------------------------------------------------------------------------
def _window(spread, seed, tn=128, lo=118):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (tn, 256), np.uint8)  # [tn, 512/2]
    scales = rng.integers(lo, lo + spread + 1, (tn, 16), np.uint8)
    scales[:, 0] = lo
    scales[:, 1] = lo + spread
    return packed, scales


@pytest.mark.parametrize("spread", [0, 2, gmm.B11_FOLD_WINDOW])
def test_b11_chunk_decode_is_exact_inside_fold_window(spread):
    packed, scales = _window(spread, seed=10 + spread)
    codes, wscale = gmm._dequantize_chunk_planar_t_b11(
        gmm.repack_planar_t_stack(jnp.asarray(packed)),
        gmm.transpose_scales_planar_t(jnp.asarray(scales)),
    )
    decoded = np.asarray(codes.astype(jnp.float32) * wscale)
    np.testing.assert_array_equal(decoded, _dequant(packed, scales).T)
    assert _scale_spread(scales) <= gmm.B11_FOLD_WINDOW


def test_b11_chunk_decode_misdecodes_past_fold_window():
    spread = gmm.B11_FOLD_WINDOW + 1
    packed, scales = _window(spread, seed=30)
    codes, wscale = gmm._dequantize_chunk_planar_t_b11(
        gmm.repack_planar_t_stack(jnp.asarray(packed)),
        gmm.transpose_scales_planar_t(jnp.asarray(scales)),
    )
    decoded = np.asarray(codes.astype(jnp.float32) * wscale)
    assert not np.array_equal(decoded, _dequant(packed, scales).T)
    assert _scale_spread(scales) == spread


# ---------------------------------------------------------------------------
# Grouped matmul kernels against the dense references
# ---------------------------------------------------------------------------
def _served_tiling(m, k, n, pairs):
    return gmm.default_tiling(
        m,
        k,
        n,
        lhs_bytes=2,
        rhs_pairs=pairs,
        expected_rows=max(32, -(-m // 8)),
        vmem_budget=gmm.V6E_ROLLED_VMEM_BUDGET,
        scoped_elem_bytes=gmm.SCOPED_ELEM_BYTES_B11,
        force_chunk=gmm.PLANAR_WINDOW,
    )


def test_mxfp4_gmm_matches_reference(weights):
    _, _, w2, s2 = weights["ckpt"]
    _, _, w2_pt, w2_sc = weights["planar_t"]
    m = 64
    h = _lhs(m, 200, k=INTER)
    group_sizes = jnp.asarray([20, 0, 44], jnp.int32)
    tiling = _served_tiling(m, INTER, K, 1)
    out = gmm.mxfp4_gmm(
        h,
        w2_pt,
        jnp.squeeze(w2_sc, 2),
        group_sizes,
        tiling=tiling,
        group_metadata=gmm._make_group_metadata(
            group_sizes, m=m, tm=tiling[0], num_groups=E
        ),
        interpret=True,
    )
    ref = _gmm_reference(
        h,
        jnp.asarray(w2),
        jnp.asarray(s2),
        group_sizes,
        preferred_element_type=jnp.float32,
    )
    assert out.shape == (m, K) and out.dtype == jnp.bfloat16
    _assert_close(out, ref)


def _moe_reference(weights, lhs, group_sizes, activation=ACT):
    w13, s13, w2, s2 = (jnp.asarray(a) for a in weights["ckpt"])
    h = _fused_gate_up_reference(
        lhs,
        w13[:, :INTER],
        s13[:, :INTER],
        w13[:, INTER:],
        s13[:, INTER:],
        group_sizes,
        activation=activation,
    )
    return _gmm_reference(h, w2, s2, group_sizes, preferred_element_type=jnp.float32)


@pytest.mark.parametrize("activation", [ACT, "silu"])
@pytest.mark.parametrize("sizes", [[3, 0, 5], [17, 9, 14]])
def test_moe_gmm_matches_reference(weights, sizes, activation):
    m = sum(sizes)
    lhs = _lhs(m, 100 + m)
    group_sizes = jnp.asarray(sizes, jnp.int32)
    with pltpu.force_tpu_interpret_mode():
        out = gmm.moe_gmm_v6e_planar_t_b11(
            lhs, *weights["planar_t"], group_sizes, activation=activation
        )
    assert out.shape == (m, K)
    _assert_close(
        out, _moe_reference(weights, lhs, group_sizes, activation=activation), rel=3e-2
    )


def test_moe_gmm_rejects_unsupported_operands(weights):
    w13_pt, w13_sc, w2_pt, w2_sc = weights["planar_t"]
    lhs = _lhs(8, 1)
    group_sizes = jnp.asarray([3, 0, 5], jnp.int32)
    with pytest.raises(NotImplementedError):
        gmm.moe_gmm_v6e_planar_t_b11(
            lhs, w13_pt, w13_sc, w2_pt, w2_sc, group_sizes, activation="gelu"
        )
    with pytest.raises(ValueError, match="envelope"):
        gmm.moe_gmm_v6e_planar_t_b11(
            lhs,
            w13_pt,
            w13_sc,
            jnp.swapaxes(w2_pt, 1, 2),
            w2_sc,
            group_sizes,
            activation=ACT,
        )


# ---------------------------------------------------------------------------
# Eligibility predicates and activation parsing
# ---------------------------------------------------------------------------
def test_layout_and_support_predicates():
    u8 = jax.ShapeDtypeStruct((E, K, INTER), jnp.uint8)
    sc = jax.ShapeDtypeStruct((E, 32, 1, INTER), jnp.uint8)
    assert gmm.is_planar_t_b11_layout(u8, sc, rhs_quant_dtype=None)
    assert not gmm.is_planar_t_b11_layout(u8, None, rhs_quant_dtype=None)
    assert not gmm.is_planar_t_b11_layout(u8, sc, rhs_quant_dtype=jnp.int4)
    v6e, v7x = pltpu.ChipVersion.TPU_V6E, pltpu.ChipVersion.TPU_7X
    supported = gmm.is_supported_by_planar_t_b11
    assert supported(3584, 3072, chip_version=v6e, activation=ACT)
    assert supported(512, 512, chip_version=v6e, activation="silu")
    assert not supported(3584, 3072, chip_version=v7x, activation=ACT)
    assert not supported(3584, 3072, chip_version=v7x, activation="silu")
    assert not supported(3584, 3000, chip_version=v6e, activation=ACT)
    assert not supported(3584, 3072, chip_version=v6e, activation="gelu")


@pytest.mark.parametrize("rhs_pairs", [1, 2])
def test_support_predicate_rejects_exactly_the_widths_without_a_tiling(rhs_pairs):
    """The width bound is `default_tiling`'s own failure, not a second model."""

    def has_tiling(k):
        try:
            gmm.default_tiling(
                gmm._MOE_GMM_ROW_ALIGN,
                k,
                gmm.PLANAR_WINDOW,
                lhs_bytes=2,
                rhs_pairs=rhs_pairs,
                expected_rows=None,
                vmem_budget=gmm.V6E_ROLLED_VMEM_BUDGET,
                scoped_elem_bytes=gmm.SCOPED_ELEM_BYTES_B11,
                force_chunk=gmm.PLANAR_WINDOW,
            )
        except ValueError:
            return False
        return True

    widths = range(gmm.PLANAR_WINDOW, 250 * gmm.PLANAR_WINDOW, gmm.PLANAR_WINDOW)
    fits = [gmm._smallest_tiling_fits(k, rhs_pairs) for k in widths]
    assert fits == [has_tiling(k) for k in widths]
    # Both outcomes occur, and the measured widths are far inside the bound.
    assert fits[0] and not fits[-1]
    assert gmm._smallest_tiling_fits(8 * 3584, rhs_pairs)


def test_support_predicate_rejects_a_width_without_a_tiling():
    v6e = pltpu.ChipVersion.TPU_V6E
    supported = gmm.is_supported_by_planar_t_b11
    too_wide = 249 * gmm.PLANAR_WINDOW
    assert not supported(too_wide, 3072, chip_version=v6e, activation=ACT)
    assert not supported(3584, too_wide, chip_version=v6e, activation=ACT)


def test_parse_situ_activation():
    assert gmm.parse_situ_activation("situ:4.0:25.0") == (4.0, 25.0)
    assert gmm.parse_situ_activation("situ:2:none") == (2.0, None)
    assert gmm.parse_situ_activation("silu") == (0.0, None)
    with pytest.raises(NotImplementedError):
        gmm.parse_situ_activation("gelu")
