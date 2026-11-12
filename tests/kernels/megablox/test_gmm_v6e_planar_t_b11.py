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
from vllm_torchtpu.kernels.megablox.gmm_v2 import situ_and_mul

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


def _dequant(packed, scales):
    """Dense f32 `[..., n, k]` weights of a checkpoint-layout MXFP4 stack."""
    lut = np.asarray(gmm._E2M1_LUT, np.float32)
    codes = np.stack([packed & 0x0F, packed >> 4], axis=-1)
    values = lut[codes.reshape(*packed.shape[:-1], -1)]
    exps = np.exp2(scales.astype(np.float32) - 127.0)
    k = values.shape[-1]
    values = values.reshape(*values.shape[:-1], k // 32, 32) * exps[..., None]
    return values.reshape(*packed.shape[:-1], k)


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
@pytest.mark.parametrize("as_jax", [False, True])
def test_planar_t_repack_round_trips(as_jax):
    packed = np.random.default_rng(1).integers(0, 256, (2, 64, 512), np.uint8)
    src = jnp.asarray(packed) if as_jax else packed
    repacked = gmm.repack_planar_t_stack(src)
    assert repacked.shape == (2, 512, 64)
    assert isinstance(repacked, jax.Array if as_jax else np.ndarray)
    np.testing.assert_array_equal(
        np.asarray(gmm.unpack_planar_t_stack(repacked)), packed
    )


def test_planar_repack_puts_window_halves_in_one_byte():
    # One 512-element window: byte j holds elements j (low) and j + 256 (high).
    codes = np.arange(512, dtype=np.int64) % 16
    interleaved = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    planar = gmm.repack_planar_stack(interleaved)
    np.testing.assert_array_equal(planar & 0x0F, codes[:256])
    np.testing.assert_array_equal(planar >> 4, codes[256:])


def test_transpose_scales_planar_t_is_an_involution():
    scales = np.random.default_rng(2).integers(0, 256, (2, 40, 32), np.uint8)
    out = gmm.transpose_scales_planar_t(scales)
    assert out.shape == (2, 32, 40)
    np.testing.assert_array_equal(gmm.transpose_scales_planar_t(out), scales)


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
        jnp.asarray(gmm.repack_planar_t_stack(packed)),
        jnp.asarray(gmm.transpose_scales_planar_t(scales)),
    )
    decoded = np.asarray(codes.astype(jnp.float32) * wscale)
    np.testing.assert_array_equal(decoded, _dequant(packed, scales).T)
    assert gmm.planar_t_b11_window_ok(scales)


def test_b11_chunk_decode_misdecodes_past_fold_window():
    spread = gmm.B11_FOLD_WINDOW + 1
    packed, scales = _window(spread, seed=30)
    codes, wscale = gmm._dequantize_chunk_planar_t_b11(
        jnp.asarray(gmm.repack_planar_t_stack(packed)),
        jnp.asarray(gmm.transpose_scales_planar_t(scales)),
    )
    decoded = np.asarray(codes.astype(jnp.float32) * wscale)
    assert not np.array_equal(decoded, _dequant(packed, scales).T)
    assert gmm.planar_t_scale_spread(scales) == spread
    assert not gmm.planar_t_b11_window_ok(scales)


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
        scoped_elem_bytes=gmm.SCOPED_ELEM_BYTES_TPU7X,
        force_chunk=gmm.PLANAR_WINDOW,
    )


@pytest.mark.parametrize(
    "dequant", [gmm.DEQUANT_PLANAR_T_B11, gmm.DEQUANT_PLANAR_T_B11_DEEP]
)
def test_mxfp4_gmm_matches_reference(weights, dequant):
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
        interpret=True,
        dequant=dequant,
        deep_vmem_budget=gmm.V6E_DEEP_VMEM_BUDGET,
    )
    ref = gmm.mxfp4_gmm_reference(
        h,
        jnp.asarray(w2),
        jnp.asarray(s2),
        group_sizes,
        preferred_element_type=jnp.float32,
    )
    assert out.shape == (m, K) and out.dtype == jnp.bfloat16
    _assert_close(out, ref)


def test_deep_and_rolled_schedules_are_bitwise_identical(weights):
    _, _, w2_pt, w2_sc = weights["planar_t"]
    m = 64
    h = _lhs(m, 201, k=INTER)
    group_sizes = jnp.asarray([32, 32, 0], jnp.int32)
    tiling = _served_tiling(m, INTER, K, 1)
    outs = [
        gmm.mxfp4_gmm(
            h,
            w2_pt,
            jnp.squeeze(w2_sc, 2),
            group_sizes,
            tiling=tiling,
            interpret=True,
            dequant=dequant,
            deep_vmem_budget=gmm.V6E_DEEP_VMEM_BUDGET,
        )
        for dequant in gmm.DEQUANT_MODES
    ]
    np.testing.assert_array_equal(np.asarray(outs[0]), np.asarray(outs[1]))


def _moe_reference(weights, lhs, group_sizes):
    w13, s13, w2, s2 = (jnp.asarray(a) for a in weights["ckpt"])
    h = gmm.mxfp4_fused_gate_up_reference(
        lhs,
        w13[:, :INTER],
        s13[:, :INTER],
        w13[:, INTER:],
        s13[:, INTER:],
        group_sizes,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    return gmm.mxfp4_gmm_reference(
        h, w2, s2, group_sizes, preferred_element_type=jnp.float32
    )


@pytest.mark.parametrize("sizes", [[3, 0, 5], [17, 9, 14]])
def test_moe_gmm_matches_reference(weights, sizes):
    m = sum(sizes)
    lhs = _lhs(m, 100 + m)
    group_sizes = jnp.asarray(sizes, jnp.int32)
    out = gmm.moe_gmm_v6e_planar_t_b11(
        lhs, *weights["planar_t"], group_sizes, activation=ACT
    )
    assert out.shape == (m, K)
    _assert_close(out, _moe_reference(weights, lhs, group_sizes), rel=3e-2)


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
# Fused small-batch kernel
# ---------------------------------------------------------------------------
def _routing(m, seed, topk=2, num_pad=0):
    rng = np.random.default_rng(seed)
    tw = np.abs(rng.standard_normal((m, topk))).astype(np.float32)
    tw /= tw.sum(-1, keepdims=True)
    ti = np.stack([rng.permutation(E)[:topk] for _ in range(m)]).astype(np.int32)
    if num_pad:
        ti[m - num_pad :] = -1
        tw[m - num_pad :] = 0.0
    return jnp.asarray(tw, jnp.bfloat16), jnp.asarray(ti)


def _fused_reference(weights, x, tw, ti):
    w13, s13, w2, s2 = weights["ckpt"]
    w13_d = jnp.asarray(_dequant(w13, s13)).astype(jnp.bfloat16)  # [E, 2N, K]
    w2_d = jnp.asarray(_dequant(w2, s2)).astype(jnp.bfloat16)  # [E, K, N]
    xf = x.astype(jnp.float32)
    out = jnp.zeros((x.shape[0], K), jnp.float32)
    for e in range(E):
        gate = xf @ w13_d[e, :INTER].astype(jnp.float32).T
        up = xf @ w13_d[e, INTER:].astype(jnp.float32).T
        h = situ_and_mul(gate, up, BETA, LINEAR_BETA).astype(jnp.bfloat16)
        down = h.astype(jnp.float32) @ w2_d[e].astype(jnp.float32).T
        weight = jnp.sum(jnp.where(ti == e, tw.astype(jnp.float32), 0.0), axis=-1)
        out = out + weight[:, None] * down
    return out


@pytest.mark.parametrize(
    "m, num_pad, trays",
    [(1, 0, "off"), (8, 2, "off"), (40, 0, "trays2"), (72, 4, "off")],
)
def test_fused_decode_matches_reference(weights, m, num_pad, trays):
    x = _lhs(m, 400 + m)
    tw, ti = _routing(m, 500 + m, num_pad=num_pad)
    with pltpu.force_tpu_interpret_mode():
        out = gmm.fused_decode_moe_v6e_planar_t_b11(
            x,
            *weights["planar_t"][:2],
            *weights["planar_t"][2:],
            tw,
            ti,
            activation=ACT,
            preferred_element_type=jnp.bfloat16,
            token_trays_mode=trays,
        )
    assert out.shape == (m, K)
    _assert_close(out, _fused_reference(weights, x, tw, ti), rel=3e-2)
    if num_pad:
        np.testing.assert_array_equal(np.asarray(out[m - num_pad :]), 0)


def test_fused_decode_rejects_bad_requests(weights):
    tw, ti = _routing(8, 1)
    with pytest.raises(ValueError, match="token_trays_mode"):
        gmm.fused_decode_moe_v6e_planar_t_b11(
            _lhs(8, 1),
            *weights["planar_t"],
            tw,
            ti,
            activation=ACT,
            token_trays_mode="trays4",
        )
    m = gmm.FUSED_VERIFY_MOE_MAX_TOKENS + 1
    tw, ti = _routing(m, 2)
    with pytest.raises(ValueError, match="at most"):
        gmm.fused_decode_moe_v6e_planar_t_b11(
            _lhs(m, 2), *weights["planar_t"], tw, ti, activation=ACT
        )


# ---------------------------------------------------------------------------
# Eligibility predicates
# ---------------------------------------------------------------------------
def _decode_operands(m=8, k=K, n=INTER, e=E, w1_dtype=jnp.uint8):
    s = jax.ShapeDtypeStruct
    return (
        s((m, k), jnp.bfloat16),
        s((e, k, n), w1_dtype),
        s((e, n // 2, k), jnp.uint8),
        s((e, 2 * k // 32, 1, n), jnp.uint8),
        s((e, n // 32, 1, k), jnp.uint8),
    )


def test_can_use_fused_decode_envelope():
    hs, w1, w2, s1, s2 = _decode_operands()
    ok = gmm.can_use_fused_decode_moe_v6e_planar_t_b11
    assert ok(hs, w1, w2, s1, s2, None, None)
    bias = jax.ShapeDtypeStruct((E, INTER), jnp.bfloat16)
    assert not ok(hs, w1, w2, s1, s2, bias, None)
    assert not ok(hs, w1, w2, None, s2, None, None)
    assert not ok(hs, w1, w2, s1, s2, None, None, rhs_quant_dtype=jnp.int4)
    assert not ok(*_decode_operands(w1_dtype=jnp.bfloat16), None, None)
    assert not ok(*_decode_operands(k=384), None, None)
    m_big = gmm.FUSED_DECODE_MOE_MAX_TOKENS + 1
    assert not ok(*_decode_operands(m=m_big), None, None)
    assert ok(
        *_decode_operands(m=m_big),
        None,
        None,
        max_tokens=gmm.FUSED_VERIFY_MOE_MAX_TOKENS,
    )


def test_can_use_kernel_requires_the_planar_window():
    assert gmm.can_use_kernel(64, 3584)
    assert not gmm.can_use_kernel(64, 96)
    assert not gmm.can_use_kernel(64, 1024, tiling=(32, 256, 512))


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
    assert not supported(3584, 3000, chip_version=v6e, activation=ACT)
    assert not supported(3584, 3072, chip_version=v6e, activation="gelu")


def test_parse_situ_activation():
    assert gmm.parse_situ_activation("situ:4.0:25.0") == (4.0, 25.0)
    assert gmm.parse_situ_activation("situ:2:none") == (2.0, None)
    assert gmm.parse_situ_activation("silu") == (0.0, None)
    with pytest.raises(NotImplementedError):
        gmm.parse_situ_activation("gelu")
