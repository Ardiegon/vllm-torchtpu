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
"""GDN v3 shared-pool write-back modes and granule offset rules.

Covers the side-effect-free verify (`StateWrite.NONE`), the masked commit
(`StateWrite.LAST` + `window_lens`), explicit `shared_conv_granule_offset`, and
`fused_conv1d_gdn` argument checks, including end-to-end execution of the real
Pallas kernel on CPU (interpret mode) and TPU (`@pytest.mark.tpu_test`).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from kernels.gdn.v3 import gdn_pool_state_utils as pool_utils

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref, metadata, wrapper

H, D, KS = 2, 128, 4
PACK, LANES = 2, 640
W = 8


@pytest.fixture(autouse=True)
def _cpu_tpu_info(monkeypatch):
    """Answer `pltpu.get_tpu_info()` on CPU without initializing a TPU."""
    if jax.default_backend() != "tpu":
        v6e = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: v6e)


def _cfg(
    state_write,
    *,
    window=W,
    tile=1,
    is_shared=True,
    shared_conv_granule_offset=None,
):
    f32 = jnp.dtype(jnp.float32)
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=f32,
            act_out=jnp.dtype(jnp.bfloat16),
            compute=f32,
            recurrent_state=f32,
            conv_state=f32,
        ),
        batch_size=32,
        dim_size=3 * H * D,
        kernel_size=KS,
        tile_size=tile,
        num_kq_heads=H,
        num_v_heads=H,
        kq_head_dim=D,
        v_head_dim=D,
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
        window_size=window,
        is_shared=is_shared,
        state_write=state_write,
        shared_conv_granule_offset=shared_conv_granule_offset,
    )


def _shared_pool():
    tok_bytes = PACK * LANES * 2
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=H * D * D * 4,
        conv_bytes=(KS - 1) * 3 * H * D * 4,
        token_bytes=tok_bytes,
    )
    rows = -(-layout.required_tokens // 10) * 10
    pool = jnp.zeros((3, rows, PACK, LANES), jnp.bfloat16)
    conv_offset = layout.ssm_tokens * (tok_bytes // 512)
    return pool, layout, conv_offset


class TestStateWrite:
    def test_values_are_the_legacy_strings(self):
        assert [str(m) for m in config.StateWrite] == ["all", "none", "last"]
        assert config.StateWrite("last") is config.StateWrite.LAST

    @pytest.mark.parametrize(
        ("state_write", "suffix"),
        [
            (config.StateWrite.ALL, "_w8_bounded"),
            (config.StateWrite.NONE, "_w8_bounded_swnone"),
            (config.StateWrite.LAST, "_w8_bounded_swlast"),
        ],
    )
    def test_kernel_name_and_metadata(self, state_write, suffix):
        cfg = _cfg(state_write)
        assert cfg.get_kernel_name() == f"fused_conv1d_kda_batched_b1_c8{suffix}"
        md = cfg.get_metadata()
        if state_write == config.StateWrite.ALL:
            assert "state_write" not in md
        else:
            assert md["state_write"] == str(state_write)
            assert type(md["state_write"]) is str

    def test_shared_conv_granule_offset_overrides_default_ssm_size(self):
        # Default assumes zero row padding (2 * 128 = 256 granules); MLA's
        # 5-granule packed rows pad 256 granules up to 52 * 5 = 260 granules.
        default_cfg = _cfg(config.StateWrite.ALL)
        assert default_cfg.conv_hbm_offset == 256
        mla_cfg = _cfg(config.StateWrite.ALL, shared_conv_granule_offset=260)
        assert mla_cfg.conv_hbm_offset == 260
        dense_cfg = _cfg(
            config.StateWrite.ALL, is_shared=False, shared_conv_granule_offset=260
        )
        assert dense_cfg.conv_hbm_offset == 0


class TestStateOutBufferedRef:
    def _stub(self, state_write, window_lens):
        cfg = _cfg(state_write)
        md = metadata.compute_batched_seq_metadata(
            cfg=cfg,
            seq_lens=jnp.asarray([108, 3, 58, 9, 0], jnp.int32),
            query_start_loc=jnp.asarray([0, 8, 11, 19, 20, 20], jnp.int32),
            state_indices=jnp.asarray([4, 1, 3, 2, 0], jnp.int32),
            read_offsets=jnp.zeros((5,), jnp.int32),
            end_seq=jnp.asarray(4, jnp.int32),
            window_lens=window_lens,
        )
        return types.SimpleNamespace(cfg=cfg, metadata_ref=md), md

    def test_masked_commit_writes_one_checkpoint_per_nonempty_row(self):
        window_lens = jnp.asarray([3, 0, 8, 1, 0], jnp.int32)
        stub, md = self._stub(config.StateWrite.LAST, window_lens)
        cls = memory_ref.StateOutBufferedRef
        written = [int(cls._num_written(stub, p, 0)) for p in range(5)]
        assert written == [1, 0, 1, 1, 0]
        r_size = [int(md.p_id_to_r_size[p, 0]) for p in range(5)]
        assert r_size == [3, 0, 8, 1, 0]

    def test_all_writes_one_checkpoint_per_valid_position(self):
        stub, _ = self._stub(config.StateWrite.ALL, None)
        cls = memory_ref.StateOutBufferedRef
        written = [int(cls._num_written(stub, p, 0)) for p in range(5)]
        assert written == [8, 3, 8, 1, 0]

    def test_none_never_counts_writes(self):
        stub, _ = self._stub(config.StateWrite.NONE, None)
        with pytest.raises(AssertionError):
            memory_ref.StateOutBufferedRef._num_written(stub, 0, 0)


class TestFusedConv1dGdnArgumentChecks:
    T = 8

    def _call(self, **over):
        pool, _, conv_offset = _shared_pool()
        rng = np.random.default_rng(0)

        def arr(shape, dtype=jnp.bfloat16):
            return jnp.asarray(rng.standard_normal(shape), dtype)

        kwargs = dict(
            qkv=arr((self.T, 3 * H * D)),
            b=arr((self.T, H)),
            a=arr((self.T, H * D)),
            states=pool,
            conv_weight=arr((3 * H * D, 1, KS), jnp.float32),
            conv_bias=None,
            a_log=arr((H,), jnp.float32),
            dt_bias=arr((H, D), jnp.float32),
            query_start_loc=jnp.asarray([0, 4, 8], jnp.int32),
            state_indices=jnp.asarray([1, 2], jnp.int32),
            distribution=jnp.asarray([2, 2, 2], jnp.int32),
            seq_lens=jnp.asarray([9, 9], jnp.int32),
            read_offsets=jnp.zeros((2,), jnp.int32),
            n_kq=H,
            n_v=H,
            d_k=D,
            d_v=D,
            kernel_size=KS,
            attention_mode=config.AttentionMode.KDA,
            gate_lower_bound=-5.0,
            recurrent_dtype=jnp.dtype(jnp.float32),
            conv_dtype=jnp.dtype(jnp.float32),
            num_spec_tokens=3,
            batched_state_write=config.StateWrite.NONE,
            shared_conv_granule_offset=conv_offset,
        )
        kwargs.update(over)
        return wrapper.fused_conv1d_gdn(**kwargs)

    def test_unknown_write_mode(self):
        with pytest.raises(ValueError):
            self._call(batched_state_write="first")

    def test_window_lens_needs_masked_commit(self):
        with pytest.raises(ValueError, match="masked commit"):
            self._call(window_lens=jnp.ones((2,), jnp.int32))

    def test_batched_and_prefill_only_are_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._call(batched_only=True, prefill_only=True)

    def test_windowed_all_needs_ckpt_indices(self):
        with pytest.raises(AssertionError, match="requires ckpt_indices"):
            self._call(batched_state_write=config.StateWrite.ALL)

    def test_write_mode_needs_shared_state_pool(self):
        with pytest.raises(AssertionError, match="shared state pool"):
            self._call(
                states=(
                    jnp.zeros((3, KS - 1, 3 * H * D), jnp.float32),
                    jnp.zeros((3, H, D, D), jnp.float32),
                ),
                shared_conv_granule_offset=None,
                batched_state_write=config.StateWrite.LAST,
            )


def _kda_prefix_reference(
    qkv: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    init_conv: np.ndarray,
    init_ssm: np.ndarray,
    conv_weight: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    *,
    gate_lower_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float64 conv1d + SiLU + KDA recurrence over one request's token prefix."""
    t_len = qkv.shape[0]
    if t_len == 0:
        return (
            np.zeros((0, H * D), np.float64),
            init_conv.astype(np.float32, copy=True),
            init_ssm.astype(np.float32, copy=True),
        )
    x_full = np.concatenate(
        [init_conv.astype(np.float32), qkv.astype(np.float32)], axis=0
    )
    acc = np.zeros((t_len, 3 * H * D), dtype=np.float32)
    for k_idx in range(KS):
        acc += x_full[k_idx : k_idx + t_len] * conv_weight[:, 0, k_idx][None, :]
    final_conv = x_full[-(KS - 1) :].astype(np.float32)
    # Match kernel bf16 rounding before SiLU when qkv is bf16.
    conv_bf16 = np.asarray(jnp.asarray(acc, jnp.bfloat16), np.float64)
    activated = conv_bf16 / (1.0 + np.exp(-conv_bf16))

    key_dim = H * D
    q_seq = activated[:, :key_dim].reshape(t_len, H, D)
    k_seq = activated[:, key_dim : 2 * key_dim].reshape(t_len, H, D)
    v_seq = activated[:, 2 * key_dim :].reshape(t_len, H, D)

    def _l2_norm(x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32)
        inv = 1.0 / np.sqrt(np.sum(xf * xf, axis=-1, keepdims=True) + 1e-6)
        return (xf * inv).astype(np.float64)

    q_seq = _l2_norm(q_seq)
    k_seq = _l2_norm(k_seq)
    beta_seq = 1.0 / (1.0 + np.exp(-b.astype(np.float64)))
    a_seq = a.astype(np.float64).reshape(t_len, H, D)
    scaled = np.exp(a_log.astype(np.float64).reshape(H, 1)) * (
        a_seq + dt_bias.astype(np.float64).reshape(H, D)
    )
    gate_seq = gate_lower_bound * np.exp(-np.logaddexp(0.0, -scaled))

    running = init_ssm.astype(np.float64).copy()
    out = np.zeros((t_len, H, D), np.float64)
    scale = float(D**-0.5)
    for tok in range(t_len):
        running *= np.exp(gate_seq[tok])[:, :, None]
        pred = np.einsum("hk,hkv->hv", k_seq[tok], running)
        delta = beta_seq[tok][:, None] * (v_seq[tok] - pred)
        running += k_seq[tok][:, :, None] * delta[:, None, :]
        out[tok] = scale * np.einsum("hk,hkv->hv", q_seq[tok], running)
    return out.reshape(t_len, H * D), final_conv, running.astype(np.float32)


def _write_pool_states_at_offset(
    pool: jax.Array,
    ssm: np.ndarray,
    conv: np.ndarray,
    idx: np.ndarray,
    *,
    conv_granule_offset: int,
) -> jax.Array:
    def _build(pool_arr, ssm_arr, conv_arr, idx_arr):
        words = pool_utils.state_words(pool_arr, split=1)
        words = pool_utils.put_words(
            words, idx_arr, 0, jnp.asarray(ssm_arr, jnp.float32)
        )
        words = pool_utils.put_words(
            words, idx_arr, conv_granule_offset, jnp.asarray(conv_arr, jnp.float32)
        )
        return pool_utils.words_to_pool(words, pool_arr)

    return pool_utils.on_cpu(_build, pool, ssm, conv, idx)


def _read_pool_states_at_offset(
    pool: jax.Array,
    idx: np.ndarray,
    *,
    conv_granule_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    def _take(pool_arr, idx_arr):
        words = pool_utils.state_words(pool_arr, split=1)
        ssm = pool_utils.take_words(words, idx_arr, 0, (H, D, D), jnp.float32)
        conv = pool_utils.take_words(
            words, idx_arr, conv_granule_offset, (KS - 1, 3 * H * D), jnp.float32
        )
        return ssm, conv

    ssm, conv = pool_utils.on_cpu(_take, pool, idx)
    return np.asarray(ssm), np.asarray(conv)


def _run_real_kernel_verify_and_commit() -> None:
    """Execute `wrapper.fused_conv1d_gdn` for `StateWrite.NONE` and `LAST`."""
    rng = np.random.default_rng(42)
    win = 4
    num_spec = win - 1
    num_reqs = 3
    total_tokens = num_reqs * win
    gate_lb = -5.0

    # MLA-style 5-granule token rows: recurrent state is 256 granules, padded to
    # 52 * 5 = 260 granules, so `conv_offset == 260 != 256` exercises the
    # explicit `shared_conv_granule_offset` override inside the Pallas kernel.
    pool, _, conv_offset = _shared_pool()
    assert conv_offset == 260
    num_blocks = 4
    # Fill every block with finite non-zero sentinel values so out-of-bounds or
    # `StateWrite.NONE` writes are caught immediately.
    sentinel = jnp.asarray(
        rng.uniform(-0.25, 0.25, (num_blocks, *pool.shape[1:])), jnp.bfloat16
    )
    state_idx_np = np.array([1, 2, 3], dtype=np.int32)
    init_ssm = rng.standard_normal((num_reqs, H, D, D)).astype(np.float32) * 0.05
    init_conv = (
        rng.standard_normal((num_reqs, KS - 1, 3 * H * D)).astype(np.float32) * 0.2
    )
    pool_in = _write_pool_states_at_offset(
        sentinel, init_ssm, init_conv, state_idx_np, conv_granule_offset=conv_offset
    )

    qkv = jnp.asarray(
        rng.standard_normal((total_tokens, 3 * H * D)) * 0.3, jnp.bfloat16
    )
    b = jnp.asarray(rng.standard_normal((total_tokens, H)) * 0.3, jnp.bfloat16)
    a = jnp.asarray(rng.standard_normal((total_tokens, H * D)) * 0.3, jnp.bfloat16)
    conv_weight = jnp.asarray(
        rng.standard_normal((3 * H * D, 1, KS)) * 0.2, jnp.float32
    )
    a_log = jnp.asarray(rng.standard_normal((H,)) * 0.2, jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal((H, D)) * 0.2, jnp.float32)
    qsl = jnp.asarray([0, win, 2 * win, 3 * win], jnp.int32)
    state_indices = jnp.asarray(state_idx_np, jnp.int32)
    # seq_lens > win so `has_initial_state` is True and initial pool state is read.
    seq_lens = jnp.asarray([20 + win, 15 + win, 10 + win], jnp.int32)
    dist = jnp.asarray([num_reqs, num_reqs, num_reqs], jnp.int32)
    read_offsets = jnp.zeros((num_reqs,), jnp.int32)

    common = dict(
        conv_weight=conv_weight,
        conv_bias=None,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=qsl,
        state_indices=state_indices,
        distribution=dist,
        seq_lens=seq_lens,
        read_offsets=read_offsets,
        n_kq=H,
        n_v=H,
        d_k=D,
        d_v=D,
        kernel_size=KS,
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=gate_lb,
        recurrent_dtype=jnp.dtype(jnp.float32),
        conv_dtype=jnp.dtype(jnp.float32),
        num_spec_tokens=num_spec,
        decode_tile_size=2,
        shared_conv_granule_offset=conv_offset,
    )

    # 1. Verify (`StateWrite.NONE`): outputs match full-window reference, pool
    # bytes are untouched bit-for-bit.
    pool_after_verify, verify_out = wrapper.fused_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        states=pool_in,
        batched_state_write=config.StateWrite.NONE,
        **common,
    )
    np.testing.assert_array_equal(
        np.asarray(pool_after_verify.view(jnp.uint16)),
        np.asarray(pool_in.view(jnp.uint16)),
    )
    qkv_np = np.asarray(qkv, np.float32)
    b_np = np.asarray(b, np.float32)
    a_np = np.asarray(a, np.float32)
    cw_np = np.asarray(conv_weight, np.float32)
    al_np = np.asarray(a_log, np.float32)
    dtb_np = np.asarray(dt_bias, np.float32)
    for r in range(num_reqs):
        s0, s1 = r * win, (r + 1) * win
        ref_out, _, _ = _kda_prefix_reference(
            qkv_np[s0:s1],
            b_np[s0:s1],
            a_np[s0:s1],
            init_conv[r],
            init_ssm[r],
            cw_np,
            al_np,
            dtb_np,
            gate_lower_bound=gate_lb,
        )
        np.testing.assert_allclose(
            np.asarray(verify_out[s0:s1], np.float32), ref_out, rtol=2e-2, atol=2e-2
        )

    # 2. Masked commit (`StateWrite.LAST` + `window_lens=[3, 1, 0]`): request 0
    # advances by 3 tokens, request 1 by 1 token, request 2 (`0`) writes nothing,
    # and block 0 stays untouched.
    accepted_lens = jnp.asarray([3, 1, 0], jnp.int32)
    pool_after_commit, _ = wrapper.fused_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        states=pool_in,
        batched_state_write=config.StateWrite.LAST,
        window_lens=accepted_lens,
        batched_only=True,
        **common,
    )
    # Block 0 and block 3 (`window_lens == 0`) must stay bit-exact.
    for untouched_blk in (0, 3):
        np.testing.assert_array_equal(
            np.asarray(pool_after_commit[untouched_blk].view(jnp.uint16)),
            np.asarray(pool_in[untouched_blk].view(jnp.uint16)),
        )
    got_ssm, got_conv = _read_pool_states_at_offset(
        pool_after_commit,
        np.array([1, 2], dtype=np.int32),
        conv_granule_offset=conv_offset,
    )
    for i, (r, prefix_len) in enumerate(((0, 3), (1, 1))):
        s0 = r * win
        _, want_conv, want_ssm = _kda_prefix_reference(
            qkv_np[s0 : s0 + prefix_len],
            b_np[s0 : s0 + prefix_len],
            a_np[s0 : s0 + prefix_len],
            init_conv[r],
            init_ssm[r],
            cw_np,
            al_np,
            dtb_np,
            gate_lower_bound=gate_lb,
        )
        _, full_conv, full_ssm = _kda_prefix_reference(
            qkv_np[s0 : s0 + win],
            b_np[s0 : s0 + win],
            a_np[s0 : s0 + win],
            init_conv[r],
            init_ssm[r],
            cw_np,
            al_np,
            dtb_np,
            gate_lower_bound=gate_lb,
        )
        np.testing.assert_allclose(got_conv[i], want_conv, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(got_ssm[i], want_ssm, rtol=2e-2, atol=2e-2)
        # Must-fail control: the masked state differs from committing the full window.
        assert not np.allclose(got_conv[i], full_conv, rtol=1e-2, atol=1e-2)
        assert not np.allclose(got_ssm[i], full_ssm, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("chip", [pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V6E])
def test_fused_conv1d_gdn_traces_state_write_dmas_and_conv_offset(monkeypatch, chip):
    """CPU trace of `wrapper.fused_conv1d_gdn` across `ALL`, `NONE`, and `LAST`.

    Note: `pltpu.force_tpu_interpret_mode()` cannot execute unified-pool DMAs on
    CPU because `memory_ref.granule_view` bitcasts/reshapes refs before
    `make_async_copy`, which `jax._src.pallas.mosaic.interpret.utils.to_range`
    rejects (`BitcastTransform` has no `indices`). Following
    `test_prefill_ckpt_metadata.py`, trace the real `fused_conv1d_gdn` entry
    point on CPU and verify exact DMA emission and `shared_conv_granule_offset`,
    paired with `test_real_kernel_verify_none_and_masked_commit_last_tpu` on TPU.
    """
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)

    pool, _, conv_offset = _shared_pool()
    assert conv_offset == 260
    win = 4
    num_reqs = 2
    total_tokens = num_reqs * win

    def _trace(state_write, *, ckpt_indices=None, window_lens=None, offset=conv_offset):
        return str(
            jax.make_jaxpr(
                lambda: wrapper.fused_conv1d_gdn(
                    qkv=jnp.zeros((total_tokens, 3 * H * D), jnp.bfloat16),
                    b=jnp.zeros((total_tokens, H), jnp.bfloat16),
                    a=jnp.zeros((total_tokens, H * D), jnp.bfloat16),
                    states=pool,
                    conv_weight=jnp.zeros((3 * H * D, 1, KS), jnp.float32),
                    conv_bias=None,
                    a_log=jnp.zeros((H,), jnp.float32),
                    dt_bias=jnp.zeros((H, D), jnp.float32),
                    query_start_loc=jnp.asarray([0, win, 2 * win], jnp.int32),
                    state_indices=jnp.asarray([1, 2], jnp.int32),
                    distribution=jnp.asarray([num_reqs, num_reqs, num_reqs], jnp.int32),
                    seq_lens=jnp.asarray([16, 16], jnp.int32),
                    read_offsets=jnp.zeros((num_reqs,), jnp.int32),
                    ckpt_indices=ckpt_indices,
                    n_kq=H,
                    n_v=H,
                    d_k=D,
                    d_v=D,
                    kernel_size=KS,
                    attention_mode=config.AttentionMode.KDA,
                    gate_lower_bound=-5.0,
                    recurrent_dtype=jnp.dtype(jnp.float32),
                    conv_dtype=jnp.dtype(jnp.float32),
                    num_spec_tokens=win - 1,
                    batched_only=True,
                    batched_state_write=state_write,
                    window_lens=window_lens,
                    shared_conv_granule_offset=offset,
                )
            )()
        )

    jaxpr_none = _trace(config.StateWrite.NONE)
    jaxpr_last = _trace(
        config.StateWrite.LAST, window_lens=jnp.asarray([3, 1], jnp.int32)
    )
    jaxpr_all = _trace(
        config.StateWrite.ALL,
        ckpt_indices=jnp.arange(num_reqs * win, dtype=jnp.int32).reshape(num_reqs, win),
    )

    # Windowed mode uses `seq_tile_size == 1`: each state region (SSM + conv = 2)
    # starts 1 copy-out in `LAST`, `win` in `ALL`, and `0` in `NONE`.
    starts_none = jaxpr_none.count("dma_start")
    starts_last = jaxpr_last.count("dma_start")
    starts_all = jaxpr_all.count("dma_start")
    assert starts_last - starts_none == 2
    assert starts_all - starts_none == 2 * win

    # `NONE` also omits the 4 copy-out waits (2 state regions x loop + epilogue).
    waits_none = jaxpr_none.count("dma_wait")
    assert jaxpr_last.count("dma_wait") - waits_none == 4
    assert jaxpr_all.count("dma_wait") - waits_none == 4

    # Changing `shared_conv_granule_offset` changes the traced conv slice offset.
    jaxpr_last_default = _trace(
        config.StateWrite.LAST,
        window_lens=jnp.asarray([3, 1], jnp.int32),
        offset=None,
    )
    assert jaxpr_last != jaxpr_last_default


@pytest.mark.tpu_test
def test_real_kernel_verify_none_and_masked_commit_last_tpu():
    _run_real_kernel_verify_and_commit()
