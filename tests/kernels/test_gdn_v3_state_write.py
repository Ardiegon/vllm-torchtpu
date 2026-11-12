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
"""GDN v3 pooled write-back modes and the bf16-as-uint32 state views (CPU).

Covers the pure-Python/XLA parts of the side-effect-free verify
(`StateWrite.NONE`) and the masked commit (`StateWrite.LAST`): which window
positions get a checkpoint, the copy_out DMA counts, the masked real sizes,
the kernel names/metadata, the `fused_conv1d_gdn` argument checks, and the
uint32 views `v3_state_source` plans for bf16 state. The fused kernel itself
needs a TPU and is covered by the TPU suites.
"""

import types
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import tpu_info

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref, metadata, wrapper

H, D, KS = 2, 128, 4
PACK, LANES = 2, 640
W = 8


@pytest.fixture(autouse=True)
def _cpu_v6e_tpu_info():
    """Answer `pltpu.get_tpu_info()` on CPU with the v6e table."""
    tpu_info.get_tpu_info.cache_clear()
    v6e = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)
    with patch.dict(tpu_info.registry, {"cpu": lambda: v6e}):
        yield
    tpu_info.get_tpu_info.cache_clear()


def _cfg(state_write, *, window=W, tile=1, plan=None):
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
        state_plan=plan,
        window_size=window,
        state_write=state_write,
    )


def _pool_rows():
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=H * D * D * 4,
        conv_bytes=(KS - 1) * 3 * H * D * 2,
        token_bytes=PACK * LANES * 2,
    )
    return layout, -(-layout.required_tokens // 16) * 16


def _plan():
    layout, rows = _pool_rows()
    pool = jnp.zeros((3, rows, PACK, LANES), jnp.bfloat16)
    return pool, pool_adapters.v3_state_source(
        pool,
        split=1,
        ssm_ntok=layout.ssm_tokens,
        conv_tok0=layout.ssm_tokens,
        conv_ntok=layout.conv_tokens,
        conv_dim=3 * H * D,
        n_v=H,
        d_k=D,
        d_v=D,
        kernel_size=KS,
    )


class TestStateWrite:
    def test_values_are_the_legacy_strings(self):
        assert [str(m) for m in config.StateWrite] == ["all", "none", "last"]
        assert config.StateWrite("last") is config.StateWrite.LAST

    @pytest.mark.parametrize(
        ("state_write", "window", "expected"),
        [
            (config.StateWrite.ALL, W, list(range(W))),
            (config.StateWrite.NONE, W, []),
            (config.StateWrite.LAST, W, [W - 1]),
            (config.StateWrite.ALL, 1, [0]),
            (config.StateWrite.LAST, 1, [0]),
            # Plain strings still select the same positions.
            ("none", W, []),
            ("last", W, [W - 1]),
        ],
    )
    def test_stored_window_positions(self, state_write, window, expected):
        cfg = _cfg(state_write, window=window)
        assert list(cfg.stored_window_positions()) == expected

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
            # The default is omitted so ALL kernels keep their old metadata.
            assert "state_write" not in md
        else:
            assert md["state_write"] == str(state_write)
            assert type(md["state_write"]) is str


class TestExternalStateOut:
    def _stub(self, state_write, window_lens):
        cfg = _cfg(state_write, plan=_plan()[1])
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
        cls = memory_ref.ExternalStateOutBufferedRef
        written = [int(cls._num_written(stub, p, 0)) for p in range(5)]
        # Rows past end_seq are invalid and rows with no accepted token are
        # skipped; every other row writes exactly one checkpoint.
        assert written == [1, 0, 1, 1, 0]
        # The masked real sizes are clipped to the query lengths.
        r_size = [int(md.p_id_to_r_size[p, 0]) for p in range(5)]
        assert r_size == [3, 0, 8, 1, 0]

    def test_all_writes_one_checkpoint_per_valid_position(self):
        stub, _ = self._stub(config.StateWrite.ALL, None)
        cls = memory_ref.ExternalStateOutBufferedRef
        written = [int(cls._num_written(stub, p, 0)) for p in range(5)]
        assert written == [8, 3, 8, 1, 0]

    def test_none_never_counts_writes(self):
        stub, _ = self._stub(config.StateWrite.NONE, None)
        with pytest.raises(AssertionError):
            memory_ref.ExternalStateOutBufferedRef._num_written(stub, 0, 0)


class TestFusedConv1dGdnArgumentChecks:
    T = 8

    def _call(self, **over):
        pool, plan = _plan()
        rng = np.random.default_rng(0)

        def arr(shape, dtype=jnp.bfloat16):
            return jnp.asarray(rng.standard_normal(shape), dtype)

        kwargs = dict(
            qkv=arr((self.T, 3 * H * D)),
            b=arr((self.T, H)),
            a=arr((self.T, H * D)),
            conv_state=None,
            recurrent_state=None,
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
            state_source=pool,
            state_plan=plan,
            num_spec_tokens=3,
            batched_state_write=config.StateWrite.NONE,
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

    def test_write_mode_needs_state_source(self):
        with pytest.raises(AssertionError, match="external state source"):
            self._call(
                state_source=None,
                state_plan=None,
                conv_state=jnp.zeros((3, KS - 1, 3 * H * D), jnp.float32),
                recurrent_state=jnp.zeros((3, H, D, D), jnp.float32),
                batched_state_write=config.StateWrite.LAST,
            )


class TestUint32StateViews:
    def test_conv_lane_split_on_mla_pool(self):
        # conv_dim = 768 is not a multiple of the 640-lane pool row, so the conv
        # region is viewed as uint32 with one d_v-wide lane group per row.
        _, plan = _plan()
        conv = plan.conv
        assert conv.lane_split == LANES // D
        assert conv.view_dtype == jnp.uint32
        bf16_rows = (KS - 1) * 3 * H * D // D
        assert conv.rows_used == bf16_rows // config.BF16_PER_U32
        assert plan.recurrent.view_dtype == jnp.float32

    def test_bf16_recurrent_state_halves_view_rows(self):
        pool = jnp.zeros((8, 64, 2048), jnp.bfloat16)
        plan = pool_adapters.v3_state_source(
            pool,
            split=1,
            ssm_ntok=32,
            conv_tok0=32,
            conv_ntok=16,
            conv_dim=6144,
            n_v=4,
            d_k=16,
            d_v=512,
            kernel_size=4,
            recurrent_state_dtype=jnp.bfloat16,
        )
        assert plan.recurrent.view_dtype == jnp.uint32
        assert plan.recurrent.rows_used == 4 * 16 * 512 // 512 // 2
        # Lane-aligned conv (6144 % 2048 == 0) keeps the bf16 view.
        assert plan.conv.view_dtype == jnp.bfloat16
        assert plan.conv.lane_split == 1

    def test_packing_factor(self):
        assert config.BF16_PER_U32 == 2
        assert pool_adapters._bf16_rows_as_u32(54) == 27
        with pytest.raises(AssertionError):
            pool_adapters._bf16_rows_as_u32(55)
