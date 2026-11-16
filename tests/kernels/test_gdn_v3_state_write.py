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
"""GDN v3 shared-pool write-back modes and granule offset rules (CPU).

Covers the pure-Python/XLA parts of the side-effect-free verify
(`StateWrite.NONE`), the masked commit (`StateWrite.LAST`), explicit
`shared_conv_granule_offset`, and the `fused_conv1d_gdn` argument checks.
"""

import types
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import tpu_info

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
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
