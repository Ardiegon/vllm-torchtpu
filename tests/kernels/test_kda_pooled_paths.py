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
"""Pooled KDA op layer on CPU: direct granule-view path selection."""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import tpu_info

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kop

H, D, KS = 2, 128, 4
PACK, LANES = 2, 640
LOWER_BOUND, EPS = -5.0, 1e-5
K_SPEC = 7
W = K_SPEC + 1


@pytest.fixture(autouse=True)
def _cpu_v6e_tpu_info():
    """Answer `pltpu.get_tpu_info()` on CPU with the v6e table."""
    tpu_info.get_tpu_info.cache_clear()
    v6e = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)
    with patch.dict(tpu_info.registry, {"cpu": lambda: v6e}):
        yield
    tpu_info.get_tpu_info.cache_clear()


class _RecordingKernel:
    """Stand-in for `fused_conv1d_gdn` that records its bound arguments."""

    def __init__(self, real):
        self.sig = inspect.signature(real)
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        n_v, d_v = call["n_v"], call["d_v"]
        v = call["qkv"][:, -n_v * d_v :].astype(jnp.float32)
        out = v + 0.25 * call["a"][:, : n_v * d_v].astype(jnp.float32)
        return call["states"], out.astype(call["qkv"].dtype)


@pytest.fixture
def kernel(monkeypatch):
    fake = _RecordingKernel(kop.gdn_wrapper.fused_conv1d_gdn)
    monkeypatch.setattr(kop.gdn_wrapper, "fused_conv1d_gdn", fake)
    return fake


def _bf16(rng, shape, scale=0.5):
    return jnp.asarray(rng.standard_normal(shape) * scale, jnp.bfloat16)


def _pool_rows():
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=H * D * D * 4,
        conv_bytes=(KS - 1) * 3 * H * D * 4,
        token_bytes=PACK * LANES * 2,
    )
    return layout, -(-layout.required_tokens // 10) * 10


def _batch(rng, query_lens, pad_rows, num_tokens):
    """Persistent-batch layout [windows / decodes][prefill][padding]."""
    layout, rows = _pool_rows()
    num_reqs = len(query_lens)
    lens = list(query_lens) + [0] * pad_rows
    return dict(
        pool=_bf16(rng, (1 + num_reqs, rows, PACK, LANES), 0.05),
        layout=layout,
        mixed_qkv=_bf16(rng, (num_tokens, 3 * H * D)),
        raw_gate=_bf16(rng, (num_tokens, H * D)),
        beta=_bf16(rng, (num_tokens, H)),
        output_gate=_bf16(rng, (num_tokens, H * D)),
        conv_weight=_bf16(rng, (KS, 3, H, D)),
        a_log=jnp.asarray(rng.standard_normal(H), jnp.float32),
        dt_bias=jnp.asarray(rng.standard_normal(H * D), jnp.float32),
        norm_weight=_bf16(rng, (D,)),
        query_start_loc=jnp.asarray(np.cumsum([0] + lens), jnp.int32),
        state_indices=jnp.asarray(
            list(range(1, num_reqs + 1)) + [0] * pad_rows, jnp.int32
        ),
        seq_lens=jnp.asarray([q + 50 for q in query_lens] + [0] * pad_rows, jnp.int32),
        pool_block_tokens=rows * PACK,
    )


def _core(b, **kw):
    return kop._pooled_kda_core(
        b["mixed_qkv"],
        b["raw_gate"],
        b["beta"],
        b["output_gate"],
        b["pool"],
        b["conv_weight"],
        b["a_log"],
        b["dt_bias"],
        b["norm_weight"],
        b["query_start_loc"],
        b["state_indices"],
        b["seq_lens"],
        kw.pop("distribution"),
        lower_bound=LOWER_BOUND,
        eps=EPS,
        pool_block_tokens=b["pool_block_tokens"],
        **kw,
    )


class TestPooledCorePathSelection:
    def _decode_batch(self):
        b = _batch(np.random.default_rng(1), [1, 1, 1, 9], pad_rows=2, num_tokens=16)
        return b, jnp.asarray([3, 4, 4], jnp.int32)

    def test_single_conv_shard_streams_from_the_pool(self, kernel):
        b, dist = self._decode_batch()
        out, pool = _core(b, distribution=dist)
        (call,) = kernel.calls
        assert call["states"] is b["pool"]
        assert call["shared_conv_granule_offset"] == b["layout"].ssm_tokens * 5
        assert call["recurrent_dtype"] == jnp.dtype(jnp.float32)
        assert call["conv_dtype"] == jnp.dtype(jnp.float32)
        assert call["read_offsets"] is None
        assert call["num_spec_tokens"] == 0
        assert call["batched_state_write"] == gdn_config.StateWrite.ALL
        assert call["conv_weight"].shape == (3 * H * D, 1, KS)
        assert out.shape == (16, H, D)
        assert pool is b["pool"]

    def test_sharded_conv_state_gathers_and_scatters(self, kernel, monkeypatch):
        b, dist = self._decode_batch()
        seen = {}

        def fake_gather_scatter(*args, **kwargs):
            seen["reqs"] = args[9]
            seen["num_spec_tokens"] = kwargs["num_spec_tokens"]
            return "gathered"

        monkeypatch.setattr(kop, "_pooled_kda_gather_scatter", fake_gather_scatter)
        assert _core(b, distribution=dist, conv_shard_heads=1) == "gathered"
        assert not kernel.calls
        assert seen["num_spec_tokens"] == 0
        assert seen["reqs"].state_indices.shape == (6,)

    def test_raiden_admission_from_one_source_rank_is_the_unsharded_layout(
        self, kernel, monkeypatch
    ):
        monkeypatch.setattr(kop.tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
        monkeypatch.setattr(kop.tpu_envs, "TPU_RAIDEN_KIMIK3_ADMISSION", True)
        monkeypatch.setattr(kop.tpu_envs, "TPU_RAIDEN_TRANSFER_PARALLELISM", 1)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": H})
            )
        )
        shard_heads = kop._pooled_conv_shard_heads(vllm_config)
        assert shard_heads == H

        b, dist = self._decode_batch()
        unsharded, one_shard = (
            kop._kda_pool_geometry(
                b["pool"],
                num_heads=H,
                head_dim=D,
                kernel_size=KS,
                pool_block_tokens=b["pool_block_tokens"],
                conv_shard_heads=heads,
            )
            for heads in (None, shard_heads)
        )
        assert one_shard == unsharded
        assert one_shard.conv_shards == 1

        _core(b, distribution=dist, conv_shard_heads=shard_heads)
        (call,) = kernel.calls
        assert call["states"] is b["pool"]
        assert call["shared_conv_granule_offset"] == b["layout"].ssm_tokens * 5

    def test_kimi_k3_tp16_conv_granule_offset_is_770(self):
        # Kimi K3 at TP=16: h=6, d=128 -> ssm_tokens=154, conv_tokens=16, 770 granules.
        pool = jnp.zeros((2, 170, PACK, LANES), jnp.bfloat16)
        geom = kop._kda_pool_geometry(
            pool,
            num_heads=6,
            head_dim=128,
            kernel_size=4,
            pool_block_tokens=340,
            conv_shard_heads=None,
        )
        assert geom.layout.ssm_tokens == 154
        assert geom.layout.conv_tokens == 16
        assert kop._shared_conv_granule_offset(geom) == 770

    def test_request_rows_are_clamped_to_the_token_bucket(self, kernel):
        b = _batch(np.random.default_rng(6), [1, 1], pad_rows=4, num_tokens=2)
        _core(b, distribution=jnp.asarray([2, 6, 6], jnp.int32))
        (call,) = kernel.calls
        assert call["state_indices"].shape == (2,)
        assert call["query_start_loc"].shape == (3,)
        np.testing.assert_array_equal(np.asarray(call["distribution"]), [2, 2, 2])


def test_jit_traces_the_direct_path(kernel):
    b = _batch(np.random.default_rng(5), [1, 1], pad_rows=0, num_tokens=2)
    dist = jnp.asarray([2, 2, 2], jnp.int32)
    out, _ = jax.jit(lambda: _core(b, distribution=dist))()
    assert out.shape == (2, H, D)
