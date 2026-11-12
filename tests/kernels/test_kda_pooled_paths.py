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
"""Pooled KDA op layer on CPU: path selection, verify/commit.

The fused conv1d + GDN v3 kernel needs a TPU, so the op-layer tests replace
`fused_conv1d_gdn` by a recording fake and check what the op layer hands to
the kernel (write mode, read offsets, conv-weight layout, clamped request
rows) and everything it does around it.
"""

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
    """Stand-in for `fused_conv1d_gdn` that records its bound arguments.

    Returns the source unchanged (pooled mode) and a deterministic output:
    the value lanes of `qkv` plus a quarter of the gate.
    """

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
        return call["state_source"], out.astype(call["qkv"].dtype)


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
        conv_bytes=(KS - 1) * 3 * H * D * 2,
        token_bytes=PACK * LANES * 2,
    )
    return -(-layout.required_tokens // 16) * 16


def _batch(rng, query_lens, pad_rows, num_tokens):
    """Persistent-batch layout [windows / decodes][prefill][padding]."""
    rows = _pool_rows()
    num_reqs = len(query_lens)
    lens = list(query_lens) + [0] * pad_rows
    return dict(
        pool=_bf16(rng, (1 + num_reqs, rows, PACK, LANES), 0.05),
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
        assert call["state_source"] is b["pool"]
        assert call["state_plan"] is not None
        assert call["read_offsets"] is None
        assert call["num_spec_tokens"] == 0
        assert call["batched_state_write"] == gdn_config.StateWrite.ALL
        # [dim, 1, kernel_size] for the kernel.
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
        # 6 request rows fit the 16-token bucket, so nothing is clamped.
        assert seen["reqs"].state_indices.shape == (6,)

    def test_request_rows_are_clamped_to_the_token_bucket(self, kernel):
        # 6 request rows but only 2 token rows: at most 2 requests hold tokens.
        b = _batch(np.random.default_rng(6), [1, 1], pad_rows=4, num_tokens=2)
        _core(b, distribution=jnp.asarray([2, 6, 6], jnp.int32))
        (call,) = kernel.calls
        assert call["state_indices"].shape == (2,)
        assert call["query_start_loc"].shape == (3,)
        np.testing.assert_array_equal(np.asarray(call["distribution"]), [2, 2, 2])


class TestVerifyAndCommit:
    def _verify(self, kernel, query_lens, pad_rows, num_tokens, cache_rows):
        b = _batch(np.random.default_rng(2), query_lens, pad_rows, num_tokens)
        n = len(query_lens)
        wdist = jnp.asarray([n, n, n], jnp.int32)
        caches = (
            jnp.zeros((cache_rows, 3 * H * D), jnp.bfloat16),
            jnp.zeros((cache_rows, H * D), jnp.bfloat16),
            jnp.zeros((cache_rows, H), jnp.bfloat16),
        )
        result = kop.pooled_kda_verify(
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
            wdist,
            wdist,
            *caches,
            lower_bound=LOWER_BOUND,
            eps=EPS,
            pool_block_tokens=b["pool_block_tokens"],
            num_spec_tokens=K_SPEC,
        )
        return b, wdist, result

    def test_verify_is_side_effect_free_and_caches_inputs(self, kernel):
        b, _, (out, pool, cq, cg, cb) = self._verify(
            kernel, [8, 5, 8], pad_rows=2, num_tokens=32, cache_rows=4 * W
        )
        (call,) = kernel.calls
        assert call["batched_state_write"] == gdn_config.StateWrite.NONE
        assert call["num_spec_tokens"] == K_SPEC
        np.testing.assert_array_equal(np.asarray(call["read_offsets"]), 0)
        assert call["ckpt_indices"] is None
        assert call["b"] is b["beta"]
        assert call["conv_weight"].shape == (3 * H * D, 1, KS)
        assert pool is b["pool"]
        assert out.shape == (32, H, D)
        np.testing.assert_array_equal(np.asarray(cq), np.asarray(b["mixed_qkv"]))
        np.testing.assert_array_equal(np.asarray(cg), np.asarray(b["raw_gate"]))
        np.testing.assert_array_equal(np.asarray(cb), np.asarray(b["beta"]))

    def test_verify_output_is_the_gated_norm_of_the_kernel_output(self, kernel):
        b, _, (out, *_rest) = self._verify(
            kernel, [8, 8], pad_rows=0, num_tokens=16, cache_rows=2 * W
        )
        (call,) = kernel.calls
        v = call["qkv"][:, -H * D :].astype(jnp.float32)
        raw = (v + 0.25 * call["a"][:, : H * D].astype(jnp.float32)).astype(
            jnp.bfloat16
        )
        want = kop._gated_output_norm(
            raw.reshape(16, H, D), b["output_gate"], b["norm_weight"], EPS, jnp.bfloat16
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(want))

    def test_verify_requires_spec_tokens(self, kernel):
        b = _batch(np.random.default_rng(3), [1], pad_rows=0, num_tokens=1)
        z = jnp.zeros((1, 1), jnp.bfloat16)
        with pytest.raises(ValueError, match="num_spec_tokens"):
            kop.pooled_kda_verify(
                *(
                    b[k]
                    for k in ("mixed_qkv", "raw_gate", "beta", "output_gate", "pool")
                ),
                *(b[k] for k in ("conv_weight", "a_log", "dt_bias", "norm_weight")),
                b["query_start_loc"],
                b["state_indices"],
                b["seq_lens"],
                jnp.zeros((3,), jnp.int32),
                jnp.zeros((3,), jnp.int32),
                z,
                z,
                z,
                lower_bound=LOWER_BOUND,
                eps=EPS,
                pool_block_tokens=b["pool_block_tokens"],
                num_spec_tokens=0,
            )

    def test_commit_writes_the_last_accepted_position(self, kernel):
        rng = np.random.default_rng(4)
        b = _batch(rng, [8, 8, 8], pad_rows=1, num_tokens=32)
        cache_rows = 3 * W
        caches = (
            _bf16(rng, (cache_rows, 3 * H * D)),
            _bf16(rng, (cache_rows, H * D)),
            _bf16(rng, (cache_rows, H)),
        )
        accepted = jnp.asarray([3, 1, 8, 0], jnp.int32)
        pool = kop.pooled_kda_commit(
            b["pool"],
            *caches,
            b["conv_weight"],
            b["a_log"],
            b["dt_bias"],
            b["query_start_loc"],
            b["state_indices"],
            b["seq_lens"],
            jnp.asarray([3, 3, 3], jnp.int32),
            accepted,
            lower_bound=LOWER_BOUND,
            pool_block_tokens=b["pool_block_tokens"],
            num_spec_tokens=K_SPEC,
        )
        (call,) = kernel.calls
        assert pool is b["pool"]
        assert call["batched_state_write"] == gdn_config.StateWrite.LAST
        assert call["batched_only"]
        assert call["qkv"] is caches[0]
        # The cache holds whole windows of 3 requests; the padded 4th request
        # row is clamped away.
        assert call["state_indices"].shape == (3,)
        np.testing.assert_array_equal(np.asarray(call["window_lens"]), [3, 1, 8])
        np.testing.assert_array_equal(np.asarray(call["read_offsets"]), 0)


class TestVerifyCommitBuilder:
    def test_requires_spec_tokens(self):
        with pytest.raises(ValueError, match="num_spec_tokens"):
            kop.build_pooled_kda_verify_commit_ops(
                "layer",
                lower_bound=LOWER_BOUND,
                eps=EPS,
                vllm_config=None,
                num_spec_tokens=0,
            )

    def test_requires_the_unified_pool(self, monkeypatch):
        monkeypatch.setattr(kop.tpu_envs, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", False)
        monkeypatch.setattr(kop.tpu_envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", False)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architecture="AnyHybridModel")
        )
        assert not kop.supports_deferred_spec_commit(vllm_config)
        with pytest.raises(NotImplementedError, match="unified"):
            kop.build_pooled_kda_verify_commit_ops(
                "layer",
                lower_bound=LOWER_BOUND,
                eps=EPS,
                vllm_config=vllm_config,
                num_spec_tokens=K_SPEC,
            )


def test_jit_traces_the_direct_path(kernel):
    # The pooled core is the body of a jax_op; the path choice is static.
    b = _batch(np.random.default_rng(5), [1, 1], pad_rows=0, num_tokens=2)
    dist = jnp.asarray([2, 2, 2], jnp.int32)
    out, _ = jax.jit(lambda: _core(b, distribution=dist))()
    assert out.shape == (2, H, D)
