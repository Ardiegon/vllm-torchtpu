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
"""moe_gmm dispatch of planar_t MXFP4 weights to the v6e kernel."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.layers.core import fused_moe_gmm

E, K, N, M = 2, 512, 512, 4


def _operands(bias=False):
    x = jnp.ones((M, K), jnp.bfloat16)
    w1 = jnp.zeros((E, K, N), jnp.uint8)
    w1_scale = jnp.full((E, 2 * K // 32, 1, N), 127, jnp.uint8)
    w2 = jnp.zeros((E, N // 2, K), jnp.uint8)
    w2_scale = jnp.full((E, N // 32, 1, K), 127, jnp.uint8)
    b1 = jnp.zeros((E, N), jnp.bfloat16) if bias else None
    return x, w1, w1_scale, b1, w2, w2_scale, None


def _run(operands):
    return fused_moe_gmm.moe_gmm(
        *operands,
        jnp.asarray([M, 0], jnp.int32),
        jnp.arange(M, dtype=jnp.int32),
        jnp.ones((M,), jnp.bfloat16),
        jnp.ones((M,), jnp.bool_),
        jnp.arange(M, dtype=jnp.int32),
        activation="situ:4.0:25.0",
        num_tokens=M,
        topk=1,
        use_ep=False,
        use_sparse_core=False,
    )


def test_planar_t_layout_dispatches_to_v6e_kernel(monkeypatch):
    calls = []

    def spy(x, w1, w1_scale, w2, w2_scale, group_sizes, **kwargs):
        calls.append(kwargs)
        return jnp.full((x.shape[0], K), 2.0, x.dtype)

    monkeypatch.setattr(fused_moe_gmm, "moe_gmm_v6e_planar_t_b11", spy)
    out = _run(_operands())
    assert calls == [
        {"activation": "situ:4.0:25.0", "preferred_element_type": jnp.bfloat16}
    ]
    assert out.shape == (M, K)
    assert bool(jnp.all(out == 2.0))


def test_planar_t_layout_rejects_expert_biases(monkeypatch):
    monkeypatch.setattr(
        fused_moe_gmm,
        "moe_gmm_v6e_planar_t_b11",
        lambda *a, **k: pytest.fail("kernel must not run with biases"),
    )
    with pytest.raises(NotImplementedError, match="biases"):
        _run(_operands(bias=True))


@pytest.mark.parametrize(
    ("max_tokens", "expect_fused_decode"),
    [(64, False), (256, True)],
)
def test_fused_moe_func_forwards_fused_decode_max_tokens(
    monkeypatch, max_tokens, expect_fused_decode
):
    """Verify batches (M=128) take the fused decode kernel only when opted in."""
    m = 128
    seen_max_tokens = []
    fused_calls = []
    grouped_calls = []

    def fake_can_use(*args, max_tokens, **kwargs):
        seen_max_tokens.append(max_tokens)
        return m <= max_tokens

    monkeypatch.setattr(
        fused_moe_gmm, "can_use_fused_decode_moe_v6e_planar_t_b11", fake_can_use
    )
    monkeypatch.setattr(
        fused_moe_gmm,
        "fused_decode_moe_v6e_planar_t_b11",
        lambda x, *a, **k: (fused_calls.append(x.shape[0]), jnp.ones_like(x))[1],
    )
    monkeypatch.setattr(
        fused_moe_gmm,
        "moe_gmm",
        lambda x, *a, **k: (grouped_calls.append(x.shape[0]), jnp.zeros_like(x))[1],
    )

    _, w1, w1_scale, _, w2, w2_scale, _ = _operands()
    out = fused_moe_gmm.fused_moe_func(
        jnp.ones((m, K), jnp.bfloat16),
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        jnp.ones((m, 1), jnp.bfloat16),
        jnp.zeros((m, 1), jnp.int32),
        topk=1,
        activation="situ:4.0:25.0",
        fused_decode_max_tokens=max_tokens,
    )
    assert seen_max_tokens == [max_tokens]
    assert bool(fused_calls) is expect_fused_decode
    assert bool(grouped_calls) is (not expect_fused_decode)
    assert out.shape == (m, K)


@pytest.mark.parametrize(
    ("chip_name", "requant_block", "expect_planar_t"),
    [
        ("TPU_V6E", None, True),
        ("TPU_7X", None, False),
        ("TPU_V6E", 128, False),
    ],
)
def test_mxfp4_loader_selects_planar_t_only_for_unrequantized_v6e(
    monkeypatch, chip_name, requant_block, expect_planar_t
):
    from types import SimpleNamespace

    import torch
    from jax.experimental.pallas import tpu as pltpu
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w4an_mxfp4 as mxfp4_mod,
    )

    chip = getattr(pltpu.ChipVersion, chip_name)
    monkeypatch.setattr(mxfp4_mod, "get_chip_version", lambda: chip)
    monkeypatch.setattr(mxfp4_mod.envs, "MOE_REQUANTIZE_BLOCK_SIZE", requant_block)
    monkeypatch.setattr(mxfp4_mod, "fresh_buffer", lambda t: t)

    planar_calls = []
    monkeypatch.setattr(
        mxfp4_mod,
        "load_w13_planar_t_mxfp4",
        lambda w, s: (planar_calls.append("w13"), (w, s.unsqueeze(2)))[1],
    )
    monkeypatch.setattr(
        mxfp4_mod,
        "load_w2_planar_t_mxfp4",
        lambda w, s: (planar_calls.append("w2"), (w, s.unsqueeze(2)))[1],
    )
    monkeypatch.setattr(
        mxfp4_mod,
        "requant_load_kmajor_fp4",
        lambda w, s, b: (w, s.unsqueeze(2)),
    )
    monkeypatch.setattr(mxfp4_mod, "load_kmajor_fp4", lambda w: w)
    monkeypatch.setattr(mxfp4_mod, "synchronize_tensors", lambda ts: None)
    monkeypatch.setattr(mxfp4_mod, "release_memory_to_os", lambda: None)
    monkeypatch.setattr(mxfp4_mod, "prebuild_fused_moe_kernel", lambda **k: None)

    method = mxfp4_mod.VllmCompressedTensorsW4ANMxfp4MoEMethod.__new__(
        mxfp4_mod.VllmCompressedTensorsW4ANMxfp4MoEMethod
    )
    method.group_size = 32
    method.moe = SimpleNamespace(
        intermediate_size_per_partition=N,
        intermediate_size_per_partition_unpadded=N,
        ep_rank=0,
        num_local_experts=E,
    )
    method._initialize_dummy_quantized_weights = lambda _layer: None
    method._loaded_data = lambda t: SimpleNamespace(to=lambda _dev: t, zero_=lambda: t)
    method._set_tpu_activation = lambda _layer: "situ:4.0:25.0"

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        experts_per_token=1,
    )
    layer.w13_weight_packed = torch.zeros((E, 2 * N, K // 2), dtype=torch.uint8)
    layer.w13_weight_scale = torch.full((E, 2 * N, K // 32), 127, dtype=torch.uint8)
    layer.w2_weight_packed = torch.zeros((E, K, N // 2), dtype=torch.uint8)
    layer.w2_weight_scale = torch.full((E, K, N // 32), 127, dtype=torch.uint8)

    method.process_weights_after_loading(layer)

    assert (planar_calls == ["w13", "w2"]) is expect_planar_t
    if expect_planar_t:
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w13_weight_scale.dtype == torch.uint8
