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
