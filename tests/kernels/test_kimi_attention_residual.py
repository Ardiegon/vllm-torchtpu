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
"""Compare fused Attention Residual with independent float64 arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.attention_residual import (
    attention_residual,
    attention_residual_bt,
)


def _float64_reference(prefix, history, weight) -> np.ndarray:
    """Attention Residual of a [tokens, blocks, hidden] history in float64."""
    values = np.concatenate(
        (np.asarray(history, np.float64), np.asarray(prefix, np.float64)[:, None]),
        axis=1,
    )
    scores = (values @ np.asarray(weight, np.float64)) / np.sqrt(
        np.mean(values * values, axis=-1) + 1e-5
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (values * probabilities[..., None]).sum(axis=1)


def _inputs(tokens: int, blocks: int, hidden: int = 128):
    rng = np.random.default_rng(96)
    prefix = jnp.asarray(rng.normal(size=(tokens, hidden)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(tokens, blocks, hidden)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.02, size=hidden), jnp.float32)
    return prefix, history, weight


@pytest.mark.parametrize("blocks", [0, 1, 2, 4, 8])
def test_fused_mix_matches_float64(blocks):
    prefix, history, weight = _inputs(16, blocks)
    actual = jax.jit(attention_residual, static_argnames=("interpret",))(
        prefix, history, weight, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        _float64_reference(prefix, history, weight),
        rtol=0.005,
        atol=0.005,
    )


# 128 and 256 tokens exercise the 128-token tile (blocks < 6) and the
# 64-token tile (blocks >= 6); 16 tokens the smallest tile.
@pytest.mark.parametrize("tokens", [16, 128, 256])
@pytest.mark.parametrize("blocks", [0, 1, 2, 4, 6, 8])
def test_fused_mix_blocks_first_matches_float64(tokens, blocks):
    prefix, history, weight = _inputs(tokens, blocks)
    history_bt = jnp.transpose(history, (1, 0, 2))
    actual = jax.jit(attention_residual_bt, static_argnames=("interpret",))(
        prefix, history_bt, weight, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        _float64_reference(prefix, history, weight),
        rtol=0.005,
        atol=0.005,
    )


def test_attention_residual_bt_rejects_bad_shapes():
    prefix, history, weight = _inputs(16, 2)
    history_bt = jnp.transpose(history, (1, 0, 2))
    with pytest.raises(ValueError, match="prefix/history_bt shapes disagree"):
        attention_residual_bt(prefix, history_bt[:, :8], weight, interpret=True)
    with pytest.raises(ValueError, match="one folded weight per hidden feature"):
        attention_residual_bt(prefix, history_bt, weight[:64], interpret=True)
    prefix8, history8, weight8 = _inputs(8, 2)
    with pytest.raises(ValueError, match="16-token tiles"):
        attention_residual_bt(
            prefix8, jnp.transpose(history8, (1, 0, 2)), weight8, interpret=True
        )
