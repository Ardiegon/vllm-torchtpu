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
"""CPU tests for the q-head padding of the MLA v2 kernel inputs."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.v2.kernel import prepare_q_inputs

# (actual heads, q dtype) -> padded heads: at least one 8-sublane tile, and
# 16 rows for 8-bit q.
_PADDED_HEADS = [
    (3, jnp.bfloat16, 8),
    (6, jnp.bfloat16, 8),
    (12, jnp.bfloat16, 16),
    (16, jnp.bfloat16, 16),
    (3, jnp.float32, 8),
    (12, jnp.float32, 16),
    (3, jnp.float8_e4m3fn, 16),
    (6, jnp.float8_e4m3fn, 16),
    (16, jnp.float8_e4m3fn, 16),
    (20, jnp.float8_e4m3fn, 32),
]


@pytest.mark.parametrize("num_heads,dtype,padded_heads", _PADDED_HEADS)
def test_prepare_q_inputs_pads_heads_to_sublane_tile(num_heads, dtype, padded_heads):
    num_tokens, head_dim = 5, 192
    q = (
        jnp.arange(num_tokens * num_heads * head_dim, dtype=jnp.float32)
        .reshape(num_tokens, num_heads, head_dim)
        .astype(dtype)
    )

    padded = prepare_q_inputs(q)

    assert padded.shape == (num_tokens, padded_heads, 256)
    assert padded.dtype == dtype
    padded_np = np.asarray(padded.astype(jnp.float32))
    np.testing.assert_array_equal(
        padded_np[:, :num_heads, :head_dim], np.asarray(q.astype(jnp.float32))
    )
    assert not padded_np[:, num_heads:, :].any()
    assert not padded_np[:, :, head_dim:].any()
