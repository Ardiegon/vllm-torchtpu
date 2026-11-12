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
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.mla.v2.kernel import prepare_q_inputs

_V6E = tpu_info.ChipVersion.TPU_V6E
_TPU7X = tpu_info.ChipVersion.TPU_7X

# (chip, actual heads, q dtype) -> padded heads. A chip that packs the dtype
# pads to the packing; TPU v6e does not pack bf16 and pads to the 8-sublane
# tile.
_PADDED_HEADS = [
    (_V6E, 3, jnp.bfloat16, 8),
    (_V6E, 6, jnp.bfloat16, 8),
    (_V6E, 12, jnp.bfloat16, 16),
    (_V6E, 16, jnp.bfloat16, 16),
    (_V6E, 3, jnp.float8_e4m3fn, 4),
    (_V6E, 3, jnp.float32, 3),
    (_TPU7X, 3, jnp.bfloat16, 4),
    (_TPU7X, 6, jnp.bfloat16, 6),
    (_TPU7X, 12, jnp.bfloat16, 12),
    (_TPU7X, 3, jnp.float8_e4m3fn, 4),
    (_TPU7X, 6, jnp.float8_e4m3fn, 8),
    (_TPU7X, 3, jnp.float32, 3),
]


@pytest.mark.parametrize("chip,num_heads,dtype,padded_heads", _PADDED_HEADS)
def test_prepare_q_inputs_pads_heads_to_the_chip_tiling(
    monkeypatch, chip, num_heads, dtype, padded_heads
):
    tpu_info.get_tpu_info.cache_clear()
    monkeypatch.setitem(
        tpu_info.registry, "cpu", lambda: tpu_info.get_tpu_info_for_chip(chip, 1)
    )
    num_tokens, head_dim = 5, 192
    q = (
        jnp.arange(num_tokens * num_heads * head_dim, dtype=jnp.float32)
        .reshape(num_tokens, num_heads, head_dim)
        .astype(dtype)
    )

    padded = prepare_q_inputs(q)
    tpu_info.get_tpu_info.cache_clear()

    assert padded.shape == (num_tokens, padded_heads, 256)
    assert padded.dtype == dtype
    padded_np = np.asarray(padded.astype(jnp.float32))
    np.testing.assert_array_equal(
        padded_np[:, :num_heads, :head_dim], np.asarray(q.astype(jnp.float32))
    )
    assert not padded_np[:, num_heads:, :].any()
    assert not padded_np[:, :, head_dim:].any()
