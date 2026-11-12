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
"""Per-head RMS norm followed by a sigmoid output gate (KDA/GDN output stage)."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Upper bound on the rows of one [rows, H * D] block. Each grid step holds
# double-buffered x, gate and output blocks, so at Kimi's per-shard head counts
# (H * D <= 1536 bf16 lanes) a 256-row block keeps them within a few MiB of
# VMEM while still amortizing the per-step overhead over many rows.
_MAX_BLOCK_ROWS = 256


def gated_output_norm(
    output: jax.Array,  # [num_rows, H, D]
    output_gate: jax.Array,  # [num_rows, H * D] raw gate
    norm_weight: jax.Array,  # [D]
    eps: float,
    activation_dtype: jnp.dtype,
) -> jax.Array:
    """Per-head ``o_norm`` followed by the sigmoid output gate."""
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output * norm_weight.astype(jnp.float32)).astype(activation_dtype)
    return output * jax.nn.sigmoid(output_gate.reshape(output.shape))


def _gated_output_norm_2d_kernel(
    x_ref: jax.Ref,
    g_ref: jax.Ref,
    w_ref: jax.Ref,
    o_ref: jax.Ref,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> None:
    w = w_ref[...].astype(jnp.float32)
    for h in range(num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        x = x_ref[:, sl].astype(jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
        y = (x * inv_rms * w).astype(o_ref.dtype)
        o_ref[:, sl] = y * jax.nn.sigmoid(g_ref[:, sl].astype(o_ref.dtype))


def gated_output_norm_2d(
    out_2d: jax.Array,  # [num_rows, H * D]
    output_gate: jax.Array,  # [num_rows, H * D]
    norm_weight: jax.Array,  # [D]
    eps: float,
    activation_dtype: jnp.dtype,
    *,
    num_heads: int,
    head_dim: int,
) -> jax.Array:
    """Sublane-preserving 2D Pallas ``o_norm`` + sigmoid gate.

    Keeps ``num_rows`` on sublanes and ``head_dim`` on lanes so XLA never
    emits ``2D <-> 3D`` sublane relayout transposes before or after
    ``o_norm``. Returns ``[num_rows, H * D]``. A ``head_dim`` that is not a
    lane multiple falls back to the 3D `gated_output_norm`.
    """
    tpu_info = pltpu.get_tpu_info()
    num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
    num_rows, total_dim = out_2d.shape
    if head_dim % num_lanes != 0:
        return gated_output_norm(
            out_2d.reshape(num_rows, num_heads, head_dim),
            output_gate,
            norm_weight,
            eps,
            activation_dtype,
        ).reshape(num_rows, total_dim)

    x_in = out_2d.astype(activation_dtype)
    g_in = output_gate.astype(activation_dtype)
    padded_rows = max(num_sublanes, pl.cdiv(num_rows, num_sublanes) * num_sublanes)
    pad_rows = padded_rows - num_rows
    if pad_rows > 0:
        x_in = jnp.pad(x_in, ((0, pad_rows), (0, 0)))
        g_in = jnp.pad(g_in, ((0, pad_rows), (0, 0)))

    block_rows = math.gcd(padded_rows, _MAX_BLOCK_ROWS)
    row_block = pl.BlockSpec((block_rows, total_dim), lambda i: (i, 0))
    weight_block = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    out_spec = jax.ShapeDtypeStruct((padded_rows, total_dim), activation_dtype)
    res = pl.pallas_call(
        functools.partial(
            _gated_output_norm_2d_kernel,
            eps=eps,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        grid=(padded_rows // block_rows,),
        in_specs=[row_block, row_block, weight_block],
        out_specs=row_block,
        out_shape=out_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        # Kept stable: profiles and compile-cache entries key on this name.
        name=f"kimi_kda_gated_output_norm_2d_h{num_heads}_d{head_dim}",
    )(x_in, g_in, norm_weight.reshape(1, head_dim))
    return res[:num_rows] if pad_rows > 0 else res
