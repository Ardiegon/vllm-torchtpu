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
"""JAX pieces of the Kimi K3 EP-overlap MoE layer exit.

`layers/adapter/custom_ops/kimi_moe_ep_overlap_op.py` composes these into one
`jax_op` (a `jax.shard_map` over the `('dp', 'tp')` mesh) that runs a whole
Kimi K3 MoE block for expert parallelism:

- `compute_shared_gu_and_act`: the shared-expert gate/up matmul and the Kimi
  `SituAndMul` activation.
- `unpack_meta_columns` / `unpack_axial_gathered`: split the gathered rows
  into the routed latent and the byte-packed routing metadata.
- `hierarchical_combine_rms_slice`: `'dp'` reduce-scatter, `'tp'` psum, the
  exact `RMSNorm(latent_dim)` of the routed output, and the slice of this
  rank's `latent_dim / tp` columns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

# =============================================================================
# Exact JAX RMSNorm and the shared-expert activation (`SituAndMul`)
# =============================================================================


def _jax_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """RMSNorm matching `vllm.ir.ops.layernorm.rms_norm` on TPU (<= 1 bf16 ULP).

    The normalized value is rounded to bf16 before the weight multiply.
    """
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    y_f32 = x_f32 * lax.rsqrt(var + float(eps))
    y_bf16 = lax.reduce_precision(y_f32, exponent_bits=8, mantissa_bits=7).astype(
        jnp.bfloat16
    )
    return (y_bf16 * weight.astype(jnp.bfloat16)).astype(jnp.bfloat16)


def _apply_situ_and_mul(
    x: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    """Apply Kimi `SituAndMul` to `[B_dp, 2 * inter_tp]`."""
    gate, up = jnp.split(x, 2, axis=-1)
    gf = gate.astype(jnp.float32)
    beta = float(situ_beta)
    gate_act = (beta * jnp.tanh(gf / beta) * jax.nn.sigmoid(gf)).astype(jnp.bfloat16)
    if situ_linear_beta is not None:
        uf = up.astype(jnp.float32)
        lbeta = float(situ_linear_beta)
        up_act = (lbeta * jnp.tanh(uf / lbeta)).astype(jnp.bfloat16)
    else:
        up_act = up
    return (gate_act * up_act).astype(jnp.bfloat16)


def compute_shared_gu_and_act(
    local_hidden: jax.Array,
    local_w_shared_gu: jax.Array,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> jax.Array:
    with jax.named_scope("kimi_shared_gu_dot"):
        shared_gu = jnp.dot(
            local_hidden,
            local_w_shared_gu,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
    with jax.named_scope("kimi_shared_gu_act"):
        return _apply_situ_and_mul(
            shared_gu,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )


def unpack_meta_columns(
    meta_2d: jax.Array,
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Unpack byte-valued bf16 metadata `[rows, >= 3 * topk]`.

    Columns `[0, topk)` and `[topk, 2 * topk)` hold the low and high bytes of
    the int16 expert ids (byte values survive bf16 flush-to-zero), columns
    `[2 * topk, 3 * topk)` the routing weights. Returns `(ids, weights)`.
    """
    ag_meta = meta_2d[:, : 3 * topk]
    low_i32 = ag_meta[:, :topk].astype(jnp.int32)
    high_i32 = ag_meta[:, topk : 2 * topk].astype(jnp.int32)
    ag_ids = (((high_i32 << 8) | low_i32) << 16) >> 16
    ag_weights = ag_meta[:, 2 * topk : 3 * topk]
    return ag_ids, ag_weights


def unpack_axial_gathered(
    gathered_dp_2d: jax.Array,
    *,
    latent_dim: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split `[dp * B_dp, latent_dim + 3 * topk]` into `(latent, ids, weights)`."""
    ag_latent = gathered_dp_2d[:, :latent_dim]
    ag_ids, ag_weights = unpack_meta_columns(gathered_dp_2d[:, latent_dim:], topk=topk)
    return ag_latent, ag_ids, ag_weights


# =============================================================================
# Hierarchical combine: DP reduce-scatter + TP psum + RMSNorm + column slice
# =============================================================================


def hierarchical_combine_rms_slice(
    moe_out: jax.Array,
    local_norm_w: jax.Array,
    local_tp_rank: jax.Array,
    *,
    dp_axis_name: str,
    tp_axis_name: str,
    tp_size: int,
    latent_dim: int,
    use_norm: bool,
    rms_eps: float,
) -> jax.Array:
    """Combine the routed expert partial sums of every rank.

    Reduce-scatter of the rows over `dp_axis_name`, psum over
    `tp_axis_name`, exact `RMSNorm(latent_dim)` (when `use_norm`), then this
    rank's `latent_dim / tp_size` columns.
    """
    latent_local = latent_dim // tp_size
    routed_dp = lax.psum_scatter(moe_out, dp_axis_name, scatter_dimension=0, tiled=True)
    routed_full = lax.psum(routed_dp, tp_axis_name)
    if use_norm:
        routed_full = _jax_rms_norm(routed_full, local_norm_w, rms_eps)
    return lax.dynamic_slice_in_dim(
        routed_full,
        local_tp_rank[0, 0] * latent_local,
        latent_local,
        axis=1,
    )
