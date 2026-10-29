# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Fused FP32 Block Attention Residual for token-local K3 activations."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _mix_kernel(prefix, history, weight, output, *, eps):
    values = jnp.concatenate((history[...], prefix[...][:, None, :]), axis=1).astype(
        jnp.float32
    )
    inv_norm = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    scores = (
        jnp.sum(values * weight[...].astype(jnp.float32), axis=-1, keepdims=True)
        * inv_norm
    )
    probabilities = jax.nn.softmax(scores, axis=1)
    output[...] = jnp.sum(values * probabilities, axis=1).astype(output.dtype)


def attention_residual(prefix, history, folded_weight, *, eps=1e-5, interpret=False):
    """Mix [tokens,hidden] prefix and [tokens,blocks,hidden] saved blocks.

    Norm and projection weights are folded to one FP32 vector at load time.
    All stored blocks are live; the production model carries their actual
    count in the history shape. No state or cache is modified.
    """
    tokens, hidden = prefix.shape
    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    if history.shape[1] == 0 or tokens == 0:
        return prefix
    if tokens % 16:
        raise ValueError("Fused Attention Residual requires 16-token tiles")
    return pl.pallas_call(
        functools.partial(_mix_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // 16,),
        in_specs=(
            pl.BlockSpec((16, hidden), lambda i: (i, 0)),
            pl.BlockSpec((16, history.shape[1], hidden), lambda i: (i, 0, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((16, hidden), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
        interpret=interpret,
        name="kimi_attnres_compact_m16",
    )(prefix, history, folded_weight)


def _mix_kernel_bt(prefix, weight, *blocks_and_output, eps):
    *blocks, output = blocks_and_output
    p = prefix[...].astype(jnp.float32)
    w = weight[...].astype(jnp.float32)
    bs = [b[...].astype(jnp.float32) for b in blocks] + [p]
    scores = []
    for v in bs:
        inv_norm = jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + eps)
        s = jnp.sum(v * w, axis=-1, keepdims=True) * inv_norm
        scores.append(s)
    max_s = scores[0]
    for s in scores[1:]:
        max_s = jnp.maximum(max_s, s)
    exps = [jnp.exp(s - max_s) for s in scores]
    sum_exp = exps[0]
    for e in exps[1:]:
        sum_exp = sum_exp + e
    inv_sum = 1.0 / sum_exp
    acc = bs[0] * (exps[0] * inv_sum)
    for v, e in zip(bs[1:], exps[1:]):
        acc = acc + v * (e * inv_sum)
    output[...] = acc.astype(output.dtype)


def attention_residual_bt(
    prefix, history_bt, folded_weight, *, eps=1e-5, interpret=False
):
    """Mix [tokens, hidden] prefix and [blocks, tokens, hidden] saved blocks.

    Storing history with `blocks` on the leading axis and loading each block as
    a 2D `(None, bm, hidden)` VMEM tile avoids TPU v6e 8-sublane padding on
    `blocks` when `blocks < 8`.
    """
    tokens, hidden = prefix.shape
    num_blocks = history_bt.shape[0]
    if history_bt.shape[1] != tokens or history_bt.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history_bt shapes disagree")
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    if num_blocks == 0 or tokens == 0:
        return prefix
    if tokens % 16:
        raise ValueError("Fused Attention Residual requires 16-token tiles")
    bm = min(128, tokens)
    if tokens % bm != 0:
        bm = 16
    block_specs = tuple(
        pl.BlockSpec((None, bm, hidden), lambda i, b=b: (b, i, 0))
        for b in range(num_blocks)
    )
    return pl.pallas_call(
        functools.partial(_mix_kernel_bt, eps=eps),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // bm,),
        in_specs=(
            pl.BlockSpec((bm, hidden), lambda i: (i, 0)),
            pl.BlockSpec((hidden,), lambda i: (0,)),
            *block_specs,
        ),
        out_specs=pl.BlockSpec((bm, hidden), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2),
        interpret=interpret,
        name=f"kimi_attnres_bt_m{bm}",
    )(prefix, folded_weight, *tuple(history_bt for _ in range(num_blocks)))

