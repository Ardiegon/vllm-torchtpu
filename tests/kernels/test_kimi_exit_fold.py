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
"""CPU tests for the pure-JAX pieces of the Kimi K3 EP-overlap MoE exit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.kimi_k3 import exit_fold


def _pack_meta(ids: np.ndarray, weights: jax.Array) -> jax.Array:
    """Byte-lane packing the MoE layer applies before the dispatch."""
    ids_j = jnp.asarray(ids, jnp.int32)
    low = (ids_j & 0xFF).astype(jnp.bfloat16)
    high = ((ids_j >> 8) & 0xFF).astype(jnp.bfloat16)
    return jnp.concatenate((low, high, weights.astype(jnp.bfloat16)), axis=-1)


def _one_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("dp", "tp"))


def test_plan_is_axial_async_hierarchical():
    for mode in ("c2", "auto", "1", "o4d", "c2b"):
        plan = exit_fold.resolve_effective_exit_fold_plan(mode)
        assert plan == ("o4b", True, "axial")
        assert plan.dispatch_kind == "axial"
    for mode in ("off", "", "o4a", "bogus"):
        with pytest.raises(ValueError):
            exit_fold.resolve_effective_exit_fold_plan(mode)


def test_unpack_meta_columns_round_trips_expert_ids():
    topk = 4
    ids = np.arange(-1, 895, dtype=np.int32)
    ids = ids[: (ids.size // topk) * topk].reshape(-1, topk)
    weights = jnp.linspace(0.0, 1.0, ids.size, dtype=jnp.float32).reshape(ids.shape)
    meta = _pack_meta(ids, weights)
    # Padding columns past 3 * topk are ignored.
    meta = jnp.pad(meta, ((0, 0), (0, 5)))
    got_ids, got_w = exit_fold._unpack_meta_columns(meta, topk=topk)
    np.testing.assert_array_equal(np.asarray(got_ids), ids)
    np.testing.assert_array_equal(
        np.asarray(got_w), np.asarray(weights.astype(jnp.bfloat16))
    )


def test_unpack_axial_gathered_splits_latent_and_meta():
    topk, latent = 2, 8
    ids = np.array([[0, 7], [-1, 300], [895, 1]], np.int32)
    weights = jnp.full(ids.shape, 0.5, jnp.float32)
    latent_rows = jnp.arange(3 * latent, dtype=jnp.float32).reshape(3, latent)
    gathered = jnp.concatenate(
        (latent_rows.astype(jnp.bfloat16), _pack_meta(ids, weights)), axis=-1
    )
    lat, got_ids, got_w = exit_fold._unpack_axial_gathered(
        gathered, latent_dim=latent, topk=topk
    )
    np.testing.assert_array_equal(np.asarray(lat), np.asarray(gathered[:, :latent]))
    np.testing.assert_array_equal(np.asarray(got_ids), ids)
    assert got_w.shape == ids.shape


def test_jax_rms_norm_matches_float_reference():
    x = jax.random.normal(jax.random.key(0), (4, 64), jnp.float32) * 3.0
    w = jax.random.normal(jax.random.key(1), (64,), jnp.float32)
    x_bf, w_bf = x.astype(jnp.bfloat16), w.astype(jnp.bfloat16)
    got = exit_fold._jax_rms_norm(x_bf, w_bf, 1e-6)
    assert got.dtype == jnp.bfloat16
    xf = x_bf.astype(jnp.float32)
    normed = xf / jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
    ref = normed.astype(jnp.bfloat16) * w_bf
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(ref, np.float32), rtol=1e-2, atol=1e-2
    )


def test_hierarchical_combine_single_rank_is_rms_norm():
    mesh = _one_device_mesh()
    moe_out = jax.random.normal(jax.random.key(2), (4, 32), jnp.float32).astype(
        jnp.bfloat16
    )
    norm_w = jnp.linspace(0.5, 1.5, 32, dtype=jnp.float32).astype(jnp.bfloat16)
    tp_rank = jnp.zeros((1, 1), jnp.int32)

    def local(m, nw, r):
        return exit_fold.hierarchical_combine_rms_slice(
            m, nw, r, dp_size=1, tp_size=1, latent_dim=32, use_norm=True, rms_eps=1e-6
        )

    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(None, None), P(None), P("dp", "tp")),
        out_specs=P(None, None),
        check_vma=False,
    )(moe_out, norm_w, tp_rank)
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(exit_fold._jax_rms_norm(moe_out, norm_w, 1e-6))
    )


def test_async_dispatch_single_rank_is_identity():
    mesh = _one_device_mesh()
    x = jnp.arange(4 * 200, dtype=jnp.float32).reshape(4, 200).astype(jnp.bfloat16)

    def local(a):
        state, tok = exit_fold.async_ep_dispatch_start(a, "tp")
        return exit_fold.async_ep_dispatch_wait(
            state, tok, "tp", rows=4, orig_lanes=200
        )

    y = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P(None, None),
        out_specs=P(None, None, None),
        check_vma=False,
    )(x)
    np.testing.assert_array_equal(np.asarray(y[0]), np.asarray(x))
