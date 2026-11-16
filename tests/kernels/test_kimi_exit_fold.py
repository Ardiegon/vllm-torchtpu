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


def test_unpack_meta_columns_round_trips_expert_ids():
    topk = 4
    ids = np.arange(-1, 895, dtype=np.int32)
    ids = ids[: (ids.size // topk) * topk].reshape(-1, topk)
    weights = jnp.linspace(0.0, 1.0, ids.size, dtype=jnp.float32).reshape(ids.shape)
    meta = _pack_meta(ids, weights)
    # Padding columns past 3 * topk are ignored.
    meta = jnp.pad(meta, ((0, 0), (0, 5)))
    got_ids, got_w = exit_fold.unpack_meta_columns(meta, topk=topk)
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
    lat, got_ids, got_w = exit_fold.unpack_axial_gathered(
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


@pytest.mark.parametrize(
    ("act_kind", "linear_beta"),
    [("situ", 25.0), ("situ", None), ("silu", None)],
)
def test_compute_shared_gu_and_act_matches_reference(act_kind, linear_beta):
    hidden = jax.random.normal(jax.random.key(3), (5, 16), jnp.float32).astype(
        jnp.bfloat16
    )
    w_gu = jax.random.normal(jax.random.key(4), (16, 12), jnp.float32).astype(
        jnp.bfloat16
    )
    got = exit_fold.compute_shared_gu_and_act(
        hidden,
        w_gu,
        act_kind=act_kind,
        situ_beta=4.0,
        situ_linear_beta=linear_beta,
    )
    assert got.shape == (5, 6)
    assert got.dtype == jnp.bfloat16
    gate, up = jnp.split(jnp.matmul(hidden, w_gu), 2, axis=-1)
    gf = gate.astype(jnp.float32)
    if act_kind == "situ":
        gate_act = (4.0 * jnp.tanh(gf / 4.0) * jax.nn.sigmoid(gf)).astype(jnp.bfloat16)
        up_act = (
            (linear_beta * jnp.tanh(up.astype(jnp.float32) / linear_beta)).astype(
                jnp.bfloat16
            )
            if linear_beta is not None
            else up
        )
    else:
        gate_act = jax.nn.silu(gf).astype(jnp.bfloat16)
        up_act = up
    ref = (gate_act * up_act).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize(
    ("dp_size", "tp_size"),
    [(1, 4), (4, 1), (1, 1)],
)
def test_hierarchical_combine_rejects_single_rank_axis(dp_size, tp_size):
    moe_out = jnp.zeros((4, 32), jnp.bfloat16)
    norm_w = jnp.ones((32,), jnp.bfloat16)
    tp_rank = jnp.zeros((1, 1), jnp.int32)
    with pytest.raises(AssertionError):
        exit_fold.hierarchical_combine_rms_slice(
            moe_out,
            norm_w,
            tp_rank,
            dp_axis_name="dp",
            tp_axis_name="tp",
            dp_size=dp_size,
            tp_size=tp_size,
            latent_dim=32,
            use_norm=True,
            rms_eps=1e-6,
            max_dma_tokens=64,
        )


def test_async_dispatch_start_rejects_single_rank_axis():
    mesh = _one_device_mesh()
    x = jnp.zeros((4, 32), jnp.bfloat16)

    def local(a):
        return exit_fold.async_ep_dispatch_start(
            a, "tp", collective_id=exit_fold.COLLECTIVE_ID_MOE_DP_DISPATCH_AG
        )

    with pytest.raises(AssertionError, match="one rank"):
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(None, None),
            out_specs=P(None, None),
            check_vma=False,
        )(x)


@pytest.mark.parametrize("above_dma_cap", [True, False])
def test_hierarchical_combine_rms_slice_matches_multi_rank_reference(
    monkeypatch, above_dma_cap
):
    """Verify both the XLA prefill branch and the guarded DMA branch."""
    dp_size, tp_size, b_dp, latent_dim = 2, 2, 4, 32
    partials = jax.random.normal(
        jax.random.key(5), (dp_size, tp_size, dp_size * b_dp, latent_dim), jnp.float32
    ).astype(jnp.bfloat16)
    norm_w = jnp.linspace(0.5, 1.5, latent_dim, dtype=jnp.float32).astype(jnp.bfloat16)
    max_dma_tokens = 2 if above_dma_cap else 64

    for dp_r in range(dp_size):
        row_sum = jnp.sum(partials[:, :, dp_r * b_dp : (dp_r + 1) * b_dp], axis=0)
        full_sum = jnp.sum(row_sum, axis=0)
        normed = exit_fold._jax_rms_norm(full_sum, norm_w, 1e-6)
        if above_dma_cap:
            monkeypatch.setattr(
                exit_fold.lax,
                "psum_scatter",
                lambda x, *a, _r=row_sum, **k: _r[0],
            )
            monkeypatch.setattr(
                exit_fold.lax, "psum", lambda x, *a, _f=full_sum, **k: _f
            )
        else:
            monkeypatch.setattr(
                exit_fold,
                "guarded_psum_scatter_rows",
                lambda x, *a, _r=row_sum, **k: _r[0],
            )
            monkeypatch.setattr(
                exit_fold, "guarded_psum", lambda x, *a, _f=full_sum, **k: _f
            )
        for tp_r in range(tp_size):
            got = exit_fold.hierarchical_combine_rms_slice(
                partials[dp_r, tp_r],
                norm_w,
                jnp.array([[tp_r]], jnp.int32),
                dp_axis_name="dp",
                tp_axis_name="tp",
                dp_size=dp_size,
                tp_size=tp_size,
                latent_dim=latent_dim,
                use_norm=True,
                rms_eps=1e-6,
                max_dma_tokens=max_dma_tokens,
            )
            expected = normed[
                :,
                tp_r * (latent_dim // tp_size) : (tp_r + 1) * (latent_dim // tp_size),
            ]
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
