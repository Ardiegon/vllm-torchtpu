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
"""CPU tests for the JAX and Pallas pieces of the Kimi K3 EP-overlap MoE exit."""

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


@pytest.mark.parametrize("linear_beta", [25.0, None])
def test_compute_shared_gu_and_act_matches_reference(linear_beta):
    hidden = jax.random.normal(jax.random.key(3), (5, 16), jnp.float32).astype(
        jnp.bfloat16
    )
    w_gu = jax.random.normal(jax.random.key(4), (16, 12), jnp.float32).astype(
        jnp.bfloat16
    )
    got = exit_fold.compute_shared_gu_and_act(
        hidden,
        w_gu,
        situ_beta=4.0,
        situ_linear_beta=linear_beta,
    )
    assert got.shape == (5, 6)
    assert got.dtype == jnp.bfloat16
    gate, up = jnp.split(jnp.matmul(hidden, w_gu), 2, axis=-1)
    gf = gate.astype(jnp.float32)
    gate_act = (4.0 * jnp.tanh(gf / 4.0) * jax.nn.sigmoid(gf)).astype(jnp.bfloat16)
    up_act = (
        (linear_beta * jnp.tanh(up.astype(jnp.float32) / linear_beta)).astype(
            jnp.bfloat16
        )
        if linear_beta is not None
        else up
    )
    ref = (gate_act * up_act).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


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
    """Verify both the XLA large-batch branch and the one-shot DMA branch."""
    dp_size, tp_size, b_dp, latent_dim = 2, 16, 4, 32
    partials = jax.random.normal(
        jax.random.key(5), (dp_size, tp_size, dp_size * b_dp, latent_dim), jnp.float32
    ).astype(jnp.bfloat16)
    norm_w = jnp.linspace(0.5, 1.5, latent_dim, dtype=jnp.float32).astype(jnp.bfloat16)
    max_dma_tokens = 2 if above_dma_cap else 64
    seen_tp_caps: list[int] = []

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
                "latency_psum_scatter_rows",
                lambda x, *a, _r=row_sum, **k: _r[0],
            )
            monkeypatch.setattr(
                exit_fold,
                "guarded_psum",
                lambda x, *a, _f=full_sum, **k: (
                    seen_tp_caps.append(k["max_dma_tokens"]) or _f
                ),
            )
        for tp_r in range(tp_size):
            got = exit_fold.hierarchical_combine_rms_slice(
                partials[dp_r, tp_r],
                norm_w,
                jnp.array([[tp_r]], jnp.int32),
                dp_axis_name="dp",
                tp_axis_name="tp",
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

    if not above_dma_cap:
        assert seen_tp_caps and set(seen_tp_caps) == {8}


def _ref_ragged_all_to_all(
    operand: jax.Array,
    output: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    axis_name: str,
) -> jax.Array:
    """Exact JAX reference for `lax.ragged_all_to_all` over a vmapped axis."""
    world = jax.lax.axis_size(axis_name)
    my_rank = jax.lax.axis_index(axis_name)
    num_updates = input_offsets.shape[0] // world

    all_op = jax.lax.all_gather(operand, axis_name, axis=0)
    all_in_off = jax.lax.all_gather(
        input_offsets.reshape(world, num_updates), axis_name, axis=0
    )
    all_send_sz = jax.lax.all_gather(
        send_sizes.reshape(world, num_updates), axis_name, axis=0
    )
    all_out_off = jax.lax.all_gather(
        output_offsets.reshape(world, num_updates), axis_name, axis=0
    )

    out_cap = output.shape[0]
    row_idx = jnp.arange(out_cap, dtype=jnp.int32)
    res = output
    for src in range(world):
        for u in range(num_updates):
            s_off = all_in_off[src, my_rank, u]
            sz = all_send_sz[src, my_rank, u]
            d_off = all_out_off[src, my_rank, u]
            rel = row_idx - d_off
            mask = (rel >= 0) & (rel < sz)
            src_idx = jnp.clip(s_off + jnp.where(mask, rel, 0), 0, operand.shape[0] - 1)
            res = jnp.where(mask[:, None], all_op[src, src_idx], res)
    return res


def _ref_expert_gmm(
    lhs: jax.Array,
    w13: jax.Array,
    _w13_scale: jax.Array,
    _w2: jax.Array,
    _w2_scale: jax.Array,
    group_sizes: jax.Array,
    *,
    activation: str,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    ep_shards: int = 1,
) -> jax.Array:
    """Reference grouped linear transform using `w13[:, :, 0]` per-expert diagonal."""
    del activation, ep_shards
    out_dtype = lhs.dtype if preferred_element_type is None else preferred_element_type
    m = lhs.shape[0]
    e_local = w13.shape[0]
    # w13 carries per-expert scale vectors in [e_local, hidden] (stored as f32/bf16).
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    row_idx = jnp.arange(m, dtype=jnp.int32)
    out = jnp.zeros_like(lhs, dtype=jnp.float32)
    for e in range(e_local):
        mask = (row_idx >= starts[e]) & (row_idx < ends[e])
        out = jnp.where(
            mask[:, None],
            lhs.astype(jnp.float32) * w13[e].astype(jnp.float32)[None, :],
            out,
        )
    return out.astype(out_dtype)


def test_compact_ep_capacity_shapes_and_alignment():
    # v6e-32 (DP=2, TP=16, E_local=7, topk=16)
    assert exit_fold.compact_ep_capacity(
        b_dp=256, dp_size=2, tp_size=16, e_local=7, topk=16
    ) == (3584, 3584)
    assert exit_fold.compact_ep_capacity(
        b_dp=512, dp_size=2, tp_size=16, e_local=7, topk=16
    ) == (4096, 4096)
    assert exit_fold.compact_ep_capacity(
        b_dp=1024, dp_size=2, tp_size=16, e_local=7, topk=16
    ) == (4096, 8192)
    # v6e-128 (DP=8, TP=16, E_local=7, topk=16 -> E=896)
    assert exit_fold.compact_ep_capacity(
        b_dp=256, dp_size=8, tp_size=16, e_local=7, topk=16
    ) == (4096, 4096)
    assert exit_fold.compact_ep_capacity(
        b_dp=512, dp_size=8, tp_size=16, e_local=7, topk=16
    ) == (4096, 4096)
    assert exit_fold.compact_ep_capacity(
        b_dp=1024, dp_size=8, tp_size=16, e_local=7, topk=16
    ) == (4096, 8192)
    # v6e-128 (DP=8, TP=16, E_local=3, topk=16)
    assert exit_fold.compact_ep_capacity(
        b_dp=64, dp_size=8, tp_size=16, e_local=3, topk=16
    ) == (1024, 1536)
    assert exit_fold.compact_ep_capacity(
        b_dp=1024, dp_size=8, tp_size=16, e_local=3, topk=16
    ) == (4096, 8192)
    # Every large-batch capacity is 512-aligned for planar_t GMM tiling.
    for b_dp in (256, 512, 1024):
        sc, rc = exit_fold.compact_ep_capacity(
            b_dp=b_dp, dp_size=8, tp_size=16, e_local=7, topk=16
        )
        assert sc % 512 == 0 and rc % 512 == 0


@pytest.mark.parametrize("scenario", ["uniform", "half_dp_padded", "skewed_hot_expert"])
@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize("use_counting_sort", [True, False])
def test_ragged_dp_dispatch_combine_matches_dense_reference(
    scenario, use_norm, use_counting_sort
):
    """Numerical equivalence of ragged DP EP dispatch/combine vs dense baseline."""
    from vllm_torchtpu.layers.core.fused_moe_gmm import _counting_sort_positions

    dp_size, tp_size, e_local, topk = 2, 4, 3, 4
    e_global = dp_size * tp_size * e_local
    b_dp, latent_dim = 16, 32
    rng = np.random.default_rng(123 if scenario == "uniform" else 456)

    lat = jnp.array(
        rng.normal(scale=0.5, size=(dp_size, b_dp, latent_dim)), dtype=jnp.bfloat16
    )
    expert_scales = jnp.array(
        rng.uniform(0.5, 1.5, size=(e_global, latent_dim)), dtype=jnp.bfloat16
    )
    norm_w = jnp.array(rng.uniform(0.5, 1.5, size=(latent_dim,)), dtype=jnp.bfloat16)

    if scenario == "skewed_hot_expert":
        # Route every token to experts owned by (dp=0, tp=1) plus random others.
        hot = np.array([e_local, e_local + 1], dtype=np.int32)
        rows = []
        for _ in range(dp_size * b_dp):
            other = rng.choice(
                [e for e in range(e_global) if e not in hot],
                size=topk - 2,
                replace=False,
            )
            rows.append(np.concatenate([hot, other]))
        ids_np = np.stack(rows).reshape(dp_size, b_dp, topk).astype(np.int32)
    else:
        ids_np = (
            np.stack(
                [
                    rng.choice(e_global, size=topk, replace=False)
                    for _ in range(dp_size * b_dp)
                ]
            )
            .reshape(dp_size, b_dp, topk)
            .astype(np.int32)
        )
    weights = jnp.array(
        rng.uniform(0.1, 0.9, size=(dp_size, b_dp, topk)), dtype=jnp.bfloat16
    )
    if scenario == "half_dp_padded":
        # Entire DP rank 1 padded (weights=0) + tail of DP rank 0 padded.
        weights = weights.at[1].set(jnp.bfloat16(0.0))
        weights = weights.at[0, -4:].set(jnp.bfloat16(0.0))

    packed_meta = jnp.stack(
        [_pack_meta(ids_np[d], weights[d]) for d in range(dp_size)], axis=0
    )
    w_sharded = expert_scales.reshape(dp_size, tp_size, e_local, latent_dim)
    tp_ranks = jnp.broadcast_to(
        jnp.arange(tp_size, dtype=jnp.int32)[None, :, None, None],
        (dp_size, tp_size, 1, 1),
    )
    lat_2d = jnp.broadcast_to(lat[:, None, :, :], (dp_size, tp_size, b_dp, latent_dim))
    meta_2d = jnp.broadcast_to(
        packed_meta[:, None, :, :], (dp_size, tp_size, b_dp, 3 * topk)
    )

    cs_fn = _counting_sort_positions if use_counting_sort else (lambda *a: (None, None))

    def _per_rank(lat_r, meta_r, w_r, tp_r):
        return exit_fold.ragged_dp_dispatch_combine_rms_slice(
            lat_r,
            meta_r,
            w_r,
            w_r,
            w_r,
            w_r,
            norm_w,
            tp_r,
            dp_axis_name="dp",
            tp_axis_name="tp",
            dp_size=dp_size,
            tp_size=tp_size,
            topk=topk,
            routed_activation="silu",
            use_norm=use_norm,
            rms_eps=1e-6,
            use_sparse_core=False,
            counting_sort_fn=cs_fn,
            ragged_gather_fn=lambda x, idx, s, e: x[idx],
            ragged_gather_reduce_fn=None,
            moe_gmm_fn=_ref_expert_gmm,
            ragged_all_to_all_fn=_ref_ragged_all_to_all,
            send_cap=b_dp * topk,
            recv_cap=dp_size * b_dp * e_local,
        )

    got = jax.vmap(
        jax.vmap(_per_rank, axis_name="tp"),
        axis_name="dp",
    )(lat_2d, meta_2d, w_sharded, tp_ranks)
    # Concatenate TP slices -> [dp_size, b_dp, latent_dim]
    got_full = jnp.concatenate([got[:, t, :, :] for t in range(tp_size)], axis=-1)

    # Exact dense mathematical reference
    ids_j = jnp.asarray(ids_np, jnp.int32)
    exp_out = (
        lat[:, :, None, :].astype(jnp.float32)
        * expert_scales[ids_j].astype(jnp.float32)
    ).astype(jnp.bfloat16)
    ref = (
        (exp_out.astype(jnp.float32) * weights[:, :, :, None].astype(jnp.float32))
        .sum(axis=2)
        .astype(jnp.bfloat16)
    )
    if use_norm:
        ref = exit_fold._jax_rms_norm(ref, norm_w, 1e-6)

    np.testing.assert_allclose(
        np.asarray(got_full, np.float32),
        np.asarray(ref, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_ragged_gather_reduce_v2_allow_fallback_false_safe_without_sparse_core(
    monkeypatch,
):
    """allow_fallback=False still falls back cleanly when SparseCore is absent."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
        wrapper as rgr_wrapper,
    )

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V5E, 1),
    )

    x = jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8).astype(jnp.bfloat16)
    indices = jnp.array([0, 3, 5, 1, 2, 4, 6, 7], dtype=jnp.int32)
    weights = jnp.array([0.5, 0.25, 0.25, 0.0, 1.0, 0.5, 0.0, 0.5], dtype=jnp.bfloat16)
    valid = jnp.array([True, True, True, False, True, True, False, True])
    got = rgr_wrapper.ragged_gather_reduce_v2(
        x, indices, weights, valid, reduce_group_size=4, allow_fallback=False
    )
    ref = rgr_wrapper.ragged_gather_reduce_v2(
        x, indices, weights, valid, reduce_group_size=4, allow_fallback=True
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
