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
"""CPU tests for the Kimi K3 DMA collective torch bridges (no TPU needed)."""

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.models.vllm.kimi_k3 import dma_collective_ops as dco
from vllm_torchtpu.models.vllm.kimi_k3 import latency_collective_ops as lco


def _mesh_1x1(axis_names=("dp", "tp")):
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names)


def test_sanitize_tag():
    assert dco.sanitize_tag("a-b.c") == "a_b_c"
    assert dco.sanitize_tag("c2b+o4d") == "c2b_o4d"
    assert dco.sanitize_tag("") == "v1"


def test_format_op_name_is_stable():
    # These names key warm compile caches; they must not change.
    name = dco.format_pallas_dma_op_name(
        kind="psum",
        spec_tag="rank_unshared",
        axis_name=("dp", "tp"),
        collective_id=4,
        recv_memory_space="vmem",
        max_dma_tokens=8,
        dp_size=8,
        tp_size=16,
        variant_tag="o2_m1_p1_lc1_w0_tp_off_ef_c2_cp_vmem_chain_sk_o6b",
    )
    assert name == (
        "pallas::kimi_psum_rank_unshared_ax_dp_tp_cid4_rmem_vmem_cap8_dp8_tp16"
        "_o2_m1_p1_lc1_w0_tp_off_ef_c2_cp_vmem_chain_sk_o6b"
    )


def test_mesh_shape_and_ids():
    dp, tp, ids = dco.mesh_shape_and_ids(_mesh_1x1())
    assert (dp, tp) == (1, 1)
    assert len(ids) == 1
    with pytest.raises(ValueError):
        dco.mesh_shape_and_ids(_mesh_1x1(("tp", "dp")))


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        (None, "off"),
        (" OFF ", "off"),
        ("none", "off"),
        ("auto", "tp16"),
        ("routed_only", "routed_only"),
        ("yes", "all"),
        ("064", "64"),
    ],
)
def test_normalize_latency_tp_psum_mode(raw, mode):
    assert lco.normalize_latency_tp_psum_mode(raw) == mode


@pytest.mark.parametrize("raw", ["-1", "0x8", "tp8", "garbage"])
def test_normalize_latency_tp_psum_mode_rejects(raw):
    with pytest.raises(ValueError):
        lco.normalize_latency_tp_psum_mode(raw)


@pytest.mark.parametrize(
    ("mode", "tp", "plan"),
    [
        ("off", 16, (False, False, 64)),
        ("all", 1, (False, False, 64)),
        ("tp16", 8, (False, False, 64)),
        ("tp16", 16, (True, True, 8)),
        ("routed_only", 4, (True, False, 64)),
        ("all", 4, (True, True, 64)),
        ("16", 4, (True, True, 16)),
    ],
)
def test_resolve_tp_psum_plan(mode, tp, plan):
    assert tuple(lco._resolve_tp_psum_plan(mode, tp, 64)) == plan


def test_dp_dispatch_pack_round_trip():
    gen = torch.Generator().manual_seed(0)
    latent = torch.randn(4, 224, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, 896, (4, 8), generator=gen, dtype=torch.int32)
    ids[0, 0] = -1
    ids[1, :4] = torch.tensor([0, 1, 127, 895], dtype=torch.int32)
    weights = torch.rand(4, 8, generator=gen).to(torch.bfloat16)

    packed = lco.pack_dp_dispatch(latent, ids, weights)
    assert packed.dtype == torch.bfloat16
    assert packed.shape == (4, 224 + 3 * 8)
    torch.testing.assert_close(
        packed[:, 224:], lco.pack_dispatch_metadata(ids, weights), rtol=0, atol=0
    )
    got_latent, got_ids, got_weights = lco.unpack_dp_dispatch(packed, 224, 8)
    assert torch.equal(got_latent, latent)
    assert torch.equal(got_ids, ids)
    assert torch.equal(got_weights, weights)


def test_pack_dispatch_metadata_signed_16_bit_range():
    ids = torch.tensor([[-32768, -2, -1, 0, 1, 32767, 256, 255]], dtype=torch.int32)
    meta = lco.pack_dispatch_metadata(ids, torch.ones(1, 8, dtype=torch.bfloat16))
    # Byte lanes stay exact small integers in bf16 (no subnormals, no NaNs).
    assert float(meta[:, :16].min()) >= 0 and float(meta[:, :16].max()) <= 255
    _, got, _ = lco.unpack_dp_dispatch(meta, 0, 8)
    assert torch.equal(got, ids)
