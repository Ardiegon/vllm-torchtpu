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
"""CPU tests for the shape/tiling helpers of the one-shot DMA collectives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import one_shot_dma as pd


@pytest.fixture(autouse=True)
def _cpu_v6e_tpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer `pltpu.get_tpu_info()` on CPU with the v6e table."""
    if jax.default_backend() != "tpu":
        v6e = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: v6e)


@pytest.mark.parametrize(
    ("axis_name", "normalized"),
    [("tp", "tp"), (("dp",), "dp"), (("dp", "tp"), ("dp", "tp"))],
)
def test_normalize_axis_name(axis_name, normalized):
    assert pd.normalize_axis_name(axis_name) == normalized


def test_normalize_axis_name_rejects_empty():
    with pytest.raises(ValueError):
        pd.normalize_axis_name(())


@pytest.mark.parametrize(
    ("dtype", "rows", "expected"),
    [
        (jnp.float32, 1, (1, 1)),
        (jnp.float32, 5, (1, 8)),
        (jnp.bfloat16, 1, (2, 2)),
        (jnp.bfloat16, 3, (2, 4)),
        (jnp.bfloat16, 64, (2, 8)),
        (jnp.int8, 3, (4, 4)),
        (jnp.int8, 64, (4, 32)),
        (jnp.int4, 3, (8, 8)),
        (jnp.int4, 128, (8, 64)),
    ],
)
def test_sublane_packing_params(dtype, rows, expected):
    x = jax.ShapeDtypeStruct((4, rows, 128), dtype)
    assert pd._sublane_packing_params(x) == expected


def test_retile_round_trip():
    x = jnp.arange(2 * 8 * 256, dtype=jnp.float32).reshape(2, 8, 256)
    x = x.astype(jnp.bfloat16)
    assert pd.can_retile(x)
    tiled = pd.retile(x)
    assert tiled.shape == (2, 8, 2, 128)
    np.testing.assert_array_equal(pd.unretile(tiled), x)


@pytest.mark.parametrize(
    ("shape", "axis_size", "problem"),
    [
        ((4, 8, 128), 4, False),
        ((4, 8, 128), 1, False),
        ((4, 8, 100), 4, True),  # lanes not a multiple of 128
        ((16, 4, 256), 4, False),
    ],
)
def test_tiling_problem_and_check(shape, axis_size, problem):
    assert (pd._tiling_problem(shape, axis_size) is not None) == problem
    if problem:
        with pytest.raises(pd.BadTilingException):
            pd._check_tiling(shape, axis_size)
        with pytest.raises(ValueError):
            pd._check_tiling(shape, axis_size)
    else:
        pd._check_tiling(shape, axis_size)


def test_needs_retile():
    bf16_rows3 = jax.ShapeDtypeStruct((4, 3, 256), jnp.bfloat16)
    assert pd.needs_retile(bf16_rows3)
    f32_rows8 = jax.ShapeDtypeStruct((4, 8, 128), jnp.float32)
    assert not pd.needs_retile(f32_rows8)
    assert not pd.needs_retile(f32_rows8, axis_size=4)
    f32_odd_lanes = jax.ShapeDtypeStruct((4, 8, 100), jnp.float32)
    assert pd.needs_retile(f32_odd_lanes, axis_size=4)
    assert not pd.needs_retile(jax.ShapeDtypeStruct((16,), jnp.float32))


@pytest.mark.parametrize(
    ("axis_size", "rows", "lanes", "dtype"),
    [
        (1, 1, 64, jnp.float32),
        (1, 4, 3632, jnp.bfloat16),
        (4, 3, 224, jnp.bfloat16),
        (2, 8, 256, jnp.float32),
        (16, 2, 448, jnp.bfloat16),
    ],
)
def test_seam_pieces_round_trip(axis_size, rows, lanes, dtype):
    rng = np.random.default_rng(axis_size * 100 + rows * 10 + lanes)
    x = jnp.asarray(rng.standard_normal((axis_size, rows, lanes)), dtype)
    packed = pd.seam_pieces_pack(x, axis_size)
    assert packed.shape[-1] % 128 == 0
    np.testing.assert_array_equal(pd.seam_pieces_unpack(packed, rows, lanes), x)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("hbm", 1, 128, 16, jnp.float32), "hbm"),
        (("auto", 1, 128, 16, jnp.float32), "vmem"),
        (("auto", 1, 224, 16, jnp.float32), "hbm"),
        (("auto", 9, 128, 4, jnp.float32), "hbm"),
        (("auto", 8, 7168, 16, jnp.bfloat16), "vmem"),
        (("auto", 8, 7168, 16, jnp.float32), "hbm"),
    ],
)
def test_resolve_recv_memory_space(args, expected):
    assert pd._resolve_recv_memory_space(*args) == expected


def test_resolve_recv_memory_space_rejects_unknown():
    with pytest.raises(ValueError):
        pd._resolve_recv_memory_space("smem", 1, 128, 2, jnp.float32)


@pytest.mark.parametrize(
    ("rows", "total_rows", "cap", "expected"),
    [
        (8, 128, 64, True),
        (64, 1024, 64, True),
        (65, 130, 64, False),
        (72, 1152, 4096, False),
    ],
)
def test_use_one_shot(rows, total_rows, cap, expected):
    assert pd._use_one_shot(rows, total_rows, cap) is expected


@pytest.mark.parametrize(
    ("rows", "expect_custom_call"),
    [(8, True), (16, False)],
)
def test_guarded_psum_switches_between_one_shot_and_lax_psum(rows, expect_custom_call):
    from jax.sharding import Mesh, PartitionSpec
    from torch_tpu._internal.pallas import pallas as pallas_impl

    tp_size = 16
    devices = [
        pallas_impl._PallasDevice(id=i, process_index=i, device_kind="TPU v6 lite")
        for i in range(tp_size)
    ]
    mesh = Mesh(np.asarray(devices, dtype=object), ("tp",))

    def step(x):
        return jax.shard_map(
            lambda s: pd.guarded_psum(
                s,
                "tp",
                collective_id=3,
                recv_memory_space="auto",
                max_dma_tokens=8,
                vmem_max_tokens=64,
            ),
            mesh=mesh,
            in_specs=(PartitionSpec("tp", None),),
            out_specs=PartitionSpec("tp", None),
            check_vma=False,
        )(x)

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        hlo = str(
            jax.jit(step)
            .lower(jax.ShapeDtypeStruct((tp_size * rows, 256), jnp.bfloat16))
            .compiler_ir("stablehlo")
        )
    assert ("@tpu_custom_call" in hlo) is expect_custom_call
    assert ("stablehlo.all_reduce" in hlo) is (not expect_custom_call)
