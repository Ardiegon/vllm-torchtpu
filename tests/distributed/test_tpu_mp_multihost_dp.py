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
"""`--nnodes` + data parallelism must export the TorchTPU DP env.

With `--nnodes N --data-parallel-size D` (D > 1, each replica's TP group
spanning hosts), TpuPlatform.check_and_update_config takes the `--nnodes`
branch rather than the DP branch, but TPUWorker.init_device still needs
TORCH_TPU_DP_SIZE / TORCH_TPU_DP_MASTER_ADDR / TORCH_TPU_DP_MASTER_PORT
whenever dp_size > 1, with the same values on every host.
"""

import copy
import os
from unittest import mock

import pytest
from vllm.config import ParallelConfig, VllmConfig

from vllm_torchtpu.distributed import tpu_mp_multihost as mp
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

MASTER_ADDR = "10.0.0.1"
MASTER_PORT = 29500
DP_ENV = ("TORCH_TPU_DP_SIZE", "TORCH_TPU_DP_MASTER_ADDR", "TORCH_TPU_DP_MASTER_PORT")
EP_DEVICE_CONTROL_PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
# Env the multi-host bootstrap exports besides the TORCH_TPU_* variables.
_BOOTSTRAP_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "TPU_NUM_HOSTS", "NODE_RANK")
# Plus what a worker process or the test launcher may have set.
_WORKER_ENV_KEYS = _BOOTSTRAP_ENV_KEYS + (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "TPU_LOCAL_RANK_OFFSET",
    "TPU_MULTIHOST_BACKEND",
)


def _is_torch_tpu_env(key: str) -> bool:
    return key.startswith(("TORCH_TPU_", "_TORCH_TPU_"))


# (nnodes, chips per host, dp, tp): DP8xTP16 on v6e-128, DP2xTP16 on v6e-32, and
# the 4 "hosts" x 2 chips v6e-8 emulation.
SHAPES = [(32, 4, 8, 16), (8, 4, 2, 16), (4, 2, 2, 4)]
SHAPE_IDS = ["v6e128_dp8_tp16", "v6e32_dp2_tp16", "v6e8emul_dp2_tp4"]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in list(os.environ):
        if _is_torch_tpu_env(key) or key in _WORKER_ENV_KEYS:
            monkeypatch.delenv(key)
    # check_and_update_config writes os.environ directly; restore afterwards.
    saved = dict(os.environ)
    monkeypatch.setattr(TpuPlatform, "device_control_env_var", "TPU_VISIBLE_CHIPS")
    monkeypatch.setattr(mp, "_keepalive_stores", [])
    yield
    os.environ.clear()
    os.environ.update(saved)


def _nnodes_dp_config(nnodes, chips, dp, tp, node_rank, ep=True):
    # Fields as EngineArgs.create_engine_config infers them for --nnodes with
    # internal DP load balancing (vllm/engine/arg_utils.py, --nnodes block).
    local_world_size = dp * tp // nnodes
    assert local_world_size == chips
    pc = ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        data_parallel_size_local=max(local_world_size // tp, 1),
        data_parallel_rank=(node_rank * local_world_size) // tp,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=MASTER_ADDR,
        master_port=MASTER_PORT,
        data_parallel_master_ip=MASTER_ADDR,
        data_parallel_backend="mp",
        is_moe_model=True,
        enable_expert_parallel=ep,
    )
    host_ips = [f"10.0.{i // 250}.{i % 250 + 1}" for i in range(nnodes)]
    with (
        mock.patch.object(
            mp, "_rendezvous_host_ips", return_value=(host_ips, "session")
        ),
        mock.patch.object(TpuPlatform, "get_device_name", return_value="TPU v6e"),
        # Keep DeviceConfig from building torch.device("tpu") off a TPU.
        mock.patch.object(TpuPlatform, "uses_host_device_handling", return_value=True),
    ):
        vllm_config = VllmConfig(parallel_config=pc)
    assert vllm_config.parallel_config.local_world_size == chips
    return vllm_config


def _clear_bootstrap_env():
    for key in list(os.environ):
        if _is_torch_tpu_env(key) or key in _BOOTSTRAP_ENV_KEYS:
            del os.environ[key]


@pytest.mark.parametrize("nnodes,chips,dp,tp", SHAPES, ids=SHAPE_IDS)
def test_nnodes_dp_exports_torch_tpu_dp_env(nnodes, chips, dp, tp):
    """Every host (DP leaders and headless TP-worker hosts) gets the same DP env."""
    nnodes_within_dp = nnodes // dp
    # Node 0 (head, DP0 leader), node 1 (headless TP workers of DP0) and the
    # last DP rank's leader host.
    nodes = sorted({0, 1, (dp - 1) * nnodes_within_dp})
    seen = []
    for node_rank in nodes:
        _clear_bootstrap_env()
        vllm_config = _nnodes_dp_config(nnodes, chips, dp, tp, node_rank)
        assert vllm_config.parallel_config.distributed_executor_backend.__name__ == (
            "TpuMultiprocExecutor"
        )
        missing = [k for k in DP_ENV if k not in os.environ]
        assert not missing, (
            f"--nnodes {nnodes} + DP{dp}xTP{tp}, node_rank {node_rank}: "
            f"{missing} not exported by TpuPlatform.check_and_update_config"
        )
        assert os.environ["TORCH_TPU_DP_SIZE"] == str(dp)
        assert os.environ["TORCH_TPU_DP_MASTER_ADDR"] == MASTER_ADDR
        assert os.environ["TORCH_TPU_DP_MASTER_PORT"].isdigit()
        seen.append(os.environ["TORCH_TPU_DP_MASTER_PORT"])
        # EP spans every DP*TP worker, as in the single-host DP branch.
        assert TpuPlatform.device_control_env_var == EP_DEVICE_CONTROL_PLACEHOLDER
    # Hosts cannot coordinate a random pick, so the port must be deterministic.
    assert len(set(seen)) == 1, seen


def test_nnodes_dp_keeps_user_exported_dp_master():
    os.environ["TORCH_TPU_DP_MASTER_ADDR"] = "10.9.9.9"
    os.environ["TORCH_TPU_DP_MASTER_PORT"] = "31000"
    _nnodes_dp_config(4, 2, 2, 4, node_rank=1)
    assert os.environ["TORCH_TPU_DP_SIZE"] == "2"
    assert os.environ["TORCH_TPU_DP_MASTER_ADDR"] == "10.9.9.9"
    assert os.environ["TORCH_TPU_DP_MASTER_PORT"] == "31000"


def test_nnodes_dense_dp_keeps_device_control_env_var():
    _nnodes_dp_config(4, 2, 2, 4, node_rank=1, ep=False)
    assert os.environ["TORCH_TPU_DP_SIZE"] == "2"
    assert TpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


def test_nnodes_without_dp_leaves_dp_env_unset():
    vllm_config = _nnodes_dp_config(4, 2, 1, 8, node_rank=1)
    assert vllm_config.parallel_config.distributed_executor_backend.__name__ == (
        "TpuMultiprocExecutor"
    )
    assert not [k for k in DP_ENV if k in os.environ]
    assert TpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


class _ReachedDistInit(Exception):
    """Raised to stop TPUWorker.init_device at vLLM's distributed init."""


def _init_worker_device(vllm_config):
    """Run TPUWorker.init_device for node 1's first worker up to dist init."""
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    pc = vllm_config.parallel_config
    worker = TPUWorker.__new__(TPUWorker)
    worker.parallel_config = copy.deepcopy(pc)
    worker.rank = pc.local_world_size * pc.node_rank_within_dp
    worker.local_rank = 0
    worker.distributed_init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
    worker.vllm_config = vllm_config
    worker.devices = ["tpu:placeholder"]  # skip torch.device("tpu") off a TPU
    # Stop at vLLM's distributed init; everything after it needs a TPU.
    with (
        mock.patch(
            "vllm_torchtpu.worker.tpu_worker.init_distributed_environment",
            side_effect=_ReachedDistInit,
        ) as dist_init,
        pytest.raises(_ReachedDistInit),
    ):
        worker.init_device()
    return worker, dist_init


@pytest.mark.parametrize("nnodes,chips,dp,tp", SHAPES, ids=SHAPE_IDS)
def test_nnodes_dp_worker_init_gets_past_dp_master_port(nnodes, chips, dp, tp):
    """TPUWorker.init_device must not die on the missing DP master port."""
    vllm_config = _nnodes_dp_config(nnodes, chips, dp, tp, node_rank=1)
    worker, dist_init = _init_worker_device(vllm_config)
    kwargs = dist_init.call_args.kwargs
    # EP-DP path: upstream re-adds dp_rank * tp, so init_rank is the TP rank.
    assert kwargs["world_size"] == tp
    assert kwargs["rank"] == worker.rank
    assert worker.parallel_config.data_parallel_master_ip == MASTER_ADDR


def test_worker_without_dp_master_port_raises_clear_error():
    vllm_config = _nnodes_dp_config(4, 2, 2, 4, node_rank=1)
    del os.environ["TORCH_TPU_DP_MASTER_PORT"]
    with pytest.raises(RuntimeError, match="TORCH_TPU_DP_MASTER_PORT"):
        _init_worker_device(vllm_config)
