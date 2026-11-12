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
"""Kimi K3 libtpu flag parsing and Tier-2 cache naming in `env_override`.

LIBTPU_INIT_ARGS is hashed verbatim, so the parsers are pinned down to the
flag order. The import-time behaviour (what `env_override` writes into the
environment) runs in a fresh interpreter per case.
"""

import json
import os
import subprocess
import sys

import pytest

from vllm_torchtpu import env_override as eo

SC_AR = "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
SC_AG = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
MSA = "--xla_msa_preferred_overlap_to_async_copy_ratio"
ASYNC = "--xla_tpu_enable_async_collective_fusion"
ASYNC_AG = "--xla_tpu_enable_async_collective_fusion_fuse_all_gather"
CPP = "--xla_msa_max_cross_program_prefetches"


@pytest.mark.parametrize("raw", ["", "0", "off", "OFF", "false", "none", " off "])
def test_off_spellings_add_nothing(raw):
    assert eo._resolve_kimi_libtpu_flags(raw, raw) == ([], {}, False)


@pytest.mark.parametrize("raw", ["auto", "1", "true", "c1", "AUTO"])
def test_overlap_preset(raw):
    flags = eo._resolve_kimi_libtpu_flags(raw, "off")
    assert flags.flags == [(MSA, "12"), (ASYNC, "true"), (ASYNC_AG, "true")]
    assert flags.sc_thresholds == {}
    assert flags.active


def test_overlap_tokens_keep_their_order():
    flags = eo._resolve_kimi_libtpu_flags("cpp_256+async_coll+msa_r8", "off")
    assert flags.flags == [
        (CPP, "256"),
        (ASYNC, "true"),
        (ASYNC_AG, "true"),
        (MSA, "8"),
    ]


def test_a_repeated_flag_keeps_its_first_position_and_last_value():
    flags = eo._resolve_kimi_libtpu_flags("msa_r8+async_coll+msa_r16", "off")
    assert flags.flags == [(MSA, "16"), (ASYNC, "true"), (ASYNC_AG, "true")]


@pytest.mark.parametrize("raw", ["msa_r0", "msa_r32", "bogus", "msa_rx"])
def test_overlap_rejects_bad_tokens(raw):
    with pytest.raises(ValueError, match="TPU_KIMI_OPT_XLA_OVERLAP"):
        eo._resolve_kimi_libtpu_flags(raw, "off")


@pytest.mark.parametrize("raw", ["auto", "c1"])
def test_sc_offload_default_is_off(raw):
    assert eo._resolve_kimi_libtpu_flags("off", raw) == ([], {}, False)


@pytest.mark.parametrize(
    "raw,flags,thresholds",
    [
        (
            "1",
            [
                ("--xla_tpu_enable_sparse_core_collective_offload_all_reduce", "true"),
                ("--xla_tpu_enable_sparse_core_collective_offload_all_gather", "true"),
                ("--xla_tpu_enable_concurrent_sparse_core_offloading", "true"),
            ],
            {SC_AR: 8192, SC_AG: 8192},
        ),
        (
            "full_2048",
            [
                ("--xla_tpu_enable_sparse_core_collective_offload_all_reduce", "true"),
                ("--xla_tpu_enable_sparse_core_collective_offload_all_gather", "true"),
                ("--xla_tpu_enable_concurrent_sparse_core_offloading", "true"),
            ],
            {SC_AR: 2048, SC_AG: 2048},
        ),
        (
            "AR_8192",
            [
                ("--xla_tpu_enable_sparse_core_collective_offload_all_reduce", "true"),
                ("--xla_tpu_enable_concurrent_sparse_core_offloading", "true"),
            ],
            {SC_AR: 8192},
        ),
        ("thresh_4096", [], {SC_AR: 4096, SC_AG: 4096}),
    ],
)
def test_sc_offload_modes(raw, flags, thresholds):
    resolved = eo._resolve_kimi_libtpu_flags("off", raw)
    assert resolved.flags == flags
    assert resolved.sc_thresholds == thresholds
    assert resolved.active


@pytest.mark.parametrize("raw", ["bogus", "ar_", "-5"])
def test_sc_offload_rejects_unknown(raw):
    with pytest.raises(ValueError, match="TPU_KIMI_OPT_SC_OFFLOAD"):
        eo._resolve_kimi_libtpu_flags("off", raw)


def test_overlap_flags_come_before_sc_flags():
    resolved = eo._resolve_kimi_libtpu_flags("async_coll", "ar_8192")
    assert [name for name, _ in resolved.flags] == [
        ASYNC,
        ASYNC_AG,
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce",
        "--xla_tpu_enable_concurrent_sparse_core_offloading",
    ]


# ---------------------------------------------------------------------------
# Import-time environment (fresh interpreter per case).
# ---------------------------------------------------------------------------

_KIMI_ENV = (
    "TPU_KIMI_OPT_SPEC_KDA",
    "TPU_KIMI_OPT_EXIT_FOLD",
    "TPU_KIMI_OPT_COMPUTE_POLISH",
    "TPU_KIMI_OPT_XLA_OVERLAP",
    "TPU_KIMI_OPT_SC_OFFLOAD",
    "TORCH_TPU_TIER2_COMPILATION_CACHE",
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
    "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
    "LIBTPU_INIT_ARGS",
)
_PRODUCTION = {
    "TPU_KIMI_OPT_SPEC_KDA": "o6b",
    "TPU_KIMI_OPT_EXIT_FOLD": "c2",
    "TPU_KIMI_OPT_COMPUTE_POLISH": "c2",
    "TPU_KIMI_OPT_XLA_OVERLAP": "auto",
    "TPU_KIMI_OPT_SC_OFFLOAD": "auto",
}


def _import_env(**overrides):
    env = {k: v for k, v in os.environ.items() if k not in _KIMI_ENV}
    env.update({"TPU_ACCELERATOR_TYPE": "v6e-8", "JAX_PLATFORMS": "cpu"})
    env.update(overrides)
    code = (
        "import json, os; import vllm_torchtpu.env_override; "
        f"print(json.dumps({{k: os.environ.get(k) for k in {_KIMI_ENV!r}}}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_production_libtpu_args():
    env = _import_env(**_PRODUCTION, TORCH_TPU_TIER2_COMPILATION_CACHE="a3")
    assert env["LIBTPU_INIT_ARGS"] == (
        "--xla_tpu_use_dynamic_smem_negotiation=true "
        f"{MSA}=12 {ASYNC}=true {ASYNC_AG}=true "
        f"{SC_AR}=134217728 {SC_AG}=134217728"
    )


def test_a_pinned_tier2_cache_is_never_renamed():
    env = _import_env(**_PRODUCTION, TORCH_TPU_TIER2_COMPILATION_CACHE="a3")
    assert env["TORCH_TPU_TIER2_COMPILATION_CACHE"] == "a3"


def test_an_enabled_tier2_cache_is_namespaced_by_the_kimi_modes(tmp_path):
    env = _import_env(
        **_PRODUCTION, TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=str(tmp_path)
    )
    name = env["TORCH_TPU_TIER2_COMPILATION_CACHE"]
    assert name.startswith("tpu_tier2_o6_o6b_")
    assert env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] == name


def test_invalid_mode_fails_at_import_even_with_a_pinned_cache():
    with pytest.raises(subprocess.CalledProcessError):
        _import_env(
            TPU_KIMI_OPT_EXIT_FOLD="bogus", TORCH_TPU_TIER2_COMPILATION_CACHE="a3"
        )
