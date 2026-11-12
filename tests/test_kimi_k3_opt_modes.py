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
"""Kimi K3 mode env parsers and the switches built on them."""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kimi_k3_opt_modes import (
    ALL_COMPUTE_POLISH_TOKENS,
    DEFAULT_COMPUTE_POLISH_TOKENS,
    OFF_TOKENS,
    normalize_compute_polish_mode,
    normalize_exit_fold_mode,
    normalize_spec_kda_mode,
)
from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.models.vllm.kimi_k3 import opt_flags

_OFF_SPELLINGS = [None, *sorted(OFF_TOKENS), " Off ", "NONE"]


@pytest.mark.parametrize("raw", _OFF_SPELLINGS)
def test_off_spellings(raw):
    assert normalize_exit_fold_mode(raw) == "off"
    assert normalize_compute_polish_mode(raw) == ("off", frozenset())
    assert normalize_spec_kda_mode(raw) == "off"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("c2", "c2"),
        (" C2 ", "c2"),
        ("auto", "c2"),
        ("1", "c2"),
        ("true", "c2"),
        ("yes", "c2"),
        ("o4d", "o4d"),
        ("c2b", "o4d"),
    ],
)
def test_exit_fold_modes(raw, expected):
    assert normalize_exit_fold_mode(raw) == expected


# The world-dispatch, owner-combine and synchronous-dispatch ablation modes
# were removed and now fail closed like any unknown value.
@pytest.mark.parametrize(
    "raw", ["o4a", "o4b", "o4a_sync", "o4b_sync", "o4d_sync", "o4c", "c3", "o4d+c2"]
)
def test_exit_fold_rejects_unknown(raw):
    with pytest.raises(ValueError, match="TPU_KIMI_OPT_EXIT_FOLD"):
        normalize_exit_fold_mode(raw)


@pytest.mark.parametrize(
    "raw,tokens",
    [
        ("c2", {"vmem_chain"}),
        ("auto", {"vmem_chain"}),
        ("trays4", {"trays"}),
        ("trays+trays4", {"trays"}),
        (" C2 + Trays ", {"vmem_chain", "trays"}),
        ("off+vmem_chain", {"vmem_chain"}),
        ("all", set(ALL_COMPUTE_POLISH_TOKENS)),
        ("all+c2", set(ALL_COMPUTE_POLISH_TOKENS)),
        ("copy_opt+kda_glue", {"copy_opt", "kda_glue"}),
        ("trays+", {"trays"}),
        ("+", set()),
    ],
)
def test_compute_polish_tokens(raw, tokens):
    tag, active = normalize_compute_polish_mode(raw)
    assert active == frozenset(tokens)
    assert tag == ("_".join(sorted(tokens)) if tokens else "off")


def test_compute_polish_preset_is_vmem_chain():
    assert DEFAULT_COMPUTE_POLISH_TOKENS == ("vmem_chain",)
    assert normalize_compute_polish_mode("c2") == ("vmem_chain", {"vmem_chain"})


@pytest.mark.parametrize(
    "raw",
    ["trays2", "trays32", "vmem_src", "kda_cond", "kda_cw_f32", "bogus", "c2+bogus"],
)
def test_compute_polish_rejects_unknown(raw):
    with pytest.raises(ValueError, match="TPU_KIMI_OPT_COMPUTE_POLISH"):
        normalize_compute_polish_mode(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("o6a", "o6a"),
        ("c3", "o6a"),
        ("auto", "o6a"),
        ("1", "o6a"),
        ("true", "o6a"),
        ("yes", "o6a"),
        ("o6b", "o6b"),
        (" O6B ", "o6b"),
    ],
)
def test_spec_kda_modes(raw, expected):
    assert normalize_spec_kda_mode(raw) == expected


@pytest.mark.parametrize("raw", ["o6c", "2", "bogus"])
def test_spec_kda_rejects_unknown(raw):
    with pytest.raises(ValueError, match="TPU_KIMI_OPT_SPEC_KDA"):
        normalize_spec_kda_mode(raw)


@pytest.mark.parametrize("raw", ["", "0", "off", "o6a", "c3", "auto", "1", "o6b"])
def test_envs_spec_kda_getter_matches_parser(monkeypatch, raw):
    # envs canonicalizes the value for the compile-hash factor; it must agree
    # with the parser the runtime switches use.
    monkeypatch.setenv("TPU_KIMI_OPT_SPEC_KDA", raw)
    assert envs.environment_variables["TPU_KIMI_OPT_SPEC_KDA"]() == (
        normalize_spec_kda_mode(raw)
    )


def _set_modes(monkeypatch, *, exit_fold=None, polish=None, spec_kda=None):
    for name, value in (
        ("TPU_KIMI_OPT_EXIT_FOLD", exit_fold),
        ("TPU_KIMI_OPT_COMPUTE_POLISH", polish),
        ("TPU_KIMI_OPT_SPEC_KDA", spec_kda),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_opt_flags_production_settings(monkeypatch):
    _set_modes(monkeypatch, exit_fold="c2", polish="c2", spec_kda="o6b")
    assert opt_flags.exit_fold_mode() == "c2"
    assert opt_flags.is_exit_fold_active()
    assert opt_flags.active_trays_mode() == "off"
    assert opt_flags.vmem_chain_mode() == "vmem_chain"
    assert not opt_flags.use_copy_opt()
    assert not opt_flags.use_kda_glue()
    assert opt_flags.spec_kda_mode() == "o6b"
    assert opt_flags.use_spec_kda_verify(7)
    assert not opt_flags.use_spec_kda_verify(0)
    assert opt_flags.use_spec_verify_extensions()


def test_opt_flags_defaults(monkeypatch):
    _set_modes(monkeypatch)
    assert opt_flags.exit_fold_mode() == "off"
    assert not opt_flags.is_exit_fold_active()
    assert opt_flags.active_trays_mode() == "off"
    assert opt_flags.vmem_chain_mode() == "off"
    assert not opt_flags.use_copy_opt()
    assert not opt_flags.use_kda_glue()
    assert not opt_flags.use_spec_kda_verify(7)
    assert not opt_flags.use_spec_verify_extensions()


def test_opt_flags_all_polish_tokens(monkeypatch):
    _set_modes(monkeypatch, polish="all", spec_kda="o6a")
    assert opt_flags.active_trays_mode() == "trays"
    assert opt_flags.vmem_chain_mode() == "vmem_chain"
    assert opt_flags.use_copy_opt()
    assert opt_flags.use_kda_glue()
    assert opt_flags.use_spec_kda_verify(1)
    assert not opt_flags.use_spec_verify_extensions()


def test_is_exit_fold_active_explicit_value(monkeypatch):
    _set_modes(monkeypatch, exit_fold="c2")
    assert not opt_flags.is_exit_fold_active("off")
    assert opt_flags.is_exit_fold_active("o4d")


@pytest.mark.parametrize(
    "polish,spec_kda,expected",
    [
        ("c2", "o6b", ("off", False, fused_moe.FUSED_VERIFY_MOE_MAX_TOKENS)),
        ("c2", "off", ("off", False, fused_moe.FUSED_DECODE_MOE_MAX_TOKENS)),
        (
            "trays+copy_opt",
            "o6a",
            ("trays", True, fused_moe.FUSED_DECODE_MOE_MAX_TOKENS),
        ),
        (None, None, ("off", False, fused_moe.FUSED_DECODE_MOE_MAX_TOKENS)),
    ],
)
def test_fused_moe_adapter_resolves_fused_decode_options(
    monkeypatch, polish, spec_kda, expected
):
    _set_modes(monkeypatch, polish=polish, spec_kda=spec_kda)
    assert fused_moe._resolve_fused_decode_options() == expected
