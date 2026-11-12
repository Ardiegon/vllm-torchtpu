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
"""Defaults, spellings and rejections of the Kimi K3 `envs.py` knobs.

The mode parsers themselves are covered by `test_kimi_k3_opt_modes.py`; these
tests pin what `envs.<KNOB>` returns, because that value is also a compile
hash factor and the raw `TPU_KIMI_OPT_EXIT_FOLD` spelling is part of the DMA
collectives' op names.
"""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.envs import env_canonical, env_str

VAR = "TPU_TEST_ONLY_KNOB"

KIMI_KNOBS = (
    "TPU_LATENT_PROJ_TP",
    "TPU_KDA_DIRECT_POOL_STREAMING",
    "TPU_KIMI_DENSE_W8A16_B11",
    "TPU_KIMI_OPT_2D_EP_MESH",
    "TPU_KIMI_OPT_PACKED_DP_AG",
    "TPU_KIMI_OPT_LATENCY_COLLECTIVES",
    "TPU_KIMI_OPT_LATENCY_TP_PSUM",
    "TPU_KIMI_OPT_XLA_OVERLAP",
    "TPU_KIMI_OPT_SC_OFFLOAD",
    "TPU_KIMI_OPT_EXIT_FOLD",
    "TPU_KIMI_OPT_COMPUTE_POLISH",
    "TPU_KIMI_OPT_SPEC_KDA",
    "TPU_KIMI_DSPARK_TAP",
)

# The values the production Kimi K3 launch sets.
PRODUCTION = {
    "TPU_KIMI_OPT_SPEC_KDA": "o6b",
    "TPU_KIMI_OPT_EXIT_FOLD": "c2",
    "TPU_KIMI_OPT_COMPUTE_POLISH": "c2",
    "TPU_KIMI_DSPARK_TAP": "prefix",
    "TPU_KIMI_OPT_XLA_OVERLAP": "auto",
    "TPU_KIMI_OPT_SC_OFFLOAD": "auto",
    "TPU_KIMI_OPT_PACKED_DP_AG": "1",
    "TPU_KIMI_OPT_LATENCY_COLLECTIVES": "1",
    "TPU_KIMI_OPT_2D_EP_MESH": "1",
    "TPU_KIMI_OPT_LATENCY_TP_PSUM": "off",
}


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for name in (VAR, *KIMI_KNOBS, "TPU_KIMI_OPT_WORLD_LATENT_AG"):
        monkeypatch.delenv(name, raising=False)


def _read(name):
    return envs.environment_variables[name]()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TPU_LATENT_PROJ_TP", True),
        ("TPU_KDA_DIRECT_POOL_STREAMING", True),
        ("TPU_KIMI_DENSE_W8A16_B11", False),
        ("TPU_KIMI_OPT_2D_EP_MESH", False),
        ("TPU_KIMI_OPT_PACKED_DP_AG", False),
        ("TPU_KIMI_OPT_LATENCY_COLLECTIVES", False),
        ("TPU_KIMI_OPT_LATENCY_TP_PSUM", "off"),
        ("TPU_KIMI_OPT_XLA_OVERLAP", "off"),
        ("TPU_KIMI_OPT_SC_OFFLOAD", "off"),
        ("TPU_KIMI_OPT_EXIT_FOLD", "off"),
        ("TPU_KIMI_OPT_COMPUTE_POLISH", "off"),
        ("TPU_KIMI_OPT_SPEC_KDA", "off"),
        ("TPU_KIMI_DSPARK_TAP", "prefix"),
    ],
)
def test_kimi_knob_defaults(name, expected):
    assert _read(name) == expected


def test_every_kimi_knob_is_registered():
    assert all(name in envs.environment_variables for name in KIMI_KNOBS)


def test_world_latent_ag_is_no_longer_a_knob(monkeypatch):
    """Setting the removed knob (production sets it to 0) is harmless."""
    monkeypatch.setenv("TPU_KIMI_OPT_WORLD_LATENT_AG", "0")
    assert "TPU_KIMI_OPT_WORLD_LATENT_AG" not in envs.environment_variables


def test_production_values(monkeypatch):
    for name, value in PRODUCTION.items():
        monkeypatch.setenv(name, value)
    assert {name: _read(name) for name in PRODUCTION} == {
        "TPU_KIMI_OPT_SPEC_KDA": "o6b",
        "TPU_KIMI_OPT_EXIT_FOLD": "c2",
        "TPU_KIMI_OPT_COMPUTE_POLISH": "c2",
        "TPU_KIMI_DSPARK_TAP": "prefix",
        "TPU_KIMI_OPT_XLA_OVERLAP": "auto",
        "TPU_KIMI_OPT_SC_OFFLOAD": "auto",
        "TPU_KIMI_OPT_PACKED_DP_AG": True,
        "TPU_KIMI_OPT_LATENCY_COLLECTIVES": True,
        "TPU_KIMI_OPT_2D_EP_MESH": True,
        "TPU_KIMI_OPT_LATENCY_TP_PSUM": "off",
    }


@pytest.mark.parametrize(
    "name",
    [
        "TPU_KIMI_OPT_LATENCY_TP_PSUM",
        "TPU_KIMI_OPT_XLA_OVERLAP",
        "TPU_KIMI_OPT_SC_OFFLOAD",
        "TPU_KIMI_OPT_EXIT_FOLD",
    ],
)
def test_raw_string_knobs_are_stripped_not_canonicalized(monkeypatch, name):
    """These are parsed downstream; the getter only strips whitespace."""
    monkeypatch.setenv(name, " Auto ")
    assert _read(name) == "Auto"
    monkeypatch.setenv(name, "")
    assert _read(name) == ""


@pytest.mark.parametrize("value", ["", "   "])
def test_compute_polish_empty_reads_as_off(monkeypatch, value):
    monkeypatch.setenv("TPU_KIMI_OPT_COMPUTE_POLISH", value)
    assert _read("TPU_KIMI_OPT_COMPUTE_POLISH") == "off"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "off"),
        ("0", "off"),
        ("OFF", "off"),
        ("none", "off"),
        ("o6a", "o6a"),
        ("c3", "o6a"),
        (" Auto ", "o6a"),
        ("1", "o6a"),
        ("yes", "o6a"),
        ("o6b", "o6b"),
        ("O6B", "o6b"),
    ],
)
def test_spec_kda_is_canonical(monkeypatch, value, expected):
    monkeypatch.setenv("TPU_KIMI_OPT_SPEC_KDA", value)
    assert _read("TPU_KIMI_OPT_SPEC_KDA") == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "prefix"),
        ("0", "prefix"),
        ("prefix", "prefix"),
        ("Prefix_Only", "prefix"),
        ("1", "attn_res"),
        ("attn_res", "attn_res"),
        (" stream ", "attn_res"),
        ("attn_res_stream", "attn_res"),
    ],
)
def test_dspark_tap_is_canonical(monkeypatch, value, expected):
    monkeypatch.setenv("TPU_KIMI_DSPARK_TAP", value)
    assert _read("TPU_KIMI_DSPARK_TAP") == expected


@pytest.mark.parametrize(
    "name,value",
    [("TPU_KIMI_OPT_SPEC_KDA", "o7"), ("TPU_KIMI_DSPARK_TAP", "bogus")],
)
def test_canonical_knobs_reject_unknown_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _read(name)


def test_env_str_empty_is_default(monkeypatch):
    monkeypatch.setenv(VAR, "  ")
    assert env_str(VAR, "off", empty_is_default=True)() == "off"
    assert env_str(VAR, "off")() == ""
    monkeypatch.setenv(VAR, " x ")
    assert env_str(VAR, "off", empty_is_default=True)() == "x"


def test_env_canonical_passes_the_raw_value(monkeypatch):
    seen = []

    def canonicalize(raw):
        seen.append(raw)
        return "mode"

    getter = env_canonical(VAR, canonicalize)
    assert getter() == "mode"
    monkeypatch.setenv(VAR, " A ")
    assert getter() == "mode"
    assert seen == [None, " A "]
