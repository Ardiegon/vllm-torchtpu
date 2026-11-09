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
"""Kimi K3 Phase O5 (`TPU_KIMI_OPT_COMPUTE_POLISH`) mode normalization & helpers.

Delegates normalization to ``vllm_torchtpu.env_override._normalize_kimi_o5_compute_polish_env``
so import-time Tier-2 cache namespacing and runtime kernel builders share a single
source of truth without importing ``jax`` prematurely during ``env_override`` (`D87`, `D88`).
"""

from __future__ import annotations

from vllm_torchtpu import envs
from vllm_torchtpu.env_override import (
    _ALL_KIMI_COMPUTE_POLISH_TOKENS,
    _DEFAULT_KIMI_COMPUTE_POLISH_C2_TOKENS,
    _normalize_kimi_o5_compute_polish_env,
)

normalize_compute_polish_mode = _normalize_kimi_o5_compute_polish_env


def _current_tokens() -> frozenset[str]:
    _, tokens = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    return tokens


def active_trays_mode() -> str:
    """Return ``'trays'``, ``'trays2'``, ``'trays32'``, or ``'off'``."""
    tokens = _current_tokens()
    for mode in ("trays", "trays2", "trays32"):
        if mode in tokens:
            return mode
    return "off"


def vmem_chain_mode() -> str:
    """Return ``'vmem_chain'``, ``'vmem_src'``, or ``'off'``."""
    tokens = _current_tokens()
    if "vmem_chain" in tokens:
        return "vmem_chain"
    if "vmem_src" in tokens:
        return "vmem_src"
    return "off"


def use_copy_opt() -> bool:
    """True when ``copy_opt`` (rank-3 MoE scale layout + unpack polish) is active."""
    return "copy_opt" in _current_tokens()


def use_kda_glue() -> bool:
    """True when ``kda_glue`` (KDA decode VPU/glue fusion) is active."""
    return "kda_glue" in _current_tokens()


def use_kda_cond() -> bool:
    """True when ``kda_cond`` (diagnostic `lax.cond` per_seq guard) is active."""
    return "kda_cond" in _current_tokens()


def use_kda_cw_f32() -> bool:
    """True when ``kda_cw_f32`` (load-time f32 conv_weight copy) is active."""
    return "kda_cw_f32" in _current_tokens()


__all__ = [
    "_ALL_KIMI_COMPUTE_POLISH_TOKENS",
    "_DEFAULT_KIMI_COMPUTE_POLISH_C2_TOKENS",
    "active_trays_mode",
    "normalize_compute_polish_mode",
    "use_copy_opt",
    "use_kda_cond",
    "use_kda_cw_f32",
    "use_kda_glue",
    "vmem_chain_mode",
]
