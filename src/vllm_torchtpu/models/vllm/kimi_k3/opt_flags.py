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
"""Env-backed switches for the Kimi K3 optimization modes.

Each accessor reads its `TPU_KIMI_OPT_*` env through `vllm_torchtpu.envs` and
canonicalizes it with `vllm_torchtpu.kimi_k3_opt_modes`, the parsers
`env_override` also uses to namespace the compile caches. Call them while
building layers and ops, never inside traced code: the resolved values are
passed to kernels as arguments.
"""

from __future__ import annotations

from vllm_torchtpu import envs
from vllm_torchtpu.kimi_k3_opt_modes import (
    normalize_compute_polish_mode,
    normalize_exit_fold_mode,
    normalize_spec_kda_mode,
)


def exit_fold_mode() -> str:
    """Canonical `TPU_KIMI_OPT_EXIT_FOLD` mode (`"off"`, `"c2"` or `"o4d"`)."""
    return normalize_exit_fold_mode(envs.TPU_KIMI_OPT_EXIT_FOLD)


def is_exit_fold_active(raw: str | None = None) -> bool:
    """Whether the fused EP-overlap MoE exit path is enabled.

    Args:
      raw: an explicit `TPU_KIMI_OPT_EXIT_FOLD` value; None reads the env.
    """
    src = envs.TPU_KIMI_OPT_EXIT_FOLD if raw is None else raw
    return normalize_exit_fold_mode(src) != "off"


def _compute_polish_tokens() -> frozenset[str]:
    _, tokens = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
    return tokens


def active_trays_mode() -> str:
    """Row-loop mode of the fused small-batch MXFP4 MoE kernel.

    Returns `"trays"` (16-row trays) when the `trays` compute-polish token is
    set, else `"off"` (one tile).
    """
    return "trays" if "trays" in _compute_polish_tokens() else "off"


def vmem_chain_mode() -> str:
    """`"vmem_chain"` when the EP-overlap collectives receive into VMEM.

    With the `vmem_chain` compute-polish token the async dispatch wait
    leaves the gathered rows in its receive buffer and the combine
    collectives use VMEM receive buffers up to their DMA token cap.
    Returns `"off"` otherwise.
    """
    return "vmem_chain" if "vmem_chain" in _compute_polish_tokens() else "off"


def use_copy_opt() -> bool:
    """Whether the fused MoE kernel takes rank-3 scale buffers (`copy_opt`)."""
    return "copy_opt" in _compute_polish_tokens()


def use_kda_glue() -> bool:
    """Whether K = 0 pooled KDA fuses its glue ops (`kda_glue`).

    The op then feeds the conv weight in the kernel layout and runs the 2D
    gated output norm.
    """
    return "kda_glue" in _compute_polish_tokens()


def spec_kda_mode() -> str:
    """Canonical `TPU_KIMI_OPT_SPEC_KDA` mode (`"off"`, `"o6a"` or `"o6b"`)."""
    return normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA)


def use_spec_kda_verify(num_spec_tokens: int) -> bool:
    """Whether KDA runs the multi-token verify plus masked commit.

    The verify op reads the state without writing it and the commit op
    writes back only each request's accepted prefix, replacing the
    checkpointed state path. Needs speculative decoding
    (`num_spec_tokens > 0`); both `"o6a"` and `"o6b"` select it.
    """
    return num_spec_tokens > 0 and spec_kda_mode() in ("o6a", "o6b")


def use_spec_verify_extensions() -> bool:
    """Whether the `"o6b"` additions to speculative verify are enabled.

    They are the multi-tray fused verify MoE (up to 256 tokens) and the
    TP-sharded DSpark context projections. Independent of the number of
    speculative tokens: the MoE path is selected by token count.
    """
    return spec_kda_mode() == "o6b"
