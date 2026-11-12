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
"""Parsers for the Kimi K3 optimization mode env values.

Standard library only: `env_override` imports this module at process start,
before jax or libtpu are loaded, to namespace the compile caches on the
canonical modes, and the model-side accessors in
`models/vllm/kimi_k3/opt_flags.py` use the same parsers at build time, so
both agree on what every spelling means. Every parser fails closed: an
unknown value raises `ValueError`.
"""

from __future__ import annotations

from enum import StrEnum

from vllm_torchtpu.envs import OFF_TOKENS, normalize_spec_kda_mode

__all__ = [
    "ALL_COMPUTE_POLISH_TOKENS",
    "DEFAULT_COMPUTE_POLISH_TOKENS",
    "DSparkTapMode",
    "OFF_TOKENS",
    "normalize_compute_polish_mode",
    "normalize_exit_fold_mode",
    "normalize_latency_tp_psum_mode",
    "normalize_spec_kda_mode",
]


class DSparkTapMode(StrEnum):
    """Target auxiliary hidden-state tap mode for the Kimi K3 DSpark drafter."""

    PREFIX = "prefix"
    ATTN_RES = "attn_res"


# `TPU_KIMI_OPT_EXIT_FOLD` spellings of the verified preset. Both the preset
# ("c2") and its explicit name ("o4d", alias "c2b") run the axial
# TP -> DP async EP dispatch with the hierarchical DP reduce-scatter +
# TP psum combine.
_EXIT_FOLD_PRESET_ALIASES: frozenset[str] = frozenset(
    {"1", "true", "yes", "auto", "c2"}
)

# `TPU_KIMI_OPT_COMPUTE_POLISH` token sets: the preset ("c2" and aliases)
# and "all".
DEFAULT_COMPUTE_POLISH_TOKENS: tuple[str, ...] = ("vmem_chain",)
ALL_COMPUTE_POLISH_TOKENS: tuple[str, ...] = (
    "trays",
    "vmem_chain",
    "kda_glue",
    "copy_opt",
)
_COMPUTE_POLISH_PRESET_ALIASES: frozenset[str] = frozenset(
    {"c2", "auto", "1", "true", "yes"}
)


def normalize_exit_fold_mode(raw: str | None) -> str:
    """Canonicalize `TPU_KIMI_OPT_EXIT_FOLD`.

    Returns `"off"` (default and the `OFF_TOKENS` spellings), `"c2"` (the
    verified preset; aliases `"auto"`, `"1"`, `"true"`, `"yes"`) or `"o4d"`
    (alias `"c2b"`), which selects the same plan as `"c2"` under its own
    name.
    """
    val = (raw or "off").strip().lower()
    if val in OFF_TOKENS:
        return "off"
    if val in ("o4d", "c2b"):
        return "o4d"
    if val in _EXIT_FOLD_PRESET_ALIASES:
        return "c2"
    raise ValueError(
        f"Invalid TPU_KIMI_OPT_EXIT_FOLD={raw!r}; expected 'off', 'o4d'/'c2b', "
        "or 'c2'/'auto'/'1'."
    )


def normalize_compute_polish_mode(raw: str | None) -> tuple[str, frozenset[str]]:
    """Canonicalize `TPU_KIMI_OPT_COMPUTE_POLISH`.

    The value is a `+`-joined list of tokens: `"trays"` (alias `"trays4"`),
    `"vmem_chain"`, `"kda_glue"`, `"copy_opt"`, `"all"` (all four) and the
    preset `"c2"` (aliases `"auto"`, `"1"`, `"true"`, `"yes"`), which expands
    to `DEFAULT_COMPUTE_POLISH_TOKENS`. `OFF_TOKENS` spellings are ignored.

    Returns:
      `(tag, tokens)`: the active token set and its canonical tag (the
      sorted tokens joined by `_`, or `"off"` when the set is empty).
    """
    val = (raw or "off").strip().lower()
    if val in OFF_TOKENS:
        return "off", frozenset()

    tokens: set[str] = set()
    for part in val.split("+"):
        tok = part.strip()
        if tok in OFF_TOKENS:
            continue
        if tok in ("trays", "trays4"):
            tokens.add("trays")
        elif tok in ("vmem_chain", "kda_glue", "copy_opt"):
            tokens.add(tok)
        elif tok == "all":
            tokens.update(ALL_COMPUTE_POLISH_TOKENS)
        elif tok in _COMPUTE_POLISH_PRESET_ALIASES:
            tokens.update(DEFAULT_COMPUTE_POLISH_TOKENS)
        else:
            raise ValueError(
                f"Invalid TPU_KIMI_OPT_COMPUTE_POLISH token {tok!r} in {raw!r}; "
                "expected 'off', 'trays'/'trays4', 'vmem_chain', 'kda_glue', "
                "'copy_opt', 'all', 'c2'/'auto'/'1', or '+'-joined combinations."
            )

    if not tokens:
        return "off", frozenset()
    frozen = frozenset(tokens)
    return "_".join(sorted(frozen)), frozen


def normalize_latency_tp_psum_mode(raw: str | None) -> str:
    """Canonicalize `TPU_KIMI_OPT_LATENCY_TP_PSUM`.

    Selects which TP psums use the one-shot remote-DMA collective:
      - `"off"` (default and the `OFF_TOKENS` spellings): none.
      - `"tp16"` (alias `"auto"`): all of them when `tp_size >= 16`, for
        batches of at most 8 rows.
      - `"routed_only"`: only the MoE routed-output psum, when `tp_size > 1`.
      - `"all"` (aliases `"1"`, `"true"`, `"yes"`): the routed-output, the
        attention/MLP exit and the MoE exit psums when `tp_size > 1`, for
        batches of at most 64 rows.
      - a positive integer `"<N>"`: like `"all"` with an `N`-row cap.
    """
    val = (raw or "off").strip().lower()
    if val in OFF_TOKENS:
        return "off"
    if val in ("tp16", "auto"):
        return "tp16"
    if val == "routed_only":
        return "routed_only"
    if val in ("1", "true", "yes", "all"):
        return "all"
    if val.isdigit() and int(val) > 0:
        return str(int(val))
    raise ValueError(
        f"Invalid TPU_KIMI_OPT_LATENCY_TP_PSUM={raw!r}; expected 'off', 'tp16'/'auto', "
        "'routed_only', 'all', or a positive integer max_dma_tokens threshold."
    )
