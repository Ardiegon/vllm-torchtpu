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
"""Kimi K3 Cycle 3 Phase O6a (`TPU_KIMI_OPT_SPEC_KDA`) mode helpers.

Delegates normalization to
``vllm_torchtpu.env_override._normalize_kimi_o6_spec_kda_env``
so import-time Tier-2 cache namespacing and the runtime KDA op builders share a
single source of truth (`D87`).
"""

from __future__ import annotations

from vllm_torchtpu import envs
from vllm_torchtpu.env_override import _normalize_kimi_o6_spec_kda_env

normalize_spec_kda_mode = _normalize_kimi_o6_spec_kda_env


def use_spec_kda_o6a(num_spec_tokens: int) -> bool:
    """True when the O6a KDA verify + masked commit replaces the checkpoint path.

    Requires speculative decoding (``num_spec_tokens > 0``); with ``K = 0``
    the KDA ops are the pre-O6a ones regardless of the env value.
    """
    return (
        num_spec_tokens > 0
        and normalize_spec_kda_mode(envs.TPU_KIMI_OPT_SPEC_KDA) == "o6a"
    )


__all__ = ["normalize_spec_kda_mode", "use_spec_kda_o6a"]
