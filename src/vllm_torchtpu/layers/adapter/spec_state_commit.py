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
"""Protocol of layers that defer their speculative-decoding state commit."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SpecStateCommitLayer(Protocol):
    """A recurrent-state layer whose verify forward has no side effects.

    During a speculative verify step such a layer runs its `1 + K` window
    without writing its recurrent state. After rejection sampling the runner
    calls `commit_spec` with each request's accepted length, and the layer
    advances its state exactly as that many sequential 1-token steps would.
    A layer takes part only while `defers_spec_state_commit` is true.
    """

    # Key of the layer's attention metadata in the per-layer metadata dict.
    prefix: str
    num_spec_tokens: int
    defers_spec_state_commit: bool

    def commit_spec(
        self,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        window_distribution: torch.Tensor,
        accepted_lens: torch.Tensor,
    ) -> None:
        """Commit the first `accepted_lens[r]` window tokens of batch row `r`."""
        ...
