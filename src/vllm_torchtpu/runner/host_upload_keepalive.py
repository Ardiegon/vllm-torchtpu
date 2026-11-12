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
"""Lifetime of host tensors uploaded with non-blocking H2D copies."""

import torch


class HostUploadKeepalive:
    """Keeps the host source of each non-blocking upload alive until fenced.

    torch_tpu hands PJRT the host pointer with
    kImmutableUntilTransferCompletes and keeps no reference to the source, so
    a temporary freed right after `.to()` returns can be reused by the
    allocator (whose free-list pointers then land in the upload) before the
    transfer has read it. That corrupted e.g. `logits_indices` on one TP rank
    in ~2% of prefills.

    TODO: delete this class and upload directly once torch_tpu keeps the
    source of a non-blocking host-to-device copy alive until the transfer
    completes.

    Each uploaded tensor goes through three states: pending (uploaded, no
    fence recorded after it yet), fenced (a fence covering the upload has
    been recorded) and released (that fence has been waited on).
    """

    def __init__(self) -> None:
        self._pending: list[torch.Tensor] = []
        self._fenced: list[torch.Tensor] = []

    def upload(self, host: torch.Tensor, device: torch.device) -> torch.Tensor:
        """`host.to(device, non_blocking=True)`, holding `host` until fenced."""
        host = host.contiguous()
        self._pending.append(host)
        return host.to(device, non_blocking=True)

    def on_fence_recorded(self) -> None:
        """A fence recorded now covers every upload issued so far."""
        self._fenced.extend(self._pending)
        self._pending = []

    def on_fence_waited(self) -> None:
        """The recorded fence completed, so its uploads have read their source."""
        self._fenced = []
