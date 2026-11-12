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
"""CPU tests for HostUploadKeepalive (pending -> fenced -> released)."""

import weakref

import torch

from vllm_torchtpu.runner.host_upload_keepalive import HostUploadKeepalive


def _upload_temporary(keepalive: HostUploadKeepalive) -> weakref.ref:
    host = torch.arange(4, dtype=torch.int32) - 1
    ref = weakref.ref(host)
    uploaded = keepalive.upload(host, torch.device("cpu"))
    assert torch.equal(uploaded, torch.tensor([-1, 0, 1, 2], dtype=torch.int32))
    return ref


def test_pending_upload_survives_a_wait_without_a_fence():
    keepalive = HostUploadKeepalive()
    ref = _upload_temporary(keepalive)
    keepalive.on_fence_waited()
    assert ref() is not None


def test_upload_released_once_its_fence_is_waited():
    keepalive = HostUploadKeepalive()
    before_fence = _upload_temporary(keepalive)
    keepalive.on_fence_recorded()
    after_fence = _upload_temporary(keepalive)
    keepalive.on_fence_waited()
    assert before_fence() is None
    assert after_fence() is not None
    keepalive.on_fence_recorded()
    keepalive.on_fence_waited()
    assert after_fence() is None


def test_upload_makes_a_non_contiguous_source_contiguous():
    keepalive = HostUploadKeepalive()
    host = torch.arange(6, dtype=torch.int32).reshape(2, 3).t()
    uploaded = keepalive.upload(host, torch.device("cpu"))
    assert torch.equal(uploaded, host)
