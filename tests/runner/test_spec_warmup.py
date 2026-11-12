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

"""CPU tests for the synthetic requests of the speculative-decoding warmup.

* `TPUModelRunner._spec_warmup_num_output_tokens` keeps the in-flight `1 + K` async
  placeholders of a synthetic request, as vLLM's V1 scheduler does.
* `TPUModelRunner._spec_warmup_bucket_grid` and `_warmup_spec_batch` never
  warm more requests than `scheduler_config.max_num_seqs`, which can be below
  the runner's padded `max_num_reqs` (`MIN_NUM_SEQS`).
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_WARMUP_REQ_ID = "__spec_warmup_0__"


def _req_state_with_placeholders() -> SimpleNamespace:
    # One sampled token plus the 1 + K (K = 7) async placeholders.
    return SimpleNamespace(output_token_ids=[42] + [0] * 8)


def test_num_output_tokens_counts_async_placeholders():
    runner = SimpleNamespace(
        use_async_scheduling=True,
        requests={_WARMUP_REQ_ID: _req_state_with_placeholders()},
    )
    assert TPUModelRunner._spec_warmup_num_output_tokens(
        runner, [_WARMUP_REQ_ID], [2]
    ) == [9]


def test_num_output_tokens_without_async_scheduling_uses_fallback():
    runner = SimpleNamespace(
        use_async_scheduling=False,
        requests={_WARMUP_REQ_ID: _req_state_with_placeholders()},
    )
    assert TPUModelRunner._spec_warmup_num_output_tokens(
        runner, [_WARMUP_REQ_ID], [2]
    ) == [2]


def _runner_with_max_num_seqs_below_min_num_seqs() -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=7)
    runner.drafter = SimpleNamespace(block_size=7)
    runner.max_num_reqs = 8  # MIN_NUM_SEQS
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4)
    runner.max_num_tokens = 1024
    runner.num_tokens_paddings = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return runner


def test_bucket_grid_capped_by_max_num_seqs():
    runner = _runner_with_max_num_seqs_below_min_num_seqs()
    # Counts 5..8 fit the padded `max_num_reqs` but exceed `max_num_seqs`;
    # count 4 shares every program bucket with count 3.
    assert runner._spec_warmup_bucket_grid(mixed=True) == [1, 2, 3]


def test_warmup_spec_batch_skips_more_requests_than_max_num_seqs():
    runner = _runner_with_max_num_seqs_below_min_num_seqs()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[], num_blocks=582)
    runner.max_model_len = 4096
    # Building the first synthetic scheduler step means the guard let R through.
    with patch.object(
        SchedulerOutput,
        "make_empty",
        side_effect=AssertionError("guard did not short-circuit"),
    ):
        runner._warmup_spec_batch(5, 4, greedy=True, short=True, mixed=True)
