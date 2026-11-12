# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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

import hashlib
import os
import re
from typing import NamedTuple

# envs.py imports nothing from this package, so it is safe to load here,
# while the package's own __init__ is still running.
from vllm_torchtpu import envs
from vllm_torchtpu.kimi_k3_opt_modes import (
    normalize_compute_polish_mode,
    normalize_exit_fold_mode,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_existing_tier2_cache = os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE")

# Configure TorchTPU compilation cache environment variables directly
if envs.TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:
    # Enable Tier-2 cache in memory (required by Tier-3)
    os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "tpu_tier2_cache")
    os.environ.setdefault(
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        envs.TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT,
    )
    os.environ.setdefault(
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "tpu_tier2_cache"
    )

# Default VLLM_CACHE_ROOT to ~/.cache/vllm if unset, ensuring a writable default path
os.environ.setdefault("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Disable forced graph breaks for functional collectives by default.
# torch_tpu commit 477f611f made the collective-ops graph-break guardrail
# active by default on module import in non-absl environments, causing a
# Dynamo Unsupported crash during vLLM startup (fullgraph=True). Setting
# this to false allows functional collectives to compile under SPMD, which
# is safe and required for Tensor Parallelism.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")

# vLLM 0.27.0 auto-enables breakable cudagraph for Kimi K3, disabling its
# torch.compile pipeline. TPU does not support CUDA graphs and requires that
# pipeline, so force the flag off before VllmConfig initialization.
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

# vLLM's "mega" AOT artifact is inductor-only by construction: it saves whatever
# a runnable's serialize() returns but reloads it through
# torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize
# (vllm/compilation/caching.py load_all), which an out-of-tree CompilerInterface
# cannot produce without forging that format. So the artifact is written -- 689 MB
# per rank on the 12-layer Qwen3.5-35B -- and always fails to load:
#   "Compiling model again due to a load failure ..., reason: a bytes-like object
#    is required, not 'BundledAOTAutogradResult'"
# every start silently re-traces, and the 0-entry artifact the failed reload then
# saves makes the *next* start raise TypeError. It defaults on for torch >= 2.12
# (vllm/envs.py use_mega_aot_artifact). Our per-range executable cache already
# persists the same bytes, keyed one entry per (compile range, subgraph), so
# turning this off loses nothing and is what lets a warm start reuse them.
os.environ.setdefault("VLLM_USE_MEGA_AOT_ARTIFACT", "0")

# Size libtpu's premapped (DMA-mapped) host buffer pool when the Raiden KV
# transfer stack is enabled. Raiden's XlaHostMemoryAllocator deliberately
# skips per-allocation DMA mapping on multi-process v7x workers
# (host_memory_allocator.cc), and torch_tpu pins TPU_PREMAPPED_BUFFER_SIZE
# to "0" when unset (discovery.cc, memoized at torch_tpu import). The
# combination routes every KV-cache D2H/H2D through libtpu's synchronous
# staged-copy slow path (~0.4 GB/s with per-transfer IOMMU map/unmap).
# A sized pool restores the pinned fast path; 16 GiB per worker covers the
# connector's host mirrors with headroom. Must run before torch_tpu is
# imported; overridable by presetting the variable.
if (
    os.getenv("TPU_KV_RESHARD_TRANSPORT") == "raiden"
    or os.getenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER") == "1"
):
    os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(16 << 30))

# TPU_GDN_CONV_QK_PAIR_LAYOUT resolves in two layers:
#   1. An explicitly set value always wins (enable or disable), for both
#      raiden and non-raiden runs — hence setdefault below.
#   2. Otherwise, when the Raiden reshard stack is enabled, the layout
#      follows the reshard's source fan-in: enabled iff
#      TPU_RAIDEN_TRANSFER_PARALLELISM >= 8.  At such degrees a rank's
#      conv Q and K segments are sub-token (512 B halves of the 1024-byte
#      physical pool token under TP8), so only the pair-blocked layout
#      gives whole-token — placement-exact — raw transfer spans; at lower
#      source degrees the segment-major layout is already token-aligned.
# The fan-in is parsed exactly like envs.TPU_RAIDEN_TRANSFER_PARALLELISM
# (unset/empty means the default degree of 8 — keep the two in sync), so
# the layout decision and the transfer stack always agree; today that
# makes the pair layout the raiden default.  The launcher exports the
# variable identically on both roles, so producer and consumer resolve the
# same layout; a mismatched pair still fails closed through the layout
# fingerprint.  An unparsable value raises here, at startup, rather than
# letting the two readers diverge.
if (
    os.getenv("TPU_KV_RESHARD_TRANSPORT") == "raiden"
    or os.getenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER") == "1"
):
    _raiden_transfer_degree = int(os.getenv("TPU_RAIDEN_TRANSFER_PARALLELISM") or "8")
    if _raiden_transfer_degree >= 8:
        os.environ.setdefault("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")

# Per-tensor-core VMEM capacity by TPU family (used to set the SC-offload
# threshold below). Values mirror JAX's pallas chip table at
# jax/_src/pallas/mosaic/tpu_info.py. We resolve from the chip family string
# because we run before libtpu is loadable and cannot query pltpu directly.
# Unknown families fall through to libtpu's default (no flag set); add new
# entries here when a new chip family ships.
_TPU_VMEM_BYTES_PER_TENSOR_CORE = {
    # TPU v5p: 64 MiB per tensor core.
    "v5p": 67108864,
    # TPU v5e (accelerator-type prefix "v5litepod"): 128 MiB per tensor core.
    "v5litepod": 134217728,
    # TPU v6e (Trillium): 128 MiB per tensor core.
    "v6e": 134217728,
    # TPU v7 (accelerator-type prefix "tpu7x" / device-kind "v7x"):
    # 64 MiB per tensor core.
    "tpu7x": 67108864,
    "v7x": 67108864,
}


def _fetch_tpu_type_from_metadata() -> str | None:
    # Inline the GCE metadata probe instead of delegating to
    # vllm_torchtpu.tpu_info, because that helper transitively imports
    # vllm.logger -> vllm -> torch -> torch_tpu autoload -> libtpu.so, which
    # would violate the "set LIBTPU_INIT_ARGS before libtpu loads" invariant
    # this file exists to enforce.
    try:
        import requests

        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/attributes/accelerator-type",
            headers={"Metadata-Flavor": "Google"},
        )
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:  # noqa: S110
        pass
    return None


def _resolve_sc_offload_threshold_bytes() -> int | None:
    # An explicit value wins, with 0 meaning no flag. None (the variable
    # was unset or "auto") falls through to the chip-family table below.
    override = envs.SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES
    if override is not None:
        return override if override > 0 else None

    tpu_type = os.getenv("TPU_ACCELERATOR_TYPE")
    if not tpu_type:
        tpu_type = _fetch_tpu_type_from_metadata()
    if not tpu_type:
        return None
    tpu_type = tpu_type.strip().lower()
    for prefix, size in _TPU_VMEM_BYTES_PER_TENSOR_CORE.items():
        if tpu_type.startswith(prefix):
            return size
    return None


_existing_libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")


def _user_has_flag(flag_name: str, args_string: str | None = None) -> bool:
    # Tokenized startswith("<flag>=") check so a hypothetical longer flag
    # that embeds ours as a prefix doesn't false-positive.
    if args_string is None:
        args_string = _existing_libtpu_args
    return any(
        token.startswith(flag_name + "=") or token == flag_name
        for token in args_string.split()
    )


def set_libtpu_flag_default(flag_name: str, value: str) -> None:
    """Append ``<flag_name>=<value>`` to ``LIBTPU_INIT_ARGS`` unless already set.

    Must be called before the PjRt client is initialized (libtpu parses
    ``LIBTPU_INIT_ARGS`` at PjRt client creation, not when ``libtpu.so`` is
    loaded). An explicit user setting in ``LIBTPU_INIT_ARGS`` (either
    ``<flag_name>=...`` or bare ``<flag_name>``) takes precedence.
    """
    current = os.environ.get("LIBTPU_INIT_ARGS", "")
    if _user_has_flag(flag_name, current):
        return
    new_args = f"{current} {flag_name}={value}".strip()
    os.environ["LIBTPU_INIT_ARGS"] = new_args
    logger.info("Appended default flag to LIBTPU_INIT_ARGS: %s", new_args)


# Default XLA overlap preset ("1"/"auto"/"c1"): overlapping async copies with
# collectives cuts the exposed DMA wait of v6e decode at no HBM cost.
# SparseCore collective offload stays off by default: its staging buffers cost
# HBM and add TensorCore DMA wait on v6e decode.
_DEFAULT_XLA_OVERLAP_TOKENS = ("msa_r12", "async_coll")

# Off spellings of TPU_KIMI_OPT_XLA_OVERLAP / TPU_KIMI_OPT_SC_OFFLOAD.
_LIBTPU_KNOB_OFF = ("", "0", "off", "false", "none")

_MSA_RATIO_RE = re.compile(r"^msa_r(\d+)$")
_SC_FULL_RE = re.compile(r"^(?:full_|min_)?(\d+)$")
_SC_AR_ONLY_RE = re.compile(r"^ar_(\d+)$")
_SC_THRESH_ONLY_RE = re.compile(r"^thresh_(\d+)$")

_SC_AR_OFFLOAD_FLAG = "--xla_tpu_enable_sparse_core_collective_offload_all_reduce"
_SC_AG_OFFLOAD_FLAG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather"
_SC_CONCURRENT_FLAG = "--xla_tpu_enable_concurrent_sparse_core_offloading"
_SC_AR_MIN_BYTES_FLAG = "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
_SC_AG_MIN_BYTES_FLAG = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"


class LibtpuFlagOverrides(NamedTuple):
    """LIBTPU_INIT_ARGS additions requested by the Kimi K3 libtpu knobs."""

    # Ordered (flag, value) pairs without duplicate flag names.
    flags: list[tuple[str, str]]
    # SparseCore `*_min_size_in_bytes` thresholds that replace the default
    # per-tensor-core VMEM cap.
    sc_thresholds: dict[str, int]
    # Whether any flag or threshold is requested.
    active: bool


def _parse_xla_overlap(raw: str) -> dict[str, str]:
    """Map `TPU_KIMI_OPT_XLA_OVERLAP` to XLA flags, in token order."""
    flags: dict[str, str] = {}
    norm = raw.strip().lower()
    if norm in _LIBTPU_KNOB_OFF:
        return flags
    tokens: list[str] = []
    for tok in norm.split("+"):
        tok = tok.strip()
        if tok in _LIBTPU_KNOB_OFF:
            continue
        if tok in ("1", "true", "auto", "c1"):
            tokens.extend(_DEFAULT_XLA_OVERLAP_TOKENS)
        else:
            tokens.append(tok)

    for tok in tokens:
        m_msa = _MSA_RATIO_RE.match(tok)
        if m_msa:
            ratio = int(m_msa.group(1))
            if not (1 <= ratio <= 31):
                raise ValueError(
                    f"TPU_KIMI_OPT_XLA_OVERLAP={raw!r}: MSA "
                    f"overlap ratio {ratio} must be in 1..31 (32 hits the "
                    "v6e VMEM prefetch capacity cliff)."
                )
            flags["--xla_msa_preferred_overlap_to_async_copy_ratio"] = str(ratio)
        elif tok == "async_coll":
            flags["--xla_tpu_enable_async_collective_fusion"] = "true"
            flags["--xla_tpu_enable_async_collective_fusion_fuse_all_gather"] = "true"
        elif tok == "cpp_256":
            flags["--xla_msa_max_cross_program_prefetches"] = "256"
        else:
            raise ValueError(
                f"Unsupported TPU_KIMI_OPT_XLA_OVERLAP token {tok!r} in "
                f"{raw!r}; expected 'off', 'auto'/'c1'/'1', "
                "'msa_r<1..31>', 'async_coll', 'cpp_256', or '+'-joined "
                "combinations."
            )
    return flags


def _parse_sc_offload(raw: str) -> tuple[dict[str, str], dict[str, int]]:
    """Map `TPU_KIMI_OPT_SC_OFFLOAD` to (offload flags, threshold overrides)."""
    norm = raw.strip().lower()
    if norm in ("1", "true"):
        norm = "8192"
    elif norm in ("auto", "c1"):
        # The default setting keeps the offload off.
        norm = "off"
    if norm in _LIBTPU_KNOB_OFF:
        return {}, {}

    m_full = _SC_FULL_RE.match(norm)
    m_ar = _SC_AR_ONLY_RE.match(norm)
    m_th = _SC_THRESH_ONLY_RE.match(norm)
    if m_full:
        min_bytes = int(m_full.group(1))
        flags = {
            _SC_AR_OFFLOAD_FLAG: "true",
            _SC_AG_OFFLOAD_FLAG: "true",
            _SC_CONCURRENT_FLAG: "true",
        }
        return flags, {
            _SC_AR_MIN_BYTES_FLAG: min_bytes,
            _SC_AG_MIN_BYTES_FLAG: min_bytes,
        }
    if m_ar:
        min_bytes = int(m_ar.group(1))
        flags = {_SC_AR_OFFLOAD_FLAG: "true", _SC_CONCURRENT_FLAG: "true"}
        return flags, {_SC_AR_MIN_BYTES_FLAG: min_bytes}
    if m_th:
        min_bytes = int(m_th.group(1))
        return {}, {_SC_AR_MIN_BYTES_FLAG: min_bytes, _SC_AG_MIN_BYTES_FLAG: min_bytes}
    raise ValueError(
        f"Unsupported TPU_KIMI_OPT_SC_OFFLOAD={raw!r}; "
        "expected 'off'/'0', '1'/'auto'/'c1', '<bytes>'/'min_<bytes>', "
        "'ar_<bytes>', or 'thresh_<bytes>'."
    )


def _resolve_kimi_libtpu_flags(
    xla_overlap_raw: str | None = None,
    sc_offload_raw: str | None = None,
) -> LibtpuFlagOverrides:
    """Resolve the Kimi K3 libtpu knobs into LIBTPU_INIT_ARGS additions.

    Args:
      xla_overlap_raw: `TPU_KIMI_OPT_XLA_OVERLAP` value; None reads the env.
      sc_offload_raw: `TPU_KIMI_OPT_SC_OFFLOAD` value; None reads the env.
    """
    if xla_overlap_raw is None:
        xla_overlap_raw = envs.TPU_KIMI_OPT_XLA_OVERLAP
    if sc_offload_raw is None:
        sc_offload_raw = envs.TPU_KIMI_OPT_SC_OFFLOAD
    flags = _parse_xla_overlap(xla_overlap_raw)
    sc_flags, sc_thresholds = _parse_sc_offload(sc_offload_raw)
    flags.update(sc_flags)
    return LibtpuFlagOverrides(
        flags=list(flags.items()),
        sc_thresholds=sc_thresholds,
        active=bool(flags or sc_thresholds),
    )


_libtpu_extra_args: list[str] = []

# --xla_tpu_use_dynamic_smem_negotiation: let libtpu size SMEM to fit the
# batched RPA attention kernel's scheduling metadata (default budget is too
# small on --attention-backend CUSTOM).
if not _user_has_flag("--xla_tpu_use_dynamic_smem_negotiation"):
    _libtpu_extra_args.append("--xla_tpu_use_dynamic_smem_negotiation=true")

_kimi_libtpu = _resolve_kimi_libtpu_flags()
for _flag_name, _flag_value in _kimi_libtpu.flags:
    if not _user_has_flag(_flag_name):
        _libtpu_extra_args.append(f"{_flag_name}={_flag_value}")

# --xla_tpu_sparse_core_{all_reduce,all_gather}_offload_min_size_in_bytes:
# raise the SC-offload threshold to the per-tensor-core VMEM capacity so
# typical per-layer AR/AG collectives run on TensorCore instead of being
# offloaded to SparseCore, unless TPU_KIMI_OPT_SC_OFFLOAD overrides a threshold.
# The chip family is detected from TPU_ACCELERATOR_TYPE / GCE metadata and
# mapped to a VMEM constant above; each flag is added only if the user hasn't
# supplied it explicitly.
_sc_offload_bytes = _resolve_sc_offload_threshold_bytes()
for _flag_name in (_SC_AR_MIN_BYTES_FLAG, _SC_AG_MIN_BYTES_FLAG):
    _target_bytes = _kimi_libtpu.sc_thresholds.get(_flag_name, _sc_offload_bytes)
    if _target_bytes is not None and not _user_has_flag(_flag_name):
        _libtpu_extra_args.append(f"{_flag_name}={_target_bytes}")

if _libtpu_extra_args:
    _assembled = " ".join(_libtpu_extra_args)
    if _existing_libtpu_args:
        _assembled = _assembled + " " + _existing_libtpu_args
    os.environ["LIBTPU_INIT_ARGS"] = _assembled


def _kimi_tier2_cache_name(
    libtpu_args: str, exit_fold: str, polish_tag: str, spec_kda: str
) -> str | None:
    """Tier-2 cache directory name for the active Kimi K3 knobs, or None.

    The most specific active knob names the directory: speculative KDA, then
    compute polish, then exit fold, then the libtpu flags. The digest folds in
    LIBTPU_INIT_ARGS and every active mode, so processes with different
    settings never share a directory.
    """

    def digest(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    # Bumped whenever the exit-fold implementation changes the compiled program.
    impl_salt = "|impl=o4d_axial_v1" if exit_fold in ("c2", "o4d") else ""
    exit_fold_key = libtpu_args + "|exit_fold=" + exit_fold + impl_salt
    polish_key = exit_fold_key + "|polish=" + polish_tag
    if spec_kda != "off":
        spec_key = polish_key + "|spec_kda=" + spec_kda
        return f"tpu_tier2_o6_{spec_kda}_{digest(spec_key)}"
    if polish_tag != "off":
        return f"tpu_tier2_o5_{polish_tag[:24]}_{digest(polish_key)}"
    if exit_fold != "off":
        return f"tpu_tier2_o4_{exit_fold}_{digest(exit_fold_key)}"
    if _kimi_libtpu.active:
        return f"tpu_tier2_o3_{digest(libtpu_args)}"
    return None


# Parsed unconditionally so an invalid value fails at import (fail closed).
_kimi_exit_fold_mode = normalize_exit_fold_mode(envs.TPU_KIMI_OPT_EXIT_FOLD)
_kimi_polish_tag, _ = normalize_compute_polish_mode(envs.TPU_KIMI_OPT_COMPUTE_POLISH)
_kimi_spec_kda_mode = envs.TPU_KIMI_OPT_SPEC_KDA

# Tier-2 cache isolation: torch_tpu's /dev/shm/torch_tpu_cache/<name> keys
# compiled executables by HLO + PjRt version, without LIBTPU_INIT_ARGS. When a
# Kimi K3 knob changes LIBTPU_INIT_ARGS or the traced program and the caller
# has not pinned TORCH_TPU_TIER2_COMPILATION_CACHE, namespace the Tier-2
# directory so processes on the same host with different settings cannot
# reuse a stale executable.
if not _existing_tier2_cache:
    _tier2_name = _kimi_tier2_cache_name(
        os.environ.get("LIBTPU_INIT_ARGS", ""),
        _kimi_exit_fold_mode,
        _kimi_polish_tag,
        _kimi_spec_kda_mode,
    )
    if _tier2_name is not None:
        os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = _tier2_name
        os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = _tier2_name


def _patch_jax_pallas_fori_lowering() -> None:
    """Elide the dead remainder ``scf.for`` from Pallas ``fori_loop`` lowering.

    JAX Pallas' Mosaic lowering unconditionally emits a remainder loop for
    dynamic-bound ``jax.lax.fori_loop`` even when ``unroll=1`` (the default),
    where the main loop already covers the full range and the remainder is a
    no-op. libtpu still lowers the empty remainder to device code, which
    materially grows the emitted Mosaic body and slows decode-heavy MoE
    workloads that rely on ``fori_loop`` in Pallas scheduler kernels.

    Rewrites ``_lower_jaxpr_to_for_loop`` in place to gate the remainder
    emission on ``unroll != 1``. Applied at import so it takes effect before
    any Pallas kernel is traced. No-op if the source no longer contains the
    exact needle (upstream fix landed, or the file was refactored), so it
    remains safe across JAX bumps.
    """
    try:
        from jax._src.pallas.mosaic import lowering as _lowering
    except Exception:
        return
    if getattr(_lowering, "_torchtpu_fori_patch_applied", False):
        return
    orig = getattr(_lowering, "_lower_jaxpr_to_for_loop", None)
    if orig is None:
        return

    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(orig))
    except Exception:
        return
    needle = "elif has_dynamic_remainder:"
    fix = "elif has_dynamic_remainder and unroll != 1:"
    if needle not in src or fix in src:
        return
    patched_src = src.replace(needle, fix, 1)
    ns: dict = {"__name__": _lowering.__name__}
    try:
        exec(compile(patched_src, _lowering.__file__, "exec"), _lowering.__dict__, ns)
    except Exception:
        return
    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return
    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._torchtpu_fori_patch_applied = True


from vllm_torchtpu import patch_registry  # noqa: E402

patch_registry.apply("import")
