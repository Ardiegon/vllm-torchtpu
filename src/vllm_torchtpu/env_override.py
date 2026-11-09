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

# envs.py imports nothing from this package, so it is safe to load here,
# while the package's own __init__ is still running.
from vllm_torchtpu import envs
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


# Cycle 1 Phase O3 empirically verified presets on v6e (msa_r12 + async_coll
# cuts exposed DMA wait by 34-47% and wins decode + T=4096 TTFT with 0 HBM
# overhead and OG-1 cos >= 0.999983; SparseCore collective offload remains off
# by default because SC staging adds +141-276 MiB stackReservedBytes and
# +372-524 us TensorCore DMA wait on v6e decode).
_DEFAULT_KIMI_XLA_OVERLAP_PRESET = "msa_r12+async_coll"
_DEFAULT_KIMI_SC_OFFLOAD_PRESET = "off"

_MSA_RATIO_RE = re.compile(r"^msa_r(\d+)$")
_SC_FULL_RE = re.compile(r"^(?:full_|min_)?(\d+)$")
_SC_AR_ONLY_RE = re.compile(r"^ar_(\d+)$")
_SC_THRESH_ONLY_RE = re.compile(r"^thresh_(\d+)$")


def resolve_kimi_o3_libtpu_flags(
    xla_overlap_raw: str | None = None,
    sc_offload_raw: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int], bool]:
    """Resolve Kimi K3 Phase O3 opt-in knobs into LIBTPU_INIT_ARGS flag pairs.

    Returns ``(extra_flag_pairs, sc_threshold_overrides, is_active)`` where:
      - ``extra_flag_pairs`` is an ordered list of ``(flag_name, value)`` pairs
        without duplicate flag names.
      - ``sc_threshold_overrides`` maps SparseCore ``*_min_size_in_bytes`` flag
        names to explicit byte thresholds that override the default per-TC VMEM
        cap (`128 MiB` on `v6e`).
      - ``is_active`` is ``True`` iff any non-baseline O3 setting is active.
    """
    if xla_overlap_raw is None:
        xla_overlap_raw = envs.TPU_KIMI_OPT_XLA_OVERLAP
    if sc_offload_raw is None:
        sc_offload_raw = envs.TPU_KIMI_OPT_SC_OFFLOAD

    flag_map: dict[str, str] = {}
    sc_thresholds: dict[str, int] = {}
    active = False

    ov_norm = xla_overlap_raw.strip().lower()
    if ov_norm not in ("", "0", "off", "false", "none"):
        expanded_tokens: list[str] = []
        for tok in ov_norm.split("+"):
            tok = tok.strip()
            if not tok or tok in ("0", "off", "false", "none"):
                continue
            if tok in ("1", "true", "auto", "c1"):
                preset = _DEFAULT_KIMI_XLA_OVERLAP_PRESET.strip().lower()
                if preset not in ("", "0", "off", "false", "none"):
                    expanded_tokens.extend(preset.split("+"))
            else:
                expanded_tokens.append(tok)

        for tok in expanded_tokens:
            m_msa = _MSA_RATIO_RE.match(tok)
            if m_msa:
                ratio = int(m_msa.group(1))
                if not (1 <= ratio <= 31):
                    raise ValueError(
                        f"TPU_KIMI_OPT_XLA_OVERLAP={xla_overlap_raw!r}: MSA "
                        f"overlap ratio {ratio} must be in 1..31 (32 hits the "
                        "v6e VMEM prefetch capacity cliff)."
                    )
                flag_map["--xla_msa_preferred_overlap_to_async_copy_ratio"] = str(
                    ratio
                )
                active = True
            elif tok == "async_coll":
                flag_map["--xla_tpu_enable_async_collective_fusion"] = "true"
                flag_map[
                    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather"
                ] = "true"
                active = True
            elif tok == "cpp_256":
                flag_map["--xla_msa_max_cross_program_prefetches"] = "256"
                active = True
            else:
                raise ValueError(
                    f"Unsupported TPU_KIMI_OPT_XLA_OVERLAP token {tok!r} in "
                    f"{xla_overlap_raw!r}; expected 'off', 'auto'/'c1'/'1', "
                    "'msa_r<1..31>', 'async_coll', 'cpp_256', or '+'-joined "
                    "combinations."
                )

    sc_norm = sc_offload_raw.strip().lower()
    if sc_norm in ("1", "true"):
        sc_norm = "8192"
    elif sc_norm in ("auto", "c1"):
        sc_norm = _DEFAULT_KIMI_SC_OFFLOAD_PRESET.strip().lower()

    if sc_norm not in ("", "0", "off", "false", "none"):
        m_full = _SC_FULL_RE.match(sc_norm)
        m_ar = _SC_AR_ONLY_RE.match(sc_norm)
        m_th = _SC_THRESH_ONLY_RE.match(sc_norm)
        if m_full:
            min_bytes = int(m_full.group(1))
            flag_map[
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce"
            ] = "true"
            flag_map[
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather"
            ] = "true"
            flag_map["--xla_tpu_enable_concurrent_sparse_core_offloading"] = (
                "true"
            )
            sc_thresholds[
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
            ] = min_bytes
            sc_thresholds[
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
            ] = min_bytes
            active = True
        elif m_ar:
            min_bytes = int(m_ar.group(1))
            flag_map[
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce"
            ] = "true"
            flag_map["--xla_tpu_enable_concurrent_sparse_core_offloading"] = (
                "true"
            )
            sc_thresholds[
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
            ] = min_bytes
            active = True
        elif m_th:
            min_bytes = int(m_th.group(1))
            sc_thresholds[
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
            ] = min_bytes
            sc_thresholds[
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
            ] = min_bytes
            active = True
        else:
            raise ValueError(
                f"Unsupported TPU_KIMI_OPT_SC_OFFLOAD={sc_offload_raw!r}; "
                "expected 'off'/'0', '1'/'auto'/'c1', '<bytes>'/'min_<bytes>', "
                "'ar_<bytes>', or 'thresh_<bytes>'."
            )

    return list(flag_map.items()), sc_thresholds, active


_libtpu_extra_args: list[str] = []

# --xla_tpu_use_dynamic_smem_negotiation: let libtpu size SMEM to fit the
# batched RPA attention kernel's scheduling metadata (default budget is too
# small on --attention-backend CUSTOM).
if not _user_has_flag("--xla_tpu_use_dynamic_smem_negotiation"):
    _libtpu_extra_args.append("--xla_tpu_use_dynamic_smem_negotiation=true")

_kimi_o3_flags, _kimi_o3_sc_thresholds, _kimi_o3_active = (
    resolve_kimi_o3_libtpu_flags()
)
for _o3_flag_name, _o3_flag_val in _kimi_o3_flags:
    if not _user_has_flag(_o3_flag_name):
        _libtpu_extra_args.append(f"{_o3_flag_name}={_o3_flag_val}")

# --xla_tpu_sparse_core_{all_reduce,all_gather}_offload_min_size_in_bytes:
# raise the SC-offload threshold to the per-tensor-core VMEM capacity so
# typical per-layer AR/AG collectives run on TensorCore instead of being
# offloaded to SparseCore, unless TPU_KIMI_OPT_SC_OFFLOAD overrides a threshold.
# The chip family is detected from TPU_ACCELERATOR_TYPE / GCE metadata and
# mapped to a VMEM constant above; each flag is added only if the user hasn't
# supplied it explicitly.
_sc_offload_bytes = _resolve_sc_offload_threshold_bytes()
for _flag_name in (
    "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes",
):
    _target_bytes = (
        _kimi_o3_sc_thresholds[_flag_name]
        if _flag_name in _kimi_o3_sc_thresholds
        else _sc_offload_bytes
    )
    if _target_bytes is not None and not _user_has_flag(_flag_name):
        _libtpu_extra_args.append(f"{_flag_name}={_target_bytes}")

if _libtpu_extra_args:
    _assembled = " ".join(_libtpu_extra_args)
    if _existing_libtpu_args:
        _assembled = _assembled + " " + _existing_libtpu_args
    os.environ["LIBTPU_INIT_ARGS"] = _assembled

# D87 Tier-2 cache isolation: torch_tpu's /dev/shm/torch_tpu_cache/<name> keys
# compiled executables by HLO + PjRt version without LIBTPU_INIT_ARGS. When O3
# LIBTPU_INIT_ARGS overrides or O4 TPU_KIMI_OPT_EXIT_FOLD modes are active and
# the caller has not explicitly pinned TORCH_TPU_TIER2_COMPILATION_CACHE,
# namespace the Tier-2 directory so distinct O3/O4 sweeps cannot reuse a stale
# executable across processes on the same host.
def _normalize_kimi_o4_exit_fold_env(raw: str | None) -> str:
    val = (raw or "off").strip().lower()
    if val in ("", "0", "false", "no", "off", "none"):
        return "off"
    if val in ("o4a", "o4b", "o4a_sync", "o4b_sync"):
        return val
    if val in ("1", "true", "yes", "auto", "c2"):
        return "c2"
    raise ValueError(
        f"Invalid TPU_KIMI_OPT_EXIT_FOLD={raw!r}; expected 'off', 'o4a', 'o4b', "
        "'o4a_sync', 'o4b_sync', or 'c2'/'auto'/'1'."
    )


_DEFAULT_KIMI_COMPUTE_POLISH_C2_TOKENS: tuple[str, ...] = (
    "vmem_chain",
)
_ALL_KIMI_COMPUTE_POLISH_TOKENS: tuple[str, ...] = (
    "trays",
    "vmem_chain",
    "kda_glue",
    "copy_opt",
)
_KIMI_O5_TRAY_TOKENS: frozenset[str] = frozenset({"trays", "trays2", "trays32"})
_KIMI_O5_VMEM_TOKENS: frozenset[str] = frozenset({"vmem_chain", "vmem_src"})
_KIMI_O5_OFF_TOKENS: frozenset[str] = frozenset(
    {"", "0", "false", "no", "off", "none"}
)


def _normalize_kimi_o5_compute_polish_env(
    raw: str | None = None,
) -> tuple[str, frozenset[str]]:
    if raw is None:
        raw = os.environ.get("TPU_KIMI_OPT_COMPUTE_POLISH", "off")
    val = (raw or "off").strip().lower()
    if val in _KIMI_O5_OFF_TOKENS:
        return "off", frozenset()

    tokens: set[str] = set()
    for part in val.split("+"):
        tok = part.strip()
        if tok in _KIMI_O5_OFF_TOKENS:
            continue
        if tok in ("trays", "trays4"):
            tokens.add("trays")
        elif tok in (
            "trays2",
            "trays32",
            "vmem_chain",
            "vmem_src",
            "kda_glue",
            "kda_cond",
            "kda_cw_f32",
            "copy_opt",
        ):
            tokens.add(tok)
        elif tok == "all":
            tokens.update(_ALL_KIMI_COMPUTE_POLISH_TOKENS)
        elif tok in ("c2", "auto", "1", "true", "yes"):
            tokens.update(_DEFAULT_KIMI_COMPUTE_POLISH_C2_TOKENS)
        else:
            raise ValueError(
                f"Invalid TPU_KIMI_OPT_COMPUTE_POLISH token {tok!r} in {raw!r}; "
                "expected 'off', 'trays'/'trays4', 'trays2', 'trays32', "
                "'vmem_chain', 'vmem_src', 'kda_glue', 'kda_cond', 'kda_cw_f32', "
                "'copy_opt', 'all', 'c2'/'auto'/'1', or '+'-joined combinations."
            )

    active_trays = tokens & _KIMI_O5_TRAY_TOKENS
    if len(active_trays) > 1:
        raise ValueError(
            f"Conflicting tray modes {sorted(active_trays)} in "
            f"TPU_KIMI_OPT_COMPUTE_POLISH={raw!r}; choose at most one of "
            "'trays'/'trays4', 'trays2', or 'trays32'."
        )

    active_vmem = tokens & _KIMI_O5_VMEM_TOKENS
    if len(active_vmem) > 1:
        raise ValueError(
            f"Conflicting VMEM chaining modes {sorted(active_vmem)} in "
            f"TPU_KIMI_OPT_COMPUTE_POLISH={raw!r}; choose at most one of "
            "'vmem_chain' or 'vmem_src'."
        )

    if not tokens:
        return "off", frozenset()

    frozen = frozenset(tokens)
    return "_".join(sorted(frozen)), frozen


_kimi_o4_exit_fold_mode = _normalize_kimi_o4_exit_fold_env(
    os.environ.get("TPU_KIMI_OPT_EXIT_FOLD")
)
_kimi_o5_polish_tag, _ = _normalize_kimi_o5_compute_polish_env(
    os.environ.get("TPU_KIMI_OPT_COMPUTE_POLISH")
)

if _kimi_o5_polish_tag != "off" and not _existing_tier2_cache:
    _o5_t2_digest = hashlib.sha256(
        (
            os.environ.get("LIBTPU_INIT_ARGS", "")
            + "|exit_fold="
            + _kimi_o4_exit_fold_mode
            + "|polish="
            + _kimi_o5_polish_tag
        ).encode()
    ).hexdigest()[:8]
    _o5_t2_name = f"tpu_tier2_o5_{_kimi_o5_polish_tag[:24]}_{_o5_t2_digest}"
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = _o5_t2_name
    os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = _o5_t2_name
elif _kimi_o4_exit_fold_mode != "off" and not _existing_tier2_cache:
    _o4_t2_digest = hashlib.sha256(
        (
            os.environ.get("LIBTPU_INIT_ARGS", "")
            + "|exit_fold="
            + _kimi_o4_exit_fold_mode
        ).encode()
    ).hexdigest()[:8]
    _o4_t2_name = f"tpu_tier2_o4_{_kimi_o4_exit_fold_mode}_{_o4_t2_digest}"
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = _o4_t2_name
    os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = _o4_t2_name
elif _kimi_o3_active and not _existing_tier2_cache:
    _o3_t2_digest = hashlib.sha256(
        os.environ.get("LIBTPU_INIT_ARGS", "").encode()
    ).hexdigest()[:8]
    _o3_t2_name = f"tpu_tier2_o3_{_o3_t2_digest}"
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = _o3_t2_name
    os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = _o3_t2_name


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
