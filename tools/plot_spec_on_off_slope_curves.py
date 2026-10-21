#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair Spec ON/OFF runs and plot their per-layer controller slopes.

The script discovers ``manifest.json`` files, forms strict pairs whose runtime
settings differ only in ``layer_recompute_async_plan_build``, and reads
``controller_update.gain_used`` from the corresponding flat JSONL traces.  A
curve compares actual independent Spec ON and Spec OFF runs at the same decode
step and layer; it is deliberately different from the counterfactual
early-vs-late adjacent-layer comparison in ``analyze_spec_slope_stability.py``.

By default, one latest complete pair is retained per
``(model, environment, workload)``.  The representative curve uses the
temporal median of well-populated common decode steps, so it is not selected
because the two curves happen to be close.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[1]
ENVIRONMENT_ORDER = ("stable", "io_contention")

# Run-specific paths, tags, profiler outputs, and ablation labels are excluded.
# Every field below must agree before two runs can be called a Spec ON/OFF pair.
MATCH_FIELDS = (
    "model",
    "model_family",
    "planner",
    "prefix_blocks",
    "num_prompts",
    "prompt_words",
    "prompt_word",
    "max_tokens",
    "fixed_output_length",
    "gpu_memory_fraction",
    "num_device_buffers",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_staging_blocks",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "h2d_copy_mode",
    "layer_recompute_use_state_machine",
    "layer_recompute_replay_allocation_policy",
    "workload",
    "benchmark_setting_name",
)

PRESSURE_FIELDS = (
    "resource_pressure_kind",
    "resource_pressure_clock",
    "resource_pressure_pattern",
    "resource_pressure_device",
    "resource_pressure_buffer_mb",
    "resource_pressure_direction",
    "resource_pressure_matrix_size",
    "resource_pressure_dtype",
    "resource_pressure_window_s",
    "resource_pressure_period_ms",
    "resource_pressure_max_fraction",
    "resource_pressure_io_calibration_s",
    "resource_pressure_io_max_gbps",
    "resource_pressure_mode",
    "resource_pressure_process_placement",
    "resource_pressure_inline_layer_period_ms",
)

PAIR_FIELDS = (
    "pair_id",
    "signature_id",
    "model",
    "environment",
    "workload",
    "completion_time_utc",
    "spec_on_manifest",
    "spec_off_manifest",
    "spec_on_flat_jsonl",
    "spec_off_flat_jsonl",
    "matched_config_json",
)

SAMPLE_FIELDS = (
    "pair_id",
    "model",
    "environment",
    "workload",
    "step",
    "layer_idx",
    "num_reqs",
    "total_scheduled_tokens",
    "spec_on_gain_ms_per_block",
    "spec_off_gain_ms_per_block",
    "spec_on_gain_magnitude_ms_per_block",
    "spec_off_gain_magnitude_ms_per_block",
    "same_sign",
    "absolute_change_ms_per_block",
    "symmetric_relative_change",
    "symmetric_relative_change_pct",
    "spec_on_sm_state",
    "spec_off_sm_state",
    "spec_on_action",
    "spec_off_action",
)

SUMMARY_FIELDS = (
    "pair_id",
    "model",
    "environment",
    "workload",
    "common_decode_steps",
    "paired_samples",
    "schedule_mismatch_samples_dropped",
    "representative_step",
    "representative_step_quantile",
    "representative_layers",
    "all_log_magnitude_pearson_r",
    "all_same_sign_fraction",
    "all_within_10pct_fraction",
    "all_median_symmetric_relative_change",
    "all_p90_symmetric_relative_change",
    "representative_log_magnitude_pearson_r",
    "representative_same_sign_fraction",
    "representative_within_10pct_fraction",
    "representative_median_symmetric_relative_change",
    "representative_p90_symmetric_relative_change",
)

INVENTORY_FIELDS = (
    "model",
    "environment",
    "workload",
    "spec_enabled",
    "status",
    "signature_id",
    "modified_time_utc",
    "manifest_path",
    "flat_jsonl_path",
)

COVERAGE_FIELDS = (
    "model",
    "environment",
    "spec_on_runs",
    "spec_off_runs",
    "strictly_matched_setting_pairs",
    "selected_latest_pairs",
)


@dataclass(frozen=True)
class RunRecord:
    manifest_path: Path
    flat_jsonl_path: Path
    manifest: dict[str, Any]
    spec_enabled: bool
    environment: str
    model_key: str
    model: str
    workload: str
    signature: str
    signature_id: str
    modified_time: float


@dataclass(frozen=True)
class RunPair:
    pair_id: str
    spec_on: RunRecord
    spec_off: RunRecord
    completion_time: float

    @property
    def model(self) -> str:
        return self.spec_on.model

    @property
    def environment(self) -> str:
        return self.spec_on.environment

    @property
    def workload(self) -> str:
        return self.spec_on.workload


@dataclass(frozen=True)
class PairingResult:
    selected_pairs: tuple[RunPair, ...]
    strict_candidates: tuple[RunPair, ...]
    status_by_manifest: dict[Path, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        help=(
            "Root recursively searched for manifests. Repeat as needed; "
            "defaults to exp_results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_artifacts/spec_on_off_slope_curves",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=ENVIRONMENT_ORDER,
        default=list(ENVIRONMENT_ORDER),
    )
    parser.add_argument(
        "--model-regex",
        help="Optional case-insensitive regex applied to the model label/path.",
    )
    parser.add_argument(
        "--step-quantile",
        type=float,
        default=0.5,
        help=(
            "Temporal quantile used to select the representative common "
            "decode step (default: 0.5, the median step)."
        ),
    )
    parser.add_argument(
        "--min-common-layers",
        type=int,
        default=4,
        help="Minimum common layers required in a candidate step.",
    )
    parser.add_argument(
        "--min-layer-coverage",
        type=float,
        default=0.8,
        help=(
            "Candidate steps must have this fraction of the maximum common "
            "layer count observed for the pair (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--all-matched-settings",
        action="store_true",
        help=(
            "Keep every strict matched setting instead of only the latest "
            "complete pair per model/environment/workload."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.step_quantile <= 1.0:
        parser.error("--step-quantile must be in [0, 1]")
    if args.min_common_layers < 1:
        parser.error("--min-common-layers must be positive")
    if not 0.0 < args.min_layer_coverage <= 1.0:
        parser.error("--min-layer-coverage must be in (0, 1]")
    return args


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return format(float(value), ".12g")
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return format(float(stripped), ".12g")
        except ValueError:
            return stripped
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _parse_passthrough_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, list):
        return {}
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = str(args[index])
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            parsed[key] = args[index + 1]
            index += 2
        else:
            parsed[key] = True
            index += 1
    return parsed


def classify_environment(manifest: dict[str, Any], path: Path) -> str | None:
    passthrough = _parse_passthrough_args(manifest.get("passthrough_args"))
    kind = manifest.get(
        "resource_pressure_kind", passthrough.get("resource_pressure_kind")
    )
    if kind is not None:
        normalized = str(kind).strip().lower().replace("-", "_")
        if normalized in {"", "none", "off", "no_throttle", "stable"}:
            return "stable"
        if normalized == "io":
            return "io_contention"
        return None

    lowered = str(path).lower()
    if "io_contention" in lowered or "io_pcie" in lowered:
        return "io_contention"
    if "no_throttle" in lowered:
        return "stable"
    # A normal benchmark with no pressure arguments is a stable run.
    return "stable"


def _model_labels(manifest: dict[str, Any]) -> tuple[str, str]:
    raw = str(manifest.get("model") or "unknown-model").rstrip("/")
    basename = Path(raw).name or raw
    lowered = basename.lower()
    opt_match = re.search(r"opt[-_]?([0-9]+(?:[._][0-9]+)?)b", lowered)
    if opt_match:
        size = opt_match.group(1).replace("_", ".")
        return f"opt-{size}b", f"OPT-{size}B"
    llama_match = re.search(r"llama[-_ ]?2[-_]?([0-9]+)b", lowered)
    if llama_match:
        size = llama_match.group(1)
        return f"llama-2-{size}b", f"Llama-2-{size}B"
    return lowered, basename


def workload_label(manifest: dict[str, Any]) -> str:
    workload = manifest.get("workload")
    if workload:
        return str(workload)
    words = manifest.get("prompt_words")
    try:
        numeric = int(words)
    except (TypeError, ValueError):
        numeric = 0
    if numeric > 0 and numeric % 1000 == 0:
        return f"{numeric // 1000}k"
    if numeric > 0:
        return str(numeric)
    setting = manifest.get("benchmark_setting_name")
    return str(setting or "unspecified")


def build_match_signature(manifest: dict[str, Any], *, environment: str) -> str:
    passthrough = _parse_passthrough_args(manifest.get("passthrough_args"))
    matched: dict[str, Any] = {"environment": environment}
    for field in MATCH_FIELDS:
        value = manifest.get(field)
        if field == "model" and value is not None:
            value = str(value).rstrip("/")
        matched[field] = _canonical_value(value)
    for field in PRESSURE_FIELDS:
        value = manifest.get(field, passthrough.get(field))
        if field == "resource_pressure_kind" and value is None:
            value = "none" if environment == "stable" else "io"
        matched[field] = _canonical_value(value)
    return json.dumps(matched, sort_keys=True, separators=(",", ":"))


def _find_flat_jsonl(run_dir: Path, manifest: dict[str, Any]) -> Path:
    local = sorted(run_dir.glob("*.flat.jsonl"))
    manifest_candidates: list[Path] = []
    for key in ("mfu_flat_jsonl_glob", "component_flat_jsonl_glob"):
        pattern = manifest.get(key)
        if pattern:
            manifest_candidates.extend(Path(item) for item in glob.glob(str(pattern)))
    existing_manifest_candidates = sorted(
        {path for path in manifest_candidates if path.exists()}
    )
    if len(local) == 1:
        return local[0]
    if len(existing_manifest_candidates) == 1:
        return existing_manifest_candidates[0]
    candidates = sorted(set(local) | set(existing_manifest_candidates))
    raise FileNotFoundError(
        f"expected exactly one flat JSONL in {run_dir}, found {len(candidates)}"
    )


def discover_runs(
    roots: Sequence[Path],
    *,
    environments: set[str],
    model_pattern: re.Pattern[str] | None = None,
) -> tuple[list[RunRecord], list[dict[str, str]]]:
    runs: list[RunRecord] = []
    issues: list[dict[str, str]] = []
    seen_manifests: set[Path] = set()
    for root in roots:
        manifest_paths = sorted(root.rglob("manifest.json"))
        if not manifest_paths:
            for flat_path in sorted(root.rglob("*.flat.jsonl")):
                issues.append(
                    {
                        "path": str(flat_path.absolute()),
                        "reason": (
                            "root has no manifest.json with an explicit Spec "
                            "switch; flat trace is not safe to pair"
                        ),
                    }
                )
        for manifest_path in manifest_paths:
            lexical_path = manifest_path.absolute()
            try:
                identity = lexical_path.resolve()
            except OSError:
                identity = lexical_path
            if identity in seen_manifests:
                continue
            seen_manifests.add(identity)
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(lexical_path), "reason": str(error)})
                continue
            spec_enabled = _coerce_bool(
                manifest.get("layer_recompute_async_plan_build")
            )
            if spec_enabled is None:
                continue
            environment = classify_environment(manifest, lexical_path)
            if environment not in environments:
                continue
            model_key, model = _model_labels(manifest)
            if model_pattern and not (
                model_pattern.search(model)
                or model_pattern.search(str(manifest.get("model", "")))
            ):
                continue
            try:
                flat_path = _find_flat_jsonl(manifest_path.parent, manifest)
            except FileNotFoundError as error:
                issues.append({"path": str(lexical_path), "reason": str(error)})
                continue
            signature = build_match_signature(manifest, environment=environment)
            signature_id = hashlib.sha256(signature.encode()).hexdigest()[:12]
            modified_time = max(
                manifest_path.stat().st_mtime,
                flat_path.stat().st_mtime,
            )
            runs.append(
                RunRecord(
                    manifest_path=lexical_path,
                    flat_jsonl_path=flat_path.absolute(),
                    manifest=manifest,
                    spec_enabled=spec_enabled,
                    environment=environment,
                    model_key=model_key,
                    model=model,
                    workload=workload_label(manifest),
                    signature=signature,
                    signature_id=signature_id,
                    modified_time=modified_time,
                )
            )
    return runs, issues


def _latest_complete_pair(group: Sequence[RunRecord]) -> RunPair | None:
    spec_on = [run for run in group if run.spec_enabled]
    spec_off = [run for run in group if not run.spec_enabled]
    if not spec_on or not spec_off:
        return None
    on, off = max(
        ((on_run, off_run) for on_run in spec_on for off_run in spec_off),
        key=lambda item: (
            min(item[0].modified_time, item[1].modified_time),
            -abs(item[0].modified_time - item[1].modified_time),
            max(item[0].modified_time, item[1].modified_time),
        ),
    )
    completion = min(on.modified_time, off.modified_time)
    pair_id = f"{on.model_key}_{on.environment}_{on.workload}_{on.signature_id}"
    return RunPair(pair_id, on, off, completion)


def pair_runs(
    runs: Sequence[RunRecord], *, latest_per_workload: bool = True
) -> PairingResult:
    by_signature: dict[str, list[RunRecord]] = defaultdict(list)
    for run in runs:
        by_signature[run.signature].append(run)

    status: dict[Path, str] = {}
    candidates: list[RunPair] = []
    for group in by_signature.values():
        candidate = _latest_complete_pair(group)
        if candidate is None:
            has_on = any(run.spec_enabled for run in group)
            label = "unpaired_spec_on_only" if has_on else "unpaired_spec_off_only"
            for run in group:
                status[run.manifest_path] = label
            continue
        candidates.append(candidate)
        for run in group:
            status[run.manifest_path] = "duplicate_same_setting"
        status[candidate.spec_on.manifest_path] = "strict_pair_candidate"
        status[candidate.spec_off.manifest_path] = "strict_pair_candidate"

    if latest_per_workload:
        by_coverage: dict[tuple[str, str, str], list[RunPair]] = defaultdict(list)
        for pair in candidates:
            key = (
                pair.spec_on.model_key,
                pair.environment,
                pair.workload,
            )
            by_coverage[key].append(pair)
        selected = [
            max(group, key=lambda pair: pair.completion_time)
            for group in by_coverage.values()
        ]
    else:
        selected = list(candidates)

    selected_ids = {pair.pair_id for pair in selected}
    for pair in candidates:
        pair_status = (
            "selected_latest_pair"
            if pair.pair_id in selected_ids
            else "matched_but_superseded"
        )
        status[pair.spec_on.manifest_path] = pair_status
        status[pair.spec_off.manifest_path] = pair_status

    selected.sort(
        key=lambda pair: (
            pair.spec_on.model_key,
            ENVIRONMENT_ORDER.index(pair.environment),
            _workload_sort_key(pair.workload),
            pair.pair_id,
        )
    )
    candidates.sort(key=lambda pair: pair.pair_id)
    return PairingResult(tuple(selected), tuple(candidates), status)


def _is_decode_only(record: dict[str, Any]) -> bool:
    phase = record.get("phase") or record.get("inference_phase")
    if phase is not None:
        return str(phase).lower().replace("-", "_") == "decode_only"
    for key in ("num_prefill_tokens", "prefill_scheduled_tokens"):
        value = record.get(key)
        if value is not None:
            return int(value) == 0
    total = record.get("total_scheduled_tokens")
    num_reqs = record.get("num_reqs")
    return total is not None and num_reqs is not None and int(total) == int(num_reqs)


def load_gain_snapshots(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    snapshots: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            layer_idx = record.get("layer_idx")
            if layer_idx is None or not _is_decode_only(record):
                continue
            update = record.get("controller_update")
            if not isinstance(update, dict) or update.get("gain_used") is None:
                continue
            gain = float(update["gain_used"])
            if not math.isfinite(gain):
                continue
            step = int(record.get("step", 0))
            layer = int(layer_idx)
            snapshots[(step, layer)] = {
                "step": step,
                "layer_idx": layer,
                "gain": gain,
                "num_reqs": _optional_int(record.get("num_reqs")),
                "total_scheduled_tokens": _optional_int(
                    record.get("total_scheduled_tokens")
                ),
                "sm_state": update.get("sm_state"),
                "action": update.get("action"),
            }
    return snapshots


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _same_schedule(on: dict[str, Any], off: dict[str, Any]) -> bool:
    return (
        on.get("num_reqs"),
        on.get("total_scheduled_tokens"),
    ) == (
        off.get("num_reqs"),
        off.get("total_scheduled_tokens"),
    )


def align_pair_snapshots(
    pair: RunPair,
    spec_on: dict[tuple[int, int], dict[str, Any]],
    spec_off: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    schedule_mismatches = 0
    for key in sorted(set(spec_on) & set(spec_off)):
        on = spec_on[key]
        off = spec_off[key]
        if not _same_schedule(on, off):
            schedule_mismatches += 1
            continue
        on_gain = float(on["gain"])
        off_gain = float(off["gain"])
        denominator = abs(on_gain) + abs(off_gain)
        symmetric_change = (
            2.0 * abs(on_gain - off_gain) / denominator if denominator > 1e-15 else 0.0
        )
        rows.append(
            {
                "pair_id": pair.pair_id,
                "model": pair.model,
                "environment": pair.environment,
                "workload": pair.workload,
                "step": key[0],
                "layer_idx": key[1],
                "num_reqs": on.get("num_reqs"),
                "total_scheduled_tokens": on.get("total_scheduled_tokens"),
                "spec_on_gain_ms_per_block": on_gain,
                "spec_off_gain_ms_per_block": off_gain,
                "spec_on_gain_magnitude_ms_per_block": abs(on_gain),
                "spec_off_gain_magnitude_ms_per_block": abs(off_gain),
                "same_sign": (on_gain == 0.0 and off_gain == 0.0)
                or (on_gain * off_gain > 0.0),
                "absolute_change_ms_per_block": abs(on_gain - off_gain),
                "symmetric_relative_change": symmetric_change,
                "symmetric_relative_change_pct": 100.0 * symmetric_change,
                "spec_on_sm_state": on.get("sm_state"),
                "spec_off_sm_state": off.get("sm_state"),
                "spec_on_action": on.get("action"),
                "spec_off_action": off.get("action"),
            }
        )
    return rows, schedule_mismatches


def select_representative_step(
    rows: Sequence[dict[str, Any]],
    *,
    quantile: float,
    min_common_layers: int,
    min_layer_coverage: float,
) -> tuple[int, list[int], int]:
    layers_by_step: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        layers_by_step[int(row["step"])].add(int(row["layer_idx"]))
    if not layers_by_step:
        raise ValueError("no schedule-matched common decode samples")
    max_layers = max(len(layers) for layers in layers_by_step.values())
    threshold = max(
        min_common_layers,
        int(math.ceil(max_layers * min_layer_coverage)),
    )
    eligible = sorted(
        step for step, layers in layers_by_step.items() if len(layers) >= threshold
    )
    if not eligible:
        threshold = max_layers
        eligible = sorted(
            step for step, layers in layers_by_step.items() if len(layers) == max_layers
        )
    index = int(math.floor(quantile * (len(eligible) - 1) + 0.5))
    return eligible[index], eligible, threshold


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0.0 else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _similarity_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    relative = [float(row["symmetric_relative_change"]) for row in rows]
    log_pairs = [
        (
            math.log10(float(row["spec_on_gain_magnitude_ms_per_block"])),
            math.log10(float(row["spec_off_gain_magnitude_ms_per_block"])),
        )
        for row in rows
        if float(row["spec_on_gain_magnitude_ms_per_block"]) > 0.0
        and float(row["spec_off_gain_magnitude_ms_per_block"]) > 0.0
    ]
    return {
        "log_magnitude_pearson_r": _pearson(
            [item[0] for item in log_pairs],
            [item[1] for item in log_pairs],
        ),
        "same_sign_fraction": (
            sum(bool(row["same_sign"]) for row in rows) / len(rows) if rows else None
        ),
        "within_10pct_fraction": (
            sum(value <= 0.10 for value in relative) / len(relative)
            if relative
            else None
        ),
        "median_symmetric_relative_change": _quantile(relative, 0.50),
        "p90_symmetric_relative_change": _quantile(relative, 0.90),
    }


def summarize_pair(
    pair: RunPair,
    rows: Sequence[dict[str, Any]],
    *,
    representative_step: int,
    representative_step_quantile: float,
    schedule_mismatches: int,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row["step"]) == representative_step]
    all_metrics = _similarity_metrics(rows)
    selected_metrics = _similarity_metrics(selected)
    return {
        "pair_id": pair.pair_id,
        "model": pair.model,
        "environment": pair.environment,
        "workload": pair.workload,
        "common_decode_steps": len({int(row["step"]) for row in rows}),
        "paired_samples": len(rows),
        "schedule_mismatch_samples_dropped": schedule_mismatches,
        "representative_step": representative_step,
        "representative_step_quantile": representative_step_quantile,
        "representative_layers": len(selected),
        **{f"all_{key}": value for key, value in all_metrics.items()},
        **{f"representative_{key}": value for key, value in selected_metrics.items()},
    }


def _workload_sort_key(value: str) -> tuple[float, str]:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)k", value.lower())
    return (float(match.group(1)) if match else math.inf, value)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model"


def plot_selected_curves(
    pairs: Sequence[RunPair],
    selected_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "runkv-spec-onoff-mpl")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    summary_by_pair = {str(row["pair_id"]): row for row in summaries}
    pairs_by_model: dict[str, list[RunPair]] = defaultdict(list)
    for pair in pairs:
        pairs_by_model[pair.model].append(pair)

    written: list[Path] = []
    for model, model_pairs in sorted(pairs_by_model.items()):
        workloads = sorted(
            {pair.workload for pair in model_pairs}, key=_workload_sort_key
        )
        environments = [
            env
            for env in ENVIRONMENT_ORDER
            if any(pair.environment == env for pair in model_pairs)
        ]
        fig, axes = plt.subplots(
            len(workloads),
            len(environments),
            figsize=(6.2 * len(environments), 3.25 * len(workloads) + 1.0),
            squeeze=False,
            constrained_layout=True,
        )
        legend_handles = None
        for row_index, workload in enumerate(workloads):
            for column_index, environment in enumerate(environments):
                ax = axes[row_index][column_index]
                matching = [
                    pair
                    for pair in model_pairs
                    if pair.workload == workload and pair.environment == environment
                ]
                if not matching:
                    ax.set_visible(False)
                    continue
                pair = max(matching, key=lambda item: item.completion_time)
                curve_rows = sorted(
                    rows_by_pair[pair.pair_id], key=lambda row: int(row["layer_idx"])
                )
                xs = [int(row["layer_idx"]) for row in curve_rows]
                on = [
                    float(row["spec_on_gain_magnitude_ms_per_block"])
                    for row in curve_rows
                ]
                off = [
                    float(row["spec_off_gain_magnitude_ms_per_block"])
                    for row in curve_rows
                ]
                handles = ax.plot(
                    xs,
                    on,
                    color="#1f77b4",
                    marker="o",
                    markersize=3.0,
                    linewidth=1.6,
                    label="Spec ON",
                )
                handles += ax.plot(
                    xs,
                    off,
                    color="#d55e00",
                    marker="s",
                    markersize=2.8,
                    linewidth=1.5,
                    linestyle="--",
                    label="Spec OFF",
                )
                legend_handles = handles
                positive = [value for value in on + off if value > 0.0]
                if positive:
                    ax.set_yscale("log")
                summary = summary_by_pair[pair.pair_id]
                environment_title = (
                    "Stable" if environment == "stable" else "IO contention"
                )
                ax.set_title(
                    f"{environment_title} · {workload} · decode step "
                    f"{summary['representative_step']}"
                )
                median_change = summary[
                    "representative_median_symmetric_relative_change"
                ]
                within = summary["representative_within_10pct_fraction"]
                ax.text(
                    0.02,
                    0.96,
                    (
                        f"median symmetric change="
                        f"{100.0 * float(median_change):.1f}%\n"
                        f"within 10%={100.0 * float(within):.1f}%"
                    ),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8.5,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "edgecolor": "none",
                    },
                )
                ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
                ax.set_xlabel("Layer index")
                if column_index == 0:
                    ax.set_ylabel(r"Controller slope magnitude $|\hat{g}|$ (ms/block)")

        if legend_handles:
            fig.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc="outside lower center",
                ncol=2,
                frameon=False,
            )
        fig.suptitle(
            f"{model} — actual Spec ON/OFF controller slope curves",
        )
        stem = output_dir / f"spec_on_off_slope_curves_{_slug(model)}"
        for suffix in ("png", "pdf"):
            path = stem.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=240, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written


def _format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_inventory_rows(
    runs: Sequence[RunRecord], pairing: PairingResult
) -> list[dict[str, Any]]:
    return [
        {
            "model": run.model,
            "environment": run.environment,
            "workload": run.workload,
            "spec_enabled": run.spec_enabled,
            "status": pairing.status_by_manifest.get(run.manifest_path, "not_selected"),
            "signature_id": run.signature_id,
            "modified_time_utc": _format_time(run.modified_time),
            "manifest_path": run.manifest_path,
            "flat_jsonl_path": run.flat_jsonl_path,
        }
        for run in sorted(
            runs,
            key=lambda item: (
                item.model_key,
                item.environment,
                _workload_sort_key(item.workload),
                item.spec_enabled,
                -item.modified_time,
            ),
        )
    ]


def build_coverage_rows(
    runs: Sequence[RunRecord], pairing: PairingResult
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        groups[(run.model, run.environment)].append(run)
    rows: list[dict[str, Any]] = []
    for (model, environment), group in sorted(groups.items()):
        rows.append(
            {
                "model": model,
                "environment": environment,
                "spec_on_runs": sum(run.spec_enabled for run in group),
                "spec_off_runs": sum(not run.spec_enabled for run in group),
                "strictly_matched_setting_pairs": sum(
                    pair.model == model and pair.environment == environment
                    for pair in pairing.strict_candidates
                ),
                "selected_latest_pairs": sum(
                    pair.model == model and pair.environment == environment
                    for pair in pairing.selected_pairs
                ),
            }
        )
    return rows


def _pair_rows(pairs: Sequence[RunPair]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        rows.append(
            {
                "pair_id": pair.pair_id,
                "signature_id": pair.spec_on.signature_id,
                "model": pair.model,
                "environment": pair.environment,
                "workload": pair.workload,
                "completion_time_utc": _format_time(pair.completion_time),
                "spec_on_manifest": pair.spec_on.manifest_path,
                "spec_off_manifest": pair.spec_off.manifest_path,
                "spec_on_flat_jsonl": pair.spec_on.flat_jsonl_path,
                "spec_off_flat_jsonl": pair.spec_off.flat_jsonl_path,
                "matched_config_json": pair.spec_on.signature,
            }
        )
    return rows


def _write_report(
    path: Path,
    *,
    roots: Sequence[Path],
    coverage: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    step_quantile: float,
    plots: Sequence[Path],
) -> None:
    lines = [
        "# Actual Spec ON/OFF slope curves",
        "",
        "Search roots: " + ", ".join(f"`{root}`" for root in roots) + ".",
        "",
        (
            "A pair must match every controller, workload, memory, copy-mode, "
            "and resource-pressure field encoded in `matched_run_pairs.csv`; "
            "only `layer_recompute_async_plan_build` may differ."
        ),
        "",
        (
            "Only decode-only records with equal `(step, layer, num_reqs, "
            "total_scheduled_tokens)` are aligned. The plotted step is the "
            f"temporal q={step_quantile:g} common step, not the step with the "
            "smallest ON/OFF difference."
        ),
        "",
        (
            "These are actual independent ON/OFF runs. They do not represent "
            "the early and late adjacent gain snapshots within one trajectory."
        ),
        "",
        "## Log coverage",
        "",
        "| Model | Environment | ON runs | OFF runs | Strict pairs | Selected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            "| {model} | {environment} | {spec_on_runs} | {spec_off_runs} | "
            "{strictly_matched_setting_pairs} | {selected_latest_pairs} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Direct ON/OFF similarity",
            "",
            ("Symmetric change is `2 * |g_on - g_off| / (|g_on| + |g_off|)`."),
            "",
            "| Model | Environment | Workload | Step | Layers | Median change "
            "| Within 10% | All-step log-r |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        log_r = row["all_log_magnitude_pearson_r"]
        lines.append(
            "| {model} | {environment} | {workload} | "
            "{representative_step} | {representative_layers} | {median:.1f}% | "
            "{within:.1f}% | {log_r} |".format(
                **row,
                median=100.0
                * float(row["representative_median_symmetric_relative_change"]),
                within=100.0 * float(row["representative_within_10pct_fraction"]),
                log_r="n/a" if log_r is None else f"{float(log_r):.3f}",
            )
        )
    if plots:
        lines.extend(["", "## Plots", ""])
        lines.extend(f"- `{plot.name}`" for plot in plots)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    roots = args.search_root or [ROOT / "exp_results"]
    roots = [
        path.expanduser() if path.expanduser().is_absolute() else ROOT / path
        for path in roots
    ]
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_pattern = (
        re.compile(args.model_regex, re.IGNORECASE) if args.model_regex else None
    )
    runs, issues = discover_runs(
        roots,
        environments=set(args.environments),
        model_pattern=model_pattern,
    )
    pairing = pair_runs(runs, latest_per_workload=not args.all_matched_settings)
    inventory = build_inventory_rows(runs, pairing)
    coverage = build_coverage_rows(runs, pairing)
    _write_csv(output_dir / "run_inventory.csv", INVENTORY_FIELDS, inventory)
    _write_csv(output_dir / "data_coverage.csv", COVERAGE_FIELDS, coverage)
    _write_csv(
        output_dir / "matched_run_pairs.csv",
        PAIR_FIELDS,
        _pair_rows(pairing.selected_pairs),
    )
    _write_csv(
        output_dir / "discovery_issues.csv",
        ("path", "reason"),
        issues,
    )

    all_samples: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for pair in pairing.selected_pairs:
        spec_on = load_gain_snapshots(pair.spec_on.flat_jsonl_path)
        spec_off = load_gain_snapshots(pair.spec_off.flat_jsonl_path)
        rows, schedule_mismatches = align_pair_snapshots(pair, spec_on, spec_off)
        representative_step, _, _ = select_representative_step(
            rows,
            quantile=args.step_quantile,
            min_common_layers=args.min_common_layers,
            min_layer_coverage=args.min_layer_coverage,
        )
        selected = [row for row in rows if int(row["step"]) == representative_step]
        all_samples.extend(rows)
        selected_samples.extend(selected)
        summaries.append(
            summarize_pair(
                pair,
                rows,
                representative_step=representative_step,
                representative_step_quantile=args.step_quantile,
                schedule_mismatches=schedule_mismatches,
            )
        )

    _write_csv(output_dir / "paired_slope_samples.csv", SAMPLE_FIELDS, all_samples)
    _write_csv(
        output_dir / "selected_slope_curves.csv", SAMPLE_FIELDS, selected_samples
    )
    _write_csv(output_dir / "curve_similarity_summary.csv", SUMMARY_FIELDS, summaries)

    plots: list[Path] = []
    if pairing.selected_pairs and not args.no_plots:
        plots = plot_selected_curves(
            pairing.selected_pairs,
            selected_samples,
            summaries,
            output_dir,
        )
    _write_report(
        output_dir / "analysis_report.md",
        roots=roots,
        coverage=coverage,
        summaries=summaries,
        step_quantile=args.step_quantile,
        plots=plots,
    )

    print(f"Discovered {len(runs)} eligible Spec runs")
    print(f"Selected {len(pairing.selected_pairs)} strict latest pairs")
    print(f"Wrote {output_dir / 'run_inventory.csv'}")
    print(f"Wrote {output_dir / 'data_coverage.csv'}")
    print(f"Wrote {output_dir / 'matched_run_pairs.csv'}")
    print(f"Wrote {output_dir / 'paired_slope_samples.csv'}")
    print(f"Wrote {output_dir / 'selected_slope_curves.csv'}")
    print(f"Wrote {output_dir / 'curve_similarity_summary.csv'}")
    print(f"Wrote {output_dir / 'analysis_report.md'}")
    for plot in plots:
        print(f"Wrote {plot}")
    if not pairing.selected_pairs:
        raise SystemExit(
            "No strict Spec ON/OFF pairs were found; inspect data_coverage.csv "
            "and run_inventory.csv."
        )


if __name__ == "__main__":
    main()
