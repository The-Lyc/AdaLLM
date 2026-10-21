#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize RunKV component ablations with matched resource settings."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_per_layer_timing import (  # noqa: E402
    extract_inference_loop_seconds,
    extract_kernel_timing_from_sqlite,
    load_nvtx,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

WORKLOADS = ("1k", "2k", "4k", "8k")
VARIANTS = [
    "full",
    "no_async_plan",
    "no_segment_dma",
    "no_segment_dma_uva",
    "no_state_machine",
]
MATCH_KEYS = [
    "model",
    "prefix_blocks",
    "num_prompts",
    "prompt_words",
    "max_tokens",
    "gpu_memory_fraction",
    "gpu_memory_utilization",
    "num_device_buffers",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "planner",
    "max_num_seqs",
    "max_staging_blocks",
    "layer_recompute_replay_allocation_policy",
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
    "resource_pressure_inline_layer_period_ms",
]
FIELDS = [
    "workload",
    "variant",
    "h2d_copy_mode",
    "io_fallback",
    "config_group",
    "prefix_blocks",
    "num_prompts",
    "prompt_words",
    "max_tokens",
    "decode_tokens",
    "gpu_memory_fraction",
    "gpu_memory_utilization",
    "num_device_buffers",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "replay_allocation_policy",
    "baseline_config_match",
    "valid_comparison",
    "comparison_note",
    "decode_duration_s",
    "throughput_duration_source",
    "decode_tokens_per_s",
    "delta_duration_s",
    "delta_decode_tokens_per_s",
    "delta_decode_tokens_per_s_pct",
    "mean_abs_imbalance_ms",
    "p95_abs_imbalance_ms",
    "mean_inter_layer_sync_ms",
    "mean_balance_gap_with_inter_layer_sync_ms",
    "mean_replay_tokens",
    "mean_replay_ratio",
    "avg_gpu_bubble_ms_per_layer",
    "pipeline_bubble_source",
    "prehook_total_ms",
    "prehook_build_plan_ms",
    "prehook_sync_plan_ms",
    "schedule_io_ms",
    "h2d_dma_enqueue_ms",
    "manifest",
]
FINAL_FIELDS = [
    "workload",
    "variant",
    "h2d_copy_mode",
    "end_to_end_throughput_tokens_per_s",
    "throughput_duration_source",
    "pipeline_bubble_ms_per_layer",
    "pipeline_bubble_source",
    "valid_comparison",
]
LONG_METRICS = {
    "decode_duration_s": ("Decode duration", "s"),
    "decode_tokens_per_s": ("Decode throughput", "tokens/s"),
    "mean_abs_imbalance_ms": ("Mean absolute imbalance", "ms"),
    "p95_abs_imbalance_ms": ("P95 absolute imbalance", "ms"),
    "mean_inter_layer_sync_ms": ("Mean inter-layer synchronization", "ms"),
    "mean_balance_gap_with_inter_layer_sync_ms": (
        "Mean |imbalance| plus inter-layer event gap",
        "ms",
    ),
    "mean_replay_tokens": ("Mean replay tokens", "tokens"),
    "mean_replay_ratio": ("Mean replay ratio", "ratio"),
    "avg_gpu_bubble_ms_per_layer": ("Average GPU bubble per layer", "ms"),
    "prehook_total_ms": ("Prehook total", "ms"),
    "prehook_build_plan_ms": ("Prehook build-plan total", "ms"),
    "prehook_sync_plan_ms": ("Synchronous full-plan build total", "ms"),
    "schedule_io_ms": ("Schedule IO total", "ms"),
    "h2d_dma_enqueue_ms": ("H2D DMA enqueue total", "ms"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "exp_results/ablation/no_throttle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exp_results/analysis/component_ablation/no_throttle",
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Analyze one matching run tag instead of the latest run per variant.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Write only end-to-end throughput and pipeline bubble.",
    )
    parser.add_argument(
        "--allow-missing-bubble",
        action="store_true",
        help=(
            "Allow final output without an Nsight kernel timeline; leave the "
            "bubble cell empty and mark its source unavailable."
        ),
    )
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _add_config_columns(row: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key in (
        "prefix_blocks",
        "num_prompts",
        "prompt_words",
        "max_tokens",
        "gpu_memory_fraction",
        "gpu_memory_utilization",
        "num_device_buffers",
        "cpu_memory_gb",
        "cpu_memory_fraction",
    ):
        row[key] = manifest.get(key)
    row["replay_allocation_policy"] = manifest.get(
        "layer_recompute_replay_allocation_policy"
    )
    row["decode_tokens"] = int(manifest["num_prompts"]) * int(manifest["max_tokens"])
    row["config_group"] = (
        f"{manifest['prompt_words']}-"
        f"{manifest['num_prompts']}x{manifest['max_tokens']}-"
        f"f{manifest['gpu_memory_fraction']}"
    )


def _flat_metrics(path_pattern: str | None) -> dict[str, float | None]:
    paths = sorted(glob.glob(path_pattern or ""))
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    # A resumed run may append a complete retry after a failed prefix. Keep the
    # last record for each (step, layer), which is the completed retry emitted
    # later in the file.
    records_by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for record in records:
        step = int(record.get("step", 0))
        layer_idx = record.get("layer_idx")
        if layer_idx is None:
            continue
        records_by_step.setdefault(step, {})[int(layer_idx)] = record
    records = [
        record
        for step_layers in records_by_step.values()
        for record in step_layers.values()
    ]
    imbalance_by_layer: dict[int, list[float]] = {}
    for record in records:
        if int(record.get("step", 0)) < 1:
            continue
        layer_idx = record.get("layer_idx")
        imbalance_ms = record.get("imbalance_ms")
        if layer_idx is None or imbalance_ms is None:
            continue
        imbalance_by_layer.setdefault(int(layer_idx), []).append(
            abs(float(imbalance_ms))
        )
    imbalance = [
        value for values in imbalance_by_layer.values() for value in values
    ]
    layer_abs_means = [
        statistics.fmean(values)
        for values in imbalance_by_layer.values()
        if values
    ]
    # Pipeline bubble has two parts for every layer after L0:
    #
    #   1. absolute IO/compute imbalance, which counts either pipeline stage
    #      becoming idle;
    #   2. the same-stream gap between the previous layer ending and this layer
    #      starting, which captures inter-layer synchronization/scheduling idle.
    #
    # Small negative inter-layer event deltas are timestamp noise, so only the
    # synchronization component is clamped at zero.
    inter_layer_sync_by_layer: dict[int, list[float]] = {}
    pipeline_bubble_by_layer: dict[int, list[float]] = {}
    for step, step_layers in records_by_step.items():
        if step < 1:
            continue
        for layer_idx, record in step_layers.items():
            if layer_idx <= 0 or record.get("imbalance_ms") is None:
                continue
            previous = step_layers.get(layer_idx - 1)
            if previous is None:
                continue
            compute_start = record.get("compute_start_ms_from_anchor")
            previous_compute_end = previous.get("compute_end_ms_from_anchor")
            if compute_start is None or previous_compute_end is None:
                continue
            absolute_imbalance = abs(float(record["imbalance_ms"]))
            inter_layer_sync = max(
                float(compute_start) - float(previous_compute_end), 0.0
            )
            inter_layer_sync_by_layer.setdefault(layer_idx, []).append(
                inter_layer_sync
            )
            pipeline_bubble_by_layer.setdefault(layer_idx, []).append(
                absolute_imbalance + inter_layer_sync
            )

    def _mean_of_layer_means(values_by_layer: dict[int, list[float]]) -> float | None:
        layer_means = [
            statistics.fmean(values)
            for values in values_by_layer.values()
            if values
        ]
        return statistics.fmean(layer_means) if layer_means else None

    replay_tokens = [float(r["replay_token_count"]) for r in records if r.get("replay_token_count") is not None]
    replay_ratio = [float(r["replay_ratio"]) for r in records if r.get("replay_ratio") is not None]
    if imbalance:
        ordered = sorted(imbalance)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    else:
        p95 = None
    return {
        "mean_abs_imbalance_ms": (
            statistics.fmean(layer_abs_means) if layer_abs_means else None
        ),
        "p95_abs_imbalance_ms": p95,
        "mean_inter_layer_sync_ms": _mean_of_layer_means(
            inter_layer_sync_by_layer
        ),
        "mean_balance_gap_with_inter_layer_sync_ms": _mean_of_layer_means(
            pipeline_bubble_by_layer
        ),
        "mean_replay_tokens": statistics.fmean(replay_tokens) if replay_tokens else None,
        "mean_replay_ratio": statistics.fmean(replay_ratio) if replay_ratio else None,
    }


def _prehook_metrics(run_dir: Path) -> dict[str, float | None]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "prehook_timing").glob("prehook_timing_*.jsonl")):
        with path.open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    keys = {
        "prehook_total_ms": "total_ms",
        "prehook_build_plan_ms": "build_plan_ms",
        "prehook_sync_plan_ms": "bp_sync_plan_ms",
        "schedule_io_ms": "schedule_io_ms",
    }
    result: dict[str, float | None] = {
        out: sum(float(r.get(src, 0.0)) for r in records) if records else None
        for out, src in keys.items()
    }
    result["h2d_dma_enqueue_ms"] = (
        sum(
            float(r.get("sio_mseg_dma_ms", 0.0))
            + float(r.get("sio_gather_dma_ms", 0.0))
            + float(r.get("sio_uva_batch_copy_ms", 0.0))
            for r in records
        )
        if records
        else None
    )
    return result


def _step_metrics_duration(path: str | None) -> float | None:
    if not path or not Path(path).exists():
        return None
    starts: list[float] = []
    ends: list[float] = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            start = record.get("step_start_monotonic_s")
            end = record.get("step_end_monotonic_s")
            if start is not None and end is not None:
                starts.append(float(start))
                ends.append(float(end))
    if not starts or not ends:
        return None
    duration = max(ends) - min(starts)
    return duration if duration > 0 else None


def _resource_steps_duration(path: str | None) -> float | None:
    if not path or not Path(path).exists():
        return None
    starts: list[float] = []
    ends: list[float] = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            start = record.get("step_start_s")
            end = record.get("step_end_s")
            if start is not None and end is not None:
                starts.append(float(start))
                ends.append(float(end))
    if not starts or not ends:
        return None
    duration = max(ends) - min(starts)
    return duration if duration > 0 else None


def _timing_metrics(
    sqlite_path: Path,
    step_metrics_path: str | None,
    resource_steps_path: str | None,
) -> dict[str, Any]:
    duration = None
    duration_source = None
    layer_timing: dict[int, dict[str, list[float]]] = {}
    if sqlite_path.exists():
        duration = extract_inference_loop_seconds(load_nvtx(str(sqlite_path)))
        layer_timing = extract_kernel_timing_from_sqlite(str(sqlite_path))
        if duration is not None:
            duration_source = "nsight_inference_loop"
    if duration is None:
        duration = _step_metrics_duration(step_metrics_path)
        if duration is not None:
            duration_source = "step_metrics_span"
    if duration is None:
        duration = _resource_steps_duration(resource_steps_path)
        if duration is not None:
            duration_source = "resource_steps_span"
    bubbles = [
        value
        for metrics in layer_timing.values()
        for value in metrics.get("gpu_bubble", [])
    ]
    return {
        "decode_duration_s": duration,
        "throughput_duration_source": duration_source,
        "avg_gpu_bubble_ms_per_layer": (
            statistics.fmean(bubbles) if bubbles else None
        ),
        "pipeline_bubble_source": (
            "nsight_gpu_span_minus_kernel_active"
            if bubbles
            else "unavailable_no_nsys_kernel_timeline"
        ),
    }


def _latest_variant_manifest(
    input_root: Path,
    workload: str,
    variant: str,
    include_smoke: bool,
    run_tag: str | None = None,
) -> Path | None:
    if run_tag is not None:
        path = input_root / workload / variant / run_tag / "manifest.json"
        if not path.exists():
            return None
        manifest = _load_json(path)
        return path if include_smoke or not manifest.get("smoke", False) else None

    candidates = []
    for path in (input_root / workload / variant).glob("**/manifest.json"):
        manifest = _load_json(path)
        if include_smoke or not manifest.get("smoke", False):
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _variant_row(
    workload: str,
    variant: str,
    path: Path,
    baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(path)
    mismatches = [
        key
        for key in MATCH_KEYS
        if str(manifest.get(key)) != str(baseline_manifest.get(key))
    ]
    sqlite_value = manifest.get("sqlite")
    sqlite_path = (
        Path(sqlite_value) if sqlite_value else path.parent / "runkv.sqlite"
    )
    row: dict[str, Any] = {
        "workload": workload,
        "variant": variant,
        "h2d_copy_mode": manifest.get("h2d_copy_mode"),
        "io_fallback": manifest.get("io_fallback"),
        "baseline_config_match": not mismatches,
        "valid_comparison": not mismatches,
        "comparison_note": (
            "matches full reference"
            if not mismatches
            else f"full reference differs: {', '.join(mismatches)}"
        ),
        "manifest": str(path),
    }
    _add_config_columns(row, manifest)
    row.update(
        _timing_metrics(
            sqlite_path,
            manifest.get("step_metrics_path"),
            manifest.get("resource_pressure_step_log_path"),
        )
    )
    decode_tokens = int(manifest["num_prompts"]) * int(manifest["max_tokens"])
    duration = row["decode_duration_s"]
    row["decode_tokens_per_s"] = decode_tokens / duration if duration else None
    row.update(_flat_metrics(manifest.get("mfu_flat_jsonl_glob")))
    row.update(_prehook_metrics(path.parent))
    if mismatches:
        row["validation_error"] = f"baseline mismatch: {', '.join(mismatches)}"
    return row


def _reference_row(
    input_root: Path,
    workload: str,
    include_smoke: bool,
    run_tag: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prefer a same-version full run, with historical data as a fallback."""
    current_path = _latest_variant_manifest(
        input_root, workload, "full", include_smoke, run_tag
    )
    if current_path is None:
        selector = f" for run tag {run_tag!r}" if run_tag is not None else ""
        raise FileNotFoundError(
            f"No same-version full manifest found for workload {workload}{selector}. "
            "Run the ablation's full variant before analysis."
        )

    manifest = _load_json(current_path)
    row = _variant_row(workload, "full", current_path, manifest)
    row.update(
        {
            "baseline_config_match": True,
            "valid_comparison": True,
            "comparison_note": "same-version full reference",
        }
    )
    return row, manifest


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Component Ablation",
        "",
        "| Workload | Variant | H2D path | Valid | Duration (s) | Decode tok/s | "
        "Delta tok/s | Delta % | Mean |imb| (ms) | P95 |imb| (ms) | "
        "Prehook (ms) | H2D enqueue (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {variant} | {h2d_path} | {valid} | {dur} | {tps} | {delta} | "
            "{pct} | {mean} | {p95} | {prehook} | {h2d} |".format(
                workload=row["workload"],
                variant=row["variant"],
                h2d_path=(
                    row.get("io_fallback") or row.get("h2d_copy_mode") or ""
                ),
                valid=_fmt(row.get("valid_comparison")),
                dur=_fmt(row.get("decode_duration_s")),
                tps=_fmt(row.get("decode_tokens_per_s")),
                delta=_fmt(row.get("delta_decode_tokens_per_s")),
                pct=_fmt(row.get("delta_decode_tokens_per_s_pct")),
                mean=_fmt(row.get("mean_abs_imbalance_ms")),
                p95=_fmt(row.get("p95_abs_imbalance_ms")),
                prehook=_fmt(row.get("prehook_total_ms")),
                h2d=_fmt(row.get("h2d_dma_enqueue_ms")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_long_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "workload",
        "variant",
        "config_group",
        "num_prompts",
        "prompt_words",
        "max_tokens",
        "gpu_memory_fraction",
        "gpu_memory_utilization",
        "baseline_config_match",
        "valid_comparison",
        "metric",
        "metric_label",
        "unit",
        "value",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for metric, (label, unit) in LONG_METRICS.items():
                value = row.get(metric)
                if value is None:
                    continue
                writer.writerow(
                    {
                        "workload": row["workload"],
                        "variant": row["variant"],
                        "config_group": row["config_group"],
                        "num_prompts": row["num_prompts"],
                        "prompt_words": row["prompt_words"],
                        "max_tokens": row["max_tokens"],
                        "gpu_memory_fraction": row["gpu_memory_fraction"],
                        "gpu_memory_utilization": row["gpu_memory_utilization"],
                        "baseline_config_match": row["baseline_config_match"],
                        "valid_comparison": row["valid_comparison"],
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "value": value,
                    }
                )


def _write_final_results(
    rows: list[dict[str, Any]], csv_path: Path, md_path: Path
) -> None:
    final_rows = [
        {
            "workload": row["workload"],
            "variant": row["variant"],
            "h2d_copy_mode": row.get("h2d_copy_mode"),
            "end_to_end_throughput_tokens_per_s": row.get(
                "decode_tokens_per_s"
            ),
            "throughput_duration_source": row.get("throughput_duration_source"),
            "pipeline_bubble_ms_per_layer": row.get(
                "avg_gpu_bubble_ms_per_layer"
            ),
            "pipeline_bubble_source": row.get("pipeline_bubble_source"),
            "valid_comparison": row.get("valid_comparison"),
        }
        for row in rows
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows(final_rows)

    lines = [
        "# Component Ablation Final Results",
        "",
        "| Workload | Variant | H2D mode | End-to-end throughput (tokens/s) | "
        "Throughput source | Pipeline bubble (ms/layer) | Bubble source | Valid |",
        "|---|---|---|---:|---|---:|---|---:|",
    ]
    for row in final_rows:
        lines.append(
            "| {workload} | {variant} | {h2d} | {throughput} | "
            "{throughput_source} | {bubble} | {bubble_source} | {valid} |".format(
                workload=row["workload"],
                variant=row["variant"],
                h2d=row.get("h2d_copy_mode") or "",
                throughput=_fmt(row.get("end_to_end_throughput_tokens_per_s")),
                throughput_source=row.get("throughput_duration_source") or "",
                bubble=_fmt(row.get("pipeline_bubble_ms_per_layer")),
                bubble_source=row.get("pipeline_bubble_source") or "",
                valid=_fmt(row.get("valid_comparison")),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")


def _plot_workload(rows: list[dict[str, Any]], workload: str, out_dir: Path) -> None:
    if not HAS_MPL:
        return
    selected = [r for r in rows if r["workload"] == workload and r.get("decode_tokens_per_s") is not None]
    if not selected:
        return
    labels = [r["variant"] for r in selected]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [r["decode_tokens_per_s"] for r in selected])
    axes[0].set_ylabel("decode tokens/s")
    axes[0].set_title(f"{workload} throughput")
    axes[1].bar(labels, [r["decode_duration_s"] for r in selected])
    axes[1].set_ylabel("seconds")
    axes[1].set_title(f"{workload} decode duration")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{workload}_throughput_latency.png", dpi=180)
    plt.close(fig)


def _plot_mechanisms(rows: list[dict[str, Any]], workload: str, out_dir: Path) -> None:
    if not HAS_MPL:
        return
    selected = [r for r in rows if r["workload"] == workload]
    metrics = [
        ("mean_abs_imbalance_ms", "mean |imbalance| (ms)"),
        ("prehook_build_plan_ms", "prehook plan build total (ms)"),
        ("h2d_dma_enqueue_ms", "H2D enqueue total (ms)"),
        ("avg_gpu_bubble_ms_per_layer", "GPU bubble/layer (ms)"),
    ]
    populated = [
        (field, title)
        for field, title in metrics
        if any(row.get(field) is not None for row in selected)
    ]
    if not populated:
        return
    fig, axes = plt.subplots(1, len(populated), figsize=(5 * len(populated), 4))
    if len(populated) == 1:
        axes = [axes]
    labels = [row["variant"] for row in selected]
    for axis, (field, title) in zip(axes, populated):
        axis.bar(labels, [row.get(field) or 0.0 for row in selected])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"{workload} mechanism metrics")
    fig.tight_layout()
    fig.savefig(out_dir / f"{workload}_mechanisms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for workload in args.workloads:
        baseline, baseline_manifest = _reference_row(
            args.input_root, workload, args.include_smoke, args.run_tag
        )
        rows.append(baseline)
        for variant in VARIANTS[1:]:
            manifest_path = _latest_variant_manifest(
                args.input_root,
                workload,
                variant,
                args.include_smoke,
                args.run_tag,
            )
            if manifest_path is None:
                if args.run_tag is not None or args.final_only:
                    raise FileNotFoundError(
                        f"Missing manifest for {workload}/{variant}"
                        + (
                            f" at run tag {args.run_tag!r}"
                            if args.run_tag is not None
                            else ""
                        )
                    )
                continue
            rows.append(
                _variant_row(workload, variant, manifest_path, baseline_manifest)
            )
        full_tps = baseline["decode_tokens_per_s"]
        full_duration = baseline["decode_duration_s"]
        for row in rows:
            if row["workload"] != workload or row["variant"] == "full":
                continue
            if row.get("decode_tokens_per_s") is not None and full_tps is not None:
                row["delta_decode_tokens_per_s"] = row["decode_tokens_per_s"] - full_tps
                row["delta_decode_tokens_per_s_pct"] = (
                    row["delta_decode_tokens_per_s"] / full_tps * 100
                )
            if row.get("decode_duration_s") is not None and full_duration is not None:
                row["delta_duration_s"] = row["decode_duration_s"] - full_duration

    final_csv_path = args.output_dir / "final_ablation_results.csv"
    final_md_path = args.output_dir / "final_ablation_results.md"
    if args.final_only:
        incomplete = [
            f"{row['workload']}/{row['variant']}"
            for row in rows
            if row.get("decode_tokens_per_s") is None
            or not row.get("valid_comparison")
        ]
        if not args.allow_missing_bubble:
            incomplete.extend(
                f"{row['workload']}/{row['variant']}"
                for row in rows
                if row.get("avg_gpu_bubble_ms_per_layer") is None
            )
        if incomplete:
            raise RuntimeError(
                "Final metrics are missing or invalid for: " + ", ".join(incomplete)
            )
    _write_final_results(rows, final_csv_path, final_md_path)
    print(f"Wrote {final_csv_path}")
    print(f"Wrote {final_md_path}")
    if args.final_only:
        return

    csv_path = args.output_dir / "component_ablation_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.output_dir / "component_ablation_summary.md"
    _write_markdown(rows, md_path)
    long_path = args.output_dir / "component_ablation_metrics_long.csv"
    _write_long_metrics(rows, long_path)
    for workload in args.workloads:
        _plot_workload(rows, workload, args.output_dir)
        _plot_mechanisms(rows, workload, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
