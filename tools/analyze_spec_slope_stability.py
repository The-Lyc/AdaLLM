#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export and plot adjacent-layer controller gains for spec-plan ablations.

For a target layer ``t``, speculative construction uses the controller state
available after source layer ``t - 2``.  Synchronous construction can wait for
source layer ``t - 1``.  This tool aligns those two gain snapshots as
``early_gain`` and ``late_gain`` and measures how slowly the estimate changes
within a decoding step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep the lexical workspace path instead of resolving its /data symlink so
# analysis outputs remain writable in workspace-sandboxed environments.
ROOT = Path(__file__).absolute().parents[1]
WORKLOADS = ("1k", "2k", "4k", "8k")
VARIANTS = ("full", "no_async_plan")
PAIR_SCOPES = ("full_plan_needed", "all_gain_snapshots")


@dataclass(frozen=True)
class Experiment:
    label: str
    input_root: Path
    run_tag: str


DEFAULT_EXPERIMENTS = (
    Experiment(
        label="no_throttle",
        input_root=ROOT / "exp_results/ablation/no_throttle",
        run_tag="opt27b_ablation_final",
    ),
    Experiment(
        label="io_contention",
        input_root=ROOT / "exp_results/ablation/io_contention",
        run_tag="io_contention_ablation_opt27b_20260810_nsysfree",
    ),
)

PER_LAYER_FIELDS = (
    "scenario",
    "workload",
    "variant",
    "speculative_enabled",
    "step",
    "layer_idx",
    "decode_only",
    "gain_used_ms_per_block",
    "budget_before_blocks",
    "budget_after_blocks",
    "imbalance_ms",
    "sm_state",
    "action",
    "plan_change_hint",
)

PAIR_FIELDS = (
    "scenario",
    "workload",
    "variant",
    "speculative_enabled",
    "step",
    "target_layer_idx",
    "early_source_layer_idx",
    "late_source_layer_idx",
    "early_sm_state",
    "late_sm_state",
    "early_action",
    "late_action",
    "late_plan_change_hint",
    "late_imbalance_ms",
    "late_stable_successor_eligible",
    "full_plan_needed",
    "early_gain_ms_per_block",
    "late_gain_ms_per_block",
    "early_gain_magnitude_ms_per_block",
    "late_gain_magnitude_ms_per_block",
    "absolute_change_ms_per_block",
    "symmetric_relative_change",
    "symmetric_relative_change_pct",
)

SUMMARY_FIELDS = (
    "scenario",
    "workload",
    "variant",
    "speculative_enabled",
    "pair_scope",
    "adjacent_pair_count",
    "pearson_r",
    "log_magnitude_pearson_r",
    "exact_match_fraction",
    "within_5pct_fraction",
    "within_10pct_fraction",
    "median_absolute_change_ms_per_block",
    "p90_absolute_change_ms_per_block",
    "median_symmetric_relative_change",
    "p90_symmetric_relative_change",
    "p95_symmetric_relative_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("LABEL", "INPUT_ROOT", "RUN_TAG"),
        help=(
            "Experiment triple. Repeat to combine scenarios. Defaults to the "
            "matched no-throttle and IO-contention ablations in exp_results."
        ),
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exp_results/analysis/spec_slope_stability",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Include prefill-containing steps; decode-only is the default.",
    )
    parser.add_argument(
        "--deadband-ms",
        type=float,
        default=0.5,
        help=(
            "Deadband used by the state-machine stable-successor gate "
            "(default: ImbalanceControllerConfig's 0.5 ms)."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _find_flat_jsonl(run_dir: Path, manifest: dict[str, Any]) -> Path:
    local_paths = sorted(run_dir.glob("*.flat.jsonl"))
    if len(local_paths) == 1:
        return local_paths[0]
    manifest_glob = manifest.get("mfu_flat_jsonl_glob")
    if manifest_glob:
        import glob

        manifest_paths = [Path(path) for path in sorted(glob.glob(manifest_glob))]
        if len(manifest_paths) == 1:
            return manifest_paths[0]
    raise FileNotFoundError(
        f"Expected exactly one component flat JSONL in {run_dir}, "
        f"found {len(local_paths)}."
    )


def _deduplicated_records(path: Path) -> list[dict[str, Any]]:
    by_step_layer: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            layer_idx = record.get("layer_idx")
            if layer_idx is None:
                continue
            key = (int(record.get("step", 0)), int(layer_idx))
            by_step_layer[key] = record
    return [by_step_layer[key] for key in sorted(by_step_layer)]


def extract_per_layer_rows(
    records: Iterable[dict[str, Any]],
    *,
    scenario: str,
    workload: str,
    variant: str,
    speculative_enabled: bool,
    decode_only_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        decode_only = _is_decode_only(record)
        if decode_only_only and not decode_only:
            continue
        update = record.get("controller_update")
        if not isinstance(update, dict):
            continue
        rows.append(
            {
                "scenario": scenario,
                "workload": workload,
                "variant": variant,
                "speculative_enabled": speculative_enabled,
                "step": int(record.get("step", 0)),
                "layer_idx": int(record["layer_idx"]),
                "decode_only": decode_only,
                "gain_used_ms_per_block": update.get("gain_used"),
                "budget_before_blocks": update.get("budget_before"),
                "budget_after_blocks": update.get("budget_after"),
                "imbalance_ms": update.get("imbalance_ms"),
                "sm_state": update.get("sm_state"),
                "action": update.get("action"),
                "plan_change_hint": update.get("plan_change_hint"),
            }
        )
    return rows


def build_adjacent_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    num_layers: int,
    deadband_ms: float = 0.5,
) -> list[dict[str, Any]]:
    rows_by_step_layer = {
        (int(row["step"]), int(row["layer_idx"])): row for row in rows
    }
    gains = {
        key: float(gain)
        for key, row in rows_by_step_layer.items()
        if (gain := row.get("gain_used_ms_per_block")) is not None
    }
    pairs: list[dict[str, Any]] = []
    for (step, late_layer), late_gain in sorted(gains.items()):
        early_layer = late_layer - 1
        target_layer = late_layer + 1
        early_gain = gains.get((step, early_layer))
        if early_gain is None or target_layer >= num_layers:
            continue
        early = rows_by_step_layer[(step, early_layer)]
        magnitude_sum = abs(early_gain) + abs(late_gain)
        symmetric_change = (
            2.0 * abs(late_gain - early_gain) / magnitude_sum
            if magnitude_sum > 1e-12
            else 0.0
        )
        first = rows_by_step_layer[(step, late_layer)]
        late_hint = first.get("plan_change_hint")
        late_imbalance = first.get("imbalance_ms")
        stable_successor_eligible = (
            late_hint in ("unchanged", "small_delta")
            and late_imbalance is not None
            and abs(float(late_imbalance)) < deadband_ms
        )
        pairs.append(
            {
                "scenario": first["scenario"],
                "workload": first["workload"],
                "variant": first["variant"],
                "speculative_enabled": first["speculative_enabled"],
                "step": step,
                "target_layer_idx": target_layer,
                "early_source_layer_idx": early_layer,
                "late_source_layer_idx": late_layer,
                "early_sm_state": early.get("sm_state"),
                "late_sm_state": first.get("sm_state"),
                "early_action": early.get("action"),
                "late_action": first.get("action"),
                "late_plan_change_hint": late_hint,
                "late_imbalance_ms": late_imbalance,
                "late_stable_successor_eligible": stable_successor_eligible,
                "full_plan_needed": not stable_successor_eligible,
                "early_gain_ms_per_block": early_gain,
                "late_gain_ms_per_block": late_gain,
                "early_gain_magnitude_ms_per_block": abs(early_gain),
                "late_gain_magnitude_ms_per_block": abs(late_gain),
                "absolute_change_ms_per_block": abs(late_gain - early_gain),
                "symmetric_relative_change": symmetric_change,
                "symmetric_relative_change_pct": 100.0 * symmetric_change,
            }
        )
    return pairs


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0 else None


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


def summarize_pairs(
    pairs: Sequence[dict[str, Any]],
    *,
    workload: str,
    pair_scope: str = "all_gain_snapshots",
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Cannot summarize an empty adjacent-pair group.")
    early = [float(row["early_gain_ms_per_block"]) for row in pairs]
    late = [float(row["late_gain_ms_per_block"]) for row in pairs]
    absolute = [float(row["absolute_change_ms_per_block"]) for row in pairs]
    relative = [float(row["symmetric_relative_change"]) for row in pairs]
    log_pairs = [
        (math.log10(abs(early_value)), math.log10(abs(late_value)))
        for early_value, late_value in zip(early, late)
        if early_value != 0.0 and late_value != 0.0
    ]
    first = pairs[0]
    return {
        "scenario": first["scenario"],
        "workload": workload,
        "variant": first["variant"],
        "speculative_enabled": first["speculative_enabled"],
        "pair_scope": pair_scope,
        "adjacent_pair_count": len(pairs),
        "pearson_r": _pearson(early, late),
        "log_magnitude_pearson_r": _pearson(
            [item[0] for item in log_pairs],
            [item[1] for item in log_pairs],
        ),
        "exact_match_fraction": sum(a == b for a, b in zip(early, late))
        / len(pairs),
        "within_5pct_fraction": sum(value <= 0.05 for value in relative)
        / len(pairs),
        "within_10pct_fraction": sum(value <= 0.10 for value in relative)
        / len(pairs),
        "median_absolute_change_ms_per_block": _quantile(absolute, 0.50),
        "p90_absolute_change_ms_per_block": _quantile(absolute, 0.90),
        "median_symmetric_relative_change": _quantile(relative, 0.50),
        "p90_symmetric_relative_change": _quantile(relative, 0.90),
        "p95_symmetric_relative_change": _quantile(relative, 0.95),
    }


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_adjacent_scatter(
    pairs: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    output_dir: Path,
    scenario_order: Sequence[str],
    variant_order: Sequence[str],
    *,
    pair_scope: str,
    file_stem: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive_values = [
        float(row[key])
        for row in pairs
        for key in (
            "early_gain_magnitude_ms_per_block",
            "late_gain_magnitude_ms_per_block",
        )
        if float(row[key]) > 0
    ]
    if not positive_values:
        return
    lower = 10 ** math.floor(math.log10(min(positive_values)))
    upper = 10 ** math.ceil(math.log10(max(positive_values)))

    fig, axes = plt.subplots(
        len(scenario_order),
        len(variant_order),
        figsize=(5.0 * len(variant_order), 4.2 * len(scenario_order)),
        squeeze=False,
        constrained_layout=True,
    )
    last_hexbin = None
    variant_titles = {"full": "Spec ON", "no_async_plan": "Spec OFF"}
    scenario_titles = {
        "no_throttle": "No Throttle",
        "io_contention": "IO Contention",
    }
    for row_idx, scenario in enumerate(scenario_order):
        for col_idx, variant in enumerate(variant_order):
            ax = axes[row_idx][col_idx]
            selected = [
                row
                for row in pairs
                if row["scenario"] == scenario and row["variant"] == variant
                and (
                    pair_scope == "all_gain_snapshots"
                    or bool(row["full_plan_needed"])
                )
                and float(row["early_gain_magnitude_ms_per_block"]) > 0
                and float(row["late_gain_magnitude_ms_per_block"]) > 0
            ]
            if not selected:
                ax.set_visible(False)
                continue
            xs = [float(row["early_gain_magnitude_ms_per_block"]) for row in selected]
            ys = [float(row["late_gain_magnitude_ms_per_block"]) for row in selected]
            last_hexbin = ax.hexbin(
                xs,
                ys,
                xscale="log",
                yscale="log",
                gridsize=48,
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            ax.plot([lower, upper], [lower, upper], "--", color="0.35", linewidth=1)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")
            summary = next(
                row
                for row in summaries
                if row["scenario"] == scenario
                and row["variant"] == variant
                and row["workload"] == "all"
                and row["pair_scope"] == pair_scope
            )
            log_corr = summary["log_magnitude_pearson_r"]
            log_corr_text = "n/a" if log_corr is None else f"{log_corr:.3f}"
            ax.text(
                0.04,
                0.96,
                (
                    f"n={summary['adjacent_pair_count']:,}\n"
                    f"log-r={log_corr_text}\n"
                    f"sym. change <=10%="
                    f"{100 * summary['within_10pct_fraction']:.1f}%"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            scenario_title = scenario_titles.get(
                scenario, scenario.replace("_", " ").title()
            )
            ax.set_title(f"{scenario_title} — {variant_titles[variant]}")
            ax.set_xlabel(r"Early $|\hat{g}_{i-1}|$ (ms/block)")
            ax.set_ylabel(r"Late $|\hat{g}_{i}|$ (ms/block)")
            ax.grid(True, which="both", linewidth=0.35, alpha=0.3)

    if last_hexbin is not None:
        colorbar = fig.colorbar(last_hexbin, ax=axes.ravel().tolist(), shrink=0.82)
        colorbar.set_label("Samples per hexbin (log scale)")
    fig.suptitle(title)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{file_stem}.{suffix}", dpi=220)
    plt.close(fig)


def _write_report(
    path: Path,
    summaries: Sequence[dict[str, Any]],
    *,
    decode_only_only: bool,
    deadband_ms: float,
) -> None:
    lines = [
        "# Speculative Plan Slope Stability",
        "",
        (
            "Rows compare the gain available for early speculative construction "
            "with the adjacent gain available to synchronous construction."
        ),
        "",
        f"Step filter: {'decode-only' if decode_only_only else 'all steps'}.",
        "",
        f"Stable-successor gate deadband: {deadband_ms:g} ms.",
        "",
        (
            "`full_plan_needed` applies the same hint + deadband gate as the "
            "pre-hook and is the scope relevant to speculative-plan consumption. "
            "`all_gain_snapshots` includes every non-null gain snapshot."
        ),
        "",
        (
            "The state-machine controller carries this smoothed estimate across "
            "layers and refreshes it during TRACKING; this measures estimator "
            "state stability, not an independent per-layer plant derivative."
        ),
        "",
        (
            "Symmetric relative change is "
            "`2 * |late - early| / (|late| + |early|)`."
        ),
        "",
        "| Scenario | Scope | Workload | Variant | Pairs | log-r | Exact | Within 10% | P90 symmetric change |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        log_corr = row["log_magnitude_pearson_r"]
        log_corr_text = "" if log_corr is None else f"{log_corr:.3f}"
        lines.append(
            "| {scenario} | {scope} | {workload} | {variant} | {count} | {corr} | "
            "{exact:.1f}% | {within:.1f}% | {p90:.1f}% |".format(
                scenario=row["scenario"],
                scope=row["pair_scope"],
                workload=row["workload"],
                variant=row["variant"],
                count=row["adjacent_pair_count"],
                corr=log_corr_text,
                exact=100 * row["exact_match_fraction"],
                within=100 * row["within_10pct_fraction"],
                p90=100 * row["p90_symmetric_relative_change"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    experiments = (
        [
            Experiment(
                label,
                (
                    Path(input_root).expanduser()
                    if Path(input_root).expanduser().is_absolute()
                    else ROOT / Path(input_root).expanduser()
                ),
                run_tag,
            )
            for label, input_root, run_tag in args.experiment
        ]
        if args.experiment
        else list(DEFAULT_EXPERIMENTS)
    )
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_layer_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for workload in args.workloads:
            for variant in args.variants:
                run_dir = (
                    experiment.input_root / workload / variant / experiment.run_tag
                )
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Missing manifest: {manifest_path}")
                manifest = _load_json(manifest_path)
                speculative_enabled = bool(
                    manifest.get("layer_recompute_async_plan_build")
                )
                expected = variant == "full"
                if speculative_enabled != expected:
                    raise ValueError(
                        f"Unexpected speculative switch for {run_dir}: "
                        f"expected {expected}, got {speculative_enabled}."
                    )
                records = _deduplicated_records(
                    _find_flat_jsonl(run_dir, manifest)
                )
                rows = extract_per_layer_rows(
                    records,
                    scenario=experiment.label,
                    workload=workload,
                    variant=variant,
                    speculative_enabled=speculative_enabled,
                    decode_only_only=not args.all_steps,
                )
                num_layers = max(int(record["layer_idx"]) for record in records) + 1
                per_layer_rows.extend(rows)
                pair_rows.extend(
                    build_adjacent_pairs(
                        rows,
                        num_layers=num_layers,
                        deadband_ms=args.deadband_ms,
                    )
                )

    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        for variant in args.variants:
            for pair_scope in PAIR_SCOPES:
                for workload in [*args.workloads, "all"]:
                    selected = [
                        row
                        for row in pair_rows
                        if row["scenario"] == experiment.label
                        and row["variant"] == variant
                        and (
                            pair_scope == "all_gain_snapshots"
                            or bool(row["full_plan_needed"])
                        )
                        and (workload == "all" or row["workload"] == workload)
                    ]
                    if selected:
                        summaries.append(
                            summarize_pairs(
                                selected,
                                workload=workload,
                                pair_scope=pair_scope,
                            )
                        )

    _write_csv(output_dir / "per_layer_slopes.csv", PER_LAYER_FIELDS, per_layer_rows)
    _write_csv(output_dir / "adjacent_slope_pairs.csv", PAIR_FIELDS, pair_rows)
    _write_csv(output_dir / "slope_stability_summary.csv", SUMMARY_FIELDS, summaries)
    _write_report(
        output_dir / "slope_stability_report.md",
        summaries,
        decode_only_only=not args.all_steps,
        deadband_ms=args.deadband_ms,
    )
    if not args.no_plots:
        _plot_adjacent_scatter(
            pair_rows,
            summaries,
            output_dir,
            [experiment.label for experiment in experiments],
            args.variants,
            pair_scope="full_plan_needed",
            file_stem="adjacent_slope_stability",
            title=(
                "Adjacent-layer controller gain stability when a full plan is needed"
            ),
        )
        _plot_adjacent_scatter(
            pair_rows,
            summaries,
            output_dir,
            [experiment.label for experiment in experiments],
            args.variants,
            pair_scope="all_gain_snapshots",
            file_stem="adjacent_slope_stability_all",
            title="Adjacent-layer controller gain stability — all snapshots",
        )

    print(f"Wrote {output_dir / 'per_layer_slopes.csv'}")
    print(f"Wrote {output_dir / 'adjacent_slope_pairs.csv'}")
    print(f"Wrote {output_dir / 'slope_stability_summary.csv'}")
    print(f"Wrote {output_dir / 'slope_stability_report.md'}")
    if not args.no_plots:
        print(f"Wrote {output_dir / 'adjacent_slope_stability.png'}")
        print(f"Wrote {output_dir / 'adjacent_slope_stability.pdf'}")
        print(f"Wrote {output_dir / 'adjacent_slope_stability_all.png'}")
        print(f"Wrote {output_dir / 'adjacent_slope_stability_all.pdf'}")


if __name__ == "__main__":
    main()
