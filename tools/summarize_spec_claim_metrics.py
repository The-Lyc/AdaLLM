#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize the two metrics used by the speculative-plan claim.

The script discovers RunKV manifests, retains the latest strict Spec ON/OFF
pair for each model/environment/workload when one exists, and otherwise keeps
the latest available unpaired run.  It reports:

* the fraction of adjacent decode-only gain snapshots whose symmetric relative
  change is at most 10%; and
* TRACKING episode length and latency to the first deadband observation.

One controller observation is emitted per model layer.  Therefore TRACKING
latencies are layer/controller observations, not inference-loop decode steps.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.analyze_spec_slope_stability import (
        _deduplicated_records,
        build_adjacent_pairs,
        extract_per_layer_rows,
    )
    from tools.analyze_spec_tracking_performance import build_tracking_episodes
    from tools.plot_spec_on_off_slope_curves import (
        ROOT,
        RunRecord,
        discover_runs,
        pair_runs,
    )
else:
    from analyze_spec_slope_stability import (  # type: ignore[no-redef]
        _deduplicated_records,
        build_adjacent_pairs,
        extract_per_layer_rows,
    )
    from analyze_spec_tracking_performance import (  # type: ignore[no-redef]
        build_tracking_episodes,
    )
    from plot_spec_on_off_slope_curves import (  # type: ignore[no-redef]
        ROOT,
        RunRecord,
        discover_runs,
        pair_runs,
    )


@dataclass(frozen=True)
class SelectedRun:
    run: RunRecord
    selection: str


@dataclass(frozen=True)
class MetricCounts:
    controller_observations: int
    adjacent_pairs: int
    adjacent_within_10pct: int
    full_plan_needed_pairs: int
    full_plan_needed_within_10pct: int
    tracking_episodes: int
    tracking_observations: int
    deadband_reached_episodes: int
    observations_to_first_deadband_sum: float


RUN_FIELDS = (
    "model",
    "environment",
    "workload",
    "spec_enabled",
    "selection",
    "manifest_path",
    "flat_jsonl_path",
)

METRIC_FIELDS = (
    "model",
    "environment",
    "workload",
    "spec_enabled",
    "selection",
    "selected_run_count",
    "controller_observations",
    "adjacent_pair_count",
    "adjacent_within_10pct_count",
    "adjacent_within_10pct_fraction",
    "full_plan_needed_pair_count",
    "full_plan_needed_within_10pct_count",
    "full_plan_needed_within_10pct_fraction",
    "tracking_episode_count",
    "mean_tracking_observations_per_episode",
    "deadband_reached_episode_count",
    "deadband_reach_fraction",
    "mean_observations_to_first_deadband",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        help="Root recursively searched for manifests; repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_artifacts/spec_claim_metrics",
    )
    parser.add_argument(
        "--deadband-ms",
        type=float,
        default=0.5,
        help="Controller deadband in milliseconds (default: 0.5).",
    )
    parser.add_argument(
        "--settle-layers",
        type=int,
        default=3,
        help="Consecutive deadband observations used by the controller.",
    )
    return parser.parse_args()


def _workload_key(value: str) -> tuple[float, str]:
    normalized = value.strip().lower()
    if normalized.endswith("k"):
        try:
            return float(normalized[:-1]) * 1000.0, normalized
        except ValueError:
            pass
    try:
        return float(normalized), normalized
    except ValueError:
        return float("inf"), normalized


def select_runs(runs: Sequence[RunRecord]) -> list[SelectedRun]:
    """Select strict pairs where possible and latest singletons otherwise."""
    paired = pair_runs(runs)
    pairs_by_group = {
        (pair.model, pair.environment, pair.workload): pair
        for pair in paired.selected_pairs
    }
    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.model, run.environment, run.workload)].append(run)

    selected: list[SelectedRun] = []
    for key, group in grouped.items():
        pair = pairs_by_group.get(key)
        if pair is not None:
            selected.extend(
                (
                    SelectedRun(pair.spec_on, "strict_pair"),
                    SelectedRun(pair.spec_off, "strict_pair"),
                )
            )
            continue
        for spec_enabled in (True, False):
            candidates = [run for run in group if run.spec_enabled is spec_enabled]
            if candidates:
                selected.append(
                    SelectedRun(
                        max(candidates, key=lambda run: run.modified_time),
                        "latest_unpaired",
                    )
                )

    return sorted(
        selected,
        key=lambda item: (
            item.run.model_key,
            item.run.environment,
            _workload_key(item.run.workload),
            not item.run.spec_enabled,
        ),
    )


def analyze_run(
    selected: SelectedRun,
    *,
    deadband_ms: float,
    settle_layers: int,
) -> MetricCounts:
    run = selected.run
    records = _deduplicated_records(run.flat_jsonl_path)
    if not records:
        raise ValueError(f"No per-layer records in {run.flat_jsonl_path}")
    rows = extract_per_layer_rows(
        records,
        scenario=run.environment,
        workload=run.workload,
        variant="spec_on" if run.spec_enabled else "spec_off",
        speculative_enabled=run.spec_enabled,
        decode_only_only=False,
    )
    decode_rows = [row for row in rows if bool(row["decode_only"])]
    num_layers = max(int(record["layer_idx"]) for record in records) + 1
    adjacent = build_adjacent_pairs(
        decode_rows,
        num_layers=num_layers,
        deadband_ms=deadband_ms,
    )
    full_plan_needed = [row for row in adjacent if row["full_plan_needed"]]

    episodes = build_tracking_episodes(
        rows,
        deadband_ms=deadband_ms,
        settle_layers=settle_layers,
    )
    # Match analyze_spec_tracking_performance.py: exclude prefill-containing
    # and left-censored episodes from latency/length means.
    eligible = [
        episode
        for episode in episodes
        if bool(episode["all_decode_only"]) and not bool(episode["left_censored"])
    ]
    reached = [
        episode
        for episode in eligible
        if bool(episode["deadband_reached"])
        and episode["observations_to_first_deadband"] is not None
    ]
    return MetricCounts(
        controller_observations=len(decode_rows),
        adjacent_pairs=len(adjacent),
        adjacent_within_10pct=sum(
            float(row["symmetric_relative_change"]) <= 0.10 for row in adjacent
        ),
        full_plan_needed_pairs=len(full_plan_needed),
        full_plan_needed_within_10pct=sum(
            float(row["symmetric_relative_change"]) <= 0.10 for row in full_plan_needed
        ),
        tracking_episodes=len(eligible),
        tracking_observations=sum(
            int(episode["tracking_state_observations"]) for episode in eligible
        ),
        deadband_reached_episodes=len(reached),
        observations_to_first_deadband_sum=sum(
            float(episode["observations_to_first_deadband"]) for episode in reached
        ),
    )


def _sum_counts(items: Sequence[MetricCounts]) -> MetricCounts:
    return MetricCounts(
        **{
            field: sum(getattr(item, field) for item in items)
            for field in MetricCounts.__dataclass_fields__
        }
    )


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric_row(
    *,
    model: str,
    environment: str,
    workload: str,
    spec_enabled: bool,
    selection: str,
    selected_run_count: int,
    counts: MetricCounts,
) -> dict[str, Any]:
    return {
        "model": model,
        "environment": environment,
        "workload": workload,
        "spec_enabled": spec_enabled,
        "selection": selection,
        "selected_run_count": selected_run_count,
        "controller_observations": counts.controller_observations,
        "adjacent_pair_count": counts.adjacent_pairs,
        "adjacent_within_10pct_count": counts.adjacent_within_10pct,
        "adjacent_within_10pct_fraction": _fraction(
            counts.adjacent_within_10pct, counts.adjacent_pairs
        ),
        "full_plan_needed_pair_count": counts.full_plan_needed_pairs,
        "full_plan_needed_within_10pct_count": (counts.full_plan_needed_within_10pct),
        "full_plan_needed_within_10pct_fraction": _fraction(
            counts.full_plan_needed_within_10pct,
            counts.full_plan_needed_pairs,
        ),
        "tracking_episode_count": counts.tracking_episodes,
        "mean_tracking_observations_per_episode": _fraction(
            counts.tracking_observations, counts.tracking_episodes
        ),
        "deadband_reached_episode_count": counts.deadband_reached_episodes,
        "deadband_reach_fraction": _fraction(
            counts.deadband_reached_episodes, counts.tracking_episodes
        ),
        "mean_observations_to_first_deadband": (
            counts.observations_to_first_deadband_sum / counts.deadband_reached_episodes
            if counts.deadband_reached_episodes
            else None
        ),
    }


def build_metric_rows(
    selected: Sequence[SelectedRun],
    counts: Sequence[MetricCounts],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detailed: list[dict[str, Any]] = []
    aggregate_groups: dict[
        tuple[str, str, bool], list[tuple[SelectedRun, MetricCounts]]
    ] = defaultdict(list)
    for item, item_counts in zip(selected, counts):
        run = item.run
        detailed.append(
            metric_row(
                model=run.model,
                environment=run.environment,
                workload=run.workload,
                spec_enabled=run.spec_enabled,
                selection=item.selection,
                selected_run_count=1,
                counts=item_counts,
            )
        )
        aggregate_groups[(run.model, run.environment, run.spec_enabled)].append(
            (item, item_counts)
        )

    aggregate: list[dict[str, Any]] = []
    for (model, environment, spec_enabled), group in sorted(aggregate_groups.items()):
        workloads = sorted({item.run.workload for item, _ in group}, key=_workload_key)
        selections = {item.selection for item, _ in group}
        aggregate.append(
            metric_row(
                model=model,
                environment=environment,
                workload="+".join(workloads),
                spec_enabled=spec_enabled,
                selection=(next(iter(selections)) if len(selections) == 1 else "mixed"),
                selected_run_count=len(group),
                counts=_sum_counts([item_counts for _, item_counts in group]),
            )
        )
    return detailed, aggregate


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _format_mean(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def write_report(
    path: Path,
    aggregate: Sequence[dict[str, Any]],
    *,
    roots: Sequence[Path],
    deadband_ms: float,
) -> None:
    lines = [
        "# Speculative-plan claim metrics",
        "",
        "Search roots: " + ", ".join(f"`{root}`" for root in roots) + ".",
        "",
        (
            "Adjacent change uses `2 * |late - early| / "
            "(|late| + |early|)` on decode-only controller gain snapshots."
        ),
        "",
        (
            f"Convergence is the first observation with `|imbalance| < "
            f"{deadband_ms:g} ms`, among non-left-censored decode-only "
            "TRACKING episodes that reach the deadband. One observation is one "
            "model layer, not one inference-loop decode step."
        ),
        "",
        (
            "`strict_pair` means ON and OFF settings match in all manifest "
            "fields used by `plot_spec_on_off_slope_curves.py`; "
            "`latest_unpaired` has no matched opposite-Spec trace."
        ),
        "",
        "| Model | Environment | Spec | Workloads | Pairing | Adjacent n | "
        "Adjacent <=10% | Plan-needed <=10% | TRACKING episodes | Mean TRACKING "
        "obs. | Deadband reach | Mean obs. to first deadband |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {model} | {environment} | {spec} | {workload} | {selection} | "
            "{pairs} | {within} | {plan_within} | {episodes} | "
            "{mean_tracking} | {reach} | {mean_hit} |".format(
                model=row["model"],
                environment=row["environment"],
                spec="ON" if row["spec_enabled"] else "OFF",
                workload=row["workload"],
                selection=row["selection"],
                pairs=row["adjacent_pair_count"],
                within=_format_percent(row["adjacent_within_10pct_fraction"]),
                plan_within=_format_percent(
                    row["full_plan_needed_within_10pct_fraction"]
                ),
                episodes=row["tracking_episode_count"],
                mean_tracking=_format_mean(
                    row["mean_tracking_observations_per_episode"]
                ),
                reach=_format_percent(row["deadband_reach_fraction"]),
                mean_hit=_format_mean(row["mean_observations_to_first_deadband"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.deadband_ms <= 0:
        raise ValueError("deadband_ms must be positive")
    if args.settle_layers <= 0:
        raise ValueError("settle_layers must be positive")
    roots = args.search_root or [ROOT / "exp_results"]
    roots = [
        root.expanduser() if root.expanduser().is_absolute() else ROOT / root
        for root in roots
    ]
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, issues = discover_runs(roots, environments={"stable", "io_contention"})
    selected = select_runs(runs)
    counts = [
        analyze_run(
            item,
            deadband_ms=args.deadband_ms,
            settle_layers=args.settle_layers,
        )
        for item in selected
    ]
    detailed, aggregate = build_metric_rows(selected, counts)

    _write_csv(
        output_dir / "selected_runs.csv",
        RUN_FIELDS,
        (
            {
                "model": item.run.model,
                "environment": item.run.environment,
                "workload": item.run.workload,
                "spec_enabled": item.run.spec_enabled,
                "selection": item.selection,
                "manifest_path": item.run.manifest_path,
                "flat_jsonl_path": item.run.flat_jsonl_path,
            }
            for item in selected
        ),
    )
    _write_csv(output_dir / "metrics_by_workload.csv", METRIC_FIELDS, detailed)
    _write_csv(
        output_dir / "metrics_by_model_environment.csv",
        METRIC_FIELDS,
        aggregate,
    )
    _write_csv(
        output_dir / "discovery_issues.csv",
        ("path", "reason"),
        issues,
    )
    write_report(
        output_dir / "report.md",
        aggregate,
        roots=roots,
        deadband_ms=args.deadband_ms,
    )


if __name__ == "__main__":
    main()
