#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare state-machine TRACKING behavior for speculative plan ON vs OFF.

The controller advances once per layer observation, so all episode lengths in
this analysis are measured in layer observations rather than inference-loop
steps.  A TRACKING episode begins when the emitted state first becomes
``tracking``.  Its final observation is the first following observation whose
emitted state is no longer ``tracking``; that observation is included because
it was processed by ``_step_tracking`` before the state transition.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__:
    from tools.analyze_spec_slope_stability import (
        DEFAULT_EXPERIMENTS,
        ROOT,
        VARIANTS,
        WORKLOADS,
        Experiment,
        _deduplicated_records,
        _find_flat_jsonl,
        _load_json,
        extract_per_layer_rows,
    )
else:
    from analyze_spec_slope_stability import (  # type: ignore[no-redef]
        DEFAULT_EXPERIMENTS,
        ROOT,
        VARIANTS,
        WORKLOADS,
        Experiment,
        _deduplicated_records,
        _find_flat_jsonl,
        _load_json,
        extract_per_layer_rows,
    )

EPISODE_FIELDS = (
    "scenario",
    "workload",
    "variant",
    "speculative_enabled",
    "episode_id",
    "start_step",
    "start_layer_idx",
    "end_step",
    "end_layer_idx",
    "decode_step_count",
    "all_decode_only",
    "left_censored",
    "right_censored",
    "exit_reason",
    "deadband_reached",
    "settled_via_deadband",
    "tracking_state_observations",
    "episode_observations_including_exit",
    "observations_to_first_deadband",
    "outside_deadband_observations_before_first_hit",
    "first_deadband_step",
    "first_deadband_layer_idx",
    "max_consecutive_deadband_observations",
)

SUMMARY_FIELDS = (
    "scenario",
    "workload",
    "variant",
    "speculative_enabled",
    "phase_scope",
    "controller_observations",
    "tracking_state_observations",
    "tracking_observation_fraction",
    "episodes_per_1000_controller_observations",
    "episode_count",
    "complete_episode_count",
    "deadband_reached_episode_count",
    "deadband_reach_fraction",
    "settled_via_deadband_episode_count",
    "settled_fraction_of_complete_episodes",
    "timeout_or_other_episode_count",
    "left_censored_episode_count",
    "right_censored_episode_count",
    "mean_tracking_state_observations_per_episode",
    "median_tracking_state_observations_per_episode",
    "mean_observations_to_first_deadband",
    "median_observations_to_first_deadband",
    "p90_observations_to_first_deadband",
    "mean_episode_observations_to_exit",
    "median_episode_observations_to_exit",
    "p90_episode_observations_to_exit",
    "mean_decode_steps_spanned_per_episode",
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
            "matched no-throttle and IO-contention ablations."
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
        default=ROOT / "analysis_artifacts/spec_tracking_performance",
    )
    parser.add_argument(
        "--deadband-ms",
        type=float,
        default=0.5,
        help="State-machine deadband in milliseconds (default: 0.5).",
    )
    parser.add_argument(
        "--settle-layers",
        type=int,
        default=3,
        help="Consecutive deadband observations required to settle (default: 3).",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Summarize all episodes instead of decode-only episodes.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _append_episode_observation(
    episode: dict[str, Any],
    row: dict[str, Any],
    *,
    deadband_ms: float,
) -> None:
    episode["_observations"].append(row)
    episode["_steps"].add(int(row["step"]))
    episode["all_decode_only"] = episode["all_decode_only"] and bool(
        row["decode_only"]
    )
    if row.get("sm_state") == "tracking":
        episode["tracking_state_observations"] += 1

    imbalance = row.get("imbalance_ms")
    in_deadband = (
        imbalance is not None and abs(float(imbalance)) < float(deadband_ms)
    )
    if in_deadband:
        episode["_consecutive_deadband"] += 1
        episode["max_consecutive_deadband_observations"] = max(
            episode["max_consecutive_deadband_observations"],
            episode["_consecutive_deadband"],
        )
        if episode["observations_to_first_deadband"] is None:
            observation_number = len(episode["_observations"])
            episode["deadband_reached"] = True
            episode["observations_to_first_deadband"] = observation_number
            episode["outside_deadband_observations_before_first_hit"] = (
                observation_number - 1
            )
            episode["first_deadband_step"] = int(row["step"])
            episode["first_deadband_layer_idx"] = int(row["layer_idx"])
    else:
        episode["_consecutive_deadband"] = 0


def _finish_episode(
    episode: dict[str, Any],
    *,
    right_censored: bool,
    settle_layers: int,
) -> dict[str, Any]:
    observations = episode.pop("_observations")
    steps = episode.pop("_steps")
    episode.pop("_consecutive_deadband")
    final = observations[-1]
    episode["end_step"] = int(final["step"])
    episode["end_layer_idx"] = int(final["layer_idx"])
    episode["decode_step_count"] = len(steps)
    episode["right_censored"] = right_censored
    episode["episode_observations_including_exit"] = len(observations)
    if right_censored:
        episode["exit_reason"] = "right_censored"
        episode["settled_via_deadband"] = False
    elif episode["max_consecutive_deadband_observations"] >= settle_layers:
        episode["exit_reason"] = "settled_via_deadband"
        episode["settled_via_deadband"] = True
    else:
        episode["exit_reason"] = "timeout_or_other"
        episode["settled_via_deadband"] = False
    return episode


def build_tracking_episodes(
    rows: Sequence[dict[str, Any]],
    *,
    deadband_ms: float = 0.5,
    settle_layers: int = 3,
) -> list[dict[str, Any]]:
    """Reconstruct contiguous TRACKING episodes from emitted controller states."""
    if settle_layers <= 0:
        raise ValueError("settle_layers must be positive")
    ordered = sorted(rows, key=lambda row: (int(row["step"]), int(row["layer_idx"])))
    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for row_index, row in enumerate(ordered):
        state = row.get("sm_state")
        if active is None:
            if state != "tracking":
                continue
            active = {
                "scenario": row["scenario"],
                "workload": row["workload"],
                "variant": row["variant"],
                "speculative_enabled": row["speculative_enabled"],
                "episode_id": len(episodes),
                "start_step": int(row["step"]),
                "start_layer_idx": int(row["layer_idx"]),
                "all_decode_only": True,
                "left_censored": row_index == 0,
                "deadband_reached": False,
                "tracking_state_observations": 0,
                "observations_to_first_deadband": None,
                "outside_deadband_observations_before_first_hit": None,
                "first_deadband_step": None,
                "first_deadband_layer_idx": None,
                "max_consecutive_deadband_observations": 0,
                "_consecutive_deadband": 0,
                "_observations": [],
                "_steps": set(),
            }
            _append_episode_observation(active, row, deadband_ms=deadband_ms)
            continue

        _append_episode_observation(active, row, deadband_ms=deadband_ms)
        if state != "tracking":
            episodes.append(
                _finish_episode(
                    active,
                    right_censored=False,
                    settle_layers=settle_layers,
                )
            )
            active = None

    if active is not None:
        episodes.append(
            _finish_episode(
                active,
                right_censored=True,
                settle_layers=settle_layers,
            )
        )
    return episodes


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


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_tracking(
    episodes: Sequence[dict[str, Any]],
    observation_rows: Sequence[dict[str, Any]],
    *,
    scenario: str,
    workload: str,
    variant: str,
    phase_scope: str,
) -> dict[str, Any]:
    all_steps = phase_scope == "all_steps"
    selected_rows = [
        row
        for row in observation_rows
        if row["scenario"] == scenario
        and row["variant"] == variant
        and (workload == "all" or row["workload"] == workload)
        and (all_steps or bool(row["decode_only"]))
    ]
    selected_episodes = [
        episode
        for episode in episodes
        if episode["scenario"] == scenario
        and episode["variant"] == variant
        and (workload == "all" or episode["workload"] == workload)
        and (all_steps or bool(episode["all_decode_only"]))
    ]
    if not selected_rows:
        raise ValueError("Cannot summarize an empty controller-observation group.")

    complete = [
        episode for episode in selected_episodes if not episode["right_censored"]
    ]
    uncensored_start = [
        episode for episode in selected_episodes if not episode["left_censored"]
    ]
    reached = [
        episode
        for episode in uncensored_start
        if episode["deadband_reached"]
        and episode["observations_to_first_deadband"] is not None
    ]
    settled = [episode for episode in complete if episode["settled_via_deadband"]]
    tracking_lengths = [
        float(episode["tracking_state_observations"])
        for episode in uncensored_start
    ]
    first_hit_lengths = [
        float(episode["observations_to_first_deadband"]) for episode in reached
    ]
    exit_lengths = [
        float(episode["episode_observations_including_exit"])
        for episode in complete
        if not episode["left_censored"]
    ]
    decode_steps_spanned = [
        float(episode["decode_step_count"]) for episode in uncensored_start
    ]
    tracking_rows = sum(row.get("sm_state") == "tracking" for row in selected_rows)
    first = selected_rows[0]
    return {
        "scenario": scenario,
        "workload": workload,
        "variant": variant,
        "speculative_enabled": first["speculative_enabled"],
        "phase_scope": phase_scope,
        "controller_observations": len(selected_rows),
        "tracking_state_observations": tracking_rows,
        "tracking_observation_fraction": tracking_rows / len(selected_rows),
        "episodes_per_1000_controller_observations": (
            1000.0 * len(selected_episodes) / len(selected_rows)
        ),
        "episode_count": len(selected_episodes),
        "complete_episode_count": len(complete),
        "deadband_reached_episode_count": len(reached),
        "deadband_reach_fraction": (
            len(reached) / len(uncensored_start) if uncensored_start else None
        ),
        "settled_via_deadband_episode_count": len(settled),
        "settled_fraction_of_complete_episodes": (
            len(settled) / len(complete) if complete else None
        ),
        "timeout_or_other_episode_count": sum(
            episode["exit_reason"] == "timeout_or_other" for episode in complete
        ),
        "left_censored_episode_count": sum(
            bool(episode["left_censored"]) for episode in selected_episodes
        ),
        "right_censored_episode_count": sum(
            bool(episode["right_censored"]) for episode in selected_episodes
        ),
        "mean_tracking_state_observations_per_episode": _mean(tracking_lengths),
        "median_tracking_state_observations_per_episode": _quantile(
            tracking_lengths, 0.50
        ),
        "mean_observations_to_first_deadband": _mean(first_hit_lengths),
        "median_observations_to_first_deadband": _quantile(
            first_hit_lengths, 0.50
        ),
        "p90_observations_to_first_deadband": _quantile(first_hit_lengths, 0.90),
        "mean_episode_observations_to_exit": _mean(exit_lengths),
        "median_episode_observations_to_exit": _quantile(exit_lengths, 0.50),
        "p90_episode_observations_to_exit": _quantile(exit_lengths, 0.90),
        "mean_decode_steps_spanned_per_episode": _mean(decode_steps_spanned),
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


def _percent(value: Any) -> str:
    return "" if value is None else f"{100 * float(value):.1f}%"


def _number(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    summaries: Sequence[dict[str, Any]],
    *,
    deadband_ms: float,
    settle_layers: int,
    phase_scope: str,
) -> None:
    lines = [
        "# Speculative Plan TRACKING Performance",
        "",
        (
            "Unit: one controller observation per model layer. The final "
            "observation that exits TRACKING is included in episode-to-exit length."
        ),
        "",
        f"Phase scope: `{phase_scope}`.",
        "",
        (
            f"Deadband: `|imbalance| < {deadband_ms:g} ms`; settling requires "
            f"{settle_layers} consecutive deadband observations."
        ),
        "",
        (
            "First-deadband latency is computed only for non-left-censored "
            "episodes that reached the deadband. Reach and settle rates are "
            "reported alongside it to expose failures and censoring."
        ),
        "",
        "| Scenario | Workload | Spec | Controller obs. | TRACKING obs. | "
        "TRACKING share | Episodes | Mean TRACKING obs./episode | Mean obs. to "
        "first deadband | P90 | Deadband reach | Mean obs. to exit | Settled |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {scenario} | {workload} | {spec} | {observations} | {tracking} | "
            "{share} | {episodes} | {mean_tracking} | {mean_hit} | {p90_hit} | "
            "{reach} | {mean_exit} | {settled} |".format(
                scenario=row["scenario"],
                workload=row["workload"],
                spec="ON" if row["speculative_enabled"] else "OFF",
                observations=row["controller_observations"],
                tracking=row["tracking_state_observations"],
                share=_percent(row["tracking_observation_fraction"]),
                episodes=row["episode_count"],
                mean_tracking=_number(
                    row["mean_tracking_state_observations_per_episode"]
                ),
                mean_hit=_number(row["mean_observations_to_first_deadband"]),
                p90_hit=_number(row["p90_observations_to_first_deadband"]),
                reach=_percent(row["deadband_reach_fraction"]),
                mean_exit=_number(row["mean_episode_observations_to_exit"]),
                settled=_percent(row["settled_fraction_of_complete_episodes"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _plot_summary(
    summaries: Sequence[dict[str, Any]],
    output_dir: Path,
    scenario_order: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aggregate = [row for row in summaries if row["workload"] == "all"]
    variants = ("full", "no_async_plan")
    labels = [scenario.replace("_", " ").title() for scenario in scenario_order]
    x = np.arange(len(scenario_order), dtype=float)
    width = 0.34
    metrics = (
        ("tracking_observation_fraction", 100.0, "TRACKING observations (%)"),
        (
            "mean_tracking_state_observations_per_episode",
            1.0,
            "Mean TRACKING-state observations / episode",
        ),
        (
            "mean_observations_to_first_deadband",
            1.0,
            "Mean observations to first deadband (reached episodes)",
        ),
        (
            "settled_fraction_of_complete_episodes",
            100.0,
            "Complete episodes settled via deadband (%)",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    colors = {"full": "#0072B2", "no_async_plan": "#D55E00"}
    names = {"full": "Spec ON", "no_async_plan": "Spec OFF"}
    for ax, (field, scale, title) in zip(axes.ravel(), metrics):
        for variant_index, variant in enumerate(variants):
            values = []
            for scenario in scenario_order:
                row = next(
                    item
                    for item in aggregate
                    if item["scenario"] == scenario and item["variant"] == variant
                )
                value = row[field]
                values.append(0.0 if value is None else scale * float(value))
            positions = x + (variant_index - 0.5) * width
            bars = ax.bar(
                positions,
                values,
                width,
                label=names[variant],
                color=colors[variant],
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False)
    fig.suptitle("State-machine TRACKING behavior: Spec ON vs OFF")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tracking_performance.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.deadband_ms <= 0:
        raise ValueError("deadband_ms must be positive")
    if args.settle_layers <= 0:
        raise ValueError("settle_layers must be positive")
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

    all_rows: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
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
                    decode_only_only=False,
                )
                all_rows.extend(rows)
                all_episodes.extend(
                    build_tracking_episodes(
                        rows,
                        deadband_ms=args.deadband_ms,
                        settle_layers=args.settle_layers,
                    )
                )

    phase_scope = "all_steps" if args.all_steps else "decode_only"
    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        for variant in args.variants:
            for workload in [*args.workloads, "all"]:
                summaries.append(
                    summarize_tracking(
                        all_episodes,
                        all_rows,
                        scenario=experiment.label,
                        workload=workload,
                        variant=variant,
                        phase_scope=phase_scope,
                    )
                )

    _write_csv(output_dir / "tracking_episodes.csv", EPISODE_FIELDS, all_episodes)
    _write_csv(output_dir / "tracking_summary.csv", SUMMARY_FIELDS, summaries)
    _write_report(
        output_dir / "tracking_report.md",
        summaries,
        deadband_ms=args.deadband_ms,
        settle_layers=args.settle_layers,
        phase_scope=phase_scope,
    )
    if not args.no_plots:
        _plot_summary(
            summaries,
            output_dir,
            [experiment.label for experiment in experiments],
        )

    print(f"Wrote {output_dir / 'tracking_episodes.csv'}")
    print(f"Wrote {output_dir / 'tracking_summary.csv'}")
    print(f"Wrote {output_dir / 'tracking_report.md'}")
    if not args.no_plots:
        print(f"Wrote {output_dir / 'tracking_performance.png'}")
        print(f"Wrote {output_dir / 'tracking_performance.pdf'}")


if __name__ == "__main__":
    main()
