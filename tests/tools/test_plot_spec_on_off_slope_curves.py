# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from tools.plot_spec_on_off_slope_curves import (
    RunPair,
    RunRecord,
    align_pair_snapshots,
    build_match_signature,
    classify_environment,
    load_gain_snapshots,
    pair_runs,
    select_representative_step,
    summarize_pair,
)


def _manifest(*, spec: bool = True, model: str = "/models/opt-2.7b-8k") -> dict:
    return {
        "model": model,
        "model_family": "opt",
        "planner": "feedback",
        "prefix_blocks": "10000000",
        "num_prompts": "32",
        "prompt_words": "1000",
        "max_tokens": "64",
        "fixed_output_length": True,
        "gpu_memory_fraction": "0.9",
        "num_device_buffers": "3",
        "gpu_memory_utilization": "0.7",
        "cpu_memory_gb": "47.9",
        "cpu_memory_fraction": "0.8",
        "h2d_copy_mode": "segment",
        "layer_recompute_use_state_machine": True,
        "layer_recompute_replay_allocation_policy": "concentrate",
        "layer_recompute_async_plan_build": spec,
        "workload": "1k",
        "resource_pressure_kind": "none",
    }


def _run(
    path: str,
    *,
    spec: bool,
    modified: float,
    manifest: dict | None = None,
) -> RunRecord:
    data = dict(manifest or _manifest(spec=spec))
    data["layer_recompute_async_plan_build"] = spec
    signature = build_match_signature(data, environment="stable")
    return RunRecord(
        manifest_path=Path(path) / "manifest.json",
        flat_jsonl_path=Path(path) / "trace.flat.jsonl",
        manifest=data,
        spec_enabled=spec,
        environment="stable",
        model_key="opt-2.7b",
        model="OPT-2.7B",
        workload="1k",
        signature=signature,
        signature_id=str(abs(hash(signature)))[:12],
        modified_time=modified,
    )


def test_environment_classification_reads_manifest_and_passthrough() -> None:
    assert (
        classify_environment({"resource_pressure_kind": "none"}, Path("run"))
        == "stable"
    )
    assert (
        classify_environment(
            {
                "passthrough_args": [
                    "--resource-pressure-kind",
                    "io",
                    "--resource-pressure-pattern",
                    "0:24,16:12",
                ]
            },
            Path("run"),
        )
        == "io_contention"
    )
    assert classify_environment({"resource_pressure_kind": "sm"}, Path("run")) is None


def test_match_signature_ignores_switch_and_run_paths_but_not_settings() -> None:
    on = _manifest(spec=True)
    on.update({"run_tag": "on", "output_dir": "/tmp/on"})
    off = _manifest(spec=False)
    off.update({"run_tag": "off", "output_dir": "/tmp/off"})

    assert build_match_signature(on, environment="stable") == build_match_signature(
        off, environment="stable"
    )

    off["h2d_copy_mode"] = "gather"
    assert build_match_signature(on, environment="stable") != build_match_signature(
        off, environment="stable"
    )


def test_pairing_uses_latest_complete_strict_pair() -> None:
    old_on = _run("old/on", spec=True, modified=10)
    old_off = _run("old/off", spec=False, modified=11)
    new_on = _run("new/on", spec=True, modified=20)
    new_off = _run("new/off", spec=False, modified=21)

    result = pair_runs([old_on, old_off, new_on, new_off])

    assert len(result.selected_pairs) == 1
    pair = result.selected_pairs[0]
    assert pair.spec_on == new_on
    assert pair.spec_off == new_off
    assert result.status_by_manifest[new_on.manifest_path] == "selected_latest_pair"
    assert result.status_by_manifest[old_on.manifest_path] == "duplicate_same_setting"


def _trace_record(
    step: int,
    layer: int,
    gain: float,
    *,
    decode: bool = True,
    num_reqs: int = 4,
) -> dict:
    return {
        "step": step,
        "layer_idx": layer,
        "num_reqs": num_reqs,
        "total_scheduled_tokens": num_reqs if decode else 64,
        "controller_update": {
            "gain_used": gain,
            "sm_state": "tracking",
            "action": "update",
        },
    }


def test_extract_align_and_temporal_median_step(tmp_path: Path) -> None:
    on_path = tmp_path / "on.flat.jsonl"
    off_path = tmp_path / "off.flat.jsonl"
    on_records = [
        _trace_record(step, layer, -0.1 * layer)
        for step in (10, 20, 30)
        for layer in (1, 2, 3)
    ] + [_trace_record(40, 1, -1.0, decode=False)]
    off_records = [
        _trace_record(step, layer, -0.11 * layer)
        for step in (10, 20, 30)
        for layer in (1, 2, 3)
    ]
    on_path.write_text("\n".join(json.dumps(row) for row in on_records) + "\n")
    off_path.write_text("\n".join(json.dumps(row) for row in off_records) + "\n")

    on_run = _run("on", spec=True, modified=10)
    off_run = _run("off", spec=False, modified=11)
    pair = RunPair("pair", on_run, off_run, 10)
    rows, mismatches = align_pair_snapshots(
        pair,
        load_gain_snapshots(on_path),
        load_gain_snapshots(off_path),
    )
    step, eligible, threshold = select_representative_step(
        rows,
        quantile=0.5,
        min_common_layers=2,
        min_layer_coverage=0.8,
    )

    assert mismatches == 0
    assert len(rows) == 9
    assert eligible == [10, 20, 30]
    assert threshold == 3
    assert step == 20
    assert rows[0]["symmetric_relative_change"] < 0.10

    summary = summarize_pair(
        pair,
        rows,
        representative_step=step,
        representative_step_quantile=0.5,
        schedule_mismatches=mismatches,
    )
    assert summary["representative_layers"] == 3
    assert summary["representative_within_10pct_fraction"] == 1.0


def test_alignment_drops_different_runtime_schedules() -> None:
    on_run = _run("on", spec=True, modified=10)
    off_run = _run("off", spec=False, modified=11)
    pair = RunPair("pair", on_run, off_run, 10)
    on = {
        (7, 1): {
            "step": 7,
            "layer_idx": 1,
            "gain": -0.1,
            "num_reqs": 4,
            "total_scheduled_tokens": 4,
        }
    }
    off = {
        (7, 1): {
            "step": 7,
            "layer_idx": 1,
            "gain": -0.1,
            "num_reqs": 3,
            "total_scheduled_tokens": 3,
        }
    }

    rows, mismatches = align_pair_snapshots(pair, on, off)

    assert rows == []
    assert mismatches == 1
