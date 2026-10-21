# SPDX-License-Identifier: Apache-2.0

from tools.analyze_spec_slope_stability import (
    build_adjacent_pairs,
    extract_per_layer_rows,
    summarize_pairs,
)


def _record(
    step: int,
    layer: int,
    gain: float | None,
    *,
    decode_only: bool = True,
    imbalance_ms: float = 0.5,
    plan_change_hint: str = "small_delta",
) -> dict:
    return {
        "step": step,
        "layer_idx": layer,
        "num_reqs": 4,
        "total_scheduled_tokens": 4 if decode_only else 64,
        "controller_update": {
            "gain_used": gain,
            "budget_before": 10,
            "budget_after": 11,
            "imbalance_ms": imbalance_ms,
            "sm_state": "tracking",
            "action": "update",
            "plan_change_hint": plan_change_hint,
        },
    }


def test_extract_and_align_adjacent_controller_gains() -> None:
    records = [
        _record(7, 0, None),
        _record(7, 1, -0.10),
        _record(7, 2, -0.11),
        _record(7, 3, -0.12),
        _record(7, 4, -0.13),
        _record(8, 1, -1.0, decode_only=False),
        _record(8, 2, -1.0, decode_only=False),
    ]
    rows = extract_per_layer_rows(
        records,
        scenario="test",
        workload="1k",
        variant="full",
        speculative_enabled=True,
        decode_only_only=True,
    )
    pairs = build_adjacent_pairs(rows, num_layers=5)

    assert len(rows) == 5
    assert len(pairs) == 2
    assert pairs[0]["target_layer_idx"] == 3
    assert pairs[0]["early_source_layer_idx"] == 1
    assert pairs[0]["late_source_layer_idx"] == 2
    assert pairs[0]["early_gain_ms_per_block"] == -0.10
    assert pairs[0]["late_gain_ms_per_block"] == -0.11
    assert pairs[0]["late_stable_successor_eligible"] is False
    assert pairs[0]["full_plan_needed"] is True
    assert pairs[1]["target_layer_idx"] == 4


def test_stable_successor_gate_marks_full_plan_unnecessary() -> None:
    records = [
        _record(7, 1, -0.10),
        _record(7, 2, -0.11, imbalance_ms=0.10, plan_change_hint="unchanged"),
    ]
    rows = extract_per_layer_rows(
        records,
        scenario="test",
        workload="1k",
        variant="full",
        speculative_enabled=True,
        decode_only_only=True,
    )
    pairs = build_adjacent_pairs(rows, num_layers=4, deadband_ms=0.25)

    assert len(pairs) == 1
    assert pairs[0]["late_stable_successor_eligible"] is True
    assert pairs[0]["full_plan_needed"] is False


def test_summary_reports_stable_adjacent_gains() -> None:
    rows = extract_per_layer_rows(
        [_record(1, 1, -0.1), _record(1, 2, -0.1), _record(1, 3, -0.1)],
        scenario="test",
        workload="1k",
        variant="no_async_plan",
        speculative_enabled=False,
        decode_only_only=True,
    )
    pairs = build_adjacent_pairs(rows, num_layers=5)
    summary = summarize_pairs(pairs, workload="1k")

    assert summary["adjacent_pair_count"] == 2
    assert summary["exact_match_fraction"] == 1.0
    assert summary["within_10pct_fraction"] == 1.0
    assert summary["median_absolute_change_ms_per_block"] == 0.0
