# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_component_ablation import _flat_metrics


def test_pipeline_bubble_adds_absolute_imbalance_and_inter_layer_sync(
    tmp_path: Path,
) -> None:
    records = [
        # Warm-up data must not affect the result.
        {
            "step": 0,
            "layer_idx": 1,
            "imbalance_ms": 1000.0,
            "compute_start_ms_from_anchor": 1010.0,
        },
        {
            "step": 1,
            "layer_idx": 0,
            "compute_end_ms_from_anchor": 10.0,
        },
        # A failed-prefix record superseded by the completed retry below.
        {
            "step": 1,
            "layer_idx": 1,
            "imbalance_ms": 100.0,
            "compute_start_ms_from_anchor": 30.0,
        },
        {
            "step": 1,
            "layer_idx": 1,
            "imbalance_ms": -10.0,
            "compute_start_ms_from_anchor": 12.0,
            "compute_end_ms_from_anchor": 20.0,
        },
        {
            "step": 1,
            "layer_idx": 2,
            "imbalance_ms": 4.0,
            "compute_start_ms_from_anchor": 21.0,
        },
        {
            "step": 2,
            "layer_idx": 0,
            "compute_end_ms_from_anchor": 10.0,
        },
        {
            "step": 2,
            "layer_idx": 1,
            "imbalance_ms": 2.0,
            "compute_start_ms_from_anchor": 11.0,
            "compute_end_ms_from_anchor": 20.0,
        },
        {
            "step": 2,
            "layer_idx": 2,
            "imbalance_ms": -1.0,
            "compute_start_ms_from_anchor": 19.0,
        },
    ]
    path = tmp_path / "component.flat.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    metrics = _flat_metrics(str(path))

    # Sync: mean(mean([2, 1]), mean([1, 0])) = 1.0 ms. The negative L2
    # event delta in step 2 is clamped to zero.
    assert metrics["mean_inter_layer_sync_ms"] == pytest.approx(1.0)
    # |imbalance| = 4.25 ms/layer, but is not the whole bubble by itself.
    assert metrics["mean_abs_imbalance_ms"] == pytest.approx(4.25)
    assert metrics["mean_balance_gap_with_inter_layer_sync_ms"] == pytest.approx(
        5.25
    )
