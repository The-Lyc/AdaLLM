# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.kernels import benchmark_runkv_segment_dma as dma


def _capture_record(
    *,
    layer_idx: int = 16,
    num_segments: int = 2,
) -> dict[str, object]:
    return {
        "layer_idx": layer_idx,
        "all_logical_ids": [2, 3, 4, 8, 9, 12],
        "skip_ids": [4, 12],
        "load_ids": [2, 3, 8, 9],
        "total_blocks": 6,
        "skip_count": 2,
        "load_count": 4,
        "num_segments": num_segments,
        "cpu_cache_shape": [2, 16, 4],
        "cpu_cache_stride": [64, 4, 1],
        "cpu_cache_is_contiguous": True,
        "gpu_buffer_shape": [2, 16, 4],
        "gpu_buffer_stride": [64, 4, 1],
        "gpu_buffer_is_contiguous": True,
        "cpu_cache_dtype": "torch.float16",
    }


def _empty_capture_record() -> dict[str, object]:
    return {
        "layer_idx": 16,
        "all_logical_ids": [5],
        "skip_ids": [5],
        "load_ids": [],
        "total_blocks": 1,
        "skip_count": 1,
        "load_count": 0,
        "num_segments": 0,
        "cpu_cache_shape": [2, 16, 4],
        "cpu_cache_stride": [64, 4, 1],
        "cpu_cache_is_contiguous": True,
        "gpu_buffer_shape": [2, 16, 4],
        "gpu_buffer_stride": [64, 4, 1],
        "gpu_buffer_is_contiguous": True,
        "cpu_cache_dtype": "torch.float16",
    }


def _write_capture(
    path: Path,
    records: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "block_size": 16,
                "blocks_dim": 1,
                "capacity": 16,
                "num_buffers": 2,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reconstructs_real_segment_source_and_destination_gaps(
    tmp_path: Path,
) -> None:
    capture = _write_capture(tmp_path / "capture.json", [_capture_record()])

    record = dma.load_capture(capture)[0]

    assert record.segments == (
        dma.DMASegment(src_start=2, dst_start=0, length=2),
        dma.DMASegment(src_start=8, dst_start=3, length=2),
    )
    assert record.outer_copies == 2
    assert record.bytes_per_outer_block == 8
    assert record.bytes_per_block == 16
    assert record.total_bytes == 64
    assert record.num_cpu_blocks == 16
    assert record.staging_capacity == 16
    assert record.byte_layout_exact is True


def test_control_plans_keep_payload_and_isolate_fragmentation(
    tmp_path: Path,
) -> None:
    capture = _write_capture(tmp_path / "capture.json", [_capture_record()])
    record = dma.load_capture(capture)[0]

    plans = dma.build_method_plans(
        record,
        methods=("real", "packed", "contiguous", "single_dma", "per_block"),
        compact_source=False,
    )

    assert plans["packed"] == (
        dma.DMASegment(src_start=0, dst_start=0, length=2),
        dma.DMASegment(src_start=2, dst_start=2, length=2),
    )
    assert plans["contiguous"] == (dma.DMASegment(src_start=0, dst_start=0, length=4),)
    assert plans["single_dma"] == (
        dma.DMASegment(src_start=0, dst_start=0, length=4),
    )
    assert plans["per_block"] == (
        dma.DMASegment(src_start=2, dst_start=0, length=1),
        dma.DMASegment(src_start=3, dst_start=1, length=1),
        dma.DMASegment(src_start=8, dst_start=3, length=1),
        dma.DMASegment(src_start=9, dst_start=4, length=1),
    )
    assert {
        method: sum(segment.length for segment in plan)
        for method, plan in plans.items()
    } == {
        "real": 4,
        "packed": 4,
        "contiguous": 4,
        "single_dma": 4,
        "per_block": 4,
    }


def test_record_selection_supports_tail_stride_and_explicit_indices(
    tmp_path: Path,
) -> None:
    capture = _write_capture(
        tmp_path / "capture.json",
        [_capture_record(layer_idx=index) for index in range(5)],
    )
    records = dma.load_capture(capture)

    tail = dma.select_records(
        records,
        record_indices=None,
        last_records=3,
        stride=2,
    )
    explicit = dma.select_records(
        records,
        record_indices=(0, -1, 0),
        last_records=0,
        stride=1,
    )

    assert [record.capture_index for record in tail] == [2, 4]
    assert [record.capture_index for record in explicit] == [0, 4]


def test_capture_directory_resolves_its_newest_json(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capture = _write_capture(
        tmp_path / "block_dist_123.json",
        [_capture_record()],
    )

    assert dma.resolve_capture_path(tmp_path) == capture
    assert f"Selected newest capture: {capture}" in capsys.readouterr().out


def test_rejects_capture_when_segment_count_does_not_reconstruct(
    tmp_path: Path,
) -> None:
    capture = _write_capture(
        tmp_path / "capture.json",
        [_capture_record(num_segments=3)],
    )

    with pytest.raises(ValueError, match="captured num_segments=3"):
        dma.load_capture(capture)


def test_describe_only_skips_records_without_h2d_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capture = _write_capture(
        tmp_path / "capture.json",
        [_empty_capture_record(), _capture_record()],
    )
    output_dir = tmp_path / "unused"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_runkv_segment_dma.py",
            "--capture",
            str(capture),
            "--output-dir",
            str(output_dir),
            "--last-records",
            "0",
            "--describe-only",
        ],
    )

    assert dma.main() == 0
    output = capsys.readouterr().out
    assert "Skipping records with no H2D payload: 0" in output
    assert "Selected 1 records" in output
    assert not output_dir.exists()


def test_rejects_noncontiguous_layout_without_explicit_approximation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _capture_record()
    record["cpu_cache_stride"] = [64, 1, 16]
    record["cpu_cache_is_contiguous"] = False
    capture = _write_capture(tmp_path / "capture.json", [record])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_runkv_segment_dma.py",
            "--capture",
            str(capture),
            "--output-dir",
            str(tmp_path / "unused"),
            "--describe-only",
        ],
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        dma.main()


def test_csv_writer_emits_one_header(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"

    dma._write_csv(output, [{"method": "real", "gbps": 10.0}])

    assert output.read_text(encoding="utf-8").splitlines() == [
        "method,gbps",
        "real,10.0",
    ]


def test_aggregate_reports_fragmentation_regimes() -> None:
    aggregate = dma.build_aggregate(
        [
            {"num_segments": 1, "real_to_contiguous_bw_ratio": 0.98},
            {"num_segments": 12, "real_to_contiguous_bw_ratio": 0.81},
            {"num_segments": 80, "real_to_contiguous_bw_ratio": 0.60},
        ]
    )

    assert [bucket["segment_count"] for bucket in aggregate["segment_count_bins"]] == [
        "1",
        "9-16",
        "65+",
    ]
    assert aggregate["segment_count_bins"][1]["real_to_contiguous_bw_ratio"][
        "median"
    ] == pytest.approx(0.81)
    assert aggregate["correlation_segments_vs_real_efficiency"] < 0
