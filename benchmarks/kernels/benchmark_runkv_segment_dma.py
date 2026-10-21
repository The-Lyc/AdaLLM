#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark RunKV segmented H2D copies with real block allocations.

The input is a ``block_dist_*.json`` file emitted by ``PagedBlockMapper`` when
``RUNKV_CAPTURE_BLOCK_DIST=1``.  Each captured record contains the exact
logical blocks needed by a real inference step and the blocks skipped because
replay/recompute will fill their staging slots.

For every selected record this benchmark transfers the same number of bytes
with controlled layouts:

``real``
    The exact source logical-block ranges and destination staging-slot holes
    reconstructed from the capture.  This mirrors RunKV's segment H2D loop.
``packed``
    The same segment lengths and number of memcpy submissions, packed back to
    back in source and destination memory.  Comparing ``real / packed`` tests
    whether address gaps themselves hurt bandwidth.
``contiguous``
    One contiguous logical range with the same total bytes.  Comparing
    ``packed / contiguous`` isolates the cost of small transfers and repeated
    memcpy submission.  It retains the production K/V outer layout, so a
    standard ``[2, blocks, ...]`` cache submits two CUDA copies.

``single_dma``
    One flattened contiguous transfer containing the complete K+V payload.
    This is the strict one-``cudaMemcpyAsync`` ideal requested when comparing
    segment DMA with a literal single DMA, rather than with one logical block
    range in the production layout.

``per_block`` is available as an opt-in worst-case baseline.  GPU time is
measured with CUDA events on a dedicated copy stream; CPU enqueue time is
measured separately around the ``Tensor.copy_`` calls.  For a standard
``[2, num_blocks, ...]`` KV layout, every logical segment produces two H2D
memcpy submissions, matching the production implementation's K/V outer loop.

Example:

.. code-block:: bash

    RUNKV_CAPTURE_BLOCK_DIST=1 \
    RUNKV_CAPTURE_BLOCK_DIST_DIR=/tmp/runkv-block-dist \
      <run the normal RunKV workload>

    python benchmarks/kernels/benchmark_runkv_segment_dma.py \
      --capture /tmp/runkv-block-dist/block_dist_12345.json \
      --last-records 32 --warmup 3 --iterations 20 \
      --output-dir /tmp/runkv-segment-dma
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_METHODS = ("real", "packed", "contiguous", "single_dma", "per_block")


@dataclass(frozen=True)
class DMASegment:
    """One contiguous logical-block transfer."""

    src_start: int
    dst_start: int
    length: int


@dataclass(frozen=True)
class AllocationRecord:
    """Validated allocation geometry reconstructed from one capture record."""

    capture_index: int
    layer_idx: int
    all_logical_ids: tuple[int, ...]
    skip_ids: tuple[int, ...]
    load_ids: tuple[int, ...]
    segments: tuple[DMASegment, ...]
    cpu_cache_shape: tuple[int, ...]
    cpu_cache_stride: tuple[int, ...] | None
    gpu_buffer_shape: tuple[int, ...] | None
    gpu_buffer_stride: tuple[int, ...] | None
    byte_layout_exact: bool | None
    cpu_cache_dtype: str
    blocks_dim: int
    num_cpu_blocks: int
    staging_capacity: int
    outer_copies: int
    bytes_per_outer_block: int
    bytes_per_block: int

    @property
    def total_blocks(self) -> int:
        return len(self.all_logical_ids)

    @property
    def load_blocks(self) -> int:
        return len(self.load_ids)

    @property
    def total_bytes(self) -> int:
        return self.load_blocks * self.bytes_per_block

    @property
    def segment_lengths(self) -> tuple[int, ...]:
        return tuple(segment.length for segment in self.segments)


@dataclass(frozen=True)
class SampleStats:
    mean: float
    median: float
    stddev: float
    p05: float
    p95: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class MethodResult:
    capture_index: int
    layer_idx: int
    method: str
    load_blocks: int
    allocation_segments: int
    logical_dma_ranges: int
    cuda_memcpy_submissions: int
    total_bytes: int
    allocation_min_segment_blocks: int
    allocation_median_segment_blocks: float
    allocation_p95_segment_blocks: float
    allocation_max_segment_blocks: int
    host_enqueue_ms: SampleStats
    gpu_elapsed_ms: SampleStats
    effective_gbps: float


def _percentile(values: Sequence[float | int], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_stats(values: Sequence[float]) -> SampleStats:
    if not values:
        raise ValueError("values must not be empty")
    return SampleStats(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stddev=statistics.pstdev(values),
        p05=_percentile(values, 0.05),
        p95=_percentile(values, 0.95),
        minimum=min(values),
        maximum=max(values),
    )


def _torch_dtype_size(dtype_name: str) -> int:
    short_name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, short_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported captured dtype: {dtype_name!r}")
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Unsupported captured dtype: {dtype_name!r}") from error


def coalesce_load_ids(
    all_logical_ids: Sequence[int],
    load_ids: Sequence[int],
) -> tuple[DMASegment, ...]:
    """Reproduce ``PagedBlockMapper.load_layer_async`` segment formation."""
    if not load_ids:
        return ()

    mapping = {logical_id: slot for slot, logical_id in enumerate(all_logical_ids)}
    missing = [logical_id for logical_id in load_ids if logical_id not in mapping]
    if missing:
        raise ValueError(
            f"load_ids contain blocks absent from all_logical_ids: {missing}"
        )

    segments: list[DMASegment] = []
    src_start = int(load_ids[0])
    dst_start = mapping[src_start]
    previous_src = src_start
    previous_dst = dst_start
    length = 1

    for logical_id_raw in load_ids[1:]:
        logical_id = int(logical_id_raw)
        destination = mapping[logical_id]
        if logical_id == previous_src + 1 and destination == previous_dst + 1:
            length += 1
        else:
            segments.append(DMASegment(src_start, dst_start, length))
            src_start = logical_id
            dst_start = destination
            length = 1
        previous_src = logical_id
        previous_dst = destination

    segments.append(DMASegment(src_start, dst_start, length))
    return tuple(segments)


def _parse_capture_record(
    raw: dict[str, Any],
    *,
    capture_index: int,
    blocks_dim: int,
    staging_capacity: int,
) -> AllocationRecord:
    all_logical_ids = tuple(int(value) for value in raw["all_logical_ids"])
    skip_ids = tuple(int(value) for value in raw.get("skip_ids", []))
    load_ids = tuple(int(value) for value in raw["load_ids"])

    if tuple(sorted(set(all_logical_ids))) != all_logical_ids:
        raise ValueError(
            f"record {capture_index}: all_logical_ids must be sorted and unique"
        )
    if tuple(sorted(set(load_ids))) != load_ids:
        raise ValueError(f"record {capture_index}: load_ids must be sorted and unique")

    skip_set = set(skip_ids)
    expected_load_ids = tuple(
        logical_id for logical_id in all_logical_ids if logical_id not in skip_set
    )
    if load_ids != expected_load_ids:
        raise ValueError(
            f"record {capture_index}: load_ids do not equal all_logical_ids - skip_ids"
        )

    if int(raw["total_blocks"]) != len(all_logical_ids):
        raise ValueError(f"record {capture_index}: total_blocks is inconsistent")
    if int(raw["load_count"]) != len(load_ids):
        raise ValueError(f"record {capture_index}: load_count is inconsistent")
    if staging_capacity <= 0 or len(all_logical_ids) > staging_capacity:
        raise ValueError(
            f"record {capture_index}: {len(all_logical_ids)} allocated blocks "
            f"do not fit staging capacity {staging_capacity}"
        )

    shape = tuple(int(value) for value in raw["cpu_cache_shape"])
    if not shape or not 0 <= blocks_dim < len(shape):
        raise ValueError(
            f"record {capture_index}: invalid blocks_dim={blocks_dim} for shape={shape}"
        )
    if shape[blocks_dim] <= 0:
        raise ValueError(
            f"record {capture_index}: CPU block dimension must be positive"
        )
    if all_logical_ids and all_logical_ids[-1] >= shape[blocks_dim]:
        raise ValueError(
            f"record {capture_index}: logical block {all_logical_ids[-1]} exceeds "
            f"CPU cache capacity {shape[blocks_dim]}"
        )

    layout_keys = (
        "cpu_cache_stride",
        "cpu_cache_is_contiguous",
        "gpu_buffer_shape",
        "gpu_buffer_stride",
        "gpu_buffer_is_contiguous",
    )
    present_layout_keys = [key for key in layout_keys if key in raw]
    if present_layout_keys and len(present_layout_keys) != len(layout_keys):
        raise ValueError(f"record {capture_index}: incomplete captured layout metadata")
    if present_layout_keys:
        cpu_cache_stride = tuple(int(value) for value in raw["cpu_cache_stride"])
        gpu_buffer_shape = tuple(int(value) for value in raw["gpu_buffer_shape"])
        gpu_buffer_stride = tuple(int(value) for value in raw["gpu_buffer_stride"])
        if not (
            len(cpu_cache_stride)
            == len(gpu_buffer_shape)
            == len(gpu_buffer_stride)
            == len(shape)
        ):
            raise ValueError(
                f"record {capture_index}: captured shape/stride ranks differ"
            )
        if gpu_buffer_shape[blocks_dim] != staging_capacity:
            raise ValueError(
                f"record {capture_index}: GPU buffer block dimension "
                f"{gpu_buffer_shape[blocks_dim]} differs from captured capacity "
                f"{staging_capacity}"
            )
        for dim, (cpu_size, gpu_size) in enumerate(zip(shape, gpu_buffer_shape)):
            if dim != blocks_dim and cpu_size != gpu_size:
                raise ValueError(
                    f"record {capture_index}: CPU/GPU shape mismatch at dimension {dim}"
                )
        cpu_is_contiguous = raw["cpu_cache_is_contiguous"]
        gpu_is_contiguous = raw["gpu_buffer_is_contiguous"]
        if not isinstance(cpu_is_contiguous, bool) or not isinstance(
            gpu_is_contiguous, bool
        ):
            raise ValueError(
                f"record {capture_index}: contiguity flags must be booleans"
            )
        byte_layout_exact = cpu_is_contiguous and gpu_is_contiguous
    else:
        cpu_cache_stride = None
        gpu_buffer_shape = None
        gpu_buffer_stride = None
        byte_layout_exact = None

    dtype_name = str(raw["cpu_cache_dtype"])
    element_size = _torch_dtype_size(dtype_name)
    outer_copies = math.prod(shape[:blocks_dim]) if blocks_dim else 1
    inner_elements = math.prod(shape[blocks_dim + 1 :])
    bytes_per_outer_block = inner_elements * element_size
    bytes_per_block = outer_copies * bytes_per_outer_block
    if bytes_per_outer_block <= 0:
        raise ValueError(f"record {capture_index}: bytes per block must be positive")

    segments = coalesce_load_ids(all_logical_ids, load_ids)
    if int(raw["num_segments"]) != len(segments):
        raise ValueError(
            f"record {capture_index}: captured num_segments={raw['num_segments']} "
            f"but reconstructed {len(segments)}"
        )

    return AllocationRecord(
        capture_index=capture_index,
        layer_idx=int(raw["layer_idx"]),
        all_logical_ids=all_logical_ids,
        skip_ids=skip_ids,
        load_ids=load_ids,
        segments=segments,
        cpu_cache_shape=shape,
        cpu_cache_stride=cpu_cache_stride,
        gpu_buffer_shape=gpu_buffer_shape,
        gpu_buffer_stride=gpu_buffer_stride,
        byte_layout_exact=byte_layout_exact,
        cpu_cache_dtype=dtype_name,
        blocks_dim=blocks_dim,
        num_cpu_blocks=shape[blocks_dim],
        staging_capacity=staging_capacity,
        outer_copies=outer_copies,
        bytes_per_outer_block=bytes_per_outer_block,
        bytes_per_block=bytes_per_block,
    )


def load_capture(path: Path) -> list[AllocationRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid RunKV block-distribution capture: {path}")
    blocks_dim = int(payload["blocks_dim"])
    staging_capacity = int(payload["capacity"])
    records = [
        _parse_capture_record(
            raw,
            capture_index=index,
            blocks_dim=blocks_dim,
            staging_capacity=staging_capacity,
        )
        for index, raw in enumerate(payload["records"])
    ]
    if not records:
        raise ValueError(f"Capture contains no records: {path}")
    return records


def resolve_capture_path(path: Path) -> Path:
    """Resolve an explicit capture file or the newest capture in a directory."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = list(path.glob("block_dist_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"Capture directory contains no block_dist_*.json files: {path}"
            )
        selected = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
        print(f"Selected newest capture: {selected}")
        return selected

    candidates = (
        list(path.parent.glob("block_dist_*.json")) if path.parent.is_dir() else []
    )
    hint = ""
    if candidates:
        newest = max(
            candidates,
            key=lambda candidate: candidate.stat().st_mtime_ns,
        )
        hint = f" Newest available capture is {newest}."
    raise FileNotFoundError(
        f"Capture does not exist: {path}.{hint} "
        "Pass that file or its containing directory to --capture."
    )


def select_records(
    records: Sequence[AllocationRecord],
    *,
    record_indices: Sequence[int] | None,
    last_records: int,
    stride: int,
) -> list[AllocationRecord]:
    if stride <= 0:
        raise ValueError("record stride must be positive")
    if last_records < 0:
        raise ValueError("last_records must be non-negative")

    if record_indices is not None:
        selected: list[AllocationRecord] = []
        seen: set[int] = set()
        for raw_index in record_indices:
            index = raw_index if raw_index >= 0 else len(records) + raw_index
            if not 0 <= index < len(records):
                raise ValueError(
                    f"record index {raw_index} is outside capture with "
                    f"{len(records)} records"
                )
            if index not in seen:
                selected.append(records[index])
                seen.add(index)
        return selected

    base = list(records[-last_records:]) if last_records else list(records)
    return base[::stride]


def packed_plan(record: AllocationRecord) -> tuple[DMASegment, ...]:
    cursor = 0
    packed: list[DMASegment] = []
    for length in record.segment_lengths:
        packed.append(DMASegment(cursor, cursor, length))
        cursor += length
    return tuple(packed)


def contiguous_plan(record: AllocationRecord) -> tuple[DMASegment, ...]:
    if not record.load_blocks:
        return ()
    return (DMASegment(0, 0, record.load_blocks),)


def per_block_plan(segments: Sequence[DMASegment]) -> tuple[DMASegment, ...]:
    return tuple(
        DMASegment(
            src_start=segment.src_start + offset,
            dst_start=segment.dst_start + offset,
            length=1,
        )
        for segment in segments
        for offset in range(segment.length)
    )


def compact_real_plan(record: AllocationRecord) -> tuple[DMASegment, ...]:
    """Preserve real segment lengths with one source-block gap between runs."""
    source_cursor = 0
    compact: list[DMASegment] = []
    for segment in record.segments:
        compact.append(DMASegment(source_cursor, segment.dst_start, segment.length))
        source_cursor += segment.length + 1
    return tuple(compact)


def build_method_plans(
    record: AllocationRecord,
    *,
    methods: Sequence[str],
    compact_source: bool,
) -> dict[str, tuple[DMASegment, ...]]:
    real = compact_real_plan(record) if compact_source else record.segments
    available = {
        "real": real,
        "packed": packed_plan(record),
        "contiguous": contiguous_plan(record),
        # The block plan is only used for common allocation accounting.  The
        # measurement path flattens both outer K/V slices and issues one copy
        # for the full ``record.total_bytes`` payload.
        "single_dma": contiguous_plan(record),
        "per_block": per_block_plan(real),
    }
    return {method: available[method] for method in methods}


def _issue_segments(
    destination: torch.Tensor,
    source: torch.Tensor,
    segments: Sequence[DMASegment],
    outer_copies: int,
) -> None:
    for outer in range(outer_copies):
        for segment in segments:
            destination[
                outer,
                segment.dst_start : segment.dst_start + segment.length,
            ].copy_(
                source[
                    outer,
                    segment.src_start : segment.src_start + segment.length,
                ],
                non_blocking=True,
            )


def _measure_once(
    *,
    destination: torch.Tensor,
    source: torch.Tensor,
    segments: Sequence[DMASegment],
    outer_copies: int,
    stream: torch.cuda.Stream,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
    flat_copy: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[float, float]:
    with torch.cuda.stream(stream):
        start_event.record(stream)
        host_start_ns = time.perf_counter_ns()
        if flat_copy is None:
            _issue_segments(destination, source, segments, outer_copies)
        else:
            flat_destination, flat_source = flat_copy
            flat_destination.copy_(flat_source, non_blocking=True)
        host_elapsed_ms = (time.perf_counter_ns() - host_start_ns) / 1e6
        end_event.record(stream)
    end_event.synchronize()
    return host_elapsed_ms, start_event.elapsed_time(end_event)


def benchmark_record(
    record: AllocationRecord,
    *,
    plans: dict[str, tuple[DMASegment, ...]],
    destination: torch.Tensor,
    source: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> list[MethodResult]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    methods = list(plans)
    if not methods:
        raise ValueError("at least one benchmark method is required")

    single_dma_copy = (
        destination.view(-1)[: record.total_bytes],
        source.view(-1)[: record.total_bytes],
    )

    for method in methods:
        for _ in range(warmup):
            _measure_once(
                destination=destination,
                source=source,
                segments=plans[method],
                outer_copies=record.outer_copies,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                flat_copy=single_dma_copy if method == "single_dma" else None,
            )

    host_samples = {method: [] for method in methods}
    gpu_samples = {method: [] for method in methods}
    # Rotate method order to reduce monotonic temperature/order bias.
    for iteration in range(iterations):
        offset = iteration % len(methods)
        iteration_order = methods[offset:] + methods[:offset]
        for method in iteration_order:
            host_ms, gpu_ms = _measure_once(
                destination=destination,
                source=source,
                segments=plans[method],
                outer_copies=record.outer_copies,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                flat_copy=single_dma_copy if method == "single_dma" else None,
            )
            host_samples[method].append(host_ms)
            gpu_samples[method].append(gpu_ms)

    allocation_lengths = record.segment_lengths
    results: list[MethodResult] = []
    for method in methods:
        host_stats = _sample_stats(host_samples[method])
        gpu_stats = _sample_stats(gpu_samples[method])
        if gpu_stats.median <= 0:
            raise RuntimeError(
                f"CUDA event returned non-positive elapsed time for {method}"
            )
        results.append(
            MethodResult(
                capture_index=record.capture_index,
                layer_idx=record.layer_idx,
                method=method,
                load_blocks=record.load_blocks,
                allocation_segments=len(record.segments),
                logical_dma_ranges=len(plans[method]),
                cuda_memcpy_submissions=(
                    1
                    if method == "single_dma"
                    else record.outer_copies * len(plans[method])
                ),
                total_bytes=record.total_bytes,
                allocation_min_segment_blocks=min(allocation_lengths),
                allocation_median_segment_blocks=statistics.median(allocation_lengths),
                allocation_p95_segment_blocks=_percentile(allocation_lengths, 0.95),
                allocation_max_segment_blocks=max(allocation_lengths),
                host_enqueue_ms=host_stats,
                gpu_elapsed_ms=gpu_stats,
                effective_gbps=record.total_bytes / gpu_stats.median / 1e6,
            )
        )
    return results


def _ratio(left: MethodResult, right: MethodResult) -> float:
    return left.effective_gbps / right.effective_gbps


def build_comparisons(results: Sequence[MethodResult]) -> list[dict[str, Any]]:
    by_record: dict[int, dict[str, MethodResult]] = {}
    for result in results:
        by_record.setdefault(result.capture_index, {})[result.method] = result

    comparisons: list[dict[str, Any]] = []
    for capture_index, methods in sorted(by_record.items()):
        sample = next(iter(methods.values()))
        row: dict[str, Any] = {
            "capture_index": capture_index,
            "layer_idx": sample.layer_idx,
            "load_blocks": sample.load_blocks,
            "num_segments": sample.allocation_segments,
            "mean_segment_blocks": (sample.load_blocks / sample.allocation_segments),
            "median_segment_blocks": sample.allocation_median_segment_blocks,
            "p95_segment_blocks": sample.allocation_p95_segment_blocks,
            "total_mib": sample.total_bytes / 2**20,
        }
        for method, result in methods.items():
            row[f"{method}_gbps"] = result.effective_gbps
            row[f"{method}_gpu_median_ms"] = result.gpu_elapsed_ms.median
            row[f"{method}_host_median_ms"] = result.host_enqueue_ms.median
            row[f"{method}_cuda_memcpy_submissions"] = result.cuda_memcpy_submissions
            row[f"{method}_mean_memcpy_kib"] = (
                result.total_bytes / result.cuda_memcpy_submissions / 2**10
            )
        if "real" in methods and "contiguous" in methods:
            row["real_to_contiguous_bw_ratio"] = _ratio(
                methods["real"], methods["contiguous"]
            )
            row["fragmentation_penalty_pct"] = (
                1.0 - row["real_to_contiguous_bw_ratio"]
            ) * 100.0
        if "real" in methods and "packed" in methods:
            row["real_to_packed_bw_ratio"] = _ratio(methods["real"], methods["packed"])
            row["address_gap_penalty_pct"] = (
                1.0 - row["real_to_packed_bw_ratio"]
            ) * 100.0
        if "packed" in methods and "contiguous" in methods:
            row["packed_to_contiguous_bw_ratio"] = _ratio(
                methods["packed"], methods["contiguous"]
            )
            row["submission_size_penalty_pct"] = (
                1.0 - row["packed_to_contiguous_bw_ratio"]
            ) * 100.0
        if "real" in methods and "single_dma" in methods:
            row["real_to_single_dma_bw_ratio"] = _ratio(
                methods["real"], methods["single_dma"]
            )
            row["single_dma_baseline_penalty_pct"] = (
                1.0 - row["real_to_single_dma_bw_ratio"]
            ) * 100.0
        if "contiguous" in methods and "single_dma" in methods:
            row["contiguous_to_single_dma_bw_ratio"] = _ratio(
                methods["contiguous"], methods["single_dma"]
            )
            row["kv_outer_split_penalty_pct"] = (
                1.0 - row["contiguous_to_single_dma_bw_ratio"]
            ) * 100.0
        comparisons.append(row)
    return comparisons


def _aggregate_metric(
    comparisons: Sequence[dict[str, Any]], key: str
) -> dict[str, float] | None:
    values = [float(row[key]) for row in comparisons if key in row]
    if not values:
        return None
    stats = _sample_stats(values)
    return asdict(stats)


def build_aggregate(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_records": len(comparisons),
        "real_to_contiguous_bw_ratio": _aggregate_metric(
            comparisons, "real_to_contiguous_bw_ratio"
        ),
        "real_to_packed_bw_ratio": _aggregate_metric(
            comparisons, "real_to_packed_bw_ratio"
        ),
        "packed_to_contiguous_bw_ratio": _aggregate_metric(
            comparisons, "packed_to_contiguous_bw_ratio"
        ),
        "real_to_single_dma_bw_ratio": _aggregate_metric(
            comparisons, "real_to_single_dma_bw_ratio"
        ),
        "contiguous_to_single_dma_bw_ratio": _aggregate_metric(
            comparisons, "contiguous_to_single_dma_bw_ratio"
        ),
        "segment_count_bins": _build_segment_count_bins(comparisons),
    }
    paired = [
        (
            int(row["num_segments"]),
            float(row["real_to_contiguous_bw_ratio"]),
        )
        for row in comparisons
        if "real_to_contiguous_bw_ratio" in row
    ]
    segment_counts = [segment_count for segment_count, _ in paired]
    efficiencies = [efficiency for _, efficiency in paired]
    if (
        len(efficiencies) >= 2
        and len(set(segment_counts)) >= 2
        and len(set(efficiencies)) >= 2
    ):
        aggregate["correlation_segments_vs_real_efficiency"] = statistics.correlation(
            segment_counts, efficiencies
        )
    else:
        aggregate["correlation_segments_vs_real_efficiency"] = None
    return aggregate


def _segment_count_bin(segment_count: int) -> str:
    if segment_count == 1:
        return "1"
    for lower, upper in ((2, 4), (5, 8), (9, 16), (17, 32), (33, 64)):
        if lower <= segment_count <= upper:
            return f"{lower}-{upper}"
    return "65+"


def _build_segment_count_bins(
    comparisons: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    order = ("1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+")
    grouped: dict[str, list[float]] = {label: [] for label in order}
    for row in comparisons:
        ratio = row.get("real_to_contiguous_bw_ratio")
        if ratio is not None:
            grouped[_segment_count_bin(int(row["num_segments"]))].append(float(ratio))
    return [
        {
            "segment_count": label,
            "num_records": len(grouped[label]),
            "real_to_contiguous_bw_ratio": asdict(_sample_stats(grouped[label])),
        }
        for label in order
        if grouped[label]
    ]


def _flatten_method_result(result: MethodResult) -> dict[str, Any]:
    row = {
        key: value
        for key, value in asdict(result).items()
        if key not in ("host_enqueue_ms", "gpu_elapsed_ms")
    }
    for prefix, stats in (
        ("host_enqueue_ms", result.host_enqueue_ms),
        ("gpu_elapsed_ms", result.gpu_elapsed_ms),
    ):
        for key, value in asdict(stats).items():
            row[f"{prefix}_{key}"] = value
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_capture_description(
    records: Sequence[AllocationRecord],
    *,
    source_blocks: int,
    destination_blocks: int,
) -> None:
    first = records[0]
    source_gib = (
        first.outer_copies * source_blocks * first.bytes_per_outer_block / 2**30
    )
    destination_gib = (
        first.outer_copies * destination_blocks * first.bytes_per_outer_block / 2**30
    )
    print(
        "Layout: "
        f"shape={first.cpu_cache_shape}, dtype={first.cpu_cache_dtype}, "
        f"blocks_dim={first.blocks_dim}, outer_copies={first.outer_copies}, "
        f"bytes/block={first.bytes_per_block:,}"
    )
    print(
        f"Selected {len(records)} records; allocation: "
        f"pinned_source={source_gib:.3f} GiB, gpu_destination={destination_gib:.3f} GiB"
    )
    layout_states = {record.byte_layout_exact for record in records}
    if layout_states == {True}:
        layout_description = "verified contiguous (byte model is exact)"
    elif False in layout_states:
        layout_description = "non-contiguous (byte model is approximate)"
    else:
        layout_description = "not recorded (legacy capture; unverified)"
    print(f"Tensor layout: {layout_description}")
    print(
        f"{'record':>6} {'layer':>5} {'blocks':>8} {'segments':>8} "
        f"{'seg_p50':>8} {'seg_p95':>8} {'MiB':>10}"
    )
    for record in records:
        lengths = record.segment_lengths
        print(
            f"{record.capture_index:>6} {record.layer_idx:>5} "
            f"{record.load_blocks:>8} {len(record.segments):>8} "
            f"{statistics.median(lengths):>8.1f} "
            f"{_percentile(lengths, 0.95):>8.1f} "
            f"{record.total_bytes / 2**20:>10.1f}"
        )


def _print_results(comparisons: Sequence[dict[str, Any]]) -> None:
    print()
    print(
        f"{'record':>6} {'seg':>5} {'blocks':>7} "
        f"{'real GB/s':>10} {'packed':>10} {'contig':>10} {'1-DMA':>10} "
        f"{'real/contig':>12} {'real/1-DMA':>11} {'gap loss':>9}"
    )
    for row in comparisons:
        real = row.get("real_gbps")
        packed = row.get("packed_gbps")
        contiguous = row.get("contiguous_gbps")
        single_dma = row.get("single_dma_gbps")
        ratio = row.get("real_to_contiguous_bw_ratio")
        single_ratio = row.get("real_to_single_dma_bw_ratio")
        gap_penalty = row.get("address_gap_penalty_pct")
        print(
            f"{row['capture_index']:>6} {row['num_segments']:>5} "
            f"{row['load_blocks']:>7} "
            f"{real if real is not None else float('nan'):>10.2f} "
            f"{packed if packed is not None else float('nan'):>10.2f} "
            f"{contiguous if contiguous is not None else float('nan'):>10.2f} "
            f"{single_dma if single_dma is not None else float('nan'):>10.2f} "
            f"{ratio if ratio is not None else float('nan'):>12.3f} "
            f"{single_ratio if single_ratio is not None else float('nan'):>11.3f} "
            f"{gap_penalty if gap_penalty is not None else float('nan'):>8.1f}%"
        )


def _validate_compatible_layouts(records: Sequence[AllocationRecord]) -> None:
    first = records[0]
    expected = (
        first.blocks_dim,
        first.num_cpu_blocks,
        first.staging_capacity,
        first.outer_copies,
        first.bytes_per_outer_block,
        first.bytes_per_block,
    )
    for record in records[1:]:
        actual = (
            record.blocks_dim,
            record.num_cpu_blocks,
            record.staging_capacity,
            record.outer_copies,
            record.bytes_per_outer_block,
            record.bytes_per_block,
        )
        if actual != expected:
            raise ValueError(
                "Selected records have incompatible KV layouts: "
                f"record {first.capture_index}={expected}, "
                f"record {record.capture_index}={actual}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="A block_dist_*.json file, or a directory whose newest capture "
        "should be used.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--record-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit capture record indices; negative indices are accepted.",
    )
    parser.add_argument(
        "--last-records",
        type=int,
        default=8,
        help="Benchmark the last N records (0 means all); ignored with "
        "--record-indices. Last records are normally decode steps.",
    )
    parser.add_argument(
        "--record-stride",
        type=int,
        default=1,
        help="Keep every Nth selected record.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=_METHODS,
        default=["real", "packed", "contiguous"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--compact-source",
        action="store_true",
        help="Reduce pinned-memory use by preserving segment lengths with one "
        "block gap instead of exact logical source offsets.",
    )
    parser.add_argument(
        "--allow-layout-approximation",
        action="store_true",
        help="Allow captures whose CPU/GPU KV tensors are non-contiguous. "
        "The benchmark then validates fragmentation geometry only.",
    )
    parser.add_argument(
        "--max-source-gib",
        type=float,
        default=8.0,
        help="Abort before allocating more pinned source memory.",
    )
    parser.add_argument(
        "--max-gpu-gib",
        type=float,
        default=8.0,
        help="Abort before allocating more GPU destination memory.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Validate and describe the capture without allocating CUDA tensors.",
    )
    parser.add_argument(
        "--min-real-to-contiguous",
        type=float,
        default=None,
        help="Optional median bandwidth-efficiency threshold; exit 2 when missed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_path = resolve_capture_path(args.capture)
    records = load_capture(capture_path)
    selected = select_records(
        records,
        record_indices=args.record_indices,
        last_records=args.last_records,
        stride=args.record_stride,
    )
    if not selected:
        raise ValueError("Record selection is empty")
    empty_record_indices = [
        record.capture_index for record in selected if not record.load_blocks
    ]
    if empty_record_indices:
        print(
            "Skipping records with no H2D payload: "
            + ", ".join(str(index) for index in empty_record_indices)
        )
        selected = [record for record in selected if record.load_blocks]
    if not selected:
        raise ValueError("Selected records contain no H2D payload")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.max_source_gib <= 0 or args.max_gpu_gib <= 0:
        raise ValueError("allocation safety limits must be positive")
    if args.min_real_to_contiguous is not None and args.min_real_to_contiguous <= 0:
        raise ValueError("--min-real-to-contiguous must be positive")
    _validate_compatible_layouts(selected)
    noncontiguous_layout_indices = [
        record.capture_index for record in selected if record.byte_layout_exact is False
    ]
    unknown_layout_indices = [
        record.capture_index for record in selected if record.byte_layout_exact is None
    ]
    if noncontiguous_layout_indices and not args.allow_layout_approximation:
        raise ValueError(
            "Captured KV tensors are non-contiguous for records "
            + ", ".join(str(index) for index in noncontiguous_layout_indices)
            + "; the flattened byte model would not reproduce Tensor.copy_. "
            "Use --allow-layout-approximation only to test allocation geometry."
        )
    if noncontiguous_layout_indices:
        print(
            "WARNING: using a contiguous byte approximation for non-contiguous "
            "KV layouts in records "
            + ", ".join(str(index) for index in noncontiguous_layout_indices)
        )
    if unknown_layout_indices:
        print(
            "WARNING: capture lacks tensor stride metadata; byte-layout "
            "equivalence is unverified for records "
            + ", ".join(str(index) for index in unknown_layout_indices)
        )

    plans_by_record = {
        record.capture_index: build_method_plans(
            record,
            methods=args.methods,
            compact_source=args.compact_source,
        )
        for record in selected
    }
    plan_source_blocks = max(
        segment.src_start + segment.length
        for plans in plans_by_record.values()
        for segments in plans.values()
        for segment in segments
    )
    source_blocks = (
        plan_source_blocks
        if args.compact_source
        else max(record.num_cpu_blocks for record in selected)
    )
    destination_blocks = max(record.staging_capacity for record in selected)
    _print_capture_description(
        selected,
        source_blocks=source_blocks,
        destination_blocks=destination_blocks,
    )
    if args.describe_only:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --describe-only is used")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must be CUDA, got {device}")
    torch.cuda.set_device(device)

    first = selected[0]
    source_bytes = first.outer_copies * source_blocks * first.bytes_per_outer_block
    destination_bytes = (
        first.outer_copies * destination_blocks * first.bytes_per_outer_block
    )
    if source_bytes > args.max_source_gib * 2**30:
        raise RuntimeError(
            f"Pinned source requires {source_bytes / 2**30:.2f} GiB, above "
            f"--max-source-gib={args.max_source_gib}. Use --compact-source or "
            "increase the explicit limit."
        )
    if destination_bytes > args.max_gpu_gib * 2**30:
        raise RuntimeError(
            f"GPU destination requires {destination_bytes / 2**30:.2f} GiB, above "
            f"--max-gpu-gib={args.max_gpu_gib}."
        )

    try:
        source = torch.empty(
            (
                first.outer_copies,
                source_blocks,
                first.bytes_per_outer_block,
            ),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to allocate pinned source memory. Check the process memlock "
            "limit or retry with --compact-source."
        ) from error
    destination = torch.empty(
        (
            first.outer_copies,
            destination_blocks,
            first.bytes_per_outer_block,
        ),
        dtype=torch.uint8,
        device=device,
    )
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)

    results: list[MethodResult] = []
    for record in selected:
        results.extend(
            benchmark_record(
                record,
                plans=plans_by_record[record.capture_index],
                destination=destination,
                source=source,
                stream=stream,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
    torch.cuda.synchronize(device)

    comparisons = build_comparisons(results)
    aggregate = build_aggregate(comparisons)
    if noncontiguous_layout_indices:
        layout_assumption = (
            "Contiguous byte approximation; allocation geometry and transfer "
            "sizes are preserved, but captured tensor strides are not."
        )
    elif unknown_layout_indices:
        layout_assumption = (
            "Contiguous byte model; legacy capture lacks strides, so exact "
            "equivalence to Tensor.copy_ is unverified."
        )
    else:
        layout_assumption = (
            "Captured CPU/GPU tensors are contiguous; the byte model preserves "
            "the production segment copy sizes and addresses."
        )
    _print_results(comparisons)
    print()
    ratio_stats = aggregate.get("real_to_contiguous_bw_ratio")
    if ratio_stats is not None:
        print(
            "Median real/contiguous bandwidth efficiency: "
            f"{ratio_stats['median']:.3f} "
            f"(median penalty {(1.0 - ratio_stats['median']) * 100.0:.1f}%)"
        )
        print("Efficiency by allocation segment count:")
        for bucket in aggregate["segment_count_bins"]:
            bucket_stats = bucket["real_to_contiguous_bw_ratio"]
            print(
                f"  {bucket['segment_count']:>5} segments: "
                f"n={bucket['num_records']}, "
                f"median={bucket_stats['median']:.3f}, "
                f"p05={bucket_stats['p05']:.3f}"
            )
    correlation = aggregate["correlation_segments_vs_real_efficiency"]
    if correlation is not None:
        print(f"Correlation(segments, efficiency): {correlation:.3f}")
    single_dma_ratio_stats = aggregate.get("real_to_single_dma_bw_ratio")
    if single_dma_ratio_stats is not None:
        print(
            "Median real/single-DMA bandwidth efficiency: "
            f"{single_dma_ratio_stats['median']:.3f} "
            f"(median penalty "
            f"{(1.0 - single_dma_ratio_stats['median']) * 100.0:.1f}%)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_results = [_flatten_method_result(result) for result in results]
    _write_csv(args.output_dir / "method_results.csv", flat_results)
    _write_csv(args.output_dir / "record_comparisons.csv", comparisons)
    summary = {
        "capture": str(capture_path.resolve()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "methods": args.methods,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "compact_source": args.compact_source,
        "allow_layout_approximation": args.allow_layout_approximation,
        "selected_capture_indices": [record.capture_index for record in selected],
        "skipped_empty_capture_indices": empty_record_indices,
        "noncontiguous_layout_capture_indices": noncontiguous_layout_indices,
        "unknown_layout_capture_indices": unknown_layout_indices,
        "layout_assumption": layout_assumption,
        "source_gib": source_bytes / 2**30,
        "destination_gib": destination_bytes / 2**30,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {args.output_dir}")

    if args.min_real_to_contiguous is not None:
        if ratio_stats is None:
            raise ValueError(
                "--min-real-to-contiguous requires real and contiguous methods"
            )
        if ratio_stats["median"] < args.min_real_to_contiguous:
            print(
                f"FAIL: median efficiency {ratio_stats['median']:.3f} < "
                f"{args.min_real_to_contiguous:.3f}"
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
