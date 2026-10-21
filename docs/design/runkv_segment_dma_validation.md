# RunKV Segment-DMA Fragmentation Validation

## Question

RunKV coalesces consecutive logical KV blocks and submits one asynchronous H2D
copy for every resulting segment and every outer K/V slice. This avoids copying
blocks that replay will regenerate, but a fragmented allocation can turn one
large transfer into many small transfers. The experiment answers two separate
questions:

1. Do gaps in the real source and staging addresses reduce H2D bandwidth?
2. How much do small copy sizes and repeated `cudaMemcpyAsync` submissions cost?

The benchmark uses allocation records captured from normal RunKV inference; it
does not synthesize a block-ID distribution.

## Capture a real allocation trace

`PagedBlockMapper` already has an opt-in capture path. It records one middle
layer per inference step, including the full logical allocation, replay-skipped
blocks, blocks actually loaded, KV-cache shape, stride, contiguity, and dtype.
For example:

```bash
CAPTURE_DIR=/tmp/runkv-block-dist
RUNKV_CAPTURE_BLOCK_DIST=1 \
RUNKV_CAPTURE_BLOCK_DIST_DIR="$CAPTURE_DIR" \
RUNKV_CAPTURE_BLOCK_DIST_MAX=500 \
DRY_RUN=0 \
H2D_COPY_MODE=segment \
ENABLE_NSYS=0 \
ENABLE_PROFILE=0 \
python examples/offline_inference/run_opt_feedback_observation.py
```

Use the same model, prompt distribution, concurrency, replay controller, and
resource-pressure settings as the end-to-end experiment. The JSON file is
flushed to `$CAPTURE_DIR/block_dist_<pid>.json` when the worker exits normally.
Later records generally represent decode; use explicit record indices if the
workload has multiple phases that should be analyzed separately.

## Run the paired microbenchmark

```bash
python benchmarks/kernels/benchmark_runkv_segment_dma.py \
  --capture /tmp/runkv-block-dist \
  --last-records 64 \
  --warmup 5 \
  --iterations 50 \
  --output-dir /tmp/runkv-segment-dma
```

Passing a directory selects and prints its newest `block_dist_*.json`. An exact
JSON path is also accepted and is preferable when archiving a published run.

The default allocates the full captured CPU-cache block dimension and GPU
staging capacity, preserving exact logical source offsets, K/V outer-slice
spacing, and staging addresses. This can require a large pinned allocation.
The benchmark aborts above 8 GiB by default instead of silently changing the
experiment. If the machine cannot allocate the exact source span,
`--compact-source` preserves the real segment lengths and destination holes
with a one-block source gap, but that run must be labeled as a compact-source
approximation. GPU staging capacity remains exact in either mode.

Add `--methods real packed contiguous single_dma per_block` to include both
the literal one-DMA ideal baseline and the one-copy-per-block worst case.
`--describe-only` validates the capture and prints its memory requirements
without allocating CUDA tensors.

## Paired controls

Every method copies exactly the same payload for a given allocation record:

| Method | Segment lengths/count | Source and destination gaps | Purpose |
| --- | --- | --- | --- |
| `real` | Real | Real | Production segment-DMA geometry |
| `packed` | Real | Removed | Isolate the cost of address gaps |
| `contiguous` | One range (two K/V copies) | Removed | Production-layout contiguous baseline |
| `single_dma` | One flattened K+V range | Removed | Literal one-`cudaMemcpyAsync` ideal baseline |
| `per_block` | One range per block | Real | Optional worst-case fragmentation |

For a cache shaped `[2, blocks, ...]`, each logical range creates two memcpy
submissions, exactly as in the production K/V outer loop. The tensors are
represented as byte views with the captured per-block byte size; this keeps
copy sizes and offsets exact without depending on the model's element dtype in
the measurement loop. New captures also include CPU/GPU tensor strides. The
benchmark verifies that both layouts are contiguous before treating the byte
model as exact. It rejects a non-contiguous layout unless
`--allow-layout-approximation` is supplied; that override preserves allocation
geometry and byte counts but does not reproduce a strided `Tensor.copy_`.

The key paired ratios are:

- `real / contiguous`: total efficiency retained under the real fragmentation.
- `real / packed`: effect of the real source/destination gaps at fixed segment
  sizes and submission count.
- `packed / contiguous`: effect of splitting the payload into small transfers
  and issuing many memcpy calls, without address gaps.
- `real / single_dma`: total efficiency retained relative to a literal single
  flattened DMA of the complete K+V payload. This strict ideal does not preserve
  the production K/V outer layout; `contiguous` does.

This decomposition is more informative than comparing raw GB/s across steps,
because total payload size varies with the live batch. A negative reported
"penalty" is possible from measurement noise and means the fragmented case was
slightly faster in that sample; it should not be truncated to zero.

## Measurements and outputs

GPU elapsed time is measured with CUDA events around the copy sequence on a
dedicated stream. Host enqueue time is measured separately around the
`Tensor.copy_` calls. Effective bandwidth uses useful payload bytes, so holes
are not counted as transferred data.

The output directory contains:

- `method_results.csv`: raw per-record method measurements, segment-size
  distribution, submission count, host time, GPU time, and effective GB/s.
- `record_comparisons.csv`: paired ratios and penalties for each real allocation.
- `summary.json`: aggregate distributions, segment-count buckets, and the
  correlation between segment count and real/contiguous efficiency.

The segment-count buckets (`1`, `2-4`, ..., `65+`) make the failure regime
visible directly. Report both the median ratio and the p05 lower tail; an aggregate
median alone can hide a small number of severely fragmented steps. For a gate,
the optional `--min-real-to-contiguous X` checks the median, but `X` should be
chosen from the application's tolerated layer-prefetch slack rather than treated
as a universal hardware constant.

## Recommended experiment matrix

Run at least the following captures on the target PCIe/NUMA placement:

1. Prefill and early decode, where allocations are comparatively compact.
2. Steady-state decode at the intended concurrency.
3. Churn-heavy admission/retirement, expected to create the most fragments.
4. Low and high replay budgets, because skipped staging slots can add
   destination holes even when source allocations are compact.

Use several process-level repetitions. Keep the GPU idle for the isolated
bandwidth result, then confirm any bad regime in an end-to-end run with normal
compute overlap and resource contention. This microbenchmark identifies the IO
mechanism; it does not by itself measure whether prefetch latency is hidden by
layer compute.

## Relation to gather and UVA modes

`gather` and `uva` are useful production alternatives, but they are not clean
controls for this specific bandwidth question. Gather adds CPU `index_select`,
an extra pinned staging buffer, and sometimes a GPU scatter; UVA replaces DMA
submissions with a GPU kernel that reads pinned host pages. Mixing those costs
into `real/contiguous` would make a lower ratio ambiguous.

First use this paired benchmark to establish whether segment size/count causes
the loss. Then run the same captured workload end to end with
`H2D_COPY_MODE=segment`, `gather`, and (when `runkv_kernels` is built) `uva` to
decide which implementation gives the best layer-ready latency under normal
compute overlap. That second comparison should report end-to-end prefetch
latency, not call its result raw DMA bandwidth.

## Interpretation examples

- `real/packed` near 1, but `packed/contiguous` low: address gaps are harmless;
  repeated small transfers are the problem.
- `real/packed` low: source/destination spacing adds a measurable penalty beyond
  the number and size of transfers.
- Both ratios near 1: segment DMA preserves link efficiency for the observed
  allocation distribution; its main remaining cost is host submission time.
- GPU efficiency is acceptable but host enqueue time grows with segment count:
  bandwidth is not yet the bottleneck, but CPU launch overhead may still affect
  speculative plan scheduling or overlap.
