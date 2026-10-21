# RTX 4090 RunKV segment-DMA measurement

Measurement date: 2026-08-30. Device: NVIDIA GeForce RTX 4090 (PCIe 4.0 x16,
24 GiB). PyTorch/CUDA: 2.11.0+cu128 / CUDA 12.8.

## Definitions and window

- A `segment` is one maximal consecutive logical-block range in the actual
  H2D load list. Production submits two copies per segment (K and V).
- `mapped blocks` includes both replay hits and H2D loads. `load blocks` is the
  subset that actually crosses PCIe. Both are reported to avoid ambiguity in
  “total blocks”.
- The historical table includes all 32 model layers. The exact-capture and
  microbenchmark tables select records 16--78: 63 decode
  DMA steps from the `prompt=1000, batch=32, decode=64` run, including drain.
  The runtime capture intentionally samples the middle layer (L16) once per
  step. Across the historical scenarios, its segment mean differs from the
  all-layer mean by at most 0.26 segment.
- Block totals are cumulative layer-DMA traffic, not counts of unique block
  IDs. Segment mean is the arithmetic mean per observed layer-DMA call.

## Historical 4090 trace matrix

These values are extracted from existing full-model NVTX/SQLite traces. Each
model aggregates all layer-DMA calls from its four tested prompt/decode
configurations.

| Model | Layer-DMA calls | Mean segments/DMA | Median | P95 | Max | Mean H2D load blocks/DMA | Cumulative H2D load blocks | Weighted blocks/segment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OPT-2.7B | 6,016 | 25.984 | 25 | 45.0 | 50 | 1,859.3 | 11,185,713 | 71.56 |
| OPT-6.7B | 4,992 | 19.003 | 18 | 35.0 | 44 | 1,672.6 | 8,349,430 | 88.02 |
| Llama-2-7B | 4,992 | 15.279 | 14 | 28.0 | 39 | 1,691.8 | 8,445,421 | 110.73 |

The per-configuration values are in
[`historical_trace_matrix.csv`](historical_trace_matrix.csv).

## Exact distributions used by the microbenchmark

| Model | Bytes/block | Mapped blocks mean / sum | H2D load blocks mean / sum | Replay-hit blocks mean | Segments mean / median / P95 / max | Weighted load blocks/segment | Pooled segment length median / P95 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPT-2.7B | 163,840 | 1,856.9 / 116,986 | 1,417.9 / 89,328 | 439.0 | 20.71 / 21 / 31.9 / 34 | 68.45 | 5 / 320 / 321 |
| OPT-6.7B | 262,144 | 1,856.9 / 116,986 | 1,534.1 / 96,648 | 322.8 | 16.97 / 16 / 25.9 / 28 | 90.41 | 7 / 449 / 513 |
| Llama-2-7B | 262,144 | 1,862.2 / 117,316 | 1,552.6 / 97,816 | 309.5 | 17.97 / 19 / 26.8 / 29 | 86.41 | 6 / 326 / 641 |

The exact logical IDs, skipped/load IDs, tensor shapes/strides, and segment
counts are preserved in each model's `capture/block_dist_*.json` file.

## Paired H2D microbenchmark result

Each of the 63 records is measured with 5 warmups and 30 timed iterations on a
dedicated CUDA stream. The method order rotates per iteration. GPU time comes
from CUDA events; useful payload bytes, not address span, are used for GB/s.

`real` replays the production source/destination gaps and submits `2 x
segments` K/V copies. `single_dma` flattens the same useful K+V bytes and makes
exactly one copy; it is an ideal ceiling and does not include the packing that
a production implementation would require.

| Model | Mean segments | Median real copy submissions | Real GB/s | One-DMA GB/s | Median bandwidth retained | P05 retained | Paired GPU overhead | Host enqueue real / one-DMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OPT-2.7B | 20.71 | 42 | 26.60 | 26.81 | 99.21% | 98.61% | +65.4 us | 174.9 / 2.04 us |
| OPT-6.7B | 16.97 | 32 | 26.72 | 26.82 | 99.63% | 99.08% | +52.5 us | 134.9 / 2.13 us |
| Llama-2-7B | 17.97 | 38 | 26.73 | 26.82 | 99.64% | 99.23% | +58.7 us | 159.8 / 2.08 us |

One-DMA reaches about 85.1% of the 31.51 GB/s PCIe 4.0 x16 encoded-rate
ceiling. Segment DMA therefore loses only 0.36--0.79% median useful bandwidth,
while host enqueue time rises by roughly 63--86x. The main observable cost is
CPU submission/overlap pressure, not PCIe payload efficiency.

The OPT-2.7B minimum retention (96.02%) comes from the drain tail with a small
payload. Restricting the analysis to the 48 full-batch/plateau records gives a
99.34% median and 99.00% minimum. Payload size matters more than segment count
alone for this tail.

### Cost decomposition

The three-way run separates address gaps from multiple copy submissions:

| Model | Gap penalty (`real` vs packed) | Split-copy penalty (packed vs contiguous) | Total (`real` vs contiguous) |
|---|---:|---:|---:|
| OPT-2.7B | 0.09% | 0.69% | 0.76% |
| OPT-6.7B | 0.04% | 0.32% | 0.36% |
| Llama-2-7B | 0.03% | 0.31% | 0.34% |

Thus the small GPU-side loss mostly comes from splitting the transfer; the
captured source/destination gaps themselves are effectively negligible.

Machine-readable consolidated values are in
[`capture_microbenchmark_summary.csv`](capture_microbenchmark_summary.csv).
Each model directory also contains raw per-record CSVs and complete JSON
summaries for both the decomposition run and the literal one-DMA comparison.
