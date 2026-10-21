# Speculative-plan claim metrics

Search roots: `/home/lyc/inference/vllm-RunKV/exp_results`, `/home/lyc/inference/vllm/exp_results`.

Adjacent change uses `2 * |late - early| / (|late| + |early|)` on decode-only controller gain snapshots.

Convergence is the first observation with `|imbalance| < 0.5 ms`, among non-left-censored decode-only TRACKING episodes that reach the deadband. One observation is one model layer, not one inference-loop decode step.

`strict_pair` means ON and OFF settings match in all manifest fields used by `plot_spec_on_off_slope_curves.py`; `latest_unpaired` has no matched opposite-Spec trace.

| Model | Environment | Spec | Workloads | Pairing | Adjacent n | Adjacent <=10% | Plan-needed <=10% | TRACKING episodes | Mean TRACKING obs. | Deadband reach | Mean obs. to first deadband |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | io_contention | ON | 1k+2k+4k+8k | latest_unpaired | 7235 | 73.9% | 57.6% | 166 | 19.92 | 31.3% | 8.46 |
| Llama-2-7B | stable | ON | 1k+2k+4k+8k | latest_unpaired | 3804 | 89.6% | 66.9% | 141 | 8.74 | 96.5% | 4.64 |
| OPT-1.3B | io_contention | ON | 2k | latest_unpaired | 1323 | 67.7% | 65.3% | 40 | 16.07 | 75.0% | 8.53 |
| OPT-2.7B | io_contention | OFF | 1k+2k+4k+8k | strict_pair | 6695 | 71.1% | 57.4% | 233 | 17.62 | 87.1% | 5.33 |
| OPT-2.7B | io_contention | ON | 1k+2k+4k+8k | strict_pair | 6590 | 70.8% | 58.7% | 238 | 17.85 | 85.7% | 5.51 |
| OPT-2.7B | stable | OFF | 1k+2k+4k+8k | strict_pair | 4737 | 96.5% | 57.6% | 132 | 4.75 | 99.2% | 2.98 |
| OPT-2.7B | stable | ON | 1k+2k+4k+8k | strict_pair | 4585 | 95.6% | 57.0% | 127 | 5.25 | 99.2% | 3.47 |
| OPT-6.7B | io_contention | ON | 1k+2k+4k+8k | latest_unpaired | 7556 | 70.5% | 57.6% | 268 | 17.38 | 85.8% | 6.31 |
