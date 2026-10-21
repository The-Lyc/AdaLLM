# Speculative-plan claim metrics

Search roots: `/home/lyc/inference/vllm-RunKV/exp_results/ablation/spec_claim`, `/home/lyc/inference/vllm-RunKV/exp_results/ablation/no_throttle`, `/home/lyc/inference/vllm-RunKV/exp_results/ablation/io_contention`.

Adjacent change uses `2 * |late - early| / (|late| + |early|)` on decode-only controller gain snapshots.

Convergence is the first observation with `|imbalance| < 0.5 ms`, among non-left-censored decode-only TRACKING episodes that reach the deadband. One observation is one model layer, not one inference-loop decode step.

`strict_pair` means ON and OFF settings match in all manifest fields used by `plot_spec_on_off_slope_curves.py`; `latest_unpaired` has no matched opposite-Spec trace.

| Model | Environment | Spec | Workloads | Pairing | Adjacent n | Adjacent <=10% | Plan-needed <=10% | TRACKING episodes | Mean TRACKING obs. | Deadband reach | Mean obs. to first deadband |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | io_contention | OFF | 1k+2k+4k+8k | strict_pair | 7240 | 86.2% | 76.2% | 162 | 20.06 | 42.6% | 10.16 |
| Llama-2-7B | io_contention | ON | 1k+2k+4k+8k | strict_pair | 7252 | 75.4% | 59.9% | 160 | 19.69 | 50.0% | 8.28 |
| Llama-2-7B | stable | OFF | 1k+2k+4k+8k | strict_pair | 3893 | 94.1% | 67.0% | 144 | 5.77 | 100.0% | 2.85 |
| Llama-2-7B | stable | ON | 1k+2k+4k+8k | strict_pair | 3841 | 86.7% | 59.6% | 140 | 9.16 | 99.3% | 4.57 |
| OPT-2.7B | io_contention | OFF | 1k+2k+4k+8k | strict_pair | 6695 | 71.1% | 57.4% | 233 | 17.62 | 87.1% | 5.33 |
| OPT-2.7B | io_contention | ON | 1k+2k+4k+8k | strict_pair | 6590 | 70.8% | 58.7% | 238 | 17.85 | 85.7% | 5.51 |
| OPT-2.7B | stable | OFF | 1k+2k+4k+8k | strict_pair | 4737 | 96.5% | 57.6% | 132 | 4.75 | 99.2% | 2.98 |
| OPT-2.7B | stable | ON | 1k+2k+4k+8k | strict_pair | 4585 | 95.6% | 57.0% | 127 | 5.25 | 99.2% | 3.47 |
| OPT-6.7B | io_contention | OFF | 1k+2k+4k+8k | strict_pair | 6851 | 74.6% | 60.9% | 250 | 15.40 | 93.6% | 4.63 |
| OPT-6.7B | io_contention | ON | 1k+2k+4k+8k | strict_pair | 7565 | 70.6% | 58.0% | 266 | 17.55 | 85.0% | 6.42 |
| OPT-6.7B | stable | OFF | 1k+2k+4k+8k | strict_pair | 6785 | 96.4% | 79.1% | 242 | 4.21 | 100.0% | 2.18 |
| OPT-6.7B | stable | ON | 1k+2k+4k+8k | strict_pair | 6619 | 93.6% | 74.7% | 243 | 6.01 | 98.8% | 2.91 |
