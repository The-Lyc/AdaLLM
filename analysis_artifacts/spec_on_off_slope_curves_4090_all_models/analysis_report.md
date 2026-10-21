# Actual Spec ON/OFF slope curves

Search roots: `/home/lyc/inference/vllm-RunKV/exp_results/ablation/spec_claim`, `/home/lyc/inference/vllm-RunKV/exp_results/ablation/no_throttle`, `/home/lyc/inference/vllm-RunKV/exp_results/ablation/io_contention`.

A pair must match every controller, workload, memory, copy-mode, and resource-pressure field encoded in `matched_run_pairs.csv`; only `layer_recompute_async_plan_build` may differ.

Only decode-only records with equal `(step, layer, num_reqs, total_scheduled_tokens)` are aligned. The plotted step is the temporal q=0.5 common step, not the step with the smallest ON/OFF difference.

These are actual independent ON/OFF runs. They do not represent the early and late adjacent gain snapshots within one trajectory.

## Log coverage

| Model | Environment | ON runs | OFF runs | Strict pairs | Selected |
|---|---|---:|---:|---:|---:|
| Llama-2-7B | io_contention | 4 | 4 | 4 | 4 |
| Llama-2-7B | stable | 4 | 4 | 4 | 4 |
| OPT-2.7B | io_contention | 23 | 6 | 5 | 4 |
| OPT-2.7B | stable | 25 | 7 | 7 | 4 |
| OPT-6.7B | io_contention | 4 | 4 | 4 | 4 |
| OPT-6.7B | stable | 4 | 4 | 4 | 4 |

## Direct ON/OFF similarity

Symmetric change is `2 * |g_on - g_off| / (|g_on| + |g_off|)`.

| Model | Environment | Workload | Step | Layers | Median change | Within 10% | All-step log-r |
|---|---|---|---:|---:|---:|---:|---:|
| Llama-2-7B | stable | 1k | 45 | 31 | 97.6% | 0.0% | -0.329 |
| Llama-2-7B | stable | 2k | 46 | 24 | 112.5% | 4.2% | -0.117 |
| Llama-2-7B | stable | 4k | 48 | 31 | 23.7% | 0.0% | 0.263 |
| Llama-2-7B | stable | 8k | 46 | 31 | 4.3% | 87.1% | -0.172 |
| Llama-2-7B | io_contention | 1k | 67 | 31 | 130.3% | 0.0% | 0.286 |
| Llama-2-7B | io_contention | 2k | 34 | 31 | 95.3% | 0.0% | 0.469 |
| Llama-2-7B | io_contention | 4k | 19 | 31 | 173.4% | 0.0% | 0.069 |
| Llama-2-7B | io_contention | 8k | 19 | 31 | 170.7% | 0.0% | -0.158 |
| OPT-2.7B | stable | 1k | 44 | 31 | 140.6% | 0.0% | 0.272 |
| OPT-2.7B | stable | 2k | 56 | 31 | 146.1% | 0.0% | 0.038 |
| OPT-2.7B | stable | 4k | 77 | 25 | 52.7% | 0.0% | -0.243 |
| OPT-2.7B | stable | 8k | 76 | 31 | 68.7% | 0.0% | -0.001 |
| OPT-2.7B | io_contention | 1k | 76 | 31 | 93.9% | 3.2% | 0.060 |
| OPT-2.7B | io_contention | 2k | 54 | 31 | 180.3% | 0.0% | 0.291 |
| OPT-2.7B | io_contention | 4k | 76 | 31 | 132.1% | 0.0% | 0.277 |
| OPT-2.7B | io_contention | 8k | 77 | 31 | 25.6% | 29.0% | 0.464 |
| OPT-6.7B | stable | 1k | 74 | 31 | 173.5% | 0.0% | 0.058 |
| OPT-6.7B | stable | 2k | 53 | 31 | 162.7% | 0.0% | 0.240 |
| OPT-6.7B | stable | 4k | 77 | 25 | 14.1% | 12.0% | 0.409 |
| OPT-6.7B | stable | 8k | 82 | 31 | 69.8% | 0.0% | 0.344 |
| OPT-6.7B | io_contention | 1k | 73 | 31 | 62.1% | 0.0% | 0.038 |
| OPT-6.7B | io_contention | 2k | 51 | 31 | 79.2% | 9.7% | 0.424 |
| OPT-6.7B | io_contention | 4k | 79 | 25 | 145.0% | 4.0% | 0.350 |
| OPT-6.7B | io_contention | 8k | 81 | 31 | 129.6% | 25.8% | 0.289 |

## Plots

- `spec_on_off_slope_curves_llama-2-7b.png`
- `spec_on_off_slope_curves_llama-2-7b.pdf`
- `spec_on_off_slope_curves_opt-2-7b.png`
- `spec_on_off_slope_curves_opt-2-7b.pdf`
- `spec_on_off_slope_curves_opt-6-7b.png`
- `spec_on_off_slope_curves_opt-6-7b.pdf`
