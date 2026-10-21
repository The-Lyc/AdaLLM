# Speculative Plan Slope Stability

Rows compare the gain available for early speculative construction with the adjacent gain available to synchronous construction.

Step filter: decode-only.

Stable-successor gate deadband: 0.5 ms.

`full_plan_needed` applies the same hint + deadband gate as the pre-hook and is the scope relevant to speculative-plan consumption. `all_gain_snapshots` includes every non-null gain snapshot.

The state-machine controller carries this smoothed estimate across layers and refreshes it during TRACKING; this measures estimator state stability, not an independent per-layer plant derivative.

Symmetric relative change is `2 * |late - early| / (|late| + |early|)`.

| Scenario | Scope | Workload | Variant | Pairs | log-r | Exact | Within 10% | P90 symmetric change |
|---|---|---|---|---:|---:|---:|---:|---:|
| no_throttle | full_plan_needed | 1k | full | 53 | 0.793 | 64.2% | 69.8% | 64.7% |
| no_throttle | full_plan_needed | 2k | full | 99 | 0.574 | 47.5% | 50.5% | 141.4% |
| no_throttle | full_plan_needed | 4k | full | 128 | 0.619 | 43.8% | 50.8% | 100.3% |
| no_throttle | full_plan_needed | 8k | full | 122 | 0.630 | 51.6% | 63.1% | 129.5% |
| no_throttle | full_plan_needed | all | full | 402 | 0.648 | 49.8% | 57.0% | 117.6% |
| no_throttle | all_gain_snapshots | 1k | full | 1642 | 0.992 | 98.7% | 98.8% | 0.0% |
| no_throttle | all_gain_snapshots | 2k | full | 1499 | 0.968 | 95.1% | 95.3% | 0.0% |
| no_throttle | all_gain_snapshots | 4k | full | 703 | 0.903 | 89.8% | 91.0% | 0.5% |
| no_throttle | all_gain_snapshots | 8k | full | 741 | 0.930 | 91.4% | 93.3% | 0.0% |
| no_throttle | all_gain_snapshots | all | full | 4585 | 0.958 | 94.9% | 95.6% | 0.0% |
| no_throttle | full_plan_needed | 1k | no_async_plan | 39 | 0.955 | 64.1% | 66.7% | 53.2% |
| no_throttle | full_plan_needed | 2k | no_async_plan | 81 | 0.829 | 50.6% | 55.6% | 115.2% |
| no_throttle | full_plan_needed | 4k | no_async_plan | 54 | 0.401 | 14.8% | 55.6% | 79.7% |
| no_throttle | full_plan_needed | 8k | no_async_plan | 83 | 0.522 | 32.5% | 56.6% | 64.4% |
| no_throttle | full_plan_needed | all | no_async_plan | 257 | 0.753 | 39.3% | 57.6% | 87.6% |
| no_throttle | all_gain_snapshots | 1k | no_async_plan | 1668 | 0.990 | 98.1% | 98.4% | 0.0% |
| no_throttle | all_gain_snapshots | 2k | no_async_plan | 1521 | 0.971 | 95.4% | 95.7% | 0.0% |
| no_throttle | all_gain_snapshots | 4k | no_async_plan | 698 | 0.969 | 92.3% | 95.4% | 0.0% |
| no_throttle | all_gain_snapshots | 8k | no_async_plan | 850 | 0.955 | 92.4% | 94.8% | 0.0% |
| no_throttle | all_gain_snapshots | all | no_async_plan | 4737 | 0.975 | 95.4% | 96.5% | 0.0% |
| io_contention | full_plan_needed | 1k | full | 2379 | 0.850 | 54.1% | 58.7% | 128.3% |
| io_contention | full_plan_needed | 2k | full | 1087 | 0.906 | 53.1% | 59.1% | 112.8% |
| io_contention | full_plan_needed | 4k | full | 503 | 0.606 | 47.7% | 55.1% | 132.5% |
| io_contention | full_plan_needed | 8k | full | 525 | 0.923 | 51.6% | 61.5% | 99.3% |
| io_contention | full_plan_needed | all | full | 4494 | 0.873 | 52.8% | 58.7% | 121.9% |
| io_contention | all_gain_snapshots | 1k | full | 3535 | 0.885 | 68.0% | 71.2% | 98.3% |
| io_contention | all_gain_snapshots | 2k | full | 1569 | 0.921 | 66.7% | 70.8% | 89.8% |
| io_contention | all_gain_snapshots | 4k | full | 750 | 0.682 | 63.9% | 68.8% | 110.3% |
| io_contention | all_gain_snapshots | 8k | full | 736 | 0.926 | 64.1% | 71.2% | 84.5% |
| io_contention | all_gain_snapshots | all | full | 6590 | 0.897 | 66.8% | 70.8% | 96.8% |
| io_contention | full_plan_needed | 1k | no_async_plan | 2274 | 0.883 | 47.4% | 56.8% | 103.5% |
| io_contention | full_plan_needed | 2k | no_async_plan | 1030 | 0.900 | 45.7% | 56.2% | 99.0% |
| io_contention | full_plan_needed | 4k | no_async_plan | 494 | 0.830 | 48.6% | 57.5% | 89.9% |
| io_contention | full_plan_needed | 8k | no_async_plan | 620 | 0.892 | 50.8% | 61.5% | 71.0% |
| io_contention | full_plan_needed | all | no_async_plan | 4418 | 0.893 | 47.6% | 57.4% | 95.5% |
| io_contention | all_gain_snapshots | 1k | no_async_plan | 3511 | 0.930 | 65.0% | 71.1% | 72.6% |
| io_contention | all_gain_snapshots | 2k | no_async_plan | 1601 | 0.930 | 64.6% | 71.5% | 69.8% |
| io_contention | all_gain_snapshots | 4k | no_async_plan | 745 | 0.880 | 65.5% | 71.4% | 66.8% |
| io_contention | all_gain_snapshots | 8k | no_async_plan | 838 | 0.913 | 62.4% | 70.3% | 65.7% |
| io_contention | all_gain_snapshots | all | no_async_plan | 6695 | 0.928 | 64.7% | 71.1% | 69.6% |
