# Speculative Plan TRACKING Performance

Unit: one controller observation per model layer. The final observation that exits TRACKING is included in episode-to-exit length.

Phase scope: `decode_only`.

Deadband: `|imbalance| < 0.5 ms`; settling requires 3 consecutive deadband observations.

First-deadband latency is computed only for non-left-censored episodes that reached the deadband. Reach and settle rates are reported alongside it to expose failures and censoring.

| Scenario | Workload | Spec | Controller obs. | TRACKING obs. | TRACKING share | Episodes | Mean TRACKING obs./episode | Mean obs. to first deadband | P90 | Deadband reach | Mean obs. to exit | Settled |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_throttle | 1k | ON | 1953 | 104 | 5.3% | 24 | 4.33 | 2.50 | 6.70 | 100.0% | 5.33 | 100.0% |
| no_throttle | 2k | ON | 1953 | 264 | 13.5% | 61 | 4.33 | 2.52 | 6.00 | 100.0% | 5.33 | 100.0% |
| no_throttle | 4k | ON | 961 | 169 | 17.6% | 23 | 7.35 | 6.04 | 11.80 | 100.0% | 8.35 | 100.0% |
| no_throttle | 8k | ON | 961 | 130 | 13.5% | 19 | 6.84 | 4.67 | 8.30 | 94.7% | 7.84 | 94.7% |
| no_throttle | all | ON | 5828 | 667 | 11.4% | 127 | 5.25 | 3.47 | 7.00 | 99.2% | 6.25 | 99.2% |
| no_throttle | 1k | OFF | 1953 | 118 | 6.0% | 31 | 3.81 | 1.97 | 5.00 | 100.0% | 4.81 | 100.0% |
| no_throttle | 2k | OFF | 1953 | 249 | 12.7% | 59 | 4.22 | 2.56 | 5.00 | 100.0% | 5.22 | 100.0% |
| no_throttle | 4k | OFF | 961 | 129 | 13.4% | 21 | 6.14 | 4.76 | 9.00 | 100.0% | 7.14 | 100.0% |
| no_throttle | 8k | OFF | 961 | 131 | 13.6% | 21 | 6.24 | 3.90 | 9.00 | 95.2% | 7.24 | 95.2% |
| no_throttle | all | OFF | 5828 | 627 | 10.8% | 132 | 4.75 | 2.98 | 7.00 | 99.2% | 5.75 | 99.2% |
| io_contention | 1k | ON | 3937 | 2114 | 53.7% | 125 | 16.91 | 4.86 | 13.00 | 84.8% | 17.91 | 31.2% |
| io_contention | 2k | ON | 1953 | 1139 | 58.3% | 62 | 18.37 | 5.79 | 12.00 | 85.5% | 19.37 | 30.6% |
| io_contention | 4k | ON | 961 | 506 | 52.7% | 25 | 19.64 | 7.17 | 12.00 | 92.0% | 20.64 | 16.0% |
| io_contention | 8k | ON | 961 | 505 | 52.5% | 26 | 19.42 | 6.23 | 16.00 | 84.6% | 20.42 | 19.2% |
| io_contention | all | ON | 7812 | 4264 | 54.6% | 238 | 17.85 | 5.51 | 13.00 | 85.7% | 18.85 | 28.2% |
| io_contention | 1k | OFF | 3937 | 2113 | 53.7% | 121 | 17.46 | 4.71 | 13.00 | 86.8% | 18.46 | 27.3% |
| io_contention | 2k | OFF | 1953 | 1053 | 53.9% | 63 | 16.71 | 5.28 | 11.00 | 85.7% | 17.71 | 38.1% |
| io_contention | 4k | OFF | 961 | 430 | 44.7% | 23 | 18.70 | 7.00 | 11.00 | 95.7% | 19.70 | 26.1% |
| io_contention | 8k | OFF | 961 | 510 | 53.1% | 26 | 19.62 | 6.77 | 18.70 | 84.6% | 20.62 | 23.1% |
| io_contention | all | OFF | 7812 | 4106 | 52.6% | 233 | 17.62 | 5.33 | 12.80 | 87.1% | 18.62 | 29.6% |
