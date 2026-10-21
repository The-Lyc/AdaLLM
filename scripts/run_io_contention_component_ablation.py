#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the OPT-2.7B component ablation under staged IO contention.

With Nsight enabled, contention workers run in an external sibling service so
the report profiles only the RunKV target process while retaining step-clock
pressure transitions.
"""

from pathlib import Path

from run_no_throttle_component_ablation import ROOT, main


if __name__ == "__main__":
    main(
        default_settings_config=(
            ROOT / "configs/staged_resource_benchmark_opt_2.7b_io.json"
        ),
        default_output_root=ROOT / "exp_results/ablation/io_contention",
        default_disable_nsys=False,
    )
