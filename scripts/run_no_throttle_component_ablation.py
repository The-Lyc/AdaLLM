#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run RunKV component ablations under one shared benchmark setting.

The default matrix runs a same-version full-on reference plus variants that
disable exactly one component. Workload and resource settings are loaded from
one benchmark-batch JSON, including optional staged resource pressure.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
OPT_RUNNER = ROOT / "examples/offline_inference/run_opt_feedback_observation.py"
LLAMA_RUNNER = ROOT / "examples/offline_inference/run_llama_feedback_observation.py"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/ablation/no_throttle"
DEFAULT_SETTINGS_CONFIG = ROOT / "configs/benchmark_batch_opt_2.7b.json"
PRESSURE_SERVER_MODULE = "benchmarks.runkv_resource_pressure.server"

RESOURCE_PRESSURE_DEFAULTS = {
    "resource_pressure_kind": "none",
    "resource_pressure_clock": "step",
    "resource_pressure_pattern": "0:0",
    "resource_pressure_device": "cuda:0",
    "resource_pressure_buffer_mb": "256",
    "resource_pressure_direction": "h2d",
    "resource_pressure_matrix_size": "4096",
    "resource_pressure_dtype": "float16",
    "resource_pressure_window_s": "0.25",
    "resource_pressure_period_ms": "100.0",
    "resource_pressure_max_fraction": "0.5",
    "resource_pressure_io_calibration_s": "0.5",
    "resource_pressure_io_max_gbps": "",
    "resource_pressure_mode": "thread",
    "resource_pressure_inline_layer_period_ms": "5.0",
}

RESOURCE_ENV_KEYS = {key: key.upper() for key in RESOURCE_PRESSURE_DEFAULTS}

RESOURCE_PRESSURE_CHOICES = {
    "resource_pressure_kind": {"none", "io", "sm"},
    "resource_pressure_clock": {"step", "time"},
    "resource_pressure_direction": {"h2d", "d2h", "bidirectional"},
    "resource_pressure_dtype": {"float16", "bfloat16"},
    "resource_pressure_mode": {"thread", "inline", "throttle", "process"},
}

REQUIRED_SETTING_FIELDS = (
    "model",
    "prefix_blocks",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "gpu_memory_utilization",
    "gpu_memory_fraction",
    "num_device_buffers",
    "batch_size",
    "prompt_length",
    "decode_length",
    "runkv_replay_allocation_policy",
)

VARIANTS: dict[str, dict[str, str | bool]] = {
    "full": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "segment",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "none",
        "io_fallback": "segment_dma",
    },
    "no_async_plan": {
        "ASYNC_PLAN_BUILD": "0",
        "H2D_COPY_MODE": "segment",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "async_spec_plan_build",
    },
    "no_segment_dma": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "gather",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "segment_dma",
        "io_fallback": "cpu_gather_dma",
    },
    "no_segment_dma_uva": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "uva",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "segment_dma",
        "io_fallback": "gpu_uva_batch_copy",
    },
    "no_state_machine": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "segment",
        "USE_STATE_MACHINE": "0",
        "disabled_component": "state_machine",
    },
}


def parse_args(
    *,
    default_settings_config: Path = DEFAULT_SETTINGS_CONFIG,
    default_output_root: Path = DEFAULT_OUTPUT_ROOT,
    default_disable_nsys: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings-config", type=Path, default=default_settings_config
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=None,
        metavar="WORKLOAD",
        help="Workload keys derived from prompt length, e.g. 1k 2k 4k 8k.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument(
        "--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        default=None,
        help="Override GPU staging-buffer fraction for selected workloads.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Override request batch size for selected workloads.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print jobs without executing."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Within --run-tag, skip runs with a complete manifest and "
            "expected artifacts."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one short request and two output tokens to validate switch paths.",
    )
    nsys = parser.add_mutually_exclusive_group()
    nsys.add_argument(
        "--enable-nsys", dest="disable_nsys", action="store_false"
    )
    nsys.add_argument(
        "--disable-nsys", dest="disable_nsys", action="store_true"
    )
    parser.set_defaults(disable_nsys=default_disable_nsys)
    parser.add_argument("--disable-profile", action="store_true")
    parser.add_argument("--skip-sqlite", action="store_true")
    parser.add_argument("--nsys-cmd", default=os.environ.get("NSYS_CMD", "nsys"))
    return parser.parse_args()


def _load_benchmark_settings(
    config_path: Path,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    resolved = config_path.expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    config = _load_json(resolved)
    if not isinstance(config, dict):
        raise ValueError("settings config root must be an object")
    defaults = config.get("defaults", {})
    tests = config.get("tests", [])
    if not isinstance(defaults, dict) or not isinstance(tests, list) or not tests:
        raise ValueError(
            "settings config requires an object 'defaults' and a non-empty 'tests'"
        )

    settings: dict[str, dict[str, Any]] = {}
    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise ValueError(f"tests[{index}] must be an object")
        setting = {**defaults, **test}
        missing = [key for key in REQUIRED_SETTING_FIELDS if key not in setting]
        if missing:
            raise ValueError(f"tests[{index}] is missing: {', '.join(missing)}")
        prompt_length = int(setting["prompt_length"])
        if prompt_length <= 0 or prompt_length % 1000 != 0:
            raise ValueError(
                f"tests[{index}].prompt_length must be a positive multiple of 1000"
            )
        workload = f"{prompt_length // 1000}k"
        if workload in settings:
            raise ValueError(
                f"duplicate workload derived from prompt length: {workload}"
            )
        settings[workload] = setting
    return resolved.resolve(), settings


def _resource_pressure_settings(setting: dict[str, Any]) -> dict[str, str]:
    pressure = {
        key: str(setting.get(key, default))
        for key, default in RESOURCE_PRESSURE_DEFAULTS.items()
    }
    for key, choices in RESOURCE_PRESSURE_CHOICES.items():
        if pressure[key] not in choices:
            choices_text = ", ".join(sorted(choices))
            raise ValueError(
                f"{key} must be one of: {choices_text}; got {pressure[key]!r}"
            )
    if pressure["resource_pressure_kind"] != "none":
        if pressure["resource_pressure_pattern"] in {"", "0:0"}:
            raise ValueError(
                "resource_pressure_pattern must define a non-zero staged "
                "schedule when resource pressure is enabled"
            )
    return pressure


def _run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    print(f"  CMD: {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def _uses_external_pressure_server(
    args: argparse.Namespace, pressure: dict[str, str]
) -> bool:
    return (
        not args.disable_nsys
        and pressure["resource_pressure_kind"] != "none"
        and pressure["resource_pressure_mode"] == "process"
    )


def _pressure_process_placement(
    args: argparse.Namespace, pressure: dict[str, str]
) -> str:
    if pressure["resource_pressure_kind"] == "none":
        return "not_applicable"
    if pressure["resource_pressure_mode"] != "process":
        return "in_target_process"
    if _uses_external_pressure_server(args, pressure):
        return "external_sibling"
    return "target_process_child"


def _pressure_server_command(
    pressure: dict[str, str], socket_path: str
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        PRESSURE_SERVER_MODULE,
        "--socket",
        socket_path,
        "--parent-pid",
        str(os.getpid()),
        "--kind",
        pressure["resource_pressure_kind"],
        "--clock",
        pressure["resource_pressure_clock"],
        "--pattern",
        pressure["resource_pressure_pattern"],
        "--device",
        pressure["resource_pressure_device"],
        "--buffer-mb",
        pressure["resource_pressure_buffer_mb"],
        "--direction",
        pressure["resource_pressure_direction"],
        "--matrix-size",
        pressure["resource_pressure_matrix_size"],
        "--dtype",
        pressure["resource_pressure_dtype"],
        "--io-calibration-s",
        pressure["resource_pressure_io_calibration_s"],
    ]
    if pressure["resource_pressure_io_max_gbps"]:
        cmd.extend(
            ["--io-max-gbps", pressure["resource_pressure_io_max_gbps"]]
        )
    return cmd


def _pressure_server_request(
    socket_path: str, request: dict[str, object], *, timeout_s: float = 2.0
) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_s)
        client.connect(socket_path)
        client.sendall((json.dumps(request) + "\n").encode())
        with client.makefile("rb") as reader:
            response_line = reader.readline()
    if not response_line:
        raise RuntimeError("Resource pressure server closed without a response.")
    response = json.loads(response_line)
    if not isinstance(response, dict) or not response.get("ok"):
        error = (
            response.get("error", "unknown error")
            if isinstance(response, dict)
            else f"malformed response {response!r}"
        )
        raise RuntimeError(
            "Resource pressure server rejected request: "
            f"{error}"
        )
    return response


@contextlib.contextmanager
def _external_pressure_server(
    pressure: dict[str, str], log_path: Path
) -> Iterator[str]:
    with tempfile.TemporaryDirectory(prefix="runkv-rp-") as temp_dir:
        socket_path = str(Path(temp_dir) / "control.sock")
        cmd = _pressure_server_command(pressure, socket_path)
        print(
            "  PRESSURE-SERVER: "
            + " ".join(shlex.quote(part) for part in cmd),
            flush=True,
        )
        with log_path.open("w") as log:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 60.0
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Resource pressure server exited during startup; "
                            f"see {log_path}."
                        )
                    try:
                        _pressure_server_request(socket_path, {"op": "status"})
                        break
                    except (OSError, RuntimeError, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "Resource pressure server did not become ready; "
                                f"see {log_path}."
                            )
                        time.sleep(0.1)
                yield socket_path
            finally:
                if process.poll() is None:
                    try:
                        _pressure_server_request(
                            socket_path, {"op": "shutdown"}, timeout_s=5.0
                        )
                    except (OSError, RuntimeError, json.JSONDecodeError):
                        pass
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _runner_for_model(model: str) -> tuple[Path, str]:
    config_path = Path(model).expanduser() / "config.json"
    try:
        model_config = _load_json(config_path)
    except (OSError, json.JSONDecodeError):
        return OPT_RUNNER, "opt"
    if model_config.get("model_type") == "llama":
        return LLAMA_RUNNER, "llama"
    return OPT_RUNNER, str(model_config.get("model_type") or "opt")


def _export_sqlite(
    args: argparse.Namespace, manifest: dict[str, Any], out: Path
) -> None:
    report = manifest.get("nsys_report")
    if not report:
        raise RuntimeError("Nsight report missing from manifest; cannot export sqlite.")
    cmd = [
        args.nsys_cmd,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "-o",
        str(out),
        str(report),
    ]
    print(f"  SQLITE: {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _is_complete(
    args: argparse.Namespace,
    manifest_path: Path,
    expected_config: dict[str, str],
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    for key, expected in expected_config.items():
        if str(manifest.get(key)) != expected:
            return False
    flat_glob = manifest.get("mfu_flat_jsonl_glob")
    if not flat_glob or not glob.glob(flat_glob):
        return False
    if not args.disable_nsys and not Path(manifest.get("nsys_report", "")).exists():
        return False
    if not args.skip_sqlite and not args.disable_nsys:
        sqlite = manifest.get("sqlite")
        if not sqlite or not Path(sqlite).exists():
            return False
    if manifest.get("resource_pressure_kind") != "none":
        for key in (
            "resource_pressure_log_path",
            "resource_pressure_step_log_path",
        ):
            path = manifest.get(key)
            if not path or not Path(path).exists():
                return False
    step_metrics_path = manifest.get("step_metrics_path")
    if args.disable_nsys and (
        not step_metrics_path or not Path(step_metrics_path).exists()
    ):
        return False
    return True


def main(
    *,
    default_settings_config: Path = DEFAULT_SETTINGS_CONFIG,
    default_output_root: Path = DEFAULT_OUTPUT_ROOT,
    default_disable_nsys: bool = False,
) -> None:
    args = parse_args(
        default_settings_config=default_settings_config,
        default_output_root=default_output_root,
        default_disable_nsys=default_disable_nsys,
    )
    output_root = args.output_root.resolve()
    try:
        settings_config, benchmark_settings = _load_benchmark_settings(
            args.settings_config
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid --settings-config: {exc}") from exc

    workloads = args.workloads or list(benchmark_settings)
    unknown = [
        workload for workload in workloads if workload not in benchmark_settings
    ]
    if unknown:
        choices = ", ".join(benchmark_settings)
        raise SystemExit(
            f"Unknown workload(s): {', '.join(unknown)}; available: {choices}"
        )

    for workload in workloads:
        setting = benchmark_settings[workload]
        try:
            pressure = _resource_pressure_settings(setting)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid resource-pressure setting for {workload}: {exc}"
            ) from exc
        for variant in args.variants:
            switches = VARIANTS[variant]
            runner, model_family = _runner_for_model(str(setting["model"]))
            process_placement = _pressure_process_placement(args, pressure)
            nsys_profile_scope = (
                "runkv_main_process_only"
                if process_placement == "external_sibling"
                else ("target_process_tree" if not args.disable_nsys else "disabled")
            )
            gpu_memory_fraction = (
                args.gpu_memory_fraction or str(setting["gpu_memory_fraction"])
            )
            num_prompts = (
                str(args.num_prompts)
                if args.num_prompts is not None and not args.smoke
                else ("1" if args.smoke else str(setting["batch_size"]))
            )
            prompt_words = "64" if args.smoke else str(setting["prompt_length"])
            max_tokens = "2" if args.smoke else str(setting["decode_length"])
            suffix = "_smoke" if args.smoke else ""
            run_tag = f"{args.run_tag}_{workload}_{variant}{suffix}"
            run_dir = output_root / workload / variant / args.run_tag
            manifest_path = run_dir / "manifest.json"
            sqlite_path = run_dir / "runkv.sqlite"
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": str(setting["model"]),
                    "MODEL_FAMILY": model_family,
                    "PREFIX_BLOCKS": str(setting["prefix_blocks"]),
                    "PROMPT_WORD": str(
                        setting.get(
                            "prompt_word",
                            "the" if model_family == "llama" else "replay",
                        )
                    ),
                    "NUM_PROMPTS": num_prompts,
                    "PROMPT_WORDS": prompt_words,
                    "MAX_TOKENS": max_tokens,
                    "GPU_MEMORY_FRACTION": gpu_memory_fraction,
                    "GPU_MEMORY_UTILIZATION": str(
                        setting["gpu_memory_utilization"]
                    ),
                    "NUM_DEVICE_BUFFERS": str(setting["num_device_buffers"]),
                    "MAX_NUM_SEQS": (
                        ""
                        if setting.get("max_num_seqs") is None
                        else str(setting["max_num_seqs"])
                    ),
                    "MAX_STAGING_BLOCKS": (
                        ""
                        if setting.get("max_staging_blocks") is None
                        else str(setting["max_staging_blocks"])
                    ),
                    "CPU_MEMORY_GB": str(setting["cpu_memory_gb"]),
                    "CPU_MEMORY_FRACTION": str(setting["cpu_memory_fraction"]),
                    "PLANNER": "feedback",
                    "DRY_RUN": "0",
                    "REPLAY_ALLOCATION_POLICY": str(
                        setting["runkv_replay_allocation_policy"]
                    ),
                    "ASYNC_PLAN_BUILD": str(switches["ASYNC_PLAN_BUILD"]),
                    "H2D_COPY_MODE": str(switches["H2D_COPY_MODE"]),
                    "USE_STATE_MACHINE": str(switches["USE_STATE_MACHINE"]),
                    "ENABLE_NVTX": "1",
                    "ENABLE_PROFILE": "0" if args.disable_profile else "1",
                    "ENABLE_OPT_COMPONENT_MFU_PROFILING": "1",
                    "RUNKV_PREHOOK_TIMING": "1",
                    "RUNKV_PREHOOK_TIMING_DIR": str(run_dir / "prehook_timing"),
                    "ENABLE_NSYS": "0" if args.disable_nsys else "1",
                    "NSYS_SAMPLE": "cpu",
                    "NSYS_EXTRA_ARGS": (
                        "--capture-range=cudaProfilerApi --capture-range-end=stop"
                    ),
                    "NSYS_OUTPUT_DIR": str(run_dir),
                    "OUTPUT_DIR": str(run_dir),
                    "RUN_TAG": run_tag,
                    "MANIFEST_FILE": str(manifest_path),
                    "STEP_METRICS_PATH": str(run_dir / "step_metrics.jsonl"),
                    "RESOURCE_PRESSURE_PROCESS_PLACEMENT": process_placement,
                    "NSYS_PROFILE_SCOPE": nsys_profile_scope,
                }
            )
            env.update(
                {
                    env_key: pressure[key]
                    for key, env_key in RESOURCE_ENV_KEYS.items()
                }
            )
            env.update(
                {
                    "RESOURCE_PRESSURE_LOG_PATH": str(run_dir / "pressure.csv"),
                    "RESOURCE_PRESSURE_STEP_LOG_PATH": str(
                        run_dir / "resource_steps.jsonl"
                    ),
                }
            )
            cmd = [sys.executable, str(runner)]
            print(f"\n[{workload}/{variant}] output={run_dir}")
            print(
                "  SETTING: "
                f"model={setting['model']} family={model_family} "
                f"runner={runner.name} bs={num_prompts} "
                f"prompt={prompt_words} decode={max_tokens} "
                f"cpu_gb={setting['cpu_memory_gb']} "
                f"cpu_frac={setting['cpu_memory_fraction']} "
                f"gpu_util={setting['gpu_memory_utilization']} "
                f"gpu_frac={gpu_memory_fraction} "
                f"buffers={setting['num_device_buffers']} "
                f"prefix_blocks={setting['prefix_blocks']} "
                f"allocation={setting['runkv_replay_allocation_policy']}"
            )
            print(
                "  SWITCHES: "
                f"async_plan={switches['ASYNC_PLAN_BUILD']} "
                f"h2d_copy={switches['H2D_COPY_MODE']} "
                f"state_machine={switches['USE_STATE_MACHINE']}"
            )
            print(
                "  RESOURCE: "
                f"kind={pressure['resource_pressure_kind']} "
                f"clock={pressure['resource_pressure_clock']} "
                f"pattern={pressure['resource_pressure_pattern']} "
                f"mode={pressure['resource_pressure_mode']} "
                f"device={pressure['resource_pressure_device']} "
                f"direction={pressure['resource_pressure_direction']} "
                f"io_max_gbps={pressure['resource_pressure_io_max_gbps'] or 'auto'} "
                f"placement={process_placement}"
            )
            print(f"  PROFILE-SCOPE: {nsys_profile_scope}")
            expected_config = {
                "model": str(setting["model"]),
                "prefix_blocks": str(setting["prefix_blocks"]),
                "num_prompts": num_prompts,
                "prompt_words": prompt_words,
                "max_tokens": max_tokens,
                "gpu_memory_fraction": gpu_memory_fraction,
                "gpu_memory_utilization": str(setting["gpu_memory_utilization"]),
                "num_device_buffers": str(setting["num_device_buffers"]),
                "max_num_seqs": str(setting.get("max_num_seqs")),
                "max_staging_blocks": str(setting.get("max_staging_blocks")),
                "cpu_memory_gb": str(setting["cpu_memory_gb"]),
                "cpu_memory_fraction": str(setting["cpu_memory_fraction"]),
                "layer_recompute_replay_allocation_policy": str(
                    setting["runkv_replay_allocation_policy"]
                ),
                "layer_recompute_async_plan_build": str(
                    switches["ASYNC_PLAN_BUILD"] == "1"
                ),
                "h2d_copy_mode": str(switches["H2D_COPY_MODE"]),
                "layer_recompute_use_state_machine": str(
                    switches["USE_STATE_MACHINE"] == "1"
                ),
                **{
                    key: ("None" if value == "" else value)
                    for key, value in pressure.items()
                },
            }
            if _uses_external_pressure_server(args, pressure):
                expected_config.update(
                    {
                        "resource_pressure_process_placement": process_placement,
                        "nsys_profile_scope": nsys_profile_scope,
                    }
                )
            if args.resume and _is_complete(args, manifest_path, expected_config):
                print("  SKIP: completed run already exists for this tag")
                continue
            if args.dry_run:
                if _uses_external_pressure_server(args, pressure):
                    pressure_cmd = _pressure_server_command(
                        pressure, "<external-pressure-socket>"
                    )
                    print(
                        "  PRESSURE-SERVER: "
                        + " ".join(
                            shlex.quote(part) for part in pressure_cmd
                        )
                    )
                    print(
                        "  ENV: RESOURCE_PRESSURE_EXTERNAL_PROCESS_SOCKET="
                        "<external-pressure-socket>"
                    )
                print(f"  CMD: {' '.join(shlex.quote(part) for part in cmd)}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            if _uses_external_pressure_server(args, pressure):
                with _external_pressure_server(
                    pressure, run_dir / "pressure_server.log"
                ) as pressure_socket:
                    run_env = env.copy()
                    run_env[
                        "RESOURCE_PRESSURE_EXTERNAL_PROCESS_SOCKET"
                    ] = pressure_socket
                    _run_logged(cmd, run_env, run_dir / "run.log")
            else:
                _run_logged(cmd, env, run_dir / "run.log")
            manifest = _load_json(manifest_path)
            manifest.update(
                {
                    "ablation": (
                        "no_throttle_component"
                        if pressure["resource_pressure_kind"] == "none"
                        else (
                            f"{pressure['resource_pressure_kind']}"
                            "_contention_component"
                        )
                    ),
                    "workload": workload,
                    "variant": variant,
                    "disabled_component": switches["disabled_component"],
                    "io_fallback": switches.get("io_fallback"),
                    "smoke": args.smoke,
                    "settings_config": str(settings_config),
                    "benchmark_setting_name": setting.get("name"),
                    "resource_pressure_process_placement": process_placement,
                    "nsys_profile_scope": nsys_profile_scope,
                }
            )
            if not args.skip_sqlite and not args.disable_nsys:
                _export_sqlite(args, manifest, sqlite_path)
                manifest["sqlite"] = str(sqlite_path.resolve())
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"  DONE: {manifest_path}")


if __name__ == "__main__":
    main()
