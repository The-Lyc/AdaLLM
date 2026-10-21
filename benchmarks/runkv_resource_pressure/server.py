# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own process-mode pressure workers outside an Nsight target process tree."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from benchmarks.runkv_resource_pressure.controller import (
    ResourcePressureConfig,
    ResourcePressureController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--kind", choices=["io", "sm"], required=True)
    parser.add_argument("--clock", choices=["step", "time"], default="step")
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer-mb", type=int, default=256)
    parser.add_argument(
        "--direction",
        choices=["h2d", "d2h", "bidirectional"],
        default="h2d",
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--io-calibration-s", type=float, default=0.5)
    parser.add_argument("--io-max-gbps", type=float, default=None)
    return parser.parse_args()


def _send_response(connection: socket.socket, response: dict[str, object]) -> None:
    connection.sendall((json.dumps(response) + "\n").encode())


def _serve_connection(
    connection: socket.socket,
    controller: ResourcePressureController,
    *,
    parent_pid: int | None,
) -> bool:
    """Serve one client. Return true when the whole server should stop."""
    connection.settimeout(1.0)
    pending = b""
    while parent_pid is None or os.getppid() == parent_pid:
        try:
            chunk = connection.recv(65536)
        except TimeoutError:
            continue
        if not chunk:
            return False
        pending += chunk
        while b"\n" in pending:
            raw_request, pending = pending.split(b"\n", 1)
            try:
                request = json.loads(raw_request)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                operation = request.get("op")
                if operation == "status":
                    response: dict[str, object] = {
                        "ok": True,
                        "kind": controller.config.kind,
                        "capacity": controller._process_capacity(),
                        "max_workers": controller._max_process_worker_count,
                        "worker_count": controller._process_worker_count,
                    }
                elif operation == "set_worker_count":
                    count = int(request["count"])
                    actual = controller._set_process_worker_count(count)
                    response = {
                        "ok": True,
                        "actual": actual,
                        "worker_count": controller._process_worker_count,
                    }
                elif operation == "shutdown":
                    controller._set_process_worker_count(0)
                    _send_response(connection, {"ok": True})
                    return True
                else:
                    raise ValueError(f"unsupported operation: {operation!r}")
            except (KeyError, TypeError, ValueError) as exc:
                response = {"ok": False, "error": str(exc)}
            _send_response(connection, response)
    return True


def serve(args: argparse.Namespace) -> None:
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    controller = ResourcePressureController(
        ResourcePressureConfig(
            kind=args.kind,
            clock=args.clock,
            pattern=args.pattern,
            device=args.device,
            buffer_mb=args.buffer_mb,
            direction=args.direction,
            matrix_size=args.matrix_size,
            dtype=args.dtype,
            io_calibration_s=args.io_calibration_s,
            io_max_gbps=args.io_max_gbps,
            mode="process",
        )
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        controller.prepare()
        controller.start()
        listener.bind(str(socket_path))
        listener.listen(2)
        listener.settimeout(1.0)
        print(
            "Resource pressure server ready: "
            f"socket={socket_path} kind={args.kind} "
            f"workers={controller._max_process_worker_count}",
            flush=True,
        )
        should_stop = False
        while not should_stop:
            if args.parent_pid is not None and os.getppid() != args.parent_pid:
                break
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                should_stop = _serve_connection(
                    connection,
                    controller,
                    parent_pid=args.parent_pid,
                )
    finally:
        controller.stop()
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
