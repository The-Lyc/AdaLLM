# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.runkv_resource_pressure import controller as controller_module
from benchmarks.runkv_resource_pressure.controller import (
    ResourcePressureConfig,
    ResourcePressureController,
)


class _ConnectedSocket:
    def __init__(self, worker_counts: list[int]) -> None:
        self._worker_counts = worker_counts
        self._responses: list[bytes] = []

    def settimeout(self, timeout: float) -> None:
        del timeout

    def connect(self, path: str) -> None:
        del path

    def sendall(self, payload: bytes) -> None:
        request = json.loads(payload)
        if request["op"] == "status":
            response = {
                "ok": True,
                "kind": "io",
                "capacity": 24.0,
                "max_workers": 2,
                "worker_count": 0,
            }
        elif request["op"] == "set_worker_count":
            count = int(request["count"])
            self._worker_counts.append(count)
            response = {
                "ok": True,
                "worker_count": count,
                "actual": 24.0 / float(count + 1),
            }
        else:
            response = {"ok": False, "error": "unexpected operation"}
        self._responses.append((json.dumps(response) + "\n").encode())

    def makefile(self, mode: str):
        assert mode == "rb"
        return self

    def readline(self) -> bytes:
        return self._responses.pop(0)

    def close(self) -> None:
        pass


def test_external_process_controller_only_updates_server_at_stage_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "pressure.sock"
    worker_counts: list[int] = []
    fake_client = _ConnectedSocket(worker_counts)
    monkeypatch.setattr(
        controller_module.socket, "socket", lambda *args: fake_client
    )
    monkeypatch.setattr(
        controller_module.torch.cuda, "is_available", lambda: True
    )

    controller = ResourcePressureController(
        ResourcePressureConfig(
            kind="io",
            clock="step",
            pattern="0:24,16:12,32:8",
            mode="process",
            io_max_gbps=None,
            external_process_socket=str(socket_path),
        )
    )
    controller.prepare()
    controller.start()
    assert controller.before_step(0)["resource_pressure_process_workers"] == 0
    assert controller.before_step(15)["resource_pressure_process_workers"] == 0
    assert controller.before_step(16)["resource_pressure_process_workers"] == 1
    assert controller.before_step(31)["resource_pressure_process_workers"] == 1
    assert controller.before_step(32)["resource_pressure_process_workers"] == 2
    assert controller.before_step(33)["resource_pressure_process_workers"] == 2
    controller.stop()

    assert worker_counts == [1, 2, 0]
    assert controller._processes == []
