"""RDJ-10: node start gate, cached Vulkan readback and bounded nvidia-smi retries.

EXP11A/motion-ab on a shared simulator host (2026-10-07) lost 33 starts before the first model
call to 120 s Vulkan readback and 15 s nvidia-smi timeouts while a single
readback took 30 s: concurrent driver-level initializations queued.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from ora.environment.backends.robodojo import bootstrap, graphics_compat, start_gate


def test_a_gate_admits_its_slots_and_refuses_after_its_timeout(tmp_path):
    with start_gate.gate("g", 2, timeout_s=1, directory=tmp_path) as first:
        with start_gate.gate("g", 2, timeout_s=1, directory=tmp_path) as second:
            assert {first["slot"], second["slot"]} == {0, 1}
            with pytest.raises(TimeoutError):
                with start_gate.gate("g", 2, timeout_s=0.6, directory=tmp_path):
                    pass
    with start_gate.gate("g", 2, timeout_s=1, directory=tmp_path) as again:
        assert again["waited_s"] < 0.5  # released on exit


def test_a_waiting_holder_enters_once_a_slot_is_released(tmp_path):
    entered = []

    def late():
        with start_gate.gate("g", 1, timeout_s=5, directory=tmp_path) as record:
            entered.append(record["waited_s"])

    with start_gate.gate("g", 1, timeout_s=1, directory=tmp_path):
        thread = threading.Thread(target=late)
        thread.start()
        time.sleep(1.0)
        assert entered == []
    thread.join(5)
    assert entered and entered[0] >= 0.9


def test_a_readback_is_measured_once_per_node_boot_and_inputs(tmp_path):
    calls = []

    def produce():
        calls.append(1)
        return {"after": [len(calls)]}

    first, p1 = start_gate.cached("r", {"files": "a"}, produce, timeout_s=5, directory=tmp_path)
    second, p2 = start_gate.cached("r", {"files": "a"}, produce, timeout_s=5, directory=tmp_path)
    other, p3 = start_gate.cached("r", {"files": "b"}, produce, timeout_s=5, directory=tmp_path)
    assert first == second == {"after": [1]} and other == {"after": [2]}
    assert (p1["cache"], p2["cache"], p3["cache"]) == ("measured", "hit", "measured")


def test_a_reboot_or_driver_change_measures_again(tmp_path, monkeypatch):
    produce = iter([{"v": 1}, {"v": 2}]).__next__
    assert start_gate.cached("r", {}, produce, timeout_s=5, directory=tmp_path)[0] == {"v": 1}
    monkeypatch.setattr(start_gate, "node_identity", lambda: {"boot_id": "other", "driver_version_sha256": "x"})
    assert start_gate.cached("r", {}, produce, timeout_s=5, directory=tmp_path)[0] == {"v": 2}


def test_a_failed_measurement_publishes_nothing(tmp_path):
    def broken():
        raise graphics_compat.GraphicsCompatibilityError("drift")

    with pytest.raises(graphics_compat.GraphicsCompatibilityError):
        start_gate.cached("r", {}, broken, timeout_s=5, directory=tmp_path)
    assert not list(tmp_path.glob("r-*.json"))


def test_the_script_entry_loads_the_gate_by_path():
    module = graphics_compat._start_gate()
    assert module.SCHEMA == start_gate.SCHEMA


def test_the_cached_readback_keys_on_the_verified_installation(tmp_path, monkeypatch):
    receipt = {role: {"sha256": role * 2} for role in ("layer", "profile", "manifest", "patch")}
    monkeypatch.setattr(graphics_compat, "installation", lambda directory: (receipt, {}))
    monkeypatch.setattr(graphics_compat, "readback", lambda directory: {"schema": graphics_compat.SCHEMA})
    monkeypatch.setattr(start_gate, "root", lambda: tmp_path)
    value, provenance = graphics_compat.cached_readback(tmp_path)
    again, hit = graphics_compat.cached_readback(tmp_path)
    assert value == again == {"schema": graphics_compat.SCHEMA}
    assert (provenance["cache"], hit["cache"]) == ("measured", "hit")


def test_nvidia_smi_is_retried_once_after_a_timeout(monkeypatch):
    answers = [subprocess.TimeoutExpired("nvidia-smi", 60), "0, GPU-x, 0000:00\n"]

    def fake(command, **kwargs):
        assert kwargs["timeout"] == bootstrap.NVIDIA_SMI_TIMEOUT_S
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(subprocess, "check_output", fake)
    assert bootstrap.nvidia_smi(["--query-gpu=index"]).startswith("0,")
    answers[:] = [subprocess.TimeoutExpired("nvidia-smi", 60)] * 2
    with pytest.raises(subprocess.TimeoutExpired):
        bootstrap.nvidia_smi(["--query-gpu=index"])
