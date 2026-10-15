"""Kit worker PID collisions with stale carb segments, and the OptiX cache path.

A Kit worker whose PID matches another user's stale
``/dev/shm/carb-RStringInternals-<pid>`` exits before Kit starts and the
launcher relaunches it under a new PID (run 683e36fb73af aborted instead).
Non-Kit workers are neither checked nor relaunched. Every ``/dev/shm`` here is
a temporary directory; worker processes are real, Kit is never started.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ora.environment import rpc, worker
from ora.environment.backends.robodojo import bootstrap
from ora.orchestrator.shm_segments import CARB_SEGMENT

ROOT = Path(__file__).resolve().parents[3]
ME = os.getuid()


def _segments(root: Path, pid: int) -> list[Path]:
    paths = [root / f"carb-RStringInternals-{pid}", root / f"sem.carb-RStringInternals-{pid}"]
    for path in paths:
        path.write_bytes(b"\0")
    return paths


def test_the_worker_checks_exactly_the_names_the_launcher_sweeps():
    assert all(CARB_SEGMENT.match(name) for name in (
        "carb-RStringInternals-4194304", "sem.carb-RStringInternals-4194304"))
    assert worker.KIT_BACKENDS == rpc.KIT_BACKENDS == {"robodojo"}


@pytest.mark.parametrize("backend", ["robotwin", "libero_pro", "synthetic"])
def test_non_kit_workers_are_unchanged(tmp_path, backend):
    _segments(tmp_path, 4242)
    assert worker.kit_pid_collision({"backend": backend}, pid=4242, uid=ME + 1, root=tmp_path) is None
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("backend", ["robodojo"])
def test_a_foreign_segment_is_a_collision_and_an_own_one_is_removed(tmp_path, backend):
    _segments(tmp_path, 4242)
    assert worker.kit_pid_collision({"backend": backend}, pid=4242, uid=ME + 1, root=tmp_path) == (
        "carb-RStringInternals-4242")
    assert worker.kit_pid_collision({"backend": backend}, pid=4242, uid=ME, root=tmp_path) is None
    assert list(tmp_path.iterdir()) == []
    assert worker.kit_pid_collision({"backend": backend}, pid=4243, uid=ME + 1, root=tmp_path) is None


def test_the_shm_root_is_read_at_call_time(tmp_path, monkeypatch):
    _segments(tmp_path, 4194304)  # above any issuable PID: never a real segment
    monkeypatch.setattr(worker, "SHM_ROOT", tmp_path)
    assert worker.kit_pid_collision({"backend": "robodojo"}, pid=4194304, uid=ME + 1) == (
        "carb-RStringInternals-4194304")


def test_a_colliding_worker_reports_the_segment_and_exits_before_the_session(tmp_path, monkeypatch):
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"schema": rpc.RPC_SCHEMA, "session_id": "env-x", "backend": "robodojo"}))
    monkeypatch.setattr(worker, "kit_pid_collision", lambda *args, **kwargs: "carb-RStringInternals-1")
    monkeypatch.setattr(worker, "_build_session", lambda config: pytest.fail("Kit must not start"))
    assert worker.serve(tmp_path / "s.sock", config) == rpc.PID_SEGMENT_COLLISION_EXIT
    report = json.loads((tmp_path / rpc.PID_COLLISION_FILE).read_text())
    assert report == {"worker_pid": os.getpid(), "segment": "carb-RStringInternals-1"}


def _fake_kit_project(tmp_path: Path) -> Path:
    """The real worker with a Synthetic session under a Kit backend name."""

    project = tmp_path / "fake-kit-worker"
    environment = project / "ora/environment"
    environment.mkdir(parents=True)
    shm = tmp_path / "fake-shm"
    shm.mkdir()
    (project / "ora/__init__.py").write_text(f"__path__.append({str(ROOT / 'ora')!r})\n")
    (environment / "__init__.py").write_text(
        f"__path__.append({str(ROOT / 'ora/environment')!r})\n")
    (environment / "worker.py").write_text(
        f"REAL_WORKER = {str(ROOT / 'ora/environment/worker.py')!r}\n"
        f"FAKE_SHM = {str(shm)!r}\n" + r'''
from pathlib import Path
from ora.environment.synthetic import SyntheticSession
namespace = {"__name__": "fixture_rpc_worker", "__package__": "ora.environment"}
exec(compile(Path(REAL_WORKER).read_text(), REAL_WORKER, "exec"), namespace)
namespace["SHM_ROOT"] = Path(FAKE_SHM)
namespace["_build_session"] = lambda config: SyntheticSession(
    Path(config["output_root"]), config["task"], config["seed"])
raise SystemExit(namespace["main"]())
''')
    return project


class _Exited:
    def __init__(self, pid: int, code: int) -> None:
        self.pid, self.returncode = pid, code

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _session(tmp_path: Path, backend: str) -> rpc.LocalRpcEnvironmentSession:
    output = tmp_path / "native"
    return rpc.LocalRpcEnvironmentSession(
        python_executable=Path(sys.executable),
        project_root=_fake_kit_project(tmp_path) if backend in rpc.KIT_BACKENDS else ROOT,
        output_root=output, socket_root=tmp_path / "t", startup_timeout_s=30, request_timeout_s=30,
        worker_config={"backend": backend, "output_root": str(output), "task": "pid", "seed": 0},
    )


def _popen(monkeypatch, codes: list[int]) -> list[int]:
    """Fake the first launches as reported collisions, then spawn for real."""

    real, spawned = subprocess.Popen, []

    def popen(command, *args, **kwargs):
        spawned.append(len(spawned))
        if codes:
            code = codes.pop(0)
            if code == rpc.PID_SEGMENT_COLLISION_EXIT:
                (Path(command[4]).parent / rpc.PID_COLLISION_FILE).write_text(json.dumps(
                    {"worker_pid": 90_000 + len(spawned), "segment": f"carb-RStringInternals-{len(spawned)}"}))
            return _Exited(90_000 + len(spawned), code)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(rpc.subprocess, "Popen", popen)
    return spawned


def test_a_kit_launcher_relaunches_after_a_collision_and_records_the_segment(tmp_path, monkeypatch):
    spawned = _popen(monkeypatch, [rpc.PID_SEGMENT_COLLISION_EXIT])
    session = _session(tmp_path, "robodojo")
    try:
        assert session.start()["task"] == "pid"
        record = json.loads((tmp_path / "native/environment-session.json").read_text())
        assert len(spawned) == 2
        assert [(item["worker_pid"], item["segment"]) for item in record["pid_collision_relaunches"]] == [
            (90_001, "carb-RStringInternals-1")]
        assert record["worker_pid"] != 90_001
        assert not (session.transport_root / rpc.PID_COLLISION_FILE).exists()
    finally:
        session.close()


@pytest.mark.parametrize("backend, code", [("synthetic", 73), ("robotwin", 73), ("robodojo", 1)])
def test_other_backends_and_other_exits_are_never_relaunched(tmp_path, monkeypatch, backend, code):
    spawned = _popen(monkeypatch, [code])
    session = _session(tmp_path, backend)
    with pytest.raises(rpc.EnvironmentRpcError, match=f"code {code}"):
        session.start()
    assert len(spawned) == 1
    record = json.loads((tmp_path / "native/environment-session.json").read_text())
    assert ("pid_collision_relaunches" in record) is (backend in rpc.KIT_BACKENDS)


def test_robodojo_moves_the_optix_cache_off_the_system_partition(tmp_path, monkeypatch):
    for variable, _ in bootstrap.PROCESS_PATHS:
        monkeypatch.setenv(variable, "restored-after-test")
    bound = bootstrap.bind_process_paths(tmp_path)
    assert bound["OPTIX_CACHE_PATH"] == str(tmp_path / "cache/optix") == os.environ["OPTIX_CACHE_PATH"]
    assert (tmp_path / "cache/optix").is_dir()
    assert all(Path(value).is_relative_to(tmp_path) for value in bound.values())
