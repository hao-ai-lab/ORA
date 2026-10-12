"""Host storage, per-run scratch and carb shared-memory admission.

Every probe is faked or pointed into ``tmp_path``: free space, device ids, the
system paths, ``/dev/shm`` and PID liveness. No test measures a real disk,
lists the real ``/dev/shm`` or unlinks a real segment.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.environment import rpc
from ora.orchestrator import cli, host_storage, paths, preflight, run_scratch, shm_segments

pytestmark = pytest.mark.host_storage_real
GIB = 1024 ** 3


def _plan(experiment: str = "exp-storage", policy: str = "scripted", **runtime: object) -> dict:
    return {"experiment": experiment, "output_workspace": f"outputs/{experiment}",
            "runtime": {"kind": "synthetic", **runtime}, "policy": {"kind": policy}}


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    base = tmp_path.resolve()
    outputs = base / "store" / "outputs"
    outputs.mkdir(parents=True)
    system = [base / "sys", base / "sys/var/tmp", base / "sys/tmp"]
    for path in system:
        path.mkdir(parents=True, exist_ok=True)
    shm = base / "shm"
    shm.mkdir()
    free: dict[str, float] = {"/": 50.0, str(base): 100.0}
    devices: dict[str, int] = {}

    def free_bytes(path: Path) -> int:
        text = str(path)
        owner = max((key for key in free
                     if key == "/" or text == key or text.startswith(key + "/")), key=len)
        return int(free[owner] * GIB)

    monkeypatch.setattr(paths, "OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(host_storage, "_free_bytes", free_bytes)
    monkeypatch.setattr(host_storage, "_device", lambda path: devices.get(str(path), 1))
    monkeypatch.setattr(host_storage, "SYSTEM_PATHS", tuple(str(path) for path in system))
    # pytest's base directory is deeper than any real outputs root; only the
    # socket-budget test restores the real AF_UNIX limit.
    monkeypatch.setattr(host_storage, "MAX_UNIX_SOCKET_PATH_BYTES", 4096)
    monkeypatch.setattr(shm_segments, "SHM_ROOT", shm)
    monkeypatch.delenv(host_storage.EXECUTION_HOST_ENV, raising=False)
    return SimpleNamespace(base=base, outputs=outputs, system=system, shm=shm,
                           free=free, devices=devices, scratch=outputs.parent / ".runtime-tmp")


def _checks(plan: dict) -> dict[str, dict]:
    return {item["check"]: item for item in host_storage.storage_checks(plan, check=preflight._check)}


def test_capacity_checks_pass_and_report_measured_values(host):
    host.free[str(host.outputs)] = 120.0
    checks = _checks(_plan())
    workspace = checks["evidence_workspace_capacity"]
    assert workspace["passed"] is True
    assert workspace["detail"]["path"] == str(host.outputs / "exp-storage")
    assert workspace["detail"]["filesystem_probe"] == str(host.outputs)
    assert (workspace["detail"]["free_bytes"], workspace["detail"]["free_gib"]) == (120 * GIB, 120.0)
    assert workspace["detail"]["minimum_free_gib"] == 1
    assert workspace["detail"]["floor_source"] == "synthetic_default"
    tmpdir = checks["runtime_tmpdir"]
    assert tmpdir["passed"] is True
    assert tmpdir["detail"]["tmp_base"] == str(host.scratch)
    assert tmpdir["detail"]["source"] == "workspace_default"
    assert tmpdir["detail"]["environment_variables"] == ["TMPDIR", "TMP", "TEMP"]
    assert tmpdir["detail"]["rpc_socket_fallback_bytes"] == (
        len(os.fsencode(str(host.scratch))) + len("/0123456789ab") + len("/0123456789abcdef/worker.sock"))
    assert tmpdir["detail"]["scratch_runs"] == {"base": str(host.scratch), "present": False}
    assert checks["system_partition_capacity"]["detail"]["required"] is False
    assert checks["shm_carb_segments"]["passed"] is True


def test_the_floor_is_inclusive(host):
    host.free[str(host.outputs)] = 20.0
    assert _checks(_plan(kind="robotwin"))["evidence_workspace_capacity"]["passed"] is True
    host.free[str(host.outputs)] = 20.0 - 1 / GIB
    low = _checks(_plan(kind="robotwin"))
    assert low["evidence_workspace_capacity"]["passed"] is False
    assert low["runtime_tmpdir"]["passed"] is True  # same filesystem, 2 GiB floor


@pytest.mark.parametrize("runtime_kind, policy, required", [
    ("robodojo", "native_api", True),
    ("synthetic", "scripted", False), ("libero_pro", "native_api", False),
    ("robotwin", "native_api", False),
])
def test_system_partition_is_checked_only_where_a_run_writes_outside_tmpdir(host, monkeypatch, runtime_kind, policy,
                                                                           required):
    # Default floors, independent of what the tracked host profiles declare.
    _profile(monkeypatch, {})
    host.free[str(host.system[0])] = 0.5
    check = _checks({**_plan(policy=policy), "runtime": {"kind": runtime_kind}})["system_partition_capacity"]
    assert check["detail"]["required"] is required
    assert check["passed"] is (not required)


def test_system_filesystems_are_measured_once_per_device(host):
    host.devices.update({str(host.system[0]): 7, str(host.system[1]): 7, str(host.system[2]): 9})
    host.free.update({str(host.system[0]): 3.0, str(host.system[2]): 0.9})
    system = _checks({**_plan(), "runtime": {"kind": "robodojo"}})["system_partition_capacity"]
    assert system["passed"] is False
    assert [(item["paths"], item["free_gib"], item["passed"]) for item in system["detail"]["filesystems"]] == [
        ([str(host.system[0]), str(host.system[1])], 3.0, True), ([str(host.system[2])], 0.9, False)]


def _profile(monkeypatch, storage, seen=None):
    def load(host_id):
        if seen is not None:
            seen.append(host_id)
        return {"runtime_storage": storage}
    monkeypatch.setattr(host_storage, "_host_profile", load)


def test_a_zero_floor_disables_that_check_and_says_so(host, monkeypatch):
    _profile(monkeypatch, {"minimum_workspace_free_gib": 0, "minimum_system_free_gib": 0})
    host.free.update({str(host.outputs): 0.0, str(host.system[0]): 0.0})
    checks = _checks({**_plan(required_host="fixture"), "runtime": {"kind": "robodojo", "required_host": "fixture"}})
    assert checks["evidence_workspace_capacity"]["passed"] is True
    assert checks["evidence_workspace_capacity"]["detail"]["disabled"] is True
    assert checks["evidence_workspace_capacity"]["detail"]["floor_source"] == "host_profile:fixture"
    assert checks["system_partition_capacity"]["passed"] is True
    assert checks["system_partition_capacity"]["detail"]["disabled"] is True


def test_profile_follows_the_execution_host_then_the_plan(host, monkeypatch):
    seen: list[str] = []
    _profile(monkeypatch, {}, seen)
    monkeypatch.setenv(host_storage.EXECUTION_HOST_ENV, "test-arm")
    assert _checks(_plan(required_host="test-host"))["runtime_tmpdir"]["detail"]["host_source"] == "execution_marker"
    monkeypatch.delenv(host_storage.EXECUTION_HOST_ENV)
    assert _checks(_plan(required_host="test-host"))["runtime_tmpdir"]["detail"]["host_source"] == "required_host"
    # No host is implied for any runtime kind.
    robotwin = _checks({**_plan(), "runtime": {"kind": "robotwin"}})["runtime_tmpdir"]["detail"]
    assert robotwin["host_source"] == "none"
    assert _checks(_plan())["runtime_tmpdir"]["detail"]["host_source"] == "none"
    assert seen == ["test-arm", "test-host"]


def test_the_tracked_profile_is_read_through_the_public_resources_loader():
    from ora.orchestrator import resources

    assert resources.load_host_profile("test-host")["id"] == "test-host"
    assert resources.load_host_profile("no-such-host") is None
    with pytest.raises(resources.ExecutionResourceError):
        resources.load_host_profile("../test-host")
    profile = host_storage._host_profile("test-host")
    # The fixture host disables only the system-partition floor.
    assert profile["id"] == "test-host" and dict(profile["runtime_storage"]) == {"minimum_system_free_gib": 0}
    assert host_storage._host_profile("no-such-host") is None
    with pytest.raises(host_storage.StorageConfigError):
        host_storage._host_profile("../test-host")


@pytest.mark.parametrize("storage", [
    {"tmp_root": "relative/tmp"}, {"tmp_root": "/"}, {"minimum_workspace_free_gib": -1},
    {"minimum_tmp_free_gib": True}, {"unexpected": 1}, ["not", "a", "mapping"],
])
def test_malformed_host_storage_configuration_fails_closed(host, monkeypatch, storage):
    _profile(monkeypatch, storage)
    checks = _checks(_plan(required_host="fixture"))
    assert checks["host_storage_configuration"]["passed"] is False
    assert set(checks) == {"host_storage_configuration", "shm_carb_segments"}


def test_scratch_base_must_leave_room_for_the_rpc_socket(host, monkeypatch):
    monkeypatch.setattr(host_storage, "MAX_UNIX_SOCKET_PATH_BYTES", rpc.MAX_UNIX_SOCKET_PATH_BYTES)
    monkeypatch.setattr(host_storage, "_writable", lambda path: True)
    fits = "/" + "f" * (rpc.MAX_UNIX_SOCKET_PATH_BYTES - run_scratch.RPC_SOCKET_SUFFIX_BYTES - 1)
    _profile(monkeypatch, {"tmp_root": fits})
    admitted = _checks(_plan(kind="robotwin", required_host="fixture"))["runtime_tmpdir"]
    assert admitted["detail"]["rpc_socket_fallback_bytes"] == rpc.MAX_UNIX_SOCKET_PATH_BYTES
    assert admitted["passed"] is True
    _profile(monkeypatch, {"tmp_root": fits + "g"})
    refused = _checks(_plan(kind="robotwin", required_host="fixture"))["runtime_tmpdir"]
    assert refused["passed"] is False
    assert refused["detail"]["rpc_socket_fallback_bytes"] == rpc.MAX_UNIX_SOCKET_PATH_BYTES + 1


def test_an_unwritable_scratch_filesystem_fails(host, monkeypatch):
    monkeypatch.setattr(host_storage, "_writable", lambda path: False)
    tmpdir = _checks(_plan())["runtime_tmpdir"]
    assert tmpdir["passed"] is False and tmpdir["detail"]["writable"] is False


def test_a_dangling_workspace_link_fails_instead_of_measuring_root(host):
    (host.outputs / "exp-storage").symlink_to(host.base / "unmounted/exp-storage")
    workspace = _checks(_plan())["evidence_workspace_capacity"]
    assert workspace["passed"] is False
    assert "dangling symlink" in workspace["detail"]["error"]


def test_a_plan_without_a_workspace_fails_only_that_check(host):
    checks = _checks({"runtime": {"kind": "synthetic"}})
    assert checks["evidence_workspace_capacity"]["passed"] is False
    assert checks["runtime_tmpdir"]["passed"] is True


def _segments(root: Path, names) -> None:
    for name in names:
        (root / name).write_bytes(b"\0" * 16)


def test_carb_name_pattern_is_exact():
    matched = ["carb-RStringInternals-12", "sem.carb-RStringInternals-12", "carb-RStringInternals-4194304"]
    unmatched = ["sem.carbonite-sharedmemory", "carb-other-12", "carb-RStringInternals-2147483648",
                 "carb-RStringInternals-", "torch_1_2_3", "xcarb-RStringInternals-12",
                 "carb-RStringInternals-\u0661\u0662"]  # Arabic-Indic digits: int() accepts them
    assert all(shm_segments.CARB_SEGMENT.match(name) for name in matched)
    assert not any(shm_segments.CARB_SEGMENT.match(name) for name in unmatched)


@pytest.mark.parametrize("owner", ["own", "foreign"])
def test_an_out_of_range_pid_never_reaches_os_kill_or_escapes(host, monkeypatch, owner):
    uid = 1000
    names = ["carb-RStringInternals-9999999", "carb-RStringInternals-2147483648",
             "sem.carb-RStringInternals-99999999999"]
    _segments(host.shm, names)
    monkeypatch.setattr(shm_segments, "segment_owner", lambda path: uid if owner == "own" else uid + 1)
    monkeypatch.setattr(shm_segments, "pid_max", lambda: 4194304)
    counted = shm_segments.census(host.shm, uid=uid)
    swept = shm_segments.sweep_own_stale(host.shm, uid=uid)
    assert counted["errors"] == [] and swept["removed"] == 0
    assert counted[f"{owner}_live"] == 1 and counted[f"{owner}_stale"] == 0
    assert sorted(path.name for path in host.shm.iterdir()) == sorted(names)


def test_pid_liveness_catches_overflow_and_only_missing_processes_are_dead(monkeypatch):
    assert shm_segments.pid_alive(2 ** 31, limit=2 ** 40) is True  # real os.kill raises OverflowError
    assert shm_segments.pid_alive(os.getpid(), limit=4194304) is True
    assert shm_segments.pid_alive(0, limit=4194304) is True
    assert shm_segments.pid_alive(4194305, limit=4194304) is True

    def kill(pid, signal):
        raise ProcessLookupError if pid == 11 else PermissionError

    monkeypatch.setattr(shm_segments.os, "kill", kill)
    assert shm_segments.pid_alive(11, limit=4194304) is False
    assert shm_segments.pid_alive(12, limit=4194304) is True


def test_census_only_counts_and_the_sweep_removes_only_own_dead_segments(host, monkeypatch):
    me, other = 1000, 2000
    owners = {
        "carb-RStringInternals-101": me, "sem.carb-RStringInternals-101": me,
        "carb-RStringInternals-102": me, "carb-RStringInternals-201": other,
        "sem.carb-RStringInternals-201": other, "carb-RStringInternals-202": other,
        "torch_123_456_789": me, "sem.carbonite-sharedmemory": me,
    }
    _segments(host.shm, owners)
    monkeypatch.setattr(shm_segments, "segment_owner", lambda path: owners[path.name])
    monkeypatch.setattr(shm_segments, "pid_alive", lambda pid, limit: pid in {102, 202})
    counted = shm_segments.census(host.shm, uid=me)
    assert (counted["own_stale"], counted["own_live"], counted["foreign_stale"],
            counted["foreign_stale_pids"], counted["foreign_live"]) == (2, 1, 2, 1, 1)
    assert len(list(host.shm.iterdir())) == len(owners)  # preflight changes nothing
    swept = shm_segments.sweep_own_stale(host.shm, uid=me)
    assert swept["removed_names"] == ["carb-RStringInternals-101", "sem.carb-RStringInternals-101"]
    assert sorted(path.name for path in host.shm.iterdir()) == sorted(
        name for name in owners if not name.endswith("-101"))


def test_real_ownership_is_read_with_lstat_and_non_files_are_never_removed(host, monkeypatch):
    _segments(host.shm, ["carb-RStringInternals-31"])
    (host.shm / "carb-RStringInternals-32").mkdir()
    monkeypatch.setattr(shm_segments, "pid_alive", lambda pid, limit: False)
    assert shm_segments.segment_owner(host.shm / "carb-RStringInternals-31") == os.getuid()
    assert shm_segments.segment_owner(host.shm / "carb-RStringInternals-32") is None
    swept = shm_segments.sweep_own_stale(host.shm, uid=os.getuid())
    assert swept["removed_names"] == ["carb-RStringInternals-31"]
    assert (host.shm / "carb-RStringInternals-32").is_dir()


def test_sweep_reports_an_absent_root_and_unlink_failures(host, monkeypatch):
    assert shm_segments.census(host.base / "absent", uid=os.getuid())["present"] is False
    _segments(host.shm, ["carb-RStringInternals-9"])
    monkeypatch.setattr(shm_segments, "pid_alive", lambda pid, limit: False)
    monkeypatch.setattr(shm_segments, "_unlink", lambda path: f"{path.name}: sticky")
    refused = shm_segments.sweep_own_stale(host.shm, uid=os.getuid())
    assert refused["removed"] == 0 and refused["errors"] == ["carb-RStringInternals-9: sticky"]


def test_scratch_lives_outside_every_scanned_evidence_tree(host):
    staged = run_scratch.stage(host.scratch, "run-a", source="workspace_default", host_id=None)
    directory = Path(staged["tmp_root"])
    runs = directory / "runs/run-a"
    runs.mkdir(parents=True)
    (runs / "plan.json").write_text("{}")
    assert not directory.is_relative_to(host.outputs)
    assert list(paths._output_run_candidates(host.outputs)) == []


def test_scratch_owner_binding_release_and_residual_census(host, monkeypatch):
    live = run_scratch.stage(host.scratch, "run-live", source="s", host_id=None)
    dead = run_scratch.stage(host.scratch, "run-dead", source="s", host_id=None)
    (host.scratch / "stranger").mkdir()
    assert run_scratch.bind_owner(live, 4242) is True
    monkeypatch.setattr(run_scratch, "pid_alive", lambda pid, limit: pid == 4242)
    assert run_scratch.census(host.scratch) == {
        "base": str(host.scratch), "present": True, "live_runs": 1,
        "residual_dead_owner": 1, "unrecognized": 1,
    }
    run_root = host.base / "run-root"
    run_root.mkdir()
    (run_root / "launch.json").write_text(json.dumps({"run_id": "run-other", "runtime_scratch": dead}))
    assert run_scratch.release(run_root) is False and Path(dead["tmp_root"]).is_dir()
    (run_root / "launch.json").write_text(json.dumps({"run_id": "run-dead", "runtime_scratch": dead}))
    assert host_storage.release_run_scratch(run_root) is True
    assert not Path(dead["tmp_root"]).exists() and Path(live["tmp_root"]).is_dir()


def test_worker_environment_carries_the_supervisor_scratch(monkeypatch, tmp_path):
    for name in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(tmp_path / "scratch"))
    environment = rpc._worker_environment(tmp_path)
    assert {name: environment[name] for name in ("TMPDIR", "TMP", "TEMP")} == {
        name: str(tmp_path / "scratch") for name in ("TMPDIR", "TMP", "TEMP")}


def test_run_preflight_includes_the_storage_checks_and_is_blocked_by_them(host, monkeypatch):
    monkeypatch.setattr(preflight, "_git", lambda _root, *args: subprocess.CompletedProcess(
        args, 0, stdout="a" * 40 if args == ("rev-parse", "HEAD") else "", stderr=""))
    monkeypatch.setattr(preflight, "_qualification", lambda _plan: {"check": "q", "passed": True})
    monkeypatch.setattr(preflight, "_runtime", lambda _plan: [])
    for name in ("_resident_service_prerequisite", "_provider_marker"):
        monkeypatch.setattr(preflight, name, lambda *_args: {"check": "stub", "passed": True})
    plan = {**_plan(), "runtime": {"kind": "robodojo"}, "source": {"commit": "a" * 40},
            "plan_id": "storage", "plan_sha256": "b" * 64}
    names = [item["check"] for item in preflight.run_preflight(plan)["checks"]]
    assert names[1:5] == ["evidence_workspace_capacity", "runtime_tmpdir",
                          "system_partition_capacity", "shm_carb_segments"]
    host.free[str(host.system[0])] = 0.2
    assert preflight.run_preflight(plan)["passed"] is False


def _spawn_fixture(host, monkeypatch, tmp_path, *, lease=None, admission=None):
    plan = {**_plan(), "plan_id": "storage-plan", "plan_sha256": "c" * 64, "source": {"commit": "test-source"}}
    report = {"schema": "ora.preflight.v1", "passed": True,
              "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"], "checks": []}
    monkeypatch.setattr(cli, "validate_run_plan", lambda *_: None)
    monkeypatch.setattr(cli, "_require_launchable", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, stdout="test-source" if "rev-parse" in command else ""))
    monkeypatch.setattr(cli, "launch_environment", lambda *_args: {"PATH": "/usr/bin", "TMPDIR": "/tmp"})
    released: list[str] = []
    monkeypatch.setattr(cli, "_create_lease", lease or (lambda *_args: {"claim_token": "claim"}))
    monkeypatch.setattr(cli, "_release_lease", lambda run_id, **kwargs: released.append(run_id))
    root = tmp_path / "run"

    def not_found(*_args):
        raise RuntimeError("run-id not found")

    for name, value in (("new_run_root", lambda *args: root), ("resolve_run_root", not_found),
                        ("workspace_topology", lambda: {}), ("register_run", lambda *args: None)):
        monkeypatch.setattr(cli.paths, name, value)
    for name, value in (("_snapshot_launch_gate", lambda *args: None),
                        ("_claim_launch_admission", admission or (lambda *args: None)),
                        ("_plan_resource_binding", lambda *args: {}), ("_assign_lease", lambda lease, pid: lease)):
        monkeypatch.setattr(cli, name, value)
    observed: dict = {}
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_a, **kwargs: observed.update(kwargs) or SimpleNamespace(pid=4242))
    return SimpleNamespace(plan=plan, report=report, root=root, observed=observed, released=released)


def test_spawn_sweeps_and_stages_before_the_lease_and_records_both(host, monkeypatch, tmp_path):
    _segments(host.shm, ["carb-RStringInternals-77"])
    monkeypatch.setattr(shm_segments, "pid_alive", lambda pid, limit: False)
    order: list[str] = []

    def lease(*_args):
        order.append("lease" if not (host.shm / "carb-RStringInternals-77").exists() else "lease-too-early")
        return {"claim_token": "claim"}

    launch = _spawn_fixture(host, monkeypatch, tmp_path, lease=lease)
    run_id = cli._spawn(launch.plan, launch.report, yes=True)
    scratch = host.scratch / run_scratch.run_key(run_id)
    assert order == ["lease"] and scratch.is_dir()
    assert {name: launch.observed["env"][name] for name in ("TMPDIR", "TMP", "TEMP")} == {
        name: str(scratch) for name in ("TMPDIR", "TMP", "TEMP")}
    record = json.loads((launch.root / "launch.json").read_text(encoding="utf-8"))
    assert record["runtime_scratch"]["tmp_root"] == str(scratch)
    assert record["shm_sweep"]["removed_names"] == ["carb-RStringInternals-77"]
    owner = json.loads((scratch / run_scratch.OWNER_FILE).read_text())
    assert (owner["run_id"], owner["supervisor_pid"]) == (run_id, 4242)


@pytest.mark.parametrize("failing", ["lease", "admission"])
def test_a_failed_launch_removes_its_scratch_but_not_the_sweep(host, monkeypatch, tmp_path, failing):
    _segments(host.shm, ["carb-RStringInternals-78"])
    monkeypatch.setattr(shm_segments, "pid_alive", lambda pid, limit: False)

    def refuse(*_args):
        raise RuntimeError(f"{failing} refused")

    launch = _spawn_fixture(host, monkeypatch, tmp_path, **{failing: refuse})
    with pytest.raises(RuntimeError, match=f"{failing} refused"):
        cli._spawn(launch.plan, launch.report, yes=True)
    assert list(host.scratch.iterdir()) == []
    assert not (host.shm / "carb-RStringInternals-78").exists()
    assert len(launch.released) == (1 if failing == "admission" else 0)
    assert launch.observed == {}


def test_discard_removes_only_a_directory_its_owner_file_names(host):
    staged = run_scratch.stage(host.scratch, "run-z", source="s", host_id=None)
    foreign = host.scratch / "not-a-run"
    foreign.mkdir()
    assert run_scratch.discard({"tmp_root": str(foreign)}) is False and foreign.is_dir()
    assert run_scratch.discard(staged) is True and not Path(staged["tmp_root"]).exists()


def test_synthetic_does_not_need_a_unix_rpc_socket(host, monkeypatch):
    monkeypatch.setattr(host_storage, "MAX_UNIX_SOCKET_PATH_BYTES", 1)
    synthetic = _checks(_plan())["runtime_tmpdir"]
    assert synthetic["passed"] is True
    assert synthetic["detail"]["rpc_socket_required"] is False
    isolated = _checks(_plan(kind="robotwin"))["runtime_tmpdir"]
    assert isolated["passed"] is False


def test_synthetic_storage_floors_are_small_but_explicit_host_settings_win(host, monkeypatch):
    host.free[str(host.outputs)] = 1.0
    checks = _checks(_plan())
    assert checks["evidence_workspace_capacity"]["passed"] is True
    assert checks["runtime_tmpdir"]["detail"]["minimum_free_gib"] == 0.25
    host.free[str(host.outputs)] = 1.0 - 1 / GIB
    assert _checks(_plan())["evidence_workspace_capacity"]["passed"] is False
    host.free[str(host.outputs)] = 12.66
    _profile(monkeypatch, {"minimum_workspace_free_gib": 20})
    explicit = _checks(_plan(required_host="fixture"))["evidence_workspace_capacity"]
    assert explicit["passed"] is False
    assert explicit["detail"]["floor_source"] == "host_profile:fixture"
