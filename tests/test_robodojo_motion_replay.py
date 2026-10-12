"""U1 v2 L0: the RoboDojo control-replay worker and its host-side report.

The worker is driven with a fake session whose "physics" sets each commanded
arm to its recorded target and which, like ``RobodojoSession``, publishes the
gripper command from its own directives rather than from physics. An exact
recording therefore replays with zero divergence only when the worker restores
the recorded directives. Nothing here starts Isaac or claims a real lease.
"""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import control_replay as worker
from ora.orchestrator import robodojo_motion_replay as launcher
from ora.orchestrator import robodojo_replay_kernel as replay_kernel
from ora.orchestrator.robodojo_motion_journal import load_run_journal
from tests.robodojo_journal_fixture import (
    FINGERPRINT, REVISIONS, ZERO, action, control, receipt, state, still, tick, write_native,
)

RIGHT_EE = {"right_ee_joint_state": {"position": [0.0, 0.0]}}
A = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
B = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]


class FakeEnvironment:
    def __init__(self) -> None:
        self.take_action_cnt = [0]
        self.controller_trace_callback = None
        self.robot_manager = SimpleNamespace(control_manager=SimpleNamespace(prev_control=[{}]))


class FakeSession:
    """Commanded arms reach their target on every physics step; the gripper command comes from directives."""

    def __init__(self, *, terminal_after: dict[int, int] | None = None, revisions: dict | None = None,
                 fingerprint: dict | None = None, applied_offset: float = 0.0) -> None:
        self.environment = FakeEnvironment()
        self.clock, self.sim_steps, self.round_index, self.round_ticks = 1000, 0, 0, 0
        self.joints = {"left": list(ZERO), "right": list(ZERO)}
        self.gripper_directives = {"left": 1.0, "right": 1.0}
        self.restarts: list[tuple[int, int | None]] = []
        self.terminal_after = dict(terminal_after or {})
        self.source = {"source_revisions": revisions or REVISIONS}
        self.fingerprint = fingerprint or FINGERPRINT
        self.applied_offset = applied_offset
        self.official_terminal = None
        self.closed = False

    def start(self) -> dict:
        return {"actual_source_fingerprint": self.fingerprint,
                "source_binding": {"source_revisions": self.source["source_revisions"]}}

    def _clock(self) -> int:
        return self.clock

    def _proprioception(self) -> dict:
        targets = {arm: "close" if value < 0.5 else "open" for arm, value in self.gripper_directives.items()}
        return {"end_effectors": {arm: state(q, arm=arm, target=targets[arm]) for arm, q in self.joints.items()}}

    def _applied_position_targets(self) -> dict:
        return {f"{arm}_arm_joint_state": [v + self.applied_offset for v in q] for arm, q in self.joints.items()}

    @property
    def is_terminal(self) -> bool:
        return self.round_ticks >= self.terminal_after.get(self.round_index, 10**9)

    def _conformance_tick(self, control_dict: dict) -> None:
        self.environment.take_action_cnt[0] += 1
        for _ in range(10):
            self.clock += 1
            for key, value in control_dict.items():
                if key.endswith("_arm_joint_state"):
                    self.joints[key.split("_")[0]] = list(value["position"])
            callback = self.environment.controller_trace_callback
            if callback is not None:
                callback({key: {"position": list(value["position"])} for key, value in control_dict.items()})
        self.sim_steps += 10
        self.round_ticks += 1

    def restart_round(self, index: int, *, seed: int | None = None) -> dict:
        self.restarts.append((index, seed))
        self.round_index, self.round_ticks = index, 0
        self.environment.take_action_cnt[0] = 0
        self.clock += 500
        self.joints = {"left": list(ZERO), "right": list(ZERO)}
        self.gripper_directives = {"left": 1.0, "right": 1.0}
        return {"round_index": index}

    def close(self) -> None:
        self.closed = True


def _exact_run(root: Path) -> Path:
    at_a, at_b = still("right", A), still("right", B)
    closed_at_a = still("right", A, target="close")
    close = receipt("set_gripper", arguments={"targets": {"right": "close"}}, gripper_floors={"right": 0.05},
                    arms={"right": {"gripper_outcome": {"state": "reached_target"}}})
    round0 = [
        action(receipt("move_pose"), [tick("move", control(right=A), right=at_a),
                                      tick("move", control(right=A), right=at_a)]),
        action(receipt("move_pose", status="planner_failed", code="ik_failed:move")),
        action(close, [tick("gripper", RIGHT_EE, right=closed_at_a)]),
        action(receipt("move_pose"), [tick("move", control(right=A), right=closed_at_a)]),
    ]
    round1 = [action(receipt("move_pose"), [tick("move", control(right=B), right=at_b)])]
    return write_native(root, [round0, round1])


def _replay(run: Path, output: Path, session: FakeSession, **kwargs) -> dict:
    config = {"output_root": str(output / "environment")}
    return worker.run_worker(config, run / "native", output, session_factory=lambda **_: session, **kwargs)


def _report(run: Path, output: Path, mode: str) -> tuple[int, dict]:
    code = launcher.main(["--run", str(run), "--output", str(output), "--mode", mode, "--report-only"])
    return code, json.loads((output / launcher.REPORT_NAME).read_text())


def test_worker_arguments_require_explicit_paths_and_a_positive_upto():
    parsed = worker.parse_arguments(["--config", "c.json", "--native", "n", "--output", "o", "--upto", "3"])
    assert (parsed.config, parsed.native, parsed.output, parsed.upto) == (Path("c.json"), Path("n"), Path("o"), 3)
    with pytest.raises(SystemExit):
        worker.parse_arguments(["--config", "c.json", "--native", "n", "--output", "o", "--upto", "0"])
    with pytest.raises(SystemExit):
        worker.parse_arguments(["--native", "n", "--output", "o"])


def test_recorded_actions_read_every_controller_trace_in_sequence(tmp_path):
    native = _exact_run(tmp_path / "run") / "native"
    actions = worker.recorded_actions(native)
    assert [(item["sequence"], item["round_index"], len(item["samples"])) for item in actions] == [
        (0, 0, 2), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 1, 1)]
    assert [item["sequence"] for item in worker.recorded_actions(native, upto=2)] == [0, 1]
    (native / "controller" / "action-0001.json").unlink()
    with pytest.raises(worker.ReplayError, match="contiguous"):
        worker.recorded_actions(native)


def test_recorded_actions_refuse_a_trace_bound_to_another_action(tmp_path):
    native = _exact_run(tmp_path / "run") / "native"
    path = native / "controller" / "action-0002.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "action_id": "action-9999"}), encoding="utf-8")
    with pytest.raises(worker.ReplayError, match="action_id"):
        worker.recorded_actions(native)


def test_exact_control_replay_has_zero_divergence_and_writes_loadable_journals(tmp_path):
    run = _exact_run(tmp_path / "run")
    session = FakeSession()
    report = _replay(run, tmp_path / "out", session, read_objects=lambda: {"block": {"position_m": [0, 0, 1]}})
    assert report["schema"] == worker.WORKER_REPORT_SCHEMA
    assert report["error"] is None and report["completed"] is True
    replay = report["replay"]
    assert replay["ticks_replayed"] == replay["ticks_recorded"] == 5
    assert replay["max_divergence"] == {"joint_rad": 0.0, "finger_m": 0.0, "gripper": 0.0}
    assert replay["first_divergence"] is None and replay["applied_target_mismatches"] == 0
    assert replay["skipped_zero_tick_actions"] == [{"action_sequence": 1, "execution_status": "planner_failed"}]
    assert session.restarts == [(1, 0)] and session.closed
    environment = tmp_path / "out" / "environment"
    assert [len(item.ticks) for item in load_run_journal(run, journal_root=environment).actions] == [2, 0, 1, 1, 1]
    lines = (environment / "controller" / "action-0000.jsonl").read_text().splitlines()
    assert len(lines) == 20 and set(json.loads(lines[0])) == {
        "action_id", "segment", "physics_clock", "native_action", "command",
        "applied_position_targets", "measured"}
    rows = [json.loads(line) for line in (tmp_path / "out" / "private" / "object-poses.jsonl").read_text().splitlines()]
    assert [(row["action_sequence"], row["tick_index"]) for row in rows] == [(0, 0), (0, 1), (2, 0), (3, 0), (4, 0)]
    assert report["object_poses"]["position_frame"] == worker.OBJECT_POSITION_FRAME
    assert json.loads((tmp_path / "out" / "replay-worker.json").read_text())["replay"]["ticks_replayed"] == 5


def test_replay_restores_the_recorded_gripper_directives_before_each_tick(tmp_path):
    run = _exact_run(tmp_path / "run")
    _replay(run, tmp_path / "out" / "pass-a", FakeSession())
    trace = json.loads((tmp_path / "out" / "pass-a" / "environment" / "controller" / "action-0003.json").read_text())
    assert trace["samples"][0]["measured"]["right"]["gripper_command_target"] == "close"
    code, report = _report(run, tmp_path / "out", "metrics-after-replay")
    assert code == 0 and report["metrics"]["differences"]["pass-a"] == []
    closed = report["metrics"]["replay"]["pass-a"]["v1_baseline_diagnostics"]["saturation_physics_step"]
    assert closed["closed_command"]["denominator"] == 1
    session = FakeSession()
    worker.restore_gripper_directives(session, {"measured": {"right": state(target="close")}})
    assert session.gripper_directives == {"left": 1.0, "right": 0.0}


def test_replay_reports_the_first_divergent_tick_on_joints_and_fingers(tmp_path):
    run = _exact_run(tmp_path / "run")
    path = run / "native" / "controller" / "action-0000.json"
    trace = json.loads(path.read_text())
    trace["samples"][1]["measured"]["right"]["gripper_finger_qpos"][0] = 0.04
    path.write_text(json.dumps(trace), encoding="utf-8")
    replay = _replay(run, tmp_path / "out", FakeSession())["replay"]
    assert replay["max_divergence"]["finger_m"] == pytest.approx(0.004)
    assert (replay["first_divergence"]["action_sequence"], replay["first_divergence"]["tick_index"]) == (0, 1)
    assert replay["max_joint_divergence_rad"] == 0.0


def test_an_early_terminal_stops_only_its_round_and_is_recorded(tmp_path):
    report = _replay(_exact_run(tmp_path / "run"), tmp_path / "out", FakeSession(terminal_after={0: 1}))
    rounds = {item["round_index"]: item for item in report["replay"]["rounds"]}
    assert rounds[0]["terminal_before_recorded_end"] is True and rounds[0]["ticks_unreplayed"] == 3
    assert rounds[1]["ticks_replayed"] == 1
    assert report["completed"] is False


def test_upto_and_object_reader_errors_are_recorded_not_raised(tmp_path):
    def broken() -> dict:
        raise RuntimeError("no layout")

    report = _replay(_exact_run(tmp_path / "run"), tmp_path / "out", FakeSession(), upto=1, read_objects=broken)
    assert report["replay"]["ticks_replayed"] == 2
    assert report["object_poses"]["errors"] == 2 and "no layout" in report["object_poses"]["first_error"]
    assert report["object_poses"]["frame_probe"]["status"] == "error"


@pytest.mark.parametrize("session, message", [
    (FakeSession(revisions={"robodojo": "x" * 40}), "source revisions"),
    (FakeSession(fingerprint={**FINGERPRINT, "installed_packages": {"isaaclab": "2"}}), "actual_source_fingerprint"),
])
def test_worker_refuses_a_plant_from_other_sources(tmp_path, session, message):
    report = _replay(_exact_run(tmp_path / "run"), tmp_path / "out", session)
    assert message in report["error"]
    assert report["replay"] is None and session.closed


def _layout_session():
    class Layout:
        instance_type_by_env = [{"block_1": "rigid", "stand_0": "geometry"}]

        def get_instance_name(self, env_idx, label):
            return label

        def get_instance_pose(self, env_idx, label=None, inst_name=None, relative=True):
            assert (env_idx, inst_name) == (0, "block_1")
            offset = 0.0 if relative else 2.0
            return np.array([0.1 + offset, 0.2, 0.8]), np.array([1.0, 0.0, 0.0, 0.0])

    session = type("S", (), {})()
    session.environment = type("E", (), {})()
    session.environment.scene_manager = type("M", (), {"layout_manager": Layout()})()
    return session


def test_object_poses_and_the_frame_probe_read_rigid_instances_through_the_evaluator_adapter():
    session = _layout_session()
    assert worker.object_pose_reader(session)() == {
        "block_1": {"position_m": [0.1, 0.2, 0.8], "orientation": [1.0, 0.0, 0.0, 0.0]}}
    probe = worker.object_frame_probe(session)
    assert probe["status"] == "read" and probe["layout_env_origin_m"] == pytest.approx([2.0, 0.0, 0.0])


def test_the_replayed_session_still_offers_the_private_methods_the_worker_drives():
    from ora.environment.backends.robodojo.session import RobodojoSession

    for name in ("start", "close", "restart_round", "_conformance_tick", "_applied_position_targets",
                 "_proprioception", "_clock"):
        assert callable(getattr(RobodojoSession, name)), name
    assert isinstance(inspect.getattr_static(RobodojoSession, "is_terminal"), property)
    assert list(inspect.signature(RobodojoSession._conformance_tick).parameters) == ["self", "control"]
    assert "self.gripper_directives" in inspect.getsource(RobodojoSession.__init__)
    assert launcher.WORKER_MODULE == worker.__name__


def test_launcher_arguments_and_output_containment(tmp_path):
    run = _exact_run(tmp_path / "run")
    parsed = launcher.parse_arguments(["--run", str(run), "--output", str(tmp_path / "out"),
                                       "--resource-profile", "robodojo-gpu4-d1", "--mode", "determinism",
                                       "--second-resource-profile", "robodojo-gpu4-d2", "--upto", "2"])
    assert (parsed.mode, parsed.upto, parsed.second_resource_profile) == ("determinism", 2, "robodojo-gpu4-d2")
    for argv in (["--resource-profile", "p", "--second-resource-profile", "q"], []):
        with pytest.raises(SystemExit):
            launcher.parse_arguments(["--run", str(run), "--output", str(tmp_path / "o"), *argv])
    with pytest.raises(launcher.ReplayLaunchError, match="inside the source run"):
        launcher.validate_output(run, run / "native" / "replay", report_only=False)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(launcher.ReplayLaunchError, match="not empty"):
        launcher.validate_output(run, tmp_path / "busy", report_only=False)
    assert launcher.validate_output(run, tmp_path / "busy", report_only=True) == (tmp_path / "busy").resolve()


def test_worker_command_and_replay_runtime_are_explicit():
    command = launcher.worker_command(Path("/py"), config=Path("/o/pass-a/worker-config.json"),
                                      native_root=Path("/r/native"), pass_root=Path("/o/pass-a"), upto=5)
    assert command == ["/py", "-m", worker.__name__, "--config", "/o/pass-a/worker-config.json",
                       "--native", "/r/native", "--output", "/o/pass-a", "--upto", "5"]
    runtime = {"kind": "robodojo", "cuda_visible_devices": "4", "renderer_gpu": 4,
               "infrastructure_qualification": {"path": "x"}, "execution_capture": {"enabled": True}}
    binding = {"profile_id": "robodojo-gpu6-d1", "worker_environment": {"CUDA_VISIBLE_DEVICES": "6"}}
    replayed = launcher.replay_runtime(runtime, binding)
    assert replayed["cuda_visible_devices"] == "6" and replayed["renderer_gpu"] == 6
    assert replayed["resource_profile"] == "robodojo-gpu6-d1" and replayed["resource_binding"] == binding
    assert "infrastructure_qualification" not in replayed and replayed["execution_capture"]["enabled"] is False
    assert runtime["cuda_visible_devices"] == "4"


def test_harness_identity_compares_kernels_and_tolerates_a_copied_tree(tmp_path, monkeypatch):
    from ora.orchestrator.paths import PROJECT_ROOT

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    same = launcher.harness_identity(head)
    assert same["worker_kernel_matches_run"] is True
    assert same["kernel_admission"]["status"] == "identical_kernel"
    assert same["worker_kernel_sha256"] == same["run_worker_kernel_sha256"]
    assert launcher.harness_identity("f" * 40)["worker_kernel_matches_run"] is None
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    copied = launcher.harness_identity(head)
    assert copied["commit"] is None and copied["worker_kernel_matches_run"] is None
    assert copied["kernel_admission"] is None


def test_launch_is_refused_without_the_host_marker_or_with_another_worker_kernel(tmp_path, monkeypatch):
    run = _exact_run(tmp_path / "run")
    (run / "plan.json").write_text(json.dumps({"runtime": {"required_host": "test-host"}, "source": {"commit": "a" * 40}}))
    monkeypatch.delenv("ORA_EXECUTION_HOST", raising=False)
    with pytest.raises(launcher.ReplayLaunchError, match="host marker"):
        launcher.launch_identity(run)
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(launcher, "harness_identity", lambda commit: {
        "commit": "b" * 40, "tracked_worktree_dirty": False,
        "kernel_admission": {"status": "control_path_changed"}})
    with pytest.raises(launcher.ReplayLaunchError, match="not admitted"):
        launcher.launch_identity(run)


def _admission(monkeypatch, files: dict[str, dict[str, bytes]], run: str, commit: str) -> dict:
    from ora.orchestrator import robodojo_kernel

    paths = tuple(sorted(files))
    monkeypatch.setattr(robodojo_kernel, "worker_kernel_paths", lambda commit, project_root: paths)
    monkeypatch.setattr(replay_kernel, "_source_at", lambda at, path: files[path].get(at))
    return replay_kernel.kernel_admission(run, commit)


SPARSE = "ora/environment/backends/robodojo/sparse_motion.py"
COMMON = "ora/environment/backends/robodojo/common.py"
BASE_SPARSE = (b'"""doc"""\nFREE_SPACE_CLEARANCE_M = 0.05\nLOCAL_JOINT_JUMP_LIMIT_RAD = 1.0\n'
               b"GLOBAL_JOINT_JUMP_LIMIT_RAD = 3.14\nSTAGED_CONTACT_STRATEGY = 'staged'\n"
               b"def gripper_closed_floor(session, arm):\n    return 0.0\n")
BASE_COMMON = b"import math\nX = 1\ndef f():\n    return X\n__all__ = ['X', 'f']\n"


@pytest.mark.parametrize("sparse, common, status, refused", [
    (BASE_SPARSE, BASE_COMMON, "identical_kernel", {}),
    (BASE_SPARSE.replace(b'"""doc"""', b'"""new doc"""') + b"def _run_global():\n    return 1\n",
     BASE_COMMON.replace(b"__all__ = ['X', 'f']", b"Y = 2\n__all__ = ['X', 'f', 'Y']"),
     "executor_drift_admitted", {}),
    (BASE_SPARSE.replace(b"0.05", b"0.06"), BASE_COMMON, "control_path_changed",
     {SPARSE: ["FREE_SPACE_CLEARANCE_M"]}),
    (BASE_SPARSE, BASE_COMMON.replace(b"return X", b"return X + 1"), "control_path_changed", {COMMON: ["f"]}),
    (BASE_SPARSE, BASE_COMMON.replace(b"'X', 'f'", b"'X'"), "control_path_changed", {COMMON: ["__all__"]}),
    (BASE_SPARSE, BASE_COMMON + b"X = 3\n", "control_path_changed", {COMMON: ["X"]}),
    (BASE_SPARSE, BASE_COMMON + b"print(X)\n", "control_path_changed", {COMMON: ["<module-level statements>"]}),
    (BASE_SPARSE + b"if X:\n    pass\n", BASE_COMMON, "control_path_changed",
     {SPARSE: ["<module-level statements>"]}),
])
def test_kernel_admission_admits_executor_drift_and_add_only_changes(monkeypatch, sparse, common, status, refused):
    files = {SPARSE: {"run": BASE_SPARSE, "head": sparse}, COMMON: {"run": BASE_COMMON, "head": common}}
    admission = _admission(monkeypatch, files, "run", "head")
    assert admission["status"] == status and admission["refused_changes"] == refused


def test_a_reached_name_bound_at_neither_commit_refuses(monkeypatch):
    missing = BASE_SPARSE.replace(b"FREE_SPACE_CLEARANCE_M = 0.05\n", b"")
    files = {SPARSE: {"run": missing, "head": missing + b"Z = 1\n"}}
    admission = _admission(monkeypatch, files, "run", "head")
    assert admission["status"] == "control_path_changed"
    assert admission["refused_changes"][SPARSE] == ["FREE_SPACE_CLEARANCE_M (bound at neither commit)"]
    assert _admission(monkeypatch, files, None, "head")["status"] == "unresolvable"


def test_the_executor_only_table_is_pinned_and_covers_every_replay_path_reference():
    project = Path(launcher.PROJECT_ROOT)
    assert {path: reached for path, (reached, _reason) in replay_kernel.CONTROL_REPLAY_EXECUTOR_ONLY.items()} == {
        f"{BACKEND}/sparse_motion.py": ("FREE_SPACE_CLEARANCE_M", "LOCAL_JOINT_JUMP_LIMIT_RAD",
                                        "GLOBAL_JOINT_JUMP_LIMIT_RAD", "STAGED_CONTACT_STRATEGY",
                                        "gripper_closed_floor"),
        f"{BACKEND}/recovery.py": ("RETRACT_THEN_UNWIND", "configured_route"),
        f"{BACKEND}/motion_stall.py": (), f"{BACKEND}/staged_contact.py": (), f"{BACKEND}/wait.py": (),
        f"{BACKEND}/ik.py": (), "ora/contracts/planning_diagnostics.py": ()}
    for path, (reached, reason) in replay_kernel.CONTROL_REPLAY_EXECUTOR_ONLY.items():
        bindings, _others = replay_kernel.module_shape((project / path).read_bytes())
        assert reason and set(reached) <= set(bindings), path
    table = {Path(path).stem for path in replay_kernel.CONTROL_REPLAY_EXECUTOR_ONLY}
    allowed = {(Path(path).stem, name) for path, (reached, _) in replay_kernel.CONTROL_REPLAY_EXECUTOR_ONLY.items()
               for name in reached} | set(replay_kernel.CONTROL_REPLAY_UNREACHED)
    for module in ("session", "bridge", "round_lifecycle", "feedback", "observation"):
        assert _table_references(project / BACKEND / f"{module}.py", table) <= allowed, module
    for (module, name), caller in replay_kernel.CONTROL_REPLAY_UNREACHED.items():
        bindings, _others = replay_kernel.module_shape((project / BACKEND / f"{module}.py").read_bytes())
        assert caller and name in bindings, (module, name)


BACKEND = "ora/environment/backends/robodojo"


def _table_references(path: Path, table: set[str]) -> set[tuple[str, str]]:
    """``module.name`` and ``from .module import name`` references into the table's modules."""
    tree = ast.parse(path.read_text())
    aliases, found = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is None and node.level == 1:
            aliases.update({alias.asname or alias.name: alias.name for alias in node.names if alias.name in table})
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] in table:
            found.update(((node.module or "").split(".")[-1], alias.name) for alias in node.names)
    found.update((aliases[node.value.id], node.attr) for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases)
    return found


def test_the_report_names_an_admitted_kernel_drift():
    assert replay_kernel.admission_note({"status": "identical_kernel"}) is None
    note = replay_kernel.admission_note({"status": "executor_drift_admitted", "differing": ["a", "b"]})
    assert "differs from the run's" in note and "AST-equivalent" in note


def test_an_unassigned_lease_is_released_by_its_claim_token(monkeypatch, tmp_path):
    from ora.orchestrator import cli

    released = []
    monkeypatch.setattr(cli, "claim_resource_lease", lambda plan, owner, pid: {"claim_token": "token"})
    monkeypatch.setattr(cli, "assign_resource_lease", lambda lease, pid: (_ for _ in ()).throw(RuntimeError("lost")))
    monkeypatch.setattr(cli, "release_resource_lease",
                        lambda owner, *, owner_pid=None, claim_token=None: released.append((owner_pid, claim_token)))
    with pytest.raises(RuntimeError, match="lost"):
        launcher._leased_spawn({}, None, [], tmp_path, 1.0, {})
    assert released == [(None, "token")]


def test_a_timed_out_worker_is_killed_with_its_process_group(tmp_path):
    session = type("S", (), {"worker_environment": {}})()
    command = [sys.executable, "-c", "import subprocess, sys, time; "
               "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"]
    assert launcher._spawn(session, command, tmp_path, 1.0, {}) == -9


def _copy_journals(run: Path, target: Path) -> Path:
    environment = target / "environment"
    shutil.copytree(run / "native" / "controller", environment / "controller")
    return environment


def test_physics_step_divergence_compares_joints_fingers_and_every_applied_target(tmp_path):
    run = _exact_run(tmp_path / "run")
    environment = _copy_journals(run, tmp_path / "pass-a")
    same = launcher.physics_step_divergence(run / "native", environment)
    assert same["lines_compared"] == 50 and set(same["max_divergence"].values()) == {0.0}
    path = environment / "controller" / "action-0002.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[3]["measured"]["right"]["arm_joint_qpos"][1] = 0.02
    lines[4]["applied_position_targets"]["left_arm_joint_state"][0] = 0.5
    lines[5]["applied_position_targets"]["support_arm0_joint_state"] = [0.0]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    moved = launcher.physics_step_divergence(run / "native", environment)
    assert moved["max_divergence"]["joint_rad"] == pytest.approx(0.02)
    assert moved["max_divergence"]["applied_target"] == pytest.approx(0.5)
    assert moved["max_divergence"]["applied_keys_differ"] == 1.0
    assert (moved["first_divergence"]["action_sequence"], moved["first_divergence"]["line"]) == (2, 3)


def test_object_pose_divergence_reports_spread_against_one_millimetre(tmp_path):
    def write(path: Path, z: float) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"action_sequence": 0, "tick_index": 0, "objects": {"block": {"position_m": [0.0, 0.0, 1.0]}}},
                {"action_sequence": 0, "tick_index": 1, "objects": {"block": {"position_m": [0.0, 0.0, z]}}}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    first = write(tmp_path / "a" / "object-poses.jsonl", 1.0)
    assert launcher.object_pose_divergence(first, write(tmp_path / "b" / "object-poses.jsonl", 1.0005))["status"] \
        == "within_bound"
    result = launcher.object_pose_divergence(first, write(tmp_path / "c" / "object-poses.jsonl", 1.003))
    assert result["status"] == "exceeds_bound" and result["max_position_divergence_m"] == pytest.approx(0.003)
    assert result["first_divergence"]["tick_index"] == 1
    assert launcher.object_pose_divergence(tmp_path / "x", tmp_path / "y")["status"] == "unavailable"


def test_report_only_metrics_after_replay_passes_on_an_exact_replay(tmp_path):
    run = _exact_run(tmp_path / "run")
    _replay(run, tmp_path / "out" / "pass-a", FakeSession())
    code, report = _report(run, tmp_path / "out", "metrics-after-replay")
    assert code == 0 and report["passed"] is True and report["failures"] == []
    assert report["schema"] == launcher.REPORT_SCHEMA and report["replay_contract"] == "control"
    assert report["passes"]["pass-a"]["divergence_vs_original"]["max_joint_divergence_rad"] == 0.0
    assert report["run"]["run_id"] == "run" and report["metric_definition_version"]


def test_report_fails_on_applied_target_mismatch_and_a_malformed_replay_journal(tmp_path):
    run = _exact_run(tmp_path / "run")
    _replay(run, tmp_path / "out" / "pass-a", FakeSession(applied_offset=0.01))
    code, report = _report(run, tmp_path / "out", "replay")
    assert code == 1
    assert "pass-a applied targets differed from the resolved command" in report["failures"]
    assert "pass-a applied_target diverges from the original" in report["failures"]
    journal = tmp_path / "out" / "pass-a" / "environment" / "controller" / "action-0000.jsonl"
    journal.write_text(journal.read_text()[:-30], encoding="utf-8")
    code, report = _report(run, tmp_path / "out", "replay")
    assert code == 1 and any("malformed" in failure for failure in report["failures"])


def test_report_only_determinism_gives_joints_and_objects_separate_statuses(tmp_path):
    run = _exact_run(tmp_path / "run")
    objects = iter([{"block": {"position_m": [0.0, 0.0, 1.0 + 0.001 * k]}} for k in range(8)])
    _replay(run, tmp_path / "out" / "pass-a", FakeSession(),
            read_objects=lambda: {"block": {"position_m": [0, 0, 1.0]}})
    _replay(run, tmp_path / "out" / "pass-b", FakeSession(), read_objects=lambda: next(objects))
    code, report = _report(run, tmp_path / "out", "determinism")
    determinism = report["determinism"]
    assert determinism["joints"]["max_joint_divergence_rad"] == 0.0
    assert determinism["objects"]["status"] == "exceeds_bound"
    assert determinism["objects"]["max_position_divergence_m"] == pytest.approx(0.004)
    assert code == 0 and report["passed"] is True


def test_uncommanded_arm_targets_rewritten_outside_a_tick_are_seeded_from_the_journal(tmp_path):
    held = {"right_arm_joint_state": {"position": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    run = write_native(tmp_path / "run", [[
        action(receipt("move_pose"), [tick("move", control(right=A), right=still("right", A))]),
        action(receipt("set_gripper"), [tick("gripper", RIGHT_EE, right=still("right", A),
                                             command={**RIGHT_EE, **held,
                                                      "support_arm0_joint_state": {"position": [9.0]}})]),
    ]])
    session = FakeSession()
    report = _replay(run, tmp_path / "out", session)
    assert report["replay"]["held_target_seeds"] == [
        {"action_sequence": 1, "tick_index": 0, "keys": ["right_arm_joint_state"]}]
    retained = session.environment.robot_manager.control_manager.prev_control[0]
    assert retained["right_arm_joint_state"] == held["right_arm_joint_state"]
    assert "support_arm0_joint_state" not in retained


def test_the_first_divergence_names_where_to_look():
    def line(**applied):
        return {"applied_position_targets": {key: list(value) for key, value in applied.items()}}

    base = line(right_arm_joint_state=[0.0], support_arm0_joint_state=[0.0])
    moved = line(right_arm_joint_state=[0.1], support_arm0_joint_state=[0.0])
    support = line(right_arm_joint_state=[0.0], support_arm0_joint_state=[0.1])
    commanded = {"right_arm_joint_state": {"position": [0.1]}}
    assert launcher.divergence_cause({}, 20, base, moved).startswith("held_target_at_tick_start")
    assert launcher.divergence_cause({}, 21, base, moved).startswith("held_target_within_tick")
    assert launcher.divergence_cause(commanded, 20, base, moved).startswith("commanded_target")
    assert launcher.divergence_cause({}, 3, base, support).startswith("support_arm_target")
    assert launcher.divergence_cause({}, 3, base, base).startswith("measured_state")


def test_an_interrupt_kills_registered_worker_groups_so_a_blocked_pool_thread_returns(tmp_path):
    session = type("S", (), {"worker_environment": {}})()
    started, codes = [], []
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    worker_thread = threading.Thread(target=lambda: codes.append(
        launcher._spawn(session, command, tmp_path, 60.0, {}, started.append)))
    worker_thread.start()
    for _ in range(200):
        if started:
            break
        threading.Event().wait(0.05)
    assert started and started[0] in launcher._ACTIVE_GROUPS
    with pytest.raises(KeyboardInterrupt):
        launcher.kill_active_groups(signal.SIGINT)
    worker_thread.join(10)
    assert not worker_thread.is_alive() and codes == [-9] and not launcher._ACTIVE_GROUPS


def test_a_report_rebuilt_outside_git_skips_the_commit_comparison(tmp_path, monkeypatch):
    run = _exact_run(tmp_path / "run")
    _replay(run, tmp_path / "out" / "pass-a", FakeSession())
    (tmp_path / "out" / "pass-a" / "launch.json").write_text(json.dumps({"identity": {
        "commit": "a" * 40, "kernel_admission": {"status": "executor_drift_admitted"}}}))
    monkeypatch.setattr(launcher, "harness_identity", lambda commit: {"commit": None, "kernel_admission": None})
    code, report = _report(run, tmp_path / "out", "replay")
    assert code == 0 and report["rebuilt_outside_git"] is True and report["kernel_notice"] is None
