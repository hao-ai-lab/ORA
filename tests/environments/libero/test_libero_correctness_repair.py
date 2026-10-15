"""CPU regression coverage of the dispatched LIBERO motion and IPC boundaries."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment.backends.libero_pro.session import (
    LiberoProError,
    LiberoProSession,
    _CameraMetadataEnvironment,
)
from ora.environment.controller_replay import ControllerReplayRecorder
from ora.environment.controllers.curobo_motion import (
    CUROBO_RESPONSE_SCHEMA,
    CuroboPlannerClient,
    CuroboPlannerError,
)
from ora.environment.controllers.hybrid_motion import (
    libero_global_endpoint_tolerances,
)


ROOT = Path(__file__).resolve().parents[3]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
TARGET = [0.25, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0]


class _Plant:
    """Only the simulator API is replaced; controller dispatch stays real."""

    def __init__(self) -> None:
        self.commands: list[list[float]] = []
        self.endpoint = list(TARGET)
        self.pose = list(START)
        self.termination_tick: int | None = None
        self.truncation_tick: int | None = None
        self.fail_restore = False
        self.malformed_observation = False
        self._action_dim = 7
        self.robot = SimpleNamespace(
            _joint_positions=np.zeros(7), controller_config={"type": "OSC_POSE"},
            action_dim=7,
        )
        self.robot._load_controller = self.load_controller
        self.robots = [self.robot]

    def load_controller(self) -> None:
        joint = self.robot.controller_config["type"] == "JOINT_POSITION"
        if not joint and self.commands and self.fail_restore:
            raise RuntimeError("injected controller restoration failure")
        self.robot.action_dim = 8 if joint else 7

    def observation(self) -> dict[str, Any]:
        return {"pose": list(self.pose), "robot0_gripper_qpos": [0.024, -0.024],
                "actual_tick": len(self.commands)}

    def step(self, command: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert len(command) == self._action_dim == 8
        self.commands.append(np.asarray(command).tolist())
        self.robot._joint_positions += np.asarray(command[:7]) * 0.05
        self.pose = list(self.endpoint)
        tick = len(self.commands)
        return (
            None if self.malformed_observation else self.observation(), 0.0,
            tick == self.termination_tick, tick == self.truncation_tick, {},
        )


class _Planner:
    def __init__(self) -> None:
        self.binding = {"revision": "qualified-model", "worker": {"pid": 11}}
        self.requests: list[dict[str, Any]] = []
        self.fk_delta = 0.0
        self.no_op = False
        self.refuse: str | None = None

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(deepcopy(request))
        if self.refuse:
            return {"reachable": False, "status": "not_reachable",
                    "planner_status": self.refuse}
        start = request["joint_positions"]
        goal = request.get("goal_joint_positions", [0.05 + x for x in start])
        return {
            "reachable": True, "status": "planned", "planner_status": "success",
            "planning_time_s": 0.01,
            "start_ee_pose_base_xyzw": [0.2 + self.fk_delta, 0, 0.9, 0, 0, 0, 1],
            "trajectory": {"positions": [start] if self.no_op else [start, goal],
                           "joint_names": JOINT_NAMES, "dt_s": 0.004},
        }


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Plant]:
    controllers = ModuleType("robosuite.controllers")
    controllers.load_controller_config = lambda *, default_controller: {  # type: ignore[attr-defined]
        "type": default_controller,
    }
    monkeypatch.setitem(sys.modules, "robosuite.controllers", controllers)
    plant = _Plant()
    worker = _CameraMetadataEnvironment(plant)
    profile = json.loads((ROOT / (
        "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    )).read_text())
    worker.harness_motion_profile = profile

    class _Environment:
        def __init__(self) -> None:
            self.current_raw_obs = [plant.observation()]
            self._elapsed_steps = 0
            self.refresh_steps = 0
            self.fail_attribute: str | None = None
            self.env = self

        def set_env_attr(self, name: str, value: Any) -> None:
            setattr(worker, name, value)

        def get_env_attr(self, name: str) -> list[Any]:
            if name == self.fail_attribute:
                raise RuntimeError("injected observation transport failure")
            return [getattr(worker, name)]

        def step(self, *_args: Any, **_kwargs: Any) -> Any:
            self.refresh_steps += 1
            raise AssertionError("joint execution must not add an OSC refresh step")

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.environment = _Environment()
    session.motion_planner = _Planner()
    session.motion_profile = profile
    session.action_sequence = 0
    session.sim_steps = 0
    session.round_index = 0
    session.terminated = False
    session.truncated = False
    session.first_termination = None
    session.world_mutation_sequence = 0
    session.world_mutation_actions = []
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.home_joint_state = [0.0] * 7
    session.last_backtrack_joint_state = [-0.1] * 7
    session.robustness_episode = None
    session.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
    session.source_manifest = {"revision": "source-a"}
    session.qualified_controller_binding = {"controller": "qualified-source"}
    session.abort_reason = None
    session.control_fault_reason = None
    session.suite = "libero_10"
    session.task_id = session.seed = 0
    session.robustness_case = "core"
    session.replay_recorder = ControllerReplayRecorder(
        tmp_path, "libero_pro", session.source_manifest, ["agentview"],
    )
    session._current_state = lambda: (  # type: ignore[method-assign]
        np.asarray(session.environment.current_raw_obs[0]["pose"][:3]),
        np.asarray(session.environment.current_raw_obs[0]["pose"][3:]), 0.6,
    )
    session._motion_state = lambda: {  # type: ignore[method-assign]
        "joint_names": JOINT_NAMES,
        "joint_positions": plant.robot._joint_positions.tolist(),
        "base_pose_world_xyzw": [0, 0, 0, 0, 0, 0, 1],
        "hand_pose_world_xyzw": [0.2, 0, 0.9, 0, 0, 0, 1],
        "grip_site_pose_world_xyzw": plant.pose,
        "external_wrench_clipped": None,
    }
    session.home_tcp_pose = list(START)
    session.last_backtrack_tcp_pose = list(START)
    monkeypatch.setattr(_CameraMetadataEnvironment, "harness_grip_site_pose", property(
        lambda self: {"position_m": np.asarray(self._delegate.pose[:3]),
                      "orientation_xyzw": np.asarray(self._delegate.pose[3:])},
    ))
    return session, plant


def _action(*, plan_ref: str | None = None) -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v3",
        "action_id": "action-0", "observation_id": "observation-0",
        "bound_observation_sha256": "0" * 64,
        "action": "move_pose" if plan_ref else "move_free_space",
        "arguments": {"plan_ref": plan_ref} if plan_ref else {
            "arm": "right", "target_pose": {
                "position_m": TARGET[:3], "orientation_xyzw": TARGET[3:],
            },
        },
    }


def _probe(session: Any) -> str:
    result = session.probe_motion({
        "schema": "ora.motion-probe-request.v1",
        "observation_id": "observation-0", "observation_sha256": "0" * 64,
        "variant_policy": "exact",
        "queries": [{"query_id": "q0", "arm": "right", "pose": {
            "position_m": TARGET[:3], "orientation_xyzw": TARGET[3:],
        }}],
    })
    return result["results"][0]["plan_ref"]


@pytest.mark.parametrize("outcome", ["termination", "truncation"])
def test_joint_first_terminal_tick_has_no_extra_step_and_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    setattr(plant, f"{outcome}_tick", 1)
    receipt = session._execute_planned_motion(_action())
    assert len(plant.commands) == receipt["steps_used"] == session.sim_steps == 1
    assert session.environment.refresh_steps == 0
    assert session.environment._elapsed_steps == 1
    assert session.environment.current_raw_obs[0]["actual_tick"] == 1
    assert receipt["after"]["right"]["pose_xyz_xyzw"] == TARGET
    assert session.last_backtrack_joint_state == [0.0] * 7
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is (outcome == "termination")
    assert verdict["first_termination"]["sim_step"] == 1
    first = tmp_path / "rounds/round-0000/first-termination.json"
    assert json.loads(first.read_text()) == verdict["first_termination"]
    tracking = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert tracking["terminal_tick"] == 1
    assert sum(tracking["waypoint_ticks"]) == 1
    assert len(tracking["commands"]) == len(tracking["samples"]) == 1
    entry = session.replay_recorder.entries[0]
    assert entry["physics_steps"] == 1
    with np.load(next((tmp_path / "controller-replay").rglob("*.npz"))) as arrays:
        assert arrays.files == ["commands"]
        np.testing.assert_allclose(arrays["commands"], plant.commands)


@pytest.mark.parametrize("failure", [
    "observation_transport", "execution_result_transport", "controller_restore",
    "observation_shape", "evidence", "termination_evidence",
])
def test_joint_official_termination_survives_post_step_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    plant.termination_tick = 1
    if failure == "observation_transport":
        session.environment.fail_attribute = "harness_joint_execution_observation"
    elif failure == "execution_result_transport":
        session.environment.fail_attribute = "harness_joint_execution_result"
    elif failure == "controller_restore":
        plant.fail_restore = True
    elif failure == "observation_shape":
        plant.malformed_observation = True
    elif failure == "termination_evidence":
        from ora.environment.backends.libero_pro import session as backend

        atomic_json = backend._atomic_json

        def fail_termination(path: Path, payload: Any) -> None:
            if path.name == "first-termination.json":
                raise RuntimeError("injected first-termination persistence failure")
            atomic_json(path, payload)

        monkeypatch.setattr(backend, "_atomic_json", fail_termination)
    else:
        def fail_evidence(_execution: Any) -> None:
            raise RuntimeError("injected evidence persistence failure")
        session._persist_joint_execution_evidence = fail_evidence
    with pytest.raises((RuntimeError, LiberoProError)):
        session._execute_planned_motion(_action())
    assert len(plant.commands) == session.sim_steps == 1
    assert session.environment.refresh_steps == 0
    assert session.last_backtrack_joint_state == [0.0] * 7
    assert session.persist_official_verdict()["official_success"] is True
    assert session.first_termination["controller"] == "JOINT_POSITION"


def test_joint_command_transport_failure_does_not_reuse_previous_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    session._execute_planned_motion(_action())
    previous = list(session.last_backtrack_joint_state)

    def failed_dispatch(_name: str, _value: Any) -> None:
        raise RuntimeError("injected command transport failure")

    session.environment.set_env_attr = failed_dispatch
    with pytest.raises(RuntimeError, match="command transport failure"):
        session._execute_planned_motion(_action())
    assert len(plant.commands) == session.sim_steps == 1
    assert session.last_backtrack_joint_state == previous
    assert session.first_termination is None


@pytest.mark.parametrize("translation,rotation,reached", [
    (0.014, 0.0, True), (0.016, 0.0, False),
    (0.0, math.radians(4.9), True), (0.0, math.radians(5.1), False),
])
@pytest.mark.parametrize("reference", [False, True])
def test_global_reach_uses_measured_tcp_for_direct_and_stored_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    translation: float, rotation: float, reached: bool, reference: bool,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += translation
    plant.endpoint[3:] = [0, 0, math.sin(rotation / 2), math.cos(rotation / 2)]
    receipt = (
        session._execute_plan_reference(_action(plan_ref=_probe(session)))
        if reference else session._execute_planned_motion(_action())
    )
    status = receipt["execution_status"] if reference else receipt["motion_status"]
    assert status == ("reached" if reached else "not_reached" if reference else "stopped")
    inner = receipt["segment"] if reference else receipt
    assert inner["feedback"].get("stop_reason") == (None if reached else "terminal_tracking_timeout")
    assert inner["feedback"]["tracking_error_l2_rad"] == 0.0
    assert inner["feedback"]["endpoint_error"]["position_m"] == pytest.approx(translation)
    assert inner["feedback"]["endpoint_error"]["orientation_rad"] == pytest.approx(rotation)
    assert inner["control_health"]["status"] == ("healthy" if reached else "degraded")
    # A miss only stops after the bounded terminal settle (40 ticks) is exhausted.
    assert len(plant.commands) == session.sim_steps == (1 if reached else 41)
    assert session.environment.refresh_steps == 0


def test_global_endpoint_tolerances() -> None:
    assert libero_global_endpoint_tolerances() == (0.015, math.radians(5))


@pytest.mark.parametrize("route", ["direct", "probe", "reference", "recovery"])
def test_start_fk_drift_rejected_before_any_dispatch_on_every_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    reference = _probe(session) if route == "reference" else None
    if reference:
        session.plan_references[reference]["planner_start_fk"][0] += 0.006
    else:
        session.motion_planner.fk_delta = 0.006
    with pytest.raises(LiberoProError, match="FK"):
        if route == "probe":
            _probe(session)
        elif reference:
            session._execute_plan_reference(_action(plan_ref=reference))
        else:
            action = _action()
            if route == "recovery":
                action.update(action="recover_motion", arguments={"strategy": "home"})
            session._execute_planned_motion(action)
    assert not plant.commands
    assert session.sim_steps == 0
    assert session.last_backtrack_joint_state == [-0.1] * 7


@pytest.mark.parametrize("drift", ["source", "planner", "profile", "controller", "round", "trajectory", "artifact"])
def test_plan_reference_rejects_changed_source_or_artifact_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    reference = _probe(session)
    stored = session.plan_references[reference]
    if drift == "source":
        session.source_manifest["revision"] = "other-source"
    elif drift == "planner":
        session.motion_planner.binding["revision"] = "other-model"
    elif drift == "profile":
        session.motion_profile["profile_id"] = "other-profile"
    elif drift == "controller":
        session.qualified_controller_binding["controller"] = "other-controller"
    elif drift == "round":
        session.round_index = 1
    elif drift == "trajectory":
        stored["trajectory"]["positions"][-1][0] += 0.01
    else:
        (tmp_path / stored["artifact"]["path"]).write_text("{}")
    with pytest.raises(LiberoProError, match="identity drifted"):
        session._execute_plan_reference(_action(plan_ref=reference))
    assert not plant.commands
    assert session.last_backtrack_joint_state == [-0.1] * 7


def test_stored_plan_survives_worker_restart_and_gripper_only_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    reference = _probe(session)
    session.motion_planner.binding["worker"]["pid"] = 22
    session.world_mutation_sequence = 1
    session.world_mutation_actions = ["set_gripper"]
    receipt = session._execute_plan_reference(_action(plan_ref=reference))
    assert receipt["execution_status"] == "reached"
    assert len(plant.commands) == 1
    assert session.last_backtrack_joint_state == [0.0] * 7


@pytest.mark.parametrize("no_dispatch", ["no_op", "refused", "stopped"])
def test_zero_dispatch_preserves_previous_state_and_uses_no_refresh_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_dispatch: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    if no_dispatch == "no_op":
        session.motion_planner.no_op = True
        plant.pose = list(TARGET)  # A no-op plan starts with the measured TCP at its target.
    elif no_dispatch == "refused":
        session.motion_planner.refuse = "IK_FAIL"
    else:
        stop = tmp_path / "control/stop.request"
        stop.parent.mkdir()
        stop.write_text("operator stop\n")
    receipt = session._execute_planned_motion(_action())
    assert receipt["state_changed"] is False
    assert receipt["steps_used"] == session.sim_steps == 0
    assert not plant.commands
    assert session.last_backtrack_joint_state == [-0.1] * 7
    assert session.environment.refresh_steps == 0


def test_recovery_does_not_replace_previous_state_with_its_own_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    action = _action()
    action.update(action="recover_motion", arguments={"strategy": "backtrack"})
    session._execute_planned_motion(action)
    assert plant.commands
    assert session.last_backtrack_joint_state == [-0.1] * 7


def test_restart_round_clears_old_abort_and_plans_but_preserves_new_stop_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    old_reference = _probe(session)
    session.task = "libero_10-t00"
    session.expected_task_language = "test reset boundary"
    session.metadata = {"reset_id": 0, "public_environment_context": {}}
    session.controller_action_contract = {"contract_sha256": "a" * 64}
    session.action_codec = "motion_action_v3"
    reset_calls = []

    def reset() -> None:
        reset_calls.append(True)
        plant.robot._joint_positions = np.zeros(7)
        plant.pose = list(START)
        session.environment.current_raw_obs = [plant.observation()]
        session.environment._elapsed_steps = 0

    session.environment.reset = reset
    session.abort("prior round aborted")
    session.last_backtrack_joint_state = [0.7] * 7
    stop_path = tmp_path / "control/stop.request"
    assert stop_path.is_file() and old_reference in session.plan_references

    reset_record = session.restart_round(1)
    assert reset_calls == [True]
    assert reset_record["round_index"] == session.round_index == 1
    assert session.abort_reason is None and session.sim_steps == 0
    assert not stop_path.exists()
    assert session.plan_references == {}
    assert session.last_backtrack_joint_state == [0.0] * 7
    with pytest.raises(LiberoProError, match="unknown or tampered"):
        session._execute_plan_reference(_action(plan_ref=old_reference))
    assert not plant.commands

    moved = session._execute_planned_motion(_action())
    assert moved["steps_used"] == 1 and moved["motion_status"] == "reached"
    assert len(plant.commands) == 1
    # An actual stop arriving after reset keeps its full zero-tick authority.
    stop_path.write_text("new external operator stop\n")
    stopped = session._execute_planned_motion(_action())
    assert stopped["steps_used"] == 0 and stopped["state_changed"] is False
    assert stopped["feedback"]["stop_reason"] == "operator_stop_requested"
    assert len(plant.commands) == session.sim_steps == 1
    assert stop_path.is_file()
    assert session.last_backtrack_joint_state == [0.0] * 7


_WORKER_SCRIPT = r'''
import json, sys, time
mode = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    if request["command"] == "close":
        break
    if request["command"] == "ready":
        response = {"status": "ready", "version": "v0.7.8"}
    else:
        if mode in {"timeout", "partial"}:
            if mode == "partial":
                sys.stdout.write('{"request_id":')
                sys.stdout.flush()
            time.sleep(5)
        if mode == "json":
            print("not-json", flush=True)
            continue
        response = {
            "schema": "ora.curobo-motion-response.v1",
            "status": "not_reachable", "reachable": False,
            "world_model": "none", "planner_status": "IK_FAIL",
        }
        if mode == "schema":
            response["schema"] = "wrong"
        if mode == "request":
            response["request"] = {k: request[k] for k in (
                "joint_names", "joint_positions", "goal_joint_positions")}
            response["request"]["joint_positions"] = [0.2] * 7
    response["request_id"] = "wrong" if mode == "identity" and request["command"] == "plan" else request["request_id"]
    print(json.dumps(response), flush=True)
'''


@pytest.mark.parametrize("failure", ["timeout", "partial", "json", "identity", "schema", "request"])
def test_curobo_transport_failure_retires_worker_and_next_plan_starts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    monkeypatch.setattr(CuroboPlannerClient, "__post_init__", lambda self: None)
    client = CuroboPlannerClient(Path(sys.executable), tmp_path, "cpu-test", tmp_path,
                                 request_timeout_s=0.25)
    processes: list[Any] = []
    real_popen = subprocess.Popen

    def launch(*_args: Any, **kwargs: Any) -> Any:
        # Production start/readiness/plan/cleanup are unchanged. Replace only
        # the unavailable GPU worker executable with the bounded CPU protocol.
        kwargs["stderr"].close()
        process = real_popen(
            [sys.executable, "-u", "-c", _WORKER_SCRIPT,
             failure if not processes else "normal"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    request = {"joint_names": JOINT_NAMES, "joint_positions": [0.0] * 7,
               "goal_joint_positions": [0.1] * 7}
    started = time.monotonic()
    try:
        with pytest.raises(CuroboPlannerError):
            client.plan(request)
        assert time.monotonic() - started < 3
        assert client._process is None and "worker" not in client.binding
        assert processes[0].poll() is not None
        assert processes[0].stdin.closed and processes[0].stdout.closed
        # The replacement worker is a cold interpreter start; under a loaded
        # parallel suite it can miss the 0.25 s bound that only the faulted
        # first worker needed.
        client.request_timeout_s = 10.0
        result = client.plan(request)
        assert result["schema"] == CUROBO_RESPONSE_SCHEMA
        assert result["planner_status"] == "IK_FAIL"
        assert len(processes) == 2 and client._process is processes[1]
    finally:
        client.close()


def test_v2_terminal_settle_refines_joint_residual_below_waypoint_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    step = plant.step

    def slow_step(command: Any) -> Any:
        before = plant.robot._joint_positions.copy()
        result = step(command)
        plant.robot._joint_positions = before + np.asarray(command[:7]) * 0.01
        return result

    plant.step = slow_step  # type: ignore[method-assign]
    receipt = session._execute_planned_motion(_action())
    evidence = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert receipt["motion_status"] == "reached"
    assert 1 < evidence["terminal_settle"]["ticks"] <= 40
    assert evidence["tracking_error_max_abs_rad"] <= 0.005
    assert evidence["terminal_settle"]["endpoint_error"]["position_m"] == pytest.approx(0)
    assert session.motion_profile["global_motion"]["joint_tracking_error_limit_rad"] == 0.08


@pytest.mark.parametrize("axis", ["translation", "orientation"])
def test_v2_settle_requires_measured_tcp_arrival_with_bounded_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axis: str,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    if axis == "translation":
        plant.endpoint[0] += 0.020
    else:
        plant.endpoint[3:] = [0, 0, math.sin(math.radians(10) / 2), math.cos(math.radians(10) / 2)]
    receipt = session._execute_planned_motion(_action())
    execution = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert execution["tracking_error_max_abs_rad"] == pytest.approx(0)
    assert execution["terminal_settle"]["ticks"] == 40
    assert len(plant.commands) == 41
    assert receipt["motion_status"] == "stopped"
    assert receipt["feedback"]["stop_reason"] == "terminal_tracking_timeout"


def test_v2_settle_does_not_step_after_first_official_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += 0.020
    plant.termination_tick = 2
    session._execute_planned_motion(_action())
    assert len(plant.commands) == session.sim_steps == 2
    assert session.terminated is True
    assert session.first_termination["sim_step"] == 2


def test_v2_recovery_uses_paired_saved_joint_and_tcp_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    plant.robot._joint_positions = np.asarray([0.03] * 7)
    plant.pose = list(START)
    plant.endpoint = list(START)
    receipt = session._execute_planned_motion({**_action(), "action": "recover_motion",
                               "arguments": {"arm": "right", "strategy": "home"}})
    assert receipt["motion_status"] == "reached"
    assert len(plant.commands) == 1
    assert receipt["execution"]["terminal_settle"]["joint_tolerance_rad"] == 0.005
    assert session.home_tcp_pose == START


def test_force_feedback_rotates_actual_sensor_frame_and_reverses_reaction() -> None:
    from ora.environment.controllers.force_feedback import sensor_reaction_feedback

    projected = sensor_reaction_feedback([30.9, 0, 0], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert projected["external_force_world_n"] == pytest.approx([0, 0, 30.9])
    assert projected["external_force_direction_world"] == pytest.approx([0, 0, 1])
    assert projected["force_direction_convention"] == "environment_on_eef_reaction_estimate"
    # Invalid/missing calibration must not turn sensor axes into world axes.
    for rotation in (None, [[1, 0, 0]] * 3, [[1, 0, 0], [0, 1, 0], [0, 0, -1]]):
        assert sensor_reaction_feedback([0, 0, 30.9], rotation)["external_force_direction_world"] is None
    assert sensor_reaction_feedback([0, 0, 0], np.eye(3))["force_direction_status"] == "zero_force"


@pytest.mark.parametrize("interaction,force,enabled,expected", [
    ("precision", 40.0, True, None),
    ("precision", 40.001, True, "contact_stop"),
    ("precision", 10.1, True, None),
    ("contact_approach", 42.7, True, None),
    ("precision", 42.7, False, None),
    ("precision", float("nan"), True, None),
])
def test_shared_precision_force_stop_has_exact_profile_and_threshold_boundary(
    interaction: str, force: float, enabled: bool, expected: str | None,
) -> None:
    from ora.environment.controllers.force_feedback import precision_force_stop_reason
    assert precision_force_stop_reason(interaction=interaction, magnitude_n=force,
                                       threshold_n=40.0, enabled=enabled) == expected


def test_actual_force_sensor_binding_is_retained_without_substituting_hand_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from enum import IntEnum
    transforms = ModuleType("robosuite.utils.transform_utils")
    transforms.mat2quat = lambda matrix: np.asarray([0, 0, 0, 1])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robosuite.utils.transform_utils", transforms)
    mujoco = ModuleType("mujoco")
    mujoco.mjtObj = IntEnum("mjtObj", {"mjOBJ_SITE": 6})  # type: ignore[attr-defined]
    mujoco.mjtSensor = IntEnum("mjtSensor", {"mjSENS_FORCE": 4})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    rotation = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    model = SimpleNamespace(body_name2id=lambda name: 0, site_name2id=lambda name: 1,
                            sensor_name2id=lambda name: 0, sensor_objid=np.asarray([0]),
                            sensor_objtype=np.asarray([6]), sensor_type=np.asarray([4]))
    data = SimpleNamespace(body_xpos=np.zeros((1, 3)), body_xmat=np.eye(3).reshape(1, 9),
                           site_xpos=np.zeros((2, 3)),
                           site_xmat=np.asarray([rotation.reshape(-1), np.eye(3).reshape(-1)]))
    robot = SimpleNamespace(
        ee_force=np.asarray([0, 0, 30.9]), ee_torque=np.zeros(3),
        robot_model=SimpleNamespace(root_body="root", eef_name="hand"),
        gripper=SimpleNamespace(important_sensors={"force_ee": "actual_force_sensor"},
                                important_sites={"grip_site": "grip"}),
        _joint_positions=np.zeros(7), _joint_velocities=np.zeros(7), robot_joints=JOINT_NAMES,
    )
    worker = _CameraMetadataEnvironment(SimpleNamespace(robots=[robot], sim=SimpleNamespace(model=model, data=data)))
    worker.harness_motion_profile = json.loads((ROOT / (
        "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    )).read_text())
    state = worker.harness_motion_state
    assert state["external_force_world_n"] == pytest.approx([0, 0, 30.9])
    assert state["external_wrench_clipped"][:3] == pytest.approx([0, 0, 30.9])
    assert state["force_source_binding"]["sensor_site_id"] == 0
    assert len(state["force_source_binding"]["sha256"]) == 64


@pytest.mark.parametrize("interaction,forces,expected_steps,stopped", [
    ("precision", [42.7], 0, True),
    ("precision", [30.9], 1, False),
    ("precision", [40.0], 1, False),
    ("precision", [9.0, 42.7], 1, True),
    ("contact_approach", [42.7], 1, False),
])
def test_v2_precision_force_stop_runs_through_sparse_dispatch_without_terminal_fault(
    tmp_path: Path, interaction: str, forces: list[float], expected_steps: int, stopped: bool,
) -> None:
    from tests.orchestrator.test_runtime_v1 import FakeLiberoEnvironment, _fake_libero_metadata
    from ora.environment.backends.libero_pro.session import (
        libero_environment_init_params, libero_public_environment_context,
    )
    from ora.environment.controllers.force_feedback import sensor_reaction_feedback

    environment = FakeLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )
    session = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]},
        max_sim_steps=100,
        environment_factory=lambda **kwargs: (environment, _fake_libero_metadata(tmp_path, public_context)),
    )
    session.start()
    session.action_codec = "motion_action_v3"
    session.motion_profile = json.loads((ROOT / (
        "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    )).read_text())

    def measured_force() -> dict[str, Any]:
        force = forces[min(environment.step_count, len(forces) - 1)]
        return {
            "joint_names": JOINT_NAMES, "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "external_force_magnitude_n": force,
            "external_wrench_clipped": [0, 0, min(force, 40)],
            **sensor_reaction_feedback([0, 0, force], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        }

    session._motion_state = measured_force  # type: ignore[method-assign]
    receipt = session.execute({
        "schema": "ora-motion-action.v3", "observation_id": "observation-0",
        "action_id": "precision-force", "bound_observation_sha256": "bound",
        "action": "move_pose", "arguments": {
            "targets": {"right": {"position_m": [0.0, 0.0, 0.49],
                                   "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
            "interactions": {"right": interaction},
        },
    })
    segment = receipt["segment"]
    assert environment.step_count == receipt["steps_used"] == expected_steps
    assert segment["motion_status"] == ("stopped" if stopped else "reached")
    assert receipt["failure_code"] == ("contact_stop" if stopped else None)
    assert receipt["failure_stage"] == ("execution" if stopped else None)
    assert segment["control_health"]["status"] == ("degraded" if stopped else "healthy")
    assert segment["control_health"]["force_telemetry"]["stop_authority"] is (interaction == "precision")
    assert segment["feedback"]["contact_force_threshold_n"] == 10.0
    assert segment["feedback"]["external_force_direction_world"] == pytest.approx([0, 0, 1])
    assert segment["feedback"]["external_force_magnitude_n"] == forces[-1]
    assert segment["command_invariant"]["verified"] is True
    assert session.status()["terminal"] is False
    tracking = json.loads((tmp_path / "native/tracking/action-0000.json").read_text())
    if interaction == "precision":
        assert tracking["samples"][-1]["force_feedback"]["external_force_magnitude_n"] == forces[-1]
    session.close()


def test_v2_persisted_command_drift_tracks_only_executed_pose_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _plant = _session(tmp_path, monkeypatch)

    def persist(action_id: str, start: float, final: float, commanded: float, steps: int = 1) -> dict[str, Any]:
        before = [0, 0, start, 0, 0, 0, 1]
        after = [0, 0, final, 0, 0, 0, 1]
        return session._persist_receipt({
            "schema": "ora.libero-sparse-motion-receipt.v1",
            "action": "move_pose", "action_id": action_id,
            "before": {"right": {"pose_xyz_xyzw": before}},
            "after": {"right": {"pose_xyz_xyzw": after}},
            "steps_used": steps, "state_changed": steps > 0,
            "segment": {"commanded_pose_xyz_xyzw": [0, 0, commanded, 0, 0, 0, 1]},
        })

    assert "command_drift" not in persist("initial", .5, .5, .5)
    assert "command_drift" not in persist("refused", .5, .5, .8, 0)
    drift = persist("hold1", .5, .52, .5)["command_drift"]["right"]
    assert drift["drift_from_last_commanded_m"] == pytest.approx([0, 0, .02])
    assert drift["drift_accumulating"] is False
    drift = persist("hold2", .52, .54, .52)["command_drift"]["right"]
    assert drift["drift_accumulating"] is True
    recorded = json.loads((tmp_path / "receipts/action-0003.json").read_text())
    assert recorded["command_drift"]["right"] == drift


def test_v2_profile_requires_its_settle_bound() -> None:
    from ora.environment.controllers.hybrid_motion import (
        HybridMotionProfileError, validate_hybrid_motion_profile,
    )
    root = ROOT / "ora/policy/embodiments/franka-panda"
    new = json.loads((root / "motion-profile-v2.json").read_text())
    validate_hybrid_motion_profile(new)
    for change in (None, 0.004, 0.006):
        invalid = deepcopy(new)
        if change is None:
            invalid["global_motion"].pop("terminal_settle_joint_tolerance_rad")
        else:
            invalid["global_motion"]["terminal_settle_joint_tolerance_rad"] = change
        with pytest.raises(HybridMotionProfileError):
            validate_hybrid_motion_profile(invalid)


def test_endpoint_routes_carry_distinct_bounds_and_reject_a_foreign_one() -> None:
    """Each live LIBERO route owns its own endpoint bound, and a receipt may
    only report the bound its own route actually applied."""

    from ora.contracts.motion import (
        POSE_ORIENTATION_TOLERANCE_RAD, POSE_POSITION_TOLERANCE_M,
    )
    from ora.environment.controllers.hybrid_motion import (
        LIBERO_ENDPOINT_ROUTE_GLOBAL,
        LIBERO_ENDPOINT_ROUTE_LOCAL,
        HybridMotionProfileError,
        libero_endpoint_contract,
        libero_endpoint_tolerances,
        libero_local_endpoint_tolerances,
    )

    assert libero_local_endpoint_tolerances() == (
        POSE_POSITION_TOLERANCE_M, POSE_ORIENTATION_TOLERANCE_RAD,
    ) == (0.005, 0.05)
    assert libero_endpoint_tolerances(LIBERO_ENDPOINT_ROUTE_GLOBAL) == (
        0.015, math.radians(5),
    )

    def contract(route: str, bound: tuple[float, float]) -> dict[str, Any]:
        return libero_endpoint_contract(
            route=route, position_tolerance_m=bound[0],
            orientation_tolerance_rad=bound[1],
            position_residual_m=0.020, orientation_residual_rad=0.010,
            within_tolerance=False, convergence_budget="osc_servo_iterations",
            ticks_used=20, ticks_budget=20, budget_exhausted=True,
            execution_phase="local_control",
        )

    local = contract(
        LIBERO_ENDPOINT_ROUTE_LOCAL, libero_local_endpoint_tolerances(),
    )
    assert local["tolerance"] == {"position_m": 0.005, "orientation_rad": 0.05}
    assert local["position_shortfall_m"] == pytest.approx(0.015)
    assert local["achieved_capability"]["meets_advertised_tolerance"] is False
    # The same 20 mm residual is a 15 mm shortfall locally and a 5 mm shortfall
    # globally. Without the route identity a policy cannot tell them apart.
    globally = contract(
        LIBERO_ENDPOINT_ROUTE_GLOBAL, libero_global_endpoint_tolerances(),
    )
    assert globally["position_shortfall_m"] == pytest.approx(0.005)
    with pytest.raises(HybridMotionProfileError, match="is not the bound route"):
        contract(LIBERO_ENDPOINT_ROUTE_LOCAL, libero_global_endpoint_tolerances())
    with pytest.raises(HybridMotionProfileError, match="outside the applied bound"):
        libero_endpoint_contract(
            route=LIBERO_ENDPOINT_ROUTE_LOCAL,
            position_tolerance_m=0.005, orientation_tolerance_rad=0.05,
            position_residual_m=0.020, orientation_residual_rad=0.0,
            within_tolerance=True, convergence_budget="osc_servo_iterations",
        )


def test_global_receipt_names_the_bound_that_judged_the_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 20 mm miss must arrive with the bound it was measured against."""

    from ora.environment.controllers.hybrid_motion import (
        LIBERO_ENDPOINT_ROUTE_GLOBAL,
    )

    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += 0.020
    receipt = session._execute_planned_motion(_action())
    assert receipt["motion_status"] == "stopped"
    contract = receipt["endpoint_contract"]
    assert receipt["feedback"]["endpoint_contract"] == contract
    assert contract["execution_route"] == LIBERO_ENDPOINT_ROUTE_GLOBAL
    assert contract["tolerance"] == {
        "position_m": 0.015, "orientation_rad": math.radians(5),
    }
    assert contract["judged_residual"]["position_m"] == pytest.approx(0.020)
    assert contract["within_tolerance"] is False
    assert contract["position_shortfall_m"] == pytest.approx(0.005)
    # The terminal settle supplies positive evidence, so the miss is a bounded
    # budget exhaustion with its achieved capability, not an unknown.
    assert contract["convergence_status"] == "budget_exhausted"
    assert contract["achieved_capability"]["position_shortfall_m"] == pytest.approx(0.005)


def test_settle_exhaustion_discloses_achieved_capability_not_bare_not_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the terminal settle cannot close the last millimetres, the receipt
    reports the capability the route actually achieved."""

    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += 0.020
    receipt = session._execute_planned_motion(_action())
    execution = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert execution["terminal_settle"]["exhausted"] is True
    assert execution["terminal_settle"]["max_ticks"] == 40
    contract = receipt["endpoint_contract"]
    assert contract["convergence"] == {
        "budget": "terminal_settle", "ticks_used": 40, "ticks_budget": 40,
        "exhausted": True,
    }
    capability = contract["achieved_capability"]
    assert capability["limit"] == "convergence_budget_exhausted"
    assert capability["meets_advertised_tolerance"] is False
    assert capability["position_residual_m"] == pytest.approx(0.020)
    assert capability["position_shortfall_m"] == pytest.approx(0.005)
    assert "20.0 mm" in capability["summary"]
    assert "15.0 mm" in capability["summary"]


def test_sparse_projection_lifts_the_applied_bound_and_never_re_derives_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The projection maps a non-reached segment to its failure code without
    re-comparing, so the bound it shows the policy has to be the executed
    route's own record."""

    from ora.environment.controllers.hybrid_motion import (
        LIBERO_ENDPOINT_ROUTE_GLOBAL,
    )

    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += 0.020
    receipt = session._execute_plan_reference(_action(plan_ref=_probe(session)))
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_code"] == "terminal_tracking_timeout"
    contract = receipt["arms"]["right"]["endpoint_contract"]
    assert contract == receipt["segment"]["endpoint_contract"]
    assert contract["execution_route"] == LIBERO_ENDPOINT_ROUTE_GLOBAL
    assert contract["tolerance"]["position_m"] == 0.015
    assert contract["judged_residual"]["position_m"] == pytest.approx(0.020)


def test_reached_projection_still_names_the_bound_it_satisfied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrival carries the contract too; a bare "reached" hides which of the
    live bounds was cleared."""

    session, plant = _session(tmp_path, monkeypatch)
    plant.endpoint[0] += 0.014
    receipt = session._execute_plan_reference(_action(plan_ref=_probe(session)))
    assert receipt["execution_status"] == "reached"
    contract = receipt["arms"]["right"]["endpoint_contract"]
    assert contract["within_tolerance"] is True
    assert contract["tolerance"]["position_m"] == 0.015
    assert contract["achieved_capability"] is None


@pytest.mark.parametrize("stop", ["waypoint_tracking_timeout", "operator_stop_requested"])
def test_waypoint_stop_preserves_progress_without_claiming_endpoint_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    from ora.agent.native.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )
    session, plant = _session(tmp_path, monkeypatch)
    if stop == "operator_stop_requested":
        (tmp_path / "control").mkdir(exist_ok=True)
        (tmp_path / "control/stop.request").touch()
    else:
        plant.endpoint = list(START)
        step = plant.step
        def stalled_step(command: Any) -> Any:
            before = plant.robot._joint_positions.copy()
            result = step(command)
            plant.robot._joint_positions = before
            plant.pose = list(START)
            return result
        plant.step = stalled_step
    receipt = session._execute_plan_reference(_action(plan_ref=_probe(session)))
    contract = receipt["arms"]["right"]["endpoint_contract"]
    assert contract["execution_phase"] == "waypoint_tracking"
    assert contract["stop_reason"] == stop
    assert contract["convergence_status"] == "not_started"
    assert contract["achieved_capability"] is None
    assert contract["waypoints_reached"] < contract["waypoints_total"]
    assert contract["convergence"]["ticks_used"] == 0
    assert "endpoint convergence has not started" in contract["summary"]
    assert contract["judged_residual"]["position_m"] > 0.015
    context = ReceiptProjectionContext(last_action_id=None, last_action_public_name=None)
    project = resolve_receipt_effect_projector(receipt["schema"])
    projected = project(receipt, context)
    expected = {key: value for key, value in contract.items() if key != "schema"}
    assert projected["detail"]["arms"]["right"]["endpoint_contract"] == expected
    inner = receipt["segment"]
    project = resolve_receipt_effect_projector(inner["schema"])
    assert project(inner, context)["detail"]["endpoint_contract"] == expected


@pytest.mark.parametrize("phase,used,budget,exhausted,stop,expected", [
    ("unknown", None, None, False, None, "unknown"),
    ("terminal_settle", 0, 40, False, None, "unknown"),
    ("terminal_settle", 0, 40, True, None, "unknown"),
    ("terminal_settle", 5, 40, False, None, "unknown"),
    ("terminal_settle", 5, 40, False, "environment_terminated", "interrupted"),
    ("terminal_settle", 40, 40, True, "environment_terminated", "interrupted"),
    ("terminal_settle", 40, 40, True, "terminal_tracking_timeout", "budget_exhausted"),
    ("local_control", 20, 20, True, "servo_iteration_budget_exhausted", "budget_exhausted"),
])
def test_endpoint_convergence_requires_positive_execution_evidence(
    phase: str, used: Any, budget: Any, exhausted: bool, stop: Any, expected: str,
) -> None:
    from ora.environment.controllers.hybrid_motion import (
        LIBERO_ENDPOINT_ROUTE_GLOBAL, libero_endpoint_contract,
    )
    contract = libero_endpoint_contract(
        route=LIBERO_ENDPOINT_ROUTE_GLOBAL,
        position_tolerance_m=0.015, orientation_tolerance_rad=math.radians(5),
        position_residual_m=0.025, orientation_residual_rad=0.01,
        within_tolerance=False, convergence_budget="terminal_settle",
        ticks_used=used, ticks_budget=budget, budget_exhausted=exhausted,
        execution_phase=phase, stop_reason=stop,
    )
    assert contract["convergence_status"] == expected
    assert (contract["achieved_capability"] is not None) == (expected == "budget_exhausted")
    assert "controller_converged_outside_tolerance" not in json.dumps(contract)


def test_endpoint_missing_residual_cannot_claim_arrival_or_budget_capability() -> None:
    from ora.environment.controllers.hybrid_motion import (
        LIBERO_ENDPOINT_ROUTE_GLOBAL, libero_endpoint_contract,
    )
    contract = libero_endpoint_contract(
        route=LIBERO_ENDPOINT_ROUTE_GLOBAL,
        position_tolerance_m=0.015, orientation_tolerance_rad=math.radians(5),
        position_residual_m=None, orientation_residual_rad=None,
        within_tolerance=None, convergence_budget="terminal_settle",
        execution_phase="terminal_settle", ticks_used=40, ticks_budget=40,
        budget_exhausted=True, stop_reason="terminal_tracking_timeout",
    )
    assert contract["convergence_status"] == "unknown"
    assert contract["within_tolerance"] is None
    assert contract["achieved_capability"] is None
    assert "unavailable" in contract["summary"]


def test_gripper_acceptance_survives_sparse_receipt_wrapping(tmp_path, monkeypatch):
    from ora.agent.native.receipt_projection import resolve_receipt_effect_projector, ReceiptProjectionContext
    session, _ = _session(tmp_path, monkeypatch)
    session._sparse_receipt_transform = lambda inner: {
        'schema': 'ora.libero-sparse-motion-receipt.v1',
        'action': 'set_gripper', 'action_id': inner['action_id'],
        'state_changed': True, 'execution_status': 'not_reached', 'segment': inner,
        'arms': {}, 'target_arms': ['right'], 'synchronization': {'mode': 'single_arm'}}
    receipt = session._persist_receipt({'action': 'set_gripper', 'action_id': 'gripper-bound',
        'arguments': {'gripper': 0.}, 'reached_gripper': .03, 'steps_used': 1, 'state_changed': True})
    check = receipt['acceptance_checks'][0]
    assert check['comparison'] == '<'
    assert check['tolerance']['opening_normalized'] == .03
    assert check['arms']['right']['within_tolerance']['opening_normalized'] is False
    context = ReceiptProjectionContext(None, 'set_gripper', lambda _: True)
    public = resolve_receipt_effect_projector(receipt['schema'])(receipt, context)['detail']
    # Every capability is on: every field of the check survives delivery.
    assert public['acceptance_checks'] == [check]
    assert public['gripper_targets_effective'] == {'right': 0.}
