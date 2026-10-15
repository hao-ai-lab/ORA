"""Environment, policy surface, planner and end-to-end Runtime contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import pytest
import scipy
from scipy.spatial.transform import Rotation
from PIL import Image

from ora.environment.controller_replay import load_controller_replay_manifest
from ora.orchestrator.evaluation import (
    EvaluationError, assess_learning_eligibility, evaluate, export, inventory,
    seal,
)
from ora.environment.workspace_profiles import resolve_workspace_profile
from ora.environment.factory import (
    EnvironmentFactoryError,
    build_environment_session,
)
from ora.orchestrator.metrics import build_runtime_metrics, verify_runtime_metrics, write_runtime_metrics
from ora.orchestrator.spec import file_sha256
from ora.orchestrator.preflight import _libero_probe_matches_plan
from ora.orchestrator.stagnation import StagnationGuard
from ora.orchestrator.coordinator import (
    SupervisorError,
    _main_observation_image,
    _quiesce_policy_after_fault,
    _qualified_runtime_binding,
    _reconciled_provider_model_calls,
    run_episode,
)
from ora.orchestrator.action_executor import _motion_reached
from ora.agent.context import ContextProjector
from ora.environment import (
    ENVIRONMENT_STATUS_SCHEMA,
    EnvironmentContractError,
    EnvironmentSession,
    LocalRpcEnvironmentSession,
    SyntheticSession,
    validate_environment_status,
    validate_environment_session,
)
from ora.environment.base import validate_restartable_environment_session
from ora.environment.rpc import _worker_environment
from ora.agent.tools.robot import TOOL_DEFINITIONS, RobotToolRuntime, ToolError
from ora.agent.evidence.policy_state import PolicyStateStore, load_policy_state, verify_policy_state
from ora.contracts.action import ActionValidationError, validate_runtime_action
from ora.contracts.workspace import (
    is_monotonic_workspace_recovery,
    workspace_state,
)
from ora.environment.backends.robotwin.session import (
    RoboTwinLiveError,
    RoboTwinLiveSession,
)
from ora.environment.backends.libero_pro.session import (
    _benchmark_data_binding,
    _build_environment,
    _inverse_controller_scale,
    LiberoProError,
    LiberoProSession,
    libero_environment_init_params,
    libero_osc_action_contract,
    libero_public_environment_context,
)
from ora.environment.backends.libero_pro.registration import (
    CORE_SUITES,
    ensure_perturbation_suites,
)
from ora.environment.benchmarks.libero import (
    LIBERO_BASE_SUITES,
    LIBERO_BASE_SUITE_TASK_COUNTS,
    LiberoBenchmarkError,
    validate_base_suite_task,
)
from ora.environment.benchmarks.libero_pro import (
    BASE_SUITES as LIBERO_PRO_BASE_SUITES,
    RUNTIME_SUITES as LIBERO_RUNTIME_SUITES,
)
from ora.environment.backends.robotwin.qualification import selected_environment_assets


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "ora/policy/core/thin-react-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENT_RULES = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def test_locked_policy_coding_stack_is_importable() -> None:
    assert np.__version__
    assert cv2.__version__
    assert scipy.__version__


def publish_synthetic(tmp_path: Path):
    source = tmp_path / "source"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    observation, _ = session.observe(source)
    workspace = tmp_path / "sandbox/workspace"
    control = tmp_path / "control"
    state_store = PolicyStateStore(tmp_path, "test-run", "test-plan")
    snapshot = state_store.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        cause="reset",
        task_instruction="Move safely once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
    )
    state_record = state_store.steps[-1]
    projector = ContextProjector(workspace, control, PROMPT, MANUAL, AGENT_RULES)
    pointer = projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="Move safely once.",
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
        policy_state={
            "step": snapshot["step"], "path": state_record["path"],
            "sha256": state_record["sha256"],
            "state_sha256": snapshot["state_sha256"],
        },
    )
    tools = RobotToolRuntime(
        control, MANUAL, tmp_path / "events.jsonl", BOUNDS,
        state_root=state_store.root,
    )
    return observation, pointer, workspace, control, tools, state_store


def test_environment_session_contract_is_transport_neutral(tmp_path: Path) -> None:
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    assert validate_environment_session(session) is session
    assert validate_environment_session(session) is session

    class MissingVerdictBackend:
        def start(self): ...
        def observe(self, destination): ...
        def execute(self, action): ...
        def status(self): return {}
        @property
        def is_terminal(self): return False
        def abort(self, reason): ...
        def finalize_operator_video(self): ...
        def close(self): ...

    with pytest.raises(EnvironmentContractError, match="persist_official_verdict"):
        validate_environment_session(MissingVerdictBackend())

    leaked = {
        "schema": ENVIRONMENT_STATUS_SCHEMA,
        "phase": "terminal", "terminal": True, "reason": "lost_object",
        "source": "test_environment", "policy_visibility": "control_only",
        "signals": [], "object_position_m": [0.1, 0.2, 0.3],
    }
    with pytest.raises(EnvironmentContractError, match="invalid projection"):
        validate_environment_status(leaked)


def test_control_factory_owns_backend_selection(tmp_path: Path, monkeypatch) -> None:
    session = build_environment_session(
        runtime_config={"kind": "synthetic"},
        output_root=tmp_path / "native",
        task="synthetic_tool_boundary",
        seed=0,
        max_sim_steps=1,
    )
    assert isinstance(session, EnvironmentSession)
    assert isinstance(session.delegate, SyntheticSession)

    with pytest.raises(EnvironmentFactoryError, match="unsupported"):
        build_environment_session(
            runtime_config={"kind": "unknown"},
            output_root=tmp_path / "unknown",
            task="task",
            seed=0,
            max_sim_steps=1,
        )

    with pytest.raises(EnvironmentFactoryError, match="transport"):
        build_environment_session(
            runtime_config={"kind": "synthetic", "transport": "remote_rpc"},
            output_root=tmp_path / "remote",
            task="task",
            seed=0,
            max_sim_steps=1,
        )

    monkeypatch.delenv("TEST_ROBOTWIN_ROOT", raising=False)
    with pytest.raises(EnvironmentFactoryError, match="missing TEST_ROBOTWIN_ROOT"):
        build_environment_session(
            runtime_config={
                "kind": "robotwin",
                "transport": "local_rpc",
                "root_env": "TEST_ROBOTWIN_ROOT",
                "python_env": "TEST_ROBOTWIN_PYTHON",
                "task_config": "demo_randomized",
                "required_revision": "revision",
                "workspace_bounds_m": BOUNDS,
            },
            output_root=tmp_path / "robotwin",
            task="adjust_bottle",
            seed=0,
            max_sim_steps=1,
        )

    monkeypatch.setenv("TEST_LIBERO_PYTHON", sys.executable)
    monkeypatch.delenv("TEST_LIBERO_ROOT", raising=False)
    with pytest.raises(EnvironmentFactoryError, match="missing TEST_LIBERO_ROOT"):
        build_environment_session(
            runtime_config={
                "kind": "libero_pro", "transport": "local_rpc",
                "python_env": "TEST_LIBERO_PYTHON",
                "libero_pro_root_env": "TEST_LIBERO_ROOT",
                "data_root_env": "TEST_LIBERO_DATA_ROOT",
                "required_libero_pro_revision": "libero-revision",
                "required_data_revision": "data-revision", "suite": "libero_goal_task",
                "task_id": 0, "expected_task_language": "put the bowl on the plate",
                "workspace_bounds_m": {
                    "x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3],
                }, "startup_timeout_s": 10, "request_timeout_s": 10,
                "servo_max_steps": 80,
            },
            output_root=tmp_path / "libero-pro",
            task="libero_goal_task-t00", seed=0, max_sim_steps=100,
        )


def test_environment_factory_preserves_virtualenv_python_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_python = tmp_path / "runtime/.venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    roots = {
        "TEST_LIBERO_ROOT": tmp_path / "LIBERO-PRO",
        "TEST_LIBERO_DATA_ROOT": tmp_path / "LIBERO-Pro-data",
    }
    monkeypatch.setenv("TEST_LIBERO_PYTHON", str(venv_python))
    for name, root in roots.items():
        root.mkdir()
        monkeypatch.setenv(name, str(root))
    session = build_environment_session(
        runtime_config={
            "kind": "libero_pro", "transport": "local_rpc",
            "python_env": "TEST_LIBERO_PYTHON",
            "libero_pro_root_env": "TEST_LIBERO_ROOT",
            "data_root_env": "TEST_LIBERO_DATA_ROOT",
            "required_libero_pro_revision": "libero-revision",
            "required_data_revision": "data-revision",
            "suite": "libero_goal_task", "task_id": 0,
            "expected_task_language": "put the bowl on the plate",
            "workspace_bounds_m": {
                "x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3],
            },
            "startup_timeout_s": 10, "request_timeout_s": 10,
            "servo_max_steps": 80,
        },
        output_root=tmp_path / "output",
        task="libero_goal_task-t00", seed=0, max_sim_steps=100,
    )
    assert isinstance(session.delegate, LocalRpcEnvironmentSession)
    assert session.delegate.python_executable == venv_python.absolute()
    assert session.delegate.python_executable.resolve() == Path(sys.executable).resolve()


def test_local_rpc_environment_runs_in_an_isolated_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOULD_NOT_REACH_ENVIRONMENT_WORKER", "secret")
    worker_environment = _worker_environment(ROOT)
    assert "SHOULD_NOT_REACH_ENVIRONMENT_WORKER" not in worker_environment
    assert worker_environment["PYTHONPATH"] == str(ROOT.resolve())
    assert worker_environment["PYTHONNOUSERSITE"] == "1"
    monkeypatch.setenv("LIBERO_TYPE", "standard")
    assert _worker_environment(ROOT)["LIBERO_TYPE"] == "standard"
    native = tmp_path / "native"
    session = LocalRpcEnvironmentSession(
        python_executable=Path(sys.executable),
        project_root=ROOT,
        output_root=native,
        socket_root=tmp_path / "state",
        worker_config={
            "backend": "synthetic",
            "output_root": str(native),
            "task": "rpc_contract",
            "seed": 7,
        },
        startup_timeout_s=10,
        request_timeout_s=10,
    )
    assert validate_environment_session(session) is session
    try:
        reset = session.start()
        assert reset["task"] == "rpc_contract"
        record = json.loads((native / "environment-session.json").read_text())
        assert record["transport"] == "local_rpc"
        assert record["worker_pid"] != os.getpid()
        assert record["state"] == "running"
        assert record["python_executable"] == os.path.abspath(sys.executable)
        observation, images = session.observe(tmp_path / "policy-observations")
        assert observation["schema"] == "ora-observation.v1"
        assert images[0].is_file()
        assert session.status() == {
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "running", "terminal": False, "reason": None,
            "source": "synthetic_session", "policy_visibility": "control_only",
            "signals": [],
        }
        action = {
            "schema": "ora-action.v3",
            "observation_id": observation["observation_id"],
            "action_id": "rpc-action-0",
            "action": "move_pose",
            "bound_observation_sha256": "test-observation-sha",
            "arguments": {
                "arm": "left",
                "position_m": [0.0, 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        assert session.execute(action)["receipt"]["state_changed"] is True
        assert session.is_terminal is True
        assert session.persist_official_verdict()["official_success"] is None
        round_reset = session.restart_round(1, seed=11)
        assert round_reset["round_index"] == 1
        assert round_reset["seed"] == 11
        assert session.status()["terminal"] is False
        restarted, _ = session.observe(tmp_path / "policy-observations")
        assert restarted["sequence"] == 1
        assert restarted["previous_execution_receipt"] is None
        record = json.loads((native / "environment-session.json").read_text())
        assert record["round_index"] == 1
        assert record["seed"] == 11
    finally:
        session.close()
    closed = json.loads((native / "environment-session.json").read_text())
    assert closed["state"] == "closed"
    assert closed["worker_returncode"] == 0
    assert not Path(closed["socket"]).exists()


def test_adjust_bottle_asset_identity_uses_constant_model_family(tmp_path: Path) -> None:
    model = tmp_path / "assets/objects/001_bottle"
    model.mkdir(parents=True)
    (model / "model_data13.json").write_text("{}", encoding="utf-8")

    class AdjustBottleReset:
        model_id = 13
        wall_texture = None
        table_texture = None

    assets = selected_environment_assets(
        AdjustBottleReset(), robotwin_root=tmp_path, task="adjust_bottle"
    )
    selected = assets["selected_object_models"][0]
    assert selected["model_name"] == "001_bottle"
    assert selected["model_id"] == 13


def synthetic_plan() -> dict:
    return {
        "schema": "ora.run-plan.v1",
        "plan_sha256": "test-plan", "plan_id": "runtime-v1-m0-test",
        "experiment": "runtime-v1", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/runtime-v1",
        "task": "synthetic_tool_boundary", "task_instruction": "Move safely once.",
        "seed": 0, "evidence_tier": "diagnostic",
        "policy": {
            "kind": "scripted", "model": "scripted-policy-v1",
            "provider": "trusted_scripted", "reasoning_effort": "none",
            "actions": [{"action": "move_pose", "arm": "left", "position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]}],
        },
        "runtime": {"kind": "synthetic", "workspace_bounds_m": BOUNDS},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 2, "max_sim_steps": 5, "wall_time_s": 30},
        "policy_context": {
            "codex_project": [{"path": "ora/policy/planners/native/project-v1.md", "sha256": file_sha256(AGENT_RULES)}],
            "episode_prefix": [{"path": "ora/policy/core/thin-react-v1.md", "sha256": file_sha256(PROMPT)}],
            "tool_manual": [{"path": "ora/policy/embodiments/aloha-agilex/manual-v1.md", "sha256": file_sha256(MANUAL)}],
            "composition_sha256": "test",
        },
        "source": {"commit": "test", "dirty": False}, "qualification": {"required": False},
    }


class FakeRobot:
    def left_plan_path(self, *_args, **_kwargs):
        return {"status": "Success", "position": np.zeros((2, 7))}

    def right_plan_path(self, *_args, **_kwargs):
        return {"status": "Success", "position": np.zeros((2, 7))}

    def get_left_gripper_val(self):
        return 1.0

    def get_right_gripper_val(self):
        return 1.0


class FakeRoboTwinEnvironment:
    def __init__(self):
        self.robot = FakeRobot()
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 100
        self.poses = {
            "left": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "right": [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }

    def setup_demo(self, **_kwargs):
        return None

    def check_success(self):
        return self.eval_success

    def get_arm_pose(self, arm):
        return self.poses[arm]

    def get_obs(self):
        return {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
                    "depth": np.full((3, 4), 1000.0, dtype=np.float32),
                    "intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4),
                    "cam2world_gl": np.eye(4),
                }
            },
            "endpose": {
                "left_endpose": self.poses["left"], "left_gripper": 1.0,
                "right_endpose": self.poses["right"], "right_gripper": 1.0,
            },
            "joint_action": {"left_arm": [0] * 6, "left_gripper": 1.0, "right_arm": [0] * 6, "right_gripper": 1.0},
        }

    def take_action(self, ee_action, action_type):
        assert action_type == "ee"
        self.poses["left"] = list(ee_action[:7])
        self.take_action_cnt += 1
        self.eval_success = True

    def close_env(self, clear_cache=False):
        assert clear_cache is False


class FakeTaskActor:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def get_functional_point(self, _index):
        return self.position.copy()


class FakeLostBottleEnvironment(FakeRoboTwinEnvironment):
    def __init__(self):
        super().__init__()
        self.bottle = FakeTaskActor([-0.1, -0.1, 0.752])

    def take_action(self, ee_action, action_type):
        assert action_type == "ee"
        self.poses["left"] = list(ee_action[:7])
        self.take_action_cnt += 1
        self.bottle.position[2] = 0.65


def test_live_session_emits_metric_observation_motion_and_boolean_verdict(tmp_path: Path) -> None:
    environment = FakeRoboTwinEnvironment()
    session = RoboTwinLiveSession(
        robotwin_root=tmp_path / "RoboTwin", output_root=tmp_path / "native",
        task="click_alarmclock", actual_seed=100, episode=0, operator_world_video=False,
        task_factory=lambda _task: environment,
        task_args_factory=lambda **_kwargs: {},
    )
    assert validate_environment_session(session) is session
    reset = session.start()
    assert reset["official_predicate_at_reset"] is False
    observation, _ = session.observe(tmp_path / "observations")
    camera = observation["cameras"]["head_camera"]
    assert observation["encoding"] == "metric_rgbd_v1"
    assert set(camera["files"]) == {"rgb_lossless_png", "depth_numpy_m_f32", "calibration"}
    action = {
        "schema": "ora-action.v3", "observation_id": observation["observation_id"],
        "action_id": "action-1", "action": "move_pose", "bound_observation_sha256": "bound",
        "arguments": {"arm": "left", "position_m": [0.02, 0.0, 1.0], "orientation_xyzw": [0, 0, 0, 1]},
    }
    receipt = session.execute(action)
    assert receipt["receipt"]["state_changed"] is True
    assert session.is_terminal is True
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True
    assert verdict["source"] == "RoboTwin.eval_success"
    session.close()


def test_robotwin_recoverability_signal_stops_lost_task_object_without_gt_leak(
    tmp_path: Path,
) -> None:
    environment = FakeLostBottleEnvironment()
    session = RoboTwinLiveSession(
        robotwin_root=tmp_path / "RoboTwin", output_root=tmp_path / "native",
        task="adjust_bottle", actual_seed=100, episode=0,
        operator_world_video=False,
        environment_signal_policy="task_recoverability_v1",
        task_factory=lambda _task: environment,
        task_args_factory=lambda **_kwargs: {},
    )
    session.start()
    assert session.status()["phase"] == "running"
    observation, _ = session.observe(tmp_path / "observations")
    session.execute({
        "schema": "ora-action.v3",
        "observation_id": observation["observation_id"],
        "action_id": "lose-bottle", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "left", "position_m": [0.02, 0.0, 1.0],
            "orientation_xyzw": [0, 0, 0, 1],
        },
    })
    status = session.status()
    assert status["terminal"] is True
    assert status["reason"] == "task_object_out_of_recoverable_workspace"
    assert status["signals"] == [{
        "code": "task_object_out_of_recoverable_workspace",
        "severity": "terminal",
        "source": "robotwin.adjust_bottle.recoverability_v1",
        "policy_visibility": "control_only",
    }]
    assert "position" not in json.dumps(status)
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False
    assert verdict["trusted_abort_reason"] == status["reason"]
    session.close()


class RestartableRoboTwinEnvironment(FakeRoboTwinEnvironment):
    """A RoboTwin double whose per-round instance records its own reset seed."""

    def __init__(self):
        super().__init__()
        self.reset_seed = None
        self.closed = False

    def setup_demo(self, **kwargs):
        self.reset_seed = kwargs.get("seed")
        return None

    def take_action(self, ee_action, action_type):
        assert action_type == "ee"
        self.poses["left"] = list(ee_action[:7])
        self.take_action_cnt += 1

    def close_env(self, clear_cache=False):
        assert clear_cache is False
        self.closed = True


def _robotwin_move_pose(observation, action_id: str, x: float) -> dict:
    return {
        "schema": "ora-action.v3",
        "observation_id": observation["observation_id"],
        "action_id": action_id,
        "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "left",
            "position_m": [x, 0.0, 1.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_robotwin_round_reset_rebuilds_the_simulator_without_leaking_episode_state(
    tmp_path: Path,
) -> None:
    built: list[RestartableRoboTwinEnvironment] = []

    def task_factory(_task):
        environment = RestartableRoboTwinEnvironment()
        built.append(environment)
        return environment

    native = tmp_path / "native"
    session = RoboTwinLiveSession(
        robotwin_root=tmp_path / "RoboTwin", output_root=native,
        task="click_alarmclock", actual_seed=100, episode=0,
        operator_world_video=False,
        task_factory=task_factory,
        task_args_factory=lambda **_kwargs: {},
    )
    assert validate_restartable_environment_session(
        validate_environment_session(session)
    ) is session

    reset = session.start()
    assert reset["round_index"] == 0
    # The Supervisor validates round 0 under the same shared seed key.
    assert reset["seed"] == 100
    assert built[0].reset_seed == 100

    observation, _ = session.observe(tmp_path / "observations")
    receipt = session.execute(_robotwin_move_pose(observation, "round-0", 0.02))
    assert receipt["receipt"]["state_changed"] is True
    assert built[0].take_action_cnt == 1

    # Poison every per-episode field a later round must not inherit.
    session.plan_references["plan-from-round-0"] = {"stale": True}
    session.pinocchio_models["left"] = {"entity": "round-0-articulation"}
    session.gripper_command_targets["left"] = "close"
    session.gripper_command_targets["right"] = "close"
    session.last_safe_pose = {
        arm: [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0] for arm in ("left", "right")
    }
    session.last_safe_joints = {arm: [9.0] * 6 for arm in ("left", "right")}
    session.home_joints = {arm: [9.0] * 6 for arm in ("left", "right")}
    session.world_mutation_sequence = 3
    session.world_mutation_actions = ["move_pose", "set_gripper", "move_pose"]

    session.abort("policy_finished_episode")
    assert session.persist_official_verdict()["official_success"] is False

    with pytest.raises(
        RoboTwinLiveError, match="previous round verdict must be archived"
    ):
        session.restart_round(1)
    (native / "official-verdict.json").unlink()
    for skipped in (0, 2, 5):
        with pytest.raises(RoboTwinLiveError, match="advance by exactly one"):
            session.restart_round(skipped)
    assert session.round_index == 0

    round_reset = session.restart_round(1)
    assert round_reset["schema"] == "ora.robotwin-round-reset.v1"
    assert round_reset["round_index"] == 1
    assert round_reset["seed"] == 100
    assert round_reset["official_predicate_at_reset"] is False
    assert (native / "round-resets/round-0001.json").is_file()

    # The previous round's simulator is closed and a new one is built at the
    # same seed, so no object pose can survive the round boundary.
    assert built[0].closed is True
    assert len(built) == 2
    assert session.environment is built[1]
    assert built[1].reset_seed == 100
    assert built[1].take_action_cnt == 0

    # Cross-round leak regression: stale planning, FK and gripper state must
    # all be gone.
    assert session.plan_references == {}
    assert session.pinocchio_models == {}
    assert session.gripper_command_targets == {
        "left": "unknown", "right": "unknown",
    }
    assert session.world_mutation_sequence == 0
    assert session.world_mutation_actions == []
    assert session.last_receipt is None
    # abort_reason is RoboTwin's only stop channel: its motion_transaction and
    # local servo executors keep the abort in a transaction local, and no
    # backend file is polled per control tick. If that ever changes, this
    # assertion and restart_round must both grow the file.
    assert session.abort_reason is None
    assert not (native / "control").exists()
    # Backtrack targets are refreshed from the new round's measurement rather
    # than cleared, on both arms, so recover_pose keeps a valid target.
    assert session.last_safe_pose == {
        "left": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "right": [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    assert session.last_safe_joints == {}
    assert session.home_joints == {}
    assert session.status()["terminal"] is False
    assert session.semantic_sim_steps == 0

    # Artifact sequences stay session-global so round 1 cannot overwrite the
    # round 0 evidence that shares the same output root.
    assert session.observation_sequence == 1
    assert session.action_sequence == 1
    assert round_reset["artifact_sequences_preserved"] == {
        "observation_sequence": 1,
        "action_sequence": 1,
        "motion_probe_sequence": 0,
        "operator_frame_sequence": 0,
        "operator_execution_frame_sequence": 0,
    }

    restarted, _ = session.observe(tmp_path / "observations")
    assert restarted["sequence"] == 1
    assert restarted["previous_execution_receipt"] is None

    # Positive evidence: the global motion path still executes in round 1. A
    # stale stop signal would leave a "planned but zero steps" receipt behind.
    round_one = session.execute(_robotwin_move_pose(restarted, "round-1", 0.05))
    assert round_one["receipt"]["state_changed"] is True
    assert built[1].take_action_cnt == 1
    assert built[1].poses["left"][0] == pytest.approx(0.05)
    assert "operator_stop_requested" not in json.dumps(round_one)
    assert (native / "receipts/action-0000.json").is_file()
    assert (native / "receipts/action-0001.json").is_file()

    replay = load_controller_replay_manifest(native / "controller-replay/manifest.json")
    assert replay["round_index"] == 1

    session.abort("policy_finished_episode")
    session.persist_official_verdict()
    (native / "official-verdict.json").unlink()
    seed_reset = session.restart_round(2, seed=4)
    assert seed_reset["seed"] == 4
    assert seed_reset["round_index"] == 2
    assert session.actual_seed == 4
    assert built[2].reset_seed == 4
    assert built[1].closed is True

    session.close()
    assert built[2].closed is True


class FakeLiberoEnvironment:
    def __init__(self):
        self.task_descriptions = ["put the bowl on the plate"]
        self.last_action = None
        self.closed = False
        self.step_count = 0
        self.gripper_hold_directive = -1.0
        self.current_raw_obs = {
            "agentview_image": np.zeros((1, 3, 4, 3), dtype=np.uint8),
            "agentview_depth": np.ones((1, 3, 4, 1), dtype=np.float32),
            "robot0_eye_in_hand_image": np.zeros((1, 3, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_depth": np.ones((1, 3, 4, 1), dtype=np.float32),
            "robot0_eef_pos": np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
            "robot0_eef_quat": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([[0.04, -0.04]], dtype=np.float32),
        }
        calibration = {
            "intrinsic_K": np.eye(3), "extrinsic_cam2world": np.eye(4),
            "depth_near": 0.1, "depth_far": 10.0, "height": 3, "width": 4,
        }

        class VectorEnvironment:
            def get_env_attr(self, name):
                if name == "harness_camera_meta":
                    return [{
                        "agentview": calibration,
                        "robot0_eye_in_hand": calibration,
                    }]
                if name == "harness_motion_state":
                    return [{
                        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "external_wrench_clipped": None,
                        "external_force_magnitude_n": None,
                    }]
                if name == "harness_gripper_hold_directive":
                    return [-1.0]
                raise AssertionError(name)

        self.env = VectorEnvironment()

    def get_grip_site_pose(self):
        raw = self.current_raw_obs
        selected = raw[0] if isinstance(raw, np.ndarray) else {
            key: value[0] for key, value in raw.items()
        }
        return {
            "frame": "robosuite_grip_site",
            "site_name": "gripper0_grip_site",
            "position_m": selected["robot0_eef_pos"],
            "orientation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    def get_gripper_hold_directive(self):
        return self.gripper_hold_directive

    def reset(self, reset_state_ids=None):
        self.last_reset_state_ids = (
            None if reset_state_ids is None
            else np.asarray(reset_state_ids).tolist()
        )
        return {}, {}

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        delta = self.last_action[0, :3] * 0.05
        if isinstance(self.current_raw_obs, np.ndarray):
            selected = dict(self.current_raw_obs[0])
        else:
            selected = {
                key: value[0].copy() for key, value in self.current_raw_obs.items()
            }
        selected["robot0_eef_pos"] += delta
        if self.last_action[0, 6] > 0:
            self.gripper_hold_directive = 1.0
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.0, 0.0], dtype=np.float32,
            )
        elif self.last_action[0, 6] < 0:
            self.gripper_hold_directive = -1.0
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.04, -0.04], dtype=np.float32,
            )
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([1.0]), np.asarray([self.step_count >= 2]), np.asarray([False]), {}

    def close(self):
        self.closed = True


class DivergingLiberoEnvironment(FakeLiberoEnvironment):
    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        selected["robot0_eef_pos"] = np.asarray([-0.9, 0.0, 0.5], dtype=np.float32)
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class CrossAxisDriftingLiberoEnvironment(FakeLiberoEnvironment):
    """Progress along X while accumulating a non-terminal lateral drift."""

    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        selected["robot0_eef_pos"][1] += 0.035
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class GraduallyDivergingLiberoEnvironment(FakeLiberoEnvironment):
    """Move away in bounded increments without a one-tick corruption jump."""

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        selected = {
            key: value[0].copy() for key, value in self.current_raw_obs.items()
        } if not isinstance(self.current_raw_obs, np.ndarray) else dict(
            self.current_raw_obs[0]
        )
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        selected["robot0_eef_pos"][0] -= 0.03
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class GraduallyRotatingAwayLiberoEnvironment(FakeLiberoEnvironment):
    """Accumulate bounded orientation error without corrupting one tick."""

    def __init__(self):
        super().__init__()
        self.orientation_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])

    def get_grip_site_pose(self):
        value = super().get_grip_site_pose()
        value["orientation_xyzw"] = self.orientation_xyzw.copy()
        return value

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        self.orientation_xyzw = (
            Rotation.from_rotvec([0.0, 0.0, 0.15])
            * Rotation.from_quat(self.orientation_xyzw)
        ).as_quat()
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class GripperContactMotionEnvironment(FakeLiberoEnvironment):
    """Block closure while contact moves the EEF in bounded increments."""

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        selected = {
            key: value[0].copy() for key, value in self.current_raw_obs.items()
        } if not isinstance(self.current_raw_obs, np.ndarray) else dict(
            self.current_raw_obs[0]
        )
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        selected["robot0_eef_pos"][0] += 0.03
        selected["robot0_gripper_qpos"] = np.asarray(
            [0.024, -0.024], dtype=np.float32,
        )
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class CurvedProgressLiberoEnvironment(FakeLiberoEnvironment):
    """Follow a bounded curved path that still reaches the requested target."""

    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        lateral_delta = {1: 0.025, 2: 0.0, 3: -0.025}.get(
            self.step_count, 0.0,
        )
        selected["robot0_eef_pos"][1] += lateral_delta
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class ContactPerturbedLiberoEnvironment(FakeLiberoEnvironment):
    """Deflect one finger without changing the pose-only command channel."""

    def step(self, action):
        observation, reward, term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        if float(np.asarray(action)[0, 6]) == 0.0:
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.032, -0.04], dtype=np.float32,
            )
            self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, term, trunc, info


class ContactBlockedGripperEnvironment(FakeLiberoEnvironment):
    """Keep a close directive while contact prevents full finger closure."""

    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        if float(np.asarray(action)[0, 6]) > 0.0:
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.024, -0.024], dtype=np.float32,
            )
            self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class RotatingLiberoEnvironment(FakeLiberoEnvironment):
    """Apply RoboSuite's configured physical rotation range end to end."""

    def __init__(self):
        super().__init__()
        self.orientation_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])
        self.actions: list[np.ndarray] = []

    def get_grip_site_pose(self):
        value = super().get_grip_site_pose()
        value["orientation_xyzw"] = self.orientation_xyzw.copy()
        return value

    def step(self, action):
        normalized = np.asarray(action, dtype=np.float64)
        self.actions.append(normalized.copy())
        physical_rotvec = normalized[0, 3:6] * 0.5
        self.orientation_xyzw = (
            Rotation.from_rotvec(physical_rotvec)
            * Rotation.from_quat(self.orientation_xyzw)
        ).as_quat()
        return super().step(action)


def _fake_libero_metadata(tmp_path: Path, public_environment_context: dict) -> dict:
    return {
        "suite": "libero_goal_task", "task_id": 0,
        "task_language": "put the bowl on the plate", "task_name": "task",
        "bddl_file": str(tmp_path / "hidden.bddl"),
        "init_states_file": "hidden.pruned_init", "reset_id": 0,
        "trial_count": 50,
        "controller_action_contract": libero_osc_action_contract({
            "control_delta": True,
            "input_min": -1, "input_max": 1,
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        }),
        "public_environment_context": public_environment_context,
    }


def test_qualified_simulator_slots_share_one_device() -> None:
    """Five LIBERO sessions per card: the lease is per slot, not per device.

    A measured LIBERO worker holds about 0.7 GiB and near-zero SM time on a
    48 GiB card, so device residency is not the scarce resource. Unprofiled
    and physical plans stay host-exclusive, and one slot still admits one run.
    """

    from ora.orchestrator.resource_admission import _resource_refusal

    def _resource_conflict(binding, held):
        return _resource_refusal(binding, held, inventory={"0": {"total_mib": 48000, "free_mib": 40000, "used_mib": 8000}, "1": {"total_mib": 48000, "free_mib": 40000, "used_mib": 8000}})

    def binding(slot: str, device: str, resource_class: str = "libero_pro_sim"):
        return {
            "schema": "ora.execution-resource-binding.v1",
            "resource_class": resource_class,
            "slot_id": slot,
            **({"minimum_free_memory_mib": 2000} if resource_class == "libero_pro_sim" else {}),
            "worker_environment": {
                "CUDA_VISIBLE_DEVICES": device,
                "MUJOCO_EGL_DEVICE_ID": device,
            },
        }

    held = [binding("gpu0", "0")]
    # A second qualified slot on the same physical card is admitted.
    assert _resource_conflict(binding("gpu0b", "0"), held) is None
    # The same slot is not.
    assert _resource_conflict(binding("gpu0", "0"), held) is not None
    # A different card is obviously fine.
    assert _resource_conflict(binding("gpu1", "1"), held) is None
    # Host-exclusive plans still refuse to coexist with anything.
    assert _resource_conflict(
        binding("host", "0", resource_class="exclusive_host"), held,
    ) is not None
    assert _resource_conflict(
        binding("gpu0b", "0"),
        [binding("host", "0", resource_class="exclusive_host")],
    ) is not None
    # Cross-class co-residency requires declared memory floors; the service
    # binding intentionally has none and therefore must fail closed.
    assert _resource_conflict(
        binding("gpu0x", "0", resource_class="learned_model_service"), held,
    ) is not None


def test_terminal_operator_stop_is_not_pollution() -> None:
    """Only a stop with later physical actions marks stale-stop pollution."""

    def events(stop_index: int) -> list[dict]:
        records = []
        for index in range(2):
            record = {"schema": "receipt", "feedback": {}}
            if index == stop_index:
                record["failure_code"] = "operator_stop_requested"
            records.append({"type": "motion_receipt", "record": record})
        return records

    plan = {"runtime": {"kind": "libero_pro"}}
    result = {"state": "completed"}
    stale = assess_learning_eligibility(plan, result, events(0))
    terminal = assess_learning_eligibility(plan, result, events(1))
    assert any(
        reason.startswith("operator_stop_execution")
        for reason in stale["reasons"]
    )
    assert not any(
        reason.startswith("operator_stop_execution")
        for reason in terminal["reasons"]
    )


def test_grip_site_frame_is_rejected_off_the_panda_surface() -> None:
    """The Panda-private frame name must fail closed on other embodiments."""

    from ora.agent.tools.runtime.core import (
        ToolError, _project_copyable_grasp_approach_targets,
    )

    def result() -> dict:
        return {"candidates": [{
            "approach_delta": {
                "frame": "grip_site",
                "translation_m": [0.0, 0.0, -0.05],
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
        }]}

    aliased = result()
    _project_copyable_grasp_approach_targets(
        aliased, allow_grip_site_alias=True,
    )
    assert aliased["candidates"][0]["approach_delta"]["position"]["frame"] == (
        "end_effector"
    )
    try:
        _project_copyable_grasp_approach_targets(
            result(), allow_grip_site_alias=False,
        )
    except ToolError as error:
        assert error.code == "grasp_approach_contract"
    else:
        raise AssertionError("grip_site must be rejected off the Panda surface")
