from __future__ import annotations

import base64
import io
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ora.environment.controller_replay import load_controller_replay_manifest
from ora.agent.composition import resolve_tool_surface
from ora.agent.evidence.trajectory import _forbidden_path
from ora.agent.tools.runtime.core import (
    ManipulationToolRuntime,
    _grasp_hover_offset_m,
)
from ora.agent.tools.robot import ToolError
from ora.agent.native.session import NativePlanner
from ora.agent.tools.grasp import (
    AlohaHybridGraspProposer,
    ALOHA_END_EFFECTOR_COMPILER_IDENTITIES,
    ALOHA_GRIP_CENTER_COMPILER_IDENTITIES,
    ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES,
    CONTACT_GRASPNET_MODEL_IDENTITY,
    PcaTopDownGraspProposer,
    SAM3_MODEL_IDENTITY,
    ContactGraspNetApiProposer,
    Sam3ApiSegmenter,
    _compile_grasp_candidate,
    _public_grasp_compiler_identity,
    grasp_plan_public_result,
)
from ora.contracts import (
    MOTION_ACTION_V3_SCHEMA,
    compile_enhanced_move_pose,
)
from ora.contracts.pose import quaternion_matrix
from ora.environment.backends.robotwin import session as robotwin_session
from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
from ora.environment.backends.robotwin.planning import (
    planner_public_summary,
)
from ora.environment.backends.robotwin.session import (
    RoboTwinControllerTrackingError,
    RoboTwinLiveSession,
)
from ora.environment.controllers.embodiment_control import (
    load_embodiment_control_profile,
)
from ora.environment.controllers.motion_transaction import (
    MotionBackendCapabilities,
    MotionExecutionFault,
)
from ora.environment.controllers.profile_capabilities import (
    FROZEN_CONTROL_PROFILE_SHA256,
    OLDEST_COMPOSED_CONTROL_PROFILE,
)
from ora.environment.definitions import resolve_runtime_descriptor
from ora.environment.session import DescribedEnvironmentSession
from ora.orchestrator.spec import RunSpecError, resolve_run_spec


ROOT = Path(__file__).resolve().parents[1]
#: The oldest control profile this tree composes; a sparse RoboTwin session
#: refuses every older one.
CONTROL_PROFILE = ROOT / (
    "ora/policy/embodiments/aloha-agilex/"
    f"control-profile-v{OLDEST_COMPOSED_CONTROL_PROFILE}.json"
)
CONTROL_PROFILE_SHA256 = FROZEN_CONTROL_PROFILE_SHA256[OLDEST_COMPOSED_CONTROL_PROFILE]


class _PlanningExtensionDelegate:
    def start(self) -> dict[str, Any]:
        return {}

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        del destination
        return {}, []

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        return dict(action)

    def probe_motion(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "probe", "request": dict(request)}

    def status(self) -> dict[str, Any]:
        return {
            "schema": "ora.environment-status.v1",
            "phase": "running",
            "terminal": False,
            "reason": None,
            "source": "test",
            "policy_visibility": "control_only",
            "signals": [],
        }

    @property
    def is_terminal(self) -> bool:
        return False

    def abort(self, reason: str) -> None:
        del reason

    def persist_official_verdict(self) -> dict[str, Any]:
        return {}

    def finalize_operator_video(self) -> None:
        return None

    def close(self) -> None:
        return None


def action_v3(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": "observation-0",
        "action_id": "action-0",
        "action": name,
        "arguments": arguments,
        "bound_observation_sha256": "0" * 64,
    }


def pose(x: float = 0.0) -> dict[str, list[float]]:
    return {
        "position_m": [x, 0.0, 1.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def primitive_runtime(
    tmp_path: Path, *, profile: str = "robotwin_toolchain_b2_closed_noservo_v2",
) -> tuple[ManipulationToolRuntime, dict[str, Any], list[dict[str, Any]]]:
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(profile)
    runtime._submitted_mutations = []
    runtime.control_root = tmp_path
    runtime.workspace_bounds_m = {
        "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
    }
    control = {
        "observation_id": "observation-0", "observation_sha256": "0" * 64,
        "generation": 0, "open": True,
    }
    observation = {
        "proprioception": {"end_effectors": {
            "left": {
                "pose_xyz_xyzw": [-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "right": {
                "pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
        }},
    }
    runtime._current_grip_state = lambda: (  # type: ignore[method-assign]
        control, tmp_path, observation,
        observation["proprioception"]["end_effectors"]["right"][
            "pose_xyz_xyzw"
        ],
        0.0,
    )
    persisted: list[dict[str, Any]] = []
    runtime._persist_action = (  # type: ignore[method-assign]
        lambda _path, _control, submitted: persisted.append(dict(submitted))
    )
    return runtime, observation, persisted


def _probe_public_result(*, observation_id: str, plan_ref: str) -> dict[str, Any]:
    planner = {
        "called": True, "status": "Success", "stage": "SUCCESS",
        "valid_query": True, "attempts": 1, "trajopt_attempts": 1,
        "used_graph": False, "goalset_index": 0, "path_steps": 4,
        "interpolation_dt_s": 0.004,
        "errors": {"position": 0.0, "rotation": 0.0, "cspace": None},
        "timings_s": {
            "solve": 0.01, "ik": 0.001, "graph": 0.0,
            "trajopt": 0.009, "finetune": 0.0, "total": 0.01,
        },
        "error": None,
    }
    pose_value = [-0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    variant = {
        "variant_id": "variant-00-tool-z-+0deg", "reachable": True,
        "target_pose_xyzw": pose_value, "planner": planner,
        "endpoint_error": {"translation_m": 0.001, "orientation_rad": 0.01},
        "trajectory_steps": 4, "trajectory_duration_s": 0.016,
        "plan_ref": plan_ref,
    }
    return {
        "schema": "ora.motion-probe-result.v1",
        "observation_id": observation_id,
        "world_mutation_sequence": 0,
        "variant_policy": "exact",
        "robot_decisions_consumed": 0,
        "world_state_changed": False,
        "results": [{
            "query_id": "q0", "arm": "left", "reachable": True,
            "selected_variant_id": variant["variant_id"],
            "plan_ref": plan_ref,
            "selected_target_pose_xyzw": pose_value,
            "variants": [variant],
        }],
    }


def test_receipt_effect_projection_is_registry_dispatched() -> None:
    """Per-backend receipt contracts live in the registry, not the session."""

    from pathlib import Path as _Path

    from ora.agent.native.receipt_projection import (
        ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS,
        register_receipt_effect_projector,
        resolve_receipt_effect_projector,
    )

    for schema in ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS:
        assert resolve_receipt_effect_projector(schema) is not None
    assert resolve_receipt_effect_projector("unknown-schema") is None
    assert resolve_receipt_effect_projector(None) is None
    with pytest.raises(ValueError, match="already registered"):
        register_receipt_effect_projector(
            ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS[:1],
            lambda record, context: {},
        )

    # Parity guard: the shared Native session keeps no per-backend receipt
    # schema branch (EXP08A cross-embodiment parity boundary).
    session_source = (
        _Path(__file__).resolve().parents[1]
        / "ora/agent/native/session.py"
    ).read_text(encoding="utf-8")
    assert "robotwin-sparse-motion-receipt" not in session_source


def test_public_motion_effect_envelope_is_capability_gated() -> None:
    """M2 ruling: unified surfaces get the cross-embodiment envelope; frozen
    surfaces keep their exact legacy projection."""

    from ora.agent.native.receipt_projection import (
        PUBLIC_MOTION_EFFECT_CAPABILITY,
        PUBLIC_MOTION_EFFECT_SCHEMA,
        ReceiptProjectionContext,
        resolve_receipt_effect_projector,
    )

    record = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action": {"action": "move_pose"},
        "action_id": "action-3",
        "execution_status": "not_reached",
        "target_arms": ["left"],
        "synchronization": {"mode": "single_arm"},
        "arms": {
            "left": {
                "targeted": True,
                "plan_validation": {"status": "accepted"},
                "plan_artifact": {"path": "private"},
            },
            "right": {"targeted": False},
        },
        "controller_tracking": {"completed": False, "arms": {}},
        "motion_strategy": "global_precontact_then_local",
        "failure_stage": "execution",
        "failure_code": "waypoint_tracking_timeout",
        "terminal_after_action": False,
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    assert projector is not None

    legacy = projector(record, ReceiptProjectionContext(
        last_action_id="action-3", last_action_public_name="move_pose",
    ))
    assert legacy["status"] == "not_reached"
    assert "schema" not in legacy
    assert legacy["failure_code"] == "controller_timeout"

    unified = projector(record, ReceiptProjectionContext(
        last_action_id="action-3", last_action_public_name="move_pose",
        surface_supports=lambda capability: (
            capability == PUBLIC_MOTION_EFFECT_CAPABILITY
        ),
    ))
    assert "schema" not in unified  # version identity lives in the capability
    assert PUBLIC_MOTION_EFFECT_SCHEMA.endswith("public-motion-effect.v1")
    assert unified["action"] == "move_pose"
    assert unified["status"] == "not_reached"
    assert unified["executor"] == [
        "robotwin_vendored_capx_planner", "robotwin_weighted_dls_local",
    ]
    assert unified["failure_code"] == "controller_timeout"
    assert unified["round_terminated"] is False
    assert unified["physical_cause"] == "unknown"
    assert unified["detail"]["arms"]["left"]["plan_validation"] == {
        "status": "accepted",
    }
    assert unified["detail"]["arms"]["left"]["plan_artifact_recorded"] is True
    # The private artifact path never reaches the policy-visible envelope.
    assert '"plan_artifact":' not in json.dumps(unified)
    assert "private" not in json.dumps(unified)


def test_described_session_forwards_read_only_planning_extensions() -> None:
    runtime_config = {
        "kind": "robotwin",
        "transport": "local_rpc",
        "required_revision": "c3ddfa8b97d5519efa828b075999bd0006778e5e",
        "policy_cameras": [
            "head_camera", "front_camera", "left_camera", "right_camera",
        ],
    }
    described = DescribedEnvironmentSession(
        delegate=_PlanningExtensionDelegate(),
        planned_descriptor=resolve_runtime_descriptor(
            runtime_config, action_codec="motion_action_v3",
        ),
        runtime_config=runtime_config,
        action_codec="motion_action_v3",
    )

    assert described.probe_motion({"query_id": "q0"}) == {
        "kind": "probe", "request": {"query_id": "q0"},
    }


class Joint:
    def __init__(self, value: float = 0.0) -> None:
        self.value = np.asarray([value], dtype=np.float64)

    def get_drive_target(self) -> np.ndarray:
        return self.value.copy()


class Robot:
    def __init__(self, *, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[str] = []
        self.left_arm_joints = [Joint() for _ in range(6)]
        self.right_arm_joints = [Joint() for _ in range(6)]
        self.left_gripper = [(Joint(),)]
        self.right_gripper = [(Joint(),)]
        self.left_original_pose = [-0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        self.right_original_pose = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    def _plan(self, arm: str, target: list[float]) -> dict[str, Any]:
        self.calls.append(arm)
        if self.fail == arm:
            return {"status": "Failure"}
        return {
            "status": "Success",
            "position": np.zeros((4, 6), dtype=np.float64),
            "velocity": np.zeros((4, 6), dtype=np.float64),
            "target": list(target),
        }

    def left_plan_path(self, target: list[float]) -> dict[str, Any]:
        return self._plan("left", target)

    def right_plan_path(self, target: list[float]) -> dict[str, Any]:
        return self._plan("right", target)

    def get_left_arm_real_jointState(self) -> list[float]:
        return [0.0] * 7

    def get_right_arm_real_jointState(self) -> list[float]:
        return [0.0] * 7


class Environment:
    def __init__(
        self, *, fail: str | None = None, execution_fault: bool = False,
        official_success: bool = False,
    ) -> None:
        self.robot = Robot(fail=fail)
        self.execution_fault = execution_fault
        self.official_success = official_success
        self.poses = {
            "left": [-0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "right": [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
        self.take_action_cnt = 0
        self.step_lim = 100
        self.eval_success = False
        self.eval_video_path = None
        self.dense_calls: list[dict[str, Any]] = []
        self.gripper_calls: list[str] = []
        self.check_calls = 0
        self.robot.get_left_tcp_pose = lambda: self._tcp_pose("left")
        self.robot.get_right_tcp_pose = lambda: self._tcp_pose("right")

    def _tcp_pose(self, arm: str) -> list[float]:
        pose = list(self.poses[arm])
        pose[0] += 0.12
        return pose

    def get_arm_pose(self, arm: str) -> list[float]:
        return list(self.poses[arm])

    def take_dense_action(self, control: dict[str, Any], save_freq: Any = None) -> None:
        self.dense_calls.append(dict(control))
        if self.execution_fault:
            raise RuntimeError("dense execution fault")
        for arm in ("left", "right"):
            planned = control[f"{arm}_arm"]
            if isinstance(planned, dict) and "target" in planned:
                self.poses[arm] = list(planned["target"])

    def set_gripper(self, set_tag: str, **kwargs: Any) -> Any:
        self.gripper_calls.append(set_tag)
        value = {
            "num_step": 1,
            "per_step": 0.1,
            "result": np.asarray([1.0]),
        }
        return (dict(value), dict(value)) if set_tag == "together" else dict(value)

    def check_success(self) -> bool:
        self.check_calls += 1
        return self.official_success

    def _update_render(self) -> None:
        return None


class Cameras:
    def __init__(self) -> None:
        self.picture_updates = 0

    def update_picture(self) -> None:
        self.picture_updates += 1

    @staticmethod
    def _rgb(value: int) -> np.ndarray:
        return np.full((24, 32, 3), value, dtype=np.uint8)

    def get_rgb(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "head_camera": {"rgb": self._rgb(50)},
            "front_camera": {"rgb": self._rgb(80)},
            "left_camera": {"rgb": self._rgb(110)},
            "right_camera": {"rgb": self._rgb(140)},
        }

    def get_observer_rgb(self) -> np.ndarray:
        return self._rgb(20)


def session(
    tmp_path: Path, environment: Environment, *, task: str = "blocks_ranking_size",
) -> RoboTwinLiveSession:
    """A sparse motion v3 session bound to the oldest composed control profile."""

    binding = load_embodiment_control_profile(CONTROL_PROFILE, CONTROL_PROFILE_SHA256)
    result = RoboTwinLiveSession(
        robotwin_root=tmp_path,
        output_root=tmp_path / "output",
        task=task,
        actual_seed=100,
        episode=0,
        action_codec="motion_action_v3",
        control_profile=binding,
        task_factory=lambda _task: environment,
    )
    result.environment = environment
    result.last_safe_pose = {
        arm: [
            *environment._tcp_pose(arm)[:3],
            *environment._tcp_pose(arm)[4:],
            environment._tcp_pose(arm)[3],
        ]
        for arm in ("left", "right")
    }
    result.last_safe_joints = {arm: [0.0] * 6 for arm in ("left", "right")}
    return result


# A sparse session on the production motion transaction against the
# source-free correctness plant (``test_robotwin_correctness_repair``), bound
# to the oldest composed control profile. Each arm's planner returns one
# straight IK branch to joint q0 = 0.02, whose public TCP is 0.02 m along +x
# from the arm's start.


def transaction_plant(**kwargs: Any) -> Any:
    """The correctness plant, with joint and gripper drive targets to digest."""

    from types import SimpleNamespace
    from test_robotwin_correctness_repair import Plant

    plant = Plant(**kwargs)
    gripper = {"left": 0.0, "right": 0.0}
    plant.gripper_writes = []

    def write_gripper(value: float, arm: str, per_step: float = 0.0) -> None:
        plant.write_gripper(value, arm, per_step)
        plant.gripper_writes.append(arm)
        gripper[arm] = float(value)

    plant.robot.set_gripper = write_gripper
    for arm in ("left", "right"):
        setattr(plant.robot, f"{arm}_arm_joints", [
            SimpleNamespace(get_drive_target=lambda a=arm, i=index: np.asarray([plant.targets[a][i]]))
            for index in range(6)
        ])
        setattr(plant.robot, f"{arm}_gripper", [
            (SimpleNamespace(get_drive_target=lambda a=arm: np.asarray([gripper[a]])),),
        ])
    return plant


def transaction_session(
    tmp_path: Path, plant: Any, *, failing: tuple[str, ...] = (),
) -> tuple[RoboTwinLiveSession, list[str]]:
    """A v10 sparse session on ``plant``; also returns the arms each planner call named."""

    import copy
    from test_robotwin_correctness_repair import branch_candidates, live_session

    live = live_session(tmp_path, plant)
    live._public_to_planner_wxyz = lambda arm, pose: pose
    calls: list[str] = []

    def planner(arm: str) -> Any:
        def plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls.append(arm)
            if arm in failing:
                return {"status": "Failure", "planner_status": "IK_FAIL"}
            raw = branch_candidates(arm)
            for candidate in raw["candidates"]:
                candidate["position"][:, 2] = 0.0
            return copy.deepcopy(raw)
        return plan

    for arm in ("left", "right"):
        setattr(plant.robot, f"{arm}_plan_path", planner(arm))
        setattr(plant.robot, f"{arm}_plan_joint_path", planner(arm))
    return live, calls


def _transaction_target(arm: str) -> dict[str, list[float]]:
    return {
        "position_m": [-0.18 if arm == "left" else 0.22, 0.0, 0.8],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def transaction_move(*arms: str) -> dict[str, Any]:
    from test_robotwin_correctness_repair import action as correctness_action

    return correctness_action("move_pose", {
        "targets": {arm: _transaction_target(arm) for arm in arms},
        "interactions": {arm: "free_space" for arm in arms},
    })


def test_partial_local_fault_records_numeric_replay_instead_of_empty_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = Environment()
    live = session(tmp_path, environment, task="place_mouse_pad")

    class IntentAdapter:
        capabilities = MotionBackendCapabilities(
            backend_id="robotwin",
            embodiment_id="aloha_agilex",
            arms=("left", "right"),
            controlled_point="parallel_jaw_grip_center_tcp",
            planner_frames={"left": "fl_link6", "right": "fr_link6"},
            synchronized_groups=(("left", "right"),),
            global_joint_planning=True,
            local_cartesian=True,
        )

        @staticmethod
        def capture_action_start() -> dict[str, Any]:
            return {"measured_joints": {"left": [0.0] * 6, "right": [0.0] * 6}}

    monkeypatch.setattr(
        robotwin_session, "RoboTwinMotionBackendAdapter",
        lambda **_kwargs: IntentAdapter(),
    )
    issued = np.asarray([
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    measured_tcp = np.asarray([
        [-0.075, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [-0.070, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    arrays = {
        "issued_left_arm__position": issued,
        "issued_left_arm__velocity": np.zeros_like(issued),
        "issued_right_arm__position": np.empty((0, 6)),
        "issued_right_arm__velocity": np.empty((0, 6)),
        "measured_left_arm__position": issued,
        "measured_right_arm__position": np.zeros((2, 6)),
        "measured_left_tcp__pose_xyzw": measured_tcp,
        "measured_right_tcp__pose_xyzw": np.asarray([
            [0.32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]),
    }
    control_sequence = {
        "left_arm": {
            "status": "IssuedLocalCartesian",
            "position": issued,
            "velocity": np.zeros_like(issued),
        },
        "left_gripper": None,
        "right_arm": None,
        "right_gripper": None,
    }
    tracking = {
        "schema": "ora.robotwin-local-cartesian-tracking.v1",
        "backend": "pinocchio_numeric_jacobian_dls",
        "target_arms": ["left"],
        "sampled_steps": 2,
        "controller_write_attempts": 2,
        "physics_step_attempts": 2,
        "execution_started": True,
        "completed": False,
        "abort": {"code": "backend_step_fault", "iteration": 1},
        "minimum_inter_arm_center_distance_m": 0.4,
        "payload_validation": None,
        "arms": {
            arm: {
                "maximum_joint_tracking_error_rad": 0.0,
                "maximum_tcp_step_m": 0.005 if arm == "left" else 0.0,
            }
            for arm in ("left", "right")
        },
    }

    def fail_local(**_kwargs: Any) -> Any:
        raise RoboTwinControllerTrackingError(
            "partial local failure", summary=tracking, arrays=arrays,
            control_sequence=control_sequence,
        )

    live._execute_local_cartesian_action = fail_local  # type: ignore[method-assign]
    receipt = live.execute({
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": "observation-0",
        "action_id": "partial-local-fault",
        "action": "move_pose",
        "arguments": {
            "targets": {"left": {
                "position_m": [-0.07, 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }},
            "interactions": {"left": "contact_approach"},
        },
        "bound_observation_sha256": "0" * 64,
    })["receipt"]

    assert receipt["motion_strategy"]["strategy"] == "local"
    assert receipt["execution_status"] == "execution_fault"
    assert receipt["state_changed"] is True
    assert receipt.get("steps_used") != 0  # A partial controller write is never refunded.
    manifest = load_controller_replay_manifest(live.output_root / "controller-replay/manifest.json")
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["physics_steps"] == 2
    assert entry["replayable"] is False
    assert entry["metadata"].get("reason") != "no_controller_sequence_executed"
    artifact = live.output_root / entry["artifact"]["path"]
    with np.load(artifact, allow_pickle=False) as replay_arrays:
        assert replay_arrays["left_arm__position"].shape == (2, 6)
        assert replay_arrays["issued_left_arm__position"].shape == (2, 6)
        assert replay_arrays["measured_left_tcp__pose_xyzw"].shape == (2, 7)


@pytest.mark.parametrize("infrastructure_fault", [False, True])
def test_transaction_planner_refusal_certifies_zero_execution(tmp_path, monkeypatch, infrastructure_fault):
    from types import SimpleNamespace
    environment = Environment()
    live = session(tmp_path, environment)
    # Exercise the receipt path without importing simulator kinematics.
    monkeypatch.setattr(robotwin_session, "RoboTwinMotionBackendAdapter",
        lambda **_kwargs: SimpleNamespace(
            capabilities=RoboTwinMotionBackendAdapter.capabilities,
            capture_action_start=lambda: {}))

    def fail_plan(state):
        state.planner_results.update({arm: {"status": "IKFail", "stage": "ik"} for arm in state.target_arms})
        state.public_pose_targets = {arm: [*target["position_m"], target["orientation_xyzw"][3],
            *target["orientation_xyzw"][:3]] for arm, target in state.targets.items()}
        if infrastructure_fault:
            raise RuntimeError("uncertain controller failure")
        return False

    monkeypatch.setattr(live, "_plan_and_validate", fail_plan)
    receipt = live.execute(action_v3("move_pose", {
        "targets": {"left": pose(-.1)}, "interactions": {"left": "free_space"},
    }))["receipt"]
    assert receipt["execution_status"] == ("execution_fault" if infrastructure_fault else "planner_failed")
    if infrastructure_fault:
        assert "steps_used" not in receipt
    else:
        assert receipt["steps_used"] == 0
    assert receipt["state_changed"] is False
    assert receipt["before"] == receipt["after"]
    assert environment.dense_calls == []


def _live_robotwin_native_spec() -> dict[str, Any]:
    """The default RoboTwin Native RunSpec, expanded to its explicit fields.

    The baseline marker is dropped so the RunPlan camera-topology and
    control-profile checks see each mutated field directly, instead of the
    baseline layer refusing the conflicting override first. EXP06B's own
    composition (pressure retention, motion_action_v2, visible_only_v1
    replay, control-profile-v1) composes only at its launch commits.
    """

    from ora.orchestrator.defaults import expand_native_defaults
    from test_native_baseline import native_spec

    spec = expand_native_defaults(native_spec("robotwin"), root=ROOT)
    spec["policy"].pop("baseline")
    return spec


def test_motion_v2_runplan_rejects_incomplete_camera_topology() -> None:
    spec = _live_robotwin_native_spec()
    spec["runtime"]["policy_cameras"] = ["head_camera", "left_camera"]
    with pytest.raises(RunSpecError, match="policy cameras"):
        resolve_run_spec(spec, root=ROOT)


def test_motion_v2_runplan_rejects_an_observer_camera() -> None:
    spec = _live_robotwin_native_spec()
    spec["runtime"]["policy_observer_camera"] = True
    with pytest.raises(RunSpecError, match="policy_observer_camera is not supported"):
        resolve_run_spec(spec, root=ROOT)

    spec = _live_robotwin_native_spec()
    spec["runtime"]["main_camera"] = "observer_camera"
    with pytest.raises(RunSpecError, match="main_camera"):
        resolve_run_spec(spec, root=ROOT)


def test_motion_v2_compilation_result_has_no_private_grip_site_vocabulary() -> None:
    public = ManipulationToolRuntime._public_end_effector_compilation({
        "current_pose_xyz_xyzw": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "target_pose_xyz_xyzw": [0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "translation_world_m": [0.01, 0.0, 0.0],
        "translation_distance_m": 0.01,
        "rotation_angle_rad": 0.0,
        "resolved_grip_site_axes_world": {"private": True},
    }, delta_frame="end_effector")
    assert public["controlled_point"] == "end_effector"
    assert public["delta_frame"] == "end_effector"
    assert "grip_site" not in json.dumps(public)
