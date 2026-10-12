"""Robotwin motion v2 profiles contract regressions."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from ora.environment.controller_replay import load_controller_replay_manifest
from ora.agent.tools.grasp import ALOHA_GRIP_CENTER_COMPILER_IDENTITIES, ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES, _compile_grasp_candidate
from ora.environment.backends.robotwin import session as robotwin_session
from ora.environment.backends.robotwin import local_cartesian as robotwin_local_cartesian
from ora.environment.backends.robotwin.session import RoboTwinControllerTrackingError, RoboTwinLiveSession
from ora.environment.controllers.embodiment_control import load_embodiment_control_profile
from ora.environment.controllers.motion_transaction import MotionBackendCapabilities, MotionExecutionFault
from ora.environment.definitions import resolve_runtime_descriptor
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from test_robotwin_motion_v2 import (
    ROOT,
    CONTROL_PROFILE,
    CONTROL_PROFILE_SHA256,
    Cameras,
    _live_robotwin_native_spec,
    transaction_move,
    transaction_plant,
    transaction_session,
)


def test_robotwin_global_transaction_checks_measured_dual_arm_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        capabilities = MotionBackendCapabilities(
            backend_id="robotwin",
            embodiment_id="aloha_agilex",
            arms=("left", "right"),
            controlled_point="parallel_jaw_grip_center_tcp",
            planner_frames={"left": "left_link6", "right": "right_link6"},
            synchronized_groups=(("left", "right"),),
            global_joint_planning=True,
            local_cartesian=True,
        )

        def __init__(self) -> None:
            self.measured = {
                "left": np.asarray([0.0, 0.0]),
                "right": np.asarray([0.1, 0.0]),
            }
            self.targets = {
                arm: value.copy() for arm, value in self.measured.items()
            }

        def read_joint_positions(self, arm: str) -> list[float]:
            return self.measured[arm].tolist()

        def read_public_tcp_pose(self, arm: str) -> list[float]:
            return [
                float(self.measured[arm][0]), 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0,
            ]

        def write_joint_target(
            self, arm: str, position: Any, velocity: Any,
        ) -> None:
            del velocity
            self.targets[arm] = np.asarray(position, dtype=np.float64)

        def step_physics(self) -> None:
            self.measured = {
                arm: target.copy() for arm, target in self.targets.items()
            }

    adapter = Adapter()
    monkeypatch.setattr(
        robotwin_session, "RoboTwinMotionBackendAdapter",
        lambda **_kwargs: adapter,
    )
    live = object.__new__(RoboTwinLiveSession)
    live.control_profile = {"profile": {
        "motion_transaction": {"global_convergence": {
            "joint_convergence_tolerance_rad": 0.03,
            "maximum_physics_steps_per_waypoint": 8,
            "maximum_total_physics_steps": 100,
            "maximum_tcp_step_m": 0.08,
            "workspace_margin_m": 0.1,
        }},
        "dual_arm_collision_check": {
            "link_radius_m": 0.045,
            "minimum_surface_clearance_m": 0.0,
        },
    }}
    live.workspace_bounds_m = {
        "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
    }
    live.operator_execution_video = False
    live._pinocchio_chain_points = (  # type: ignore[method-assign]
        lambda _environment, _arm, joints: [[float(joints[0]), 0.0, 1.0]]
    )
    control_sequence = {
        "left_arm": {
            "position": np.asarray([[0.02, 0.0]]),
            "velocity": np.asarray([[0.0, 0.0]]),
        },
        "left_gripper": None,
        "right_arm": None,
        "right_gripper": None,
    }

    with pytest.raises(RoboTwinControllerTrackingError) as raised:
        live._execute_motion_transaction_dense_action(
            environment=object(), control_sequence=control_sequence,
            action={"action_id": "collision-test"},
        )

    fault = raised.value
    assert fault.summary["abort"]["code"] == "inter_arm_collision"
    assert fault.summary["sampled_steps"] == 1
    assert fault.arrays["issued_left_arm__position"].shape == (1, 2)
    assert fault.control_sequence is not None
    assert fault.control_sequence["left_arm"]["position"].shape == (1, 2)
    assert fault.control_sequence["right_arm"] is None


def test_robotwin_local_fault_merges_prior_iterations_into_replay_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "left": np.asarray([-0.20, 0.0]),
        "right": np.asarray([0.20, 0.0]),
    }

    class Adapter:
        capabilities = MotionBackendCapabilities(
            backend_id="robotwin",
            embodiment_id="aloha_agilex",
            arms=("left", "right"),
            controlled_point="parallel_jaw_grip_center_tcp",
            planner_frames={"left": "left_link6", "right": "right_link6"},
            synchronized_groups=(("left", "right"),),
            global_joint_planning=True,
            local_cartesian=True,
        )

        @staticmethod
        def compute_local_joint_target(
            arm: str, current_joints: Any, current_pose: Any,
            target_pose: Any, profile: Any,
        ) -> dict[str, list[float]]:
            del arm, current_pose, target_pose, profile
            return {
                "position": [float(current_joints[0]) + 0.01, 0.0],
                "velocity": [0.0, 0.0],
            }

    adapter = Adapter()
    monkeypatch.setattr(
        robotwin_session, "RoboTwinMotionBackendAdapter",
        lambda **_kwargs: adapter,
    )
    transaction_calls = 0

    def transaction(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal transaction_calls
        transaction_calls += 1
        next_left = np.asarray(
            kwargs["paths"]["left"]["position"][0], dtype=np.float64,
        )
        state["left"] = next_left
        report = {
            "execution_started": True,
            "controller_write_attempts": 1,
            "physics_step_attempts": 1,
            "completed": transaction_calls == 1,
            "abort": (
                None if transaction_calls == 1
                else {"code": "backend_step_fault", "stage_index": 0}
            ),
            "waypoint_wait_steps": [1] if transaction_calls == 1 else [],
            "arms": {
                arm: {
                    "maximum_joint_tracking_error_rad": 0.0,
                    "maximum_tcp_step_m": 0.01 if arm == "left" else 0.0,
                }
                for arm in ("left", "right")
            },
        }
        arrays = {
            "issued_left_arm__position": np.asarray([state["left"]]),
            "issued_left_arm__velocity": np.asarray([[0.0, 0.0]]),
            "issued_right_arm__position": np.empty((0, 2)),
            "issued_right_arm__velocity": np.empty((0, 2)),
            "measured_left_arm__position": np.asarray([state["left"]]),
            "measured_right_arm__position": np.asarray([state["right"]]),
            "measured_left_tcp__pose_xyzw": np.asarray([[
                float(state["left"][0]), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ]]),
            "measured_right_tcp__pose_xyzw": np.asarray([[
                float(state["right"][0]), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ]]),
        }
        if transaction_calls == 2:
            raise MotionExecutionFault(
                "second local iteration fault", report=report, arrays=arrays,
            )
        return report, arrays

    # The local Cartesian executor lives in its own module since the 2026-10-01
    # session split; patch the name where that code reads it.
    monkeypatch.setattr(
        robotwin_local_cartesian, "execute_measured_convergence", transaction,
    )
    live = object.__new__(RoboTwinLiveSession)
    live.control_profile = {"profile": {
        "profile_id": "robotwin_aloha_motion_transaction_dual_arm_v5",
        "local_cartesian": {
            "maximum_translation_m": 0.05,
            "maximum_rotation_rad": 0.2617993877991494,
            "maximum_iterations": 60,
            "position_tolerance_m": 0.003,
            "orientation_tolerance_rad": 0.03,
            "stall_window": 8,
        },
        "controller_tracking": {
            "maximum_tcp_step_m": 0.08,
            "workspace_margin_m": 0.1,
        },
        "dual_arm_collision_check": {
            "link_radius_m": 0.045,
            "minimum_surface_clearance_m": 0.0,
        },
        "motion_transaction": {"local_convergence": {}},
    }}
    live.workspace_bounds_m = {
        "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
    }
    live.operator_execution_video = False
    live._measured_arm_joints = (  # type: ignore[method-assign]
        lambda _environment, arm: state[arm].tolist()
    )
    live._public_pose_wxyz = (  # type: ignore[method-assign]
        lambda _environment, arm: [
            float(state[arm][0]), 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ]
    )
    live._pinocchio_chain_points = (  # type: ignore[method-assign]
        lambda _environment, _arm, joints: [[float(joints[0]), 0.0, 1.0]]
    )
    local_environment = type("LocalEnvironment", (), {
        "scene": type("Scene", (), {"step": lambda self: None})(),
        "robot": object(),
    })()

    with pytest.raises(RoboTwinControllerTrackingError) as raised:
        live._execute_local_cartesian_action(
            environment=local_environment,
            target_pose_xyzw={
                "left": [-0.18, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            action={"action_id": "partial-local-fault"},
        )

    fault = raised.value
    assert transaction_calls == 2
    assert fault.summary["abort"] == {
        "code": "backend_step_fault", "stage_index": 0, "iteration": 1,
    }
    assert fault.summary["sampled_steps"] == 2
    assert fault.summary["controller_write_attempts"] == 2
    assert fault.summary["physics_step_attempts"] == 2
    assert fault.arrays["issued_left_arm__position"].shape == (2, 2)
    assert fault.arrays["measured_left_tcp__pose_xyzw"].shape == (2, 7)
    assert fault.control_sequence is not None
    assert fault.control_sequence["left_arm"]["position"].shape == (2, 2)
    assert fault.control_sequence["right_arm"] is None


def test_control_profile_and_runtime_descriptor_bind_topology() -> None:
    binding = load_embodiment_control_profile(
        CONTROL_PROFILE, CONTROL_PROFILE_SHA256,
    )
    assert set(binding["profile"]["arms"]) == {"left", "right"}
    descriptor = resolve_runtime_descriptor({
        "kind": "robotwin",
        "transport": "local_rpc",
        "required_revision": "c3ddfa8b97d5519efa828b075999bd0006778e5e",
        "policy_cameras": [
            "head_camera", "front_camera", "left_camera", "right_camera",
        ],
    }, action_codec="motion_action_v3")
    public = descriptor.as_dict()
    assert public["schema"].endswith(".v2")
    assert set(public["controlled_end_effectors"]) == {"left", "right"}
    assert public["control_groups"][0]["members"] == ["left", "right"]
    assert public["camera_roles"]["left_camera"] == "left_wrist"
    assert public["capabilities"]["policy_observer_camera"] is False


def test_robotwin_defaults_to_compact_controller_replay_not_live_frames(
    tmp_path: Path,
) -> None:
    plant = transaction_plant()
    live, _calls = transaction_session(tmp_path, plant)
    receipt = live.execute(transaction_move("left"))["receipt"]
    assert receipt["execution_status"] == "reached"
    steps = receipt["controller_tracking"]["sampled_steps"]
    assert steps == plant.steps > 0
    manifest_path = live.output_root / "controller-replay/manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    assert manifest["schema"] == (
        "ora.controller-replay-manifest.v2"
    )
    assert manifest["runtime_media_capture"] is False
    assert manifest["default_video_source"] == (
        "viewer_derived_controller_replay"
    )
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["controller"] == "robotwin_dense_joint_targets"
    assert entry["physics_steps"] == steps
    assert entry["execution"] == {
        "status": "reached", "settle_steps": 0, "fault": None,
    }
    artifact = live.output_root / entry["artifact"]["path"]
    assert artifact.is_file()
    with np.load(artifact, allow_pickle=False) as arrays:
        assert arrays["left_arm__position"].shape == (steps, 6)
        # The inactive arm receives no command; its measured state is evidence.
        assert "right_arm__position" not in arrays
        assert arrays["measured_right_arm__position"].shape == (steps, 6)
    assert not (live.output_root / "operator-execution-frames").exists()


def test_robotwin_records_all_five_cameras_on_actual_dense_control_steps(
    tmp_path: Path,
) -> None:
    plant = transaction_plant()
    plant.cameras = Cameras()
    live, _calls = transaction_session(tmp_path, plant)
    live.operator_world_video = True
    live.operator_execution_video = True
    receipt = live.execute(transaction_move("left"))["receipt"]
    assert receipt["execution_status"] == "reached"
    steps = receipt["controller_tracking"]["sampled_steps"]
    assert steps == plant.steps > 0
    timeline = live.output_root / "operator-execution-frames/timeline.jsonl"
    records = [json.loads(line) for line in timeline.read_text().splitlines()]
    # One sample before the first write, then one after every physics step.
    assert [record["phase"] for record in records] == [
        "action_start", *["motion_transaction_global"] * steps,
    ]
    assert records[0]["control_step_index"] is None
    assert all(record["action_id"] == "action-0" for record in records)
    assert records[0]["cameras"] == [
        "observer_camera", "head_camera", "front_camera",
        "left_camera", "right_camera",
    ]
    assert plant.cameras.picture_updates == len(records)
    video = live.finalize_operator_video()
    assert video is not None
    assert video["schema"] == (
        "ora.robotwin-operator-execution-video.v2"
    )
    assert video["frame_count"] == len(records)
    assert video["capture_semantics"] == (
        "same_environment_session_actual_physics_control_frames"
    )
    assert video["timeline"]["path"].endswith("timeline.jsonl")


def test_robotwin_operator_capture_failure_is_auxiliary_to_execution(
    tmp_path: Path,
) -> None:
    plant = transaction_plant()
    plant.cameras = Cameras()

    def fail_picture_update() -> None:
        raise RuntimeError("render capture unavailable")

    plant.cameras.update_picture = fail_picture_update
    live, _calls = transaction_session(tmp_path, plant)
    live.operator_world_video = True
    live.operator_execution_video = True
    receipt = live.execute(transaction_move("left"))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert receipt["execution_fault"] is None
    assert live.operator_capture_failure == (
        "RuntimeError: render capture unavailable"
    )
    assert (live.output_root / "operator-execution-capture-failure.json").is_file()
    video = live.finalize_operator_video()
    assert video is not None
    assert video["schema"] == "ora.operator-world-video.v1"
    assert video["capture_semantics"] == "action_boundary_keyframes"


def test_v3_grip_center_compiler_accepts_100mm_but_v2_remains_50mm() -> None:
    candidate = {
        "candidate_id": "candidate-100mm",
        "rank": 0,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "source_gripper_origin_xyz_m": [0.0, 0.0, 1.0],
        "source_pregrasp_origin_xyz_m": [-0.10, 0.0, 1.0],
        "approach_world": [1.0, 0.0, 0.0],
        "jaw_world": [0.0, 1.0, 0.0],
        "gripper_opening_m": 0.04,
        "backend_score": 0.8,
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    compiled = _compile_grasp_candidate(
        candidate,
        compiler_identity=ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES[
            "left"
        ],
    )
    assert compiled["pregrasp_pose"]["position_m"][0] == pytest.approx(0.0034)
    assert compiled["approach_delta"]["translation_m"] == pytest.approx(
        [0.10, 0.0, 0.0]
    )
    assert compiled["compiler_checks"]["embodiment_mapping"].endswith(
        "_left_grip_center_v3"
    )
    with pytest.raises(ValueError, match="bounded servo distance"):
        _compile_grasp_candidate(
            candidate,
            compiler_identity=ALOHA_GRIP_CENTER_COMPILER_IDENTITIES["left"],
        )


def test_motion_v2_runplan_rejects_missing_control_profile() -> None:
    spec = _live_robotwin_native_spec()
    del spec["runtime"]["control_profile"]
    with pytest.raises(RunSpecError, match="control_profile"):
        resolve_run_spec(spec, root=ROOT)
