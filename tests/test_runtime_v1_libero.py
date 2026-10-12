"""Runtime v1 libero contract regressions."""

from __future__ import annotations
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from ora.environment.controller_replay import load_controller_replay_manifest
from ora.orchestrator.spec import file_sha256
from ora.orchestrator.preflight import _libero_probe_matches_plan
from ora.orchestrator.coordinator import SupervisorError, _main_observation_image, _qualified_runtime_binding
from ora.contracts.workspace import is_monotonic_workspace_recovery, workspace_state
from ora.environment.backends.libero_pro.session import _benchmark_data_binding, _build_environment, _inverse_controller_scale, LiberoProError, LiberoProSession, libero_environment_init_params, libero_osc_action_contract, libero_public_environment_context
from ora.environment.backends.libero_pro.registration import CORE_SUITES, ensure_perturbation_suites
from ora.environment.benchmarks.libero import LIBERO_BASE_SUITES, LIBERO_BASE_SUITE_TASK_COUNTS, LiberoBenchmarkError, validate_base_suite_task
from ora.environment.benchmarks.libero_pro import BASE_SUITES as LIBERO_PRO_BASE_SUITES, RUNTIME_SUITES as LIBERO_RUNTIME_SUITES
from test_runtime_v1 import (
    ContactBlockedGripperEnvironment,
    ContactPerturbedLiberoEnvironment,
    CurvedProgressLiberoEnvironment,
    DivergingLiberoEnvironment,
    FakeLiberoEnvironment,
    GraduallyDivergingLiberoEnvironment,
    GraduallyRotatingAwayLiberoEnvironment,
    GripperContactMotionEnvironment,
    ROOT,
    RotatingLiberoEnvironment,
    _fake_libero_metadata,
)


def test_libero_pro_session_projects_metric_rgbd_and_uses_official_termination(
    tmp_path: Path,
) -> None:
    environment = FakeLiberoEnvironment()
    # RoboSuite's public raw quaternion belongs to the hand body, not grip_site.
    # The Session must ignore it and use the controller-owned site orientation.
    environment.current_raw_obs["robot0_eef_quat"] = np.asarray(
        [[0.0, 0.0, 2**-0.5, 2**-0.5]], dtype=np.float32,
    )
    environment.current_raw_obs["robot0_gripper_qpos"] = np.asarray(
        [[0.01, -0.01]], dtype=np.float32,
    )
    public_environment_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )
    factory_arguments = {}

    def factory(**kwargs):
        factory_arguments.update(kwargs)
        return environment, _fake_libero_metadata(
            tmp_path, public_environment_context,
        )

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test", workspace_bounds_m={
            "x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3],
        }, max_sim_steps=100, environment_factory=factory,
    )
    reset = session.start()
    assert factory_arguments["rlinf_root"] == (tmp_path / "RLinf").resolve()
    assert factory_arguments["libero_pro_root"] == (
        tmp_path / "LIBERO-PRO"
    ).resolve()
    assert factory_arguments["data_root"] == (tmp_path / "data").resolve()
    assert factory_arguments["runtime_config_root"] == (
        tmp_path / "native/runtime-config/libero"
    ).resolve()
    assert reset["official_predicate_at_reset"] is None
    assert reset["public_environment_context"] == public_environment_context
    assert reset["controller_action_contract"]["output_max"][3:] == [0.5, 0.5, 0.5]
    observation, images = session.observe(tmp_path / "observations")
    assert observation["main_camera"] == "agentview"
    assert set(observation["cameras"]) == {"agentview", "wrist"}
    assert all(path.is_file() for path in images)
    observation_root = tmp_path / "observations" / observation["observation_id"]
    assert _main_observation_image(observation, observation_root) == (
        observation_root / "agentview/rgb.png"
    )
    calibration = json.loads(
        (tmp_path / "observations" / observation["observation_id"]
         / "agentview/calibration.json").read_text()
    )
    assert calibration["source_camera"] == "agentview"
    assert calibration["extrinsic_cv"] == np.eye(4).tolist()
    assert calibration["valid_depth_range_m"] == {"near": 0.1, "far": 10.0}
    assert observation["proprioception"]["end_effectors"]["right"][
        "pose_frame"
    ] == "robosuite_grip_site"
    assert observation["proprioception"]["end_effectors"]["right"][
        "pose_xyz_xyzw"
    ][3:] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert "model_inputs" not in observation
    action = {
        "schema": "ora-action.v3",
        "observation_id": observation["observation_id"],
        "action_id": "libero-action-1", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.02, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    receipt = session.execute(action)
    assert receipt["steps_used"] == 1
    assert receipt["preserved_dimension"]["field"] == "gripper_actuator_target"
    assert receipt["preserved_dimension"]["verified"] is True
    assert receipt["command_invariant"] == {
        "schema": "ora.command-invariant.v2",
        "controlled_dimension": "pose",
        "held_dimension": "gripper_actuator_target",
        "directive_before": pytest.approx(-1.0),
        "directive_after": pytest.approx(-1.0),
        "action_dimension_command": 0.0,
        "verified": True,
    }
    assert receipt["control_health"]["status"] == "healthy"
    assert environment.last_action[0, 6] == pytest.approx(0.0)
    assert receipt["gripper_evidence"]["measured_before"] == pytest.approx(0.25)
    assert receipt["gripper_evidence"]["measured_after"] == pytest.approx(0.25)
    assert environment.gripper_hold_directive == pytest.approx(-1.0)
    next_observation, _images = session.observe(tmp_path / "observations")
    close = {
        "schema": "ora-action.v3",
        "observation_id": next_observation["observation_id"],
        "action_id": "libero-action-2", "action": "set_gripper",
        "bound_observation_sha256": "bound-2",
        "arguments": {"arm": "right", "gripper": 0.0},
    }
    close_receipt = session.execute(close)
    assert close_receipt["reached_gripper"] == pytest.approx(0.0)
    assert close_receipt["preserved_dimension"]["field"] == "eef_pose"
    assert close_receipt["preserved_dimension"]["verified"] is True
    assert close_receipt["command_invariant"]["controlled_dimension"] == "gripper"
    assert close_receipt["command_invariant"]["held_dimension"] == "eef_pose_target"
    assert close_receipt["command_invariant"]["verified"] is True
    assert environment.last_action[0, 6] == pytest.approx(1.0)
    assert session.is_terminal is True
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True
    assert verdict["source"] == "LIBERO environment termination"
    with pytest.raises(
        LiberoProError, match="previous round verdict must be archived"
    ):
        session.restart_round(1)
    (tmp_path / "native/official-verdict.json").unlink()
    qualified_controller_binding = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": reset["controller_action_contract"]["contract_sha256"],
        "binding": {"source": "test-qualified-controller"},
    }
    session.qualified_controller_binding = qualified_controller_binding
    stale_stop = tmp_path / "native/control/stop.request"
    stale_stop.parent.mkdir(parents=True, exist_ok=True)
    stale_stop.write_text("policy_finished_episode\n", encoding="utf-8")
    session.plan_references["plan-from-previous-round"] = {"stale": True}
    round_reset = session.restart_round(1)
    # The prior round's stop request must not leak into the next round:
    # executors poll it per tick and would refuse every global motion.
    assert not stale_stop.exists()
    # Neither may a prior round's plan references survive the reset.
    assert session.plan_references == {}
    assert round_reset["round_index"] == 1
    assert round_reset["schema"] == "ora.libero-pro-round-reset.v1"
    assert round_reset["qualified_controller_binding"] == (
        qualified_controller_binding
    )
    assert session.status()["terminal"] is False
    assert session.sim_steps == 0
    assert session.action_sequence == 2
    assert session.last_receipt is None
    restarted_observation, _ = session.observe(tmp_path / "observations")
    assert restarted_observation["sequence"] == 2
    assert restarted_observation["previous_execution_receipt"] is None
    assert (tmp_path / "native/round-resets/round-0001.json").is_file()
    seed_reset = session.restart_round(2, seed=4)
    assert seed_reset["seed"] == 4
    assert seed_reset["reset_state_id"] == 4
    assert seed_reset["qualified_controller_binding"] == (
        qualified_controller_binding
    )
    assert environment.last_reset_state_ids == [4]
    assert session.seed == 4
    assert session.metadata["reset_id"] == 4
    replay = load_controller_replay_manifest(tmp_path / "native/controller-replay/manifest.json")
    assert replay["backend"] == "libero_pro"
    assert replay["runtime_media_capture"] is False
    assert [entry["controller"] for entry in replay["entries"]] == [
        "libero_osc_pose", "libero_osc_pose",
    ]
    assert all(entry["artifact"]["bytes"] > 0 for entry in replay["entries"])
    assert not (tmp_path / "native/operator-execution-frames").exists()
    session.close()
    assert environment.closed is True

    blocked_environment = ContactBlockedGripperEnvironment()

    def blocked_factory(**_kwargs):
        return blocked_environment, _fake_libero_metadata(
            tmp_path, public_environment_context,
        )

    blocked = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "blocked-native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=200, environment_factory=blocked_factory,
    )
    blocked.start()
    first_observation, _ = blocked.observe(tmp_path / "blocked-observations")
    blocked.execute({
        "schema": "ora-action.v3",
        "observation_id": first_observation["observation_id"],
        "action_id": "blocked-close", "action": "set_gripper",
        "bound_observation_sha256": "blocked-bound-0",
        "arguments": {"arm": "right", "gripper": 0.0},
    })
    assert blocked_environment.gripper_hold_directive == pytest.approx(1.0)
    second_observation, _ = blocked.observe(tmp_path / "blocked-observations")
    preserved = blocked.execute({
        "schema": "ora-action.v3",
        "observation_id": second_observation["observation_id"],
        "action_id": "blocked-move", "action": "move_pose",
        "bound_observation_sha256": "blocked-bound-1",
        "arguments": {
            "arm": "right", "position_m": [0.02, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    assert preserved["command_invariant"]["directive_before"] == pytest.approx(1.0)
    assert preserved["command_invariant"]["directive_after"] == pytest.approx(1.0)
    assert preserved["command_invariant"]["verified"] is True
    assert blocked_environment.last_action[0, 6] == pytest.approx(0.0)
    blocked.close()


def test_libero_contact_deflection_is_degraded_physics_not_infra_fault(
    tmp_path: Path,
) -> None:
    environment = ContactPerturbedLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, environment_factory=factory,
    )
    session.start()
    receipt = session.execute({
        "schema": "ora-action.v3", "observation_id": "observation-0",
        "action_id": "contact-deflection", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.02, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    assert receipt["command_invariant"]["verified"] is True
    assert receipt["preserved_dimension"]["verified"] is True
    assert receipt["preserved_dimension"]["measured_aperture_delta"] != 0
    assert receipt["control_health"]["status"] == "healthy"
    assert receipt["control_health"]["codes"] == []
    assert session.status()["terminal"] is False
    session.close()


def test_libero_motion_force_is_telemetry_without_stop_authority(
    tmp_path: Path,
) -> None:
    environment = FakeLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "motion-force-reflex",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, environment_factory=factory,
    )
    session.start()
    session.action_codec = "motion_action_v3"
    session.motion_profile = json.loads((
        ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    ).read_text(encoding="utf-8"))
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "joint_velocities": [0.0] * 7,
        "external_wrench_clipped": [41.0, 0.0, 0.0],
    }

    receipt = session.execute({
        "schema": "ora-motion-action.v1",
        "observation_id": "observation-0",
        "action_id": "force-observed-servo",
        "action": "servo_delta",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right",
            "target_pose": {
                "position_m": [0.0, 0.0, 0.49],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "delta_pose": {
                "translation_m": [0.0, 0.0, -0.01],
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
            "frame": "world",
            "max_iterations": 5,
        },
    })

    assert environment.step_count == 1
    assert receipt["motion_status"] == "reached"
    assert receipt["control_health"]["status"] == "healthy"
    assert receipt["control_health"]["codes"] == []
    assert receipt["control_health"]["tracking_guard"]["outcome"] == (
        "not_triggered"
    )
    assert receipt["control_health"]["tracking_guard"]["reason"] is None
    assert receipt["control_health"]["force_telemetry"] == {
        "threshold_n": 40.0,
        "exceeded": True,
        "first_threshold_tick": 0,
        "stop_authority": False,
        "stop_reason": None,
        "stopped_tick": None,
    }
    assert receipt["feedback"]["contact_estimated"] is True
    assert receipt["feedback"]["force_threshold_exceeded"] is True
    assert receipt["feedback"]["collision"] is None
    assert receipt["feedback"]["reflex"] is False
    assert session.status()["terminal"] is False
    session.close()


def test_libero_osc_rotation_uses_physical_controller_range() -> None:
    contract = libero_osc_action_contract({
        "control_delta": True,
        "input_min": -1, "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    })
    normalized = _inverse_controller_scale(
        np.asarray([0.025, 0.0, 0.0, 0.0, 0.0, 0.10]), contract,
    )
    assert normalized.tolist() == pytest.approx([0.5, 0, 0, 0, 0, 0.2])


def test_libero_nonzero_rotation_closes_the_servo_loop_in_radians(
    tmp_path: Path,
) -> None:
    environment = RotatingLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf",
        libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data",
        output_root=tmp_path / "native-rotation",
        task="libero_goal_task-t00",
        seed=0,
        suite="libero_goal_task",
        task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100,
        servo_max_steps=4,
        environment_factory=factory,
    )
    session.start()
    target_angle = 0.2
    receipt = session.execute({
        "schema": "ora-action.v3",
        "observation_id": "observation-0",
        "action_id": "rotate-z",
        "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right",
            "position_m": [0.0, 0.0, 0.5],
            "orientation_xyzw": [
                0.0, 0.0, float(np.sin(target_angle / 2.0)),
                float(np.cos(target_angle / 2.0)),
            ],
        },
    })
    assert receipt["steps_used"] == 2
    assert [action[0, 5] for action in environment.actions] == pytest.approx(
        [0.2, 0.2], abs=1e-6,
    )
    assert receipt["control_health"][
        "target_orientation_error_before_rad"
    ] == pytest.approx(target_angle, abs=1e-6)
    assert receipt["control_health"][
        "target_orientation_error_after_rad"
    ] == pytest.approx(0.0, abs=1e-6)
    assert receipt["control_health"]["status"] == "healthy"
    session.close()


def test_libero_session_allows_only_monotonic_out_of_bounds_recovery(
    tmp_path: Path,
) -> None:
    environment = FakeLiberoEnvironment()
    environment.current_raw_obs["robot0_eef_pos"] = np.asarray(
        [[-0.7, 0.0, 0.5]], dtype=np.float32,
    )
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf",
        libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data",
        output_root=tmp_path / "native-recovery",
        task="libero_goal_task-t00",
        seed=0,
        suite="libero_goal_task",
        task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100,
        environment_factory=factory,
    )
    session.start()
    action = {
        "schema": "ora-action.v3",
        "observation_id": "observation-0",
        "action_id": "workspace-recovery",
        "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right",
            "position_m": [-0.75, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    with pytest.raises(LiberoProError, match="not a monotonic recovery"):
        session.execute(action)
    action["arguments"]["position_m"] = [-0.65, 0.0, 0.5]
    receipt = session.execute(action)
    assert receipt["control_health"]["monotonic_workspace_recovery"] is True
    assert receipt["control_health"]["status"] == "degraded"
    assert receipt["control_health"]["codes"] == [
        "eef_outside_action_workspace",
    ]
    session.close()


def test_workspace_recovery_is_monotonic_and_cannot_expand_violation() -> None:
    bounds = {"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]}
    assert workspace_state([-0.9, 0.0, 0.5], bounds)["max_violation_m"] == pytest.approx(0.3)
    assert is_monotonic_workspace_recovery(
        [-0.9, 0.0, 0.5], [-0.7, 0.0, 0.5], bounds,
    )
    assert is_monotonic_workspace_recovery(
        [-0.9, 0.0, 0.5], [-0.5, 0.0, 0.5], bounds,
    )
    assert not is_monotonic_workspace_recovery(
        [-0.9, 0.0, 0.5], [-0.95, 0.0, 0.5], bounds,
    )
    assert not is_monotonic_workspace_recovery(
        [-0.9, 0.0, 0.5], [-0.7, 0.6, 0.5], bounds,
    )


def test_libero_control_health_fault_stops_after_gross_divergence(
    tmp_path: Path,
) -> None:
    environment = DivergingLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, servo_max_steps=1, environment_factory=factory,
    )
    session.start()
    receipt = session.execute({
        "schema": "ora-action.v3", "observation_id": "observation-0",
        "action_id": "diverge", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.02, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    assert receipt["control_health"]["status"] == "fault"
    assert set(receipt["control_health"]["codes"]) >= {
        "gross_target_divergence", "gross_workspace_overrun",
    }
    status = session.status()
    assert status["terminal"] is True
    assert status["reason"] == "libero_control_health_fault"
    assert status["signals"][0]["policy_visibility"] == "control_only"
    assert session.persist_official_verdict()["official_success"] is False
    session.close()


def test_libero_tracking_guard_records_cross_axis_drift_without_stopping_progress(
    tmp_path: Path,
) -> None:
    environment = CurvedProgressLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "tracking-native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, servo_max_steps=80, environment_factory=factory,
    )
    session.start()
    receipt = session.execute({
        "schema": "ora-action.v3", "observation_id": "observation-0",
        "action_id": "cross-axis-drift", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.1, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    assert receipt["steps_used"] == 4
    assert receipt["motion_status"] == "reached"
    assert receipt["control_health"]["status"] == "healthy"
    guard = receipt["control_health"]["tracking_guard"]
    assert guard["outcome"] == "not_triggered"
    assert guard["reason"] is None
    assert guard["thresholds"]["cross_axis_drift"] == (
        "telemetry_only_no_stop_authority"
    )
    assert session.status()["terminal"] is False
    tracking = receipt["tracking_evidence"]
    tracking_path = tmp_path / "tracking-native" / tracking["path"]
    assert tracking_path.is_file()
    evidence = json.loads(tracking_path.read_text())
    assert evidence["policy_visibility"] == "control_only"
    assert evidence["guard"]["outcome"] == "not_triggered"
    assert max(
        sample["cross_axis_drift_m"] for sample in evidence["samples"]
    ) > 0.02
    session.close()


def test_libero_tracking_growth_is_telemetry_without_stop_authority(
    tmp_path: Path,
) -> None:
    environment = GraduallyDivergingLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "diverging-native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, servo_max_steps=3, environment_factory=factory,
    )
    session.start()
    receipt = session.execute({
        "schema": "ora-action.v3", "observation_id": "observation-0",
        "action_id": "cross-axis-divergence", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.1, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    guard = receipt["control_health"]["tracking_guard"]
    assert receipt["motion_status"] == "not_reached"
    assert receipt["steps_used"] == 3
    assert guard["outcome"] == "not_triggered"
    assert guard["reason"] is None
    assert "position_error_growth" in guard["telemetry_thresholds_crossed"]
    assert guard["stopped_tick"] is None
    assert "gross_target_divergence" in receipt["control_health"]["codes"]
    assert receipt["control_health"]["status"] == "degraded"
    assert session.status()["terminal"] is False
    session.close()


def test_libero_orientation_divergence_is_action_failure_not_infra(
    tmp_path: Path,
) -> None:
    environment = GraduallyRotatingAwayLiberoEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "orientation-native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, servo_max_steps=3, environment_factory=factory,
    )
    session.start()
    receipt = session.execute({
        "schema": "ora-action.v3", "observation_id": "observation-0",
        "action_id": "orientation-divergence", "action": "move_pose",
        "bound_observation_sha256": "bound",
        "arguments": {
            "arm": "right", "position_m": [0.01, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })
    assert receipt["motion_status"] == "not_reached"
    assert receipt["control_health"]["status"] == "degraded"
    assert "gross_orientation_divergence" in receipt["control_health"]["codes"]
    assert "orientation_error_growth" in receipt["control_health"][
        "tracking_guard"
    ]["telemetry_thresholds_crossed"]
    assert session.status()["terminal"] is False
    session.close()


def test_libero_gripper_contact_motion_is_degraded_not_collision_truth(
    tmp_path: Path,
) -> None:
    environment = GripperContactMotionEnvironment()
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )

    def factory(**_kwargs):
        return environment, _fake_libero_metadata(tmp_path, public_context)

    session = LiberoProSession(
        rlinf_root=tmp_path / "RLinf", libero_pro_root=tmp_path / "LIBERO-PRO",
        data_root=tmp_path / "data", output_root=tmp_path / "gripper-motion-native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_rlinf_revision="rlinf-test",
        required_libero_pro_revision="libero-test",
        required_data_revision="data-test",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
        },
        max_sim_steps=100, servo_max_steps=3, environment_factory=factory,
    )
    session.start()
    session.action_codec = "motion_action_v3"
    session.motion_profile = json.loads((
        ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    ).read_text(encoding="utf-8"))
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "joint_velocities": [0.0] * 7,
        "external_wrench_clipped": [12.0, 0.0, 0.0],
        "external_force_magnitude_n": 12.0,
    }
    receipt = session.execute({
        "schema": "ora-motion-action.v1",
        "observation_id": "observation-0",
        "action_id": "gripper-contact-motion",
        "action": "set_gripper",
        "bound_observation_sha256": "bound",
        "arguments": {"arm": "right", "gripper": 0.0},
    })
    assert receipt["motion_status"] == "not_reached"
    assert receipt["control_health"]["status"] == "degraded"
    assert "gripper_action_moved_eef" in receipt["control_health"]["codes"]
    assert receipt["feedback"]["contact_estimated"] is True
    assert receipt["feedback"]["collision"] is None
    assert receipt["feedback"]["reflex"] is False
    assert session.status()["terminal"] is False
    session.close()


def test_libero_runtime_rechecks_bound_controller_source_files(tmp_path: Path) -> None:
    """Exercise the live-only binding branch without importing the simulator."""

    rlinf_root = tmp_path / "RLinf"
    libero_root = tmp_path / "LIBERO-PRO"
    data_root = tmp_path / "data"
    robosuite_root = tmp_path / "site-packages/robosuite"
    files = {
        "rlinf_module": rlinf_root / "rlinf/envs/libero/libero_env.py",
        "libero_module": libero_root / "libero/libero/__init__.py",
        "benchmark_module": libero_root / "libero/libero/benchmark.py",
        "bddl_root": libero_root / "libero/libero/bddl_files",
        "init_states_root": libero_root / "libero/libero/init_files",
        "robosuite_module": robosuite_root / "__init__.py",
        "controller_config_file": robosuite_root / "controllers/config/osc_pose.json",
        "panda_gripper_module": robosuite_root / "models/grippers/panda_gripper.py",
        "panda_gripper_file": robosuite_root / "models/assets/grippers/panda_gripper.xml",
    }
    controller_config = {
        "control_delta": True,
        "input_min": -1,
        "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    }
    for name, path in files.items():
        if name.endswith("_root"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(controller_config)
            if name == "controller_config_file" else name
        )
        path.write_text(content, encoding="utf-8")
    bddl_file = libero_root / "libero/libero/bddl_files/libero_object/task.bddl"
    init_file = (
        libero_root / "libero/libero/init_files/libero_object/task.pruned_init"
    )
    for path in (bddl_file, init_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.environment_factory = _build_environment
    session.rlinf_root = rlinf_root
    session.libero_pro_root = libero_root
    session.data_root = data_root
    session.suite = "libero_object"
    session.metadata = {
        "bddl_file": str(bddl_file),
        "init_states_file": str(init_file),
        "runtime_bindings": {name: str(path) for name, path in files.items()},
    }
    session.controller_action_contract = libero_osc_action_contract(controller_config)
    session.source_manifest = {"source": "test"}

    session._bind_dependency_files()
    bound_paths = {
        item["path"] for item in session.source_manifest["dependencies"]
    }
    assert str(files["controller_config_file"].resolve()) in bound_paths
    assert str(files["panda_gripper_module"].resolve()) in bound_paths

    session.qualified_controller_binding = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": session.controller_action_contract["contract_sha256"],
        "binding": {
            path_field: str(files[path_field].resolve())
            for path_field in (
                "robosuite_module", "controller_config_file",
                "panda_gripper_module", "panda_gripper_file",
            )
        } | {
            digest_field: file_sha256(files[path_field])
            for path_field, digest_field in (
                ("robosuite_module", "robosuite_module_sha256"),
                ("controller_config_file", "controller_config_sha256"),
                ("panda_gripper_module", "panda_gripper_module_sha256"),
                ("panda_gripper_file", "panda_gripper_sha256"),
            )
        },
    }
    session._validate_qualified_controller_binding()
    original_gripper_source = files["panda_gripper_module"].read_text(
        encoding="utf-8"
    )
    files["panda_gripper_module"].write_text("drift", encoding="utf-8")
    with pytest.raises(LiberoProError, match="changed after preflight"):
        session._validate_qualified_controller_binding()
    files["panda_gripper_module"].write_text(
        original_gripper_source, encoding="utf-8"
    )

    files["controller_config_file"].write_text(
        json.dumps({**controller_config, "output_max": [0.1] * 6}),
        encoding="utf-8",
    )
    with pytest.raises(LiberoProError, match="differs from the active contract"):
        session._bind_dependency_files()


def test_libero_supervisor_requires_the_passing_controller_preflight(
    tmp_path: Path,
) -> None:
    runtime = {"kind": "libero_pro"}
    with pytest.raises(SupervisorError, match="lacks its passing"):
        _qualified_runtime_binding(tmp_path, runtime)

    detail = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": "a" * 64,
        "binding": {},
    }
    (tmp_path / "preflight.json").write_text(json.dumps({
        "checks": [{
            "check": "libero_controller_action_contract",
            "passed": True,
            "detail": detail,
        }],
    }), encoding="utf-8")
    assert _qualified_runtime_binding(tmp_path, runtime) == detail

    value = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    value["checks"][0]["passed"] = False
    (tmp_path / "preflight.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SupervisorError, match="absent or invalid"):
        _qualified_runtime_binding(tmp_path, runtime)


def test_libero_pro_registration_uses_base_order_and_bddl_language(tmp_path: Path) -> None:
    task_type = namedtuple(
        "Task", "name language problem problem_folder bddl_file init_states_file"
    )
    task_maps = {}
    for family in ("libero_object", "libero_goal", "libero_spatial", "libero_10"):
        task_maps[family] = {
            "base_task": task_type(
                "base_task", "base language", "Libero", family,
                "base_task.bddl", "base_task.pruned_init",
            )
        }
    mapping = {}
    benchmark = SimpleNamespace()

    class FakeBenchmark:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

        def _make_benchmark(self):
            self.tasks = list(benchmark.task_maps[self.name].values())

        def get_task(self, index):
            return self.tasks[index]

    benchmark.Task = task_type
    benchmark.Benchmark = FakeBenchmark
    benchmark.task_maps = task_maps
    benchmark.get_benchmark_dict = lambda: mapping
    benchmark.register_benchmark = lambda cls: mapping.__setitem__(cls.__name__.lower(), cls)
    benchmark.get_libero_path = lambda kind: str(tmp_path / kind)
    for suite in CORE_SUITES:
        bddl = tmp_path / "bddl_files" / suite / "base_task.bddl"
        init_states = tmp_path / "init_states" / suite / "base_task.pruned_init"
        bddl.parent.mkdir(parents=True, exist_ok=True)
        init_states.parent.mkdir(parents=True, exist_ok=True)
        bddl.write_text(f"(define (problem x)\n (:language language for {suite})\n)")
        init_states.write_bytes(b"init")

    ensure_perturbation_suites(benchmark)
    assert set(mapping) == set(CORE_SUITES)
    selected = mapping["libero_goal_task"]()
    assert selected.get_task(0).name == "base_task"
    assert selected.get_task(0).language == "language for libero_goal_task"


def test_libero_runtime_admits_libero_90_without_pro_augmentation() -> None:
    assert LIBERO_BASE_SUITE_TASK_COUNTS == {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_90": 90,
        "libero_10": 10,
    }
    assert "libero_90" in LIBERO_RUNTIME_SUITES
    assert "libero_90" not in LIBERO_PRO_BASE_SUITES
    assert not any(suite.startswith("libero_90_") for suite in CORE_SUITES)
    validate_base_suite_task("libero_90", 0)
    validate_base_suite_task("libero_90", 89)
    with pytest.raises(LiberoBenchmarkError, match=r"\[0,89\]"):
        validate_base_suite_task("libero_90", 90)


def test_original_libero_suites_use_code_checkout_assets_not_pro_dataset(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "LIBERO-PRO"
    data_root = tmp_path / "LIBERO-Pro-dataset"
    for suite in LIBERO_BASE_SUITES:
        root, source = _benchmark_data_binding(
            libero_pro_root=code_root, data_root=data_root, suite=suite,
        )
        assert root == (code_root / "libero/libero").resolve()
        assert source == "pinned_libero_pro_code_original_base_suite"
    pro_root, pro_source = _benchmark_data_binding(
        libero_pro_root=code_root, data_root=data_root, suite="libero_object_swap",
    )
    assert pro_root == data_root.resolve()
    assert pro_source == "pinned_libero_pro_dataset"


def test_libero_preflight_requires_a_nonempty_init_state_set() -> None:
    runtime = {
        "robustness_case": None,
        "public_environment_context": None,
    }
    observed = {"language_matches": True, "init_state_count": 50}
    assert _libero_probe_matches_plan(
        observed, runtime, motion_enabled=False,
    )
    assert not _libero_probe_matches_plan(
        {**observed, "init_state_count": 0}, runtime, motion_enabled=False,
    )
    assert not _libero_probe_matches_plan(
        {"language_matches": True}, runtime, motion_enabled=False,
    )


def test_libero_preflight_refuses_a_goal_authoritative_language_divergence() -> None:
    """The evaluated goal, not the registry file name, owns the instruction.

    LIBERO-Pro perturbation suites rewrite the goal while keeping the base
    BDDL file name, so the registry language can instruct one task while the
    evaluator scores another. The launch gate must refuse that unless the
    plan records the exact divergence.
    """

    runtime = {
        "robustness_case": None,
        "public_environment_context": None,
        "expected_task_language": "put both moka pots on the stove",
    }
    aligned = {
        "language_matches": True,
        "init_state_count": 50,
        "bddl_task_language": "put both moka pots on the stove",
        "bddl_language_matches": True,
    }
    assert _libero_probe_matches_plan(aligned, runtime, motion_enabled=False)

    # The live 2026-08-29 defect: the registry says "both moka pots" while the
    # bound BDDL — and therefore the scored predicate — says "the left" one.
    diverged = {
        **aligned,
        "bddl_task_language": "put the left moka pot on the stove",
        "bddl_language_matches": False,
    }
    assert not _libero_probe_matches_plan(diverged, runtime, motion_enabled=False)

    # An exact recorded acknowledgement re-opens that cell deliberately.
    acknowledged = {
        **runtime,
        "task_language_divergence_ack": {
            "bddl_task_language": "put the left moka pot on the stove",
            "policy_task_language": "put both moka pots on the stove",
        },
    }
    assert _libero_probe_matches_plan(
        diverged, acknowledged, motion_enabled=False,
    )
    # A stale or mismatched acknowledgement does not.
    assert not _libero_probe_matches_plan(
        diverged,
        {
            **runtime,
            "task_language_divergence_ack": {
                "bddl_task_language": "put the right moka pot on the stove",
                "policy_task_language": "put both moka pots on the stove",
            },
        },
        motion_enabled=False,
    )
    # Older probes without the field stay launchable (no silent regression).
    assert _libero_probe_matches_plan(
        {"language_matches": True, "init_state_count": 50},
        runtime, motion_enabled=False,
    )
