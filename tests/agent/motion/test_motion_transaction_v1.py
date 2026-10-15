from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.contracts.action import (
    ActionValidationError,
    MOTION_ACTION_V3_SCHEMA,
    validate_runtime_action,
)

from ora.agent.tools.runtime.core import (
    _project_copyable_grasp_approach_targets,
)
from ora.agent.composition import resolve_tool_surface
from ora.agent.native.session import NativePlanner
from ora.agent.contracts import PolicySessionError
from ora.environment.controllers.motion_transaction import (
    MOTION_PLAN_ARTIFACT_SCHEMA,
    MotionBackendCapabilities,
    MotionExecutionFault,
    MotionTransactionError,
    classify_motion_status,
    execute_measured_convergence,
    require_motion_capabilities,
    select_post_global_phase,
    select_motion_strategy,
    select_nearest_joint_solution,
    validate_motion_backend_adapter,
    validate_motion_backend_capabilities,
    write_motion_plan_artifact,
)
from ora.environment.controllers.embodiment_control import (
    load_embodiment_control_profile,
)
from ora.environment.controllers.libero_osc import (
    LiberoOscContractError,
    libero_osc_action_contract,
    validate_libero_osc_action_contract,
)


PROFILE = {
    "joint_convergence_tolerance_rad": 0.03,
    "maximum_physics_steps_per_waypoint": 8,
    "maximum_total_physics_steps": 100,
    "maximum_tcp_step_m": 0.08,
    "workspace_margin_m": 0.1,
}
WORKSPACE = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
ROOT = Path(__file__).resolve().parents[3]
V10_PROFILE = (
    ROOT / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
)
V10_PROFILE_SHA256 = (
    "9730bb978d99f62e474f7701a5701123f482662d460fab9061b3c58cd3770d9e"
)


class FakeAdapter:
    capabilities = MotionBackendCapabilities(
        backend_id="fake",
        embodiment_id="fake_dual",
        arms=("left", "right"),
        controlled_point="parallel_jaw_grip_center_tcp",
        planner_frames={"left": "left_link", "right": "right_link"},
        synchronized_groups=(("left", "right"),),
        global_joint_planning=True,
        local_cartesian=True,
    )

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.measured = {"left": np.zeros(2), "right": np.zeros(2)}
        self.targets = {"left": np.zeros(2), "right": np.zeros(2)}
        self.writes: list[str] = []
        self.steps = 0

    def read_joint_positions(self, arm: str):
        return self.measured[arm].tolist()

    def read_public_tcp_pose(self, arm: str):
        q = self.measured[arm]
        sign = -1.0 if arm == "left" else 1.0
        return [sign * 0.2 + q[0], q[1], 0.8, 0.0, 0.0, 0.0, 1.0]

    def write_joint_target(self, arm: str, position, velocity):
        self.targets[arm] = np.asarray(position, dtype=np.float64)
        self.writes.append(arm)

    def step_physics(self):
        self.steps += 1
        for arm in ("left", "right"):
            self.measured[arm] += self.gain * (
                self.targets[arm] - self.measured[arm]
            )


def test_backend_capability_contract_and_fail_closed_selection() -> None:
    capabilities = validate_motion_backend_capabilities(
        FakeAdapter.capabilities.as_dict()
    )
    require_motion_capabilities(
        capabilities, requested_arms=["left", "right"],
        require_global=True, require_synchronized=True,
    )
    with pytest.raises(MotionTransactionError):
        require_motion_capabilities(
            capabilities, requested_arms=["unknown"], require_global=True,
        )

    undeclared = SimpleNamespace(
        capabilities=MotionBackendCapabilities(
            backend_id="fake",
            embodiment_id="fake_dual",
            arms=("left", "right"),
            controlled_point="parallel_jaw_grip_center_tcp",
            planner_frames={"left": "left_link", "right": "right_link"},
            synchronized_groups=(("left", "right"),),
            global_joint_planning=True,
            local_cartesian=False,
        ),
        capture_action_start=lambda: {},
        compile_public_target=lambda *_args: None,
        plan_global=lambda *_args: {},
        plan_batch_global=lambda *_args: {},
        fk_control_pose=lambda *_args: [0, 0, 0, 0, 0, 0, 1],
        fk_public_tcp_pose=lambda *_args: [0, 0, 0, 0, 0, 0, 1],
        joint_limits=lambda *_args: [[-1, 1]],
        workspace_bounds_m=lambda: WORKSPACE,
        validate_collision=lambda *_args: {},
        read_joint_positions=lambda *_args: [0],
        read_public_tcp_pose=lambda *_args: [0, 0, 0, 0, 0, 0, 1],
        write_joint_target=lambda *_args: None,
        step_physics=lambda: None,
        compute_local_joint_target=lambda *_args: {},
    )

    with pytest.raises(MotionTransactionError, match="undeclared"):
        validate_motion_backend_adapter(undeclared)


def test_nearest_ik_uses_wrapped_joint_distance() -> None:
    selected, index, distances = select_nearest_joint_solution(
        [[-3.13, 0.0], [2.0, 0.0]], [3.13, 0.0],
    )
    assert index == 0
    assert selected == [-3.13, 0.0]
    assert distances[0] < distances[1]


def test_measured_convergence_waits_and_never_writes_inactive_arm() -> None:
    adapter = FakeAdapter(gain=0.1)
    report, arrays = execute_measured_convergence(
        adapter=adapter,
        paths={"left": {
            "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
        }},
        all_arms=("left", "right"), profile=PROFILE,
        workspace_bounds_m=WORKSPACE, mode="global",
    )
    assert report["completed"] is True
    assert report["sampled_steps"] > 1
    assert set(adapter.writes) == {"left"}
    assert arrays["issued_right_arm__position"].shape == (0,)
    assert arrays["measured_right_tcp__pose_xyzw"].shape[0] == report["sampled_steps"]


def test_global_terminal_wait_uses_measured_tcp_not_loose_waypoint_tolerance() -> None:
    adapter = FakeAdapter(gain=0.5)
    target = [-0.16, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    report, _arrays = execute_measured_convergence(
        adapter=adapter,
        paths={"left": {
            "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
        }},
        all_arms=("left", "right"), profile=PROFILE,
        workspace_bounds_m=WORKSPACE, mode="global",
        endpoint_public_tcp_targets={"left": target},
        endpoint_position_tolerance_m=0.005,
        endpoint_orientation_tolerance_rad=0.05,
    )
    assert report["completed"] is True
    assert report["waypoint_wait_steps"][0] > 1
    terminal = report["terminal_endpoint_check"]
    assert terminal["enabled"] is True
    assert terminal["final_errors"]["left"]["translation_m"] <= 0.005
    assert set(adapter.writes) == {"left"}


def test_global_terminal_tcp_timeout_is_distinct_from_waypoint_timeout() -> None:
    class FrozenTcpAdapter(FakeAdapter):
        def read_public_tcp_pose(self, arm: str):
            sign = -1.0 if arm == "left" else 1.0
            return [sign * 0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]

    adapter = FrozenTcpAdapter(gain=1.0)
    report, _arrays = execute_measured_convergence(
        adapter=adapter,
        paths={"left": {
            "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
        }},
        all_arms=("left", "right"), profile={
            **PROFILE, "maximum_physics_steps_per_waypoint": 4,
        },
        workspace_bounds_m=WORKSPACE, mode="global",
        endpoint_public_tcp_targets={
            "left": [-0.16, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        },
        endpoint_position_tolerance_m=0.005,
        endpoint_orientation_tolerance_rad=0.05,
    )
    assert report["completed"] is False
    assert report["abort"]["code"] == "terminal_endpoint_tracking_timeout"
    assert report["abort"]["stage_index"] == 0


def test_timeout_stops_before_later_waypoint() -> None:
    adapter = FakeAdapter(gain=0.0)
    report, _arrays = execute_measured_convergence(
        adapter=adapter,
        paths={"left": {
            "position": [[0.2, 0.0], [0.4, 0.0]],
            "velocity": [[0.1, 0.0], [0.1, 0.0]],
        }},
        all_arms=("left", "right"), profile={
            **PROFILE, "maximum_physics_steps_per_waypoint": 4,
        },
        workspace_bounds_m=WORKSPACE, mode="global",
    )
    assert report["completed"] is False
    assert report["abort"]["code"] == "waypoint_tracking_timeout"
    assert report["abort"]["stage_index"] == 0
    assert adapter.writes.count("left") == 4


def test_one_arm_timeout_stops_synchronized_dual_arm_tail() -> None:
    class OneArmStalledAdapter(FakeAdapter):
        def step_physics(self):
            self.steps += 1
            self.measured["left"] = self.targets["left"].copy()
            # The right plant intentionally never tracks. The shared
            # transaction must stop both arms before stage 1.

    adapter = OneArmStalledAdapter()
    report, _arrays = execute_measured_convergence(
        adapter=adapter,
        paths={
            "left": {
                "position": [[0.04, 0.0], [0.08, 0.0]],
                "velocity": [[0.1, 0.0], [0.1, 0.0]],
            },
            "right": {
                "position": [[-0.04, 0.0], [-0.08, 0.0]],
                "velocity": [[-0.1, 0.0], [-0.1, 0.0]],
            },
        },
        all_arms=("left", "right"),
        profile={**PROFILE, "maximum_physics_steps_per_waypoint": 3},
        workspace_bounds_m=WORKSPACE,
        mode="global",
    )
    assert report["completed"] is False
    assert report["abort"] == {
        "code": "waypoint_tracking_timeout",
        "stage_index": 0,
        "maximum_physics_steps": 3,
    }
    assert adapter.measured["left"].tolist() == pytest.approx([0.04, 0.0])
    assert adapter.targets["left"].tolist() == pytest.approx([0.04, 0.0])
    assert adapter.targets["right"].tolist() == pytest.approx([-0.04, 0.0])


def test_gross_execution_fault_preserves_partial_numeric_evidence() -> None:
    adapter = FakeAdapter(gain=1.0)
    with pytest.raises(MotionExecutionFault) as raised:
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.2, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile={
                **PROFILE, "maximum_tcp_step_m": 0.05,
            },
            workspace_bounds_m=WORKSPACE, mode="global",
        )
    assert raised.value.report["abort"]["code"] == "gross_tcp_step"
    assert raised.value.report["sampled_steps"] == 1
    assert raised.value.arrays["issued_left_arm__position"].shape == (1, 2)


def test_measured_state_guard_runs_before_endpoint_completion() -> None:
    adapter = FakeAdapter(gain=1.0)
    with pytest.raises(MotionExecutionFault) as raised:
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile=PROFILE,
            workspace_bounds_m=WORKSPACE, mode="local",
            measured_state_guard=lambda measured: (
                {"code": "inter_arm_collision", "arm": "right"}
                if measured["left"][0] == pytest.approx(0.04) else None
            ),
        )
    assert raised.value.report["completed"] is False
    assert raised.value.report["abort"]["code"] == "inter_arm_collision"
    assert adapter.writes == ["left"]


@pytest.mark.parametrize(
    "workspace",
    [
        {"x": [], "y": [-1.0, 1.0], "z": [0.0, 2.0]},
        {"x": [1.0, -1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]},
        {"x": [float("nan"), 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]},
        {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
    ],
)
def test_workspace_contract_fails_before_any_controller_write(workspace) -> None:
    adapter = FakeAdapter()
    with pytest.raises(MotionTransactionError, match="workspace"):
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile=PROFILE,
            workspace_bounds_m=workspace, mode="global",
        )
    assert adapter.writes == []
    assert adapter.steps == 0


def test_post_step_joint_shape_drift_is_partial_execution_fault() -> None:
    class ShapeDriftAdapter(FakeAdapter):
        def step_physics(self):
            super().step_physics()
            self.measured["right"] = np.zeros(3)

    adapter = ShapeDriftAdapter()
    with pytest.raises(MotionExecutionFault) as raised:
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile=PROFILE,
            workspace_bounds_m=WORKSPACE, mode="global",
        )
    assert raised.value.report["abort"]["code"] == "invalid_measured_state"
    assert raised.value.report["execution_started"] is True
    assert raised.value.report["controller_write_attempts"] == 1
    assert raised.value.report["sampled_steps"] == 1


def test_guard_exception_is_wrapped_with_partial_execution_evidence() -> None:
    adapter = FakeAdapter()

    def broken_guard(_measured):
        raise ValueError("broken FK")

    with pytest.raises(MotionExecutionFault) as raised:
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile=PROFILE,
            workspace_bounds_m=WORKSPACE, mode="local",
            measured_state_guard=broken_guard,
        )
    assert raised.value.report["abort"] == {
        "code": "measured_state_guard_fault",
        "stage_index": 0,
        "detail": "ValueError",
    }
    assert raised.value.report["execution_started"] is True
    assert raised.value.arrays["issued_left_arm__position"].shape == (1, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_physics_steps_per_waypoint", True),
        ("maximum_physics_steps_per_waypoint", 8.7),
        ("maximum_total_physics_steps", "100"),
        ("joint_convergence_tolerance_rad", "0.03"),
    ],
)
def test_convergence_profile_rejects_coercible_noncanonical_numbers(
    field, value,
) -> None:
    adapter = FakeAdapter()
    with pytest.raises(MotionTransactionError, match="invalid"):
        execute_measured_convergence(
            adapter=adapter,
            paths={"left": {
                "position": [[0.04, 0.0]], "velocity": [[0.1, 0.0]],
            }},
            all_arms=("left", "right"), profile={**PROFILE, field: value},
            workspace_bounds_m=WORKSPACE, mode="global",
        )
    assert adapter.writes == []


def test_status_requires_measured_endpoint() -> None:
    status, errors = classify_motion_status(
        convergence_completed=True,
        target_poses={"left": [0, 0, 0.8, 0, 0, 0, 1]},
        measured_poses={"left": [0.02, 0, 0.8, 0, 0, 0, 1]},
        position_tolerance_m=0.005, orientation_tolerance_rad=0.05,
    )
    assert status == "not_reached"
    assert errors["left"]["translation_m"] == pytest.approx(0.02)
    rejected, _ = classify_motion_status(
        plan_rejected=True, position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    )
    assert rejected == "plan_rejected"
    with pytest.raises(MotionTransactionError, match="quaternion is degenerate"):
        classify_motion_status(
            convergence_completed=True,
            target_poses={"left": [0, 0, 0.8, 0, 0, 0, 0]},
            measured_poses={"left": [0, 0, 0.8, 0, 0, 0, 1]},
            position_tolerance_m=0.005, orientation_tolerance_rad=0.05,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_tolerance_m", float("nan")),
        ("position_tolerance_m", 0.0),
        ("orientation_tolerance_rad", float("inf")),
        ("orientation_tolerance_rad", -0.1),
    ],
)
def test_status_endpoint_tolerances_fail_closed(field, value) -> None:
    arguments = {
        "convergence_completed": True,
        "target_poses": {"left": [0, 0, 0.8, 0, 0, 0, 1]},
        "measured_poses": {"left": [0, 0, 0.8, 0, 0, 0, 1]},
        "position_tolerance_m": 0.005,
        "orientation_tolerance_rad": 0.05,
        field: value,
    }
    with pytest.raises(MotionTransactionError, match="tolerances are invalid"):
        classify_motion_status(**arguments)


def test_backend_neutral_artifact_is_numeric_and_hash_bound(tmp_path: Path) -> None:
    reference = write_motion_plan_artifact(
        tmp_path, transaction_id="action-0001", backend_id="fake",
        embodiment_id="fake_dual", binding={"source": "test"},
        validation={"status": "accepted"},
        arrays={"planned_joint_position": [[0.0, 0.1]]},
    )
    manifest = json.loads((tmp_path / reference["path"]).read_text())
    assert reference["schema"] == MOTION_PLAN_ARTIFACT_SCHEMA
    assert manifest["backend_id"] == "fake"
    with pytest.raises(MotionTransactionError):
        write_motion_plan_artifact(
            tmp_path, transaction_id="action-0002", backend_id="fake",
            embodiment_id="fake_dual", binding={}, validation={},
            arrays={"bad": [float("nan")]},
        )


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("input_min", lambda value: value[:-1]),
        ("input_max", lambda value: [value[0] - 2.0, *value[1:]]),
        ("control_delta", lambda _value: False),
        ("dimensions", lambda value: ["bad", *value[1:]]),
        ("servo_clip", lambda value: {**value, "translation_m": 0.05}),
        (
            "gripper_control",
            lambda value: {**value, "pose_action_input": 1.0},
        ),
    ],
)
def test_libero_osc_contract_rejects_hash_valid_malformed_content(
    field, mutate,
) -> None:
    contract = libero_osc_action_contract({
        "control_delta": True,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    })
    malformed = json.loads(json.dumps(contract))
    malformed[field] = mutate(malformed[field])
    body = {
        key: value for key, value in malformed.items()
        if key != "contract_sha256"
    }
    malformed["contract_sha256"] = hashlib.sha256(json.dumps(
        body, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    with pytest.raises(LiberoOscContractError):
        validate_libero_osc_action_contract(malformed)


def test_motion_v3_rejects_policy_selected_controller_and_servo_delta() -> None:
    base = {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": "observation-1",
        "action_id": "action-1",
        "bound_observation_sha256": "digest",
    }
    valid = validate_runtime_action({
        **base,
        "action": "move_pose",
        "arguments": {
            "targets": {
                "left": {
                    "position_m": [0.0, 0.0, 0.8],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "interactions": {"left": "precision"},
        },
    })
    assert valid["arguments"]["interactions"] == {"left": "precision"}
    with pytest.raises(ActionValidationError):
        validate_runtime_action({
            **base,
            "action": "move_pose",
            "arguments": {
                **valid["arguments"], "execution_mode": "local_cartesian",
            },
        })
    with pytest.raises(ActionValidationError):
        validate_runtime_action({
            **base,
            "action": "servo_delta",
            "arguments": {"targets": {}},
        })


def test_motion_strategy_v2_routes_intent_without_exposing_controller_choice() -> None:
    control = load_embodiment_control_profile(V10_PROFILE, V10_PROFILE_SHA256)[
        "profile"
    ]
    assert control["local_cartesian"]["jacobian_backend"] == (
        "sapien_pinocchio_spatial_root_to_public_tcp"
    )
    assert control["local_cartesian"]["sapien_jacobian_frame"] == (
        "articulation_root_spatial_about_root_origin"
    )
    assert control["local_cartesian"][
        "maximum_numerical_crosscheck_relative_error"
    ] == 0.02
    assert control["local_cartesian"]["maximum_joint_step_rad"] == 0.04
    assert control["motion_transaction"]["local_convergence"][
        "joint_convergence_tolerance_rad"
    ] == 0.015
    assert control["ik_topk_planning"][
        "validated_exact_ik_joint_segment"
    ] == {
        "nearby_distance_rad": 0.6,
        "maximum_distance_rad": 3.0,
        "validation_joint_step_rad": 0.005,
        "execution_joint_step_rad": 0.025,
        "constraint_backend": "curobo_motion_gen_check_constraints",
    }
    profile = control["strategy_router"]
    start_pose = [-0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]

    def route(target, interaction):
        return select_motion_strategy(
            action="move_pose",
            start_public_tcp_poses={"left": start_pose},
            target_public_tcp_poses={"left": target},
            interactions={"left": interaction},
            capabilities=FakeAdapter.capabilities,
            profile=profile,
        )

    assert route(
        [-0.17, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], "free_space",
    ).strategy == "global"
    precision = route(
        [-0.17, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], "precision",
    )
    assert precision.strategy == "global_then_local_settle"
    assert precision.endpoint_settle_enabled is True
    assert route(
        [-0.195, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], "contact_approach",
    ).strategy == "local"

    staged = route(
        [-0.16, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], "contact_approach",
    )
    assert staged.strategy == "global_precontact_then_local"
    assert staged.endpoint_settle_enabled is False
    assert staged.precontact_public_tcp_poses is not None
    precontact = staged.precontact_public_tcp_poses["left"]
    assert np.linalg.norm(
        np.asarray(precontact[:3]) - np.asarray([-0.16, 0, 0.8])
    ) == pytest.approx(0.003)

    # A target reconstructed from a fresh measured snapshot may differ from
    # the nominal 10 mm boundary by sub-micron floating-point noise.  It must
    # retain the direct-local route instead of becoming a staged global move.
    boundary = route(
        [-0.1899994, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        "contact_approach",
    )
    assert boundary.strategy == "local"

    angle = np.deg2rad(10.0)
    rotation = route(
        [-0.2, 0.0, 0.8, np.sin(angle / 2.0), 0.0, 0.0, np.cos(angle / 2.0)],
        "precision",
    )
    assert rotation.strategy == "local"
    assert rotation.reason == "bounded_in_place_rotation"
    rotation_return = route(
        [-0.197, 0.0, 0.8, np.sin(angle / 2.0), 0.0, 0.0, np.cos(angle / 2.0)],
        "precision",
    )
    assert rotation_return.strategy == "local"
    assert rotation_return.reason == "bounded_in_place_rotation"
    outside_rotation_drift = route(
        [-0.194, 0.0, 0.8, np.sin(angle / 2.0), 0.0, 0.0, np.cos(angle / 2.0)],
        "precision",
    )
    assert outside_rotation_drift.strategy == "global_then_local_settle"


def test_precision_post_global_settle_repairs_only_bounded_residual_band() -> None:
    stage = {"left": {"translation_m": 0.006, "orientation_rad": 0.02}}
    assert select_post_global_phase(
        staged_contact=False,
        tracking_completed=True,
        stage_endpoint_errors=stage,
        final_endpoint_errors=stage,
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
    ) == "run_local"
    assert select_post_global_phase(
        staged_contact=False,
        tracking_completed=True,
        stage_endpoint_errors=stage,
        final_endpoint_errors={
            "left": {"translation_m": 0.011, "orientation_rad": 0.02},
        },
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
    ) == "outside_settle_envelope"


def test_staged_contact_requires_precontact_arrival_before_local_phase() -> None:
    assert select_post_global_phase(
        staged_contact=True,
        tracking_completed=True,
        stage_endpoint_errors={
            "left": {"translation_m": 0.006, "orientation_rad": 0.01},
        },
        final_endpoint_errors={
            "left": {"translation_m": 0.009, "orientation_rad": 0.01},
        },
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
    ) == "precontact_global_not_reached"
    assert select_post_global_phase(
        staged_contact=True,
        tracking_completed=True,
        stage_endpoint_errors={
            "left": {"translation_m": 0.006, "orientation_rad": 0.01},
        },
        final_endpoint_errors={
            "left": {"translation_m": 0.009, "orientation_rad": 0.01},
        },
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
        stage_arrival_position_tolerance_m=0.007,
        stage_arrival_orientation_tolerance_rad=0.05,
    ) == "run_local"
    assert select_post_global_phase(
        staged_contact=True,
        tracking_completed=True,
        stage_endpoint_errors={
            "left": {"translation_m": 0.002, "orientation_rad": 0.01},
        },
        final_endpoint_errors={
            "left": {"translation_m": 0.0049, "orientation_rad": 0.02},
        },
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
        stage_arrival_position_tolerance_m=0.007,
        stage_arrival_orientation_tolerance_rad=0.05,
        final_arrival_position_tolerance_m=0.003,
        final_arrival_orientation_tolerance_rad=0.03,
    ) == "run_local"
    assert select_post_global_phase(
        staged_contact=True,
        tracking_completed=False,
        stage_endpoint_errors={
            "left": {"translation_m": 0.001, "orientation_rad": 0.01},
        },
        final_endpoint_errors={
            "left": {"translation_m": 0.009, "orientation_rad": 0.01},
        },
        arrival_position_tolerance_m=0.005,
        arrival_orientation_tolerance_rad=0.05,
        local_translation_envelope_m=0.01,
        local_rotation_envelope_rad=np.deg2rad(5.0),
    ) == "global_tracking_not_converged"


def test_motion_strategy_v2_rejects_mixed_dual_arm_interactions() -> None:
    profile = load_embodiment_control_profile(V10_PROFILE, V10_PROFILE_SHA256)[
        "profile"
    ]["strategy_router"]
    with pytest.raises(MotionTransactionError, match="cannot mix"):
        select_motion_strategy(
            action="move_pose",
            start_public_tcp_poses={
                "left": [-0.2, 0, 0.8, 0, 0, 0, 1],
                "right": [0.2, 0, 0.8, 0, 0, 0, 1],
            },
            target_public_tcp_poses={
                "left": [-0.195, 0, 0.8, 0, 0, 0, 1],
                "right": [0.195, 0, 0.8, 0, 0, 0, 1],
            },
            interactions={"left": "contact_approach", "right": "free_space"},
            capabilities=FakeAdapter.capabilities,
            profile=profile,
        )


def test_v10_surface_exposes_intent_not_controller_choice() -> None:
    surface = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")
    assert surface.supports("motion_action_v3")
    assert surface.supports("motion_intent_router_v1")
    assert not surface.supports("local_cartesian_dls_v1")
    definition = next(
        item for item in surface.definitions if item["name"] == "move_pose"
    )
    serialized = json.dumps(definition, sort_keys=True)
    input_serialized = json.dumps(definition["inputSchema"], sort_keys=True)
    assert "contact_approach" in serialized
    assert "execution_mode" not in input_serialized
    assert "servo_delta" not in input_serialized
    result = {
        "candidates": [{
            "approach_delta": {
                "frame": "end_effector",
                "translation_m": [0.04, 0.0, 0.0],
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
        }],
    }
    _project_copyable_grasp_approach_targets(
        result, semantic_interaction=True,
    )
    assert result["candidates"][0]["approach_delta"]["interaction"] == (
        "contact_approach"
    )


def test_v9_grasp_approach_is_directly_copyable_into_move_pose_target() -> None:
    result = {
        "candidates": [{
            "approach_delta": {
                "frame": "end_effector",
                "translation_m": [0.04, 0.0, 0.0],
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
        }],
    }
    _project_copyable_grasp_approach_targets(result)
    assert result["candidates"][0]["approach_delta"] == {
        "position": {
            "mode": "delta",
            "translation_m": [0.04, 0.0, 0.0],
            "frame": "end_effector",
        },
        "orientation": {"mode": "hold_current"},
    }


def test_native_projection_accepts_v6_endpoint_verified_receipt() -> None:
    receipt = {
        "schema": "ora.robotwin-sparse-motion-receipt.v6",
        "action": "move_pose",
        "arguments": {},
        "target_arms": ["left"],
        "execution_status": "not_reached",
        "execution_fault": None,
        "state_changed": True,
        "terminal_after_action": False,
        "synchronization": {},
        "controller_tracking": {
            "schema": "ora.motion-execution.v1",
            "completed": True,
        },
        "arms": {
            arm: {
                "targeted": arm == "left", "planner": {}, "pose_error": None,
                "inactive_arm_drift": None, "command_preservation": {},
                "hold_controller": {}, "motion_backend": None,
                "plan_validation": (
                    {
                        "schema": "ora.motion-plan-validation.v1",
                        "status": "accepted",
                    }
                    if arm == "left" else None
                ),
                "plan_artifact": (
                    {
                        "schema": "ora.motion-plan-artifact.v1",
                        "path": "motion-transactions/private/manifest.json",
                        "sha256": "private-control-hash",
                    }
                    if arm == "left" else None
                ),
            }
            for arm in ("left", "right")
        },
    }
    effect = NativePlanner._minimal_action_effect(
        object.__new__(NativePlanner), receipt,
    )
    assert effect["status"] == "not_reached"
    assert effect["controller_tracking"] == {"completed": True}
    assert effect["arms"]["left"]["plan_validation"] == {
        "status": "accepted",
    }
    assert effect["arms"]["left"]["plan_artifact_recorded"] is True
    assert effect["arms"]["right"]["plan_artifact_recorded"] is False
    serialized = json.dumps(effect, sort_keys=True)
    assert "motion-transactions/private" not in serialized
    assert "private-control-hash" not in serialized


def test_native_projection_accepts_v7_motion_strategy_without_control_provenance() -> None:
    receipt = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action_id": "action-rotate-1",
        "action": "move_pose",
        "arguments": {},
        "target_arms": ["right"],
        "execution_status": "reached",
        "failure_stage": "execution",
        "failure_code": "local_servo_stalled",
        "execution_fault": None,
        "state_changed": True,
        "terminal_after_action": False,
        "synchronization": {},
        "motion_strategy": {
            "schema": "ora.motion-strategy-decision.v1",
            "strategy": "global_then_local_settle",
            "interactions": {"right": "free_space"},
            "physical_delta": {
                "right": {"translation_m": 0.12, "orientation_rad": 0.0},
            },
            "settle_attempted": True,
            "settle_result": "reached",
        },
        "controller_tracking": {
            "completed": False,
            "backend": "robotwin",
            "mode": "local_cartesian_dls",
            "sampled_steps": 12,
            "planned_stages": 60,
            "abort": {
                "code": "local_servo_stalled",
                "arm": "right",
                "iteration": 8,
                "maximum_iterations": 60,
                "maximum_physics_steps": 64,
            },
            "arms": {
                "right": {
                    "targeted": True,
                    "target_error": {
                        "translation_m": 0.01,
                        "orientation_rad": 0.02,
                    },
                    "execution_mode": "local_cartesian_dls",
                },
            },
        },
        "arms": {
            arm: {
                "targeted": arm == "right", "planner": {}, "pose_error": None,
                "inactive_arm_drift": None, "command_preservation": {},
                "hold_controller": {}, "motion_backend": "global_curobo",
                "plan_validation": (
                    {
                        "schema": "ora.motion-plan-validation.v1",
                        "status": "accepted",
                        "backend_id": "robotwin",
                        "joint_order": ["private_joint"],
                        "planning_endpoint_error": {
                            "translation_m": 0.001,
                            "orientation_rad": 0.01,
                        },
                    }
                    if arm == "right" else None
                ),
                "plan_artifact": None,
            }
            for arm in ("left", "right")
        },
    }
    planner = object.__new__(NativePlanner)
    planner.engine = SimpleNamespace(
        last_action_id="action-rotate-1",
        last_action_public_name="rotate_wrist",
    )
    effect = NativePlanner._minimal_action_effect(planner, receipt)
    assert effect["action"] == "rotate_wrist"
    assert effect["motion_strategy"]["strategy"] == "global_then_local_settle"
    assert effect["motion_strategy"]["settle_result"] == "reached"
    assert effect["failure_code"] == "controller_stalled"
    assert effect["controller_tracking"]["abort"] == {
        "code": "controller_stalled", "arm": "right",
    }
    assert effect["arms"]["right"]["plan_validation"] == {
        "status": "accepted",
        "planning_endpoint_error": {
            "translation_m": 0.001,
            "orientation_rad": 0.01,
        },
    }
    serialized = json.dumps(effect, sort_keys=True)
    assert "motion-strategy-decision.v1" not in serialized
    for private in (
        "backend_id", "global_curobo", "local_cartesian_dls", "OSC",
        "maximum_iterations", "maximum_physics_steps", "joint_order",
    ):
        assert private not in serialized


def test_environment_receipt_rejects_mismatched_submitted_action_id() -> None:
    planner = object.__new__(NativePlanner)
    planner.engine = SimpleNamespace(last_action_id="action-expected")
    planner.artifacts = SimpleNamespace(put_json=lambda *_args, **_kwargs: {
        "sha256": "receipt-hash",
    })
    with pytest.raises(PolicySessionError, match="does not match"):
        planner.record_environment_receipt({"action_id": "action-delayed"})


def _sparse_v3_session(tmp_path: Path) -> Any:
    """A minimal live-session shell for the v3 dispatch (EXP08A wiring v1)."""

    import numpy as np

    from ora.environment.backends.libero_pro.session import (
        LiberoProSession,
    )

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.terminated = False
    session.truncated = False
    session.world_mutation_sequence = 0
    session.world_mutation_actions = []
    session._current_state = lambda: (  # type: ignore[method-assign]
        np.asarray([0.0, 0.0, 0.8]),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        0.6,
    )
    session._remember_backtrack_state = lambda: None  # type: ignore[method-assign]
    return session


def _sparse_v3_action(
    *, interaction: str, position: list[float],
    orientation: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v3",
        "observation_id": "observation-0",
        "action_id": "action-0",
        "bound_observation_sha256": "0" * 64,
        "action": "move_pose",
        "arguments": {
            "targets": {"right": {
                "position_m": position,
                "orientation_xyzw": orientation or [0.0, 0.0, 0.0, 1.0],
            }},
            "interactions": {"right": interaction},
        },
    }


def test_libero_sparse_v3_routes_to_the_qualified_executors(
    tmp_path: Path,
) -> None:
    """free_space→global planned; small precision→local OSC; one receipt each."""

    calls: list[tuple[str, dict[str, Any]]] = []

    session = _sparse_v3_session(tmp_path / "global")

    def fake_planned(action: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(("planned", dict(action)))
        return session._persist_receipt({
            "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
            "action_id": action["action_id"], "action": action["action"],
            "arguments": action["arguments"], "consumed": True,
            "state_changed": True, "motion_status": "reached",
            "planner": {"private": "path"},
            "feedback": {"stop_reason": None},
            "before": {"right": {"pose_xyz_xyzw": [0, 0, 0.8, 0, 0, 0, 1]}},
            "after": {"right": {
                "pose_xyz_xyzw": [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            }},
            "steps_used": 5, "sim_steps_total": 5,
            "control_health": {"status": "healthy", "codes": []},
        })

    session._execute_planned_motion = fake_planned  # type: ignore[method-assign]
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="free_space", position=[0.1, 0.0, 0.8],
    ))
    assert receipt["schema"] == (
        "ora.libero-sparse-motion-receipt.v1"
    )
    assert receipt["execution_status"] == "reached"
    # The decided strategy is reported even when the settle turns out to be
    # unnecessary; the executor list says what actually ran.
    assert receipt["motion_strategy"] == "global_then_local_settle"
    assert receipt["executor"] == ["libero_curobo_joint_position"]
    assert receipt["arms"]["right"]["settle"] == {
        "attempted": False, "reason": "global_route_met_the_local_bound",
    }
    assert receipt["arms"]["right"]["pose_error"]["translation_m"] == (
        pytest.approx(0.0)
    )
    assert receipt["segment"]["planner"] == {"private": "path"}
    assert calls[0][1]["action"] == "move_free_space"
    persisted = json.loads(
        (tmp_path / "global/receipts/action-0000.json").read_text()
    )
    assert persisted["schema"] == receipt["schema"]
    assert session.action_sequence == 1

    local = _sparse_v3_session(tmp_path / "local")

    def fake_execute(payload: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(("servo", dict(payload)))
        return local._persist_receipt({
            "schema": "ora.libero-pro-motion-receipt.v2",
            "action_id": payload["action_id"], "action": payload["action"],
            "arguments": payload["arguments"], "consumed": True,
            "state_changed": True, "motion_status": "stopped",
            "feedback": {"stop_reason": "waypoint_tracking_timeout"},
            "before": {"right": {"pose_xyz_xyzw": [0, 0, 0.8, 0, 0, 0, 1]}},
            "after": {"right": {
                "pose_xyz_xyzw": [0.004, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            }},
            "steps_used": 20, "sim_steps_total": 20,
            "control_health": {"status": "degraded", "codes": []},
        })

    local.execute = fake_execute  # type: ignore[method-assign]
    receipt = local._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="precision", position=[0.006, 0.0, 0.8],
    ))
    assert receipt["execution_status"] == "not_reached"
    assert receipt["motion_strategy"] == "local"
    assert receipt["executor"] == ["libero_robosuite_osc_pose"]
    assert receipt["failure_stage"] == "execution"
    assert receipt["failure_code"] == "waypoint_tracking_timeout"
    assert calls[-1][1]["action"] == "servo_delta"
    assert calls[-1][1]["arguments"]["max_iterations"] == 20


def _settle_session(tmp_path: Path, *, landed: list[float], status: str = "reached") -> Any:
    """A v3 session whose global route lands exactly where the test says."""

    session = _sparse_v3_session(tmp_path)
    dispatched: list[dict[str, Any]] = []

    def fake_planned(action: Mapping[str, Any]) -> dict[str, Any]:
        dispatched.append(dict(action))
        return session._persist_receipt({
            "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
            "action_id": action["action_id"], "action": action["action"],
            "arguments": action["arguments"], "consumed": True,
            "state_changed": True, "motion_status": status,
            "planner": {"private": "path"},
            "feedback": {"stop_reason": None},
            "endpoint_contract": {
                "schema": "ora.libero-endpoint-contract.v1",
                "execution_route": "curobo_global_joint_position",
                "within_tolerance": status == "reached",
            },
            "before": {"right": {"pose_xyz_xyzw": [0, 0, 0.8, 0, 0, 0, 1]}},
            "after": {"right": {"pose_xyz_xyzw": [*landed, 0.0, 0.0, 0.0, 1.0]}},
            "steps_used": 5, "sim_steps_total": 5,
            "control_health": {"status": "healthy", "codes": []},
        })

    def fake_execute(payload: Mapping[str, Any]) -> dict[str, Any]:
        dispatched.append(dict(payload))
        target = payload["arguments"]["target_pose"]["position_m"]
        return session._persist_receipt({
            "schema": "ora.libero-pro-motion-receipt.v2",
            "action_id": payload["action_id"], "action": payload["action"],
            "arguments": payload["arguments"], "consumed": True,
            "state_changed": True, "motion_status": "reached",
            "feedback": {"stop_reason": None},
            "endpoint_contract": {
                "schema": "ora.libero-endpoint-contract.v1",
                "execution_route": "osc_local_pose", "within_tolerance": True,
            },
            "before": {"right": {"pose_xyz_xyzw": [*landed, 0.0, 0.0, 0.0, 1.0]}},
            "after": {"right": {"pose_xyz_xyzw": [*target, 0.0, 0.0, 0.0, 1.0]}},
            "steps_used": 7, "sim_steps_total": 7,
            "control_health": {"status": "healthy", "codes": []},
        })

    session._execute_planned_motion = fake_planned  # type: ignore[method-assign]
    session.execute = fake_execute  # type: ignore[method-assign]
    return session, dispatched


def test_libero_sparse_v3_settles_locally_when_the_global_route_leaves_a_residual(
    tmp_path: Path,
) -> None:
    """A global route inside its own bound is still 12 mm short of the local one."""

    session, dispatched = _settle_session(tmp_path, landed=[0.288, 0.0, 0.8])
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="free_space", position=[0.3, 0.0, 0.8],
    ))
    assert receipt["motion_strategy"] == "global_then_local_settle"
    assert receipt["executor"] == [
        "libero_curobo_joint_position", "libero_robosuite_osc_pose",
    ]
    assert [item["action"] for item in dispatched] == [
        "move_free_space", "servo_delta",
    ]
    # The settle closes the residual in task space; the global route alone
    # could only re-drive the same joint goal.
    assert receipt["arms"]["right"]["pose_error"]["translation_m"] == pytest.approx(0.0)
    assert receipt["execution_status"] == "reached"
    assert receipt["arms"]["right"]["endpoint_contract"]["execution_route"] == (
        "osc_local_pose"
    )
    # Both physical segments stay readable and their costs add up.
    assert [segment["action"] for segment in receipt["segments"]] == [
        "move_free_space", "servo_delta",
    ]
    assert receipt["steps_used"] == 12
    assert receipt["sim_steps_total"] == 12


def test_libero_sparse_v3_skips_the_settle_when_the_global_route_already_lands(
    tmp_path: Path,
) -> None:
    """Inside the local bound there is nothing to close, so no second segment."""

    session, dispatched = _settle_session(tmp_path, landed=[0.298, 0.0, 0.8])
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="free_space", position=[0.3, 0.0, 0.8],
    ))
    assert [item["action"] for item in dispatched] == ["move_free_space"]
    assert receipt["executor"] == ["libero_curobo_joint_position"]
    assert receipt["execution_status"] == "reached"
    assert receipt["arms"]["right"]["endpoint_contract"]["execution_route"] == (
        "curobo_global_joint_position"
    )


def test_libero_sparse_v3_does_not_settle_outside_the_servo_envelope(
    tmp_path: Path,
) -> None:
    """A 300 mm miss is not a settle; attempting one would be refused anyway."""

    session, dispatched = _settle_session(
        tmp_path, landed=[0.0, 0.0, 0.8], status="stopped",
    )
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="free_space", position=[0.3, 0.0, 0.8],
    ))
    assert [item["action"] for item in dispatched] == ["move_free_space"]
    assert receipt["execution_status"] == "not_reached"
    assert receipt["arms"]["right"]["settle"] == {
        "attempted": False,
        "reason": "residual_outside_local_servo_envelope",
    }


def test_libero_publishes_the_commanded_gripper_beside_the_encoder(
    tmp_path: Path,
) -> None:
    """The directive the controller is holding is a measurement, not a guess."""

    from ora.environment.backends.libero_pro.motion_settle import (
        public_gripper_command as _public_gripper_command,
    )

    # -1 holds the jaws open, +1 holds them closed; the encoding is the one
    # the executor compiles, not a convention invented here.
    assert _public_gripper_command(-1.0, "open") == {
        "gripper_command": 1.0, "gripper_command_target": "open",
    }
    assert _public_gripper_command(1.0, "close") == {
        "gripper_command": 0.0, "gripper_command_target": "close",
    }
    # A partially held directive is reported as the number it is.
    assert _public_gripper_command(0.0, "unknown") == {
        "gripper_command": 0.5, "gripper_command_target": "unknown",
    }


def test_libero_gripper_target_is_unknown_until_one_is_accepted(
    tmp_path: Path,
) -> None:
    """A reset directive is a hold, not evidence that anything was commanded."""

    session = _sparse_v3_session(tmp_path)
    assert session.public_gripper_command_target() == "unknown"
    session._persist_receipt({
        "schema": "ora.libero-pro-motion-receipt.v2",
        "action_id": "action-0", "action": "set_gripper",
        "arguments": {"arm": "right", "gripper": 0.0},
        "reached_gripper": 0.02, "consumed": True, "state_changed": True,
        "motion_status": "reached", "feedback": {"stop_reason": None},
    })
    assert session.public_gripper_command_target() == "close"
    session._persist_receipt({
        "schema": "ora.libero-pro-motion-receipt.v2",
        "action_id": "action-1", "action": "set_gripper",
        "arguments": {"arm": "right", "gripper": 1.0},
        "reached_gripper": 0.95, "consumed": True, "state_changed": True,
        "motion_status": "reached", "feedback": {"stop_reason": None},
    })
    assert session.public_gripper_command_target() == "open"


def test_libero_pose_motion_does_not_relabel_the_gripper_target(
    tmp_path: Path,
) -> None:
    """move_pose holds the current directive; it commands no new jaw target."""

    session = _sparse_v3_session(tmp_path)
    session._gripper_command_target = "close"
    session._persist_receipt({
        "schema": "ora.libero-pro-motion-receipt.v2",
        "action_id": "action-0", "action": "move_free_space",
        "arguments": {"arm": "right"}, "consumed": True,
        "state_changed": True, "motion_status": "reached",
        "feedback": {"stop_reason": None},
    })
    assert session.public_gripper_command_target() == "close"


def test_libero_precontact_pose_leaves_the_local_segment_inside_the_servo_envelope() -> None:
    """The global route's own bound is the reserve, so its miss cannot overflow."""

    from ora.environment.backends.libero_pro.motion_settle import (
        precontact_pose, precontact_refusal,
    )

    pose = precontact_pose(
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
    )
    # 50 mm servo envelope minus the 15 mm global endpoint bound.
    assert pose == pytest.approx([0.165, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0])
    assert precontact_refusal(
        terminated=False, motion_status="refused", endpoint_measured=True,
        within_local_envelope=True,
    ) == "precontact_segment_refused"
    assert precontact_refusal(
        terminated=False, motion_status="reached", endpoint_measured=False,
        within_local_envelope=True,
    ) == "measured_endpoint_unavailable"
    assert precontact_refusal(
        terminated=False, motion_status="reached", endpoint_measured=True,
        within_local_envelope=True,
    ) is None


def test_libero_sparse_v3_stages_a_large_contact_approach(tmp_path: Path) -> None:
    """Global to a pre-contact pose, then the bounded local approach, as RoboTwin."""

    session, dispatched = _settle_session(tmp_path, landed=[0.165, 0.0, 0.8])
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="contact_approach", position=[0.2, 0.0, 0.8],
    ))
    assert receipt["motion_strategy"] == "global_precontact_then_local"
    assert [item["action"] for item in dispatched] == [
        "move_free_space", "servo_delta",
    ]
    assert dispatched[0]["arguments"]["target_pose"]["position_m"] == (
        pytest.approx([0.165, 0.0, 0.8])
    )
    # The contact itself is made by the bounded local segment only.
    translation = dispatched[1]["arguments"]["delta_pose"]["translation_m"]
    assert translation == pytest.approx([0.035, 0.0, 0.0])
    assert receipt["executor"] == [
        "libero_curobo_joint_position", "libero_robosuite_osc_pose",
    ]
    assert receipt["execution_status"] == "reached"
    assert receipt["arms"]["right"]["interaction"] == "contact_approach"
    assert receipt["arms"]["right"]["precontact"]["attempted"] is True
    assert receipt["arms"]["right"]["precontact"]["reason"] is None
    assert receipt["arms"]["right"]["precontact"]["pose_xyz_xyzw"] == (
        pytest.approx([0.165, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0])
    )
    assert "settle" not in receipt["arms"]["right"]
    assert receipt["steps_used"] == 12


def test_libero_sparse_v3_stops_a_staged_approach_the_global_route_left_short(
    tmp_path: Path,
) -> None:
    """A pre-contact segment that did not arrive does not license touching."""

    session, dispatched = _settle_session(
        tmp_path, landed=[0.05, 0.0, 0.8], status="stopped",
    )
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="contact_approach", position=[0.2, 0.0, 0.8],
    ))
    assert [item["action"] for item in dispatched] == ["move_free_space"]
    assert receipt["execution_status"] == "not_reached"
    assert receipt["arms"]["right"]["precontact"]["attempted"] is False
    assert receipt["arms"]["right"]["precontact"]["reason"] == (
        "precontact_segment_not_reached"
    )


def test_libero_sparse_v3_does_not_report_a_precontact_arrival_as_the_target(
    tmp_path: Path,
) -> None:
    """Arriving at the pre-contact pose is not arriving at the requested one."""

    session, dispatched = _settle_session(tmp_path, landed=[0.165, 0.0, 0.8])
    planned = session._execute_planned_motion

    def terminating(action: Mapping[str, Any]) -> dict[str, Any]:
        session.terminated = True
        return planned(action)

    session._execute_planned_motion = terminating  # type: ignore[method-assign]
    receipt = session._execute_sparse_motion_v3(_sparse_v3_action(
        interaction="contact_approach", position=[0.2, 0.0, 0.8],
    ))
    assert [item["action"] for item in dispatched] == ["move_free_space"]
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_stage"] == "precontact"
    assert receipt["failure_code"] == "episode_terminated_before_contact_approach"
    assert receipt["arms"]["right"]["pose_error"]["translation_m"] == (
        pytest.approx(0.035)
    )


def test_libero_sparse_projector_emits_the_unified_envelope() -> None:
    from ora.agent.native.receipt_projection import (
        ReceiptProjectionContext,
        resolve_receipt_effect_projector,
    )

    record = {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose",
        "action_id": "action-7",
        "execution_status": "not_reached",
        "motion_strategy": "global",
        "failure_stage": "execution",
        "failure_code": "waypoint_tracking_timeout",
        "executor": ["libero_curobo_joint_position"],
        "target_arms": ["right"],
        "synchronization": {"mode": "single_arm"},
        "arms": {"right": {
            "targeted": True,
            "pose_error": {"translation_m": 0.02, "orientation_rad": 0.1},
        }},
        "segment": {"planner": {"private": "path"}},
        "terminal_after_action": False,
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    assert projector is not None
    projected = projector(record, ReceiptProjectionContext(
        last_action_id="action-7", last_action_public_name="move_pose",
    ))
    assert projected["status"] == "not_reached"
    assert projected["executor"] == ["libero_curobo_joint_position"]
    assert projected["failure_code"] == "controller_timeout"
    assert projected["physical_cause"] == "unknown"
    assert projected["detail"]["arms"]["right"]["pose_error"][
        "translation_m"
    ] == 0.02
    assert "segment" not in projected["detail"]
    assert "private" not in json.dumps(projected)


from ora.agent.tools.robot import ToolError


class _FlatSurfaceStub:
    """Single-arm flat unified-surface capability set (M1/H3/G1 rulings)."""

    profile_id = "libero_unified_manip_stub"
    capabilities = (
        "single_arm_flat_targets_v1", "atomic_gripper_v1",
        "enhanced_move_pose_v1", "motion_action_v3",
        "motion_intent_router_v1",
    )
    mutation_tools = (
        "move_to", "move_pose", "rotate_wrist", "rotate_pitch",
        "recover_pose", "open_gripper", "close_gripper",
    )
    lifecycle_tools = ("finish_episode",)
    result_schemas: dict[str, Any] = {}
    view_config = None

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


def _flat_runtime(tmp_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    from ora.agent.tools.runtime import ManipulationToolRuntime

    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = _FlatSurfaceStub()
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
            "right": {"pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        }},
    }
    runtime._current_grip_state = lambda: (  # type: ignore[method-assign]
        control, tmp_path, observation, [0.2, 0.0, 1.0, 0, 0, 0, 1], 0.0,
    )
    persisted: list[dict[str, Any]] = []
    runtime._persist_action = (  # type: ignore[method-assign]
        lambda _path, _control, submitted: persisted.append(dict(submitted))
    )
    return runtime, persisted


def test_flat_single_arm_surface_compiles_to_the_shared_sparse_codec(
    tmp_path: Path,
) -> None:
    """M1 ruling: flat single-arm tool forms, shared sparse codec inside."""

    runtime, persisted = _flat_runtime(tmp_path)

    submitted = runtime._motion_move_pose_v2({
        "interaction": "free_space",
        "position": {"mode": "absolute", "position_m": [0.1, 0.0, 0.95]},
        "orientation": {"mode": "hold_current"},
    })
    assert submitted["status"] == "submitted"
    action = persisted[-1]
    assert action["schema"] == "ora-motion-action.v3"
    assert set(action["arguments"]["targets"]) == {"right"}
    assert action["arguments"]["interactions"] == {"right": "free_space"}

    with pytest.raises(ToolError, match="without\\s+a targets wrapper"):
        runtime._motion_move_pose_v2({
            "targets": {"right": {"position_m": [0.1, 0.0, 0.95],
                                  "orientation_xyzw": [0, 0, 0, 1]}},
        })

    recover = runtime._motion_recover_pose_v2({"strategy": "previous"})
    assert recover["status"] == "submitted"
    assert persisted[-1]["arguments"]["targets"] == {
        "right": "backtrack_last_safe",
    }


def test_atomic_gripper_tools_compile_open_and_close(tmp_path: Path) -> None:
    """H3/G1/G2 rulings: EXP01B-lineage atomic verbs over the v3 codec."""

    runtime, persisted = _flat_runtime(tmp_path)
    closed = runtime._atomic_close_gripper({})
    assert closed["action"] == "close_gripper"
    assert persisted[-1]["action"] == "set_gripper"
    assert persisted[-1]["arguments"]["targets"] == {"right": "close"}
    opened = runtime._atomic_open_gripper({})
    assert opened["action"] == "open_gripper"
    assert persisted[-1]["arguments"]["targets"] == {"right": "open"}
    with pytest.raises(ToolError, match="no arguments"):
        runtime._atomic_close_gripper({"arm": "right"})

    class _DualStub(_FlatSurfaceStub):
        capabilities = (
            "dual_arm_sparse_targets_v1", "atomic_gripper_v1",
            "enhanced_move_pose_v1", "motion_action_v3",
        )

    runtime.surface = _DualStub()
    dual_observation = {
        "proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": [-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
            "right": {"pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        }},
    }
    runtime._current_grip_state = lambda: (  # type: ignore[method-assign]
        {
            "observation_id": "observation-1",
            "observation_sha256": "1" * 64,
            "generation": 1, "open": True,
        },
        tmp_path, dual_observation, [0.2, 0.0, 1.0, 0, 0, 0, 1], 0.0,
    )
    closed = runtime._atomic_close_gripper({"arm": "left"})
    assert persisted[-1]["arguments"]["targets"] == {"left": "close"}
    with pytest.raises(ToolError, match="arm=left or arm=right"):
        runtime._atomic_close_gripper({})


def _probe_session(tmp_path: Path) -> Any:
    import numpy as np

    from ora.environment.backends.libero_pro.session import (
        LiberoProSession,
    )

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.terminated = False
    session.truncated = False
    session.world_mutation_sequence = 0
    session.world_mutation_actions = []
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.round_index = 0
    session.motion_profile = {}
    session.qualified_controller_binding = {}
    session.workspace_bounds_m = {
        "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
    }
    identity = [0.0, 0.0, 0.0, 1.0]
    session._motion_state = lambda: {  # type: ignore[method-assign]
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "joint_positions": [0.1] * 7,
        "base_pose_world_xyzw": [0.0, 0.0, 0.0] + identity,
        "hand_pose_world_xyzw": [0.2, 0.0, 0.9] + identity,
        "grip_site_pose_world_xyzw": [0.2, 0.0, 0.8] + identity,
    }
    session._current_state = lambda: (  # type: ignore[method-assign]
        np.asarray([0.2, 0.0, 0.8]), np.asarray(identity), 0.6,
    )
    session._remember_backtrack_state = lambda: None  # type: ignore[method-assign]

    class _Planner:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
            self.requests.append(dict(request))
            return {
                "reachable": True,
                "planner_status": "Success",
                "planning_time_s": 0.01,
                "start_ee_pose_base_xyzw": [0.2, 0.0, 0.9, 0, 0, 0, 1],
                "trajectory": {
                    "positions": [[0.1] * 7, [0.2] * 7],
                    "joint_names": request["joint_names"],
                    "dt_s": 0.004,
                },
            }

    session.motion_planner = _Planner()
    return session


def test_libero_probe_plans_stores_and_survives_gripper_only_writes(
    tmp_path: Path,
) -> None:
    """EXP08A: probe → plan_ref → gripper-only survival → execution."""

    session = _probe_session(tmp_path)
    result = session.probe_motion({
        "schema": "ora.motion-probe-request.v1",
        "observation_id": "observation-0",
        "observation_sha256": "0" * 64,
        "variant_policy": "parallel_jaw_equivalents",
        "queries": [{
            "query_id": "q0", "arm": "right",
            "pose": {
                "position_m": [0.25, 0.0, 0.85],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }],
    })
    assert result["schema"] == "ora.motion-probe-result.v1"
    assert result["robot_decisions_consumed"] == 0
    assert result["world_state_changed"] is False
    row = result["results"][0]
    assert row["reachable"] is True
    assert len(row["variants"]) == 2
    assert {v["variant_id"] for v in row["variants"]} == {
        "variant-00-exact-+0deg", "variant-01-jaw-flip-180deg",
    }
    plan_ref = row["plan_ref"]
    assert plan_ref in session.plan_references
    # jaw flip about approach +Z: q ⊗ [0,0,1,0] == [y,-x,w,-z]
    flip = [v for v in row["variants"] if "flip" in v["variant_id"]][0]
    assert flip["target_pose_xyzw"][3:] == [0.0, -0.0, 1.0, -0.0]
    # An artifact was written with hash-bound numeric arrays.
    artifacts = list((tmp_path / "motion-probes").rglob("*.json"))
    assert artifacts

    executed: list[dict[str, Any]] = []

    def fake_planned_motion(
        action: Mapping[str, Any], *, preplanned=None,
    ) -> dict[str, Any]:
        executed.append({"action": dict(action), "preplanned": preplanned})
        assert preplanned is not None
        planned, evidence = preplanned
        assert planned["trajectory"]["positions"] == [[0.1] * 7, [0.2] * 7]
        assert evidence["plan_ref"] == plan_ref
        return session._persist_receipt({
            "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
            "action_id": action["action_id"], "action": action["action"],
            "arguments": action["arguments"], "consumed": True,
            "state_changed": True, "motion_status": "reached",
            "feedback": {"stop_reason": None},
            "before": {"right": {"pose_xyz_xyzw": [0.2, 0, 0.8, 0, 0, 0, 1]}},
            "after": {"right": {
                "pose_xyz_xyzw": [0.25, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0],
            }},
            "steps_used": 3, "sim_steps_total": 3,
            "control_health": {"status": "healthy", "codes": []},
        })

    session._execute_planned_motion = fake_planned_motion  # type: ignore

    def plan_action() -> dict[str, Any]:
        return {
            "schema": "ora-motion-action.v3",
            "observation_id": "observation-1",
            "action_id": "action-1",
            "bound_observation_sha256": "1" * 64,
            "action": "move_pose",
            "arguments": {"plan_ref": plan_ref},
        }

    # A gripper-only world write keeps the plan alive.
    session.world_mutation_sequence = 1
    session.world_mutation_actions = ["set_gripper"]
    receipt = session._execute_sparse_motion_v3(plan_action())
    assert receipt["schema"] == (
        "ora.libero-sparse-motion-receipt.v1"
    )
    assert receipt["execution_status"] == "reached"
    assert receipt["executor"] == ["libero_curobo_joint_position"]
    assert receipt["arms"]["right"]["plan_ref"] == plan_ref
    assert receipt["arms"]["right"]["pose_error"]["translation_m"] == (
        pytest.approx(0.0)
    )
    # The executed world write expired the plan for any later use.
    assert session.world_mutation_actions[-1] == "move_pose"
    with pytest.raises(Exception, match="stale"):
        session._execute_sparse_motion_v3(plan_action())

    # Non-gripper intervening writes fail closed too.
    fresh = _probe_session(tmp_path / "fresh")
    fresh.plan_references = dict(session.plan_references)
    fresh.world_mutation_sequence = 1
    fresh.world_mutation_actions = ["move_to"]
    with pytest.raises(Exception, match="stale"):
        fresh._execute_sparse_motion_v3(plan_action())
