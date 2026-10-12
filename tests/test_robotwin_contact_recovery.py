"""Offline v10 clearance, measured blocking and source-bound recovery contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robotwin import session as session_module
from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected, ValidatedRoboTwinPlan, validate_inter_arm_clearance,
)
from ora.environment.backends.robotwin.session import RoboTwinLiveSession, _SparseMotionState
from ora.environment.controllers.embodiment_control import load_embodiment_control_profile
from ora.environment.controllers.inter_arm_clearance import InterArmClearancePolicy
from ora.environment.controllers.motion_transaction import (
    MotionBackendCapabilities, execute_measured_convergence,
)
from ora.environment.controllers.profile_capabilities import FROZEN_CONTROL_PROFILE_SHA256


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "ora/policy/embodiments/aloha-agilex"
WORKSPACE = {"x": [-1., 1.], "y": [-1., 1.], "z": [0., 2.]}


def binding():
    return load_embodiment_control_profile(PROFILE_ROOT / "control-profile-v10.json", FROZEN_CONTROL_PROFILE_SHA256[10])


def path(distances):
    points = np.asarray([[distance, 0, 1, 0, 0, 0, 1] for distance in distances], dtype=float)
    joints = points[:, :3]
    return ValidatedRoboTwinPlan(joints, np.zeros_like(joints), points, points, {"status": "accepted"})


def test_v10_binds_distinct_plan_execution_thresholds():
    policy = InterArmClearancePolicy.from_profile(binding()["profile"]["dual_arm_collision_check"], .1)
    assert policy.planning_center_distance_m == pytest.approx(.095)
    assert policy.execution_center_distance_m == pytest.approx(.09)
    assert not policy.assess(.093, planning=True)["allowed"]
    assert policy.assess(.093)["allowed"]


@pytest.mark.parametrize("distances,accepted", [
    ([.0896, .089, .1], True),
    ([.0896, .0885, .1], False),
    ([.0896, .0893, .089, .0887, .0884, .1], False),
    ([.0896, .0895], False),
    ([.0896, .0896], False),
    ([.0896, .090], True),
])
def test_escape_plan_uses_original_snapshot_and_requires_outward_endpoint(distances, accepted):
    kwargs = dict(
        plans={"left": path(distances)}, measured_joints={"left": [.0896, 0, 1], "right": [0, 0, 1]},
        fk_chain_points=lambda _arm, joints: [list(joints)],
        profile=binding()["profile"]["dual_arm_collision_check"],
    )
    if accepted:
        result = validate_inter_arm_clearance(**kwargs)
        assert result["escape_allowed"] is True
        assert result["current_center_distance_m"] == pytest.approx(.0896)
    else:
        with pytest.raises(RoboTwinPlanRejected) as error:
            validate_inter_arm_clearance(**kwargs)
        assert error.value.code == "inter_arm_collision"
        assert error.value.details["escape_allowed"] is True
        assert error.value.details["current_center_distance_m"] == pytest.approx(.0896)


class Plant:
    capabilities = MotionBackendCapabilities(
        backend_id="robotwin", embodiment_id="aloha_agilex", arms=("left", "right"),
        controlled_point="parallel_jaw_grip_center_tcp",
        planner_frames={"left": "left_link6", "right": "right_link6"},
        synchronized_groups=(("left", "right"),), global_joint_planning=True, local_cartesian=True,
    )

    def __init__(self, *, separation=.4, gain=1.):
        self.measured = {"left": np.array([-separation / 2, 0., 1.]), "right": np.array([separation / 2, 0., 1.])}
        self.targets = copy.deepcopy(self.measured)
        self.gain = gain
        self.steps = 0
        self.writes = []

    def read_joint_positions(self, arm):
        return self.measured[arm].tolist()

    def read_public_tcp_pose(self, arm):
        return [*self.read_joint_positions(arm), 0., 0., 0., 1.]

    def write_joint_target(self, arm, position, velocity):
        self.targets[arm] = np.asarray(position, dtype=float)
        self.writes.append((arm, list(position)))

    def step_physics(self):
        self.steps += 1
        for arm in self.measured:
            self.measured[arm] += self.gain * (self.targets[arm] - self.measured[arm])

    def capture_action_start(self):
        return {"measured_joint_positions": copy.deepcopy(self.measured)}

    def compute_local_joint_target(self, arm, joints, current, target, _profile):
        return {"position": list(target[:3]), "velocity": [0.] * 3,
                "predicted_score_before": float(np.linalg.norm(np.asarray(target[:3]) - current[:3])),
                "predicted_score_after": 0., "line_search_scale": 1., "solver": {"fixture": True}}


def live_session(monkeypatch, plant):
    live = object.__new__(RoboTwinLiveSession)
    live.control_profile = binding()
    live.workspace_bounds_m = WORKSPACE
    live.operator_execution_video = False
    live.last_receipt = None
    live.environment = SimpleNamespace(scene=SimpleNamespace(step=plant.step_physics), robot=object(), eval_success=False)
    live._pinocchio_chain_points = lambda _env, _arm, joints: [list(joints)]
    live._measured_arm_joints = lambda _env, arm: plant.read_joint_positions(arm)
    live._public_pose_wxyz = lambda _env, arm: [*plant.read_joint_positions(arm), 1., 0., 0., 0.]
    live._after_physics_step = lambda *_args, **_kwargs: False
    monkeypatch.setattr(session_module, "RoboTwinMotionBackendAdapter", lambda **_kwargs: plant)
    return live


def test_unchanged_blocked_target_stops_at_16_actual_steps_but_v9_waits_64():
    profile = binding()["profile"]
    for enabled, expected, code in ((True, 16, "blocked"), (False, 64, "waypoint_tracking_timeout")):
        plant = Plant(gain=0.)
        target = plant.read_joint_positions("left")
        target[2] -= .04
        result, arrays = execute_measured_convergence(
            adapter=plant, paths={"left": {"position": [target], "velocity": [[0.] * 3]}},
            all_arms=("left", "right"), profile=profile["motion_transaction"]["global_convergence"],
            workspace_bounds_m=WORKSPACE, mode="global",
            **({"blocked_policy": profile["blocked_motion_recovery"]} if enabled else {}),
        )
        assert plant.steps == result["sampled_steps"] == expected
        assert result["abort"]["code"] == code
        assert result["completed"] is False
        assert arrays["issued_left_arm__position"].shape[0] == expected
        assert set(arm for arm, _ in plant.writes) == {"left"}
        if enabled:
            assert result["abort"]["arms"] == ["left"]
            assert result["stall_shape"] == "from_start"


def test_terminal_stops_before_blocked_detector_without_more_physics():
    plant = Plant(gain=0.)
    profile = binding()["profile"]
    result, _ = execute_measured_convergence(
        adapter=plant, paths={"left": {"position": [[-.2, 0, .95]], "velocity": [[0.] * 3]}},
        all_arms=("left", "right"), profile=profile["motion_transaction"]["global_convergence"],
        workspace_bounds_m=WORKSPACE, mode="global", blocked_policy=profile["blocked_motion_recovery"],
        terminal_check=lambda _stage, _wait: plant.steps >= 7,
    )
    assert result["completed"] and result["abort"] is None
    assert plant.steps == 7 and result["termination"]["reason"] == "environment_terminal"


def test_global_execution_allows_measured_escape_and_separates_margin(monkeypatch):
    for separation, target_x in ((.0896, -.055), (.10, -.043)):
        plant = Plant(separation=separation)
        live = live_session(monkeypatch, plant)
        report, _arrays, _sequence = live._execute_motion_transaction_dense_action(
            environment=live.environment, action={"action_id": "offline-guard"},
            control_sequence={"left_arm": {"position": np.array([[target_x, 0, 1]]), "velocity": np.zeros((1, 3))},
                              "right_arm": None, "left_gripper": None, "right_gripper": None},
        )
        assert report["completed"]
        assert plant.steps == 1
        assert report["inter_arm_clearance"]["escape_allowed"] is (separation < .09)


def test_local_iteration_zero_admits_outward_escape_without_global_planner(monkeypatch):
    plant = Plant(separation=.0896)
    live = live_session(monkeypatch, plant)
    target = plant.read_public_tcp_pose("left")
    target[0] -= .01
    report, arrays, sequence = live._execute_local_cartesian_action(
        environment=live.environment, target_pose_xyzw={"left": target}, action={"action_id": "escape"},
    )
    assert report["completed"] and report["sampled_steps"] == 1
    assert report["inter_arm_clearance"]["escape_allowed"] is True
    assert report["inter_arm_clearance"]["current_center_distance_m"] == pytest.approx(.0896)
    assert plant.read_joint_positions("left")[0] == pytest.approx(target[0])
    assert sequence["right_arm"] is None
    assert arrays["issued_right_arm__position"].size == 0


def state_for(live, plant, tmp_path):
    return _SparseMotionState(
        action={"action": "recover_motion", "action_id": "recover-1"}, receipt_path=tmp_path / "receipt.json", started=0.,
        environment=live.environment, grip_center_profile=True, arguments={"targets": {"left": "home"}},
        targets={"left": "home"}, target_arms=("left",), motion_adapter=plant,
        before={arm: {"pose_xyz_xyzw": plant.read_public_tcp_pose(arm)} for arm in ("left", "right")},
        measured_joint_snapshot={arm: plant.read_joint_positions(arm) for arm in ("left", "right")},
        public_pose_targets={"left": [-.3, 0., 1.2, 1., 0., 0., 0.]},
    )


def prior_blocked(direction):
    return {"receipt": {
        "action_id": "blocked-0", "failure_code": "blocked", "target_arms": ["left"],
        "controller_tracking": {"abort": {"code": "blocked", "arms": ["left"]}},
        "blocked_motion": {"arms": {"left": {"approach_direction_world": direction,
                                                "source": {"kind": "validated_tcp_segment"}}}},
    }}


def test_recovery_retreat_uses_recorded_direction_and_refreshes_global_snapshot(monkeypatch, tmp_path):
    plant = Plant()
    live = live_session(monkeypatch, plant)
    live.last_receipt = prior_blocked([0., 0., -1.])
    state = state_for(live, plant, tmp_path)
    assert live._retreat_before_blocked_recovery(state)
    assert plant.read_joint_positions("left")[2] == pytest.approx(1.02)
    assert state.before["left"]["pose_xyz_xyzw"][2] == 1.
    assert state.planning_start_public_poses["left"][2] == pytest.approx(1.02)
    assert state.motion_action_start["measured_joint_positions"]["left"][2] == pytest.approx(1.02)
    assert state.blocked_recovery["global_planning_after_retreat"]
    assert state.recovery_prefix is not None and state.executed
    assert set(arm for arm, _ in plant.writes) == {"left"}


@pytest.mark.parametrize("direction", [None, [0., 0., 0.], [float("nan"), 0., 1.]])
def test_recovery_missing_direction_refuses_before_any_write(monkeypatch, tmp_path, direction):
    plant = Plant()
    live = live_session(monkeypatch, plant)
    live.last_receipt = prior_blocked(direction)
    state = state_for(live, plant, tmp_path)
    assert not live._retreat_before_blocked_recovery(state)
    assert not plant.writes and plant.steps == 0
    assert state.plan_rejections["left"]["code"] == "blocked_recovery_direction_unavailable"


def test_blocked_retreat_does_not_attempt_global_recovery(monkeypatch, tmp_path):
    plant = Plant(gain=0.)
    live = live_session(monkeypatch, plant)
    live.last_receipt = prior_blocked([0., 0., -1.])
    state = state_for(live, plant, tmp_path)
    assert not live._retreat_before_blocked_recovery(state)
    assert plant.steps == 16 and state.tracking_not_reached_reason == "blocked"
    assert state.blocked_recovery["global_planning_after_retreat"] is False
    assert state.recovery_prefix is None


def test_automatic_retreat_outside_workspace_is_a_no_write_refusal(monkeypatch, tmp_path):
    plant = Plant()
    plant.measured["left"][2] = 1.99
    live = live_session(monkeypatch, plant)
    live.last_receipt = prior_blocked([0., 0., -1.])
    state = state_for(live, plant, tmp_path)
    assert not live._retreat_before_blocked_recovery(state)
    assert not plant.writes and plant.steps == 0
    assert state.plan_rejections["left"]["code"] == "blocked_recovery_retreat_outside_workspace"


def test_failed_global_phase_keeps_executed_retreat_and_controller_rows(monkeypatch, tmp_path):
    plant = Plant()
    live = live_session(monkeypatch, plant)
    live.last_receipt = prior_blocked([0., 0., -1.])
    state = state_for(live, plant, tmp_path)
    assert live._retreat_before_blocked_recovery(state)
    prefix_samples = state.recovery_prefix[0]["sampled_steps"]
    live._combine_blocked_recovery_prefix(state)
    assert state.executed
    assert state.controller_tracking["sampled_steps"] == prefix_samples
    assert not state.controller_tracking["completed"]
    assert state.tracking_not_reached_reason == "recovery_global_planning_failed"
    assert state.controller_tracking_arrays["recovery_retreat__issued_left_arm__position"].shape[0] == prefix_samples
    assert state.control_sequence_record["left_arm"]["position"][-1, 2] == pytest.approx(1.02)


def test_blocked_direction_uses_last_validated_segment_not_net_target(monkeypatch, tmp_path):
    plant = Plant()
    live = live_session(monkeypatch, plant)
    state = state_for(live, plant, tmp_path)
    points = np.array([[-.2, 0, 1, 0, 0, 0, 1], [-.1, 0, 1, 0, 0, 0, 1], [-.1, 0, .96, 0, 0, 0, 1]])
    state.validated_plans["left"] = ValidatedRoboTwinPlan(points[:, :3], np.zeros((3, 3)), points, points, {})
    state.controller_tracking = {"planned_stages": 3, "abort": {
        "code": "blocked", "stage_index": 2, "arms": ["left"], "frozen_steps": 16,
    }}
    evidence = live._blocked_motion_evidence(state)
    assert evidence["arms"]["left"]["approach_direction_world"] == pytest.approx([0, 0, -1])
    assert evidence["arms"]["left"]["source"] == {"kind": "validated_tcp_segment", "start_index": 1, "end_index": 2}
