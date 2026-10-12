"""RoboTwin publishes a measured jaw opening in metres, unconditionally.

The public observation used to carry only the last accepted command target,
with a real encoder reading appearing solely under a bound robot-geometry profile.
A policy that can see only its own last command cannot tell an accepted close
from jaws stopped early by an object, and a normalized 0..1 opening cannot be
compared against an object extent. Both gaps are contract, not cosmetics, so
they are pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.environment.backends.robotwin.gripper_state import public_gripper_state
from ora.environment.backends.robotwin.session import RoboTwinLiveSession


# Source: RoboTwin assets/embodiments/aloha-agilex.  ``gripper_scale`` comes
# from config.yml; the two prismatic finger rails and their 1:1 mimic binding
# come from urdf/arx5_description_isaac.urdf.
_GRIPPER_SCALE = [-0.01, 0.045]
_FINGER_TRAVEL_M = 0.021


def _robot(*, travel: float = _FINGER_TRAVEL_M, kind: str = "prismatic") -> SimpleNamespace:
    """An aloha-shaped articulation: two prismatic finger rails per arm."""

    robot = SimpleNamespace()
    for arm in ("left", "right"):
        primary = SimpleNamespace(get_drive_target=lambda: [0.045], type=kind)
        mimic = SimpleNamespace(get_drive_target=lambda: [0.045], type=kind)
        other = SimpleNamespace(type="revolute")
        setattr(robot, arm + "_gripper", [(primary, 1.0, 0.0), (mimic, 1.0, 0.0)])
        setattr(robot, arm + "_gripper_scale", list(_GRIPPER_SCALE))
        setattr(robot, arm + "_gripper_val", 1.0)
        setattr(robot, arm + "_entity", SimpleNamespace(
            get_qpos=lambda travel=travel: [0.3, travel, travel],
            get_active_joints=lambda other=other, primary=primary, mimic=mimic: [
                other, primary, mimic,
            ],
        ))
    return robot


def test_measured_jaw_separation_is_published_in_metres_not_only_normalized() -> None:
    """The encoder reading reaches the policy as jaw separation in metres."""

    state = public_gripper_state(_robot())
    for arm in ("left", "right"):
        # Both finger rails travel outward from a closed jaw, so the pad
        # faces are separated by the sum of the measured travels.
        assert state[arm]["gripper_opening_m"] == pytest.approx(2 * _FINGER_TRAVEL_M)
        assert state[arm]["opening_m_semantics"].endswith("prismatic_finger_travel")
        # The normalized value survives, but it is no longer the only reading.
        assert state[arm]["measured_opening_normalized"] == pytest.approx(
            (_FINGER_TRAVEL_M - _GRIPPER_SCALE[0])
            / (_GRIPPER_SCALE[1] - _GRIPPER_SCALE[0])
        )


def test_derived_grasp_or_retention_flag_is_gone() -> None:
    """It was false on every encoder reading of a run, including carried loads.

    A field that never varies teaches its reader to skip it, so the measured
    quantity replaces the derived claim outright.
    """

    for arm, value in public_gripper_state(_robot()).items():
        assert "grasp_or_retention_established" not in value, arm


def test_jaw_separation_fails_closed_on_a_non_prismatic_chain() -> None:
    """A wrong metre value is worse than an explicitly absent one."""

    state = public_gripper_state(_robot(kind="revolute"))
    for arm in ("left", "right"):
        assert state[arm]["gripper_opening_m"] is None
        assert state[arm]["jaw_separation_unavailable_reason"]
        # The normalized evidence is unaffected by the metre-value refusal.
        assert state[arm]["measured_opening_normalized"] is not None


def _session(*, public_robot_geometry=None) -> RoboTwinLiveSession:
    """A session stub carrying only what proprioception assembly reads."""

    session = object.__new__(RoboTwinLiveSession)
    session.action_codec = "motion_action_v3"
    session.control_profile = {
        "profile": {
            "profile_id": "robotwin_aloha_contact_recovery_dual_arm_v10",
        },
    }
    session.terminal_public_poses = {
        arm: [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0] for arm in ("left", "right")
    }
    session.terminal_gripper_targets = None
    session.gripper_command_targets = {"left": "close", "right": "open"}
    session.public_robot_geometry = public_robot_geometry
    return session


_RAW = {
    "endpose": {
        "left_endpose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "right_endpose": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
        "left_gripper": 1.0,
        "right_gripper": 0.0,
    },
    "joint_action": {"left_arm": [0.0] * 6, "right_arm": [0.0] * 6},
}


def test_public_proprioception_measures_the_jaw_without_a_geometry_binding() -> None:
    """The measurement is unconditional: no robot-geometry profile is bound here."""

    session = _session(public_robot_geometry=None)
    environment = SimpleNamespace(robot=_robot())

    proprioception = session._public_proprioception(environment, _RAW)

    assert proprioception["gripper_evidence_semantics"] != "command_target_only"
    for arm in ("left", "right"):
        effector = proprioception["end_effectors"][arm]
        # Command and measurement sit side by side; neither replaces the other.
        assert effector["gripper_command_target"] in {"open", "close"}
        assert effector["gripper_opening_m"] == pytest.approx(2 * _FINGER_TRAVEL_M)
        # The verbose normalized/drive-target block needs a geometry binding.
        assert "gripper_state" not in effector


def test_geometry_binding_still_receives_the_full_encoder_block() -> None:
    session = _session(public_robot_geometry={"descriptor_id": "fixture"})
    environment = SimpleNamespace(robot=_robot())

    proprioception = session._public_proprioception(environment, _RAW)

    for arm in ("left", "right"):
        effector = proprioception["end_effectors"][arm]
        assert effector["gripper_opening_m"] == pytest.approx(2 * _FINGER_TRAVEL_M)
        assert effector["gripper_state"]["measured_primary_joint_position"] == (
            pytest.approx(_FINGER_TRAVEL_M)
        )


def test_frozen_terminal_gripper_state_is_preferred_over_the_live_robot() -> None:
    """A terminal observation replays its frozen reading, not a later one."""

    session = _session()
    environment = SimpleNamespace(robot=_robot())
    raw = dict(_RAW, public_gripper_state={
        arm: {"gripper_opening_m": 0.004} for arm in ("left", "right")
    })

    proprioception = session._public_proprioception(environment, raw)

    for arm in ("left", "right"):
        assert proprioception["end_effectors"][arm]["gripper_opening_m"] == 0.004
