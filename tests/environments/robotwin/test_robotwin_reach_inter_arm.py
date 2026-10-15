"""The other-arm screen for RoboTwin's read-only reach check (0059 M6).

RoboTwin admission refuses plans whose link spheres meet the other arm held at
its measured joints; the reach check had no collision model. ``bind_screen``
applies the admission's own ``InterArmClearancePolicy`` and sphere distance to
a reach-check solution: the reached configuration alone (``endpoint``).
Offline: a toy chain whose link points are the joints themselves, and the
RoboTwin control profile's ``dual_arm_collision_check`` values. The wiring
(session flag, worker, RunSpec, adapter, surface) is
``test_robotwin_reach_inter_arm_wiring.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.contracts.reach_check import (
    BACKEND_REASONS, ROBODOJO_REASONS, ROBOTWIN_INTER_ARM_REASONS, ROBOTWIN_REASONS,
)
from ora.environment.backends.robotwin import reach_inter_arm as screen_module

ROOT = Path(__file__).resolve().parents[3]
PROFILE = json.loads((ROOT / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json")
                     .read_text(encoding="utf-8"))["dual_arm_collision_check"]
#: Each arm: two link points, a fixed shoulder and a "hand" at the first three joints.
SHOULDERS = {"left": [-0.3, 0.0, 1.0], "right": [0.3, 0.0, 1.0]}
MEASURED = {"left": [-0.3, 0.0, 0.8, 0.0, 0.0, 0.0], "right": [0.3, 0.0, 0.8, 0.0, 0.0, 0.0]}


def _chain(arm, joints):
    return [SHOULDERS[arm], [float(value) for value in joints[:3]]]


def _screen(arm="left", **kwargs):
    return screen_module.bind_screen(arm, measured_joints=MEASURED, fk_chain_points=_chain, profile=PROFILE,
                                     **kwargs)


def test_the_screen_uses_the_admission_sphere_model():
    assert PROFILE["link_radius_m"] == 0.045 and PROFILE["minimum_surface_clearance_m"] == 0.005
    required = 2 * PROFILE["link_radius_m"] + PROFILE["minimum_surface_clearance_m"]
    clear = _screen()([0.3 - required - 0.01, 0.0, 0.8, 0, 0, 0])
    assert clear is None
    blocked = _screen()([0.3 - required + 0.01, 0.0, 0.8, 0, 0, 0])
    assert blocked["reason"] == screen_module.REASONS["endpoint"]
    assert (blocked["stage_index"], blocked["stages"]) == (0, 1)
    assert blocked["required_center_distance_m"] == pytest.approx(required)


def test_the_screen_reads_the_endpoint_alone():
    # The hand passes through the right hand's position on its way to the far side.
    crossing = [0.6, 0.0, 0.8, 0, 0, 0]
    assert _screen()(crossing) is None
    blocked = _screen()([0.3, 0.0, 0.8, 0, 0, 0])
    assert (blocked["stage_index"], blocked["stages"]) == (0, 1)
    assert blocked["reason"] == "inter_arm_collision_endpoint"


def test_the_right_arm_is_screened_against_the_held_left_arm():
    blocked = _screen("right")([-0.3, 0.0, 0.8, 0, 0, 0])
    assert blocked["reason"] == screen_module.REASONS["endpoint"]
    assert _screen("right")([0.4, 0.1, 0.9, 0, 0, 0]) is None
    with pytest.raises(ValueError, match="left or right"):
        _screen("both")


def test_the_mode_follows_the_flag():
    assert screen_module.screen_mode(screen=False) is None
    assert screen_module.screen_mode(screen=True) == "endpoint"
    with pytest.raises(ValueError, match="Boolean"):
        screen_module.screen_mode(screen=1)


def test_the_screen_has_its_own_reason_set_model_and_claim():
    reason = screen_module.REASONS["endpoint"]
    assert reason in BACKEND_REASONS
    # The unscreened set stays what every surface without the screen publishes.
    assert reason not in ROBOTWIN_REASONS
    assert ROBOTWIN_INTER_ARM_REASONS["endpoint"] == (*ROBOTWIN_REASONS, reason)
    basis = screen_module.batch_basis("endpoint")
    assert basis["ik_model"] == screen_module.IK_MODELS["endpoint"]
    assert basis["matches_move_pose_admission"] is False
    assert "other arm held at its measured joints" in basis["claim_boundary"]
    assert "not the 128-seed collision-checked CuRobo IK move_pose plans through" in basis["claim_boundary"]
    assert reason not in ROBODOJO_REASONS
    assert screen_module.IK_MODELS["endpoint"] != "robotwin_single_seed_damped_least_squares_v1"
    assert "at that configuration only and not along any path" in screen_module.CLAIM_BOUNDARIES["endpoint"]
