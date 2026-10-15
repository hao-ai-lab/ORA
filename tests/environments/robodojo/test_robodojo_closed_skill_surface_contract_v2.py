"""The RoboDojo v2 default against its RoboTwin v2 sibling (0059 unit A0).

The v2 generation was `robodojo_placement_closed_skills_v1` plus a short
documented list (that predecessor left the tree), and it pairs with
`robotwin_toolchain_b2_closed_noservo_v2`:

- the tool set, order and effect classes are the RoboTwin v2 sibling's;
- the capabilities differ from the sibling only by documented backend facts;
- the definition bytes stay under a reviewed ceiling against the sibling.
"""
from __future__ import annotations

import json

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface

ROBODOJO = "robodojo_placement_closed_skills_v2"
SIBLING = "robotwin_toolchain_b2_closed_noservo_v2"
#: The sibling difference, as backend facts (ci/backend-parity.json classifies each).
DOJO_ONLY = {
    "contact_graspnet_robodojo_x5_link6_compiler_v1", "grasp_plan_jaw_symmetry_headroom_v1",
    "gripper_stall_hold_evidence_v1", "move_pose_jaw_symmetry_headroom_v1", "robodojo_controller_feedback_v1",
    "robodojo_retract_then_unwind_recovery_v1", "robodojo_rgbd_v1",
    "robodojo_sdk_ik_joint_interpolation_v1_non_parity", "robodojo_staged_contact_approach_v1",
    "robodojo_x5_public_rgbd_topdown_grasp_v1",
    # Version split: RoboDojo keeps the v1 reach wording (parity exceptions P0059-D01..D03).
    "move_pose_reach_along_request_v1", "pose_primitive_reach_along_request_v1", "reach_nearest_orientation_v1",
}
ROBOTWIN_ONLY = {
    "analytic_tcp_local_cartesian_v1", "contact_graspnet_robotwin_aloha_grip_center_compiler_v3",
    "measured_jaw_separation_v1", "robotwin_aloha_public_rgbd_topdown_grasp_v1",
    "robotwin_reach_check_inter_arm_screen_v1", "robotwin_rgbd_v1", "standard_tcp_pose_handoff_v1",
    "topk_ik_joint_planning_v1",
    "move_pose_reach_along_request_v2", "pose_primitive_reach_along_request_v2", "reach_nearest_orientation_v2",
}


def _size(definitions) -> int:
    return len(json.dumps(list(definitions), sort_keys=True, separators=(",", ":")).encode())


@pytest.fixture(scope="module")
def robodojo():
    return resolve_tool_surface(ROBODOJO)


def test_the_tool_set_and_effects_are_the_robotwin_v2_sibling_s(robodojo):
    sibling = resolve_tool_surface(SIBLING)
    assert [d["name"] for d in robodojo.definitions] == [d["name"] for d in sibling.definitions]
    assert len(robodojo.definitions) == 19 and "wait_ticks" in robodojo.mutation_tools
    for key in ("mutation_tools", "lifecycle_tools", "observation_tools", "call_effect_profile", "runtime_id"):
        assert getattr(robodojo, key) == getattr(sibling, key), key


def test_capabilities_differ_from_the_sibling_only_by_documented_backend_facts(robodojo):
    sibling = resolve_tool_surface(SIBLING)
    assert set(robodojo.capabilities) - set(sibling.capabilities) == DOJO_ONLY
    assert set(sibling.capabilities) - set(robodojo.capabilities) == ROBOTWIN_ONLY


def test_definition_bytes_stay_under_the_reviewed_ceilings(robodojo):
    """A later RoboDojo fact must displace text, not raise this unreviewed.

    Against the RoboTwin v2 sibling (+8,417 measured): the
    RoboDojo backend facts the v1 contract reviewed -- the compiled ARX
    controller contract, the Dojo wording and tick help, recovery, jaw hold,
    top-down, jaw-symmetry and route-equivalent reach wording -- less what the
    sibling now carries too (the shared wait_ticks definition, the grasp IK
    selection paragraph, the finish v2 contract) and less the RoboTwin
    probe-count sentence on grasp_execute, which RoboDojo states in its own words.

    observation_delivery_trim_v1 (2026-10-05) adds 6,886 B to RoboDojo v2 and
    6,012 B to the RoboTwin sibling: each surface's delivery-trim layer states
    once, on observe, what delivered copies leave out and the rounding, and on
    each producing tool the fixed claim texts its results and effects no longer
    repeat (RoboDojo carries five more: the two IK scopes, the jaw-symmetry and
    jaw-hold claims and its own reach and nearest-orientation claims, against
    RoboTwin's longer reach and align claims). Ceiling: 8,417 + 874.
    """
    assert _size(robodojo.definitions) - _size(resolve_tool_surface(SIBLING).definitions) <= 9_291
