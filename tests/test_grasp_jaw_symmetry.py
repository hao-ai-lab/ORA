"""grasp_plan publishes the parallel-jaw member with the most X5 wrist-roll headroom.

EXP11A g007 stack_bowls (run ``849f55dccacd``, +38.8 min): the right-arm 9 mm
wall pinch was the pool's only reachable candidate, published in its compiled
orientation because that orientation solved. The precommit probe recorded both
members: the compiled one at joint6 2.977 rad (pregrasp) and 3.004 rad
(contact), its 180-degree jaw flip at -0.164 and -0.137 rad with joints 1-5
within 3e-6 rad. The arm closed and lifted at joint6 3.00, 0.14 rad from the
+/-3.14 stop, and every carry move and wrist rotation after it was refused.
The joint vectors below are those probe records; the rows are what the
RoboDojo reach check computes from them (``controllers.reach_check``).

Everything is offline: a stub IK bridge, the fake Dojo plant observation and
the real dispatcher with result-schema admission.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.grasp_jaw_symmetry_schema import (
    CAPABILITY, CLAIM_BOUNDARY, REASONS, RULE, bind_grasp_jaw_symmetry,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.tools import grasp_jaw_symmetry as symmetry
from ora.agent.tools import reach_check as policy_reach
from ora.contracts.reach_check import RESULT_SCHEMA
from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

DOJO = "robodojo_placement_closed_skills_v2"
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v2",)
#: The live articulation's limits as the g007 IK diagnostics recorded them.
X5_LIMITS = [[-9.999999046325684, 9.999999046325684]] * 5 + [[-3.1399998664855957, 3.1399998664855957]]
MEASURED = [6.886e-13, -1.436e-15, 1.080e-16, 7.720e-16, -1.147e-12, 5.051e-12]
#: observation-0071 right arm, precommit probe joint targets per member and stage.
G007 = {
    "exact": ([-0.15811031659971064, 2.3524513880359095, 2.2083142439471644, -1.6691091696368616,
               -0.28960913022886103, 2.9773340225219727],
              [-0.13161274790763855, 2.3049986998187464, 1.978759765625, -1.4792472680462438,
               -0.29587095975875854, 3.004225015640259]),
    "jaw_flip": ([-0.15811031659971064, 2.352450195943014, 2.208311382924215, -1.6691082159625452,
                  -0.28960925737489873, -0.16425862908363342],
                 [-0.13161273300647736, 2.3049984614001673, 1.9787591139422815, -1.4792469183551233,
                  -0.29587095579038447, -0.1373676061630249]),
}
#: observation-0000 left arm: the compiled member already has the headroom.
G007_KEEP = {
    "exact": ([0.0, 1.0, 1.0, -1.0, 0.1, 3.14 - 2.677], [0.0, 1.0, 1.0, -1.0, 0.1, 3.14 - 2.689]),
    "jaw_flip": ([0.0, 1.0, 1.0, -1.0, 0.1, 3.14 - 2.677 - math.pi],
                 [0.0, 1.0, 1.0, -1.0, 0.1, 3.14 - 2.689 - math.pi]),
}
RULE_X5 = {"measured_joints": MEASURED, "isolation_bound_rad": symmetry.JOINT6_ISOLATION_BOUND_RAD}


def _row(joints, reference, *, reachable=True):
    if not reachable:
        return {"reachable": False, "reason": "ik_failed", "joint_limit_margin_rad": None,
                "joint_delta_rad": None}
    return {"reachable": True, "reason": None,
            "joint_limit_margin_rad": round(joint_limit_margin_rad(joints, X5_LIMITS), 5),
            "joint_delta_rad": round(joint_delta_rad(joints, reference), 5)}


def _rows(recorded, *, measured=MEASURED, unreachable=()):
    return {member: (_row(pre, measured, reachable=member not in unreachable),
                     _row(contact, pre, reachable=member not in unreachable))
            for member, (pre, contact) in recorded.items()}


# --------------------------------------------------------------------------
# The member choice, from the recorded joints
# --------------------------------------------------------------------------

def test_the_g007_wall_pinch_is_published_as_its_jaw_flip():
    choice = symmetry.choose(_rows(G007), RULE_X5)
    assert choice["member"] == "jaw_flip"
    assert choice["jaw_symmetry"]["variant"] == "jaw_flip"
    assert choice["jaw_symmetry"]["reason"] == "larger_joint6_margin"
    margins = choice["jaw_symmetry"]["joint6_margin_rad"]
    assert margins["exact"] == pytest.approx(3.14 - 3.004225, abs=1e-4)  # the contact solution binds
    assert margins["jaw_flip"] == pytest.approx(3.14 - 0.164259, abs=1e-4)
    headroom = choice["joint_headroom"]
    # Joints 1-5 sit far inside +/-10 rad, so the minimum margin is joint6's.
    assert headroom["pregrasp"] == {"min_joint_limit_margin_rad": pytest.approx(2.97574, abs=1e-4),
                                    "joint6_margin_rad": pytest.approx(2.97574, abs=1e-4)}
    assert headroom["approach"]["joint6_margin_rad"] == pytest.approx(3.00263, abs=1e-4)


def test_a_compiled_member_with_more_headroom_is_kept():
    choice = symmetry.choose(_rows(G007_KEEP), RULE_X5)
    assert (choice["member"], choice["jaw_symmetry"]["reason"]) == ("exact", "larger_joint6_margin")
    assert choice["jaw_symmetry"]["joint6_margin_rad"]["exact"] == pytest.approx(2.677, abs=1e-4)
    # The flip would sit at joint6 -2.679 and -2.691 rad: 0.449 rad from the stop.
    assert choice["jaw_symmetry"]["joint6_margin_rad"]["jaw_flip"] == pytest.approx(0.4494, abs=1e-3)
    assert choice["joint_headroom"]["pregrasp"]["joint6_margin_rad"] == pytest.approx(2.677, abs=1e-4)


def test_equal_margins_fall_to_the_smaller_move_then_to_the_compiled_member():
    rows = {member: ({"reachable": True, "joint_limit_margin_rad": 1.5, "joint_delta_rad": delta},
                     {"reachable": True, "joint_limit_margin_rad": 1.5, "joint_delta_rad": 0.1})
            for member, delta in (("exact", 1.2), ("jaw_flip", 0.8))}
    choice = symmetry.choose(rows, RULE_X5)
    assert (choice["member"], choice["jaw_symmetry"]["reason"]) == ("jaw_flip", "margin_tie_smaller_joint_delta")
    rows["jaw_flip"][0]["joint_delta_rad"] = 1.2005
    choice = symmetry.choose(rows, RULE_X5)
    assert (choice["member"], choice["jaw_symmetry"]["reason"]) == ("exact", "margin_and_joint_delta_tie")


def test_one_or_no_reachable_member():
    only = symmetry.choose(_rows(G007, unreachable=("exact",)), RULE_X5)
    assert (only["member"], only["jaw_symmetry"]["reason"]) == ("jaw_flip", "only_reachable_variant")
    assert only["jaw_symmetry"]["joint6_margin_rad"]["exact"] is None
    neither = symmetry.choose(_rows(G007, unreachable=("exact", "jaw_flip")), RULE_X5)
    assert neither["member"] is None
    assert neither["jaw_symmetry"] == {"variant": "exact", "reason": "no_reachable_variant",
                                       "joint6_margin_rad": {"exact": None, "jaw_flip": None}}
    assert neither["joint_headroom"] == {"pregrasp": None, "approach": None}


def test_joint6_is_reported_only_when_the_measured_joints_isolate_it():
    # A base wound 4.5 rad plus the 2.35-2.98 rad pregrasp move could reach
    # |q1| > 10 - 3.14, where joint1 might be the binding joint instead.
    wound = [4.5, *MEASURED[1:]]
    choice = symmetry.choose(_rows(G007, measured=wound), {**RULE_X5, "measured_joints": wound})
    assert choice["jaw_symmetry"]["joint6_margin_rad"] == {"exact": None, "jaw_flip": None}
    assert choice["joint_headroom"]["pregrasp"]["joint6_margin_rad"] is None
    assert choice["joint_headroom"]["pregrasp"]["min_joint_limit_margin_rad"] is not None
    assert (choice["member"], choice["jaw_symmetry"]["reason"]) == ("jaw_flip", "larger_joint_limit_margin")
    unmeasured = symmetry.choose(_rows(G007), {**RULE_X5, "measured_joints": None})
    assert unmeasured["jaw_symmetry"]["reason"] == "larger_joint_limit_margin"


def test_the_isolation_bound_is_the_x5_limit_table():
    assert symmetry.JOINT6_ISOLATION_BOUND_RAD == pytest.approx(9.999999 - 3.14 - 0.001)
    assert symmetry.ISOLATION_BOUNDS == {"arx_x5_link6": symmetry.JOINT6_ISOLATION_BOUND_RAD}
    observation = {"proprioception": {"end_effectors": {"right": {"arm_joint_qpos": MEASURED}}}}
    assert symmetry.measured_arm_joints(observation, "right") == pytest.approx(MEASURED)
    assert symmetry.measured_arm_joints(observation, "left") is None
    assert symmetry.measured_arm_joints({"proprioception": {"end_effectors": {"right": {
        "arm_joint_qpos": [0.0] * 5}}}}, "right") is None


# --------------------------------------------------------------------------
# Surfaces: only the live Dojo surface declares it; RoboTwin is unchanged
# --------------------------------------------------------------------------

def test_the_live_dojo_surface_declares_it_and_publishes_the_fields():
    surface = resolve_tool_surface(DOJO)
    assert surface.supports(CAPABILITY)
    grasp = surface.result_schemas["grasp_plan"]
    annotation = grasp["$defs"]["candidate"]["properties"]["ik_reachability"]
    assert {"jaw_symmetry", "joint_headroom"} <= set(annotation["required"])
    assert annotation["properties"]["jaw_symmetry"]["properties"]["reason"]["enum"] == REASONS
    assert grasp["properties"]["ik_reachability"]["properties"]["jaw_symmetry"]["properties"]["rule"][
        "const"] == RULE
    help_text = next(item["description"] for item in surface.definitions if item["name"] == "grasp_plan")
    assert "jaw_symmetry" in help_text and "joint_headroom" in help_text
    assert "not a carry guarantee" in help_text


@pytest.mark.parametrize("profile", ROBOTWIN_LIVE)
def test_robotwin_surfaces_keep_the_released_annotation(profile):
    surface = resolve_tool_surface(profile)
    assert not surface.supports(CAPABILITY)
    annotation = surface.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]
    assert "jaw_symmetry" not in annotation["properties"]
    host = SimpleNamespace(surface=surface, workspace_bounds_m={}, align_plan_callback=None)
    selection = policy_reach.grasp_candidate_selection(host, "right", {"proprioception": {}})
    assert "jaw_symmetry" not in selection["identity"]


def test_the_binder_refuses_a_surface_it_cannot_describe():
    surface = resolve_tool_surface(DOJO)
    results = copy.deepcopy(dict(surface.result_schemas))
    with pytest.raises(ToolSurfaceError, match="already|overwrite"):
        bind_grasp_jaw_symmetry(results, surface.capabilities, surface.view_config)
    released = copy.deepcopy(dict(resolve_tool_surface(ROBOTWIN_LIVE[0]).result_schemas))
    with pytest.raises(ToolSurfaceError, match="RoboDojo"):
        bind_grasp_jaw_symmetry(released, ["grasp_plan_ik_reachability_annotation_v1", "robotwin_rgbd_v1"],
                                {"controlled_frame_convention": "aloha_agilex_tcp"})
    with pytest.raises(ToolSurfaceError, match="X5"):
        bind_grasp_jaw_symmetry(released, ["grasp_plan_ik_reachability_annotation_v1", "robodojo_rgbd_v1"],
                                {"controlled_frame_convention": "aloha_agilex_tcp"})


# --------------------------------------------------------------------------
# Through the real dispatcher: the published poses are the chosen member
# --------------------------------------------------------------------------

def _x5_bridge(requests, *, flip_better=True):
    """Every pose solves; the compiled member sits near the joint6 stop, the flip far from it."""
    from test_reach_check import _local_y, _pool_ik

    def bridge(request):
        requests.append(copy.deepcopy(request))
        rows = []
        for query in request["queries"]:
            start = query["from_index"]
            unstarted = start is not None and not rows[start]["reachable"]
            if unstarted or not _pool_ik(query):
                rows.append({"reachable": False, "reason": "start_unreachable" if unstarted else "ik_failed",
                             "route": None, "position_error_m": None, "orientation_error_rad": None,
                             "joint_limit_margin_rad": None, "joint_delta_rad": None})
                continue
            compiled = _local_y(query["pose_xyzw"][3:])[1] > 0
            near_stop = compiled == flip_better
            pregrasp = query["interaction"] == "free_space"
            rows.append({"reachable": True, "reason": None, "route": "global" if pregrasp else "local",
                         "position_error_m": 0.0001, "orientation_error_rad": 0.001,
                         "joint_limit_margin_rad": (0.16278 if pregrasp else 0.13577) if near_stop
                         else (2.97574 if pregrasp else 3.00263),
                         "joint_delta_rad": (2.97733 if near_stop else 2.35245) if pregrasp else 0.22955})
        return {"schema": RESULT_SCHEMA, "arm": request["arm"],
                "ik_model": "robodojo_sdk_ik_endpoint_selection_v1", "matches_move_pose_admission": True,
                "motion_planner_called": False, "claim_boundary": "stub IK", "results": rows}
    return bridge


def test_grasp_plan_publishes_and_executes_the_member_it_chose(tmp_path):
    from test_reach_check import POOL, _local_x, _local_y, _pool_grasp
    from ora.agent.tools.skill_candidates import _grasp_candidate

    requests = []
    runtime, _backend, result = _pool_grasp(tmp_path, _x5_bridge(requests))
    ids = [f"grasp-pool{index:020d}" for index in range(len(POOL))]
    # Order and cap are the released ones: reachable near-vertical first.
    assert [item["candidate_id"] for item in result["candidates"]] == [ids[2], ids[3], ids[4]]
    # Both members for all five candidates, in pregrasp/approach pairs.
    assert [len(item["queries"]) for item in requests] == [10, 10]
    manifest_path = next(runtime.artifact_root.glob("grasp-queries/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    raw = {item["candidate_id"]: item for item in manifest["candidates"]}
    for candidate in result["candidates"]:
        annotation = candidate["ik_reachability"]
        assert annotation["member"] == "jaw_flip" and annotation["reachable"] is True
        assert annotation["jaw_symmetry"]["variant"] == "jaw_flip"
        # The pose the policy and grasp_execute see is the flipped one: the jaw
        # axis reverses, the approach and the position do not.
        for key in ("pregrasp_pose", "contact_pose"):
            orientation = candidate[key]["orientation_xyzw"]
            assert np.dot(_local_y(orientation), raw[candidate["candidate_id"]]["jaw_world"]) == pytest.approx(-1)
            assert np.dot(_local_x(orientation), raw[candidate["candidate_id"]]["approach_world"]) == (
                pytest.approx(1))
    by_id = {item["candidate_id"]: item["ik_reachability"] for item in result["candidates"]}
    assert by_id[ids[2]]["jaw_symmetry"] == {
        "variant": "jaw_flip", "reason": "larger_joint6_margin",
        "joint6_margin_rad": {"exact": 0.13577, "jaw_flip": 2.97574}}
    assert by_id[ids[2]]["joint_headroom"] == {
        "pregrasp": {"min_joint_limit_margin_rad": 2.97574, "joint6_margin_rad": 2.97574},
        "approach": {"min_joint_limit_margin_rad": 3.00263, "joint6_margin_rad": 3.00263}}
    assert by_id[ids[4]]["jaw_symmetry"]["reason"] == "only_reachable_variant"
    assert result["ik_reachability"]["jaw_symmetry"] == {"rule": RULE, "claim_boundary": CLAIM_BOUNDARY}
    record = manifest["candidate_selection"]
    assert record["identity"]["jaw_symmetry"]["capability"] == CAPABILITY
    assert record["evidence"]["jaw_symmetry"]["measured_joints"] == [0.0] * 6
    assert set(record["evidence"]["jaw_symmetry"]["published_jaw_flip"]) >= {ids[2], ids[3], ids[4]}
    # grasp_execute commits the manifest's compiled poses: the flipped ones.
    committed = _grasp_candidate(runtime, ids[2], manifest["identity"]["observation_id"])["candidate"]
    assert committed["contact_pose"] == next(
        item for item in result["candidates"] if item["candidate_id"] == ids[2])["contact_pose"]


def test_grasp_plan_keeps_the_compiled_member_when_it_has_the_headroom(tmp_path):
    from test_reach_check import POOL, _local_y, _pool_grasp

    _runtime, _backend, result = _pool_grasp(tmp_path, _x5_bridge([], flip_better=False))
    ids = [f"grasp-pool{index:020d}" for index in range(len(POOL))]
    by_id = {item["candidate_id"]: item for item in result["candidates"]}
    assert by_id[ids[2]]["ik_reachability"]["jaw_symmetry"]["variant"] == "exact"
    assert _local_y(by_id[ids[2]]["pregrasp_pose"]["orientation_xyzw"])[1] > 0
    # The x=0.1 candidate solves only as its flip, so the flip is still published.
    assert by_id[ids[4]]["ik_reachability"]["jaw_symmetry"]["variant"] == "jaw_flip"


def test_an_unavailable_ik_reference_publishes_the_compiled_member(tmp_path):
    from test_reach_check import _pool_grasp

    def broken(request):
        raise RuntimeError("worker died")

    _runtime, _backend, result = _pool_grasp(tmp_path, broken)
    for candidate in result["candidates"]:
        assert candidate["ik_reachability"]["jaw_symmetry"]["reason"] == "ik_reference_unavailable"
        assert candidate["ik_reachability"]["joint_headroom"] == {"pregrasp": None, "approach": None}


def test_the_released_selection_is_unchanged_without_a_rule():
    """No rule: exact for all, the flip only where exact failed, poses untouched."""
    from test_reach_check import _local_y, _stub_bridge

    def compiled(index, jaw_sign):
        orientation = [0.0, 1.0, 0.0, 0.0] if jaw_sign > 0 else [1.0, 0.0, 0.0, 0.0]
        pose = {"position_m": [0.1 * index, 0.0, 0.9], "orientation_xyzw": orientation}
        return {"candidate_id": f"grasp-{index}", "rank": index, "pregrasp_pose": dict(pose),
                "contact_pose": {**pose, "position_m": [0.1 * index, 0.0, 0.86]}}

    pool = [compiled(0, 1), compiled(1, 1)]
    requests = []
    bridge = _stub_bridge(requests, reachable=lambda query: not (
        query["pose_xyzw"][0] > 0.05 and _local_y(query["pose_xyzw"][3:])[1] < 0))
    chosen = policy_reach.select_grasp_candidates(
        pool, arm="left", bridge=bridge, workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        approach_axis_local=[1.0, 0.0, 0.0])
    assert [len(item["queries"]) for item in requests] == [4]
    assert all("jaw_symmetry" not in item["ik_reachability"] for item in chosen["ordered"])
    assert "jaw_symmetry" not in chosen["summary"] and "jaw_symmetry" not in chosen["evidence"]
    assert [item["pregrasp_pose"] for item in chosen["ordered"]] == [item["pregrasp_pose"] for item in pool]


def test_a_dojo_surface_without_the_capability_sends_the_released_identity(tmp_path):
    from test_reach_check import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    runtime.surface = replace(runtime.surface, capabilities=tuple(
        item for item in runtime.surface.capabilities if item != CAPABILITY))
    selection = policy_reach.grasp_candidate_selection(runtime, "left", {"proprioception": {}})
    assert "jaw_symmetry" not in selection["identity"]
