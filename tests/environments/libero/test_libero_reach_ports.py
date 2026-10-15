"""The reach features on the LIBERO default generation, over a LIBERO bridge (0059 M1-M3).

M2 gives LIBERO a reach check (``test_libero_reach_check.py``), so the
capability-gated reach features that ran only on RoboTwin and RoboDojo compose
on the LIBERO generation A0 declared them on (the default
``libero_placement_direct_overhead_v1`` carries them):

* M2 ``grasp_plan_ik_reachability_annotation_v1`` with ``libero_pro_rgbd_v1``:
  the pooled grasp_plan selection, with LIBERO's closed reason set, on a
  single-arm surface whose grasp_plan takes no arm argument;
* M1 ``move_pose_reach_along_request_v2`` / ``pose_primitive_reach_along_request_v2``:
  the reach pre-check names LIBERO's CuRobo IK, ``matches_move_pose_admission:
  false`` and a null measured disagreement (not yet measured);
* M3 ``reach_nearest_orientation_v2``: the sweep holds the grip site itself.

The bridge is the real LIBERO path -- session ``align_plan`` -> ``check_reach``
-> ``reach_check_batch`` -> frame conversion -- over a stand-in CuRobo IK that
admits a pose when its hand position is within reach of the base and its
approach is within 35 degrees of straight down.
"""
from __future__ import annotations

import copy
import json
import math

import pytest
from scipy.spatial.transform import Rotation

from ora.agent.composition import reach_reference_schema as schema
from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.move_pose_reach_schema import LINE_FRACTIONS
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.reach_orientation_schema import ROTATIONS_DEG
from ora.agent.composition.tool_surface import validate_schema_instance
from ora.agent.tools import move_pose_reach
from ora.agent.tools import reach_check as policy_reach
from ora.contracts.reach_check import LIBERO_REASONS, SHARED_REASONS
from ora.environment.backends.libero_pro import reach_check as libero_reach
from ora.environment.controllers.curobo_ik import (
    validate_curobo_ik_request, validate_curobo_ik_response,
)
from tests.environments.libero.test_libero_reach_check import DOWN, FakeIk, _Session

DEFAULT = "libero_placement_direct_overhead_v1"
PORTS = PROFILE_CATALOG[DEFAULT].parents[1] / "closed_skills_ports_v1"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
CURRENT = [-0.1, 0.1, 1.1, *DOWN]
TILT_LIMIT_DEG = 35.0


class TiltIk(FakeIk):
    """Also refuses a hand whose approach (local +Z) is tilted beyond the limit."""

    def solve_ik(self, request):
        answer = super().solve_ik(request)
        for query, row in zip(request["queries"], answer["results"]):
            approach = Rotation.from_quat(query["target_pose_base_xyzw"][3:]).apply([0.0, 0.0, 1.0])
            if row["solved"] and math.degrees(math.acos(max(-1.0, min(1.0, -approach[2])))) > TILT_LIMIT_DEG:
                row.update(solved=False, joint_positions=None, ee_pose_base_xyzw=None)
        # Chains onto a refused query are skipped, as the worker skips them.
        for index, query in enumerate(request["queries"]):
            start = query["from_index"]
            if start is not None and not answer["results"][start]["solved"]:
                answer["results"][index] = {"solved": False, "skipped": True, "joint_positions": None,
                                            "ee_pose_base_xyzw": None, "position_error_m": None,
                                            "rotation_error": None}
        return validate_curobo_ik_response(answer, validate_curobo_ik_request(request))


def _bridge(requests=None, planner=None):
    session = _Session(planner or TiltIk())

    def bridge(request):
        if requests is not None:
            requests.append(copy.deepcopy(request))
        return session.align_plan(request)
    return bridge


@pytest.fixture
def libero():
    return resolve_tool_surface(DEFAULT)


def _host(surface, bridge):
    from types import SimpleNamespace

    return SimpleNamespace(surface=surface, align_plan_callback=bridge, workspace_bounds_m=BOUNDS)


def _reach_schema(surface, tool):
    return surface.result_schemas[tool]["properties"]["compiled_pose"]["properties"]["targets"][
        "additionalProperties"]["properties"]["reach_along_request"]


def _tilted(degrees):
    return [float(item) for item in (Rotation.from_euler("x", degrees, degrees=True)
                                     * Rotation.from_quat(DOWN)).as_quat()]


def _plan(surface, bridge, target, interaction="free_space"):
    return move_pose_reach.plan_target(
        _host(surface, bridge), "right", current_pose=list(CURRENT), target_pose=list(target),
        interaction=interaction, orientation_mode="absolute_quaternion",
        observation={"proprioception": {"end_effectors": {"right": {"gripper_command_target": "open"}}}})


def test_only_the_libero_v2_generation_declares_the_libero_identity():
    """Integration 0059 unit A0 declared M2 on the LIBERO v2 default; of that generation only its
    overhead successor (the default since integration 0084) is registered."""
    declaring = {profile for profile, path in PROFILE_CATALOG.items()
                 if "libero_pro_rgbd_v1" in json.loads(path.read_text(encoding="utf-8")).get("capabilities", [])}
    assert declaring == {DEFAULT}


def test_the_candidate_binds_libero_s_reason_set_and_the_v2_reference(libero):
    for stage in ("pregrasp", "approach"):
        reason = libero.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"][
            "properties"][stage]["properties"]["reason"]
        assert reason["enum"] == [*SHARED_REASONS, *LIBERO_REASONS, None]
    for tool in ("move_pose", "move_to", "rotate_wrist", "rotate_pitch"):
        assert "ik_model" in json.dumps(_reach_schema(libero, tool))
    slots = {item["slot"] for item in libero.components}
    assert "derived:grasp_plan_ik_reachability_annotation_v1:libero_pro_rgbd_v1" in slots
    # The bridge needs no capability, and align_plan stays unpublished on LIBERO.
    names = {item["name"] for item in libero.definitions}
    assert "align_plan" not in names and "reach_check" not in names
    assert not libero.supports("read_only_ik_reachability_v1")


def test_a_solved_target_names_libero_s_curobo_ik_and_no_measured_error(libero):
    requests = []
    target = [-0.1, 0.1, 1.0, *DOWN]
    pose, notes = _plan(libero, _bridge(requests), target)
    assert pose == target and len(requests) == 1 and requests[0]["arm"] == "right"
    reach = notes["reach_along_request"]
    assert reach["target_reachable"] is True
    assert reach["ik_model"] == libero_reach.IK_MODEL and reach["matches_move_pose_admission"] is False
    assert reach["measured_disagreement"] is None  # not yet measured
    assert reach["claim_boundary"] == libero_reach.CLAIM_BOUNDARY
    assert libero_reach.IK_MODEL not in schema.MEASURED_DISAGREEMENT
    validate_schema_instance(reach, _reach_schema(libero, "move_pose"))


def test_an_unsolved_target_samples_its_path_and_the_nearest_turn_holds_the_grip_site(libero):
    requests = []
    target = [-0.1, 0.1, 1.0, *_tilted(60.0)]
    pose, notes = _plan(libero, _bridge(requests), target)
    assert pose == target
    reach = notes["reach_along_request"]
    validate_schema_instance(reach, _reach_schema(libero, "move_pose"))
    assert reach["target_reachable"] is False and reach["reason"] == "ik_failed"
    assert reach["ik_model"] == libero_reach.IK_MODEL and reach["measured_disagreement"] is None
    assert reach["claim_boundary"].endswith(libero_reach.CLAIM_BOUNDARY)
    # The orientation is slerped from straight down: up to 35 of 60 degrees solve.
    assert [sample["reachable"] for sample in reach["samples"]] == [
        fraction * 60.0 <= TILT_LIMIT_DEG for fraction in LINE_FRACTIONS]
    assert reach["farthest_reachable_fraction"] == max(f for f in LINE_FRACTIONS if f * 60.0 <= TILT_LIMIT_DEG)
    nearest = reach["nearest_reachable_orientation"]
    assert nearest["status"] == "found" and nearest["rotation_from_request_deg"] == 30
    assert nearest["tilt_from_straight_down_deg"] <= TILT_LIMIT_DEG
    # Grip-site frame: the controlled point is the jaw centre, so it stays put.
    assert nearest["jaw_center_m"] == pytest.approx(target[:3], abs=1e-5)
    assert nearest["pose"]["position_m"] == pytest.approx(target[:3], abs=1e-5)
    assert all(request["arm"] == "right" for request in requests)


def test_a_target_out_of_reach_in_every_turn_reports_none_within_the_sweep(libero):
    _pose, notes = _plan(libero, _bridge(planner=TiltIk(reach_m=0.3)), [0.5, 0.3, 1.3, *DOWN])
    nearest = notes["reach_along_request"]["nearest_reachable_orientation"]
    assert nearest["status"] == "none_within_sweep" and nearest["queries"] == len(ROTATIONS_DEG) * 12


def test_the_primitives_carry_the_same_reference(libero):
    host = _host(libero, _bridge())
    solved = move_pose_reach.primitive_reference(
        host, "right", {"target_pose_xyz_xyzw": [-0.1, 0.1, 1.0, *DOWN]}, current_pose=CURRENT,
        interaction="free_space")["reach_along_request"]
    assert solved["target_reachable"] is True and solved["ik_model"] == libero_reach.IK_MODEL
    validate_schema_instance(solved, _reach_schema(libero, "move_to"))


def test_the_help_states_the_libero_model_without_another_backend_s_numbers(libero):
    help_text = {item["name"]: item["description"] for item in libero.definitions}
    move_pose = help_text["move_pose"]
    assert f"ik_model {libero_reach.IK_MODEL}, matches_move_pose_admission false" in move_pose
    assert "has not yet been measured (measured_disagreement null)" in move_pose
    assert "no table, object or scene model" in move_pose
    assert "with the grip site held where the request put it" in move_pose
    for foreign in ("7,860", "128-seed", "damped-least-squares", "about 5%", "about 4%", "grip-centre TCP",
                    "link6", "route move_pose takes"):
        assert foreign not in move_pose, foreign
    sweep = move_pose.split("nearest_reachable_orientation")[1]
    assert "15 to 75 degrees in 15-degree steps" in sweep and ROTATIONS_DEG == (15, 30, 45, 60, 75)
    grasp = help_text["grasp_plan"]
    assert "Reachability is delegated to move_pose." not in grasp
    assert "IK reference at planning time, not a guarantee" in grasp and "pool of up to 32" in grasp
    assert f"{policy_reach.POOL_LIMIT}" in grasp and "within 30 degrees of straight down" in grasp
    assert policy_reach.VERTICAL_PREFERENCE_DEG == 30.0
    assert f"ik_model {libero_reach.IK_MODEL}" in grasp and "not yet been measured" in grasp
    assert "jaw flip about the grip site's local +Z" in grasp
    for tool in ("move_to", "rotate_wrist", "rotate_pitch"):
        assert help_text[tool].endswith("the same approximate pre-check as move_pose's.")
    for name in ("libero-move-pose-reach-reference-help-v1.json", "libero-reach-nearest-orientation-help-v1.json",
                 "libero-grasp-plan-ik-selection-help-v1.json"):
        assert (PORTS / name).is_file()


# --------------------------------------------------------------------------
# grasp_plan on the single-arm LIBERO surface
# --------------------------------------------------------------------------

def _grasp_runtime(tmp_path, monkeypatch, bridge, grasp_backend=None):
    from tests.agent.perception.test_grounding_grasp_runtime import _runtime

    runtime, _ = _runtime(tmp_path, grasp_backend=grasp_backend)
    runtime.surface = resolve_tool_surface(DEFAULT)
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    runtime.align_plan_callback = bridge
    return runtime


def test_libero_grasp_plan_checks_its_whole_pool_on_the_right_arm_through_result_admission(tmp_path, monkeypatch):
    """test_reach_check's five-candidate pool (tilts 180, 40, 0, 20, 0 degrees) on LIBERO's compiler."""
    from tests.agent.motion.test_reach_check import PoolProposer

    requests = []
    backend = PoolProposer()
    runtime = _grasp_runtime(tmp_path, monkeypatch, _bridge(requests, planner=TiltIk(reach_m=5.0)), backend)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object"})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"]})
    assert backend.requested == [policy_reach.POOL_LIMIT]
    ids = [f"grasp-pool{index:020d}" for index in range(5)]
    # Reachable and near-vertical (2, 3, 4) in backend order; the 40-degree and
    # upward candidates are refused by the 35-degree stand-in and fall below the cap.
    assert [item["candidate_id"] for item in result["candidates"]] == ids[2:]
    summary = result["ik_reachability"]
    assert summary["status"] == "annotated" and summary["ik_model"] == libero_reach.IK_MODEL
    assert summary["matches_move_pose_admission"] is False
    assert summary["checked_candidate_count"] == 5 and summary["reachable_candidate_count"] == 3
    # The exact member for the whole pool, then the jaw flip for the two it refused.
    assert [len(request["queries"]) for request in requests] == [10, 4]
    assert all(request["arm"] == "right" for request in requests)
    pair = requests[0]["queries"][:2]
    assert [query["interaction"] for query in pair] == ["free_space", "contact_approach"]
    assert [query["from_index"] for query in pair] == [None, 0]
    manifest = json.loads(next(runtime.artifact_root.glob("grasp-queries/*/manifest.json")).read_text())
    refused = {item["candidate_id"]: item["ik_reachability"] for item in manifest["candidate_selection"]["pool"]}
    for candidate_id in ids[:2]:
        assert refused[candidate_id]["reachable"] is False
        assert refused[candidate_id]["pregrasp"] == {"reachable": False, "reason": "ik_failed"}
        assert refused[candidate_id]["approach"] == {"reachable": False, "reason": "start_unreachable"}
    assert "arm" not in result  # single-arm grasp_plan keeps its released result shape


def test_a_libero_bridge_fault_keeps_the_backend_order(tmp_path, monkeypatch):
    def broken(request):
        raise RuntimeError("CuRobo worker closed without a response")

    runtime = _grasp_runtime(tmp_path, monkeypatch, broken)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object"})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"]})
    assert result["ik_reachability"]["status"] == "unavailable"
    assert all(item["ik_reachability"]["reachable"] is None for item in result["candidates"])

