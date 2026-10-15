"""move_to, rotate_wrist and rotate_pitch carry the move_pose reach reference.

EXP11A g009 stack_bowls issued 37 move_to against 6 move_pose, and two refused
lifts in its last round reported only planner_failed (E11G9-001). Under
pose_primitive_reach_along_request_v1 each primitive target is checked with
the read-only reach check on its own route and an unsolved one reports how far
along its straight path that route solves; the primitive is submitted
unchanged. Offline: a stub IK bridge and the fake Dojo plant observation with
result-schema admission.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.move_pose_reach_schema import LINE_FRACTIONS
from ora.agent.composition.pose_primitive_reach_schema import (
    CAPABILITY, PRIMITIVES, bind_pose_primitive_reach,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.contracts.reach_check import RESULT_SCHEMA

DOJO = "robodojo_placement_closed_skills_v2"
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v2", "robotwin_toolchain_b2_closed_noservo_v1")


def _bridge(requests, *, target_solves=False, line=lambda fraction: fraction <= 0.5):
    def bridge(request):
        requests.append(copy.deepcopy(request))
        sampled = len(request["queries"]) == len(LINE_FRACTIONS)
        rows = []
        for index, _query in enumerate(request["queries"]):
            ok = line(LINE_FRACTIONS[index]) if sampled else target_solves
            rows.append({"reachable": ok, "reason": None if ok else "ik_failed", "route": "global",
                         "position_error_m": 0.0001 if ok else None, "orientation_error_rad": 0.001 if ok else None,
                         "joint_limit_margin_rad": 1.0 if ok else None, "joint_delta_rad": 0.2 if ok else None})
        return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": "robodojo_sdk_ik_endpoint_selection_v2",
                "matches_move_pose_admission": True, "motion_planner_called": False,
                "claim_boundary": "stub IK", "results": rows}
    return bridge


def _runtime(tmp_path, *, declared=True):
    from tests.agent.motion.test_move_pose_reach import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    if not declared:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            item for item in runtime.surface.capabilities if item != CAPABILITY))
    return runtime


def test_an_unsolved_move_to_reports_reach_along_its_free_space_route(tmp_path):
    runtime = _runtime(tmp_path)
    requests = []
    runtime.align_plan_callback = _bridge(requests)
    result = runtime.invoke("move_to", {"targets": {"left": {"position_m": [0.3, -0.1, 0.9]}}})
    reach = result["compiled_pose"]["targets"]["left"]["reach_along_request"]
    assert reach["target_reachable"] is False and reach["farthest_reachable_fraction"] == 0.5
    assert [query["interaction"] for query in requests[0]["queries"]] == ["free_space"]
    assert {query["interaction"] for query in requests[1]["queries"]} == {"free_space"}
    # Translation only: every sample holds the measured orientation.
    current = [0.0, 0.0, 0.0, 1.0]
    assert all(query["pose_xyzw"][3:] == pytest.approx(current) for query in requests[1]["queries"])
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert submitted["arguments"]["targets"]["left"]["position_m"] == pytest.approx([0.3, -0.1, 0.9])
    assert submitted["arguments"]["targets"]["left"]["orientation_xyzw"] == pytest.approx(current)


@pytest.mark.parametrize("tool", ["rotate_wrist", "rotate_pitch"])
def test_a_rotation_is_checked_on_its_precision_route(tmp_path, tool):
    runtime = _runtime(tmp_path)
    requests = []
    runtime.align_plan_callback = _bridge(requests, target_solves=True)
    result = runtime.invoke(tool, {"targets": {"right": 30.0}})
    assert result["compiled_pose"]["targets"]["right"]["reach_along_request"] == {"target_reachable": True}
    assert [query["interaction"] for query in requests[0]["queries"]] == ["precision"]
    assert len(requests) == 1


def test_an_unavailable_reference_never_blocks_the_primitive(tmp_path):
    runtime = _runtime(tmp_path)

    def broken(request):
        raise RuntimeError("worker died")

    runtime.align_plan_callback = broken
    result = runtime.invoke("move_to", {"targets": {"left": {"position_m": [0.3, -0.1, 0.9]}}})
    assert result["status"] == "submitted"
    assert result["compiled_pose"]["targets"]["left"]["reach_along_request"] == {
        "status": "unavailable", "failure": "RuntimeError"}


def test_a_surface_without_the_capability_submits_as_released(tmp_path):
    runtime = _runtime(tmp_path, declared=False)
    requests = []
    runtime.align_plan_callback = _bridge(requests)
    result = runtime.invoke("move_to", {"targets": {"left": {"position_m": [0.3, -0.1, 0.9]}}})
    assert requests == [] and "reach_along_request" not in result["compiled_pose"]["targets"]["left"]


def test_the_live_dojo_surface_declares_it_and_robotwin_does_not():
    dojo = resolve_tool_surface(DOJO)
    assert dojo.supports(CAPABILITY)
    for name in PRIMITIVES:
        targets = dojo.result_schemas[name]["properties"]["compiled_pose"]["properties"]["targets"]
        # observed_clearance is observed_clearance_reference_v1's (test_observed_clearance_runtime.py) and
        # target_axes_world is pose_orientation_from_world_axes_v1's (test_pose_world_axes.py).
        assert set(targets["additionalProperties"]["properties"]) == {
            "reach_along_request", "observed_clearance", "target_axes_world"}
    for profile in ROBOTWIN_LIVE:
        assert not resolve_tool_surface(profile).supports(CAPABILITY)


def test_the_binder_refuses_a_surface_it_cannot_describe():
    results = {"move_to": {"type": "object", "properties": {"compiled_pose": {"type": "object"}}}}
    with pytest.raises(ToolSurfaceError, match="semantic_pose_primitives_v1"):
        bind_pose_primitive_reach(copy.deepcopy(results), [CAPABILITY])
    full = [CAPABILITY, "semantic_pose_primitives_v1", "motion_intent_router_v1"]
    with pytest.raises(ToolSurfaceError, match="at least one"):
        bind_pose_primitive_reach({}, full)
    with pytest.raises(ToolSurfaceError, match="move_to compiled_pose"):
        bind_pose_primitive_reach({"move_to": {"type": "object"}}, full)
