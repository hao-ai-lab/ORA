"""give_up records the plan candidates offered but not tried this round (F4).

09-09 condition (a): EXP11A g008/g009 finish records named options the policy
had not tried ("handover ... not attempted this round given the timings").
User decision 2026-10-04: record whether a give_up reason names such
candidates -- by candidate id on RoboTwin and RoboDojo, approximately on
LIBERO -- and never refuse a finish because of it. The ledger is read from the
run's own event log since the round's reset; nothing the policy asserts enters
it except the reason text being searched.

Offline: event-log fixtures in the exact shapes the dispatcher appends, and the
real dispatcher on test-local v2 surfaces (``finish_v2_surfaces``).
"""

from __future__ import annotations

import json

import pytest

import finish_v2_surfaces as v2
from ora.agent.tools import finish_candidates
from ora.agent.tools.finish_candidates import (
    CANDIDATE_ID, ORIENTATION_TOLERANCE_DEG, POSE_MATCH, POSITION_TOLERANCE_M, ledger,
)

A, B, C = (f"grasp-{letter * 24}" for letter in "abc")
IDENTITY = (CANDIDATE_ID, POSE_MATCH)
LIBERO = (POSE_MATCH,)
QUAT = [0.0, 0.0, 0.0, 1.0]
#: 30 degrees about z.
TURNED = [0.0, 0.0, 0.25881904510252074, 0.9659258262890683]
RESET = {"type": "reset", "time": 1.0, "record": {"round_index": 1}}


def _pose(x):
    return {"position_m": [x, 0.1, 0.9], "orientation_xyzw": QUAT}


def _grasp_plan(*ids):
    return {"type": "tool_result", "tool": "grasp_plan", "arguments": {"target_ref": "t"},
            "result": {"status": "candidates", "candidates": [
                {"candidate_id": cid, "rank": rank, "pregrasp_pose": _pose(0.1 * rank),
                 "contact_pose": _pose(0.1 * rank + 0.05)}
                for rank, cid in enumerate(ids, start=1)]},
            "started": 1.0, "finished": 1.0}


def _place_plan(count, *, dual_arm=True):
    def move(x):
        target = {"position_mode": "absolute", "position_m": [x, -0.2, 1.0],
                  "orientation_mode": "absolute_quaternion", "orientation_xyzw": QUAT}
        return {"targets": {"left": target}} if dual_arm else target

    return {"type": "tool_result", "tool": "place_plan", "arguments": {},
            "result": {"status": "candidates", "candidates": [
                {"candidate_id": f"P{index}", "move_pose": move(0.3 + 0.1 * index)}
                for index in range(count)]},
            "started": 1.0, "finished": 1.0}


def _execute(tool, candidate_id):
    return {"type": "tool_result", "tool": tool,
            "arguments": {"candidate_id": candidate_id, "rationale": "best ranked"},
            "result": {"success": True}, "started": 1.0, "finished": 1.0}


def _refused_execute(candidate_id):
    return {"type": "tool_error", "tool": "grasp_execute",
            "arguments": {"candidate_id": candidate_id, "rationale": "x"},
            "error": {"code": "stale_candidate_observation", "message": "x", "retryable": True}, "time": 1.0}


def _move_pose(position, orientation=QUAT, *, mode="absolute"):
    arguments = {"position_mode": mode, "orientation_mode": "absolute_quaternion",
                 "orientation_xyzw": orientation}
    arguments["position_m" if mode == "absolute" else "translation_m"] = position
    return {"type": "tool_result", "tool": "move_pose", "arguments": arguments,
            "result": {"success": True}, "started": 1.0, "finished": 1.0}


def _lines(*events):
    return [json.dumps(event) for event in events]


def _untried(value):
    return value["untried_candidate_ids"]


def test_offered_grasp_candidates_are_tried_by_id_and_the_reason_is_searched():
    value = ledger(_lines(RESET, _grasp_plan(A, B, C), _execute("grasp_execute", A),
                          _refused_execute(B)),
                   reason=f"{B} would collide with the bowl, so this round cannot finish", methods=IDENTITY)
    assert value["methods"] == list(IDENTITY)
    assert _untried(value) == [B, C]
    assert value["named_untried_candidate_ids"] == [B] and value["reason_names_untried"] is True
    tried = {item["candidate_id"]: item["tried_by"] for item in value["offered"]}
    assert tried == {A: [CANDIDATE_ID], B: [], C: []}


def test_a_reason_that_names_no_untried_candidate_is_recorded_as_such():
    value = ledger(_lines(RESET, _grasp_plan(A, B), _execute("grasp_execute", A), _execute("grasp_execute", B)),
                   reason=f"both candidates {A} and {B} slipped during the close", methods=IDENTITY)
    assert _untried(value) == [] and value["reason_names_untried"] is False
    assert value["named_untried_candidate_ids"] == []


def test_only_the_current_round_counts():
    value = ledger(_lines(_grasp_plan(A), RESET, _grasp_plan(B)), reason=f"{A} {B}", methods=IDENTITY)
    assert [item["candidate_id"] for item in value["offered"]] == [B]
    assert value["named_untried_candidate_ids"] == [B]


def test_place_ids_belong_to_the_plan_call_that_offered_them():
    value = ledger(_lines(RESET, _place_plan(2), _execute("place_execute", "P0"), _place_plan(1)),
                   reason="P1 was outside the shelf, and P0 slid off", methods=IDENTITY)
    offered = [(item["candidate_id"], item["call"], item["tried_by"]) for item in value["offered"]]
    assert offered == [("P0", 1, [CANDIDATE_ID]), ("P1", 1, []), ("P0", 2, [])]
    # An id the policy tried from any call is not reported untried: the reason
    # cannot say which call's P0 it means.
    assert _untried(value) == ["P1"]
    assert value["named_untried_candidate_ids"] == ["P1"]


def test_an_id_tried_from_a_later_plan_call_is_not_reported_untried():
    value = ledger(_lines(RESET, _place_plan(2), _place_plan(1), _execute("place_execute", "P0")),
                   reason="P0 was placed but the box tipped over afterwards", methods=IDENTITY)
    offered = [(item["candidate_id"], item["call"], item["tried_by"]) for item in value["offered"]]
    assert offered == [("P0", 1, []), ("P1", 1, []), ("P0", 2, [CANDIDATE_ID])]
    assert _untried(value) == ["P1"]
    assert value["named_untried_candidate_ids"] == [] and value["reason_names_untried"] is False


def test_a_repeated_grasp_identity_is_one_candidate():
    value = ledger(_lines(RESET, _grasp_plan(A, B), _grasp_plan(A)), reason="x", methods=IDENTITY)
    assert [item["candidate_id"] for item in value["offered"]] == [A, B]


def test_libero_matches_a_copied_pose_approximately():
    near = [0.1 + POSITION_TOLERANCE_M / 2, 0.1, 0.9]
    far = [0.2 + POSITION_TOLERANCE_M * 3, 0.1, 0.9]
    value = ledger(_lines(RESET, _grasp_plan(A, B, C), _move_pose(near),
                          _move_pose(far), _move_pose([0.3, 0.1, 0.9], TURNED),
                          _move_pose([0.1, 0.1, 0.9], mode="delta")),
                   reason=f"{B} and {C} were never reachable", methods=LIBERO)
    assert value["methods"] == [POSE_MATCH]
    assert value["position_tolerance_m"] == POSITION_TOLERANCE_M
    assert value["orientation_tolerance_deg"] == ORIENTATION_TOLERANCE_DEG
    tried = {item["candidate_id"]: item["tried_by"] for item in value["offered"]}
    # A's pregrasp was copied; B was missed by more than the tolerance; C's
    # position was reached with a 30-degree turn, which is not its pose.
    assert tried == {A: [POSE_MATCH], B: [], C: []}
    assert value["named_untried_candidate_ids"] == [B, C]


def test_libero_place_candidates_match_flat_move_pose_targets():
    value = ledger(_lines(RESET, _place_plan(2, dual_arm=False), _move_pose([0.3, -0.2, 1.0])),
                   reason="P1 is outside the drawer", methods=LIBERO)
    assert {item["candidate_id"]: item["tried_by"] for item in value["offered"]} == {
        "P0": [POSE_MATCH], "P1": []}
    assert value["reason_names_untried"] is True


def test_an_unreadable_line_is_not_a_candidate():
    value = ledger(["not json", json.dumps({"type": "tool_result", "tool": "grasp_plan"}),
                    json.dumps([1])], reason="x", methods=IDENTITY)
    assert value["offered"] == [] and value["reason_names_untried"] is False


@pytest.mark.parametrize("backend,methods", [
    ("robotwin", IDENTITY), ("robodojo", IDENTITY), ("libero", LIBERO)])
def test_the_method_follows_the_surface_tools(monkeypatch, tmp_path, backend, methods):
    surface = v2.resolve(monkeypatch, tmp_path, backend)
    assert finish_candidates.methods_for(surface) == methods


@pytest.mark.parametrize("backend", tuple(v2.DEFAULTS))
def test_give_up_naming_untried_candidates_is_admitted_and_recorded(monkeypatch, tmp_path, backend):
    import test_finish_outcomes as outcomes

    runtime = outcomes._runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    outcomes._log(runtime, RESET, _grasp_plan(A, B), _execute("grasp_execute", A))
    reason = f"candidate {B} was offered but the bowl blocks it; the round cannot finish"
    accepted = runtime.invoke("finish_episode", {"outcome": "give_up", "reason": reason,
                                                 "category": "unreachable"})
    assert accepted["outcome"] == "give_up"
    record = json.loads((runtime.control_root / "policy-finish.json").read_text())
    untried = record["untried_candidates"]
    assert untried["named_untried_candidate_ids"] == [B] and untried["reason_names_untried"] is True
    expected = [B] if backend != "libero" else [A, B]
    assert untried["untried_candidate_ids"] == expected
