"""LIBERO answers the v2 motion probe with CuRobo IK candidate counts (integration 0059 M2).

A surface declaring ``motion_probe_ik_candidate_feedback_v1`` sends
``ora.motion-probe-request.v2``; until now only RoboDojo answered it
(``test_robodojo_probe_ik_feedback.py``) and LIBERO refused it as drift. Each
LIBERO probe variant is a read-only CuRobo plan whose IK stage the worker's IK
observer already records; the v2 answer keeps that seed population as counts
in the shared vocabulary, exactly the three planner fields RoboDojo adds.
``motion_probe.py`` (a pinned embodiment-descriptor source) is unchanged: the
v2 answer wraps it.

Offline: the real ``LiberoProSession`` over the CPU plant of
``test_libero_correctness_repair`` and a planner stand-in whose responses
carry captured IK attempts in the worker's format.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.tools import grasp_feasibility
from ora.agent.tools.motion_probe_feedback import (
    IK_CANDIDATE_FEEDBACK_CAPABILITY, summarize_motion_probe,
)
from ora.agent.tools.robot import ToolError
from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS, PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
    validate_planning_diagnostics,
)
from ora.environment.backends.libero_pro.planning_diagnostics import ik_candidate_tally
from ora.environment.backends.libero_pro.session import LiberoProError
from tests.environments.libero.test_libero_correctness_repair import JOINT_NAMES, TARGET, _Planner, _session

#: The released LIBERO v1 probe planner record.
V1_PLANNER_KEYS = {
    "called", "status", "stage", "valid_query", "attempts", "trajopt_attempts", "used_graph",
    "goalset_index", "path_steps", "interpolation_dt_s", "errors", "timings_s", "error",
}


def _attempt(*, accepted=0, outside=0, infeasible=0, unconverged=0):
    """One captured IK seed population in ``planning_diagnostics.capture_attempt``'s format."""
    rows = ([(True, True, True, False)] * accepted + [(False, False, True, True)] * outside
            + [(False, False, True, False)] * infeasible + [(False, True, False, False)] * unconverged)
    return {"status": "captured", "accepted": [row[0] for row in rows], "feasible": [row[1] for row in rows],
            "converged": [row[2] for row in rows], "outside_position_limits": [row[3] for row in rows],
            "joint_positions": [[0.0] * 7 for _ in rows]}


def _diagnostics(outcome, phase):
    return validate_planning_diagnostics({
        "phase": phase, "outcome": outcome, "details_available": True, "recovery": [],
        "candidate_counts": {"optimized": 32, "returned": 4}, "rejection_counts": {"position_tolerance": 3},
        "unknown_reasons": ["per_term_constraint_costs_not_retained_by_sdk"],
        "evidence_scope": "last_recorded_original_ik_attempt", "note": "released LIBERO note",
    })


class _FeedbackPlanner(_Planner):
    """Each plan returns the next scripted IK attempt beside the released response."""

    def __init__(self, attempts, *, refuse=False):
        super().__init__()
        self.attempts = list(attempts)
        self.refuse_all = refuse

    def plan(self, request):
        attempt = self.attempts.pop(0)
        if self.refuse_all:
            self.requests.append(deepcopy(request))
            body = {"reachable": False, "status": "not_reachable", "planner_status": "MotionGenStatus.IK_FAIL"}
        else:
            body = super().plan(request)
        phase = "ik" if self.refuse_all else "trajectory_planning"
        return {**body, "planning_diagnostics": _diagnostics(body["planner_status"], phase),
                "planning_diagnostic_evidence": {"schema": "ora.libero-ik-diagnostics.v1",
                                                 "attempts": [] if attempt is None else [attempt],
                                                 "attempts_dropped": 0, "world_model": "none",
                                                 "extra_solver_calls": 0}}


def _request(*, v2=True, policy="exact"):
    request = {"schema": "ora.motion-probe-request.v1", "observation_id": "observation-0",
               "observation_sha256": "0" * 64, "variant_policy": policy,
               "queries": [{"query_id": "q0", "arm": "right", "pose": {
                   "position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}}]}
    if v2:
        request.update(schema=PROBE_FEEDBACK_REQUEST_SCHEMA, planner_feedback=PROBE_IK_CANDIDATE_FEEDBACK)
    return request


def test_the_v1_request_keeps_the_released_planner_record(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    session.motion_planner = _FeedbackPlanner([_attempt(accepted=5, unconverged=27)])
    planner = session.probe_motion(_request(v2=False))["results"][0]["variants"][0]["planner"]
    assert set(planner) == V1_PLANNER_KEYS
    assert not (tmp_path / "motion-probes/ik-diagnostics").exists()


def test_a_refused_variant_keeps_the_ik_seed_population_it_was_refused_on(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    real = _FeedbackPlanner([_attempt(outside=4, infeasible=10, unconverged=18)], refuse=True)
    session.motion_planner = real
    result = session.probe_motion(_request())
    assert result["schema"] == "ora.motion-probe-result.v1"
    assert session.motion_planner is real  # the recorder never outlives the call
    variant = result["results"][0]["variants"][0]
    assert variant["reachable"] is False and variant["failure"]["code"] == "MotionGenStatus.IK_FAIL"
    planner = variant["planner"]
    assert set(planner) - V1_PLANNER_KEYS == {
        "constraint_rejection_counts", "planning_diagnostics", "ik_validation_artifact"}
    counts = planner["constraint_rejection_counts"]
    assert list(counts) == list(IK_CONSTRAINT_KEYS)
    assert counts == {"non_finite_solution": 0, "solver_unsuccessful": 10, "joint_limit": 4,
                      "fk_residual_exceeded": 18, "joint_jump": 0, "pitch_joint_wrap": 0}
    diagnostics = validate_planning_diagnostics(planner["planning_diagnostics"])
    assert diagnostics["candidate_counts"] == {"returned": 32, "valid_endpoint": 0}
    assert diagnostics["rejection_counts"] == {"joint_limit": 4, "solver_unsuccessful": 10,
                                               "fk_residual_exceeded": 18}
    assert diagnostics["outcome"] == "MotionGenStatus.IK_FAIL" and diagnostics["phase"] == "ik"
    assert "not LIBERO filters" in diagnostics["note"]
    reference = planner["ik_validation_artifact"]
    path = tmp_path / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    record = json.loads(path.read_text())
    assert record["schema"] == reference["schema"] == "ora.libero-probe-ik.v1"
    assert record["query_id"] == "q0" and record["variant_id"] == variant["variant_id"]
    assert record["tally"]["returned"] == 32 and len(record["ik_attempt"]["accepted"]) == 32
    planner_evidence = tmp_path / record["planner_evidence"]["path"]
    assert hashlib.sha256(planner_evidence.read_bytes()).hexdigest() == record["planner_evidence"]["sha256"]
    assert json.loads(planner_evidence.read_text())["planner_status"] == "MotionGenStatus.IK_FAIL"


def test_each_variant_carries_its_own_plan_s_counts(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    session.motion_planner = _FeedbackPlanner([_attempt(accepted=7, infeasible=25), _attempt(accepted=2, outside=30)])
    result = session.probe_motion(_request(policy="parallel_jaw_equivalents"))
    exact, flip = result["results"][0]["variants"]
    assert exact["reachable"] and flip["reachable"]
    assert exact["planner"]["planning_diagnostics"]["candidate_counts"] == {"returned": 32, "valid_endpoint": 7}
    assert exact["planner"]["constraint_rejection_counts"]["solver_unsuccessful"] == 25
    assert flip["planner"]["planning_diagnostics"]["candidate_counts"] == {"returned": 32, "valid_endpoint": 2}
    assert flip["planner"]["constraint_rejection_counts"]["joint_limit"] == 30
    assert exact["planner"]["ik_validation_artifact"]["path"] != flip["planner"]["ik_validation_artifact"]["path"]
    assert result["results"][0]["plan_ref"] is not None  # an admitted plan is still stored


def test_no_captured_population_is_said_not_invented(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    session.motion_planner = _FeedbackPlanner([None], refuse=True)
    planner = session.probe_motion(_request())["results"][0]["variants"][0]["planner"]
    assert set(planner["constraint_rejection_counts"].values()) == {0}
    assert planner["planning_diagnostics"]["details_available"] is False
    assert planner["planning_diagnostics"]["candidate_counts"] == {}
    record = json.loads((tmp_path / planner["ik_validation_artifact"]["path"]).read_text())
    assert record["tally"] is None and record["ik_attempt"] is None


@pytest.mark.parametrize("damage", [
    {"planner_feedback": "everything"}, {"planner_feedback": None}, {"extra": 1},
])
def test_a_malformed_feedback_request_is_refused(tmp_path, monkeypatch, damage):
    session, _plant = _session(tmp_path, monkeypatch)
    session.motion_planner = _FeedbackPlanner([_attempt(accepted=1)])
    with pytest.raises(LiberoProError, match="drifted"):
        session.probe_motion({**_request(), **damage})
    with pytest.raises(LiberoProError, match="drifted"):  # the released v1 check still applies
        session.probe_motion({**_request(v2=False), "planner_feedback": PROBE_IK_CANDIDATE_FEEDBACK})


def test_the_planner_is_restored_when_the_probe_fails(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    real = _FeedbackPlanner([])  # popping from an empty script raises inside the probe
    session.motion_planner = real
    with pytest.raises(IndexError):
        session.probe_motion(_request())
    assert session.motion_planner is real


def test_the_tally_is_exclusive_and_sums_to_the_rejected_seeds():
    tally = ik_candidate_tally(_attempt(accepted=3, outside=2, infeasible=5, unconverged=22))
    assert tally["returned"] == 32 and tally["valid_endpoint"] == 3
    assert sum(tally["rejections"].values()) == 32 - 3
    assert ik_candidate_tally({"status": "unavailable", "reason": "ik_metric_capture_failed"}) is None
    unknown_limits = {**_attempt(outside=2), "outside_position_limits": None}
    assert ik_candidate_tally(unknown_limits)["rejections"]["solver_unsuccessful"] == 2
    assert ik_candidate_tally(unknown_limits)["limits_known"] is False


def test_the_summary_and_the_pregrasp_refusal_deliver_the_counts(tmp_path, monkeypatch):
    """The grasp_execute precommit path a declaring LIBERO surface would take."""
    session, _plant = _session(tmp_path, monkeypatch)
    session.motion_planner = _FeedbackPlanner([_attempt(unconverged=32), _attempt(infeasible=32)] * 3, refuse=True)
    result = session.probe_motion(_request())
    row = summarize_motion_probe(result, ik_candidates=True)[0]
    assert row["ik_candidate_counts"] == {"returned": 32, "valid_endpoint": 0}
    assert row["ik_outcome_counts"] == {"MotionGenStatus.IK_FAIL": 1}
    assert "ik_candidate_counts" not in summarize_motion_probe(result)[0]

    surface = SimpleNamespace(supports=lambda name: name == IK_CANDIDATE_FEEDBACK_CAPABILITY)

    def probe(arguments: dict[str, Any]) -> dict[str, Any]:
        return session.probe_motion({**_request(), **arguments})

    pose = {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}
    host = SimpleNamespace(surface=surface, _motion_probe=probe)
    commitment = {"arm": "right", "observation_id": "observation-0", "candidate_id": "grasp-a1",
                  "poses": {"pregrasp": dict(pose), "approach": dict(pose)}}
    with pytest.raises(ToolError) as refused:
        grasp_feasibility.check_pregrasp(host, commitment)
    assert refused.value.code == "grasp_pregrasp_unreachable"
    summary = refused.value.constraints["summary"]
    assert summary["ik_candidate_counts"]["returned"] >= 32
    assert summary["ik_candidate_counts"]["valid_endpoint"] == 0
    assert JOINT_NAMES[0] == "panda_joint1"
