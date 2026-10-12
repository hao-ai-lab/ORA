"""D3L-R002: a refused Dojo pregrasp keeps the IK evidence it was refused on.

The Dojo worker solves every probe variant with a bounded SDK IK call that
returns up to 32 candidates and records why each was rejected. The released v1
probe record reduces that to ``status: Fail``, and the probe summary the policy
receives reads only ``constraint_rejection_counts``, so a pregrasp refusal
arrived as ``Fail:1 / plan_search_or_admission``.

The frozen ``robodojo_unified_manip_v1`` probe result schema is closed and was
accepted with the v1 record, so the fix is opt-in by surface: a surface
declaring ``motion_probe_ik_candidate_feedback_v1`` sends the v2 request and
only that request receives the extra planner fields.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from tests.test_robodojo_backend import fake_sdk_candidate_boundary  # noqa: F401

import test_robodojo_backend as backend
from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import grasp_feasibility
from ora.agent.tools.motion_probe_feedback import (
    IK_CANDIDATE_FEEDBACK_CAPABILITY, probe_request_fields, summarize_motion_probe,
)
from ora.agent.tools.robot import ToolError
from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS, PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
)
from ora.environment.backends.robodojo.bridge import RobodojoError

DOJO_CLOSED = "robodojo_placement_closed_skills_v2"
V1_PLANNER_KEYS = {
    "called", "status", "stage", "valid_query", "attempts", "trajopt_attempts", "used_graph",
    "goalset_index", "path_steps", "interpolation_dt_s", "errors", "timings_s", "error",
    "planner", "parity",
}


@pytest.fixture
def sparse(tmp_path):
    instance = backend.sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def _v2(request):
    return {**request, "schema": PROBE_FEEDBACK_REQUEST_SCHEMA,
            "planner_feedback": PROBE_IK_CANDIDATE_FEEDBACK}


def _refused(sparse, tmp_path, *, feedback: bool):
    sparse.environment.robot_manager.fail_ik = True
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    try:
        return sparse.probe_motion(_v2(request) if feedback else request)
    finally:
        sparse.environment.robot_manager.fail_ik = False


def test_the_v1_request_keeps_the_released_planner_record(sparse, tmp_path):
    result = _refused(sparse, tmp_path, feedback=False)
    planner = result["results"][0]["variants"][0]["planner"]
    assert set(planner) == V1_PLANNER_KEYS


def test_the_v2_request_keeps_the_sdk_candidate_evidence(sparse, tmp_path):
    result = _refused(sparse, tmp_path, feedback=True)
    variant = result["results"][0]["variants"][0]
    assert variant["reachable"] is False and variant["failure"]["code"] == "Fail"
    planner = variant["planner"]
    assert set(planner) - V1_PLANNER_KEYS == {
        "constraint_rejection_counts", "planning_diagnostics", "ik_validation_artifact",
    }
    assert list(planner["constraint_rejection_counts"]) == list(IK_CONSTRAINT_KEYS)
    assert planner["constraint_rejection_counts"]["solver_unsuccessful"] == 1
    assert sum(planner["constraint_rejection_counts"].values()) == 1
    diagnostics = planner["planning_diagnostics"]
    assert diagnostics["outcome"] == "no_valid_endpoint"
    assert diagnostics["candidate_counts"] == {"returned": 1, "valid_endpoint": 0, "continuous": 0}
    reference = planner["ik_validation_artifact"]
    path = sparse.output_root / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    assert json.loads(path.read_text())["ik"]["ik_selection"]["outcome"] == "no_valid_endpoint"


def test_an_admitted_variant_counts_no_rejection(sparse, tmp_path):
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    planner = sparse.probe_motion(_v2(request))["results"][0]["variants"][0]["planner"]
    assert set(planner["constraint_rejection_counts"].values()) == {0}
    assert planner["planning_diagnostics"]["outcome"] == "admitted"


@pytest.mark.parametrize("damage", [
    {"planner_feedback": "everything"},
    {"schema": "ora.motion-probe-request.v1"},
    {"planner_feedback": None},
])
def test_a_malformed_feedback_request_is_refused(sparse, tmp_path, damage):
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    with pytest.raises(RobodojoError, match="drifted"):
        sparse.probe_motion({**_v2(request), **damage})


def test_the_summary_names_ik_counts_only_when_asked(sparse, tmp_path):
    result = _refused(sparse, tmp_path, feedback=True)
    plain = summarize_motion_probe(result)[0]
    assert "ik_candidate_counts" not in plain and "ik_outcome_counts" not in plain
    assert plain["constraint_rejection_counts"]["solver_unsuccessful"] == 1
    row = summarize_motion_probe(result, ik_candidates=True)[0]
    assert row["ik_candidate_counts"] == {"returned": 1, "valid_endpoint": 0, "continuous": 0}
    assert row["ik_outcome_counts"] == {"no_valid_endpoint": 1}
    # A v1 record carries no diagnostics, so asking adds nothing invented.
    v1 = summarize_motion_probe(_refused(sparse, tmp_path, feedback=False), ik_candidates=True)[0]
    assert "ik_candidate_counts" not in v1 and "constraint_rejection_counts" not in v1


@pytest.mark.parametrize("profile,schema", [
    (DOJO_CLOSED, PROBE_FEEDBACK_REQUEST_SCHEMA),
    ("robotwin_toolchain_b2_closed_noservo_v2", PROBE_FEEDBACK_REQUEST_SCHEMA),
    ("libero_placement_direct_overhead_v1", "ora.motion-probe-request.v1"),
])
def test_only_the_declaring_surface_sends_the_feedback_request(profile, schema):
    fields = probe_request_fields(resolve_tool_surface(profile))
    assert fields["schema"] == schema
    assert ("planner_feedback" in fields) is (schema == PROBE_FEEDBACK_REQUEST_SCHEMA)


def test_the_pregrasp_refusal_delivers_the_counts(sparse, tmp_path):
    """The exact D3L-R002 path: grasp_execute's precommit check refuses."""
    surface = resolve_tool_surface(DOJO_CLOSED)
    assert surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY)
    sparse.environment.robot_manager.fail_ik = True
    observation, _request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])

    def probe(arguments):
        return sparse.probe_motion({
            **probe_request_fields(surface), "observation_id": observation["observation_id"],
            "observation_sha256": "a" * 64, **arguments,
        })

    host = SimpleNamespace(surface=surface, _motion_probe=probe)
    pose = {"position_m": [0.05, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]}
    commitment = {"arm": "left", "observation_id": observation["observation_id"],
                  "candidate_id": "grasp-a1", "poses": {"pregrasp": pose, "approach": pose}}
    with pytest.raises(ToolError) as refused:
        grasp_feasibility.check_pregrasp(host, commitment)
    assert refused.value.code == "grasp_pregrasp_unreachable"
    summary = refused.value.constraints["summary"]
    assert summary["ik_outcome_counts"] == {"no_valid_endpoint": 2}
    assert summary["ik_candidate_counts"]["returned"] >= 2
    assert summary["ik_candidate_counts"]["valid_endpoint"] == 0
    assert summary["constraint_rejection_counts"]["solver_unsuccessful"] >= 2


def test_a_pitch_joint_wrap_rejection_is_counted_not_dropped():
    """Review U1: the v2 planner record counts the pitch-joint refusal beside the other filters."""
    from ora.environment.backends.robodojo import sparse_motion

    solved = {"joint_target": None, "ik_status": "Fail", "planning_diagnostics": None,
              "candidates": [{"rejection": "pitch_joint_wrap"}, {"rejection": "joint_jump"}]}
    record = sparse_motion._planner_summary(solved, ik_feedback={"path": "p", "schema": "s", "sha256": "0" * 64})
    assert record["constraint_rejection_counts"]["pitch_joint_wrap"] == 1
    assert sum(record["constraint_rejection_counts"].values()) == 2
