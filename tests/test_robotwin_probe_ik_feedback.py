"""RoboTwin answers the v2 motion-probe request (``motion_probe_ik_candidate_feedback_v1``).

The agent side (``agent/tools/motion_probe_feedback.py``) is backend-neutral:
a surface declaring the capability sends ``ora.motion-probe-request.v2``
with ``planner_feedback: ik_candidate_counts_v1``, and the ``grasp_execute``
pregrasp refusal sums each variant's ``candidate_counts``. Until 0059 only the
RoboDojo worker answered it. These tests pin RoboTwin's answer to the RoboDojo
contract on the same candidate set -- every solution one bounded exact-pose IK
solve returned for one variant -- and in the same unit, a count of those
solutions:

* ``returned``: solutions the solve returned;
* ``valid_endpoint``: those that passed every endpoint filter (finite, solver
  success -- which on CuRobo already includes its pose tolerance and collision
  feasibility -- and joint limits);
* ``continuous``: those valid endpoints inside the backend's joint-jump bound.
  RoboTwin applies no joint-jump filter to IK endpoints (continuity is decided
  by trajectory planning on at most 16 deduplicated joint goals), so the count
  is known only when no endpoint is valid (0) and is ``None`` otherwise -- never
  a different quantity under the same key.

Released v1 requests keep the released record byte for byte.
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import grasp_feasibility
from ora.agent.tools.motion_probe_feedback import (
    IK_CANDIDATE_FEEDBACK_CAPABILITY, probe_request_fields, summarize_motion_probe,
)
from ora.agent.tools.robot import ToolError
from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS, PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
    SCHEMA as DIAGNOSTICS_SCHEMA, validate_planning_diagnostics,
)
from ora.environment.backends.robotwin.planning import (
    ValidatedRoboTwinPlan, planner_public_summary, select_topk_validated_plan,
)
from ora.environment.backends.robotwin.probe_feedback import (
    CONTINUITY_UNKNOWN_REASON, NO_ADMITTED_TRAJECTORY, ik_candidate_feedback,
)
from ora.environment.backends.robotwin.session import RoboTwinLiveError, RoboTwinLiveSession
from ora.environment.controllers.profile_capabilities import PROFILE_VERSION_IDS
from test_robotwin_correctness_repair import vendored_planner_with_fake_solver
from test_robotwin_ik_feedback import _plan

V1_REQUEST = "ora.motion-probe-request.v1"
POSE = {"position_m": [0.0, 0.0, 1.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _no_valid_endpoint():
    """The actual vendored solve: two seeds, both refused by the solver."""
    return _plan([[.1] * 6, [.2] * 6], [False, False])


def _valid_but_unplanned():
    """Two valid endpoints whose every planned joint goal failed trajectory planning."""
    planner = vendored_planner_with_fake_solver(AttributeError, all_candidates_error=True)
    return planner._plan_pose_via_nearest_ik(object(), [0.0] * 6, object())


def _valid_and_planned():
    """Two valid endpoints; the first planned joint goal produced a trajectory."""
    planner = vendored_planner_with_fake_solver(ValueError)
    return planner._plan_pose_via_nearest_ik(object(), [0.0] * 6, object())


def _validated():
    return ValidatedRoboTwinPlan(
        position=np.zeros((2, 6)), velocity=np.zeros((2, 6)),
        planned_link6_pose_xyzw=np.zeros((2, 7)), planned_tcp_pose_xyzw=np.zeros((2, 7)),
        summary={"planning_endpoint_error": {"translation_m": 0.001, "orientation_rad": 0.01}},
    )


def _session(tmp_path, raw_factory, *, version=10, admit=False):
    environment = SimpleNamespace(robot=SimpleNamespace(left_plan_multi_path=lambda targets: {
        "schema": "robotwin.curobo-topk-batch-result.v1",
        "results": [raw_factory() for _ in targets],
    }))
    session = object.__new__(RoboTwinLiveSession)
    session.control_profile = {
        "profile": {"profile_id": PROFILE_VERSION_IDS[version], "plan_validation": {}},
        "sha256": "c" * 64,
    }
    session.output_root = tmp_path
    session.source_manifest = {"robotwin_revision": "test-source"}
    session.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
    session.world_mutation_sequence = 0
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session._require_environment = lambda: environment
    session._uses_motion_transaction_profile = lambda: False
    session._measured_arm_joints = lambda _env, _arm: [0.0] * 6
    session._public_pose_wxyz = lambda _env, _arm: [0, 0, 1, 1, 0, 0, 0]
    session._public_to_planner_wxyz = lambda _arm, pose: pose
    session._arm_joint_limits = lambda _env, _arm: [[-1, 1]] * 6
    session._arm_profile = lambda _arm: {"planner_joint_names": [f"j{i}" for i in range(6)]}
    if admit:
        # The real top-K selection over the vendored candidates; only the
        # FK/workspace validator and the plan-reference writer are faked.
        session._select_motion_probe_candidate = lambda probe, arm, variant, raw: (
            *select_topk_validated_plan(raw, validator=lambda _candidate: _validated())[:2], {})
        session._register_motion_probe_plan = lambda *args: "plan-" + "b" * 32
    return session


def _request(*, feedback, variant_policy="exact", queries=None):
    request = {
        "schema": V1_REQUEST, "observation_id": "obs-1", "observation_sha256": "a" * 64,
        "variant_policy": variant_policy,
        "queries": queries or [{"query_id": "q", "arm": "left", "pose": copy.deepcopy(POSE)}],
    }
    if feedback:
        request.update(schema=PROBE_FEEDBACK_REQUEST_SCHEMA, planner_feedback=PROBE_IK_CANDIDATE_FEEDBACK)
    return request


def _planner(result, variant=0):
    return result["results"][0]["variants"][variant]["planner"]


def _assert_shared_diagnostics(diagnostics):
    validate_planning_diagnostics({**diagnostics, "schema": DIAGNOSTICS_SCHEMA})
    assert "schema" not in diagnostics  # the public record drops it, as on RoboDojo
    assert diagnostics["phase"] == "ik_candidate_validation"
    assert set(diagnostics["candidate_counts"]) == {"returned", "valid_endpoint", "continuous"}


def test_the_v1_request_keeps_the_released_planner_record(tmp_path):
    raw = _no_valid_endpoint()
    session = _session(tmp_path, lambda: copy.deepcopy(raw))
    planner = _planner(session.probe_motion(_request(feedback=False)))
    expected = planner_public_summary(
        {**copy.deepcopy(raw), "ik_validation_artifact": planner["ik_validation_artifact"]},
        include_topk_selection=False, include_constraint_rejection_counts=True,
    )
    assert planner == expected
    assert list(planner["constraint_rejection_counts"]) == [
        "non_finite_solution", "solver_unsuccessful", "joint_limit"]
    assert "retained" in planner["planning_diagnostics"]["candidate_counts"]


def test_no_valid_endpoint_counts_every_returned_seed(tmp_path):
    session = _session(tmp_path, _no_valid_endpoint)
    result = session.probe_motion(_request(feedback=True))
    variant = result["results"][0]["variants"][0]
    assert variant["reachable"] is False
    planner = variant["planner"]
    assert list(planner["constraint_rejection_counts"]) == list(IK_CONSTRAINT_KEYS)
    assert planner["constraint_rejection_counts"] == {
        "non_finite_solution": 0, "solver_unsuccessful": 2, "joint_limit": 0,
        # Filters only a bounded SDK solve applies: not zero, not applied.
        "fk_residual_exceeded": None, "joint_jump": None, "pitch_joint_wrap": None,
    }
    diagnostics = planner["planning_diagnostics"]
    _assert_shared_diagnostics(diagnostics)
    assert diagnostics["outcome"] == "no_valid_endpoint"
    assert diagnostics["candidate_counts"] == {"returned": 2, "valid_endpoint": 0, "continuous": 0}
    assert diagnostics["rejection_counts"] == {"solver_unsuccessful": 2}
    assert diagnostics["details_available"] is True
    assert "does not identify collision" in diagnostics["recovery"][0]
    reference = planner["ik_validation_artifact"]
    data = (tmp_path / reference["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference["sha256"]
    assert len(json.loads(data)["candidate_validation"]) == 2
    assert result["world_state_changed"] is False and result["robot_decisions_consumed"] == 0


def test_valid_endpoints_without_an_admitted_trajectory_leave_continuity_unknown(tmp_path):
    session = _session(tmp_path, _valid_but_unplanned)
    variant = session.probe_motion(_request(feedback=True))["results"][0]["variants"][0]
    assert variant["reachable"] is False and variant["failure"]["code"] == "topk_no_admitted_plan"
    planner = variant["planner"]
    diagnostics = planner["planning_diagnostics"]
    _assert_shared_diagnostics(diagnostics)
    assert diagnostics["outcome"] == NO_ADMITTED_TRAJECTORY
    assert diagnostics["candidate_counts"] == {"returned": 2, "valid_endpoint": 2, "continuous": None}
    assert diagnostics["unknown_reasons"] == [CONTINUITY_UNKNOWN_REASON]
    assert diagnostics["rejection_counts"] == {}
    assert "not proof that the target is unreachable" in diagnostics["recovery"][0]
    assert planner["constraint_rejection_counts"]["joint_jump"] is None
    assert planner["constraint_rejection_counts"]["solver_unsuccessful"] == 0


def test_an_admitted_variant_names_no_recovery(tmp_path):
    session = _session(tmp_path, _valid_and_planned, admit=True)
    query = session.probe_motion(_request(feedback=True))["results"][0]
    assert query["reachable"] is True and query["plan_ref"] == "plan-" + "b" * 32
    planner = query["variants"][0]["planner"]
    diagnostics = planner["planning_diagnostics"]
    _assert_shared_diagnostics(diagnostics)
    assert diagnostics["outcome"] == "admitted" and diagnostics["recovery"] == []
    assert diagnostics["candidate_counts"] == {"returned": 2, "valid_endpoint": 2, "continuous": None}
    assert set(planner["constraint_rejection_counts"].values()) == {0, None}


def test_a_planner_record_without_ik_evidence_invents_nothing():
    feedback = ik_candidate_feedback({"status": "Fail", "planner_status": "IK_FAIL"}, admitted=False)
    assert feedback == {"constraint_rejection_counts": dict.fromkeys(IK_CONSTRAINT_KEYS)}
    unmeasured = _plan([[.1] * 6, [.2] * 6], [True], expose_limits=False)
    diagnostics = ik_candidate_feedback(unmeasured, admitted=False)["planning_diagnostics"]
    # The success mask did not align with the seeds, so no seed is a known valid endpoint.
    assert diagnostics["candidate_counts"] == {"returned": 2, "valid_endpoint": None, "continuous": None}
    assert diagnostics["outcome"] == "unknown"


@pytest.mark.parametrize("damage", [
    {"planner_feedback": "everything"},
    {"schema": V1_REQUEST},
    {"planner_feedback": None},
    {"extra": True},
])
def test_a_malformed_feedback_request_is_refused(tmp_path, damage):
    session = _session(tmp_path, _no_valid_endpoint)
    with pytest.raises(RoboTwinLiveError, match="drifted"):
        session.probe_motion({**_request(feedback=True), **damage})


def test_the_feedback_request_needs_the_ik_constraint_control_profile(tmp_path):
    session = _session(tmp_path, _no_valid_endpoint, version=9)
    with pytest.raises(RoboTwinLiveError, match="control profile v10"):
        session.probe_motion(_request(feedback=True))
    assert session.probe_motion(_request(feedback=False))["results"][0]["reachable"] is False


def test_the_summary_never_sums_an_unknown_count(tmp_path):
    no_valid = _session(tmp_path / "a", _no_valid_endpoint).probe_motion(_request(feedback=True))
    unplanned = _session(tmp_path / "b", _valid_but_unplanned).probe_motion(_request(feedback=True))
    variants = [no_valid["results"][0]["variants"][0], unplanned["results"][0]["variants"][0]]
    mixed = {**no_valid, "results": [{**no_valid["results"][0], "variants": variants}]}
    row = summarize_motion_probe(mixed, ik_candidates=True)[0]
    # 0 known + unknown is unknown, not 0: the continuity key is withheld.
    assert row["ik_candidate_counts"] == {"returned": 4, "valid_endpoint": 2}
    assert row["ik_outcome_counts"] == {"no_valid_endpoint": 1, NO_ADMITTED_TRAJECTORY: 1}
    only_no_valid = summarize_motion_probe(no_valid, ik_candidates=True)[0]
    assert only_no_valid["ik_candidate_counts"] == {"returned": 2, "valid_endpoint": 0, "continuous": 0}
    assert summarize_motion_probe(no_valid)[0].keys().isdisjoint({"ik_candidate_counts", "ik_outcome_counts"})


def test_the_pregrasp_refusal_delivers_the_robotwin_counts(tmp_path):
    """The grasp_execute precommit check, as a surface declaring the capability runs it."""
    surface = SimpleNamespace(supports=lambda capability: capability == IK_CANDIDATE_FEEDBACK_CAPABILITY)
    session = _session(tmp_path, _no_valid_endpoint)
    requests = []

    def probe(arguments):
        request = {**probe_request_fields(surface), "observation_id": "obs-1",
                   "observation_sha256": "a" * 64, **arguments}
        requests.append(request)
        return session.probe_motion(request)

    host = SimpleNamespace(surface=surface, _motion_probe=probe)
    commitment = {"arm": "left", "observation_id": "obs-1", "candidate_id": "grasp-a1",
                  "poses": {"pregrasp": copy.deepcopy(POSE), "approach": copy.deepcopy(POSE)}}
    with pytest.raises(ToolError) as refused:
        grasp_feasibility.check_pregrasp(host, commitment)
    assert requests[0]["schema"] == PROBE_FEEDBACK_REQUEST_SCHEMA
    assert refused.value.code == "grasp_pregrasp_unreachable"
    summary = refused.value.constraints["summary"]
    variants = summary["variant_count"]  # parallel-jaw equivalents of the pregrasp
    assert variants == 6
    assert summary["ik_outcome_counts"] == {"no_valid_endpoint": variants}
    assert summary["ik_candidate_counts"] == {
        "returned": 2 * variants, "valid_endpoint": 0, "continuous": 0}
    assert summary["constraint_rejection_counts"] == {
        "non_finite_solution": 0, "solver_unsuccessful": 2 * variants, "joint_limit": 0}


def test_no_frozen_robotwin_surface_sends_the_feedback_request():
    """Declaration belongs to the v2 surface (A0); the kept v1 surface stays v1."""
    surface = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    assert not surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY)
    assert probe_request_fields(surface) == {"schema": V1_REQUEST}


def test_the_robotwin_v2_default_sends_the_feedback_request_and_states_its_counts():
    """Integration 0059 unit A0: RoboTwin's own grasp_execute wording for the counts."""
    surface = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")
    assert surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY)
    assert probe_request_fields(surface)["schema"] != V1_REQUEST
    help_text = next(item for item in surface.definitions if item["name"] == "grasp_execute")["description"]
    for phrase in ("one CuRobo exact-pose IK solve with 128 seeds", "within the joint limits",
                   "continuous is null because this backend applies no joint-jump filter",
                   "no_admitted_trajectory"):
        assert phrase in help_text, phrase
    assert "joint-jump limit" not in help_text
