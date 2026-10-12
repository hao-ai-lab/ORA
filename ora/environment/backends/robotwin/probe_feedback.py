"""RoboTwin's answer to the v2 motion-probe request (``ik_candidate_counts_v1``).

A surface declaring ``motion_probe_ik_candidate_feedback_v1`` sends
``ora.motion-probe-request.v2`` with ``planner_feedback:
ik_candidate_counts_v1`` (``agent/tools/motion_probe_feedback.py``). Each
variant's planner record then carries the shared contract RoboDojo already
answers (``backends/robodojo/sparse_motion.py``): ``constraint_rejection_counts``
over ``IK_CONSTRAINT_KEYS``, the validated ``planning_diagnostics`` and the
``ik_validation_artifact`` reference. v1 requests keep the released record.

The candidate set and unit are RoboDojo's: every solution one bounded
exact-pose IK solve returned for this variant, counted. Here the solve is the
vendored planner's single CuRobo ``IKSolver.solve_single`` (128 seeds, the
measured joints among them; ``vendored/planner.py`` ``_plan_pose_via_nearest_ik``).

* ``returned`` -- ``ik.returned_solution_count``.
* ``valid_endpoint`` -- ``ik.valid_solution_count``: finite, solver success
  (CuRobo's success already includes its 5 mm / 0.05 rad pose tolerance and
  self/world collision feasibility, which RoboDojo checks as SDK feasibility
  plus an FK residual) and inside the joint limits. Unknown when the success
  mask did not align with the seeds, because the solver filter then never ran.
* ``continuous`` -- valid endpoints inside the joint-jump bound. RoboTwin
  applies no joint-jump filter to IK endpoints: it plans a trajectory to at
  most 16 deduplicated joint goals and admits a variant by that trajectory.
  The count is therefore known only when no endpoint is valid (0); otherwise
  it is ``None`` with ``CONTINUITY_UNKNOWN_REASON``, never a different count
  under the shared key.

``constraint_rejection_counts`` keeps RoboTwin's three measured filters and
reports the three RoboDojo-only filters (``fk_residual_exceeded``,
``joint_jump``, ``pitch_joint_wrap``) as ``None``: not applied, not zero.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS,
    PROBE_FEEDBACK_REQUEST_SCHEMA,
    PROBE_IK_CANDIDATE_FEEDBACK,
    SCHEMA,
    public_planning_diagnostics,
    validate_planning_diagnostics,
)

PROBE_REQUEST_SCHEMA = "ora.motion-probe-request.v1"
PROBE_REQUEST_FIELDS = frozenset({
    "schema", "observation_id", "observation_sha256", "queries", "variant_policy",
})
#: The filters this backend's IK selection measures, in the shared spelling.
MEASURED_FILTERS = ("non_finite_solution", "solver_unsuccessful", "joint_limit")
NO_ADMITTED_TRAJECTORY = "no_admitted_trajectory"
CONTINUITY_UNKNOWN_REASON = "joint_jump_filter_not_applied"
VALID_ENDPOINT_UNKNOWN_REASON = "solver_success_filter_unmeasured"
EVIDENCE_SCOPE = (
    "One bounded CuRobo exact-pose IK solve from the measured joints and its endpoint filters "
    "(finite, solver success, joint limits); continuity was decided only by trajectory planning "
    "on the deduplicated joint goals; no path, payload or task-success proof."
)
#: The released v1 sentence for the same outcome (``planning.ik_planning_diagnostics``).
NO_VALID_ENDPOINT_RECOVERY = (
    "No candidate survived this bounded exact-pose IK solve. "
    "Solver failure does not identify collision or prove the target unreachable. "
    "Inspect the reported filters and consider a different exact pose or intermediate pose."
)
NO_ADMITTED_TRAJECTORY_RECOVERY = (
    "Valid IK endpoints exist, but no planned joint goal produced a trajectory that passed "
    "admission from the current joints, so nothing moved. Consider a different exact pose or "
    "an intermediate pose; this is not proof that the target is unreachable."
)
UNKNOWN_RECOVERY = (
    "Candidate filter details were not recorded; the cause is unknown. "
    "Consider a different exact pose or intermediate pose; failure does not prove unreachability."
)


def probe_request_feedback(request: Any) -> bool:
    """Whether a well-formed probe request asks for IK candidate feedback.

    v1 is the released request; v2 differs only by naming the feedback it
    asks for. Anything else is drift and raises ``ValueError``.
    """

    if not isinstance(request, Mapping):
        raise ValueError("motion probe request fields drifted")
    if request.get("schema") == PROBE_REQUEST_SCHEMA and set(request) == PROBE_REQUEST_FIELDS:
        return False
    if (request.get("schema") == PROBE_FEEDBACK_REQUEST_SCHEMA
            and set(request) == PROBE_REQUEST_FIELDS | {"planner_feedback"}
            and request.get("planner_feedback") == PROBE_IK_CANDIDATE_FEEDBACK):
        return True
    raise ValueError("motion probe request fields drifted")


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _ik_summary(result: Any) -> Mapping[str, Any] | None:
    """The solve's IK summary, top level or kept on the selected top-K record."""

    if not isinstance(result, Mapping):
        return None
    source = result.get("ik")
    topk = result.get("topk_selection")
    if not isinstance(source, Mapping) and isinstance(topk, Mapping):
        source = topk.get("ik")
    return source if isinstance(source, Mapping) else None


def _outcome(valid: int | None, *, admitted: bool) -> str:
    if admitted:
        return "admitted"
    if valid == 0:
        return "no_valid_endpoint"
    return "unknown" if valid is None else NO_ADMITTED_TRAJECTORY


def _diagnostics(ik: Mapping[str, Any], filters: Mapping[str, int | None], *,
                 admitted: bool) -> dict[str, Any]:
    rows = ik.get("candidate_validation")
    returned = _count(ik.get("returned_solution_count"))
    # Without the solver filter no seed is a known valid endpoint.
    valid = _count(ik.get("valid_solution_count")) if filters["solver_unsuccessful"] is not None else None
    outcome = _outcome(valid, admitted=admitted)
    recovery = {
        "admitted": [], "no_valid_endpoint": [NO_VALID_ENDPOINT_RECOVERY],
        NO_ADMITTED_TRAJECTORY: [NO_ADMITTED_TRAJECTORY_RECOVERY], "unknown": [UNKNOWN_RECOVERY],
    }[outcome]
    unknown = ([] if valid is not None else [VALID_ENDPOINT_UNKNOWN_REASON]) + (
        [CONTINUITY_UNKNOWN_REASON] if valid != 0 else [])
    rejections = Counter(
        str(row["rejection"]) for row in rows if isinstance(row, Mapping) and row.get("rejection")
    ) if isinstance(rows, list) else Counter()
    return public_planning_diagnostics(validate_planning_diagnostics({
        "schema": SCHEMA, "phase": "ik_candidate_validation", "outcome": outcome,
        "candidate_counts": {"returned": returned, "valid_endpoint": valid,
                             "continuous": 0 if valid == 0 else None},
        "rejection_counts": dict(rejections),
        "details_available": isinstance(rows, list) and returned is not None and len(rows) == returned,
        "recovery": recovery, "evidence_scope": EVIDENCE_SCOPE,
        **({"unknown_reasons": unknown} if unknown else {}),
    }))


def ik_candidate_feedback(result: Any, *, admitted: bool) -> dict[str, Any]:
    """The v2 planner fields for one variant's raw planner result.

    ``admitted`` is whether the probe admitted a plan for this variant. A
    result that kept no IK summary gets unknown filter counts and no
    diagnostics; nothing is inferred from a final status.
    """

    ik = _ik_summary(result)
    raw = ik.get("constraint_rejection_counts") if ik is not None else None
    raw = raw if isinstance(raw, Mapping) else {}
    filters = {key: (_count(raw.get(key)) if key in MEASURED_FILTERS else None)
               for key in IK_CONSTRAINT_KEYS}
    feedback: dict[str, Any] = {"constraint_rejection_counts": filters}
    if ik is not None:
        feedback["planning_diagnostics"] = _diagnostics(ik, filters, admitted=admitted)
    return feedback


def with_ik_candidate_feedback(record: Mapping[str, Any], result: Any, *,
                               admitted: bool) -> dict[str, Any]:
    """A new planner record carrying the v2 fields in place of the v1 ones."""

    updated = {key: value for key, value in record.items() if key != "planning_diagnostics"}
    return {**updated, **ik_candidate_feedback(result, admitted=admitted)}


__all__ = [
    "CONTINUITY_UNKNOWN_REASON", "EVIDENCE_SCOPE", "MEASURED_FILTERS", "NO_ADMITTED_TRAJECTORY",
    "NO_ADMITTED_TRAJECTORY_RECOVERY", "NO_VALID_ENDPOINT_RECOVERY", "UNKNOWN_RECOVERY",
    "VALID_ENDPOINT_UNKNOWN_REASON",
    "ik_candidate_feedback", "probe_request_feedback", "with_ik_candidate_feedback",
]
