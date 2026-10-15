"""One observation-bound backend check before committing a prepared grasp."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .motion_probe_feedback import IK_CANDIDATE_FEEDBACK_CAPABILITY, summarize_motion_probe
from .robot import ToolError
from .skill_geometry import _lifted, validate_prepared_targets

CAPABILITY = "grasp_precommit_motion_check_v1"

#: A parallel jaw closes on the same two faces whichever way round the tool is
#: rolled, so one compiled orientation is one representative of an equivalence
#: class, not the only way to hold the object. Probing ``exact`` asked whether
#: that single representative solved and refused the grasp when it did not:
#: across two recorded RoboTwin batches and a clean under-30 arm, 32 refusals each probed
#: ``variant_count: 1`` and reported ``admitted_count: 0`` with 3,200
#: ``solver_unsuccessful`` and zero ``joint_limit`` rejections, while the same
#: arm in the same scenes reached commanded poses through move_pose 496 times
#: against 5 failures. The backend already enumerates the class under
#: ``parallel_jaw_equivalents`` and already returns which member it admitted.
VARIANT_POLICY = "parallel_jaw_equivalents"

#: Stages probed together, under the names the commitment actually uses:
#: ``skill_driver`` stores the candidate's contact pose as ``approach`` and
#: derives ``lift`` from it. Adopting a rolled pregrasp while approaching the
#: originally compiled contact would commit a sequence whose two halves
#: disagree, so both are asked in one call and both must be admitted by the
#: same member of the class.
PROBED_STAGES = ("pregrasp", "approach")

#: How far a member's target may sit from the compiled pose and still be that
#: pose. Backends echo the requested position and re-normalise or sign-flip the
#: quaternion, so agreement is float noise, not a tolerance on the grasp; a
#: rolled member differs by at least the smallest slack step any backend
#: enumerates (10 degrees).
SAME_POSE_TOLERANCE = 1e-6


def _pose_vector(value: Any) -> list[float] | None:
    """A backend target as the ``[x, y, z, qx, qy, qz, qw]`` it is published as.

    Every backend that answers ``parallel_jaw_equivalents`` (LIBERO, RoboTwin,
    RoboDojo) publishes ``selected_target_pose_xyzw`` and each admitted
    variant's ``target_pose_xyzw`` as that 7-element list, and ``motion.probe``
    refuses any other shape once a plan handle exists. No backend produces a
    Mapping here, so none is accepted: reading one as "not named" is what kept
    every admitted equivalent from ever being adopted.
    """

    if (not isinstance(value, (list, tuple)) or len(value) != 7
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in value)):
        return None
    return [float(item) for item in value]


def _contract(message: str) -> ToolError:
    return ToolError(message, code="environment_contract", retryable=False)


def _same_pose(left: list[float], right: list[float]) -> bool:
    """Same position and same rotation (``q`` and ``-q`` are one rotation).

    Quaternions are compared after normalising: ``motion.probe`` normalises the
    compiled orientation before the backend sees it, and prepared geometry
    admits a norm within 1e-3 of one.
    """

    norms = math.hypot(*left[3:]) * math.hypot(*right[3:])
    if norms == 0.0:
        return False
    dot = abs(sum(a * b for a, b in zip(left[3:], right[3:]))) / norms
    return (math.dist(left[:3], right[:3]) <= SAME_POSE_TOLERANCE
            and dot >= 1.0 - SAME_POSE_TOLERANCE)


def _compiled_vector(pose: Mapping[str, Any]) -> list[float]:
    return [float(v) for v in [*pose["position_m"], *pose["orientation_xyzw"]]]


def _admitted_members(query: Mapping[str, Any]) -> dict[str, list[float]]:
    """Every member of the class the backend admitted for one stage, by id.

    Empty when the backend reports reachability without naming members. That
    is a weaker answer, not a broken one. A named member without a usable
    target is a broken one: adopting from it would be adopting a guess.
    """

    rows = [row for row in query.get("variants") or []
            if isinstance(row, Mapping) and row.get("reachable") is True]
    named = [(row.get("variant_id"), row.get("target_pose_xyzw")) for row in rows]
    if query.get("selected_variant_id") is not None:
        named.append((query.get("selected_variant_id"),
                      query.get("selected_target_pose_xyzw")))
    members: dict[str, list[float]] = {}
    for variant_id, target in named:
        if variant_id is None and target is None:
            continue
        pose = _pose_vector(target)
        if not isinstance(variant_id, str) or not variant_id or pose is None:
            raise _contract("pregrasp probe named an admitted member without a "
                            "7-element target_pose_xyzw")
        if variant_id in members and not _same_pose(members[variant_id], pose):
            raise _contract("pregrasp probe gave one admitted member two targets")
        members[variant_id] = pose
    return members


def _choose_member(
    stages: list[str], members: Mapping[str, Mapping[str, list[float]]],
    compiled: Mapping[str, list[float]], backend_selected: Mapping[str, Any],
) -> tuple[str | None, str]:
    """The one member every probed stage admitted, and why it was chosen.

    The compiled representative stays whenever every stage admitted it: the
    backend ranks members by its own motion cost, and a cheaper rolled or
    slack member is not a reason to change the grasp the compiler chose.
    Otherwise the backend's own choice for the earliest stage that every other
    stage also admitted is taken, then any other common member in the order the
    backend enumerated them. ``None`` means no member holds for the whole
    sequence.
    """

    first = members[stages[0]]
    for variant_id in first:
        if all(variant_id in members[stage]
               and _same_pose(members[stage][variant_id], compiled[stage])
               for stage in stages):
            return variant_id, "compiled_representative_admitted"
    common = [variant_id for variant_id in first
              if all(variant_id in members[stage] for stage in stages)]
    preferred = [backend_selected[stage] for stage in stages
                 if backend_selected.get(stage) in common]
    ordered = [*preferred, *(item for item in common if item not in preferred)]
    if not ordered:
        return None, "no_common_member"
    return ordered[0], "compiled_representative_not_admitted"


def _adopt(host: Any, commitment: dict[str, Any], stages: list[str],
           members: Mapping[str, Mapping[str, list[float]]],
           compiled: Mapping[str, list[float]], variant_id: str) -> list[str]:
    """Replace each probed stage's pose by the member that was admitted.

    Only the orientation can differ: every member is the compiled position
    under a roll of the jaw. A member at another position is a backend that
    answered a different question, so it is refused rather than adopted.
    """

    poses = dict(commitment["poses"])
    adopted: list[str] = []
    for stage in stages:
        member = members[stage][variant_id]
        if math.dist(member[:3], compiled[stage][:3]) > SAME_POSE_TOLERANCE:
            raise _contract("pregrasp probe admitted a member at another position")
        if _same_pose(member, compiled[stage]):
            continue
        poses[stage] = {"position_m": list(poses[stage]["position_m"]),
                        "orientation_xyzw": list(member[3:])}
        adopted.append(stage)
    if "lift" in poses and "approach" in adopted:
        from . import skill_retention
        poses["lift"] = _lifted(poses["approach"], skill_retention.LIFT_M)
    if adopted and {"pregrasp", "approach"} <= set(poses):
        # The same geometry gate the compiled poses passed: unit quaternions
        # inside the workspace, and a pregrasp that still shares the contact's
        # orientation.
        validate_prepared_targets(
            {stage: poses[stage] for stage in ("pregrasp", "approach")},
            host.workspace_bounds_m,
        )
    commitment["poses"] = poses
    return adopted


def check_pregrasp(host: Any, commitment: dict[str, Any]) -> dict[str, Any]:
    """Check the chosen grasp from the current measured joint state.

    The environment's read-only bridge owns IK/planning/admission; nothing here
    moves the arm. When the backend does not admit the compiled representative
    for every probed stage but does admit one rolled or slack-adjusted member of
    the parallel-jaw equivalence class for all of them, that member is adopted
    into the commitment so execution runs the pose that was actually admitted,
    and ``pose_selection`` names the substitution rather than applying it
    silently. Execution still goes through the Supervisor and revalidates after
    opening.
    """

    stages = [stage for stage in PROBED_STAGES if commitment["poses"].get(stage)]
    result = host._motion_probe({
        "queries": [{"query_id": stage, "arm": commitment["arm"],
                     "pose": dict(commitment["poses"][stage])} for stage in stages],
        "variant_policy": VARIANT_POLICY,
    })
    queries = {str(item.get("query_id")): item for item in result["results"]}
    summaries = {item["query_id"]: item for item in summarize_motion_probe(
        result, ik_candidates=host.surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY),
    )}
    for stage in stages:
        if stage not in queries:
            raise ToolError(f"pregrasp probe omitted its {stage} result",
                            code="environment_contract", retryable=False)
        if type(queries[stage].get("reachable")) is not bool:
            raise ToolError("pregrasp probe omitted its reachability verdict",
                            code="environment_contract", retryable=False)
    evidence = {
        "stage": "pregrasp", "observation_id": commitment["observation_id"],
        "query_count": len(stages), "variant_policy": VARIANT_POLICY,
        "probed_stages": list(stages),
        "summary": summaries[stages[0]],
        **({"stage_summaries": [summaries[stage] for stage in stages]}
           if len(stages) > 1 else {}),
        "claim_boundary": (
            "Current-state planning/admission for the probed stages, with the "
            "selected backend's collision coverage. Does not certify contact, "
            "lift, retention or task success; execution revalidates after "
            "opening."
        ),
    }
    unreachable = [stage for stage in stages if not queries[stage]["reachable"]]
    if unreachable:
        evidence["unreachable_stages"] = unreachable
        raise ToolError(
            "Selected grasp failed the backend check before opening or "
            "committing. Every parallel-jaw equivalent of the compiled "
            "orientation was tried. Choose another candidate or reposition "
            "from fresh evidence; repeating the same candidate is not recovery.",
            code="grasp_pregrasp_unreachable", retryable=True,
            field="candidate_id", received=commitment["candidate_id"],
            retry_requires_changed_arguments=True, constraints=evidence,
        )
    members = {stage: _admitted_members(queries[stage]) for stage in stages}
    backend_selected = {stage: queries[stage].get("selected_variant_id")
                        for stage in stages}
    if not all(members.values()):
        # Reachable, but at least one stage names no member, so no single
        # member can be shown to hold for the whole sequence. The grasp
        # proceeds on its compiled representative, as before the class was
        # probed, and nothing is adopted on a guess.
        evidence["selected_variant_id"] = None
        evidence["pose_selection"] = {
            "decision": "kept_compiled", "reason": "members_not_named",
            "backend_selected_variants": backend_selected,
        }
        return evidence
    compiled = {stage: _compiled_vector(commitment["poses"][stage]) for stage in stages}
    variant_id, reason = _choose_member(stages, members, compiled, backend_selected)
    if variant_id is None:
        # Each stage solved, but under different members of the class. Rolling
        # the approach one way and the close the other is not a grasp, and
        # picking one of them here would be inventing the pairing.
        evidence["stage_variants"] = backend_selected
        evidence["stage_admitted_variants"] = {stage: list(members[stage])
                                               for stage in stages}
        raise ToolError(
            "The probed stages were admitted under different parallel-jaw "
            "equivalents, so no single orientation holds for the whole "
            "sequence. Choose another candidate or reposition from fresh "
            "evidence.",
            code="grasp_pregrasp_unreachable", retryable=True,
            field="candidate_id", received=commitment["candidate_id"],
            retry_requires_changed_arguments=True, constraints=evidence,
        )
    compiled_orientation = list(commitment["poses"][stages[0]]["orientation_xyzw"])
    adopted = _adopt(host, commitment, stages, members, compiled, variant_id)
    evidence["selected_variant_id"] = variant_id
    evidence["pose_selection"] = {
        "decision": "adopted_equivalent" if adopted else "kept_compiled",
        "reason": reason, "variant_id": variant_id,
        "backend_selected_variants": backend_selected,
        **({"compiled_orientation_xyzw": compiled_orientation,
            "adopted_orientation_xyzw": list(
                commitment["poses"][adopted[0]]["orientation_xyzw"])}
           if adopted else {}),
    }
    if adopted:
        evidence["adopted_variant_stages"] = adopted
    return evidence
