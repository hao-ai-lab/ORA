"""Agent-side edge of the internal read-only reach check.

``grasp_plan`` checks every compiled candidate of its pool with the backend's
read-only IK before the display cap, then shows reachable candidates first
without dropping any; ``move_pose`` and the pose primitives reuse the same
check for their reach reference. No ToolSurface declares a ``reach_check``
tool.

The computation belongs to the environment (``controllers/reach_check.py`` and
each backend's ``reach_check_batch``); it travels over the existing read-only
IK bridge (``align_plan_callback``) under its own request schema. This module
owns the workspace pre-check every motion tool applies too, chaining across
skipped poses, the translation of refusals into field-naming tool errors and
the public result shape.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.pose import canonicalize_quaternion
from ora.contracts.reach_check import (
    MAXIMUM_POSES, OUTSIDE_WORKSPACE, REQUEST_SCHEMA, RESULT_FIELDS,
    RESULT_SCHEMA, ROW_FIELDS, START_UNREACHABLE,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.contracts.workspace import workspace_state
from .robot import ToolError

GRASP_ANNOTATION_CAPABILITY = "grasp_plan_ik_reachability_annotation_v1"
#: Public numbers are rounded to 0.01 mm / 0.01 mrad: sixteen rows stay inside
#: the externalised-preview budget, and no decision turns on the sixth decimal.
DECIMALS = 5
NO_BACKEND_CLAIM = (
    "No pose reached the solver (outside the action workspace, or chained onto "
    "one that was); move_pose refuses such targets before any motion."
)
Bridge = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _number(value: Any) -> float | None:
    return None if value is None else round(float(value), DECIMALS)


def _backend_answer(bridge: Bridge, arm: str, queries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        answer = bridge({"schema": REQUEST_SCHEMA, "arm": arm,
                         "queries": [dict(query) for query in queries]})
    except EnvironmentRefusal as error:
        raise ToolError(
            str(error), code=error.code, retryable=error.retryable, field=error.field,
            received=error.received, constraints={"allowed": error.allowed},
        ) from error
    if (not isinstance(answer, Mapping) or set(answer) != RESULT_FIELDS
            or answer.get("schema") != RESULT_SCHEMA or answer.get("arm") != arm
            or answer.get("motion_planner_called") is not False
            or type(answer.get("matches_move_pose_admission")) is not bool
            or not isinstance(answer.get("results"), list)
            or len(answer["results"]) != len(queries)
            or any(not isinstance(row, Mapping) or set(row) != ROW_FIELDS
                   for row in answer["results"])):
        raise ToolError("environment reach-check result drifted",
                        code="environment_contract", retryable=False)
    return answer


def _blank(index: int, reason: str) -> dict[str, Any]:
    return {"index": index, "reachable": False, "reason": reason, "route": None,
            "position_error_m": None, "orientation_error_rad": None,
            "joint_limit_margin_rad": None, "joint_delta_rad": None}


def check_queries(
    arm: str, queries: Sequence[Mapping[str, Any]], *, bridge: Bridge | None,
    workspace_bounds_m: Mapping[str, Any],
) -> dict[str, Any]:
    """One row per query, in order.

    A pose outside the action workspace never reaches the backend, and neither
    does a pose chained onto one that did not (``start_unreachable``); every
    other ``from_index`` is renumbered into the batch actually sent.
    """

    if bridge is None:
        raise ToolError("reach_check requires the environment read-only IK bridge",
                        code="tool_unavailable", retryable=False)
    local: dict[int, int] = {}
    skipped: dict[int, str] = {}
    sent: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        start = query["from_index"]
        if not workspace_state(query["pose_xyzw"][:3], workspace_bounds_m)["inside"]:
            skipped[index] = OUTSIDE_WORKSPACE
        elif start is not None and start in skipped:
            skipped[index] = START_UNREACHABLE
        else:
            local[index] = len(sent)
            sent.append({**query, "from_index": None if start is None else local[start]})
    answer = _backend_answer(bridge, arm, sent) if sent else None
    rows = []
    for index in range(len(queries)):
        if index in skipped:
            rows.append(_blank(index, skipped[index]))
            continue
        row = answer["results"][local[index]]
        rows.append({"index": index, "reachable": bool(row["reachable"]),
                     "reason": row["reason"], "route": row["route"],
                     **{key: _number(row[key]) for key in (
                         "position_error_m", "orientation_error_rad",
                         "joint_limit_margin_rad", "joint_delta_rad")}})
    return {
        "arm": arm,
        "ik_model": None if answer is None else str(answer["ik_model"]),
        "matches_move_pose_admission": (None if answer is None
                                        else bool(answer["matches_move_pose_admission"])),
        "motion_planner_called": False,
        "reachable_indices": [row["index"] for row in rows if row["reachable"]],
        "results": rows,
        "claim_boundary": NO_BACKEND_CLAIM if answer is None else str(answer["claim_boundary"]),
    }


# --------------------------------------------------------------------------
# grasp_plan: check the whole compiled pool, then choose what the cap shows
# --------------------------------------------------------------------------

#: The compiled pool (at most POOL_LIMIT) is checked before the display cap
#: takes its top k; the result schema binder owns both constants.
from ora.agent.composition.grasp_ik_reachability_schema import ORDER, POOL_LIMIT  # noqa: E402
from . import grasp_jaw_symmetry  # noqa: E402
#: Within reachable candidates, an approach within this angle of straight down
#: is preferred before the backend's own order. In a recorded batch, refused
#: candidates were tilted a median ~36 degrees from vertical, executed ones ~10.
VERTICAL_PREFERENCE_DEG = 30.0
GRASP_ANNOTATION_CLAIM = (
    "IK reference at planning time for the compiled pregrasp (free_space from the "
    "measured joints) and approach (contact_approach from that pregrasp solution), "
    "trying the exact orientation and its 180-degree jaw flip; not a guarantee. "
    "Execution runs its own backend check, and any world action stales this."
)
#: Members of the parallel-jaw class tried, in the order the precommit prefers
#: them: the compiled representative first.
MEMBERS = ("exact", "jaw_flip")


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz]


def _rotate(orientation: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = orientation
    rotated = _quaternion_multiply(_quaternion_multiply([x, y, z, w], [*vector, 0.0]), [-x, -y, -z, w])
    return rotated[:3]


def _member(orientation: Sequence[float], member: str, axis: Sequence[float]) -> list[float]:
    """The compiled orientation, or it turned 180 degrees about the local approach axis."""
    if member == "exact":
        return [float(item) for item in orientation]
    norm = math.sqrt(sum(float(item) ** 2 for item in axis))
    half = [float(item) / norm for item in axis] + [0.0]  # sin(pi/2) = 1, cos(pi/2) = 0
    return canonicalize_quaternion(_quaternion_multiply(orientation, half), "jaw_flip")


def approach_tilt_deg(candidate: Mapping[str, Any], axis: Sequence[float]) -> float:
    """Angle between the candidate's approach direction and straight down, in degrees."""
    direction = _rotate(candidate["pregrasp_pose"]["orientation_xyzw"], axis)
    norm = math.sqrt(sum(item * item for item in direction)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -direction[2] / norm))))


def _stage(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if row is None else {"reachable": bool(row["reachable"]), "reason": row["reason"]}


def _pairs(candidates: Sequence[Mapping[str, Any]], member: str, axis: Sequence[float]) -> list[dict[str, Any]]:
    """Pregrasp (free_space, measured start) then approach chained onto it."""
    queries = []
    for candidate in candidates:
        for stage, key, interaction in (("pregrasp", "pregrasp_pose", "free_space"),
                                        ("approach", "contact_pose", "contact_approach")):
            pose = candidate[key]
            queries.append({
                "pose_xyzw": [*map(float, pose["position_m"]),
                              *_member(pose["orientation_xyzw"], member, axis)],
                "interaction": interaction,
                "from_index": None if stage == "pregrasp" else len(queries) - 1,
            })
    return queries


def _check_member(arm: str, candidates: Sequence[Mapping[str, Any]], member: str, *,
                  bridge: Bridge, workspace_bounds_m: Mapping[str, Any],
                  axis: Sequence[float]) -> tuple[list[tuple[dict, dict]], str | None, bool | None]:
    """Stage rows per candidate for one member, in bridge-sized chunks."""
    per_call = MAXIMUM_POSES // 2
    rows: list[tuple[dict, dict]] = []
    model: str | None = None
    parity: bool | None = None
    for start in range(0, len(candidates), per_call):
        chunk = candidates[start:start + per_call]
        answer = check_queries(arm, _pairs(chunk, member, axis), bridge=bridge,
                               workspace_bounds_m=workspace_bounds_m)
        model = answer["ik_model"] or model
        parity = answer["matches_move_pose_admission"] if parity is None else parity
        results = answer["results"]
        rows.extend((results[2 * i], results[2 * i + 1]) for i in range(len(chunk)))
    return rows, model, parity


def _released_verdicts(arm: str, candidates: Sequence[Mapping[str, Any]], *, bridge: Bridge,
                       workspace_bounds_m: Mapping[str, Any], axis: Sequence[float]) -> tuple:
    """Exact for the whole pool, then the jaw flip only where exact was not admitted."""
    verdicts: dict[str, dict[str, Any]] = {}
    model: str | None = None
    parity: bool | None = None
    solved = 0
    remaining = list(candidates)
    for member in MEMBERS:
        if not remaining:
            break
        rows, member_model, member_parity = _check_member(
            arm, remaining, member, bridge=bridge, workspace_bounds_m=workspace_bounds_m, axis=axis)
        model = model or member_model
        parity = member_parity if parity is None else parity
        solved += 2 * len(remaining)
        still = []
        for candidate, (pregrasp, approach) in zip(remaining, rows):
            reachable = bool(pregrasp["reachable"] and approach["reachable"])
            if reachable or candidate["candidate_id"] not in verdicts:
                verdicts[candidate["candidate_id"]] = {
                    "reachable": reachable, "member": member if reachable else None,
                    "pregrasp": _stage(pregrasp), "approach": _stage(approach),
                }
            if not reachable:
                still.append(candidate)
        remaining = still
    return verdicts, model, parity, solved


def _symmetric_verdicts(arm: str, candidates: Sequence[Mapping[str, Any]], *, bridge: Bridge,
                        workspace_bounds_m: Mapping[str, Any], axis: Sequence[float],
                        rule: Mapping[str, Any]) -> tuple:
    """Both members for every candidate; ``grasp_jaw_symmetry.choose`` names the one to publish.

    The stages reported are the chosen member's, or the compiled orientation's
    when neither member solved -- the released meaning of ``pregrasp`` and
    ``approach``.
    """
    rows: dict[str, list[tuple[dict, dict]]] = {}
    model: str | None = None
    parity: bool | None = None
    for member in MEMBERS:
        rows[member], member_model, member_parity = _check_member(
            arm, candidates, member, bridge=bridge, workspace_bounds_m=workspace_bounds_m, axis=axis)
        model = model or member_model
        parity = member_parity if parity is None else parity
    verdicts: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        pairs = {member: rows[member][index] for member in MEMBERS}
        choice = grasp_jaw_symmetry.choose(pairs, rule)
        pregrasp, approach = pairs[choice["member"] or "exact"]
        verdicts[candidate["candidate_id"]] = {
            "reachable": choice["member"] is not None, "member": choice["member"],
            "pregrasp": _stage(pregrasp), "approach": _stage(approach),
            "jaw_symmetry": choice["jaw_symmetry"], "joint_headroom": choice["joint_headroom"],
        }
    return verdicts, model, parity, 2 * len(MEMBERS) * len(candidates)


def _publish_jaw_flip(candidate: dict[str, Any], axis: Sequence[float]) -> None:
    """Turn the published poses to the jaw flip: exactly the quaternions the IK was asked."""
    for key in ("pregrasp_pose", "contact_pose"):
        pose = candidate[key]
        candidate[key] = {**pose, "orientation_xyzw": _member(pose["orientation_xyzw"], "jaw_flip", axis)}


def select_grasp_candidates(
    compiled: Sequence[Mapping[str, Any]], *, arm: str, bridge: Bridge | None,
    workspace_bounds_m: Mapping[str, Any], approach_axis_local: Sequence[float],
    jaw_symmetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate every pool candidate; order reachable, then near-vertical, then backend.

    Returns ``ordered`` (all candidates, annotated, never dropped), the public
    ``summary`` and private ``evidence``. Released: the exact member is checked
    for the whole pool first; the jaw flip only for candidates the exact member
    did not admit, which is what the precommit adopts. With a ``jaw_symmetry``
    rule (``grasp_jaw_symmetry.rule_for``) both members are checked for every
    candidate and the published poses carry the member that rule chooses; a
    roll about the approach axis changes neither reachability nor the tilt, so
    the order is the same. Any failure leaves the backend order and marks every
    candidate unknown: an IK reference never blocks a proposal.
    """

    candidates = [dict(item) for item in compiled]
    started = time.monotonic()
    tilt = {item["candidate_id"]: approach_tilt_deg(item, approach_axis_local) for item in candidates}
    verdicts: dict[str, dict[str, Any]] = {}
    model: str | None = None
    parity: bool | None = None
    failure: str | None = None
    solved = 0
    passes = {"bridge": bridge, "workspace_bounds_m": workspace_bounds_m, "axis": approach_axis_local}
    try:
        verdicts, model, parity, solved = (
            _released_verdicts(arm, candidates, **passes) if jaw_symmetry is None
            else _symmetric_verdicts(arm, candidates, rule=jaw_symmetry, **passes))
    except ToolError as error:
        failure, verdicts = error.code, {}
    except Exception as error:  # an IK reference must never block a proposal
        failure, verdicts = type(error).__name__, {}
    annotated = bool(verdicts) and failure is None
    flipped: list[str] = []
    for candidate in candidates:
        verdict = verdicts.get(candidate["candidate_id"]) if annotated else None
        blank = {"reachable": None, "member": None, "pregrasp": None, "approach": None,
                 **(grasp_jaw_symmetry.unavailable() if jaw_symmetry is not None else {})}
        candidate["ik_reachability"] = {
            **(verdict or blank),
            "approach_tilt_from_vertical_deg": round(tilt[candidate["candidate_id"]], 1),
        }
        if verdict is not None and verdict.get("jaw_symmetry", {}).get("variant") == "jaw_flip":
            _publish_jaw_flip(candidate, approach_axis_local)
            flipped.append(candidate["candidate_id"])

    def tier(item: Mapping[str, Any]) -> int:
        verdict = item["ik_reachability"]
        if verdict["reachable"] is not True:
            return 2
        return 0 if verdict["approach_tilt_from_vertical_deg"] <= VERTICAL_PREFERENCE_DEG else 1

    ordered = sorted(candidates, key=tier) if annotated else candidates
    reachable = sum(1 for item in candidates if item["ik_reachability"]["reachable"] is True)
    summary = {
        "status": "annotated" if annotated else ("no_candidate" if not candidates else "unavailable"),
        "ik_model": model if annotated else None,
        "matches_move_pose_admission": parity if annotated else None,
        "order": ORDER,
        "checked_candidate_count": len(candidates) if annotated else 0,
        "reachable_candidate_count": reachable,
        "claim_boundary": GRASP_ANNOTATION_CLAIM,
    }
    evidence = {
        "type": "grasp_plan_ik_reachability", "arm": arm, "status": summary["status"],
        "failure": failure, "pool_size": len(candidates), "queries_sent_upper_bound": solved,
        "elapsed_s": round(time.monotonic() - started, 4),
        "pool_order": [item["candidate_id"] for item in ordered],
    }
    if jaw_symmetry is not None:
        summary["jaw_symmetry"] = grasp_jaw_symmetry.summary()
        evidence["jaw_symmetry"] = {
            "measured_joints": jaw_symmetry.get("measured_joints"),
            "isolation_bound_rad": jaw_symmetry.get("isolation_bound_rad"),
            "published_jaw_flip": flipped,
        }
    return {"ordered": ordered, "summary": summary, "evidence": evidence}


# --------------------------------------------------------------------------
# Runtime edges (kept here so the oversized runtime module only calls them)
# --------------------------------------------------------------------------

def grasp_candidate_selection(host: Any, arm: Any, observation: Any = None) -> dict[str, Any] | None:
    """The pool selection a declaring surface passes to the grasp query, else None.

    ``observation`` is the one the grasp query is bound to; a surface declaring
    ``grasp_plan_jaw_symmetry_headroom_v1`` reads its measured arm joints, and
    only that surface's query identity gains the ``jaw_symmetry`` rule.
    """

    if not host.surface.supports(GRASP_ANNOTATION_CAPABILITY):
        return None
    if arm is None and host.surface.supports("single_arm_flat_targets_v1"):
        # A single-arm flat surface (LIBERO) takes no arm argument; its one
        # arm is the right arm, as motion.probe and the flat motion tools resolve it.
        arm = "right"
    from .alignment import frame_local_axes

    convention = (host.surface.view_config or {}).get("controlled_frame_convention")
    axis = frame_local_axes(str(convention))["approach"]
    identity = {"schema": "ora.grasp-ik-candidate-selection.v1",
                "pool_limit": POOL_LIMIT, "order": ORDER, "members": list(MEMBERS),
                "vertical_preference_deg": VERTICAL_PREFERENCE_DEG, "arm": str(arm)}
    rule = grasp_jaw_symmetry.rule_for(host.surface, str(arm), observation)
    if rule is not None:  # released query refs keep their hashes
        identity["jaw_symmetry"] = dict(rule["identity"])
    return {
        "identity": identity,
        "select": lambda compiled: select_grasp_candidates(
            compiled, arm=str(arm), bridge=getattr(host, "align_plan_callback", None),
            workspace_bounds_m=host.workspace_bounds_m, approach_axis_local=axis, jaw_symmetry=rule),
    }


def publish_grasp_selection(host: Any, result: dict[str, Any], manifest: Mapping[str, Any]) -> None:
    """Copy the selection summary into a declaring surface's grasp_plan result."""

    if host.surface.supports(GRASP_ANNOTATION_CAPABILITY):
        result["ik_reachability"] = dict(manifest["candidate_selection"]["summary"])
