"""Rank approach poses by inverse kinematics, never by motion planning.

Backend-neutral on purpose, and structurally so: this module takes an adapter's
batch-reachability callable and a control profile, and it has no import path to
CuRobo, MPlib or any planner. Reachability used to cost one motion plan per
pose, because ``plan_batch_global`` is the only batch entry and it always plans.
Answering the same question from forward kinematics lets a policy discard
unreachable approaches before it spends a world mutation.

It lives beside ``reachability.py`` rather than under a backend because
``controllers`` never imports ``backends``: a solver placed under one backend is
unreachable from the other, which is how the first draft of this work ended up
RoboTwin-only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.alignment import (
    APPROACH_SIDES, MAXIMUM_POSES, REQUEST_FIELDS, REQUEST_SCHEMA, RESULT_SCHEMA,
    AlignmentRequestDrift,
)
from ora.contracts.refusal import EnvironmentRefusal


def _selected_arms(arm: Any, available: Sequence[str]) -> tuple[str, ...]:
    """Resolve a requested arm against the arms this embodiment actually has.

    The vocabulary is the embodiment's, not RoboTwin's. A single-arm backend
    reports its one arm as the only allowed value instead of refusing a
    dual-arm name the policy had no way to know was wrong.
    """

    if len(available) > 1 and arm == "both":
        return tuple(available)
    if arm in available:
        return (str(arm),)
    allowed = list(available) + (["both"] if len(available) > 1 else [])
    raise EnvironmentRefusal(
        "arm is not an arm of this embodiment",
        field="arm", received=arm, allowed=allowed,
    )


def _validated_poses(poses: Any) -> list[list[float]]:
    if (not isinstance(poses, list) or not 1 <= len(poses) <= MAXIMUM_POSES
            or any(not isinstance(pose, list) or len(pose) != 7 for pose in poses)):
        raise EnvironmentRefusal(
            f"target poses must be one to {MAXIMUM_POSES} seven-element world poses",
            field="target_poses_xyzw",
            received=len(poses) if isinstance(poses, list) else poses,
            allowed=[f"1..{MAXIMUM_POSES} poses of [x, y, z, qx, qy, qz, qw]"],
        )
    return [[float(value) for value in pose] for pose in poses]


def plan_alignment(
    request: Mapping[str, Any],
    *,
    adapter: Any,
    control_profile: Mapping[str, Any],
    available_arms: Sequence[str],
) -> dict[str, Any]:
    """Answer a batch of approach poses with per-pose reachability.

    A correctable input is refused as ``EnvironmentRefusal``, not as a backend
    live error: the refusal envelope survives RPC and reaches the policy as a
    retryable ``invalid_arguments`` naming the field, while a live error arrives
    as a non-retryable internal fault (H6-015).

    The read-only IK bridge also carries ``reach_check`` requests under their
    own schema; they are routed here so no backend session method (both are
    pinned embodiment-descriptor sources) has to change to serve them.
    """

    from ora.contracts.reach_check import REQUEST_SCHEMA as REACH_CHECK_SCHEMA

    if request.get("schema") == REACH_CHECK_SCHEMA:
        from .reach_check import check_reach

        return check_reach(request, adapter=adapter, control_profile=control_profile,
                           available_arms=available_arms)
    if set(request) != REQUEST_FIELDS or request.get("schema") != REQUEST_SCHEMA:
        raise AlignmentRequestDrift("alignment request fields drifted")
    side = request.get("approach_side")
    if side not in APPROACH_SIDES:
        raise EnvironmentRefusal(
            "approach side is not a declared direction",
            field="approach_side", received=side, allowed=list(APPROACH_SIDES),
        )
    arms = _selected_arms(request.get("arm"), available_arms)
    poses = _validated_poses(request.get("target_poses_xyzw"))
    # arrival_check, not local_cartesian. local_cartesian is how precisely the
    # local servo controller must settle; arrival_check is what counts as having
    # reached a pose, and it is the same 0.005 m / 0.05 rad the backend planner
    # accepts. Reading the servo tolerance demanded 0.003 m / 0.03 rad and
    # reported reachable: false for every pose whose best solution landed in the
    # 2 mm band the planner would have solved -- a refusal written into the
    # threshold rather than found by the solver.
    tolerances = (control_profile.get("profile") or {}).get("arrival_check")
    if not isinstance(tolerances, Mapping):
        # A missing block must not escape as a KeyError: the dispatch wraps that
        # into a non-retryable internal_tool_error, which tells a policy nothing
        # it can act on. Name the block instead.
        raise AlignmentRequestDrift(
            "control profile declares no arrival_check tolerances for reachability"
        )

    candidates: list[dict[str, Any]] = []
    for selected in arms:
        solutions = adapter.solve_batch_reachability(
            selected, poses,
            position_tolerance_m=float(tolerances["position_tolerance_m"]),
            orientation_tolerance_rad=float(tolerances["orientation_tolerance_rad"]),
        )
        for index, solved in enumerate(solutions):
            # A rejected candidate is kept and named: a policy cannot cite an
            # untried approach it never saw. source_index makes every entry of a
            # batch attributable to the pose that produced it, and (arm,
            # source_index) is the key when more than one arm was asked.
            candidates.append({
                "source_index": index,
                "arm": selected,
                "align_pose": {
                    "position_m": poses[index][:3],
                    "orientation_xyzw": poses[index][3:],
                },
                "reachable": bool(solved["reachable"]),
                "position_error_m": solved["position_error_m"],
                "orientation_error_rad": solved["orientation_error_rad"],
                "iterations": solved["iterations"],
                "rejection_reason": solved.get("reason"),
                "claim_boundary": solved["claim_boundary"],
            })
    return {
        "schema": RESULT_SCHEMA,
        "target_ref": request.get("target_ref"),
        "approach_side": side,
        "requested_arm": request.get("arm"),
        "motion_planner_called": False,
        "candidates": candidates,
    }
