"""Bounded, backend-neutral public explanations of plan admission outcomes.

Never copy validator messages or nested details: they may contain private
geometry, paths or actor identities. A candidate's planner success is distinct
from admission, execution, endpoint arrival and task success.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Mapping


_REASONS = {
    "planner_failed": (
        "The planner did not produce a successful trajectory for this candidate.",
        "Inspect the current pose and consider a different approach or intermediate target; this does not prove the goal is unreachable.",
    ),
    "workspace_path": (
        "The generated TCP path leaves the allowed workspace; this is a path rejection, not an endpoint reachability verdict.",
        "Use the current pose and advertised workspace to choose a different approach or intermediate target before replanning.",
    ),
    "joint_limit": (
        "The generated path exceeds a robot joint limit.",
        "Consider a different approach orientation or intermediate target and replan.",
    ),
    "inter_arm_collision": (
        "The proposed arm paths failed inter-arm clearance admission.",
        "Inspect both arms and replan their approaches; this is predicted clearance, not evidence of an actual collision.",
    ),
    "payload_collision": (
        "The proposed path failed the configured payload clearance check.",
        "Reconsider the approach using fresh observations; predicted clearance does not establish actual contact.",
    ),
    "planning_endpoint_error": (
        "The planned endpoint failed the configured target accuracy check.",
        "Inspect the requested pose and replan; no endpoint arrival is established.",
    ),
    "start_joint_error": (
        "The path does not start at the measured robot joint state.",
        "Obtain fresh state and replan; do not execute the mismatched path.",
    ),
    "public_tcp_error": (
        "The planned start pose disagrees with the measured TCP pose.",
        "Use fresh state and check the commanded frame before replanning.",
    ),
    "planned_tcp_step": (
        "The generated path contains a TCP step larger than the allowed bound.",
        "Replan the approach; increasing execution time does not repair the rejected path.",
    ),
    "unknown_rejection": (
        "This candidate failed plan admission; no recognized public reason is available.",
        "Inspect fresh state and the recorded failure stage; do not infer a specific collision or unreachable goal.",
    ),
}
_MAX_OUTCOMES = 8


def _index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _clearance(value: Any) -> dict[str, Any]:
    """Only robot-link admission measurements; never scene collision details."""
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key in (
        "center_distance_m", "required_center_distance_m", "current_center_distance_m",
        "planning_center_distance_m", "execution_center_distance_m",
    ):
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number) and number >= 0:
            result[key] = number
    if _index(value.get("stage_index")):
        result["stage_index"] = value["stage_index"]
    pair = value.get("link_pair")
    if isinstance(pair, list) and len(pair) == 2 and all(_index(i) for i in pair):
        result["link_pair"] = list(pair)
    if type(value.get("escape_allowed")) is bool:
        result["escape_allowed"] = value["escape_allowed"]
    if result:
        result["distance_semantics"] = "robot_link_sphere_centers_not_end_effector_gap_or_measured_contact"
    identity = value.get("link_identity")
    if isinstance(identity, Mapping) and identity.get("frame") == "world":
        public = {"frame": "world", "geometry": "sphere_at_named_link_origin"}
        for arm in ("left", "right"):
            link = identity.get(arm)
            if not isinstance(link, Mapping):
                continue
            row = {}
            if _index(link.get("index")):
                row["index"] = link["index"]
            name = link.get("name")
            if isinstance(name, str) and 0 < len(name) <= 128:
                row["name"] = name
            center = link.get("center_m")
            if isinstance(center, list) and len(center) == 3 and all(
                type(x) in (int, float) and math.isfinite(x) for x in center
            ):
                row["center_m"] = list(center)
            radius = link.get("radius_m")
            if type(radius) in (int, float) and math.isfinite(radius) and radius >= 0:
                row["radius_m"] = radius
            public[arm] = row
        result["link_identity"] = public
    return result


def _outcome(value: Mapping[str, Any]) -> dict[str, Any]:
    rejection = value.get("rejection")
    rejection = rejection if isinstance(rejection, Mapping) else value
    planning_failed = value.get("admission") == "planner_failed"
    code = "planner_failed" if planning_failed else rejection.get("code")
    if not isinstance(code, str) or code not in _REASONS:
        code = "unknown_rejection"
    explanation, recovery = _REASONS[code]
    result: dict[str, Any] = {
        "stage": "planner" if planning_failed else "plan_validation",
        "code": code, "explanation": explanation, "recovery": recovery,
    }
    if code == "inter_arm_collision":
        clearance = _clearance(rejection.get("details"))
        if clearance:
            result["clearance"] = clearance
    if _index(value.get("candidate_index")):
        result["candidate_index"] = value["candidate_index"]
    indices = value.get("candidate_indices")
    if isinstance(indices, Mapping):
        public_indices = {arm: indices[arm] for arm in ("left", "right")
                          if arm in indices and _index(indices[arm])}
        if public_indices:
            result["candidate_indices"] = public_indices
    return result


def public_plan_validation(value: Any) -> dict[str, Any] | None:
    """Project known numeric checks and explain every *reported* failure class.

    Counts describe source records, not an unobserved exhaustive search. Pair
    candidates and single-arm candidates share this boundary. Outcomes are
    bounded while counts preserve failure classes beyond the displayed prefix.
    """
    if not isinstance(value, Mapping):
        return None
    result = {key: copy.deepcopy(value[key]) for key in (
        "status", "code", "planning_endpoint_error", "public_tcp_start_error",
        "maximum_start_joint_error_rad", "maximum_planned_tcp_step_m",
    ) if key in value}
    details = value.get("details")
    code = value.get("code")
    if value.get("status") == "rejected" and isinstance(code, str) and code in _REASONS:
        result["explanation"], result["recovery"] = _REASONS[code]
        if code == "inter_arm_collision":
            clearance = _clearance(details)
            if clearance:
                result["clearance"] = clearance
    candidates = details.get("candidates") if isinstance(details, Mapping) else None
    if value.get("status") == "rejected" and isinstance(candidates, list):
        outcomes = [_outcome(item) for item in candidates if isinstance(item, Mapping)]
        counts = Counter((item["stage"], item["code"]) for item in outcomes)
        result["candidate_summary"] = {
            "reported_count": len(candidates),
            "uninterpretable_count": len(candidates) - len(outcomes),
            "omitted_outcome_count": max(0, len(outcomes) - _MAX_OUTCOMES),
            "reason_counts": [
                {"stage": stage, "code": code, "count": count,
                 "explanation": _REASONS[code][0], "recovery": _REASONS[code][1]}
                for (stage, code), count in sorted(counts.items())
            ],
            "outcomes": outcomes[:_MAX_OUTCOMES],
            "claim": "reported_candidate_failures_not_goal_unreachability_or_execution",
        }
    return result
