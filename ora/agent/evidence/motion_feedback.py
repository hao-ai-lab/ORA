"""Policy-safe measured motion feedback and evidence-based cause estimates."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping


def public_motion_feedback(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {str(arm): {key: copy.deepcopy(record[key]) for key in (
        "contact_estimated", "contact_force_threshold_n", "force_stop_threshold_n",
        "external_force_magnitude_n", "external_force_world_n",
        "external_force_direction_world", "force_frame", "force_direction_convention",
        "force_direction_status", "force_threshold_exceeded",
        "blocked_evidence", "frozen_steps", "progress", "stall_shape",
        "current_center_distance_m", "execution_minimum_center_distance_m",
        "planning_minimum_center_distance_m", "escape_allowed",
    ) if key in record} for arm, record in value.items() if isinstance(record, Mapping)}
    for arm, record in value.items():
        if isinstance(record, Mapping):
            clearance = public_inter_arm_clearance(record.get("inter_arm_clearance"))
            if clearance:
                result[str(arm)]["inter_arm_clearance"] = clearance
    return result


def public_inter_arm_clearance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: copy.deepcopy(value[key]) for key in (
        "allowed", "escape_allowed", "current_center_distance_m", "center_distance_m",
        "required_center_distance_m", "planning_center_distance_m",
        "execution_center_distance_m", "escape_endpoint_must_increase",
    ) if key in value}

def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in value):
        return None
    return [float(item) for item in value]


def _norm(value: list[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def public_contact_motion_cause(evidence: Mapping[str, Any]) -> str:
    """Infer only from explicitly framed force or the measured blocked test.

    A force sensor includes inertia and gravity. This is a public estimate,
    never privileged contact or object-state truth. Missing frame/sign or
    displacement evidence cannot establish a directional cause.
    """
    if evidence.get("blocked_evidence") == "measured_joint_stall_at_unchanged_target":
        return "blocked_by_contact"
    if evidence.get("contact_estimated") is not True:
        return "unknown"
    realized = _vector3(evidence.get("realized_translation_world_m"))
    if realized is None:
        return "unknown"
    direction = _vector3(evidence.get("external_force_direction_world"))
    magnitude = evidence.get("external_force_magnitude_n")
    threshold = evidence.get("contact_force_threshold_n")
    framed = (
        evidence.get("force_frame") == "world"
        and evidence.get("force_direction_convention")
        == "environment_on_eef_reaction_estimate"
    )
    if (framed and direction is not None and _norm(direction) > 0
            and _norm(realized) > 0
            and isinstance(magnitude, (int, float)) and not isinstance(magnitude, bool)
            and isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(magnitude) and math.isfinite(threshold)
            and threshold >= 0 and magnitude > threshold
            and _dot(realized, direction) / (_norm(realized) * _norm(direction)) > .5):
        return "pushed_by_contact"
    requested = _vector3(evidence.get("requested_translation_world_m"))
    if _norm(realized) < .003 or (
        requested is not None and _dot(realized, requested) < 0
    ):
        return "blocked_by_contact"
    return "unknown"


#: Per-arm blocks in which a RoboDojo executor names how each arm's motion ended
#: (``environment/backends/robodojo/motion_stall.py``): a recovery's overall
#: outcome first, else the arm's latest driven segment.
_ARM_PROGRESS_BLOCKS = ("recovery_progress", "motion_progress")


def stall_evidenced_code(record: Mapping[str, Any], code: str | None) -> str | None:
    """Keep ``controller_timeout`` only while the receipt says an arm was still moving.

    A timeout tells the policy the window ran out on a moving arm, so calling
    again continues the motion. When the receipt's own per-arm progress says
    every arm that did not arrive stood still, the stop was a stall; naming it
    a timeout invites repeating a blocked motion. Receipts without progress
    blocks (released RoboDojo moves, other backends) keep their code.
    """
    arms = record.get("arms")
    if code != "controller_timeout" or not isinstance(arms, Mapping):
        return code
    outcomes = []
    for effect in arms.values():
        if isinstance(effect, Mapping) and effect.get("targeted"):
            block = next((effect[key] for key in _ARM_PROGRESS_BLOCKS if isinstance(effect.get(key), Mapping)), {})
            outcomes.append(block.get("outcome"))
    unarrived = [outcome for outcome in outcomes if outcome != "arrived"]
    return "controller_stalled" if unarrived and all(item == "stalled" for item in unarrived) else code


def _public_command_drift(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("command_drift")
    if not isinstance(value, Mapping):
        return {}
    return {str(arm): {key: copy.deepcopy(state[key]) for key in (
        "drift_from_last_commanded_m", "start_drift_from_last_commanded_m",
        "drift_accumulating", "drift_frame",
    ) if key in state} for arm, state in value.items() if isinstance(state, Mapping)}


_PUBLIC_CONTACT_DETAIL_FIELDS = frozenset({
    "contact_estimated", "contact_force_threshold_n", "external_force_magnitude_n",
    "external_force_world_n", "external_force_direction_world", "force_frame",
    "force_direction_convention", "force_direction_status", "force_threshold_exceeded",
    "realized_translation_world_m", "requested_translation_world_m",
})


def is_public_contact_feedback_field(parent_path: str, key: str, value: Any) -> bool:
    """Recognize narrow existing sensor-feedback fields, never arbitrary contact data."""
    if key == "contact_motion":
        return (parent_path.endswith(".last_action_effect.detail")
                and isinstance(value, Mapping) and bool(value)
                and set(value) <= _PUBLIC_CONTACT_DETAIL_FIELDS)
    feedback_parent = (parent_path.endswith((".feedback", ".motion_feedback",
                                            ".last_action_effect.detail.contact_motion"))
                       or parent_path.endswith((".motion_feedback.left", ".motion_feedback.right")))
    if not feedback_parent:
        return False
    if key == "contact_estimated":
        return value is None or isinstance(value, bool)
    if key == "contact_force_threshold_n":
        return (value is None or type(value) in (int, float)
                and math.isfinite(value) and value >= 0)
    return False
