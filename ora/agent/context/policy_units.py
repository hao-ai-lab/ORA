"""Degrees for every angle a policy reads (``policy_angles_degrees_v1``).

Receipts, facts and tool results report angles in radians: an acceptance
residual of 2.09 rad against a 0.1 rad tolerance, a ``joint_target`` of six
bare radians. Models misread those figures. On a surface that declares this
capability the delivered copy states every angle in degrees; the canonical
receipt, the tool-result event and every controller keep radians.

One explicit table names each radian key and the key it becomes. It also
names the two joint vectors that carry no unit label (``joint_target`` and
``stored_joints``, RoboDojo segment and recovery targets); those convert only
when their value is a numeric vector, because ``joint_target`` is also an
action-abstraction name. ``tests/test_policy_units.py`` holds the table to
every radian key a producer in ``ora`` can write.

Inputs a policy still types in radians (``RADIAN_INPUT_KEYS``) keep their
name and unit everywhere, so what the policy reads back matches what it sends.

Prose follows the values: ``<number> rad`` becomes ``<number> deg``, a radian
field name becomes its degree name, and a JSON pair inside text is converted
with its key. Image payloads are never read as prose.
"""

from __future__ import annotations

import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

CAPABILITY = "policy_angles_degrees_v1"

#: Decimal places of a converted value (1e-6 deg is about 1.7e-8 rad).
DEGREE_DECIMALS = 6

RADIAN_TO_DEGREE_KEYS: Mapping[str, str] = MappingProxyType({
    "accumulated_rotation_rad": "accumulated_rotation_deg",
    "actual_rotation_rad": "actual_rotation_deg",
    "canonical_interval_rad": "canonical_interval_deg",
    "canonical_jump_limit_rad": "canonical_jump_limit_deg",
    "comparison_epsilon_rad": "comparison_epsilon_deg",
    "correction_rad": "correction_deg",
    "current_rad": "current_deg",
    "deduplication_threshold_rad": "deduplication_threshold_deg",
    "direct_local_rotation_rad": "direct_local_rotation_deg",
    "error_rad": "error_deg",
    "excess_rad": "excess_deg",
    "execution_joint_step_rad": "execution_joint_step_deg",
    "final_joint_tracking_error_rad": "final_joint_tracking_error_deg",
    "finite_difference_epsilon_rad": "finite_difference_epsilon_deg",
    "global_joint_jump_limit_rad": "global_joint_jump_limit_deg",
    "gross_orientation_error_growth_rad": "gross_orientation_error_growth_deg",
    "ik_distance_rad": "ik_distance_deg",
    "isolation_bound_rad": "isolation_bound_deg",
    "joint6_isolation_bound_rad": "joint6_isolation_bound_deg",
    "joint6_margin_rad": "joint6_margin_deg",
    "joint_command_step_rad": "joint_command_step_deg",
    "joint_convergence_tolerance_rad": "joint_convergence_tolerance_deg",
    "joint_delta_rad": "joint_delta_deg",
    "joint_delta_tie_rad": "joint_delta_tie_deg",
    "joint_distance_before_rad": "joint_distance_before_deg",
    "joint_distance_end_rad": "joint_distance_end_deg",
    "joint_distance_start_rad": "joint_distance_start_deg",
    "joint_error_max_abs_rad": "joint_error_max_abs_deg",
    "joint_hold_tolerance_rad": "joint_hold_tolerance_deg",
    "joint_jump_limit_rad": "joint_jump_limit_deg",
    "joint_limit_epsilon_rad": "joint_limit_epsilon_deg",
    "joint_limit_margin_rad": "joint_limit_margin_deg",
    "joint_rad": "joint_deg",
    "joint_residual_max_abs_rad": "joint_residual_max_abs_deg",
    "joint_step_rad": "joint_step_deg",
    "joint_tolerance_rad": "joint_tolerance_deg",
    "joint_tracking_error_limit_rad": "joint_tracking_error_limit_deg",
    "joint_tracking_error_rad": "joint_tracking_error_deg",
    "joint_tracking_tolerance_rad": "joint_tracking_tolerance_deg",
    "jump_limit_rad": "jump_limit_deg",
    "last_joint_step_rad": "last_joint_step_deg",
    "limiting_joint_delta_rad": "limiting_joint_delta_deg",
    "local_joint_jump_limit_rad": "local_joint_jump_limit_deg",
    "lower_rad": "lower_deg",
    "margin_tie_rad": "margin_tie_deg",
    "max_deviation_rad": "max_deviation_deg",
    "max_joint_divergence_rad": "max_joint_divergence_deg",
    "max_rotation_rad": "max_rotation_deg",
    "max_step_rad": "max_step_deg",
    "max_step_rotation_rad": "max_step_rotation_deg",
    "maximum_deviation_rad": "maximum_deviation_deg",
    "maximum_distance_rad": "maximum_distance_deg",
    "maximum_endpoint_orientation_error_rad": "maximum_endpoint_orientation_error_deg",
    "maximum_frozen_joint_step_rad": "maximum_frozen_joint_step_deg",
    "maximum_joint_delta_rad": "maximum_joint_delta_deg",
    "maximum_joint_error_rad": "maximum_joint_error_deg",
    "maximum_joint_step_rad": "maximum_joint_step_deg",
    "maximum_joint_tracking_error_rad": "maximum_joint_tracking_error_deg",
    "maximum_local_rotation_rad": "maximum_local_rotation_deg",
    "maximum_negative_view_direction_change_rad": "maximum_negative_view_direction_change_deg",
    "maximum_per_joint_delta_rad": "maximum_per_joint_delta_deg",
    "maximum_public_tcp_start_orientation_error_rad": "maximum_public_tcp_start_orientation_error_deg",
    "maximum_rotation_rad": "maximum_rotation_deg",
    "maximum_rotation_step_rad": "maximum_rotation_step_deg",
    "maximum_start_joint_error_rad": "maximum_start_joint_error_deg",
    "maximum_step_rad": "maximum_step_deg",
    "measured_joints_rad": "measured_joints_deg",
    "min_joint_limit_margin_rad": "min_joint_limit_margin_deg",
    "minimum_progress_rad": "minimum_progress_deg",
    "nearby_distance_rad": "nearby_distance_deg",
    "numerical_crosscheck_epsilon_rad": "numerical_crosscheck_epsilon_deg",
    "orientation_error_growth_telemetry_rad": "orientation_error_growth_telemetry_deg",
    "orientation_error_rad": "orientation_error_deg",
    "orientation_error_reduction_rad": "orientation_error_reduction_deg",
    "orientation_rad": "orientation_deg",
    "orientation_residual_rad": "orientation_residual_deg",
    "orientation_shortfall_rad": "orientation_shortfall_deg",
    "orientation_tolerance_rad": "orientation_tolerance_deg",
    "peak_joint_step_rad": "peak_joint_step_deg",
    "precontact_arrival_reserve_rad": "precontact_arrival_reserve_deg",
    "probe_remaining_rotation_rad": "probe_remaining_rotation_deg",
    "probe_rotation_used_rad": "probe_rotation_used_deg",
    "realized_rotation_angle_rad": "realized_rotation_angle_deg",
    "remaining_orientation_error_rad": "remaining_orientation_error_deg",
    "requested_rotation_angle_rad": "requested_rotation_angle_deg",
    "requested_rotation_rad": "requested_rotation_deg",
    "requested_rotation_rotvec_rad": "requested_rotation_rotvec_deg",
    "rotation_angle_rad": "rotation_angle_deg",
    "rotation_rad": "rotation_deg",
    "rotation_used_rad": "rotation_used_deg",
    "rx_rad": "rx_deg",
    "ry_rad": "ry_deg",
    "rz_rad": "rz_deg",
    "selected_distance_rad": "selected_distance_deg",
    "single_tick_rotation_rad": "single_tick_rotation_deg",
    "target_orientation_error_after_rad": "target_orientation_error_after_deg",
    "target_orientation_error_before_rad": "target_orientation_error_before_deg",
    "target_orientation_error_growth_rad": "target_orientation_error_growth_deg",
    "target_orientation_error_rad": "target_orientation_error_deg",
    "target_rad": "target_deg",
    "terminal_settle_joint_tolerance_rad": "terminal_settle_joint_tolerance_deg",
    "tolerance_rad": "tolerance_deg",
    "tracking_error_l2_rad": "tracking_error_l2_deg",
    "tracking_error_max_abs_rad": "tracking_error_max_abs_deg",
    "tracking_error_rad": "tracking_error_deg",
    "unwrapped_l2_rad": "unwrapped_l2_deg",
    "upper_rad": "upper_deg",
    "validation_joint_step_rad": "validation_joint_step_deg",
    "value_rad": "value_deg",
    "view_direction_change_rad": "view_direction_change_deg",
    "waypoint_tracking_tolerance_rad": "waypoint_tracking_tolerance_deg",
    "wrapped_candidate_distance_rad": "wrapped_candidate_distance_deg",
    # Joint vectors that carry no unit label (inventory D2, Q17).
    "joint_target": "joint_target_deg",
    "stored_joints": "stored_joints_deg",
})

#: Inputs a policy still types in radians. They keep name and unit in what it
#: reads back until their own degree alias exists (inventory Q23).
RADIAN_INPUT_KEYS = frozenset({
    "rotation_rotvec_rad", "rotation_vector_rad",
})

#: Values that are payloads rather than prose or measurements.
OPAQUE_KEYS = frozenset({"image_base64", "images_base64", "_runtime"})

_UNLABELED = frozenset(key for key in RADIAN_TO_DEGREE_KEYS if not key.endswith("_rad"))
_LABELED = "|".join(sorted((key for key in RADIAN_TO_DEGREE_KEYS if key not in _UNLABELED),
                           key=len, reverse=True))
_NUMBER = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_JSON_PAIR = re.compile(rf'"({_LABELED})"(\s*:\s*)(-?{_NUMBER})')
_ASSIGNMENT = re.compile(rf"\b({_LABELED})(\s*[=:]\s*)(-?{_NUMBER})(?![\w.])")
_FIELD_NAME = re.compile(rf"\b({_LABELED})\b")
_NUMBER_RAD = re.compile(rf"(?<![\w.])({_NUMBER}) rad\b")
_PI_RAD = re.compile(r"\bpi rad\b")
_IN_RADIANS = re.compile(r"\b([a-z][a-z0-9_]*_deg) in radians\b")


def degrees(value: float) -> float:
    """One radian value as delivered degrees."""
    return round(math.degrees(value), DEGREE_DECIMALS)


def _prose_number(value: float) -> str:
    """Two decimals from one degree up, three significant digits below it."""
    if value == 0:
        return "0"
    text = f"{value:.2f}" if abs(value) >= 1 else f"{value:.3g}"
    return text.rstrip("0").rstrip(".") if "." in text and "e" not in text else text


def degrees_text(text: str) -> str:
    """The same sentence with every stated radian quantity in degrees.

    A field name is never renamed apart from a value it states: a JSON pair or
    ``name=value`` / ``name: value`` converts the number with the name.
    """
    text = _JSON_PAIR.sub(lambda match: (
        f'"{RADIAN_TO_DEGREE_KEYS[match.group(1)]}"{match.group(2)}'
        f"{json.dumps(degrees(float(match.group(3))))}"), text)
    text = _ASSIGNMENT.sub(lambda match: (
        f"{RADIAN_TO_DEGREE_KEYS[match.group(1)]}{match.group(2)}"
        f"{json.dumps(degrees(float(match.group(3))))}"), text)
    text = _PI_RAD.sub("180 deg", text)
    text = _NUMBER_RAD.sub(
        lambda match: f"{_prose_number(math.degrees(float(match.group(1))))} deg", text)
    text = _FIELD_NAME.sub(lambda match: RADIAN_TO_DEGREE_KEYS[match.group(1)], text)
    return _IN_RADIANS.sub(r"\1 in degrees", text)


def _numeric_vector(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _angles(value: Any) -> Any:
    """Every number under a radian key, in degrees; flags and absences unchanged."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return degrees(float(value))
    if isinstance(value, list):
        return [_angles(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _angles(item) for key, item in value.items()}
    if isinstance(value, str):
        return degrees_text(value)
    return value


def policy_degrees(value: Any) -> Any:
    """A new copy of one policy-visible value with every angle in degrees.

    A radian key whose degree name already sits beside it keeps its own label
    and value, so neither copy is overwritten.
    """
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            renamed = RADIAN_TO_DEGREE_KEYS.get(key)
            if key in OPAQUE_KEYS:
                result[key] = child
            elif renamed is not None and renamed not in value and (
                    key not in _UNLABELED or _numeric_vector(child)):
                result[renamed] = _angles(child)
            else:
                result[key] = policy_degrees(child)
        return result
    if isinstance(value, list):
        return [policy_degrees(item) for item in value]
    if isinstance(value, str):
        return degrees_text(value)
    return value


__all__ = [
    "CAPABILITY", "DEGREE_DECIMALS", "OPAQUE_KEYS", "RADIAN_INPUT_KEYS", "RADIAN_TO_DEGREE_KEYS",
    "degrees", "degrees_text", "policy_degrees",
]
