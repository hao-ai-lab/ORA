"""Public gripper intent, drive target and encoder evidence, kept distinct.

No controller constants live here. The environment supplies the actual target,
measurement and arrival bound; projections validate and preserve that evidence.

``two_finger_encoder_mean_v1`` (schema v2, EXP11A g012 E11G12-001) is a
measurement from both fingers: the mean of the two finger joint positions on the
primary finger's command scale, published with the signed lateral offset of the
finger pair. It is a normalized encoder reading, not a calibrated aperture.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


GRIPPER_FEEDBACK_SCHEMA = "ora.gripper-feedback.v1"
GRIPPER_FEEDBACK_SCHEMA_V2 = "ora.gripper-feedback.v2"
#: The mean of both finger encoders on the primary command scale; requires
#: ``finger_lateral_offset_m`` = (primary - secondary) / 2 in metres, positive
#: when the finger pair sits toward the primary finger's opening direction.
TWO_FINGER_MEASUREMENT_SOURCE = "two_finger_encoder_mean_v1"
MEASUREMENT_SOURCES = frozenset({"primary_finger_encoder", "normalized_measured_aperture",
                                 TWO_FINGER_MEASUREMENT_SOURCE})
_NUMBERS = ("requested_opening_normalized", "drive_target_opening_normalized",
            "measured_opening_normalized", "effective_target_opening_normalized")
_SOURCES = {"requested_command", "drive_target_source", "measurement_source", "pose_hold_semantics"}


def gripper_feedback(*, requested_opening_normalized: float | None,
                     drive_target_opening_normalized: float,
                     measured_opening_normalized: float,
                     effective_target_opening_normalized: float | None,
                     tolerance_normalized: float, comparison: str = "<=",
                     drive_target_source: str, measurement_source: str,
                     pose_hold_semantics: str,
                     finger_lateral_offset_m: float | None = None) -> dict[str, Any]:
    """Build an observation-bound comparison, never infer intent from a drive.

    ``finger_lateral_offset_m`` accompanies exactly the two-finger source.
    """
    numbers = {
        "requested_opening_normalized": requested_opening_normalized,
        "drive_target_opening_normalized": drive_target_opening_normalized,
        "measured_opening_normalized": measured_opening_normalized,
        "effective_target_opening_normalized": effective_target_opening_normalized,
    }
    for name, value in numbers.items():
        if value is None and name in {"requested_opening_normalized", "effective_target_opening_normalized"}:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite evidence")
    if (type(tolerance_normalized) not in (int, float) or not math.isfinite(tolerance_normalized)
            or tolerance_normalized <= 0 or comparison not in {"<", "<="}):
        raise ValueError("gripper arrival bound is invalid")
    if (requested_opening_normalized is None) != (effective_target_opening_normalized is None):
        raise ValueError("an unknown directive cannot have a claimed arrival target")
    if drive_target_source not in {"resolved_sdk_joint_position_target", "actuator_joint_position_target"}:
        raise ValueError("unknown public gripper drive source")
    if measurement_source not in MEASUREMENT_SOURCES:
        raise ValueError("unknown public gripper measurement source")
    two_finger = measurement_source == TWO_FINGER_MEASUREMENT_SOURCE
    if two_finger != (finger_lateral_offset_m is not None) or two_finger and (
            type(finger_lateral_offset_m) not in (int, float) or not math.isfinite(finger_lateral_offset_m)):
        raise ValueError("a finite finger lateral offset accompanies exactly the two-finger measurement")
    if pose_hold_semantics not in {"preserve_last_applied_drive_target", "preserve_actuator_target"}:
        raise ValueError("unknown public gripper hold semantics")
    residual = (None if effective_target_opening_normalized is None else
                abs(measured_opening_normalized - effective_target_opening_normalized))
    within = (None if residual is None else residual < tolerance_normalized
              if comparison == "<" else residual <= tolerance_normalized)
    return {
        "schema": GRIPPER_FEEDBACK_SCHEMA_V2 if two_finger else GRIPPER_FEEDBACK_SCHEMA,
        **numbers,
        "requested_command": {0.0: "close", 1.0: "open"}.get(requested_opening_normalized, "unknown"),
        "arrival_check": {"comparison": comparison, "residual_normalized": residual,
                          "tolerance_normalized": tolerance_normalized, "within_tolerance": within},
        "drive_target_source": drive_target_source,
        "measurement_source": measurement_source,
        "pose_hold_semantics": pose_hold_semantics,
        **({"finger_lateral_offset_m": finger_lateral_offset_m} if two_finger else {}),
    }


def validate_gripper_feedback(value: Any) -> dict[str, Any]:
    """Reject dropped, contradictory or undeclared fields at each projection."""
    fields = {"schema", "arrival_check", *_NUMBERS, *_SOURCES}
    if not isinstance(value, Mapping) or set(value) not in (fields, fields | {"finger_lateral_offset_m"}):
        raise ValueError("gripper feedback fields differ from the public contract")
    check = value["arrival_check"]
    if not isinstance(check, Mapping) or set(check) != {
        "comparison", "residual_normalized", "tolerance_normalized", "within_tolerance",
    }:
        raise ValueError("gripper arrival fields differ from the public contract")
    canonical = gripper_feedback(**{name: value[name] for name in _NUMBERS},
        comparison=check["comparison"], tolerance_normalized=check["tolerance_normalized"],
        finger_lateral_offset_m=value.get("finger_lateral_offset_m"),
        **{name: value[name] for name in _SOURCES - {"requested_command"}})
    if canonical != value:
        raise ValueError("gripper feedback disagrees with its source quantities")
    return canonical


def validate_controller_feedback(value: Any) -> dict[str, Any]:
    """Allowlist a source-bound controller description without new defaults."""
    keys = {"schema", "position_tolerance_m", "orientation_tolerance_rad",
            "gripper_tolerance_normalized", "arrival_comparison",
            "native_actions_per_command", "gripper_drive_limit"}
    if (not isinstance(value, Mapping) or set(value) != keys
            or value["schema"] != "ora.controller-feedback.v1"
            or value["arrival_comparison"] not in {"<", "<="}):
        raise ValueError("controller feedback fields differ")
    for key in ("position_tolerance_m", "orientation_tolerance_rad", "gripper_tolerance_normalized"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("controller feedback tolerance must be positive finite evidence")
    if type(value["native_actions_per_command"]) is not int or value["native_actions_per_command"] <= 0:
        raise ValueError("controller feedback window is invalid")
    rule = value["gripper_drive_limit"]
    expected = {"kind": "measured_primary_relative", "reference": "current_primary_finger_encoder",
                "coupling": "both_drive_targets_from_limited_primary_target",
                "applies_to": "explicit_gripper_commands", "pose_hold": "preserve_last_applied_drive_target"}
    if (not isinstance(rule, Mapping) or set(rule) != {*expected, "maximum_normalized_offset"}
            or any(rule[k] != v for k, v in expected.items())):
        raise ValueError("controller feedback drive rule is invalid")
    bound = rule["maximum_normalized_offset"]
    if type(bound) not in (int, float) or not math.isfinite(bound) or not 0 < bound <= 1:
        raise ValueError("controller feedback drive bound is invalid")
    return {**value, "gripper_drive_limit": dict(rule)}
