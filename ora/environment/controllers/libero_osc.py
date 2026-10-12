"""Source-bound RoboSuite OSC_POSE input/output and safety contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


OSC_ACTION_CONTRACT_SCHEMA = "ora.libero-osc-action-contract.v2"
CONTROL_HEALTH_NUMERIC_TOLERANCE = 1e-6
TRACKING_POSITION_GROWTH_TELEMETRY_M = 0.02
TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M = 0.02
TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD = 0.18
TRACKING_CONSECUTIVE_TICKS = 2
GROSS_SINGLE_TICK_TRANSLATION_M = 0.05
GROSS_ORIENTATION_GROWTH_RAD = 0.25
GROSS_SAFETY_ENVELOPE_OVERRUN_M = 0.05
# Normalized measured aperture; both execution and public acceptance use the
# strict comparison against this bound. This is not a jaw length in metres.
GRIPPER_TOLERANCE_NORMALIZED = 0.03


class LiberoOscContractError(ValueError):
    """The pinned controller configuration violated the closed OSC contract."""


def libero_osc_action_contract(controller_config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind physical OSC output ranges to normalized controller input."""

    def vector(name: str) -> list[float]:
        raw = controller_config.get(name)
        if isinstance(raw, bool) or not isinstance(
            raw, (int, float, list, tuple),
        ):
            raise LiberoOscContractError(f"OSC controller config lacks {name}")
        values = [raw] * 6 if isinstance(raw, (int, float)) else list(raw)
        if len(values) != 6 or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in values
        ):
            raise LiberoOscContractError(
                f"OSC controller config has invalid {name}"
            )
        return [float(item) for item in values]

    if controller_config.get("control_delta") is not True:
        raise LiberoOscContractError("LIBERO OSC_POSE must use delta control")
    input_min = vector("input_min")
    input_max = vector("input_max")
    output_min = vector("output_min")
    output_max = vector("output_max")
    if any(
        input_min[index] >= input_max[index]
        or output_min[index] >= 0
        or output_max[index] <= 0
        for index in range(6)
    ):
        raise LiberoOscContractError(
            "OSC controller ranges are not signed and increasing"
        )
    body = {
        "schema": OSC_ACTION_CONTRACT_SCHEMA,
        "controller": "OSC_POSE",
        "control_delta": True,
        "dimensions": ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"],
        "input_min": input_min,
        "input_max": input_max,
        "output_min": output_min,
        "output_max": output_max,
        "servo_clip": {"translation_m": 0.025, "rotation_rad": 0.10},
        "tracking_envelope": {
            "position_error_growth_telemetry_m": (
                TRACKING_POSITION_GROWTH_TELEMETRY_M
            ),
            "cross_axis_drift": "telemetry_only_no_stop_authority",
            "travel_overshoot_telemetry_m": (
                TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M
            ),
            "orientation_error_growth_telemetry_rad": (
                TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD
            ),
            "consecutive_ticks": TRACKING_CONSECUTIVE_TICKS,
            "gross_single_tick_translation_m": GROSS_SINGLE_TICK_TRANSLATION_M,
            "gross_orientation_error_growth_rad": GROSS_ORIENTATION_GROWTH_RAD,
            "gross_safety_envelope_overrun_m": GROSS_SAFETY_ENVELOPE_OVERRUN_M,
            "ordinary_tracking_effect": "telemetry_only_no_stop_authority",
            "terminal_stop": (
                "nonfinite_or_command_invariant_or_gross_state_corruption"
            ),
        },
        "gripper_control": {
            "backend": "robosuite_panda_sign_integrator",
            "pose_action_input": 0.0,
            "pose_action_precondition": "existing_integrator_directive_preserved",
            "pose_action_hold_semantics": (
                "actuator_target_not_measured_aperture"
            ),
            "set_gripper_inputs": {"open": -1.0, "close": 1.0},
        },
        "source": "pinned_robosuite_controller_config",
    }
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {**body, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def validate_libero_osc_action_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete hash-bound OSC command boundary."""

    expected = {
        "schema", "controller", "control_delta", "dimensions",
        "input_min", "input_max", "output_min", "output_max", "servo_clip",
        "tracking_envelope", "gripper_control", "source", "contract_sha256",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected:
        raise LiberoOscContractError("OSC action contract fields drifted")
    body = {key: contract[key] for key in contract if key != "contract_sha256"}
    actual = hashlib.sha256(json.dumps(
        body, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if (
        contract.get("schema") != OSC_ACTION_CONTRACT_SCHEMA
        or contract.get("controller") != "OSC_POSE"
        or contract.get("control_delta") is not True
        or contract.get("dimensions")
        != ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"]
        or contract.get("source") != "pinned_robosuite_controller_config"
        or not isinstance(contract.get("contract_sha256"), str)
        or contract["contract_sha256"] != actual
    ):
        raise LiberoOscContractError("OSC action contract identity is invalid")

    def vector(name: str) -> list[float]:
        raw = contract.get(name)
        if not isinstance(raw, list) or len(raw) != 6 or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in raw
        ):
            raise LiberoOscContractError(f"OSC action contract {name} is invalid")
        return [float(item) for item in raw]

    input_min = vector("input_min")
    input_max = vector("input_max")
    output_min = vector("output_min")
    output_max = vector("output_max")
    if any(
        input_min[index] >= input_max[index]
        or output_min[index] >= output_max[index]
        or output_min[index] >= 0.0
        or output_max[index] <= 0.0
        for index in range(6)
    ):
        raise LiberoOscContractError("OSC action contract ranges are invalid")
    if contract.get("servo_clip") != {
        "translation_m": 0.025, "rotation_rad": 0.10,
    }:
        raise LiberoOscContractError("OSC action contract servo clip drifted")
    if contract.get("tracking_envelope") != {
        "position_error_growth_telemetry_m": TRACKING_POSITION_GROWTH_TELEMETRY_M,
        "cross_axis_drift": "telemetry_only_no_stop_authority",
        "travel_overshoot_telemetry_m": TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M,
        "orientation_error_growth_telemetry_rad": (
            TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD
        ),
        "consecutive_ticks": TRACKING_CONSECUTIVE_TICKS,
        "gross_single_tick_translation_m": GROSS_SINGLE_TICK_TRANSLATION_M,
        "gross_orientation_error_growth_rad": GROSS_ORIENTATION_GROWTH_RAD,
        "gross_safety_envelope_overrun_m": GROSS_SAFETY_ENVELOPE_OVERRUN_M,
        "ordinary_tracking_effect": "telemetry_only_no_stop_authority",
        "terminal_stop": "nonfinite_or_command_invariant_or_gross_state_corruption",
    }:
        raise LiberoOscContractError("OSC action contract tracking envelope drifted")
    if contract.get("gripper_control") != {
        "backend": "robosuite_panda_sign_integrator",
        "pose_action_input": 0.0,
        "pose_action_precondition": "existing_integrator_directive_preserved",
        "pose_action_hold_semantics": "actuator_target_not_measured_aperture",
        "set_gripper_inputs": {"open": -1.0, "close": 1.0},
    }:
        raise LiberoOscContractError("OSC action contract gripper boundary drifted")
    return dict(contract)


def inverse_controller_scale(
    physical_delta: Any, contract: Mapping[str, Any],
) -> Any:
    """Invert RoboSuite's affine physical-output to normalized-input mapping."""

    import numpy as np

    validated = validate_libero_osc_action_contract(contract)
    physical = np.asarray(physical_delta, dtype=np.float64)
    if physical.shape != (6,) or not np.isfinite(physical).all():
        raise LiberoOscContractError(
            "OSC physical delta must contain six finite values"
        )
    input_min = np.asarray(validated["input_min"], dtype=np.float64)
    input_max = np.asarray(validated["input_max"], dtype=np.float64)
    output_min = np.asarray(validated["output_min"], dtype=np.float64)
    output_max = np.asarray(validated["output_max"], dtype=np.float64)
    scale = (output_max - output_min) / (input_max - input_min)
    output_transform = (output_max + output_min) / 2.0
    input_transform = -(input_max + input_min) / 2.0
    normalized = (physical - output_transform) / scale - input_transform
    return np.clip(normalized, input_min, input_max).astype(np.float32)


__all__ = [
    "CONTROL_HEALTH_NUMERIC_TOLERANCE",
    "GROSS_ORIENTATION_GROWTH_RAD",
    "GROSS_SAFETY_ENVELOPE_OVERRUN_M",
    "GROSS_SINGLE_TICK_TRANSLATION_M",
    "LiberoOscContractError",
    "OSC_ACTION_CONTRACT_SCHEMA",
    "TRACKING_CONSECUTIVE_TICKS",
    "TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD",
    "TRACKING_POSITION_GROWTH_TELEMETRY_M",
    "TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M",
    "inverse_controller_scale",
    "libero_osc_action_contract",
    "validate_libero_osc_action_contract",
]
