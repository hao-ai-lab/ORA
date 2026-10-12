"""Versioned, host-safe profile for the hybrid Panda motion boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.motion import (
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
)


HYBRID_MOTION_PROFILE_SCHEMA = "ora.hybrid-motion-profile.v1"
LIBERO_MOTION_PROFILE_V2 = "franka_panda_libero_pro_hybrid_motion_v2"

# One identity per live LIBERO endpoint-verification route. The endpoint bound
# differs by route and each bound is correct for its own executor, so a receipt
# that reports a residual without its route cannot be read by the policy.
LIBERO_ENDPOINT_CONTRACT_SCHEMA = "ora.libero-endpoint-contract.v1"
LIBERO_ENDPOINT_ROUTE_GLOBAL = "curobo_global_joint_position"
LIBERO_ENDPOINT_ROUTE_LOCAL = "osc_local_pose"
LIBERO_ENDPOINT_ROUTES = (
    LIBERO_ENDPOINT_ROUTE_GLOBAL, LIBERO_ENDPOINT_ROUTE_LOCAL,
)


def precision_force_stop_enabled(profile: Mapping[str, Any] | None) -> bool:
    """Select the versioned precision stop; historical v1 stays telemetry-only."""
    return profile is not None and profile.get("profile_id") == LIBERO_MOTION_PROFILE_V2


def libero_global_endpoint_tolerances() -> tuple[float, float]:
    """Return the existing Panda global-route qualification bound.

    This source-bound contract applies to CuRobo transport, including precision
    requests routed globally. OSC retains its tighter action-contract endpoint
    checks. Execution and qualification must import this same definition.
    """

    return 0.015, math.radians(5.0)


class HybridMotionProfileError(ValueError):
    """The selected motion profile is malformed or no longer hash-bound."""


def libero_local_endpoint_tolerances() -> tuple[float, float]:
    """Return the local OSC route's endpoint bound.

    Local execution is judged by the shared action contract, not by the looser
    CuRobo transport bound. Execution, receipt disclosure and qualification must
    import this same definition so the number a policy is judged against and the
    number a policy is told about can never drift apart.
    """

    return POSE_POSITION_TOLERANCE_M, POSE_ORIENTATION_TOLERANCE_RAD


def libero_endpoint_tolerances(route: str) -> tuple[float, float]:
    """Return the endpoint bound that one named execution route applies."""

    if route == LIBERO_ENDPOINT_ROUTE_GLOBAL:
        return libero_global_endpoint_tolerances()
    if route == LIBERO_ENDPOINT_ROUTE_LOCAL:
        return libero_local_endpoint_tolerances()
    raise HybridMotionProfileError(f"unknown LIBERO endpoint route: {route}")


def _optional_finite(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridMotionProfileError(f"{field} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise HybridMotionProfileError(f"{field} must be finite")
    return number


def _optional_count(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise HybridMotionProfileError(f"{field} must be an integer or null")
    if value < 0:
        raise HybridMotionProfileError(f"{field} must not be negative")
    return int(value)


def libero_endpoint_contract(
    *,
    route: str,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    position_residual_m: Any,
    orientation_residual_rad: Any,
    within_tolerance: bool | None,
    convergence_budget: str,
    ticks_used: Any = None,
    ticks_budget: Any = None,
    budget_exhausted: bool = False,
    execution_phase: str = "unknown",
    stop_reason: str | None = None,
    waypoints_reached: Any = None,
    waypoints_total: Any = None,
) -> dict[str, Any]:
    """Compile the policy-visible endpoint-verification contract for one action.

    ``position_tolerance_m`` / ``orientation_tolerance_rad`` are the values the
    caller actually compared against, and they are re-checked here against the
    single definition owned by ``route``. A projection layer must lift this
    record, never re-derive a default bound of its own.

    Residuals describe the measured current pose, including interrupted paths.
    An exhausted flag alone never proves endpoint convergence was attempted.
    Capability evidence requires a known convergence phase and consumed budget;
    missing historical phase evidence remains unknown. Bounds are never relaxed.
    """

    if route not in LIBERO_ENDPOINT_ROUTES:
        raise HybridMotionProfileError(f"unknown LIBERO endpoint route: {route}")
    expected = libero_endpoint_tolerances(route)
    applied = (float(position_tolerance_m), float(orientation_tolerance_rad))
    if not all(math.isfinite(item) for item in applied):
        raise HybridMotionProfileError("endpoint tolerance must be finite")
    if not all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(applied, expected, strict=True)
    ):
        raise HybridMotionProfileError(
            f"reported endpoint tolerance {applied} is not the bound route "
            f"{route} applies {expected}"
        )
    if not isinstance(convergence_budget, str) or not convergence_budget.strip():
        raise HybridMotionProfileError("convergence budget must be named")
    position = _optional_finite(position_residual_m, "position_residual_m")
    orientation = _optional_finite(
        orientation_residual_rad, "orientation_residual_rad",
    )
    used = _optional_count(ticks_used, "ticks_used")
    budget = _optional_count(ticks_budget, "ticks_budget")
    exhausted = bool(budget_exhausted)
    if execution_phase not in {"unknown", "waypoint_tracking", "terminal_settle", "local_control"}:
        raise HybridMotionProfileError("unknown endpoint execution phase")
    reached_waypoints = _optional_count(waypoints_reached, "waypoints_reached")
    total_waypoints = _optional_count(waypoints_total, "waypoints_total")
    convergence = {
        "budget": convergence_budget, "ticks_used": used,
        "ticks_budget": budget, "exhausted": exhausted,
    }
    contract: dict[str, Any] = {
        "schema": LIBERO_ENDPOINT_CONTRACT_SCHEMA,
        "execution_route": route,
        "execution_phase": execution_phase,
        "stop_reason": stop_reason,
        "waypoints_reached": reached_waypoints,
        "waypoints_total": total_waypoints,
        "tolerance": {
            "position_m": applied[0], "orientation_rad": applied[1],
        },
        "convergence": convergence,
    }
    if position is None or orientation is None:
        contract.update({
            "judged_residual": None, "within_tolerance": None,
            "position_shortfall_m": None, "orientation_shortfall_rad": None,
            "achieved_capability": None,
            "convergence_status": "unknown",
            "summary": "Endpoint residual evidence is unavailable; endpoint convergence is not established.",
        })
        return contract
    if within_tolerance is None:
        raise HybridMotionProfileError(
            "a measured endpoint residual requires its applied verdict"
        )
    verdict = bool(within_tolerance)
    if verdict and (position > applied[0] or orientation > applied[1]):
        raise HybridMotionProfileError(
            "endpoint verdict claims arrival outside the applied bound"
        )
    shortfall_position = max(0.0, position - applied[0])
    shortfall_orientation = max(0.0, orientation - applied[1])
    contract.update({
        "judged_residual": {
            "position_m": position, "orientation_rad": orientation,
        },
        "within_tolerance": verdict,
        "position_shortfall_m": shortfall_position,
        "orientation_shortfall_rad": shortfall_orientation,
        "achieved_capability": None,
    })
    residual_phrase = (
        f"current residual {position * 1000.0:.1f} mm / {orientation:.4f} rad "
        f"against {applied[0] * 1000.0:.1f} mm / {applied[1]:.4f} rad"
    )
    if verdict:
        contract["convergence_status"] = "reached"
        contract["summary"] = f"Measured pose meets the endpoint bound; {residual_phrase}."
        return contract
    convergence_phase = execution_phase in {"terminal_settle", "local_control"}
    budget_stop = stop_reason in {None, "terminal_tracking_timeout", "servo_iteration_budget_exhausted"}
    consumed_budget = used is not None and budget is not None and 0 < budget <= used
    if convergence_phase and exhausted and consumed_budget and budget_stop:
        contract["convergence_status"] = "budget_exhausted"
        contract["summary"] = (
            f"route {route} exhausted its {budget}-step {convergence_budget} budget; "
            f"{residual_phrase}. This is the residual achieved in this bounded "
            "attempt, not a general controller capability limit."
        )
        contract["achieved_capability"] = {
            "position_residual_m": position,
            "orientation_residual_rad": orientation,
            "position_shortfall_m": shortfall_position,
            "orientation_shortfall_rad": shortfall_orientation,
            "meets_advertised_tolerance": False,
            "limit": "convergence_budget_exhausted",
            "summary": contract["summary"],
        }
    elif execution_phase == "waypoint_tracking":
        contract["convergence_status"] = "not_started"
        progress = (
            f"{reached_waypoints}/{total_waypoints} waypoints reached"
            if reached_waypoints is not None and total_waypoints is not None
            else "waypoint progress unavailable"
        )
        contract["summary"] = (
            f"Motion stopped during waypoint tracking ({stop_reason or 'reason unavailable'}; "
            f"{progress}); endpoint convergence has not started; {residual_phrase}."
        )
    elif convergence_phase and stop_reason is not None:
        contract["convergence_status"] = "interrupted"
        contract["summary"] = (
            f"Endpoint convergence was interrupted ({stop_reason}) after {used} "
            f"{convergence_budget} steps; {residual_phrase}."
        )
    else:
        contract["convergence_status"] = "unknown"
        contract["summary"] = (
            f"Endpoint convergence completion is not established; {residual_phrase}."
        )
    return contract


def _finite_number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridMotionProfileError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise HybridMotionProfileError(
            f"{field} must be in [{minimum}, {maximum}]"
        )
    return number


def validate_hybrid_motion_profile(value: Any) -> dict[str, Any]:
    """Validate the small execution/safety profile used by both control modes."""

    if not isinstance(value, Mapping):
        raise HybridMotionProfileError("motion profile must be an object")
    required = {
        "schema", "profile_id", "embodiment_id", "gripper", "servo",
        "global_motion", "sensing",
    }
    if set(value) != required:
        raise HybridMotionProfileError("motion profile fields drifted")
    if value.get("schema") != HYBRID_MOTION_PROFILE_SCHEMA:
        raise HybridMotionProfileError("motion profile schema drifted")
    if value.get("embodiment_id") != "franka_panda":
        raise HybridMotionProfileError("motion profile embodiment drifted")
    if not isinstance(value.get("profile_id"), str) or not value["profile_id"].strip():
        raise HybridMotionProfileError("motion profile id must be non-empty")
    sections = {
        name: value.get(name) for name in (
            "gripper", "servo", "global_motion", "sensing",
        )
    }
    if not all(isinstance(item, Mapping) for item in sections.values()):
        raise HybridMotionProfileError("motion profile sections must be objects")
    gripper = sections["gripper"]
    if set(gripper) != {
        "open_directive", "close_directive", "maximum_aperture_m",
        "command_speed_normalized_per_tick", "closure_contact_min_normalized",
    }:
        raise HybridMotionProfileError("gripper profile fields drifted")
    if (
        _finite_number(gripper["open_directive"], "gripper.open_directive", minimum=0, maximum=0) != 0
        or _finite_number(gripper["close_directive"], "gripper.close_directive", minimum=1, maximum=1) != 1
    ):
        raise HybridMotionProfileError("gripper directives must remain discrete")
    _finite_number(gripper["maximum_aperture_m"], "gripper.maximum_aperture_m", minimum=0.01, maximum=0.2)
    _finite_number(gripper["command_speed_normalized_per_tick"], "gripper.command_speed_normalized_per_tick", minimum=0.01, maximum=1)
    _finite_number(
        gripper["closure_contact_min_normalized"],
        "gripper.closure_contact_min_normalized", minimum=0.01, maximum=0.5,
    )
    servo = sections["servo"]
    if set(servo) != {"max_translation_m", "max_rotation_rad", "max_iterations"}:
        raise HybridMotionProfileError("servo profile fields drifted")
    if _finite_number(servo["max_translation_m"], "servo.max_translation_m", minimum=0.05, maximum=0.05) != 0.05:
        raise HybridMotionProfileError("servo translation limit drifted")
    _finite_number(servo["max_rotation_rad"], "servo.max_rotation_rad", minimum=math.radians(15), maximum=math.radians(15))
    if isinstance(servo["max_iterations"], bool) or servo["max_iterations"] != 20:
        raise HybridMotionProfileError("servo max_iterations must be 20")
    global_motion = sections["global_motion"]
    global_fields = {
        "joint_tracking_error_limit_rad", "waypoint_tracking_tolerance_rad",
        "joint_command_step_rad", "max_ticks_per_waypoint",
        "max_terminal_settle_ticks", "max_execution_ticks", "max_waypoints",
    }
    if precision_force_stop_enabled(value):
        global_fields.add("terminal_settle_joint_tolerance_rad")
    if set(global_motion) != global_fields:
        raise HybridMotionProfileError("global motion profile fields drifted")
    if precision_force_stop_enabled(value):
        _finite_number(
            global_motion["terminal_settle_joint_tolerance_rad"],
            "global_motion.terminal_settle_joint_tolerance_rad", minimum=0.005,
            maximum=0.005,
        )
    _finite_number(global_motion["joint_tracking_error_limit_rad"], "global_motion.joint_tracking_error_limit_rad", minimum=0.001, maximum=0.2)
    _finite_number(
        global_motion["waypoint_tracking_tolerance_rad"],
        "global_motion.waypoint_tracking_tolerance_rad", minimum=0.001,
        maximum=0.1,
    )
    _finite_number(
        global_motion["joint_command_step_rad"],
        "global_motion.joint_command_step_rad", minimum=0.005, maximum=0.2,
    )
    for field, maximum in (
        ("max_ticks_per_waypoint", 100),
        ("max_terminal_settle_ticks", 500),
        ("max_execution_ticks", 20000),
    ):
        if (
            isinstance(global_motion[field], bool)
            or not isinstance(global_motion[field], int)
            or not 1 <= global_motion[field] <= maximum
        ):
            raise HybridMotionProfileError(f"global_motion.{field} is invalid")
    if (
        isinstance(global_motion["max_waypoints"], bool)
        or not isinstance(global_motion["max_waypoints"], int)
        or not 1 <= global_motion["max_waypoints"] <= 5000
    ):
        raise HybridMotionProfileError("global_motion.max_waypoints is invalid")
    sensing = sections["sensing"]
    if set(sensing) != {
        "wrench_clip_force_n", "wrench_clip_torque_nm",
        "contact_force_threshold_n", "force_telemetry_threshold_n",
    }:
        raise HybridMotionProfileError("sensing profile fields drifted")
    _finite_number(sensing["wrench_clip_force_n"], "sensing.wrench_clip_force_n", minimum=1, maximum=100)
    _finite_number(sensing["wrench_clip_torque_nm"], "sensing.wrench_clip_torque_nm", minimum=0.1, maximum=100)
    _finite_number(sensing["contact_force_threshold_n"], "sensing.contact_force_threshold_n", minimum=1, maximum=100)
    _finite_number(sensing["force_telemetry_threshold_n"], "sensing.force_telemetry_threshold_n", minimum=1, maximum=1000)
    return json.loads(json.dumps(value))


def load_hybrid_motion_profile(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load one exact local profile and reject substitution or path drift."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise HybridMotionProfileError("motion profile file is unavailable")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise HybridMotionProfileError("motion profile hash drifted")
    value = validate_hybrid_motion_profile(json.loads(resolved.read_text(encoding="utf-8")))
    return {"path": str(resolved), "sha256": digest, "profile": value}


def motion_stop_reason(
    *, stop_request_path: Path,
) -> str | None:
    """Return the trusted asynchronous operator stop for one control tick.

    Force-stop policy is checked separately by the selected execution adapter;
    this function owns only asynchronous operator interruption.
    """

    if stop_request_path.is_file():
        return "operator_stop_requested"
    return None


def aggregate_motion_contact(
    *, external_force_xyz: Any, force_threshold_n: float,
    closing: bool, measured_aperture_normalized: float,
    closure_contact_min_normalized: float,
) -> bool | None:
    """Aggregate wrist force and blocked closure without inferring a payload."""

    closure_contact = (
        closing
        and math.isfinite(float(measured_aperture_normalized))
        and float(measured_aperture_normalized)
        > float(closure_contact_min_normalized)
    )
    if external_force_xyz is None:
        return True if closure_contact else None
    if (
        not isinstance(external_force_xyz, (list, tuple))
        or len(external_force_xyz) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in external_force_xyz
        )
    ):
        return None
    force_contact = math.sqrt(
        sum(float(item) ** 2 for item in external_force_xyz)
    ) > float(force_threshold_n)
    return force_contact or closure_contact


def bounded_joint_position_command(
    current: Any, target: Any, *, maximum_step_rad: float,
) -> list[float]:
    """Compile one normalized RoboSuite JOINT_POSITION delta command."""

    if (
        not isinstance(current, (list, tuple))
        or not isinstance(target, (list, tuple))
        or len(current) != 7 or len(target) != 7
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in [*current, *target]
        )
    ):
        raise HybridMotionProfileError("joint command states must be finite 7-vectors")
    step = _finite_number(
        maximum_step_rad, "maximum_step_rad", minimum=0.005, maximum=0.2,
    )
    return [
        max(-1.0, min(1.0, (float(goal) - float(measured)) / step))
        for measured, goal in zip(current, target)
    ]


__all__ = [
    "HYBRID_MOTION_PROFILE_SCHEMA", "HybridMotionProfileError",
    "aggregate_motion_contact", "bounded_joint_position_command",
    "load_hybrid_motion_profile",
    "libero_endpoint_contract", "libero_endpoint_tolerances",
    "libero_global_endpoint_tolerances", "libero_local_endpoint_tolerances",
    "LIBERO_ENDPOINT_CONTRACT_SCHEMA", "LIBERO_ENDPOINT_ROUTES",
    "LIBERO_ENDPOINT_ROUTE_GLOBAL", "LIBERO_ENDPOINT_ROUTE_LOCAL",
    "LIBERO_MOTION_PROFILE_V2", "precision_force_stop_enabled",
    "motion_stop_reason",
    "validate_hybrid_motion_profile",
]
