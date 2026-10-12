"""Pure decisions and projections for LIBERO's settling motion route.

These are the parts of the sparse-v3 route that depend only on numbers, not on
a live simulator: whether a global segment should hand off to a local one, how
the two segments compose into one receipt, and how the held gripper directive
becomes a public command. Keeping them here lets them be tested without a
LIBERO environment and keeps ``session.py`` from growing further.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.action import (
    SERVO_MAX_ROTATION_RAD, SERVO_MAX_TRANSLATION_M,
)
from ora.environment.controllers.hybrid_motion import (
    libero_global_endpoint_tolerances,
)
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, GripperStallRule,
    gripper_outcome as _shared_gripper_outcome,
    gripper_stalled as _shared_gripper_stalled,
)
from ora.environment.controllers.libero_osc import (
    GRIPPER_TOLERANCE_NORMALIZED,
)
from ora.environment.controllers.motion_transaction import (
    _precontact_pose,
)

__all__ = [
    "GRIPPER_STALL_REASON", "LIBERO_CONTROL_PERIOD_S", "LIBERO_GRIPPER_STALL", "gripper_outcome",
    "gripper_stalled", "merge_settled_receipt", "precontact_pose",
    "precontact_refusal", "public_gripper_command", "settle_refusal",
]

# LIBERO's BDDL environments run robosuite at control_freq=20
# (libero/envs/bddl_base_domain.py), so one controller tick is 50 ms.
LIBERO_CONTROL_PERIOD_S = 0.05
# Measured on LIBERO: a free-air close never moves less than 0.0034 of the
# normalized opening per tick before it arrives, while a close on an object
# drops below 0.0003 per tick once the fingers meet it.
GRIPPER_STALL_STEP = 0.001
GRIPPER_STALL_WINDOW = 3
GRIPPER_STALL_MINIMUM_TICKS = 5
# The shared rule reads this backend's measurements with its own strict
# arrival test; the thresholds and the trace (start sample plus one per tick)
# are unchanged from the LIBERO-only implementation.
LIBERO_GRIPPER_STALL = GripperStallRule(
    still_step=GRIPPER_STALL_STEP, window=GRIPPER_STALL_WINDOW,
    minimum_samples=max(GRIPPER_STALL_MINIMUM_TICKS, GRIPPER_STALL_WINDOW + 1),
    arrival_tolerance=GRIPPER_TOLERANCE_NORMALIZED,
    tick_period_s=LIBERO_CONTROL_PERIOD_S, tick_unit="osc_control_tick",
    arrival_comparison="<",
)


def settle_refusal(
    *, strategy: str | None, terminated: bool, motion_status: str,
    endpoint_measured: bool, residual_m: float | None,
    residual_rad: float | None, bound_m: float, bound_rad: float,
    within_local_envelope: bool,
) -> str | None:
    """Name the reason not to settle, or ``None`` to run the local segment.

    The global route is judged against its own 15 mm bound while the local
    route is judged against 5 mm. A global segment can therefore finish
    "reached" and still be several global tolerances away from what a local
    segment would accept. Re-driving the same joint goal cannot close that gap
    -- only task-space control can -- so the settle is the difference between
    reporting a residual and removing it.
    """

    if strategy != "global_then_local_settle":
        return "strategy_does_not_settle"
    if terminated:
        return "episode_terminated_before_settle"
    if motion_status == "refused":
        return "global_segment_refused"
    if not endpoint_measured or residual_m is None or residual_rad is None:
        return "measured_endpoint_unavailable"
    if residual_m <= bound_m and residual_rad <= bound_rad:
        return "global_route_met_the_local_bound"
    if not within_local_envelope:
        # A miss this large is not a settle. Requesting one anyway would be
        # rejected by the servo contract and would replace an honest endpoint
        # report with a contract violation.
        return "residual_outside_local_servo_envelope"
    return None


def precontact_pose(
    start: Sequence[float], target: Sequence[float],
) -> list[float]:
    """The pose a staged contact approach reaches globally before touching.

    It is RoboTwin's staged pose, reserved by LIBERO's own global endpoint
    bound: the global route may land anywhere inside that bound and the rest
    of the approach still fits the local servo envelope.
    """

    reserve_m, reserve_rad = libero_global_endpoint_tolerances()
    return _precontact_pose(
        start, target,
        local_translation_m=SERVO_MAX_TRANSLATION_M - reserve_m,
        local_rotation_rad=SERVO_MAX_ROTATION_RAD - reserve_rad,
    )


def precontact_refusal(
    *, terminated: bool, motion_status: str, endpoint_measured: bool,
    within_local_envelope: bool,
) -> str | None:
    """Name the reason not to make contact, or ``None`` to run the approach.

    Stricter than a settle: only a pre-contact segment that converged may hand
    off, because the next segment moves toward contact rather than closing a
    residual.
    """

    if terminated:
        return "episode_terminated_before_contact_approach"
    if motion_status == "refused":
        return "precontact_segment_refused"
    if motion_status != "reached":
        return "precontact_segment_not_reached"
    if not endpoint_measured:
        return "measured_endpoint_unavailable"
    if not within_local_envelope:
        return "residual_outside_local_servo_envelope"
    return None


def gripper_stalled(trace: Sequence[float], *, target: float) -> bool:
    """True once the jaws stopped moving short of their commanded opening.

    ``trace`` is the measured normalized opening at each tick so far. A close
    that meets an object stops within a few ticks and then only creeps; waiting
    out the whole budget changes nothing but the time spent.
    """

    return _shared_gripper_stalled(trace, target=target, rule=LIBERO_GRIPPER_STALL)


def gripper_outcome(
    *, state: str, target: float, measured: float, ticks_used: int,
    ticks_budget: int, maximum_aperture_m: float,
) -> dict[str, Any]:
    """What one jaw command did, in the units the policy compares against."""

    return _shared_gripper_outcome(
        rule=LIBERO_GRIPPER_STALL, state=state, target=target, measured=measured,
        ticks_used=ticks_used, ticks_budget=ticks_budget,
        extra={"measured_aperture_mm": float(measured) * float(maximum_aperture_m) * 1000.0},
    )


def merge_settled_receipt(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose one receipt from a global segment and its local settle.

    Both halves are real physical segments and both were persisted. The
    composed receipt reports the endpoint the policy is actually standing at,
    which is the second segment's, while keeping the first segment's start and
    the sum of what the two cost. ``segments`` carries both; ``segment`` stays
    the last one so existing readers keep working.
    """

    def total(field: str) -> Any:
        values = [part.get(field) for part in (first, second)]
        numbers = [value for value in values if isinstance(value, int)]
        return sum(numbers) if len(numbers) == len(values) else values[-1]

    merged = dict(second)
    merged["motion_strategy"] = first.get("motion_strategy")
    merged["before"] = first.get("before")
    merged["steps_used"] = total("steps_used")
    merged["sim_steps_total"] = total("sim_steps_total")
    merged["state_changed"] = bool(
        first.get("state_changed") or second.get("state_changed")
    )
    merged["segments"] = [
        *(first.get("segments") or []), *(second.get("segments") or []),
    ]
    return merged


def public_gripper_command(directive: float, target: str) -> dict[str, Any]:
    """Project the held gripper directive and the last accepted jaw target.

    ``directive`` is the value the executor compiles and the environment holds:
    -1 keeps the jaws open, +1 keeps them closed, and the normalized opening is
    the linear inverse. It is a measurement of what the controller is being
    told, which is exactly the number a policy needs beside the encoder to ask
    whether its own close took effect.

    ``target`` is a separate claim and stays ``unknown`` until a jaw command is
    actually accepted: a reset directive is a hold, not evidence that anything
    was commanded.
    """

    return {
        "gripper_command": (1.0 - float(directive)) / 2.0,
        "gripper_command_target": target,
    }
