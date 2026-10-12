"""Host-safe jaw-stall rule and outcome shared by every backend's gripper loop.

A jaw command that meets an object stops short of its commanded opening and
then only creeps. Waiting out the command budget changes nothing but the time
spent, and reporting the result as a generic miss hides the one fact a policy
needs: the jaws stopped on something. Each backend owns its measured
thresholds and its control tick (``GripperStallRule``); the decision, the
outcome states and the claim boundary are implemented once here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

__all__ = [
    "GRIPPER_OUTCOME_STATES", "GRIPPER_STALL_CLAIM_BOUNDARY", "GRIPPER_STALL_REASON",
    "GripperStallRule", "gripper_outcome", "gripper_outcome_state", "gripper_stalled",
]

GRIPPER_STALL_REASON = "gripper_stalled_before_target"
GRIPPER_OUTCOME_STATES = (
    "reached_target", "stalled_before_target", "moving_at_budget", "interrupted",
)
GRIPPER_STALL_CLAIM_BOUNDARY = (
    "stalled_before_target means the jaws stopped short of the commanded "
    "opening; it is consistent with an object between the fingers and is "
    "not proof of a grasp"
)


@dataclass(frozen=True)
class GripperStallRule:
    """One backend's measured stall thresholds, in its own control ticks.

    ``trace`` samples are normalized openings: the measurement at command
    start followed by one sample after every control tick. A stall needs at
    least ``minimum_samples`` samples, an opening still farther than
    ``arrival_tolerance`` from the target, and ``window`` consecutive ticks
    that each moved less than ``still_step`` of the normalized opening.
    ``arrival_comparison`` is the backend's own arrival test, so a stall is
    never reported for an opening that backend already counts as arrived.
    """

    still_step: float
    window: int
    minimum_samples: int
    arrival_tolerance: float
    tick_period_s: float
    tick_unit: str
    arrival_comparison: str = "<="

    def __post_init__(self) -> None:
        numbers = (self.still_step, self.arrival_tolerance, self.tick_period_s)
        if not all(math.isfinite(value) and value > 0 for value in numbers):
            raise ValueError("gripper stall thresholds and tick period must be positive")
        if self.window < 1 or self.minimum_samples < self.window + 1:
            raise ValueError("a stall window needs window + 1 samples")
        if not self.tick_unit:
            raise ValueError("a stall rule must name its control tick")
        if self.arrival_comparison not in {"<", "<="}:
            raise ValueError("arrival comparison must be '<' or '<='")

    def arrived(self, measured: float, target: float) -> bool:
        residual = abs(float(measured) - float(target))
        if self.arrival_comparison == "<":
            return residual < self.arrival_tolerance
        return residual <= self.arrival_tolerance


def gripper_stalled(trace: Sequence[float], *, target: float, rule: GripperStallRule) -> bool:
    """True once the jaws stopped moving short of their commanded opening."""

    if len(trace) < rule.minimum_samples:
        return False
    if rule.arrived(trace[-1], target):
        return False
    recent = [float(value) for value in trace[-(rule.window + 1):]]
    return all(abs(b - a) < rule.still_step for a, b in zip(recent, recent[1:]))


def gripper_outcome_state(
    *, reached: bool, stalled: bool, ticks_used: int, ticks_budget: int,
) -> str:
    """Name how one jaw command ended; arrival always wins over a stall."""

    if reached:
        return "reached_target"
    if stalled:
        return "stalled_before_target"
    if ticks_used >= ticks_budget:
        return "moving_at_budget"
    return "interrupted"


def gripper_outcome(
    *, rule: GripperStallRule, state: str, target: float, measured: float,
    ticks_used: int, ticks_budget: int, extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What one jaw command did, in the units the policy compares against.

    ``extra`` carries backend-owned measurements (for example a calibrated
    aperture, or an SDK-limited drive target) and is placed after the two
    openings so every backend's outcome reads in the same order.
    """

    if state not in GRIPPER_OUTCOME_STATES:
        raise ValueError(f"unknown gripper outcome state: {state!r}")
    return {
        "state": state,
        "commanded_opening_normalized": float(target),
        "measured_opening_normalized": float(measured),
        **dict(extra or {}),
        "ticks_used": int(ticks_used),
        "ticks_budget": int(ticks_budget),
        "seconds_used": int(ticks_used) * rule.tick_period_s,
        "claim_boundary": GRIPPER_STALL_CLAIM_BOUNDARY,
    }
