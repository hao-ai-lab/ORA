"""Measured drift from the preceding physically issued pose target."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping


def _position(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in value):
        return None
    return [float(x) for x in value]


class CommandDriftTracker:
    """Session-local measured evidence; refusals cannot replace a command.

    Call only for pose actions, after physical execution. Repeated persistence
    of the same action is idempotent. Reset by creating a fresh tracker with
    the environment episode. Contact causation is deliberately not inferred.
    """

    def __init__(self) -> None:
        self.targets: dict[str, list[float]] = {}
        self.previous_drift: dict[str, list[float]] = {}
        self.last_action_id: str | None = None
        self.last_result: dict[str, Any] = {}

    def record(self, *, action_id: str,
               arms: Mapping[str, Mapping[str, Any]],
               steps_used: int) -> dict[str, Any]:
        if action_id == self.last_action_id:
            return copy.deepcopy(self.last_result)
        if steps_used <= 0:
            return {}
        result: dict[str, Any] = {}
        for arm, state in arms.items():
            start = _position(state.get("start_position_m"))
            final = _position(state.get("final_position_m"))
            commanded = _position(state.get("commanded_position_m"))
            if start is None or final is None or commanded is None:
                # A missing executed pose target breaks the causal chain.
                self.targets.pop(arm, None)
                self.previous_drift.pop(arm, None)
                continue
            previous_target = self.targets.get(arm)
            position_held = all(abs(x - y) <= 1e-9 for x, y in zip(commanded, start))
            if previous_target is not None:
                start_drift = [x - y for x, y in zip(start, previous_target)]
                drift = [x - y for x, y in zip(final, previous_target)]
                prior = self.previous_drift.get(arm)
                norm = math.sqrt(sum(x * x for x in drift))
                prior_norm = math.sqrt(sum(x * x for x in prior)) if prior else 0.0
                accumulating = (
                    position_held and norm > .01 and prior_norm > .01 and prior is not None
                    and sum(x * y for x, y in zip(drift, prior))
                    / (norm * prior_norm) > .5
                )
                result[arm] = {
                    "drift_from_last_commanded_m": drift,
                    "start_drift_from_last_commanded_m": start_drift,
                    "drift_accumulating": accumulating,
                    "drift_frame": "world",
                }
                if position_held:
                    self.previous_drift[arm] = drift
                else:
                    self.previous_drift.pop(arm, None)
            self.targets[arm] = commanded
        self.last_action_id = action_id
        self.last_result = result
        return copy.deepcopy(result)
