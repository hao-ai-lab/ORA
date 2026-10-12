"""Measured progress helpers shared by motion executors and public receipts."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def classify_stall_shape(stage_index: int, planned_stages: int,
                         waypoint_wait_steps: Sequence[int]) -> str | None:
    """Describe the measured wait prefix; it does not infer physical contact."""
    if planned_stages > 0 and stage_index / planned_stages < .1:
        return "from_start"
    prefix = list(waypoint_wait_steps[:stage_index])
    if sum(wait > 4 for wait in prefix) >= 3:
        return "creeping"
    if len(prefix) >= 12 and all(wait <= 2 for wait in prefix[:12]):
        return "abrupt"
    return None


class JointBlockedMonitor:
    """Count physical samples frozen against one unchanged joint target.

    A target change resets its arm's count. The reference for joint motion is
    always the preceding actual sample, including the action-start sample.
    """

    def __init__(self, initial_joints: Mapping[str, Sequence[float]], *,
                 consecutive_physics_steps: int, maximum_joint_step_rad: float,
                 reset_on_target_change: bool = True):
        self.previous = {arm: np.asarray(joints, dtype=np.float64)
                         for arm, joints in initial_joints.items()}
        self.targets: dict[str, np.ndarray] = {}
        self.counts = {arm: 0 for arm in initial_joints}
        self.required = consecutive_physics_steps
        self.maximum_step = maximum_joint_step_rad
        self.reset_on_target_change = reset_on_target_change

    def sample(self, measured: Mapping[str, Sequence[float]],
               targets: Mapping[str, Sequence[float]], *, tolerance_rad: float,
               stage_index: int, planned_stages: int,
               waypoint_wait_steps: Sequence[int]) -> dict[str, Any] | None:
        blocked = []
        errors = {}
        for arm, joints in measured.items():
            actual = np.asarray(joints, dtype=np.float64)
            prior = self.previous[arm]
            target = np.asarray(targets[arm], dtype=np.float64) if arm in targets else None
            if target is None:
                self.counts[arm] = 0
            else:
                previous_target = self.targets.get(arm)
                if self.reset_on_target_change and (previous_target is None or not np.array_equal(previous_target, target)):
                    self.counts[arm] = 0
                error = float(np.max(np.abs(actual - target)))
                step = float(np.max(np.abs(actual - prior)))
                self.counts[arm] = self.counts[arm] + 1 if (
                    error > tolerance_rad and step < self.maximum_step
                ) else 0
                self.targets[arm] = target.copy()
                if self.counts[arm] >= self.required:
                    blocked.append(arm)
                    errors[arm] = round(error, 8)
            self.previous[arm] = actual.copy()
        if not blocked:
            return None
        return {
            "code": "blocked", "stage_index": stage_index,
            "blocked_at_stage": stage_index, "arms": blocked,
            "frozen_steps": max(self.counts[arm] for arm in blocked),
            "progress": round(stage_index / planned_stages, 8),
            "stall_shape": classify_stall_shape(stage_index, planned_stages, waypoint_wait_steps),
            "joint_tracking_error_rad": errors,
            "maximum_frozen_joint_step_rad": self.maximum_step,
            "cause_evidence": ("measured_joint_stall_at_unchanged_target" if self.reset_on_target_change
                               else "measured_joint_stall_despite_requested_motion"),
        }
