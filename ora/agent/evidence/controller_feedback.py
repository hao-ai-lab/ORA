"""Public source-bound controller and gripper receipt details.

No constants are inferred from the current backend or a historical schema.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
from ora.contracts.gripper_feedback import validate_gripper_feedback, validate_controller_feedback


def public_gripper_detail(record: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Preserve producer-owned intent/drive/encoder distinctions through delivery."""
    if not context.supports("current_public_robot_facts_v1"):
        return {}
    result = {}
    if "controller_feedback" in record:
        result["controller_feedback"] = validate_controller_feedback(record["controller_feedback"])
    after = record.get("after")
    if "controller_feedback" in record and (
        not isinstance(after, Mapping) or any(
            not isinstance(after.get(arm), Mapping) or "gripper_feedback" not in after[arm]
            for arm in ("left", "right")
        )
    ):
        raise ValueError("controller feedback requires both arm gripper snapshots")
    grippers = {}
    if isinstance(after, Mapping):
        for arm in ("left", "right"):
            state = after.get(arm)
            if isinstance(state, Mapping) and "gripper_feedback" in state:
                grippers[arm] = validate_gripper_feedback(state["gripper_feedback"])
    if grippers:
        result["gripper_feedback"] = grippers
    targets = record.get("gripper_targets_effective")
    if isinstance(targets, Mapping):
        result["gripper_targets_effective"] = {
            arm: float(targets[arm]) for arm in ("left", "right")
            if arm in targets and type(targets[arm]) in (int, float) and math.isfinite(targets[arm])
        }
    if type(record.get("native_action_budget")) is int and type(record.get("native_actions")) is int:
        result["execution_window"] = {"native_actions_used": record["native_actions"],
                                      "native_action_budget": record["native_action_budget"]}
    return result
