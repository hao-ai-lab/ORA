"""How a closed skill's jaw stage ends on a sparse RoboDojo receipt.

Closing against an object need not reach the empty-jaw endpoint: the jaws
stall on the object (``gripper_stalled_before_target``) or, for receipts
recorded before the stall rule, spend their budget (``gripper_not_reached``).
Such a close continues to verification only with applied-command evidence;
an incomplete open must stop before exit.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["DOJO_SPARSE_RECEIPT_SCHEMA", "apply_dojo_gripper_stage"]

DOJO_SPARSE_RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"
_APPLIED_CLOSE_CODES = frozenset({"gripper_not_reached", "gripper_stalled_before_target"})


def apply_dojo_gripper_stage(record: dict[str, Any], execution: Mapping[str, Any],
                             targets: Mapping[str, str]) -> None:
    """Set the stage status/reason and carry each arm's measured jaw outcome."""
    applied_close = (
        bool(targets) and all(value == "close" for value in targets.values())
        and execution.get("failure_code") in _APPLIED_CLOSE_CODES
        and execution.get("execution_status") == "not_reached"
        and execution.get("native_actions", 0) > 0
        and all(execution.get("gripper_directives", {}).get(arm) == 0.0 for arm in targets)
    )
    outcomes = {
        arm: dict(effect["gripper_outcome"])
        for arm, effect in (execution.get("arms") or {}).items()
        if isinstance(effect, Mapping) and isinstance(effect.get("gripper_outcome"), Mapping)
    }
    if outcomes:
        record["measurements"].append({"quantity": "gripper_outcome", "value": outcomes})
    if not (execution.get("consumed") is True
            and execution.get("command_invariant", {}).get("verified") is True
            and (execution.get("execution_status") == "reached" or applied_close)):
        record.update(status="not_reached", reason=str(
            execution.get("failure_code") or "gripper_command_unavailable"))
    elif applied_close and execution.get("failure_code") == "gripper_stalled_before_target":
        # Executed, but say how: the jaws stopped on something.
        record["reason"] = "gripper_stalled_before_target"
