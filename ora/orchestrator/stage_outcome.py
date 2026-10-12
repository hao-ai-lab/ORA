"""Classify one dispatched stage action: did it do what it was sent to do?

Closed-skill sequences (``action_executor._execute_skill_sequence``) and
composed tools (``composed_tool_job.execute_composed_tool``) both classify
through here, so the two cannot disagree about what an arrival is.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts import TARGET_STATE_ACTION_SCHEMA

from .action_executor import _ARRIVAL_CHECKED_ACTIONS, _motion_reached, _record
from .gripper_stage import DOJO_SPARSE_RECEIPT_SCHEMA, apply_dojo_gripper_stage
from .motion_stage import dojo_motion_stage_evidence


def classify_stage_action(
    record: dict[str, Any], stage_action: Mapping[str, Any], stage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Mark ``record`` (which starts as executed) with its arrival or jaw evidence.

    A pose stage gains its endpoint residual (and RoboDojo motion progress)
    and becomes ``not_reached`` when it did not arrive; a gripper stage
    carries RoboDojo jaw evidence or becomes ``unavailable`` when the backend
    did not execute it.
    """

    pose_stage = stage_action["action"] in _ARRIVAL_CHECKED_ACTIONS or (
        stage_action["schema"] == TARGET_STATE_ACTION_SCHEMA
        and "pose" in stage_action["arguments"]
    )
    if pose_stage:
        reached, residual = _motion_reached(stage_receipt)
        residual, stopped = dojo_motion_stage_evidence(_record(stage_receipt), residual)
        record["pose_residual"] = residual
        record["measurements"].extend([{"quantity": "endpoint_residual", "value": residual}, *stopped])
        if not reached:
            record["status"] = "not_reached"
            record["reason"] = str(
                residual.get("reason") or "endpoint_residual_exceeded"
            )
    elif stage_action["action"] == "set_gripper":
        execution = _record(stage_receipt)
        if execution.get("schema") == DOJO_SPARSE_RECEIPT_SCHEMA:
            apply_dojo_gripper_stage(record, execution, stage_action["arguments"].get("targets", {}))
        elif "execution_status" in execution and execution["execution_status"] != "executed":
            record.update(status="unavailable", reason=str(
                execution.get("failure_code") or execution["execution_status"]
            ))
    return record
