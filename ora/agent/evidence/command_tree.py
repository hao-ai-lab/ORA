"""Join trusted composite/skill commands to consumed worker receipt leaves."""
from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.action import target_state_phases
from .receipt_binding import public_worker_arguments_match


def command_leaves(action: Mapping[str, Any], receipt: Mapping[str, Any], resolve):
    if receipt.get("action_id") != action.get("action_id") or receipt.get("consumed") is not True:
        raise ValueError("receipt parent mismatch")
    schema = receipt.get("schema")
    if schema == "ora.composite-motion-receipt.v1":
        if receipt.get("action") != action:
            raise ValueError("composite action mismatch")
        phases = target_state_phases(action)
        records = receipt.get("phase_receipts", [])
        if len(records) != len(phases):
            raise ValueError("incomplete composite phases")
        for phase_index, (expected, record) in enumerate(zip(phases, records)):
            if record.get("action") != expected:
                raise ValueError("composite phase mismatch")
            status = record.get("status", "")
            if status.startswith("skipped_"):
                if record.get("receipt") is not None:
                    raise ValueError("skipped phase has receipt")
                if status == "skipped_motion_not_reached":
                    if (expected.get("action") != "set_gripper" or phase_index == 0
                            or records[phase_index - 1].get("status") != "not_reached"):
                        raise ValueError("skipped gripper lacks failed pose prefix")
                elif status != "skipped_environment_terminal":
                    raise ValueError("unknown skipped phase")
                continue
            child = resolve(record.get("receipt"))
            if not isinstance(child, Mapping):
                raise ValueError("missing composite child")
            yield from command_leaves(expected, child, resolve)
    elif schema == "ora.manipulation-skill-sequence-receipt.v1":
        if receipt.get("action") != action or receipt.get("committed_sequence") != action.get("arguments", {}).get("sequence"):
            raise ValueError("skill commitment mismatch")
        arguments = action["arguments"]
        previous = arguments["resume_from"] - 1
        records = receipt.get("stage_receipts", [])
        if not isinstance(records, list):
            raise ValueError("missing skill stages")
        if receipt.get("status") == "completed" and len(records) != len(arguments["sequence"]) - arguments["resume_from"]:
            raise ValueError("completed skill lacks stages")
        dispatched = sum(record.get("action_id") is not None for record in records)
        if receipt.get("dispatched_stage_actions", dispatched) != dispatched:
            raise ValueError("skill dispatch count mismatch")
        for record in records:
            stage = record.get("stage")
            if stage not in arguments["sequence"]:
                raise ValueError("uncommitted stage")
            index = arguments["sequence"].index(stage)
            if index != previous + 1:
                raise ValueError("missing or reordered stage")
            previous = index
            if not record.get("action_id"):
                if record.get("receipt") is not None or record.get("action") is not None:
                    raise ValueError("anonymous stage has command evidence")
                continue
            child = resolve(record.get("receipt"))
            expected_id = f"{action['action_id']}:{index:02d}-{stage}"
            if not isinstance(child, Mapping) or record["action_id"] != expected_id:
                raise ValueError("skill child identity mismatch")
            child_action = record.get("action", child.get("action"))
            if not isinstance(child_action, Mapping) or child_action.get("action_id") != expected_id or any(
                child_action.get(key) != action.get(key) for key in ("observation_id", "bound_observation_sha256")
            ):
                raise ValueError("skill child observation drift")
            yield from command_leaves(child_action, child, resolve)
    else:
        if receipt.get("action") != action.get("action") or not public_worker_arguments_match(action, receipt):
            raise ValueError("command arguments mismatch")
        yield action, receipt
