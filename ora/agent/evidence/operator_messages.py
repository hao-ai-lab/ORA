"""Validate the human-input boundary without trusting UI acknowledgement files."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def operator_trace_errors(events: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    pending_tools: set[Any] = set()
    pending_action = False
    commands: set[str] = set()
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind == "tool_call_proposed":
            pending_tools.add(event.get("tool_call_id"))
        elif kind == "tool_result":
            pending_tools.discard(event.get("tool_call_id"))
        elif kind == "action_proposed":
            pending_action = True
        elif kind == "fresh_observation_published":
            pending_action = False
        elif kind == "operator_message":
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                errors.append(f"event:{index}:operator_message_contract")
                continue
            command_id = payload.get("id")
            if (
                plan.get("operator_control", {}).get("mode") != "file_inbox_v1"
                or plan.get("experiment") != "ora-interactive"
                or payload.get("schema") != "ora.ora-operator-command.v1"
                or payload.get("type") not in {"steer", "follow_up"}
                or not isinstance(command_id, str) or not command_id
                or command_id in commands
                or payload.get("run_id") != event.get("run_id")
                or payload.get("plan_sha256") != plan.get("plan_sha256")
                or type(payload.get("submitted_unix_ns")) is not int
                or type(payload.get("round_index")) is not int
                or type(payload.get("observation_step")) is not int
                or payload.get("observation_step") != event.get("observation_step")
                or not isinstance(payload.get("before_model_call_id"), str)
                or not isinstance(payload.get("message"), str) or not payload["message"].strip()
                or event.get("visibility") != "exact"
            ):
                errors.append(f"event:{index}:operator_message_contract")
            if pending_tools or pending_action:
                errors.append(f"event:{index}:operator_message_before_tool_completion")
            if isinstance(command_id, str):
                commands.add(command_id)
    return errors
