"""Join immutable public placement command history, including closed stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence.receipt_binding import (
    canonical_public_receipt, public_actions_match,
)
from ora.contracts import target_state_phases, validate_runtime_action
from .robot import ToolError


def _stage_receipt(value: Any, run_root: Path | None) -> Mapping[str, Any] | None:
    """Use the shared resolver while retaining placement's integrity error."""
    if not isinstance(value, Mapping):
        return None
    wrapped = bool({"path", "sha256", "receipt"}.intersection(value))
    if wrapped and run_root is None:
        raise ToolError("placement receipt source identity drift", code="artifact_hash_mismatch", retryable=False)
    receipt = canonical_public_receipt({"previous_execution_receipt": value}, run_root)
    if wrapped and receipt is None:
        raise ToolError("placement receipt source identity drift", code="artifact_hash_mismatch", retryable=False)
    return receipt


def _command_states(states: list[Mapping[str, Any]], run_root: Path | None) -> list[Mapping[str, Any]]:
    """Read committed skill commands without changing their evidence identities.

    New stages record their action explicitly; historical target-state stages
    carry it inside the composite. Missing or mismatched children stay unknown
    instead of being discarded as if the command had never happened.
    """
    expanded: list[Mapping[str, Any]] = []

    def leaves(action: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        if receipt.get("action_id") != action.get("action_id") or receipt.get("consumed") is not True:
            raise ValueError("unjoined command receipt")
        schema = receipt.get("schema")
        if schema == "ora.composite-motion-receipt.v1":
            if not public_actions_match(action, receipt.get("action")):
                raise ValueError("composite action drift")
            phases = target_state_phases(action)
            records = receipt.get("phase_receipts")
            if not isinstance(records, list) or len(records) != len(phases):
                raise ValueError("incomplete composite")
            result = []
            for expected, item in zip(phases, records, strict=True):
                if not isinstance(item, Mapping) or not public_actions_match(expected, item.get("action")):
                    raise ValueError("phase action drift")
                if item.get("receipt") is None and item.get("status") in {
                    "skipped_environment_terminal", "skipped_motion_not_reached",
                }:
                    continue
                child = _stage_receipt(item.get("receipt"), run_root)
                if child is None:
                    raise ValueError("missing phase receipt")
                result.extend(leaves(expected, child))
            return result
        if (schema == "ora.manipulation-skill-sequence-receipt.v1"
                and action.get("schema") == "ora-composed-tool-action.v1"):
            if not public_actions_match(action, receipt.get("action")):
                raise ValueError("composed action drift")
            sequence = validate_runtime_action(action)["arguments"]["committed_sequence"]
            stages = receipt.get("stage_receipts")
            if not isinstance(stages, list) or len(stages) > len(sequence):
                raise ValueError("invalid stage sequence")
            result = []
            previous = -1
            for item in stages:
                if not isinstance(item, Mapping) or item.get("stage") not in sequence:
                    raise ValueError("stage order drift")
                index = sequence.index(item["stage"])
                if index <= previous:
                    raise ValueError("stage order drift")
                previous = index
                if item.get("action_id") is None:
                    continue
                base = f"{action['action_id']}:{index:02d}-{item['stage']}"
                for number, attempt in enumerate([*(item.get("attempts") or []), item], start=1):
                    child = _stage_receipt(attempt.get("receipt"), run_root)
                    child_action = attempt.get("action") or (child.get("action") if child else None)
                    expected_id = base if number == 1 else f"{base}.{number}"
                    if (not isinstance(child_action, Mapping) or child is None
                        or attempt.get("action_id") != expected_id or child_action.get("action_id") != expected_id
                        or child_action.get("observation_id") != action.get("observation_id")
                        or child_action.get("bound_observation_sha256") != action.get("bound_observation_sha256")):
                        raise ValueError("stage binding drift")
                    result.extend(leaves(child_action, child))
            return result
        if schema == "ora.manipulation-skill-sequence-receipt.v1":
            if not public_actions_match(action, receipt.get("action")):
                raise ValueError("skill action drift")
            checked = validate_runtime_action(action)
            args = checked["arguments"]
            sequence = args["sequence"][args["resume_from"]:]
            stages = receipt.get("stage_receipts")
            if not isinstance(stages, list) or len(stages) > len(sequence):
                raise ValueError("invalid stage sequence")
            if receipt.get("status") == "completed" and len(stages) != len(sequence):
                raise ValueError("missing completed stage")
            result = []
            for index, item in enumerate(stages, start=args["resume_from"]):
                if not isinstance(item, Mapping) or item.get("stage") != args["sequence"][index]:
                    raise ValueError("stage order drift")
                if item.get("action_id") is None:
                    if item.get("stage") not in {"verify", "stop"} and item.get("status") not in {"skipped", "refused"}:
                        raise ValueError("missing mutating stage")
                    continue
                child = _stage_receipt(item.get("receipt"), run_root)
                child_action = item.get("action") or (child.get("action") if child else None)
                expected_id = f"{action['action_id']}:{index:02d}-{item['stage']}"
                if (not isinstance(child_action, Mapping) or child is None
                    or item["action_id"] != expected_id or child_action.get("action_id") != expected_id
                    or child_action.get("observation_id") != action.get("observation_id")
                    or child_action.get("bound_observation_sha256") != action.get("bound_observation_sha256")):
                    raise ValueError("stage binding drift")
                result.extend(leaves(child_action, child))
            return result
        validate_runtime_action(action)
        return [(action, receipt)]

    for index, state in enumerate(states):
        action = state.get("previous_public_action")
        if not isinstance(action, Mapping) or action.get("action") not in {"execute_skill_sequence", "execute_composed_tool"}:
            expanded.append(state)
            continue
        receipt = _stage_receipt(state.get("previous_execution_receipt"), run_root)
        previous = states[index - 1].get("observation", {}) if index else {}
        try:
            if receipt is None:
                raise ValueError("missing skill receipt")
            commands = leaves(action, receipt)
        except (ValueError, TypeError, KeyError):
            expanded.append(state)  # Existing reader marks the unresolved envelope unknown.
            continue
        expanded.extend({**state, "previous_public_action": command,
                         "previous_execution_receipt": record,
                         "_skill_command": True, "_command_input_observation": previous}
                        for command, record in commands)
    return expanded
