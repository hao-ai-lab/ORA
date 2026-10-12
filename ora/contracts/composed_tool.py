"""The action envelope that executes one committed composed tool.

A composed tool call is one robot decision. Like a closed-skill sequence, its
action carries the commitment and no geometry: the targets were derived at
call time, persisted beside the tool runtime's control state, and are replayed
by the trusted executor stage by stage. This module owns only the
transport-neutral shape; ``ora.agent.tools.composed`` owns what the
tool does.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

COMPOSED_TOOL_ACTION_SCHEMA = "ora-composed-tool-action.v1"
COMPOSED_TOOL_ACTION = "execute_composed_tool"
ARM_PROFILES = ("single_arm", "dual_arm")
#: One receipt reports one sequence: the closed-skill stage ceiling.
MAX_SEQUENCE_STAGES = 6
MAX_RATIONALE_CHARS = 200

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^c[0-9]+$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARGUMENTS = {"tool", "version", "arm_profile", "compiled_sha256", "rationale", "committed_sequence"}


def validate_composed_tool_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized envelope or raise ValueError naming the field."""

    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} must be non-empty")
    if payload.get("action") != COMPOSED_TOOL_ACTION:
        raise ValueError(f"a composed tool action must be {COMPOSED_TOOL_ACTION}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != _ARGUMENTS:
        raise ValueError(f"composed tool arguments must be exactly {sorted(_ARGUMENTS)}")
    if not isinstance(arguments["tool"], str) or not _NAME.match(arguments["tool"]):
        raise ValueError("tool must be a lowercase snake_case name")
    if not isinstance(arguments["version"], str) or not _VERSION.match(arguments["version"]):
        raise ValueError("version must look like c1")
    if arguments["arm_profile"] not in ARM_PROFILES:
        raise ValueError(f"arm_profile must be one of {list(ARM_PROFILES)}")
    if not isinstance(arguments["compiled_sha256"], str) or not _SHA.match(arguments["compiled_sha256"]):
        raise ValueError("compiled_sha256 must be sha256:<64 hex>")
    rationale = arguments["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale.strip()) > MAX_RATIONALE_CHARS:
        raise ValueError(f"rationale must be 1..{MAX_RATIONALE_CHARS} characters")
    sequence = arguments["committed_sequence"]
    if (not isinstance(sequence, list) or not 1 <= len(sequence) <= MAX_SEQUENCE_STAGES
            or len(set(sequence)) != len(sequence)
            or not all(isinstance(item, str) and _NAME.match(item) for item in sequence)):
        raise ValueError(f"committed_sequence must list 1..{MAX_SEQUENCE_STAGES} distinct stage names")
    return {
        "schema": COMPOSED_TOOL_ACTION_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": COMPOSED_TOOL_ACTION,
        "arguments": {
            "tool": arguments["tool"], "version": arguments["version"],
            "arm_profile": arguments["arm_profile"], "compiled_sha256": arguments["compiled_sha256"],
            "rationale": rationale.strip(), "committed_sequence": list(sequence),
        },
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }
