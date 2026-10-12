"""Policy tool ``wait_ticks``: submit one shared motion-v3 ``wait`` of every current command.

The tool writes nothing but the action: no arm, pose or gripper is named, so
the Supervisor's executor holds each applied command and advances ``ticks``
control ticks (see ``contracts/wait.py``). Validation happens here, before the
action exists, with the same bound the action contract enforces.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA
from ora.contracts.wait import WAIT_MAX_TICKS, validate_wait_arguments

from .robot import ToolError

CAPABILITY = "shared_hold_wait_action_v1"
#: Not ``wait``: planner layers already say "wait for the next actual observation",
#: and prompt-bundle audits forbid a tool name in those layers.
TOOL = "wait_ticks"


def submit(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    host._ensure_mutation_available(TOOL)
    if not host.surface.supports(CAPABILITY) or not host.surface.supports("motion_intent_router_v1"):
        raise ToolError(f"{TOOL} is not selected on this surface", code="tool_unavailable", retryable=False)
    try:
        values = validate_wait_arguments(dict(arguments))
    except ValueError as exc:
        raise ToolError(str(exc), code="invalid_arguments", field="ticks", received=arguments.get("ticks"),
                        constraints={"minimum": 1, "maximum": WAIT_MAX_TICKS, "type": "integer"}) from exc
    control, _root, _observation, _pose, _gripper = host._current_grip_state()
    return host._atomic_submit(control=control, action_name="wait", arguments=values,
                               schema=MOTION_ACTION_V3_SCHEMA, public_action_name=TOOL)


__all__ = ["CAPABILITY", "TOOL", "submit"]
