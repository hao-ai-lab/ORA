"""Host-safe contract of the LIBERO ``wait`` hold and its receipt.

The isolated worker (``backends/libero_pro/wait.py``) executes the hold; the
Supervisor-side audits (``orchestrator/evaluation_learning.py`` and
``orchestrator/control_audit.py``) judge its receipt. Both read this module,
so the hold command, the command-invariant record and the receipt checks have
one owner and no simulator import.

LIBERO's OSC_POSE controller runs in delta mode (``libero_osc`` requires
``control_delta``): each control step's goal is the pose measured at that step
plus the commanded delta, so no pose target survives between steps. The hold
is the zero-delta 7-vector: a zero pose delta makes every step's goal the pose
measured at its start, and a zero gripper dimension leaves the Panda gripper's
stateful directive unchanged (robosuite integrates ``sign(action)``) -- the
value a pose action already sends in that dimension. The pose part is the
normalized input of a zero physical delta under the reset-bound OSC action
contract (``libero_osc.inverse_controller_scale``, the one owner of that
mapping): all zeros for the pinned symmetric ranges, never assumed.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ora.contracts.wait import WAIT_RECEIPT_SCHEMA
from ora.environment.controllers.libero_osc import inverse_controller_scale

LIBERO_WAIT_EXECUTOR = "libero_osc_zero_delta_hold"
LIBERO_WAIT_TICK_UNIT = "control_step"
LIBERO_WAIT_HELD_DIMENSION = "eef_pose_target_and_gripper_directive"
LIBERO_WAIT_POSE_COMMAND = "zero_osc_delta_from_measured_pose"
COMMAND_INVARIANT_SCHEMA = "ora.command-invariant.v2"
LIBERO_SPARSE_RECEIPT_SCHEMA = "ora.libero-sparse-motion-receipt.v1"
#: One LIBERO OSC command: three position deltas, three rotation deltas, gripper.
LIBERO_OSC_COMMAND_DIMENSIONS = 7


def libero_hold_command(contract: Mapping[str, Any]) -> list[float]:
    """The OSC_POSE command that holds the measured pose and the gripper directive.

    ``contract`` is the reset-bound OSC action contract; an invalid one raises.
    """

    pose = inverse_controller_scale([0.0] * 6, contract)
    return [*(float(value) for value in pose), 0.0]


def libero_wait_command_invariant(*, directive_before: float | None, directive_after: float | None,
                                  commands: list[list[float]], hold_command: list[float]) -> dict[str, Any]:
    """The command-invariant record of one wait: nothing controlled, both dimensions held."""

    held = (len(hold_command) == LIBERO_OSC_COMMAND_DIMENSIONS and hold_command[-1] == 0.0
            and all(list(command) == list(hold_command) for command in commands))
    return {
        "schema": COMMAND_INVARIANT_SCHEMA,
        "controlled_dimension": "none",
        "held_dimension": LIBERO_WAIT_HELD_DIMENSION,
        "pose_command": LIBERO_WAIT_POSE_COMMAND,
        "hold_command": list(hold_command),
        "directive_before": directive_before,
        "directive_after": directive_after,
        "action_dimension_command": 0.0,
        "commands_issued": len(commands),
        "verified": bool(
            held and directive_before is not None and directive_after == directive_before
        ),
    }


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _invariant_issues(invariant: Any, steps: int) -> list[str]:
    if not isinstance(invariant, Mapping) or invariant.get("schema") != COMMAND_INVARIANT_SCHEMA:
        return ["modern_command_invariant_missing_or_invalid"]
    if (invariant.get("controlled_dimension"), invariant.get("held_dimension")) != (
            "none", LIBERO_WAIT_HELD_DIMENSION):
        return ["recorded_command_invariant_violation"]
    before, after = invariant.get("directive_before"), invariant.get("directive_after")
    if not (_finite(before) and _finite(after)) or invariant.get("commands_issued") != steps:
        return ["modern_command_invariant_evidence_invalid"]
    if before != after or invariant.get("action_dimension_command") != 0.0 or invariant.get("verified") is not True:
        return ["recorded_command_invariant_violation"]
    return []


def libero_wait_receipt_issues(receipt: Mapping[str, Any]) -> list[str]:
    """Structural and command-invariant issues of one LIBERO wait receipt.

    Control health (schema, contract binding, fault) stays with each caller's
    existing health rule; this judges only what is specific to a wait.
    """

    steps = receipt.get("steps_used")
    arguments = receipt.get("arguments")
    ticks = arguments.get("ticks") if isinstance(arguments, Mapping) else None
    block = receipt.get("wait")
    issues: list[str] = []
    if (receipt.get("schema") != LIBERO_SPARSE_RECEIPT_SCHEMA or receipt.get("action") != "wait"
            or type(steps) is not int or type(ticks) is not int or not 0 <= steps <= ticks
            or receipt.get("consumed") is not True or receipt.get("state_changed") is not (steps > 0)
            or receipt.get("target_arms") != [] or receipt.get("executor") != [LIBERO_WAIT_EXECUTOR]
            or (receipt.get("synchronization") or {}).get("mode") != "hold"):
        return ["wait_receipt_contract_mismatch"]
    per_tick = block.get("physics_steps_per_tick") if isinstance(block, Mapping) else None
    if (not isinstance(block, Mapping) or block.get("schema") != WAIT_RECEIPT_SCHEMA
            or block.get("tick_unit") != LIBERO_WAIT_TICK_UNIT or type(per_tick) is not int or per_tick < 1
            or block.get("ticks_requested") != ticks or block.get("ticks_used") != steps
            or block.get("physics_steps") != steps * per_tick
            or receipt.get("physics_steps") != block.get("physics_steps")):
        issues.append("wait_receipt_contract_mismatch")
    return issues + _invariant_issues(receipt.get("command_invariant"), steps)


__all__ = [
    "COMMAND_INVARIANT_SCHEMA", "LIBERO_OSC_COMMAND_DIMENSIONS", "LIBERO_SPARSE_RECEIPT_SCHEMA",
    "LIBERO_WAIT_EXECUTOR", "LIBERO_WAIT_HELD_DIMENSION", "LIBERO_WAIT_POSE_COMMAND",
    "LIBERO_WAIT_TICK_UNIT", "libero_hold_command", "libero_wait_command_invariant",
    "libero_wait_receipt_issues",
]
