"""The shared ``wait`` action on the RoboDojo sparse v3 route.

Every applied SDK command is held: the native ticks run with an empty command,
and ``bridge._fixed_meta_control`` restores each dimension the command does not
name, so no drive target, gripper target or limiter is recomputed. Each tick
is one native action, the unit the task's native-action limit counts; an
interactive scene's scripted demonstrator or support arms advance exactly as
many native actions as the wait consumes. The per-physics-step journal still
verifies that every applied target equals the held one.
"""

from __future__ import annotations

from typing import Any

from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
from ora.contracts.wait import wait_receipt_block

WAIT_EXECUTOR = "robodojo_sdk_hold_applied_commands"


def execute_wait(execution: Any) -> dict[str, Any]:
    """Advance ``ticks`` native actions holding every command; report measured drift."""
    session = execution.session
    ticks = int(execution.action["arguments"]["ticks"])
    execution.begin([])
    used = session._run_native_control(
        action_id=str(execution.action["action_id"]), control={}, held_targets=execution.held_targets,
        journal=execution.journal, tracking=execution.tracking, segment="wait",
        reached=lambda _current: False, budget=min(ticks, execution.remaining()),
    )
    execution.used += used
    terminal = bool(session.is_terminal)
    block = wait_receipt_block(
        ticks_requested=ticks, physics_steps=used * int(ARX_X5_CONTRACT["physics_steps_per_native_action"]),
        tick_unit="native_action", tick_period_s=1.0 / float(ARX_X5_CONTRACT["control_hz"]),
        terminated_early=terminal and used < ticks,
    )
    return execution.finish(
        status="not_reached" if terminal and used < ticks else "reached", strategy=None,
        stage="execution" if terminal and used < ticks else None,
        code="episode_terminal_during_execution" if terminal and used < ticks else None,
        executor=[WAIT_EXECUTOR], extra={"wait": block},
    )


__all__ = ["WAIT_EXECUTOR", "execute_wait"]
