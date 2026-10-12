"""The shared ``wait`` motion action: hold every current command and let time pass.

A ``wait`` names no arm, pose or gripper. The backend keeps each drive target
and gripper directive it already applied -- nothing is re-planned, re-ramped
or re-limited -- and advances ``ticks`` control ticks before the Supervisor
captures a fresh observation. One tick is 10 physics steps: one RoboDojo
native action (the unit the task's native-action limit counts), or 10
RoboTwin physics steps; both are 40 ms at their 4 ms physics step. LIBERO
cannot split a robosuite control step, so its tick is one control step: one
``env.step`` of 25 physics steps of 2 ms, 50 ms (control_freq 20); the
receipt names the backend's own ``physics_steps_per_tick``. At most
20 ticks fit one call, the per-command native-action reservation of the
RoboDojo controller contract, so one ``wait`` never outlasts any other single
command.

What changes during a wait is whatever the world does on its own: a scripted
demonstrator or support arm advancing, an object settling or sliding, a held
object slipping. The receipt reports the measured drift; the action itself
claims nothing about it.
"""

from __future__ import annotations

from typing import Any, Mapping

WAIT_ACTION = "wait"
WAIT_MAX_TICKS = 20
WAIT_PHYSICS_STEPS_PER_TICK = 10
WAIT_RECEIPT_SCHEMA = "ora.wait-receipt.v1"


def validate_wait_arguments(arguments: Mapping[str, Any]) -> dict[str, int]:
    """``{"ticks": 1..20}`` exactly; anything else is refused before any write."""
    if set(arguments) != {"ticks"}:
        raise ValueError("motion v3 wait arguments must contain exactly ['ticks']")
    ticks = arguments["ticks"]
    if type(ticks) is not int or not 1 <= ticks <= WAIT_MAX_TICKS:
        raise ValueError(f"motion v3 wait ticks must be an integer in 1..{WAIT_MAX_TICKS}")
    return {"ticks": ticks}


def wait_receipt_block(*, ticks_requested: int, physics_steps: int, tick_unit: str,
                       tick_period_s: float, terminated_early: bool,
                       physics_steps_per_tick: int = WAIT_PHYSICS_STEPS_PER_TICK) -> dict[str, Any]:
    """The backend-neutral evidence one executed wait reports.

    ``physics_steps_per_tick`` is the backend's tick length in physics steps
    (10 on RoboDojo and RoboTwin, the LIBERO control step's 25 there).
    """
    if type(physics_steps_per_tick) is not int or physics_steps_per_tick < 1:
        raise ValueError("wait physics_steps_per_tick must be a positive integer")
    return {
        "schema": WAIT_RECEIPT_SCHEMA, "ticks_requested": int(ticks_requested),
        "ticks_used": -(-int(physics_steps) // physics_steps_per_tick),
        "physics_steps": int(physics_steps), "physics_steps_per_tick": physics_steps_per_tick,
        "tick_unit": tick_unit, "tick_period_s": float(tick_period_s),
        "terminated_early": bool(terminated_early),
        "held": "every_applied_drive_target_and_gripper_directive",
        "claim_boundary": "time passed with commands held; drift, settling, slip or scripted scene motion "
                          "are measured evidence in the fresh observation, not effects of this action",
    }


_PUBLIC_WAIT_KEYS = ("ticks_requested", "ticks_used", "physics_steps", "physics_steps_per_tick",
                     "tick_unit", "tick_period_s", "terminated_early", "held", "claim_boundary")


def public_wait_detail(record: Mapping[str, Any]) -> dict[str, Any]:
    """The policy-visible part of a receipt's ``wait`` block, or nothing."""
    block = record.get("wait")
    if not isinstance(block, Mapping) or block.get("schema") != WAIT_RECEIPT_SCHEMA:
        return {}
    return {"wait": {key: block[key] for key in _PUBLIC_WAIT_KEYS if key in block}}


__all__ = ["WAIT_ACTION", "WAIT_MAX_TICKS", "WAIT_PHYSICS_STEPS_PER_TICK", "WAIT_RECEIPT_SCHEMA",
           "public_wait_detail", "validate_wait_arguments", "wait_receipt_block"]
