"""The shared ``wait`` action on the RoboTwin sparse v3 motion-transaction route.

Nothing is written to any drive: SAPIEN keeps each joint and gripper drive
target it was last given, so stepping the scene holds every current command.
One tick is 10 physics steps (40 ms at the 4 ms RoboTwin physics step), the
same wall-clock unit as a RoboDojo native action. Each physics step runs the
official-success latch, so a wait stops at an official terminal exactly as a
motion does. The receipt's per-arm ``command_preservation`` digests prove the
drive targets did not change; ``inactive_arm_drift`` reports what moved anyway.
There is no controller sequence to replay, so the replay record says so.

``session.py`` is hash-pinned by the ALOHA embodiment descriptors (their
``robotwin_public_adapter`` provenance), so this path enters from the
execution-capture ``execute`` wrapper and drives the session's own sparse
helpers -- admission bookkeeping, decision count, settle/classify and receipt
composition -- rather than adding a branch to the pinned file. The receipt
composer predates ``wait``; its ``wait`` block and hold backend name are added
to the same receipt file before anything reads it.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Mapping

from ora.contracts.wait import WAIT_PHYSICS_STEPS_PER_TICK, wait_receipt_block

WAIT_CONTROLLER = "robotwin_hold_current_drive_targets"
HOLD_BACKEND = "hold_current_drive_targets"


def is_wait(payload: Mapping[str, Any]) -> bool:
    return payload.get("schema") == "ora-motion-action.v3" and payload.get("action") == "wait"


def hold_current_targets(session: Any, state: Any) -> None:
    """Step the scene ``ticks`` x 10 times without writing a target."""
    ticks = int(state.arguments["ticks"])
    environment = state.environment
    state.execution_attempted = True
    steps, terminal = 0, False
    for _ in range(ticks * WAIT_PHYSICS_STEPS_PER_TICK):
        environment.scene.step()
        steps += 1
        if session._after_physics_step(environment, action=state.action, phase="wait"):
            terminal = True
            break
    state.executed = steps > 0
    state.wait = wait_receipt_block(
        ticks_requested=ticks, physics_steps=steps, tick_unit="ten_physics_steps",
        tick_period_s=float(environment.scene.get_timestep()) * WAIT_PHYSICS_STEPS_PER_TICK,
        terminated_early=terminal and steps < ticks * WAIT_PHYSICS_STEPS_PER_TICK,
    )
    session.replay_recorder.record(
        action_sequence=session.action_sequence, action=state.action, controller=WAIT_CONTROLLER,
        arrays={}, physics_steps=steps, replayable=False,
        metadata={"reason": "hold_without_new_drive_targets", "ticks_requested": ticks},
    )
    state.replay_recorded = True


def execute_wait(session: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Admit, hold, classify and persist one wait with the session's own sparse helpers."""
    from ora.contracts import validate_runtime_action
    from .session import RoboTwinLiveError, _SparseMotionState, _atomic_json, _file_hash

    environment = session._require_environment()
    if session.is_terminal:
        raise RoboTwinLiveError("episode is already terminal")
    action = validate_runtime_action(payload)
    if not session._uses_motion_transaction_profile():
        raise RoboTwinLiveError("wait requires the motion transaction control profile")
    receipt_path = session.output_root / "receipts" / f"action-{session.action_sequence:04d}.json"
    if receipt_path.exists():
        raise RoboTwinLiveError(f"refusing to overwrite receipt: {receipt_path}")
    state = _SparseMotionState(
        action=action, receipt_path=receipt_path, started=time.time(), environment=environment,
        grip_center_profile=session._uses_grip_center_profile(), arguments=action["arguments"],
        targets={}, target_arms=(),
    )
    state.wait = None
    session._route_motion_strategy(state)
    session._begin_sparse_decision(state.environment)
    if session.operator_execution_video:
        session._capture_operator_execution_frame(action=action, phase="action_start", control_step_index=None)
    try:
        hold_current_targets(session, state)
    except Exception as exc:
        session._classify_sparse_exception(state, exc)
    session._combine_blocked_recovery_prefix(state)
    session._persist_motion_artifacts(state)
    session._settle_and_classify(state)
    record = session._compose_sparse_receipt(state)
    receipt = deepcopy(record["receipt"])
    if state.wait is not None:
        receipt["wait"] = state.wait
    if "motion_backend" in receipt:
        receipt["motion_backend"] = HOLD_BACKEND
    _atomic_json(receipt_path, receipt)
    record = {"path": record["path"], "sha256": _file_hash(receipt_path), "receipt": receipt}
    session.last_receipt = record
    return record


__all__ = ["HOLD_BACKEND", "WAIT_CONTROLLER", "execute_wait", "hold_current_targets", "is_wait"]
