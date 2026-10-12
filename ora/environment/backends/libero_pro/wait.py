"""The shared ``wait`` action on the LIBERO sparse v3 route.

A wait names no arm, pose or gripper. LIBERO's OSC_POSE controller is a delta
controller and keeps no pose target between control steps, so the hold is the
zero-delta command (``controllers/libero_wait.py``; all zeros for the pinned
symmetric ranges): each step's goal is the pose measured at its start, and the
Panda gripper's stateful directive is left as it is -- the same gripper value
every pose action sends. Nothing is planned, clipped or re-limited; what the
arm or the scene does meanwhile is measured, not corrected:
``arms.right.inactive_arm_drift`` and the fresh observation.

One tick is one LIBERO control step, one ``env.step``: robosuite runs it as
``control_timestep / model_timestep`` physics steps, 0.05 s / 0.002 s = 25 at
the LIBERO BDDL environments' control_freq of 20. It is the unit of
``max_sim_steps`` and of the environment's own horizon. The executor reads the
live timing from the worker and refuses a wait whose control period is not
``LIBERO_CONTROL_PERIOD_S``, so the receipt and the published tick length
cannot silently disagree with the simulator. Each step latches the official
termination exactly as a motion step does.

The session files that compose, judge and persist LIBERO receipts are
hash-pinned by the Franka embodiment descriptors (``session-split-v1``
provenance), so this path enters from the unpinned execution-capture
``execute`` wrapper and uses the session's own helpers -- the outcome latch,
the robustness event, ``_persist_receipt`` -- rather than adding a branch to a
pinned file.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from ora.contracts import MOTION_ACTION_V3_SCHEMA
from ora.contracts.action import validate_runtime_action
from ora.contracts.wait import WAIT_ACTION, wait_receipt_block
from ora.contracts.workspace import workspace_state
from ora.environment.controllers.libero_osc import CONTROL_HEALTH_NUMERIC_TOLERANCE
from ora.environment.controllers.libero_wait import (
    LIBERO_SPARSE_RECEIPT_SCHEMA,
    LIBERO_WAIT_EXECUTOR,
    LIBERO_WAIT_TICK_UNIT,
    libero_hold_command,
    libero_wait_command_invariant,
)

from .common import CONTROL_HEALTH_SCHEMA, LiberoProError, _quaternion_error_rotvec
from .motion_settle import LIBERO_CONTROL_PERIOD_S

#: The replay controller the zero commands are re-stepped through.
REPLAY_CONTROLLER = "libero_osc_pose"
DIRECTIVE_CHANGED = "gripper_directive_changed_during_wait"
UNREADABLE_STATE = "wait_state_unreadable"
#: The local executor's gross-overrun bound (``local_motion_report``), in metres.
GROSS_WORKSPACE_OVERRUN_M = 0.05


def is_wait(payload: Mapping[str, Any]) -> bool:
    return payload.get("schema") == MOTION_ACTION_V3_SCHEMA and payload.get("action") == WAIT_ACTION


def wait_timing(session: Any) -> tuple[int, float]:
    """Physics steps per control step and the physics step, read from the worker."""

    timing = session._worker_attr("harness_servo_timing")
    per_tick = timing.get("native_quantum_physics_steps") if isinstance(timing, Mapping) else None
    physics_dt = timing.get("physics_dt_s") if isinstance(timing, Mapping) else None
    if (type(per_tick) is not int or per_tick < 1 or type(physics_dt) not in (int, float)
            or not math.isfinite(physics_dt) or physics_dt <= 0):
        raise LiberoProError("LIBERO wait timing is unavailable")
    if not math.isclose(per_tick * physics_dt, LIBERO_CONTROL_PERIOD_S, rel_tol=0.0, abs_tol=1e-9):
        raise LiberoProError(
            f"LIBERO control period {per_tick * physics_dt:g} s differs from the "
            f"{LIBERO_CONTROL_PERIOD_S:g} s wait tick"
        )
    return per_tick, float(physics_dt)


def _admit(session: Any, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Every refusal happens here, before any command or step."""

    session._require_environment()
    if session.is_terminal:
        raise LiberoProError("episode is already terminal")
    action = validate_runtime_action(payload)
    if not is_wait(action):
        raise LiberoProError("the LIBERO hold executor accepts only a motion v3 wait")
    if session.action_codec != "motion_action_v3":
        raise LiberoProError("wait requires the sparse motion_action_v3 codec")
    return action


def _measured(session: Any) -> dict[str, Any] | None:
    try:
        position, orientation, gripper = session._current_state()
        directive = session._gripper_hold_directive()
    except (LiberoProError, ValueError, FloatingPointError):
        return None
    if not all(math.isfinite(float(value)) for value in (*position, *orientation, gripper, directive)):
        return None
    return {"position": position, "orientation": orientation, "gripper": float(gripper),
            "directive": float(directive)}


def _issued_hold_command(session: Any) -> list[float]:
    """The hold command exactly as the float32 environment port receives it."""

    import numpy as np

    try:
        command = libero_hold_command(session.controller_action_contract)
    except ValueError as exc:
        raise LiberoProError(f"LIBERO OSC action contract is invalid: {exc}") from exc
    return np.asarray(command, dtype=np.float32).astype(np.float64).tolist()


def _hold(session: Any, environment: Any, action: Mapping[str, Any], ticks: int,
          hold_command: list[float]) -> list[list[float]]:
    """Step ``ticks`` control steps with the hold command; stop at any terminal."""

    import numpy as np

    commands: list[list[float]] = []
    for _ in range(ticks):
        command = np.asarray(hold_command, dtype=np.float32)
        commands.append(command.astype(np.float64).tolist())
        _obs, _reward, terminated, truncated, _info = environment.step(command[None])
        session._latch_environment_outcome(
            terminated=terminated, truncated=truncated, steps=1, action=action, controller="OSC_POSE",
        )
        if session.terminated or session.truncated or session.sim_steps >= session.max_sim_steps:
            session.truncated = session.truncated or session.sim_steps >= session.max_sim_steps
            break
    return commands


def _pose(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {"pose_xyz_xyzw": [*map(float, state["position"]), *map(float, state["orientation"])],
            "gripper": state["gripper"]}


def _drift(before: Mapping[str, Any], after: Mapping[str, Any] | None) -> dict[str, float] | None:
    import numpy as np

    if after is None:
        return None
    return {
        "translation_m": float(np.linalg.norm(np.asarray(after["position"]) - np.asarray(before["position"]))),
        "orientation_rad": float(np.linalg.norm(_quaternion_error_rotvec(
            np.asarray(after["orientation"], dtype=np.float64),
            np.asarray(before["orientation"], dtype=np.float64)))),
    }


def _health(session: Any, before: Mapping[str, Any], after: Mapping[str, Any] | None,
            verified: bool) -> dict[str, Any]:
    """LIBERO control health v2 of a hold, bound to the OSC contract it stepped through."""

    workspace_before = workspace_state([float(x) for x in before["position"]], session.workspace_bounds_m)
    workspace_after = (None if after is None
                       else workspace_state([float(x) for x in after["position"]], session.workspace_bounds_m))
    codes: list[str] = []
    if after is None:
        codes.append(UNREADABLE_STATE)
    elif not verified:
        codes.append(DIRECTIVE_CHANGED)
    if workspace_after is not None and workspace_after["max_violation_m"] > CONTROL_HEALTH_NUMERIC_TOLERANCE:
        codes.append("eef_outside_action_workspace")
        if workspace_after["max_violation_m"] > GROSS_WORKSPACE_OVERRUN_M + CONTROL_HEALTH_NUMERIC_TOLERANCE:
            codes.append("gross_workspace_overrun")
    fault = {UNREADABLE_STATE, DIRECTIVE_CHANGED, "gross_workspace_overrun"}.intersection(codes)
    return {
        "schema": CONTROL_HEALTH_SCHEMA,
        "status": "fault" if fault else "degraded" if codes else "healthy",
        "codes": codes,
        "controller_action_contract_sha256": session.controller_action_contract["contract_sha256"],
        "workspace_before": workspace_before,
        "workspace_after": workspace_after,
    }


def _outcome(health: Mapping[str, Any], *, steps: int, ticks: int, terminal: bool) -> tuple[str, Any, Any]:
    if health["status"] == "fault":
        return "execution_fault", "execution", health["codes"][0]
    if terminal and steps < ticks:
        return "not_reached", "execution", "episode_terminal_during_execution"
    return "reached", None, None


def execute_wait(session: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Hold for ``ticks`` LIBERO control steps; persist and return the sparse receipt."""

    action = _admit(session, payload)
    per_tick, physics_dt = wait_timing(session)
    environment = session._require_environment()
    ticks = int(action["arguments"]["ticks"])
    hold_command = _issued_hold_command(session)
    started = time.time()
    before = _measured(session)
    if before is None:
        raise LiberoProError("LIBERO state is unreadable before the wait")
    commands = _hold(session, environment, action, ticks, hold_command)
    steps = len(commands)
    after = _measured(session)
    invariant = libero_wait_command_invariant(
        directive_before=before["directive"], directive_after=None if after is None else after["directive"],
        commands=commands, hold_command=hold_command,
    )
    health = _health(session, before, after, invariant["verified"])
    terminal = bool(session.terminated or session.truncated)
    status, stage, code = _outcome(health, steps=steps, ticks=ticks, terminal=terminal)
    receipt = {
        "schema": LIBERO_SPARSE_RECEIPT_SCHEMA, "action_id": action["action_id"], "action": WAIT_ACTION,
        "arguments": dict(action["arguments"]), "consumed": True, "state_changed": steps > 0,
        "execution_status": status, "motion_strategy": None, "failure_stage": stage, "failure_code": code,
        "executor": [LIBERO_WAIT_EXECUTOR], "target_arms": [],
        "synchronization": {"mode": "hold", "requested": False, "executed": steps > 0},
        "arms": {"right": {"targeted": False, "interaction": None, "pose_error": None,
                           "endpoint_contract": None, "plan_validation": None,
                           "inactive_arm_drift": _drift(before, after)}},
        "before": {"right": _pose(before)}, "after": {"right": _pose(after)},
        "steps_used": steps, "sim_steps_total": session.sim_steps, "physics_steps": steps * per_tick,
        "command_invariant": invariant, "control_health": health,
        "wait": wait_receipt_block(
            ticks_requested=ticks, physics_steps=steps * per_tick, tick_unit=LIBERO_WAIT_TICK_UNIT,
            tick_period_s=per_tick * physics_dt, terminated_early=terminal and steps < ticks,
            physics_steps_per_tick=per_tick,
        ),
        "robustness_event": session._persist_new_robustness_event(),
        "terminal_after_action": terminal,
        "started_unix_s": started, "finished_unix_s": time.time(),
    }
    if health["status"] == "fault":
        session.control_fault_reason = "libero_control_health_fault"
    session.replay_recorder.record(
        action_sequence=session.action_sequence, action=action, controller=REPLAY_CONTROLLER,
        arrays={"commands": commands}, physics_steps=steps, replayable=bool(commands),
        metadata={"controller_action_contract_sha256": session.controller_action_contract["contract_sha256"],
                  "hold": invariant["pose_command"], "ticks_requested": ticks},
    )
    session.replay_recorder.update(session.action_sequence, execution_status=status,
                                   execution_fault=code if status == "execution_fault" else None)
    return session._persist_receipt(receipt)


__all__ = ["REPLAY_CONTROLLER", "execute_wait", "is_wait", "wait_timing"]
