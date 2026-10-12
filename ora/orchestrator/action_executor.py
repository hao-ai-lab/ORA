"""Trusted execution adapter for atomic v3 and staged target-state v4 actions."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.contracts import (
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
)
from ora.environment import (
    EnvironmentSession, TARGET_STATE_ACTION_SCHEMA, target_state_phases,
    validate_runtime_action,
)
from ora.contracts import SKILL_SEQUENCE_ACTION_SCHEMA


def _record(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = receipt.get("receipt")
    return nested if isinstance(nested, Mapping) else receipt


def _motion_reached(receipt: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    record = _record(receipt)
    if record.get("schema") == "ora.composite-motion-receipt.v1":
        phases = record.get("phase_receipts")
        if isinstance(phases, list):
            pose_phases = [phase for phase in phases if phase.get("phase") == "pose"]
            if len(pose_phases) != 1 or not isinstance(pose_phases[0].get("receipt"), Mapping):
                return False, {"reason": "receipt_lacks_pose_evidence"}
            return _motion_reached(pose_phases[0]["receipt"])
        residual = record.get("pose_residual")
        return record.get("pose_reached") is True, (
            dict(residual) if isinstance(residual, Mapping)
            else {"reason": "composite_lacks_pose_evidence"}
        )
    if record.get("schema") in {
        "ora.robotwin-sparse-motion-receipt.v7",
        "ora.robodojo-sparse-motion-receipt.v1",
        "ora.libero-sparse-motion-receipt.v1",
    }:
        # The selected controller owns its endpoint thresholds and per-arm
        # checks. Do not reinterpret planner success as physical arrival.
        errors = {}
        arms = record.get("arms", {})
        for arm, detail in arms.items() if isinstance(arms, Mapping) else ():
            error = detail.get("pose_error") if isinstance(detail, Mapping) else None
            if arm not in {"left", "right"} or not isinstance(error, Mapping):
                continue
            errors[arm] = {
                key: float(value) for key, value in error.items()
                if key in {"position_m", "translation_m", "orientation_rad"}
                and not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value)
            }
        return record.get("execution_status") == "reached" and record.get("consumed") is True, {
            "reason": str(record.get("failure_code") or record.get("not_reached_reason")
                          or record.get("execution_status") or "receipt_lacks_execution_status"),
            **({"errors_by_arm": errors} if errors else {}),
        }
    commanded = record.get("commanded_pose_xyz_xyzw")
    reached = record.get("reached_pose_xyz_xyzw")
    if not (
        isinstance(commanded, list) and len(commanded) == 7
        and isinstance(reached, list) and len(reached) == 7
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in [*commanded, *reached]
        )
    ):
        return False, {"reason": "receipt_lacks_pose_evidence"}
    translation = math.sqrt(sum(
        (float(commanded[index]) - float(reached[index])) ** 2
        for index in range(3)
    ))
    commanded_norm = math.sqrt(sum(
        float(commanded[index]) ** 2 for index in range(3, 7)
    ))
    reached_norm = math.sqrt(sum(
        float(reached[index]) ** 2 for index in range(3, 7)
    ))
    if commanded_norm <= 1e-12 or reached_norm <= 1e-12:
        return False, {"reason": "receipt_lacks_normalized_pose_evidence"}
    dot = abs(sum(
        float(commanded[index]) * float(reached[index])
        for index in range(3, 7)
    ) / (commanded_norm * reached_norm))
    orientation = 2.0 * math.acos(min(1.0, max(0.0, dot)))
    return (
        translation <= POSE_POSITION_TOLERANCE_M
        and orientation <= POSE_ORIENTATION_TOLERANCE_RAD
    ), {
        "translation_residual_m": translation,
        "orientation_residual_rad": orientation,
        "thresholds": {
            "translation_m": POSE_POSITION_TOLERANCE_M,
            "orientation_rad": POSE_ORIENTATION_TOLERANCE_RAD,
        },
    }


@contextmanager
def _action_sequence(runtime: EnvironmentSession, action: Mapping[str, Any]):
    """Keep observation-consuming backends bound to one trusted parent."""
    begin = getattr(runtime, "begin_action_sequence", None)
    end = getattr(runtime, "end_action_sequence", None)
    if not callable(begin):
        yield
        return
    if not callable(end):
        raise ValueError("action sequence extension requires cleanup")
    begin(action)
    try:
        yield
    finally:
        end(str(action["action_id"]))


def execute_runtime_action(
    runtime: EnvironmentSession, payload: Mapping[str, Any], *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    action = validate_runtime_action(payload)
    if action["schema"] == TARGET_STATE_ACTION_SCHEMA:
        with _action_sequence(runtime, action):
            return _execute_runtime_action(runtime, action, should_stop=should_stop)
    return _execute_runtime_action(runtime, action, should_stop=should_stop)


def _execute_runtime_action(
    runtime: EnvironmentSession, payload: Mapping[str, Any], *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one validated proposal through its trusted action boundary."""

    action = validate_runtime_action(payload)
    if action["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA:
        raise ValueError(
            "a committed skill sequence requires the coordinator-owned stage loop"
        )
    if action["schema"] != TARGET_STATE_ACTION_SCHEMA:
        return runtime.execute(action)
    started = time.time()
    phases = target_state_phases(action)
    phase_receipts: list[dict[str, Any]] = []
    pose_reached = True
    residual: dict[str, Any] | None = None
    for phase in phases:
        environment_status = runtime.status()
        if environment_status.get("terminal") is True:
            phase_receipts.append({
                "phase": "pose" if phase["action"] == "move_pose" else "gripper",
                "status": "skipped_environment_terminal",
                "action": phase,
                "receipt": None,
                "terminal_reason": environment_status.get("reason"),
            })
            continue
        if phase["action"] == "set_gripper" and not pose_reached:
            phase_receipts.append({
                "phase": "gripper",
                "status": "skipped_motion_not_reached",
                "action": phase,
                "receipt": None,
            })
            continue
        receipt = runtime.execute(phase)
        status = "executed"
        if phase["action"] == "move_pose":
            pose_reached, residual = _motion_reached(receipt)
            status = "reached" if pose_reached else "not_reached"
        phase_receipts.append({
            "phase": "pose" if phase["action"] == "move_pose" else "gripper",
            "status": status,
            "action": phase,
            "receipt": receipt,
        })
    return {
        "schema": "ora.composite-motion-receipt.v1",
        "action_id": action["action_id"],
        "action": action,
        "consumed": True,
        "pose_reached": pose_reached if "pose" in action["arguments"] else None,
        "pose_residual": residual,
        "phase_receipts": phase_receipts,
        "started_unix_s": started,
        "finished_unix_s": time.time(),
    }


_ARRIVAL_CHECKED_ACTIONS = frozenset({
    "move_pose", "move_free_space", "servo_delta", "recover_motion",
})


def execute_skill_sequence(
    runtime: EnvironmentSession, payload: Mapping[str, Any], **kwargs: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], Path]]:
    action = validate_runtime_action(payload)
    if action["schema"] != SKILL_SEQUENCE_ACTION_SCHEMA:
        raise ValueError("execute_skill_sequence requires a skill sequence action")
    with _action_sequence(runtime, action):
        return _execute_skill_sequence(runtime, action, **kwargs)


def _execute_skill_sequence(
    runtime: EnvironmentSession,
    payload: Mapping[str, Any],
    *,
    initial_observation: Mapping[str, Any],
    initial_observation_root: Path,
    drive_stage: Callable[..., Mapping[str, Any]],
    build_receipt: Callable[..., Mapping[str, Any]],
    capture_observation: Callable[[], tuple[dict[str, Any], Path]],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], Path]]:
    """Drive one committed manipulation sequence without opening an Agent turn.

    This function owns the trust boundary: it validates that every driven stage
    action belongs to the submitted commitment, dispatches it, decides whether
    the sequence may continue, and guarantees that the sequence reports exactly
    once however it ends. ``drive_stage`` owns the geometry -- it is the tool
    runtime, which holds the candidate artifacts and their observation binding
    -- and ``build_receipt`` owns the ledger those stages advance.

    ``build_receipt`` is called on **every** exit path, including a refused
    stage, a pose that did not arrive, a terminal environment and a raised
    backend fault. A closed skill that could end without reporting would leave
    the ledger holding a decision the policy cannot take back, which is worse
    than the three explicit calls it replaced.
    """

    from .stage_outcome import classify_stage_action  # stage_outcome imports this module

    action = validate_runtime_action(payload)
    if action["schema"] != SKILL_SEQUENCE_ACTION_SCHEMA:
        raise ValueError("execute_skill_sequence requires a skill sequence action")
    started = time.time()
    arguments = action["arguments"]
    sequence = list(arguments["sequence"])
    latest: tuple[dict[str, Any], Path] = (
        dict(initial_observation), initial_observation_root,
    )
    captured = False

    def capture_fresh() -> tuple[dict[str, Any], Path]:
        nonlocal latest, captured
        latest = capture_observation()
        if not str(latest[0].get("observation_id") or ""):
            raise ValueError("skill sequence fresh observation lacks identity")
        captured = True
        return latest

    stages: list[dict[str, Any]] = []
    status = "completed"
    stop_reason = "sequence_completed"
    for index in range(int(arguments["resume_from"]), len(sequence)):
        stage = sequence[index]
        if should_stop is not None and should_stop():
            stages.append(_skipped_stage(stage, "operator_interrupt"))
            status, stop_reason = "aborted", "operator_interrupt"
            break
        environment_status = runtime.status()
        if environment_status.get("terminal") is True:
            stages.append(_skipped_stage(stage, "environment_terminal"))
            status, stop_reason = "terminated", "environment_terminal"
            break
        try:
            driven = drive_stage(
                action, stage=stage, stage_index=index,
                observation=latest[0], observation_root=latest[1],
                completed=list(stages), capture_observation=capture_fresh,
            )
        except Exception:
            # The driver is expected to turn its own refusals into records; a
            # raise from it is a fault, and it reaches here before any world
            # mutation for this stage. It still has to report, for the same
            # reason a raised backend fault does.
            stages.append(_skipped_stage(stage, "stage_driver_fault"))
            build_receipt(
                action=action, status="aborted", stages=stages,
                observation=latest[0], stop_reason="stage_driver_fault",
            )
            raise
        kind = driven.get("kind")
        if kind == "refusal":
            stages.append(_driven_stage(stage, driven, default_status="refused"))
            status = "aborted"
            stop_reason = str(driven.get("reason") or "stage_refused")
            break
        if kind == "measurement":
            record = _driven_stage(stage, driven, default_status="executed")
            environment_status = runtime.status()
            if environment_status.get("terminal") is True:
                record.update(status="terminated", reason="environment_terminal")
                stages.append(record)
                status, stop_reason = "terminated", "environment_terminal"
                break
            stages.append(record)
            if record["status"] == "stopped":
                status = "stopped"
                stop_reason = "stopped_for_close_view_before_release"
                break
            continue
        if kind != "mutation":
            raise ValueError("skill sequence stage produced an unknown record")
        stage_action = validate_runtime_action(driven.get("action"))
        expected_id = f"{action['action_id']}:{index:02d}-{stage}"
        if (
            stage_action["action_id"] != expected_id
            or stage_action["observation_id"] != action["observation_id"]
            or stage_action["bound_observation_sha256"]
            != action["bound_observation_sha256"]
        ):
            raise ValueError("skill sequence stage drifted from its committed action")
        try:
            # A driver may have captured a baseline immediately before this
            # stage. Once a mutation is dispatched it is no longer the final
            # scene, including when the endpoint check below aborts the job.
            captured = False
            stage_receipt = execute_runtime_action(
                runtime, stage_action, should_stop=should_stop,
            )
        except Exception:
            stages.append({
                "stage": stage, "status": "unavailable",
                "reason": "trusted_execution_fault",
                "action_id": stage_action["action_id"], "measurements": [],
            })
            build_receipt(
                action=action, status="aborted", stages=stages,
                observation=latest[0], stop_reason="trusted_execution_fault",
            )
            raise
        record = _driven_stage(stage, driven, default_status="executed")
        record["action_id"] = stage_action["action_id"]
        record["action"] = dict(stage_action)
        record["reason"] = "executed"
        record["receipt"] = stage_receipt
        # The trusted world may end during this subaction. Preserve its raw
        # execution/pose evidence, but do not turn an interrupted endpoint into
        # a failed approach or authorize another stage from an arrival guess.
        environment_status = runtime.status()
        if environment_status.get("terminal") is True:
            record.update(status="terminated", reason="environment_terminal")
            stages.append(record)
            status, stop_reason = "terminated", "environment_terminal"
            break
        classify_stage_action(record, stage_action, stage_receipt)
        stages.append(record)
        if record["status"] != "executed":
            status, stop_reason = "aborted", f"{stage}_not_reached"
            break
    if not captured:
        capture_fresh()
    skill_receipt = build_receipt(
        action=action, status=status, stages=stages,
        observation=latest[0], stop_reason=stop_reason,
    )
    return {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": action["action_id"],
        "action": dict(action),
        "skill": arguments["skill"],
        "candidate_id": arguments["candidate_id"],
        "bound_observation_id": action["observation_id"],
        "committed_sequence": sequence,
        "resumed_from": int(arguments["resume_from"]),
        "stage_receipts": stages,
        # One coordinator execution is one robot decision, exactly as a combined
        # v4 target state is. The simulator advances once per
        # dispatched stage, which is a different budget, so both numbers are
        # stated rather than one standing in for the other.
        "robot_decisions": 1,
        "dispatched_stage_actions": sum(
            1 for item in stages if item.get("action_id") is not None
        ),
        "status": status,
        "stop_reason": stop_reason,
        "terminal_after_action": status == "terminated",
        "skill_receipt": dict(skill_receipt),
        "fresh_observation_id": str(latest[0].get("observation_id") or ""),
        "consumed": True,
        "started_unix_s": started,
        "finished_unix_s": time.time(),
    }, latest


def _skipped_stage(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage, "status": "skipped", "reason": reason,
        "action_id": None, "measurements": [],
    }


def _driven_stage(
    stage: str, driven: Mapping[str, Any], *, default_status: str,
) -> dict[str, Any]:
    record = {
        "stage": stage,
        "status": str(driven.get("status") or default_status),
        "reason": str(driven.get("reason") or default_status),
        "action_id": None,
        "measurements": [dict(item) for item in driven.get("measurements", ())],
    }
    # A verifying stage's structured evidence is the one thing in a stage record
    # that is not reducible to its measurement list, and it is what the receipt's
    # verification block is built from. Dropping it here would leave every
    # receipt reporting no verification while the stage reported a measurement.
    verification = driven.get("verification")
    if isinstance(verification, Mapping):
        record["verification"] = dict(verification)
    return record
