"""One plain-language line at the head of the last-action result.

``last_action_summary_v1``: the policy reads, before the detail, a line such as
``left arm 104 mm / 120° from target, plan rejected: inter arm collision, not
moved``. It restates only what the projected effect already carries:

* distance from target per targeted arm -- the arm's ``pose_error``, else the
  last endpoint acceptance residual, else LIBERO's judged endpoint residual;
  for a closed skill, the last stage that measured an endpoint residual;
* for a gripper command, measured opening and effective target;
* the outcome -- ``status`` and, when it adds something, ``failure_code``
  (a closed skill also names its ``failure_stage``);
* whether the arm moved -- ``synchronization.executed``, the native-action
  window, or the measured TCP translation, else ``motion unknown``.

The line never adds a hint, a cause or a next step. Angles are in degrees
whatever unit the detail uses. The shared contact detail measures the right
arm's TCP only, so its translation is cited only when the right arm is the one
target, or when the effect names no target arm (LIBERO's single-arm hybrid
receipts). A surface declaring ``per_arm_realized_translation_v1`` carries each
targeted arm's own translation in ``detail.arms``, and those are cited instead.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

CAPABILITY = "last_action_summary_v1"
SUMMARY_KEY = "action_summary"
_ENDPOINT_STAGES = ("final_endpoint", "controller_terminal_endpoint")


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _millimetres(metres: float) -> str:
    value = metres * 1000.0
    return f"{value:.0f} mm" if abs(value) >= 10 else f"{value:.1f} mm"


def _degrees(radians: float) -> str:
    value = math.degrees(radians)
    return f"{value:.0f}°" if abs(value) >= 10 else f"{value:.1f}°"


def _words(code: Any) -> str:
    return str(code).replace("_", " ")


def _from_target(arm: Any, position: Any, orientation: Any) -> str | None:
    position, orientation = _finite(position), _finite(orientation)
    parts = [*([_millimetres(position)] if position is not None else []),
             *([_degrees(orientation)] if orientation is not None else [])]
    if not parts:
        return None
    return f"{f'{arm} arm ' if arm else ''}{' / '.join(parts)} from target"


def _pose_residuals(detail: Mapping[str, Any]) -> list[str]:
    rows: list[str] = []
    arms, targets = detail.get("arms"), detail.get("target_arms")
    if isinstance(arms, Mapping) and isinstance(targets, list):
        for arm in targets:
            state = arms.get(arm)
            error = state.get("pose_error") if isinstance(state, Mapping) else None
            if isinstance(error, Mapping):
                row = _from_target(arm, error.get("translation_m", error.get("position_m")),
                                   error.get("orientation_rad"))
                rows.extend([row] if row else [])
    if rows:
        return rows
    for check in reversed(detail.get("acceptance_checks") or []):
        if isinstance(check, Mapping) and check.get("stage") in _ENDPOINT_STAGES \
                and isinstance(check.get("arms"), Mapping):
            for arm, value in sorted(check["arms"].items()):
                residual = value.get("residual") if isinstance(value, Mapping) else None
                if isinstance(residual, Mapping):
                    row = _from_target(arm, residual.get("position_m"), residual.get("orientation_rad"))
                    rows.extend([row] if row else [])
            if rows:
                return rows
    contract = detail.get("endpoint_contract")
    judged = contract.get("judged_residual") if isinstance(contract, Mapping) else None
    if isinstance(judged, Mapping):
        row = _from_target(None, judged.get("position_m"), judged.get("orientation_rad"))
        rows.extend([row] if row else [])
    return rows


def _gripper(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> list[str]:
    checks = [check for check in detail.get("acceptance_checks") or [] if isinstance(check, Mapping)]
    gripper_checks = [check for check in checks if check.get("stage") == "gripper_endpoint"]
    if effect.get("action") != "set_gripper" and not gripper_checks:
        return []
    feedback = detail.get("gripper_feedback")
    targets = detail.get("target_arms") if isinstance(detail.get("target_arms"), list) else []
    rows = []
    for arm in targets:
        state = feedback.get(arm) if isinstance(feedback, Mapping) else None
        measured = _finite(state.get("measured_opening_normalized")) if isinstance(state, Mapping) else None
        if measured is None:
            continue
        target = _finite(state.get("effective_target_opening_normalized"))
        rows.append(f"{arm} gripper opening {measured:.2f}"
                    + (f" / target {target:.2f}" if target is not None else ""))
    if rows:
        return rows
    for check in gripper_checks:
        for arm, value in sorted((check.get("arms") or {}).items()):
            residual = value.get("residual") if isinstance(value, Mapping) else None
            error = _finite(residual.get("opening_normalized")) if isinstance(residual, Mapping) else None
            if error is not None:
                rows.append(f"{arm} gripper {error:.2f} from target opening")
    return rows


def _outcome(effect: Mapping[str, Any], *, name_stage: bool = False) -> str:
    status = effect.get("status")
    text = _words(status)
    stage = effect.get("failure_stage")
    if name_stage and isinstance(stage, str) and stage:
        text += f" at {_words(stage)}"
    code = effect.get("failure_code")
    if isinstance(code, str) and code and code != status:
        text += f": {_words(code)}"
    cause = effect.get("physical_cause")
    if isinstance(cause, str) and cause and cause != "unknown":
        text += f" ({_words(cause)})"
    return text


def _measured_translation(detail: Mapping[str, Any]) -> float | None:
    # The contact detail reads the right arm (LIBERO's only arm) whatever was
    # targeted; LIBERO's hybrid receipts name no target arms at all.
    if detail.get("target_arms", ["right"]) != ["right"]:
        return None
    contact = detail.get("contact_motion")
    vector = contact.get("realized_translation_world_m") if isinstance(contact, Mapping) else None
    if not isinstance(vector, list) or len(vector) != 3 or any(_finite(item) is None for item in vector):
        return None
    return math.dist(vector, (0.0, 0.0, 0.0))


def _arm_translations(detail: Mapping[str, Any]) -> list[tuple[str, float]]:
    arms, targets = detail.get("arms"), detail.get("target_arms")
    if not isinstance(arms, Mapping) or not isinstance(targets, list):
        return []
    rows = []
    for arm in targets:
        state = arms.get(arm)
        vector = state.get("realized_translation_world_m") if isinstance(state, Mapping) else None
        if isinstance(vector, list) and len(vector) == 3 and all(_finite(item) is not None
                                                                  for item in vector):
            rows.append((str(arm), math.dist(vector, (0.0, 0.0, 0.0))))
    return rows


def _motion(detail: Mapping[str, Any]) -> str:
    sync = detail.get("synchronization")
    executed = sync.get("executed") if isinstance(sync, Mapping) else None
    window = detail.get("execution_window")
    used = window.get("native_actions_used") if isinstance(window, Mapping) else None
    budget = window.get("native_action_budget") if isinstance(window, Mapping) else None
    used = used if isinstance(used, int) and not isinstance(used, bool) else None
    if executed is False or used == 0:
        return "not moved"
    moved = _arm_translations(detail)
    if len(moved) == 1:
        return f"TCP moved {_millimetres(moved[0][1])}"
    if moved:
        return ", ".join(f"{arm} TCP moved {_millimetres(value)}" for arm, value in moved)
    translation = _measured_translation(detail)
    if translation is not None:
        return f"TCP moved {_millimetres(translation)}"
    if used is not None:
        return f"ran {used} of {budget} native actions" if isinstance(budget, int) else (
            f"ran {used} native actions")
    return "executed" if executed is True else "motion unknown"


def _skill_summary(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> list[str]:
    stages = [stage for stage in detail["stages"] if isinstance(stage, Mapping)]
    rows: list[str] = []
    for stage in reversed(stages):
        for measurement in stage.get("measurements") or []:
            value = measurement.get("value") if isinstance(measurement, Mapping) else None
            errors = value.get("errors_by_arm") if isinstance(value, Mapping) else None
            if measurement.get("quantity") == "endpoint_residual" and isinstance(errors, Mapping):
                for arm, error in sorted(errors.items()):
                    if isinstance(error, Mapping):
                        row = _from_target(arm, error.get("translation_m", error.get("position_m")),
                                           error.get("orientation_rad"))
                        rows.extend([row] if row else [])
        if rows:
            break
    parts = [*rows, _outcome(effect, name_stage=True)]
    committed = detail.get("committed_sequence")
    if isinstance(committed, list) and committed:
        parts.append(f"{len(stages)} of {len(committed)} stages attempted")
    return parts


def action_summary(effect: Any) -> str | None:
    """The summary line for one projected effect, or ``None`` for an unknown shape."""
    if not isinstance(effect, Mapping) or not isinstance(effect.get("status"), str):
        return None
    detail = effect.get("detail") if isinstance(effect.get("detail"), Mapping) else {}
    if isinstance(detail.get("stages"), list):
        parts = _skill_summary(effect, detail)
    else:
        parts = [*_pose_residuals(detail), *_gripper(effect, detail), _outcome(effect),
                 _motion(detail)]
    if effect.get("round_terminated") is True:
        parts.append("round ended")
    return ", ".join(parts)


__all__ = ["CAPABILITY", "SUMMARY_KEY", "action_summary"]
