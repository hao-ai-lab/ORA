"""Strict backend-neutral action validation for robot environment sessions."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


RUNTIME_ACTION_SCHEMA = "ora-action.v3"
TARGET_STATE_ACTION_SCHEMA = "ora-action.v4"
MOTION_ACTION_SCHEMA = "ora-motion-action.v1"
MOTION_ACTION_V3_SCHEMA = "ora-motion-action.v3"
MOTION_INTERACTIONS = frozenset({"free_space", "precision", "contact_approach"})
# Re-exported, not redefined: `skill_sequence` owns the committed-sequence
# contract and a second literal here would be a second thing to keep in step.
from .skill_sequence import SKILL_SEQUENCE_ACTION_SCHEMA  # noqa: E402
SERVO_MAX_TRANSLATION_M = 0.05
SERVO_MAX_ROTATION_RAD = math.radians(15.0)
FORBIDDEN_KEY_FRAGMENTS = {
    "actor",
    "object_gt",
    "ground_truth",
    "segmentation",
    "contact",
    "success_predicate",
    "play_once",
    "task_source",
    "env",
}


class ActionValidationError(ValueError):
    """Raised before an untrusted action reaches the execution bridge."""


def validate_runtime_action(payload: Any) -> dict[str, Any]:
    """Validate the thin Harness action after trusted identity binding."""

    if not isinstance(payload, Mapping):
        raise ActionValidationError("runtime action must be an object")
    _reject_forbidden_keys(payload)
    expected = {
        "schema", "observation_id", "action_id", "action", "arguments",
        "bound_observation_sha256",
    }
    if set(payload) != expected:
        raise ActionValidationError("runtime action fields differ")
    if payload["schema"] == "ora-composed-tool-action.v1":
        from .composed_tool import validate_composed_tool_action
        try:
            return validate_composed_tool_action(payload)
        except ValueError as exc:
            raise ActionValidationError(str(exc)) from exc
    if payload["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA:
        from .skill_sequence import validate_skill_sequence_action
        try:
            return validate_skill_sequence_action(payload)
        except ValueError as exc:
            raise ActionValidationError(str(exc)) from exc
    if payload["schema"] == TARGET_STATE_ACTION_SCHEMA:
        return _validate_target_state_action(payload)
    if payload["schema"] == MOTION_ACTION_SCHEMA:
        return _validate_motion_action(payload)
    if payload["schema"] == MOTION_ACTION_V3_SCHEMA:
        return _validate_motion_action_v3(payload)
    if payload["schema"] != RUNTIME_ACTION_SCHEMA:
        raise ActionValidationError(
            "runtime action schema must be "
            f"{RUNTIME_ACTION_SCHEMA}, {TARGET_STATE_ACTION_SCHEMA}, "
            f"{MOTION_ACTION_SCHEMA}, or {MOTION_ACTION_V3_SCHEMA}"
        )
    if payload["action"] not in {"move_pose", "set_gripper"}:
        raise ActionValidationError("action must be move_pose or set_gripper")
    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ActionValidationError(f"{field} must be non-empty")
    arguments = payload["arguments"]
    if not isinstance(arguments, Mapping):
        raise ActionValidationError("arguments must be an object")
    if payload["action"] == "move_pose":
        normalized_arguments = _validate_move(arguments)
    else:
        normalized_arguments = _validate_gripper(arguments)
    return {
        "schema": RUNTIME_ACTION_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": payload["action"],
        "arguments": normalized_arguments,
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }


def _validate_motion_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ActionValidationError(f"{field} must be non-empty")
    action = payload.get("action")
    if action not in {
        "move_free_space", "servo_delta", "set_gripper", "recover_motion",
    }:
        raise ActionValidationError("motion action is unsupported")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ActionValidationError("arguments must be an object")
    if arguments.get("arm") not in {"left", "right"}:
        raise ActionValidationError("arm must be left or right")
    normalized: dict[str, Any]
    if action == "move_free_space":
        if set(arguments) != {"arm", "target_pose"}:
            raise ActionValidationError(
                "move_free_space arguments must contain arm and target_pose"
            )
        normalized = {
            "arm": arguments["arm"],
            "target_pose": _validate_pose(arguments["target_pose"]),
        }
    elif action == "servo_delta":
        expected = {"arm", "target_pose", "delta_pose", "frame", "max_iterations"}
        if set(arguments) != expected:
            raise ActionValidationError(
                "servo_delta arguments must contain arm, target_pose, delta_pose, "
                "frame and max_iterations"
            )
        if arguments["frame"] not in {"world", "grip_site"}:
            raise ActionValidationError("servo_delta frame must be world or grip_site")
        iterations = arguments["max_iterations"]
        if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 20:
            raise ActionValidationError("servo_delta max_iterations must be in [1, 20]")
        delta = arguments["delta_pose"]
        if not isinstance(delta, Mapping) or set(delta) != {
            "translation_m", "rotation_rotvec_rad",
        }:
            raise ActionValidationError(
                "servo_delta delta_pose must contain translation_m and rotation_rotvec_rad"
            )
        translation = _finite_vector(delta["translation_m"], 3, "translation_m")
        rotation = _finite_vector(
            delta["rotation_rotvec_rad"], 3, "rotation_rotvec_rad"
        )
        if math.sqrt(sum(value * value for value in translation)) > SERVO_MAX_TRANSLATION_M + 1e-12:
            raise ActionValidationError("servo_delta translation exceeds 0.05 m")
        if math.sqrt(sum(value * value for value in rotation)) > SERVO_MAX_ROTATION_RAD + 1e-12:
            raise ActionValidationError("servo_delta rotation exceeds 15 degrees")
        normalized = {
            "arm": arguments["arm"],
            "target_pose": _validate_pose(arguments["target_pose"]),
            "delta_pose": {
                "translation_m": translation,
                "rotation_rotvec_rad": rotation,
            },
            "frame": arguments["frame"],
            "max_iterations": iterations,
        }
    elif action == "set_gripper":
        normalized = _validate_gripper(arguments)
        if normalized["gripper"] not in {0.0, 1.0}:
            raise ActionValidationError("motion set_gripper accepts only open or close")
    else:
        if set(arguments) != {"arm", "strategy"}:
            raise ActionValidationError(
                "recover_motion arguments must contain arm and strategy"
            )
        if arguments["strategy"] not in {"backtrack_last_safe", "home"}:
            raise ActionValidationError(
                "recover_motion strategy must be backtrack_last_safe or home"
            )
        normalized = {"arm": arguments["arm"], "strategy": arguments["strategy"]}
    return {
        "schema": MOTION_ACTION_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": action,
        "arguments": normalized,
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }


def _validate_pose(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != {
        "position_m", "orientation_xyzw",
    }:
        raise ActionValidationError(
            "target_pose must contain position_m and orientation_xyzw"
        )
    position = _finite_vector(value["position_m"], 3, "position_m")
    orientation = _finite_vector(value["orientation_xyzw"], 4, "orientation_xyzw")
    norm = math.sqrt(sum(component * component for component in orientation))
    if not 0.98 <= norm <= 1.02:
        raise ActionValidationError("orientation_xyzw must be normalized")
    return {
        "position_m": position,
        "orientation_xyzw": [component / norm for component in orientation],
    }


def _validate_target_state_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ActionValidationError(f"{field} must be non-empty")
    if payload.get("action") != "target_state":
        raise ActionValidationError("v4 action must be target_state")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ActionValidationError("arguments must be an object")
    if set(arguments) - {"arm", "pose", "gripper"}:
        raise ActionValidationError("target_state arguments contain unknown fields")
    if arguments.get("arm") not in {"left", "right"}:
        raise ActionValidationError("arm must be left or right")
    has_pose = "pose" in arguments and arguments.get("pose") is not None
    has_gripper = "gripper" in arguments and arguments.get("gripper") is not None
    if not has_pose and not has_gripper:
        raise ActionValidationError("target_state requires pose or gripper")
    normalized: dict[str, Any] = {"arm": arguments["arm"]}
    if has_pose:
        pose = arguments["pose"]
        if not isinstance(pose, Mapping) or set(pose) != {
            "position_m", "orientation_xyzw",
        }:
            raise ActionValidationError(
                "pose must contain position_m and orientation_xyzw"
            )
        position = _finite_vector(pose["position_m"], 3, "position_m")
        orientation = _finite_vector(
            pose["orientation_xyzw"], 4, "orientation_xyzw"
        )
        norm = math.sqrt(sum(component * component for component in orientation))
        if not 0.98 <= norm <= 1.02:
            raise ActionValidationError("orientation_xyzw must be normalized")
        normalized["pose"] = {
            "position_m": position,
            "orientation_xyzw": [component / norm for component in orientation],
        }
    if has_gripper:
        gripper = arguments["gripper"]
        if (
            isinstance(gripper, bool)
            or not isinstance(gripper, (int, float))
            or not math.isfinite(gripper)
            or not 0 <= gripper <= 1
        ):
            raise ActionValidationError("gripper must be a finite number in [0, 1]")
        normalized["gripper"] = float(gripper)
    return {
        "schema": TARGET_STATE_ACTION_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": "target_state",
        "arguments": normalized,
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }


def target_state_phases(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand one validated v4 target state into ordered v3 backend phases."""

    action = validate_runtime_action(payload)
    if action["schema"] != TARGET_STATE_ACTION_SCHEMA:
        return [action]
    arguments = action["arguments"]
    phases: list[dict[str, Any]] = []
    if "pose" in arguments:
        pose = arguments["pose"]
        phases.append({
            "schema": RUNTIME_ACTION_SCHEMA,
            "observation_id": action["observation_id"],
            "action_id": f"{action['action_id']}:pose",
            "action": "move_pose",
            "arguments": {
                "arm": arguments["arm"],
                "position_m": pose["position_m"],
                "orientation_xyzw": pose["orientation_xyzw"],
            },
            "bound_observation_sha256": action["bound_observation_sha256"],
        })
    if "gripper" in arguments:
        phases.append({
            "schema": RUNTIME_ACTION_SCHEMA,
            "observation_id": action["observation_id"],
            "action_id": f"{action['action_id']}:gripper",
            "action": "set_gripper",
            "arguments": {
                "arm": arguments["arm"], "gripper": arguments["gripper"],
            },
            "bound_observation_sha256": action["bound_observation_sha256"],
        })
    return phases


def _validate_move(arguments: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"arm", "position_m", "orientation_xyzw"}
    if set(arguments) != expected:
        raise ActionValidationError(
            "move_pose arguments must contain arm, position_m and orientation_xyzw"
        )
    if arguments["arm"] not in {"left", "right"}:
        raise ActionValidationError("arm must be left or right")
    position = _finite_vector(arguments["position_m"], 3, "position_m")
    quat = _finite_vector(arguments["orientation_xyzw"], 4, "orientation_xyzw")
    norm = math.sqrt(sum(component * component for component in quat))
    if not 0.98 <= norm <= 1.02:
        raise ActionValidationError("orientation_xyzw must be normalized")
    return {
        "arm": arguments["arm"],
        "position_m": position,
        "orientation_xyzw": [component / norm for component in quat],
    }


def _validate_gripper(arguments: Mapping[str, Any]) -> dict[str, Any]:
    if set(arguments) != {"arm", "gripper"}:
        raise ActionValidationError("set_gripper arguments must contain arm and gripper")
    if arguments["arm"] not in {"left", "right"}:
        raise ActionValidationError("arm must be left or right")
    gripper = arguments["gripper"]
    if (
        isinstance(gripper, bool)
        or not isinstance(gripper, (int, float))
        or not math.isfinite(gripper)
        or not 0 <= gripper <= 1
    ):
        raise ActionValidationError("gripper must be a finite number in [0, 1]")
    return {"arm": arguments["arm"], "gripper": float(gripper)}


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ActionValidationError(f"{label} must contain {length} numbers")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ActionValidationError(f"{label} must contain finite numbers")
        output.append(float(item))
    return output


def _reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise ActionValidationError(f"forbidden policy field at {path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _validate_sparse_targets(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionValidationError("targets must be an object")
    if not value or set(value) - {"left", "right"}:
        raise ActionValidationError("targets must specify left, right, or both")
    return dict(value)


def _validate_motion_action_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate semantic motion intent without exposing controller selection."""

    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ActionValidationError(f"{field} must be non-empty")
    action = payload.get("action")
    if action not in {"move_pose", "move_free_space", "set_gripper", "recover_motion", "wait"}:
        raise ActionValidationError("motion v3 action is unsupported")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ActionValidationError("arguments must be an object")

    if action == "move_pose" and set(arguments) == {"plan_ref"}:
        plan_ref = arguments.get("plan_ref")
        if (
            not isinstance(plan_ref, str)
            or re.fullmatch(r"plan-[0-9a-f]{32}", plan_ref) is None
        ):
            raise ActionValidationError("move_pose plan_ref is invalid")
        normalized_arguments: dict[str, Any] = {"plan_ref": plan_ref}
    elif action == "wait":
        from .wait import validate_wait_arguments
        try:
            normalized_arguments = validate_wait_arguments(arguments)
        except ValueError as exc:
            raise ActionValidationError(str(exc)) from exc
    else:
        expected = {"targets", "interactions"} if action in {
            "move_pose", "move_free_space",
        } else {"targets"}
        if set(arguments) != expected:
            raise ActionValidationError(
                f"motion v3 {action} arguments must contain {sorted(expected)}"
                + (" or only plan_ref" if action == "move_pose" else "")
            )
        raw_targets = _validate_sparse_targets(arguments["targets"])
        if action in {"move_pose", "move_free_space"}:
            raw_interactions = arguments["interactions"]
            if (
                not isinstance(raw_interactions, Mapping)
                or set(raw_interactions) != set(raw_targets)
            ):
                raise ActionValidationError(
                    "motion v3 interactions must specify exactly the targeted arms"
                )
            normalized_interactions: dict[str, str] = {}
            for arm, value in raw_interactions.items():
                if value not in MOTION_INTERACTIONS:
                    raise ActionValidationError(
                        "motion interaction must be free_space, precision, or "
                        "contact_approach"
                    )
                if action == "move_free_space" and value != "free_space":
                    raise ActionValidationError(
                        "move_free_space requires free_space interaction"
                    )
                normalized_interactions[arm] = str(value)
            normalized_arguments = {
                "targets": {
                    arm: _validate_pose(value) for arm, value in raw_targets.items()
                },
                "interactions": normalized_interactions,
            }
        elif action == "set_gripper":
            normalized_targets = {}
            for arm, value in raw_targets.items():
                if value not in {"open", "close"}:
                    raise ActionValidationError(
                        "motion v3 set_gripper targets must be open or close"
                    )
                normalized_targets[arm] = str(value)
            normalized_arguments = {"targets": normalized_targets}
        else:
            normalized_targets = {}
            for arm, value in raw_targets.items():
                if value not in {"backtrack_last_safe", "home"}:
                    raise ActionValidationError(
                        "motion v3 recover_motion strategy must be "
                        "backtrack_last_safe or home"
                    )
                normalized_targets[arm] = str(value)
            normalized_arguments = {"targets": normalized_targets}

    return {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": action,
        "arguments": normalized_arguments,
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }
