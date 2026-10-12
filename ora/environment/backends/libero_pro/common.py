"""Constants, the backend error and module-level helpers of the LIBERO-Pro session.

``session`` re-exports every name defined here, so imports from
``ora.environment.backends.libero_pro.session`` keep working.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.action import (
    SERVO_MAX_ROTATION_RAD,
    SERVO_MAX_TRANSLATION_M,
)


OBSERVATION_SCHEMA = "ora-observation.v1"
RECEIPT_SCHEMA = "ora.libero-pro-motion-receipt.v2"
HYBRID_RECEIPT_SCHEMA = "ora.libero-pro-hybrid-motion-receipt.v2"
LIBERO_SPARSE_RECEIPT_SCHEMA = "ora.libero-sparse-motion-receipt.v1"
VERDICT_SCHEMA = "ora.libero-pro-official-verdict.v1"
RESET_SCHEMA = "ora.libero-pro-reset.v1"
CONTROL_HEALTH_SCHEMA = "ora.libero-control-health.v2"
TRACKING_EVIDENCE_SCHEMA = "ora.libero-pose-tracking-evidence.v1"
LIBERO_POLICY_CAMERA_NAMES = {
    "agentview": "agentview",
    "robot0_eye_in_hand": "wrist",
}
LIBERO_EMBODIMENTS = {("Panda",): "franka_panda"}


def _sparse_planning_failure(reason: str) -> tuple[str, str, str]:
    if reason.startswith("route_"):
        return "plan_rejected", "route", reason
    if reason == "planner_error":
        return "execution_fault", "planner", reason
    if "INVALID_START_STATE_" in reason:
        return "plan_rejected", "planner", reason
    return "planner_failed", "planner", reason


def _public_gripper_command(directive: float, target: str) -> dict[str, Any]:
    """Project the held gripper directive and the last accepted jaw target.

    ``directive`` is the value the executor compiles and the environment holds:
    -1 keeps the jaws open, +1 keeps them closed, and the normalized opening is
    the linear inverse. It is a measurement of what the controller is being
    told, which is exactly the number a policy needs beside the encoder to ask
    whether its own close took effect.

    ``target`` is a separate claim and stays ``unknown`` until a jaw command is
    actually accepted: a reset directive is a hold, not evidence that anything
    was commanded.
    """

    return {
        "gripper_command": (1.0 - float(directive)) / 2.0,
        "gripper_command_target": target,
    }


def _within_local_servo_envelope(
    translation_m: float, rotation_rad: float,
) -> bool:
    """Route a sparse v3 segment locally only when the contract will accept it.

    The threshold and tolerance must be exactly the servo_delta contract's
    (0.05 m / 15 degrees, +1e-12): a boundary-exact request — models do aim
    at exactly 15 degrees — that routes local under a looser tolerance is
    then rejected by validate_runtime_action and escalates to a fatal fault
    instead of taking the global route.
    """

    return (
        translation_m <= SERVO_MAX_TRANSLATION_M + 1e-12
        and rotation_rad <= SERVO_MAX_ROTATION_RAD + 1e-12
    )


class LiberoProError(RuntimeError):
    """The pinned LIBERO-Pro runtime or public boundary was violated."""


def bddl_task_language(path: Path) -> str | None:
    """Read only the public language field from the already bound task asset."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    marker = text.find("(:language")
    if marker < 0:
        return None
    depth = 0
    for index in range(marker, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return " ".join(text[marker + len("(:language"):index].split()).strip() or None
    return None


def resolve_episode_task_language(metadata: Mapping[str, Any], descriptions: Any,
                                  expected: str) -> tuple[str, str | None]:
    """Match the evaluated asset's language even when its file name is retained."""
    actual = str(metadata["task_language"]).strip()
    if descriptions:
        actual = str(descriptions[0]).strip()
    language = bddl_task_language(Path(metadata["bddl_file"])) if metadata.get("bddl_file") else None
    if language == expected:
        actual = language
    if actual != expected:
        raise LiberoProError(f"task language mismatch: expected {expected!r}, got {actual!r}")
    return actual, language


def libero_environment_init_params(*, max_sim_steps: int) -> dict[str, Any]:
    """Return the environment-owned robot/controller/camera construction."""

    return {
        "robots": ["Panda"],
        "controller": "OSC_POSE",
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_depths": True,
        "horizon": max_sim_steps,
    }

def libero_public_environment_context(
    task_metadata: Any, *, init_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse one public prompt descriptor from task and environment metadata."""

    def task_field(name: str) -> Any:
        if isinstance(task_metadata, Mapping):
            return task_metadata.get(name)
        return getattr(task_metadata, name, None)

    if str(task_field("problem")).strip().lower() != "libero":
        raise LiberoProError("task metadata does not identify a LIBERO problem")
    problem_folder = task_field("problem_folder")
    if not isinstance(problem_folder, str) or not problem_folder.strip():
        raise LiberoProError("task metadata omits the LIBERO problem folder")
    robots = init_params.get("robots")
    controller = init_params.get("controller")
    cameras = init_params.get("camera_names")
    if not isinstance(robots, (list, tuple)) or not all(
        isinstance(item, str) and item for item in robots
    ):
        raise LiberoProError("environment metadata has invalid robot models")
    robot_models = tuple(robots)
    embodiment_id = LIBERO_EMBODIMENTS.get(robot_models)
    if embodiment_id is None:
        raise LiberoProError(
            f"environment robot models have no prompt embodiment: {list(robot_models)}"
        )
    if not isinstance(controller, str) or not controller:
        raise LiberoProError("environment metadata has no controller")
    if not isinstance(cameras, (list, tuple)) or not all(
        isinstance(item, str) and item for item in cameras
    ):
        raise LiberoProError("environment metadata has invalid public cameras")
    try:
        public_cameras = [LIBERO_POLICY_CAMERA_NAMES[item] for item in cameras]
    except KeyError as exc:
        raise LiberoProError(
            f"environment camera has no policy-visible name: {exc.args[0]}"
        ) from exc
    if len(public_cameras) != len(set(public_cameras)):
        raise LiberoProError("environment cameras map to duplicate public names")
    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "libero_pro",
        "embodiment_id": embodiment_id,
        "robot_models": list(robot_models),
        "controller": controller,
        "public_cameras": public_cameras,
    }


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not public canonical JSON: {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=60, check=False,
    )
    if result.returncode:
        raise LiberoProError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _metric_depth(depth: Any, camera_meta: Mapping[str, Any]) -> Any:
    import numpy as np

    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        value = float(near) / (1.0 - value * (1.0 - float(near) / float(far)))
    return value


def _quaternion_error_rotvec(target_xyzw: Any, current_xyzw: Any) -> Any:
    """Return the shortest target * inverse(current) rotation vector."""

    import numpy as np

    target = np.asarray(target_xyzw, dtype=np.float64)
    current = np.asarray(current_xyzw, dtype=np.float64)
    current_inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    x1, y1, z1, w1 = target
    x2, y2, z2, w2 = current_inverse
    error = np.asarray([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])
    error /= np.linalg.norm(error)
    if error[3] < 0:
        error = -error
    vector_norm = float(np.linalg.norm(error[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(error[3]))
    return error[:3] * (angle / vector_norm)


def _pose_tracking_sample(
    *, tick: int, before_position: Any, before_orientation: Any,
    target_position: Any, target_orientation: Any, position: Any,
    orientation: Any, previous_position: Any, measured_aperture: float,
    actuator_target: float | None, initial_position_error_m: float,
    initial_orientation_error_rad: float,
) -> dict[str, Any]:
    """Project one control-only, high-level tracking sample."""

    import numpy as np

    requested = np.asarray(target_position) - np.asarray(before_position)
    realized = np.asarray(position) - np.asarray(before_position)
    requested_distance = float(np.linalg.norm(requested))
    realized_distance = float(np.linalg.norm(realized))
    if requested_distance > 1e-9:
        direction = requested / requested_distance
        along = float(np.dot(realized, direction))
        cross_axis = float(np.linalg.norm(realized - direction * along))
    else:
        along = 0.0
        cross_axis = realized_distance
    position_error = float(np.linalg.norm(
        np.asarray(target_position) - np.asarray(position)
    ))
    orientation_error = float(np.linalg.norm(
        _quaternion_error_rotvec(target_orientation, orientation)
    ))
    realized_rotation = float(np.linalg.norm(
        _quaternion_error_rotvec(orientation, before_orientation)
    ))
    return {
        "tick": tick,
        "position_xyz_m": np.asarray(position, dtype=float).tolist(),
        "orientation_xyzw": np.asarray(orientation, dtype=float).tolist(),
        "target_position_error_m": position_error,
        "target_position_error_growth_m": position_error - initial_position_error_m,
        "target_orientation_error_rad": orientation_error,
        "target_orientation_error_growth_rad": (
            orientation_error - initial_orientation_error_rad
        ),
        "realized_translation_world_m": realized.tolist(),
        "realized_translation_distance_m": realized_distance,
        "realized_rotation_angle_rad": realized_rotation,
        "along_request_translation_m": along,
        "cross_axis_drift_m": cross_axis,
        "travel_overshoot_m": max(0.0, along - requested_distance),
        "single_tick_translation_m": float(np.linalg.norm(
            np.asarray(position) - np.asarray(previous_position)
        )),
        "measured_aperture": float(measured_aperture),
        "gripper_actuator_target": actuator_target,
    }


def _normalized_gripper(qpos: Any) -> float:
    """Map public finger displacement to Harness 0=closed, 1=open."""

    import numpy as np

    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise LiberoProError("LIBERO gripper qpos is unavailable")
    return float(np.clip(np.mean(np.abs(values)) / 0.04, 0.0, 1.0))

