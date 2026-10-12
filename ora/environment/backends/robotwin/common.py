"""Constants, errors and helpers shared by the RoboTwin session and its mixins.

Moved verbatim from ``session.py`` on 2026-10-01; ``session.py`` re-exports
every name, so existing imports from it keep working.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.backends.robotwin.planning import ValidatedRoboTwinPlan


OBSERVATION_SCHEMA = "ora-observation.v1"


RECEIPT_SCHEMA = "ora.robotwin-motion-receipt.v1"


SPARSE_RECEIPT_SCHEMA = "ora.robotwin-sparse-motion-receipt.v2"


SPARSE_RECEIPT_V3_SCHEMA = "ora.robotwin-sparse-motion-receipt.v3"


SPARSE_RECEIPT_V4_SCHEMA = "ora.robotwin-sparse-motion-receipt.v4"


SPARSE_RECEIPT_V5_SCHEMA = "ora.robotwin-sparse-motion-receipt.v5"


SPARSE_RECEIPT_V6_SCHEMA = "ora.robotwin-sparse-motion-receipt.v6"


SPARSE_RECEIPT_V7_SCHEMA = "ora.robotwin-sparse-motion-receipt.v7"


VERDICT_SCHEMA = "ora.robotwin-official-verdict.v1"


OPERATOR_EXECUTION_FRAME_SCHEMA = (
    "ora.robotwin-operator-execution-frame.v1"
)


OPERATOR_EXECUTION_VIDEO_SCHEMA = (
    "ora.robotwin-operator-execution-video.v2"
)


OPERATOR_CAMERA_ORDER = (
    "observer_camera",
    "head_camera",
    "front_camera",
    "left_camera",
    "right_camera",
)


DEFAULT_REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"


#: RoboTwin tasks this backend may construct as a live EnvironmentSession.
#: Admission is deliberately separate from qualification registration: a
#: new task enters this set only once it is in ``qualification.DEFAULT_TASKS``
#: with a ``TASK_MODEL_ATTRIBUTES`` entry, a tracked seed qualification has
#: recorded ``reset_predicate_false`` and ``oracle_success`` at the pinned
#: revision, its per-revision task-contract audit is recorded, and any
#: expert-only evaluator state it reads is restored by
#: ``session._restore_expert_only_evaluator_state``. The seed
#: qualifier exercises this module's exact reset contract, so that record
#: is what proves the task constructible -- not a reviewer's reading of
#: ``check_success``. Tasks meeting none of that stay in
#: ``DEFERRED_SESSION_TASKS`` so the gap is named rather than silent.
#: A/B placement is admitted for the reviewed G8 launcher: left uses the
#: qualified seed 3 and both tasks require their public named-object instruction
#: and exact source/seed witness bindings. Registration alone grants no launch.
SUPPORTED_TASKS = {
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
}


#: Qualification-registered tasks deliberately not executable yet, each
#: with the reason it fails the admission criteria above. A task must be
#: in exactly one of these two sets; the test suite enforces that, because
#: a silent gap here once let thirty-six qualified tasks fail at worker
#: startup instead of at review time.
DEFERRED_SESSION_TASKS: set[str] = set()


DEFAULT_WORKSPACE_BOUNDS_M = {
    "x": [-0.60, 0.60],
    "y": [-0.40, 0.40],
    "z": [0.70, 1.45],
}


ENVIRONMENT_SIGNAL_POLICIES = {"lifecycle_only", "task_recoverability_v1"}


TASK_RECOVERABILITY_ACTORS = {"adjust_bottle": ("bottle", 0)}


class RoboTwinLiveError(RuntimeError):
    """A pinned runtime, episode, action or artifact contract was violated."""


class RoboTwinControllerTrackingError(RoboTwinLiveError):
    """A dense controller path crossed a profile-bound fail-closed guard."""

    def __init__(
        self, message: str, *, summary: Mapping[str, Any], arrays: Mapping[str, Any],
        control_sequence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.summary = dict(summary)
        self.arrays = dict(arrays)
        self.control_sequence = (
            dict(control_sequence) if control_sequence is not None else None
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RoboTwinLiveError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _load_task_args(
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
    runtime_output: Path,
) -> dict[str, Any]:
    import yaml

    config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args.update(
        {
            "task_name": task,
            "task_config": task_config,
            "save_path": str(runtime_output),
            "need_plan": True,
            "save_data": False,
            "collect_data": False,
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
        }
    )
    data_type = args.setdefault("data_type", {})
    data_type.update(
        {
            "rgb": True,
            "depth": True,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
            "pointcloud": False,
        }
    )

    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    configs = Path(CONFIGS_PATH)
    embodiment_map = yaml.safe_load(
        (configs / "_embodiment_config.yml").read_text(encoding="utf-8")
    )

    def embodiment_file(name: str) -> str:
        path = embodiment_map[name]["file_path"]
        if not path:
            raise RoboTwinLiveError(f"embodiment {name!r} has no file_path")
        return str(path)

    def embodiment_config(path: str) -> dict[str, Any]:
        return yaml.safe_load((Path(path) / "config.yml").read_text(encoding="utf-8"))

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise RoboTwinLiveError("embodiment must have one or three entries")
    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
    return args


def _task_factory(task: str) -> Any:
    module = importlib.import_module(f"envs.{task}")
    return getattr(module, task)()


def _wxyz_to_xyzw(pose: Any) -> list[float]:
    values = [float(value) for value in pose]
    if len(values) != 7:
        raise RoboTwinLiveError("RoboTwin end-effector pose must have seven values")
    x, y, z, w, qx, qy, qz = values
    return [x, y, z, qx, qy, qz, w]


def _xyzw_to_wxyz(position: Any, orientation: Any) -> list[float]:
    xyz = [float(value) for value in position]
    qx, qy, qz, w = [float(value) for value in orientation]
    return xyz + [w, qx, qy, qz]


def _translate_pose_local_wxyz(
    pose: Any, translation_local_m: Any,
) -> list[float]:
    """Translate a world pose along its own axes without changing orientation."""

    values = [float(value) for value in pose]
    offset = [float(value) for value in translation_local_m]
    if len(values) != 7 or len(offset) != 3:
        raise RoboTwinLiveError("controlled-point transform is malformed")
    x, y, z = offset
    w, qx, qy, qz = values[3:]
    norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RoboTwinLiveError("controlled-point orientation is invalid")
    w, qx, qy, qz = (item / norm for item in (w, qx, qy, qz))
    rotated = [
        (1 - 2 * (qy * qy + qz * qz)) * x
        + 2 * (qx * qy - qz * w) * y
        + 2 * (qx * qz + qy * w) * z,
        2 * (qx * qy + qz * w) * x
        + (1 - 2 * (qx * qx + qz * qz)) * y
        + 2 * (qy * qz - qx * w) * z,
        2 * (qx * qz - qy * w) * x
        + 2 * (qy * qz + qx * w) * y
        + (1 - 2 * (qx * qx + qy * qy)) * z,
    ]
    return [
        values[index] + rotated[index] for index in range(3)
    ] + [w, qx, qy, qz]


def _pose_error(commanded_xyzw: list[float], reached_xyzw: list[float]) -> dict[str, float]:
    translation = math.sqrt(
        sum((commanded_xyzw[index] - reached_xyzw[index]) ** 2 for index in range(3))
    )
    q1 = commanded_xyzw[3:]
    q2 = reached_xyzw[3:]
    dot = min(1.0, abs(sum(a * b for a, b in zip(q1, q2))))
    orientation = 2.0 * math.acos(dot)
    return {
        "translation_m": round(translation, 6),
        "orientation_rad": round(orientation, 6),
    }


def _validate_workspace_target(
    position: list[float], bounds: Mapping[str, Any], *, argument_field: str | None = None,
) -> None:
    for index, axis in enumerate(("x", "y", "z")):
        interval = bounds.get(axis)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)
        ):
            raise RoboTwinLiveError(f"invalid workspace bound for {axis}")
        if not float(interval[0]) <= position[index] <= float(interval[1]):
            if argument_field is not None:
                raise EnvironmentRefusal(
                    f"motion probe target {axis}={position[index]} is outside {interval}",
                    field=argument_field, received=position, allowed=bounds,
                )
            raise RoboTwinLiveError(
                f"move_pose target {axis}={position[index]} is outside {interval}"
            )


def probe_variant_sort_key(row: Mapping[str, Any]) -> tuple:
    """Order reachable probe variants for selection.

    A reported duration of 0.0 means the plan carried no honest time
    parameterization; such untimed plans rank strictly behind every timed
    plan so a missing dt can never win as a zero-duration trajectory.
    """

    duration = float(row["trajectory_duration_s"])
    return (
        duration == 0.0,
        duration,
        float(row["endpoint_error"]["translation_m"]),
        float(row["endpoint_error"]["orientation_rad"]),
        str(row["variant_id"]),
    )


@dataclass
class _SparseMotionState:
    """Mutable per-action state shared by the sparse-motion helper methods.

    Fields mirror the historical local variables of ``_execute_sparse_motion``
    one to one; the split is a pure relocation with no semantic change.
    """

    action: Mapping[str, Any]
    receipt_path: Path
    started: float
    environment: Any
    grip_center_profile: bool
    arguments: Mapping[str, Any]
    targets: dict[str, Any]
    target_arms: tuple[str, ...]
    local_cartesian_requested: bool = False
    global_then_local_settle_requested: bool = False
    global_precontact_then_local_requested: bool = False
    motion_strategy: dict[str, Any] | None = None
    motion_strategy_error: str | None = None
    pose_action: bool = False
    before: dict[str, dict[str, Any]] = field(default_factory=dict)
    measured_joint_snapshot: dict[str, list[float]] = field(
        default_factory=dict
    )
    motion_adapter: Any = None
    motion_action_start: Any = None
    planner_results: dict[str, Any] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    executed: bool = False
    execution_attempted: bool = False
    execution_fault: str | None = None
    replay_recorded: bool = False
    public_pose_targets: dict[str, list[float]] = field(default_factory=dict)
    planner_public_pose_targets: dict[str, list[float]] = field(
        default_factory=dict
    )
    planner_pose_targets: dict[str, Any] = field(default_factory=dict)
    hold_engaged: dict[str, bool] = field(
        default_factory=lambda: {"left": False, "right": False}
    )
    settle_steps: int = 0
    control_sequence_record: Mapping[str, Any] | None = None
    controller_tracking: dict[str, Any] | None = None
    acceptance_checks: list[dict[str, Any]] = field(default_factory=list)
    controller_tracking_arrays: dict[str, Any] = field(default_factory=dict)
    validated_plans: dict[str, ValidatedRoboTwinPlan] = field(
        default_factory=dict
    )
    plan_rejections: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    collision_validation: dict[str, Any] | None = None
    tracking_not_reached_reason: str | None = None
    payload_validation: dict[str, Any] | None = None
    invalidated_payload_refs: list[str] = field(default_factory=list)
    recovery_joint_targets: dict[str, list[float]] = field(
        default_factory=dict
    )
    reached: dict[str, list[float]] = field(default_factory=dict)
    execution_status: str | None = None
    planning_start_public_poses: dict[str, list[float]] = field(default_factory=dict)
    blocked_recovery: dict[str, Any] | None = None
    recovery_prefix: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
