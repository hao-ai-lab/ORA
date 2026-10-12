"""Measurement, gripper, route-geometry and evidence helpers shared by the RoboDojo session modules."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers.gripper_stall import GripperStallRule
from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT

from .bridge import RobodojoError


POSITION_TOLERANCE_M = float(ARX_X5_CONTRACT["position_tolerance_m"])
ORIENTATION_TOLERANCE_RAD = float(ARX_X5_CONTRACT["orientation_tolerance_rad"])
GRIPPER_TOLERANCE = float(ARX_X5_CONTRACT["gripper_tolerance_normalized"])
GRIPPER_COMMAND_SCALE = tuple(
    float(value) for value in ARX_X5_CONTRACT["gripper_command"]["primary_joint_range"]
)
# ``X5A.urdf`` limits the prismatic finger joints ``joint7``/``joint8`` to
# ``[0, 0.044]`` m while ``robot_config.yml`` drives them over the wider
# command scale above, so the command origin lies 10 mm below the
# articulation's own lower limit: a ``close`` directive of 0.0 is written and
# held, but the plant stops at that limit, which normalizes to about 0.19.
# ``sparse_motion.gripper_closed_floor`` measures the same floor from the live
# articulation and is what an executed ``close`` is checked against; this
# source-derived value is its host-side projection. Neither number is a
# measured jaw separation.
FINGER_JOINT_LIMIT_M = (0.0, 0.044)
CLOSED_COMMAND_FLOOR = (
    (FINGER_JOINT_LIMIT_M[0] - GRIPPER_COMMAND_SCALE[0])
    / (GRIPPER_COMMAND_SCALE[1] - GRIPPER_COMMAND_SCALE[0])
)
# The command is normalized and quantization-free, so the categorical read
# only has to separate the two admitted directives.
COMMAND_TARGET_TOLERANCE = 0.02
# One native action is 10 physics steps of 4 ms at 25 Hz: 40 ms per tick and
# at most 20 ticks (0.8 s) per command. On the 198 sparse set_gripper receipts
# recorded before this rule, a free-air jaw moved at least 0.0083 of the
# normalized opening per tick until it arrived, while a jaw stopped on an
# object moved under 0.005 per tick (usually under 0.001) and then held the
# SDK-limited drive target 0.2 below its first-finger encoder. Three such still
# ticks end the command instead of spending the remaining budget.
ARX_GRIPPER_STALL = GripperStallRule(
    still_step=0.005, window=3, minimum_samples=4,
    arrival_tolerance=GRIPPER_TOLERANCE,
    tick_period_s=1.0 / float(ARX_X5_CONTRACT["control_hz"]),
    tick_unit="native_action", arrival_comparison="<=",
)

# The principal range of one revolute coordinate. No narrower ARX X5 joint
# range is recorded in source: X5A.urdf limits joint1-joint5 to +/-10 rad, and
# admission checks the live articulation limits (``ik.articulation_limits``).
# IK ranking prefers continuous candidates inside it (EXP11A g005 audit D3).
CANONICAL_JOINT_INTERVAL_RAD = (-math.pi, math.pi)
# A pitch joint already outside that interval may stay out by up to this much
# more without counting as winding: a held joint overshoots its command
# slightly (review U1 round 2 decision). A joint inside gets no allowance.
PITCH_WRAP_HOLD_TOLERANCE_RAD = 1e-3


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=json_default, allow_nan=False))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def measured_pose(robot_manager: Any, robot: Any) -> list[float]:
    """Read the link6 pose as policy xyzw from the SDK's wxyz measurement."""
    wxyz = array(robot_manager.get_real_endpose(robot, env_idx_list=[0])[0])
    if wxyz.shape != (7,) or not np.isfinite(wxyz).all():
        raise RobodojoError("robot end-effector measurement is not finite")
    return np.concatenate((wxyz[:3], wxyz[4:], wxyz[3:4])).tolist()


def native_pose(pose_xyz_xyzw: Any) -> list[float]:
    """Order one policy xyzw pose as the SDK's wxyz IK target."""
    pose = [float(item) for item in pose_xyz_xyzw]
    if len(pose) != 7 or not all(math.isfinite(item) for item in pose):
        raise RobodojoError("pose target must contain seven finite numbers")
    return pose[:3] + [pose[6]] + pose[3:6]


def gripper_opening(robot: Any, physical: float) -> float:
    low, high = robot.gripper_scale
    value = (physical - low) / (high - low)
    return float(np.clip(value if robot.gripper_move["sign"] == 1 else 1 - value, 0, 1))


def finger_primary_coordinates(robot: Any, fingers: Sequence[float]) -> tuple[float, float]:
    """Both finger joints in the primary joint's coordinate, the second through the inverse mimic.

    ``robot_config.yml`` drives joint8 as ``mimic[1] * joint7 + mimic[2]``
    (X5: 1 and 0), so a symmetric pair maps to two equal values.
    """
    primary, secondary = (float(value) for value in fingers[:2])
    multiplier, offset = (float(value) for value in robot.gripper_move["mimic"][1:3])
    if not math.isfinite(primary) or not math.isfinite(secondary) or not multiplier:
        raise RobodojoError("finger measurement or mimic coupling is invalid")
    return primary, (secondary - offset) / multiplier


def two_finger_opening(robot: Any, fingers: Sequence[float]) -> float:
    """The public jaw reading (EXP11A g012 E11G12-001): both fingers' mean on the command scale.

    The released reading normalized ``fingers[0]`` alone. Under lateral load the
    two prismatic fingers shift together to one side (by_language seq 61:
    41.4 / 9.5 mm read 0.952 against a 0.657 mean), and the mean is the one
    value that follows both. With symmetric fingers it equals the released
    reading. It is a normalized encoder value, not a calibrated pad aperture.
    """
    primary, secondary = finger_primary_coordinates(robot, fingers)
    return gripper_opening(robot, (primary + secondary) / 2)


def finger_lateral_offset_m(fingers: Sequence[float]) -> float:
    """Signed shift of the finger pair along link6 +Y: ``(joint7 - joint8) / 2`` in metres.

    X5A.urdf opens joint7 (``finger_1``) along link6 +Y and joint8
    (``finger_2``) along -Y, so a pair displaced toward ``finger_1``'s opening
    direction is positive and a symmetric pair is zero.
    """
    return (float(fingers[0]) - float(fingers[1])) / 2


def primary_limit_holds_close(robot: Any, fingers: Sequence[float], *, requested: float | None,
                              drive: float, effective: float | None) -> bool:
    """Whether a close stands where the SDK primary-relative drive limit holds it.

    The SDK limits an explicit gripper command to ``maximum_normalized_offset``
    below the primary finger's encoder and drives both fingers to that target.
    True when the directive is a close, the drive target is above the
    effective close target, it equals the primary finger's reading less that
    limit, and the second finger stands at it -- a factual reading of the two
    fingers and the drive target, not a statement about contact.
    """
    if requested != 0.0 or effective is None:
        return False
    primary, secondary = (gripper_opening(robot, value) for value in finger_primary_coordinates(robot, fingers))
    limit = float(ARX_X5_CONTRACT["gripper_drive_limit"]["maximum_normalized_offset"])
    return bool(drive > effective + GRIPPER_TOLERANCE
                and abs(drive - (primary - limit)) <= GRIPPER_TOLERANCE
                and abs(secondary - drive) <= GRIPPER_TOLERANCE)


def gripper_command_target(gripper_command: Any) -> str:
    """Project one normalized X5 gripper command to its categorical target.

    The argument is the *commanded* directive -- the normalized value read back
    from the SDK control buffer the session wrote -- never a measured finger
    reading. ``open`` and ``close`` are the only directives the ToolSurface
    accepts, so the public read is categorical: the open end of the command
    scale is ``open``, anything at or below the closed floor is ``close``, and
    anything else (including an unpublished command) stays ``unknown`` rather
    than inventing a category. A jaw measuring at the closed floor is a
    commanded ``close`` and nothing more; it is never contact, retention or a
    known pad separation.
    """
    if not isinstance(gripper_command, (int, float)) or isinstance(gripper_command, bool):
        return "unknown"
    value = float(gripper_command)
    if abs(value - float(ARX_X5_CONTRACT["gripper_command"]["open"])) <= COMMAND_TARGET_TOLERANCE:
        return "open"
    if value <= CLOSED_COMMAND_FLOOR + COMMAND_TARGET_TOLERANCE:
        return "close"
    return "unknown"


def physical_gripper(robot: Any, opening: float) -> list[float]:
    low, high = robot.gripper_scale
    value = low + (opening if robot.gripper_move["sign"] == 1 else 1 - opening) * (high - low)
    return [value, value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]


def pose_error(target: Any, measured: Any) -> tuple[float, float]:
    target_array, measured_array = np.asarray(target, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    distance = float(np.linalg.norm(target_array[:3] - measured_array[:3]))
    qa, qb = target_array[3:], measured_array[3:]
    norm = np.linalg.norm(qa) * np.linalg.norm(qb)
    if norm < 1e-8:
        raise RobodojoError("invalid measured orientation quaternion")
    angle = float(2 * math.acos(float(np.clip(abs(np.dot(qa, qb)) / norm, 0, 1))))
    return distance, angle


def pose_reached(target: Any, measured: Any) -> bool:
    distance, angle = pose_error(target, measured)
    return distance <= POSITION_TOLERANCE_M and angle <= ORIENTATION_TOLERANCE_RAD


def workspace_violation(position: Any, bounds: Any) -> str | None:
    if not isinstance(bounds, Mapping):
        raise RobodojoError("workspace_bounds_m is required")
    for index, axis in enumerate(("x", "y", "z")):
        if not bounds[axis][0] <= float(position[index]) <= bounds[axis][1]:
            return axis
    return None


def equivalent_turns(value: float, reference: float, lower: float, upper: float) -> int:
    """Whole turns ``k`` putting ``value + k*2*pi`` nearest ``reference`` inside ``[lower, upper]``.

    Zero unless every input is finite, ``value`` itself lies inside the limits
    and the equivalent is strictly nearer ``reference``: a bounded joint cannot
    cross its stop by taking a shortcut, and a value already outside its limits
    is never moved a whole turn away from the arm.
    """
    if not all(math.isfinite(x) for x in (value, reference, lower, upper)) or not lower <= value <= upper:
        return 0
    first = math.ceil((lower - value) / math.tau)
    last = math.floor((upper - value) / math.tau)
    turns = min(max(round((reference - value) / math.tau), first), last)
    proposed = value + turns * math.tau
    nearer = abs(proposed - reference) < abs(value - reference)
    return turns if turns and nearer and lower <= proposed <= upper else 0


def nearest_equivalent_joints(
    values: Sequence[float], reference: Sequence[float],
    limits: Sequence[Sequence[float]], revolute: Sequence[bool],
) -> tuple[list[float], list[int]]:
    """Each revolute coordinate's legal whole-turn equivalent nearest ``reference``.

    The result is actual command coordinates, never a wrapped distance: the
    joint interpolator drives these values literally. Coordinates not marked
    revolute are returned unchanged. IK candidate selection and the recovery
    unwind target both use this one rule.
    """
    target: list[float] = []
    turns: list[int] = []
    for value, near, bound, periodic in zip(values, reference, limits, revolute, strict=True):
        count = equivalent_turns(float(value), float(near), float(bound[0]), float(bound[1])) if periodic else 0
        turns.append(count)
        target.append(float(value) + count * math.tau)
    return target, turns


def canonical_joints(values: Sequence[float], revolute: Sequence[bool]) -> bool:
    """Whether every revolute coordinate lies in ``CANONICAL_JOINT_INTERVAL_RAD``."""
    lower, upper = CANONICAL_JOINT_INTERVAL_RAD
    return all(lower <= float(value) <= upper
               for value, periodic in zip(values, revolute, strict=True) if periodic)


def canonical_excess(value: float) -> float:
    """How far one coordinate lies outside the symmetric canonical interval (0 inside)."""
    return max(0.0, abs(float(value)) - CANONICAL_JOINT_INTERVAL_RAD[1])


def pitch_joint_wraps(target: Sequence[float], current: Sequence[float], pitch: Sequence[bool]) -> list[int]:
    """Indices of pitch joints ``target`` carries farther outside the canonical interval than ``current``.

    From inside the interval any exit counts; from outside only a move more
    than ``PITCH_WRAP_HOLD_TOLERANCE_RAD`` farther out does, so a wound pitch
    joint may return toward the interval or hold.
    """
    def allowed(now: float) -> float:
        excess = canonical_excess(now)
        return excess + PITCH_WRAP_HOLD_TOLERANCE_RAD if excess > 0 else 0.0

    return [index for index, (goal, now, bending) in enumerate(zip(target, current, pitch, strict=True))
            if bending and canonical_excess(goal) > allowed(now)]


def _degenerate(pose: Sequence[float], reference: Sequence[float]) -> bool:
    distance, angle = pose_error(pose, reference)
    return distance <= 1e-6 and angle <= 1e-6


def free_space_segments(
    session: Any, before: Mapping[str, Any], targets: Mapping[str, Sequence[float]],
) -> list[tuple[str, dict[str, list[float]]]]:
    """Lift to a clearance height, translate at that height, then descend."""
    z_max = float(session.config["workspace_bounds_m"]["z"][1])
    per_arm: dict[str, dict[str, list[float]]] = {}
    for arm, goal in targets.items():
        start = [float(item) for item in before[arm]["pose_xyz_xyzw"]]
        clearance = min(max(start[2], goal[2]) + session.free_space_clearance_m, z_max)
        per_arm[arm] = {
            "lift": [start[0], start[1], clearance, *start[3:]],
            "translate": [goal[0], goal[1], clearance, *goal[3:]],
            "descend": list(goal),
        }
    segments: list[tuple[str, dict[str, list[float]]]] = []
    previous = {arm: list(before[arm]["pose_xyz_xyzw"]) for arm in targets}
    for name in ("lift", "translate", "descend"):
        poses = {arm: per_arm[arm][name] for arm in targets}
        if all(_degenerate(poses[arm], previous[arm]) for arm in targets):
            continue
        segments.append((name, poses))
        previous = poses
    return segments or [("descend", {arm: list(goal) for arm, goal in targets.items()})]


__all__ = [
    "ARX_GRIPPER_STALL", "CANONICAL_JOINT_INTERVAL_RAD", "CLOSED_COMMAND_FLOOR", "COMMAND_TARGET_TOLERANCE",
    "FINGER_JOINT_LIMIT_M", "GRIPPER_COMMAND_SCALE", "GRIPPER_TOLERANCE", "ORIENTATION_TOLERANCE_RAD",
    "PITCH_WRAP_HOLD_TOLERANCE_RAD", "POSITION_TOLERANCE_M",
    "array", "canonical_excess", "canonical_joints", "equivalent_turns", "file_hash", "finger_lateral_offset_m",
    "finger_primary_coordinates", "free_space_segments",
    "gripper_command_target", "gripper_opening", "json_default", "jsonable", "measured_pose",
    "native_pose", "nearest_equivalent_joints", "physical_gripper", "pitch_joint_wraps", "pose_error",
    "primary_limit_holds_close", "two_finger_opening",
    "pose_reached", "workspace_violation", "write_json",
]
