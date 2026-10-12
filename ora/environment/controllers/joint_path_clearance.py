"""Straight joint-path self-collision screening with a solver's own sphere model.

An IK endpoint that the solver checked for self-collision can still be reached
only through self-collision when the joint interpolator drives every joint
linearly from the current configuration. EXP11A g001 action 0121 is the
recorded case: the RoboDojo SDK IK admitted an endpoint with ARX X5 joint3 at
-2.59 rad from a folded start at +0.02 rad. The plant stopped joint3 at -0.036
rad on the first native action while joint1 kept sweeping, ending 0.515 m and
2.53 rad from the target. Sampled along that straight joint path, the SDK's own
CuRobo collision spheres interpenetrate (base_link/link4, -57 mm); none of the
other 77 executed g001/g002 segments does.

This module is host-safe NumPy. It reads a URDF kinematic chain and a
CuRobo-style ``collision_spheres`` / ``self_collision_ignore`` /
``self_collision_buffer`` description, exactly as the solver loaded them. The
result is sphere-model evidence: a clear path is not a collision-free
certificate, and nothing here models the scene, the table or a held object.

``InterArmSphereScreen`` places two arms' sphere models at their world bases and
screens the moving arm(s) against the other one (EXP11A g012, E11G12-004a): 15
of 80 recorded RoboDojo stalls were one ARX arm driving into the other, which no
per-arm self-collision check can see.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = "ora.joint-path-self-clearance.v1"
#: Largest joint change between two consecutive path samples. At the ARX X5
#: link lengths (<0.3 m) 0.05 rad moves any sphere under 15 mm per sample.
DEFAULT_MAX_STEP_RAD = 0.05
CLAIM_BOUNDARY = (
    "solver self-collision spheres along the straight joint interpolation; "
    "not a scene, table, payload or mesh collision certificate"
)


INTER_ARM_SCHEMA = "ora.joint-path-inter-arm-clearance.v1"
#: Sphere-surface clearance a joint path must keep between the two arms.
#: Replayed over all 982 executed EXP11A g012 arm moves (measured start, the
#: recorded IK joint targets, the other arm held at its measured joints or moved
#: with it in lockstep), a 0 or 5 mm margin refuses all 15 moves that stalled
#: against the other arm and 11 of the 902 that did not stall; on its executed
#: trajectory each of those 11 came within 2.0 mm of the other arm's ARX.usd
#: collision hulls (10 within 0.1 mm or overlapping), so the arms touched. A
#: 10 mm margin adds one grasp_execute (three stages) whose hulls stayed 8.6 to
#: 12.4 mm apart. 5 mm is the largest margin with no such refusal; it absorbs
#: float and settling noise in the measured start.
INTER_ARM_MINIMUM_CLEARANCE_M = 0.005
#: Largest joint change between two lockstep samples of the inter-arm screen.
#: The X5 base-to-link6 chain is at most 0.822 m and its link6 spheres reach
#: about 0.08 m further, so 0.02 rad moves any sphere at most 18 mm per sample
#: per arm (the self-collision step's 0.05 rad, up to 45 mm, is coarse beside a
#: 5 mm margin). The sampled minimum can still overestimate the continuous one.
#: On the 982 recorded g012 moves 0.05 and 0.02 rad refuse exactly the same
#: moves; 0.02 rad costs about 25 ms per move there (a 3 rad sweep, 90 ms).
INTER_ARM_MAX_STEP_RAD = 0.02
INTER_ARM_CLAIM_BOUNDARY = (
    "solver collision spheres of both arms at their world bases along the straight joint "
    "interpolation of the moving arm(s), in lockstep; the finger links are not in the "
    "sphere model; not a scene, table, payload or mesh collision certificate"
)


class JointPathClearanceError(ValueError):
    """The kinematic chain or sphere model cannot be evaluated honestly."""


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    values = [default] * count if text is None else [float(item) for item in text.split()]
    if len(values) != count or not all(math.isfinite(item) for item in values):
        raise JointPathClearanceError("URDF origin or axis is not a finite vector")
    return values


def parse_urdf_joints(urdf_text: str) -> list[_Joint]:
    root = ET.fromstring(urdf_text)
    joints = []
    for node in root.findall("joint"):
        kind = node.get("type")
        if kind not in {"revolute", "continuous", "prismatic", "fixed"}:
            raise JointPathClearanceError(f"unsupported URDF joint type: {kind}")
        origin_node = node.find("origin")
        xyz = _floats(None if origin_node is None else origin_node.get("xyz"), 3)
        rpy = _floats(None if origin_node is None else origin_node.get("rpy"), 3)
        origin = np.eye(4)
        origin[:3, :3], origin[:3, 3] = _rpy_matrix(*rpy), xyz
        axis_node = node.find("axis")
        axis = np.asarray(_floats(None if axis_node is None else axis_node.get("xyz"), 3, 0.0)
                          if axis_node is not None else [1.0, 0.0, 0.0])
        if kind != "fixed":
            norm = float(np.linalg.norm(axis))
            if not norm:
                raise JointPathClearanceError("URDF moving joint has a zero axis")
            axis = axis / norm
        joints.append(_Joint(str(node.get("name")), kind, node.find("parent").get("link"),
                             node.find("child").get("link"), origin, axis))
    return joints


class SphereSelfCollisionModel:
    """Link spheres of one serial arm and the link pairs its solver checks."""

    def __init__(self, *, joints: Sequence[_Joint], base_link: str, active_joint_names: Sequence[str],
                 spheres: Mapping[str, Sequence[Mapping[str, Any]]],
                 ignore: Mapping[str, Sequence[str]] | None = None,
                 buffers: Mapping[str, float] | None = None) -> None:
        self.active_joint_names = list(active_joint_names)
        by_name = {joint.name: joint for joint in joints}
        if len(by_name) != len(joints) or not set(self.active_joint_names) <= set(by_name):
            raise JointPathClearanceError("active joints are absent from the kinematic chain")
        order, known = [], {base_link}
        pending = list(joints)
        while pending:
            ready = [joint for joint in pending if joint.parent in known]
            if not ready:
                break
            for joint in ready:
                order.append(joint)
                known.add(joint.child)
                pending.remove(joint)
        self._order = order
        self._spheres: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        #: Radii without the self-collision buffer, for checks against another body.
        self._raw_radii: dict[str, np.ndarray] = {}
        buffers = dict(buffers or {})
        for link, rows in spheres.items():
            if link not in known:
                raise JointPathClearanceError(f"sphere link {link} is not connected to {base_link}")
            centers = np.asarray([row["center"] for row in rows], dtype=float).reshape(-1, 3)
            radii = np.asarray([row["radius"] for row in rows], dtype=float).reshape(-1)
            if (len(centers) != len(radii) or not np.isfinite(centers).all()
                    or not np.isfinite(radii).all() or np.any(radii < 0)):
                raise JointPathClearanceError(f"sphere model for {link} is invalid")
            self._spheres[link] = (centers, radii + float(buffers.get(link, 0.0)))
            self._raw_radii[link] = radii
        ignore = {key: set(value or []) for key, value in (ignore or {}).items()}
        self.pairs = tuple(
            (a, b) for a, b in itertools.combinations(sorted(self._spheres), 2)
            if b not in ignore.get(a, set()) and a not in ignore.get(b, set())
            and len(self._spheres[a][1]) and len(self._spheres[b][1])
        )
        if not self.pairs:
            raise JointPathClearanceError("sphere model checks no link pairs")

    @classmethod
    def from_solver_kinematics(cls, kinematics: Mapping[str, Any], *, urdf_text: str,
                               active_joint_names: Sequence[str]) -> "SphereSelfCollisionModel":
        """Bind the exact CuRobo-style kinematics mapping a solver was built from."""
        required = ("base_link", "collision_spheres")
        if not isinstance(kinematics, Mapping) or any(key not in kinematics for key in required):
            raise JointPathClearanceError("solver kinematics lack a base link or collision spheres")
        return cls(joints=parse_urdf_joints(urdf_text), base_link=str(kinematics["base_link"]),
                   active_joint_names=active_joint_names, spheres=kinematics["collision_spheres"],
                   ignore=kinematics.get("self_collision_ignore"),
                   buffers=kinematics.get("self_collision_buffer"))

    def _link_frames(self, joints: Sequence[float]) -> dict[str, np.ndarray]:
        values = np.asarray(joints, dtype=float).reshape(-1)
        if values.shape != (len(self.active_joint_names),) or not np.isfinite(values).all():
            raise JointPathClearanceError("joint configuration is not finite or has the wrong size")
        position = dict(zip(self.active_joint_names, values.tolist()))
        frames = {self._order[0].parent: np.eye(4)} if self._order else {}
        for joint in self._order:
            transform = joint.origin.copy()
            angle = position.get(joint.name, 0.0)
            if joint.kind in {"revolute", "continuous"}:
                motion = np.eye(4)
                motion[:3, :3] = _axis_angle(joint.axis, angle)
                transform = transform @ motion
            elif joint.kind == "prismatic":
                motion = np.eye(4)
                motion[:3, 3] = joint.axis * angle
                transform = transform @ motion
            frames[joint.child] = frames[joint.parent] @ transform
        return frames

    def link_spheres(self, joints: Sequence[float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Every sphere's base-frame centre, unbuffered radius and link, for another body's check."""
        frames = self._link_frames(joints)
        centers, radii, links = [], [], []
        for link, (local, _) in self._spheres.items():
            frame = frames[link]
            centers.append(local @ frame[:3, :3].T + frame[:3, 3])
            radii.append(self._raw_radii[link])
            links.extend([link] * len(local))
        return np.vstack(centers), np.concatenate(radii), links

    def clearance(self, joints: Sequence[float]) -> tuple[float, tuple[str, str]]:
        """Smallest sphere surface gap over checked pairs (negative = overlap)."""
        frames = self._link_frames(joints)
        world = {}
        for link, (centers, radii) in self._spheres.items():
            frame = frames[link]
            world[link] = (centers @ frame[:3, :3].T + frame[:3, 3], radii)
        worst, pair = math.inf, self.pairs[0]
        for a, b in self.pairs:
            (ca, ra), (cb, rb) = world[a], world[b]
            gaps = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1) - ra[:, None] - rb[None, :]
            value = float(gaps.min())
            if value < worst:
                worst, pair = value, (a, b)
        return worst, pair

    def straight_path(self, start: Sequence[float], end: Sequence[float], *,
                      max_step_rad: float = DEFAULT_MAX_STEP_RAD) -> dict[str, Any]:
        """Screen the linear joint interpolation from ``start`` to ``end``.

        A path is blocked when some sample after the start overlaps and is
        deeper than the start itself. A start already inside the sphere model
        therefore never forbids a motion that does not make the overlap worse.
        """
        first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if first.shape != last.shape or not np.isfinite([max_step_rad]).all() or max_step_rad <= 0:
            raise JointPathClearanceError("joint path endpoints or sampling step are invalid")
        span = float(np.max(np.abs(last - first))) if first.size else 0.0
        count = max(1, math.ceil(span / max_step_rad))
        start_gap, _ = self.clearance(first)
        worst = (math.inf, None, None)
        for index in range(1, count + 1):
            fraction = index / count
            gap, pair = self.clearance(first + (last - first) * fraction)
            if gap < worst[0]:
                worst = (gap, pair, fraction)
        blocked = worst[0] < 0.0 and worst[0] < start_gap - 1e-9
        return {"schema": SCHEMA, "status": "self_collision" if blocked else "clear",
                "start_clearance_m": start_gap, "min_clearance_m": worst[0],
                "limiting_pair": list(worst[1]) if worst[1] else None,
                "limiting_fraction": worst[2], "samples": count, "max_step_rad": float(max_step_rad),
                "claim_boundary": CLAIM_BOUNDARY}


def _quaternion_matrix(wxyz: Sequence[float]) -> np.ndarray:
    values = np.asarray(wxyz, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(values)) if values.shape == (4,) and np.isfinite(values).all() else 0.0
    if norm < 1e-9:
        raise JointPathClearanceError("arm base quaternion is not a finite non-zero wxyz rotation")
    w, x, y, z = values / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class InterArmSphereScreen:
    """Two arms' solver spheres placed at their world bases, screened against each other.

    Radii are the solver's own, inflated only by its world-collision buffer:
    the self-collision buffer describes an arm against itself, not another body.
    """

    def __init__(self, models: Mapping[str, SphereSelfCollisionModel], bases: Mapping[str, Any], *,
                 minimum_clearance_m: float = INTER_ARM_MINIMUM_CLEARANCE_M,
                 sphere_buffers_m: Mapping[str, float] | None = None) -> None:
        if len(models) != 2 or set(models) != set(bases):
            raise JointPathClearanceError("inter-arm screening needs exactly two placed arms")
        self.arms = tuple(sorted(models))
        self._models = dict(models)
        self._bases: dict[str, np.ndarray] = {}
        for arm in self.arms:
            base = np.asarray(bases[arm], dtype=float)
            if (base.shape != (4, 4) or not np.isfinite(base).all() or not np.allclose(base[3], [0, 0, 0, 1])
                    or not np.allclose(base[:3, :3] @ base[:3, :3].T, np.eye(3), atol=1e-6)):
                raise JointPathClearanceError(f"{arm} arm base is not a finite rigid transform")
            self._bases[arm] = base
        buffers = {arm: float((sphere_buffers_m or {}).get(arm, 0.0)) for arm in self.arms}
        if not all(math.isfinite(value) for value in [minimum_clearance_m, *buffers.values()]) \
                or minimum_clearance_m < 0:
            raise JointPathClearanceError("inter-arm clearance margin or sphere buffer is invalid")
        self._buffers = buffers
        self.minimum_clearance_m = float(minimum_clearance_m)

    @staticmethod
    def base_transform(position: Sequence[float], quaternion_wxyz: Sequence[float],
                       frame_bias: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """World transform of a solver base: ``world = R (p - frame_bias) + position``."""
        rotation = _quaternion_matrix(quaternion_wxyz)
        offset, bias = (np.asarray(value, dtype=float).reshape(-1) for value in (position, frame_bias))
        if offset.shape != (3,) or bias.shape != (3,) or not np.isfinite([*offset, *bias]).all():
            raise JointPathClearanceError("arm base position or frame bias is not a finite 3-vector")
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, offset - rotation @ bias
        return transform

    def _world(self, arm: str, joints: Sequence[float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        centers, radii, links = self._models[arm].link_spheres(joints)
        base = self._bases[arm]
        return centers @ base[:3, :3].T + base[:3, 3], radii + self._buffers[arm], links

    def clearance(self, joints: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, str]]:
        """Smallest surface gap between the two arms' spheres (negative = overlap) and its links."""
        first, second = self.arms
        ca, ra, la = self._world(first, joints[first])
        cb, rb, lb = self._world(second, joints[second])
        gaps = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1) - ra[:, None] - rb[None, :]
        i, j = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        return float(gaps[i, j]), {first: la[i], second: lb[j]}

    def straight_paths(self, starts: Mapping[str, Sequence[float]], ends: Mapping[str, Sequence[float]], *,
                       max_step_rad: float = INTER_ARM_MAX_STEP_RAD) -> dict[str, Any]:
        """Screen both arms' straight joint lines in lockstep; a held arm has ``end == start``.

        Blocked when some sample after the start comes closer than the margin
        and closer than the start itself: arms that already touch may separate.
        """
        if set(starts) != set(self.arms) or set(ends) != set(self.arms):
            raise JointPathClearanceError("inter-arm screening needs a start and an end for both arms")
        first = {arm: np.asarray(starts[arm], dtype=float) for arm in self.arms}
        last = {arm: np.asarray(ends[arm], dtype=float) for arm in self.arms}
        if (any(first[arm].shape != last[arm].shape for arm in self.arms)
                or not math.isfinite(max_step_rad) or max_step_rad <= 0):
            raise JointPathClearanceError("joint path endpoints or sampling step are invalid")
        span = max(float(np.max(np.abs(last[arm] - first[arm]))) if first[arm].size else 0.0 for arm in self.arms)
        count = max(1, math.ceil(span / max_step_rad))
        start_gap, _ = self.clearance(first)
        worst: tuple[float, dict[str, str] | None, float | None] = (math.inf, None, None)
        for index in range(1, count + 1):
            fraction = index / count
            gap, links = self.clearance({arm: first[arm] + (last[arm] - first[arm]) * fraction for arm in self.arms})
            if gap < worst[0]:
                worst = (gap, links, fraction)
        blocked = worst[0] < self.minimum_clearance_m and worst[0] < start_gap - 1e-9
        return {"schema": INTER_ARM_SCHEMA, "status": "inter_arm_collision" if blocked else "clear",
                "moving_arms": [arm for arm in self.arms if not np.array_equal(first[arm], last[arm])],
                "start_clearance_m": start_gap, "min_clearance_m": worst[0],
                "required_clearance_m": self.minimum_clearance_m, "limiting_links": worst[1],
                "limiting_fraction": worst[2], "samples": count, "max_step_rad": float(max_step_rad),
                "claim_boundary": INTER_ARM_CLAIM_BOUNDARY}


__all__ = ["CLAIM_BOUNDARY", "DEFAULT_MAX_STEP_RAD", "INTER_ARM_CLAIM_BOUNDARY", "INTER_ARM_MAX_STEP_RAD",
           "INTER_ARM_MINIMUM_CLEARANCE_M",
           "INTER_ARM_SCHEMA", "InterArmSphereScreen", "JointPathClearanceError", "SCHEMA",
           "SphereSelfCollisionModel", "parse_urdf_joints"]
