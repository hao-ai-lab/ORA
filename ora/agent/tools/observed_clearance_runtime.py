"""Runtime half of ``observed_clearance_reference_v2``: public inputs and the policy's own records.

``observed_clearance`` (the measurement) is pure. This module gathers what it
measures from, all of which the policy already receives:

- the current observation's depth images and per-observation calibration
  (hash-checked against the observation record, as every camera read is);
- the measured state of every arm (pose, finger positions, last gripper
  command, normalized opening) and each arm's wrist-camera position, which the
  same calibration publishes;
- the public geometry descriptor the surface binds (finger collision bounds,
  opening directions, empty-close encoder reference, jaw contact centre);
- the held object's extent from the policy's own localization of the target it
  named when the jaws closed: ``set_gripper`` close with ``target_ref``, or the
  ``grasp_execute`` candidate's ``target_ref``. The localization's world box
  (bottom lowered to its single-view support height when reported) is fixed to
  link6 at the close pose (the measured pose at ``set_gripper``, the committed
  contact pose for ``grasp_execute``); a box farther than ``RECENTER_MARGIN_M``
  beyond its half extent from the jaw centre is moved under the jaws and says so.

An arm is holding when its last gripper command is ``close`` and its measured
opening exceeds the descriptor's empty-close reference by more than the arm's
published arrival tolerance. No failure here blocks or changes a move.

The inputs are read through a per-embodiment jaw model
(``observed_clearance_geometry``): where the controlled point is
already the grip centre (ALOHA, Panda) the jaw centre is that point, fingers
open by the published measurement (Panda signed finger positions, ALOHA half
the measured jaw separation each), and the jaws hold when the last command was
``close`` and the measured jaw separation exceeds the Harness's near-closed
bound. The defaults of :class:`JawGeometry` are the released v1 rules, which
the ARX X5 model keeps.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import observed_clearance as oc

RECENTER_MARGIN_M = 0.02
_ROOT = Path(__file__).resolve().parents[3]
_HOLDS = "_observed_clearance_holds"
#: Per-host caches, read and written by name like ``_HOLDS``.
_GEOMETRY = "_observed_clearance_geometry"
_VIEWS = "_observed_clearance_views"
V2_CAPABILITY = "observed_clearance_reference_v2"
#: Finger opening read from the measured state (``_qpos``).
OPENING_ENCODER = "finger_qpos_minus_reference"
OPENING_SIGNED = "signed_finger_qpos"
OPENING_SYMMETRIC = "half_measured_jaw_separation"
#: Holding test (``jaws_hold``).
HOLD_ENCODER = "normalized_opening_above_empty_close_reference"
HOLD_SEPARATION = "measured_jaw_separation_above_near_closed"


@dataclass(frozen=True)
class JawGeometry:
    fingers: tuple[oc.FingerBounds, ...]
    reference_finger_qpos: tuple[float, ...]
    empty_close_reference: float | None
    contact_center_local: tuple[float, float, float]
    wrist_cameras: tuple[str, ...]
    finger_opening: str = OPENING_ENCODER
    finger_signs: tuple[float, ...] = ()
    hold_rule: str = HOLD_ENCODER
    near_closed_m: float | None = None
    root_face: tuple[int, ...] = oc.ROOT_FACE


@dataclass(frozen=True)
class HeldExtent:
    target_ref: str | None
    local_corners: Any = None
    recentered: bool = False


def load_jaw_geometry(path: Path) -> dict[str, JawGeometry]:
    """Per-arm jaw geometry from a public embodiment descriptor."""
    from ora.agent.composition.embodiment import load_embodiment_descriptor

    descriptor = load_embodiment_descriptor(Path(path))
    points = {point["id"]: point for point in descriptor["control"]["controlled_points"]}
    cameras = descriptor.get("sensing", {}).get("cameras", [])
    result: dict[str, JawGeometry] = {}
    for gripper in descriptor["geometry"]["grippers"]:
        jaws, parameters = gripper.get("contact_geometry") or {}, gripper.get("reference_parameters") or {}
        point = points[jaws["controlled_point"]]
        manipulator = str(point["manipulator_id"])
        arm = manipulator.removesuffix("_arm")
        approach = [float(value) for value in point["local_axes"]["approach"]]
        distance = float(parameters["contact_center_distance_m"])
        result[arm] = JawGeometry(
            fingers=tuple(oc.FingerBounds(minimum=finger["collision_bounds_m"]["minimum"],
                                          maximum=finger["collision_bounds_m"]["maximum"],
                                          opening_direction=finger["opening_direction_local"])
                          for finger in jaws["fingers"]),
            reference_finger_qpos=tuple(float(value) for value in jaws["reference_finger_qpos_m"]),
            empty_close_reference=float(parameters["empty_close_encoder_reference"]),
            contact_center_local=tuple(float(base) + distance * axis
                                       for base, axis in zip(point["offset_m"], approach)),
            wrist_cameras=tuple(str(camera["public_name"]) for camera in cameras
                                if camera.get("attached_to") == manipulator),
        )
    return result


def jaws_hold(effector: Mapping[str, Any], geometry: JawGeometry, observation: Mapping[str, Any]) -> bool:
    if effector.get("gripper_command_target") != "close":
        return False
    if geometry.hold_rule == HOLD_SEPARATION:
        from .localization import measured_jaw_separation_m

        separation = measured_jaw_separation_m(effector)
        return separation is not None and separation > float(geometry.near_closed_m)
    feedback = (effector.get("gripper_feedback") or {}).get("arrival_check") or {}
    tolerance = feedback.get("tolerance_normalized",
                             (observation.get("controller_feedback") or {}).get("gripper_tolerance_normalized"))
    opening = effector.get("gripper")
    if not isinstance(tolerance, (int, float)) or not isinstance(opening, (int, float)):
        return False
    return float(opening) - geometry.empty_close_reference > float(tolerance)


def _qpos(effector: Mapping[str, Any], geometry: JawGeometry) -> list[float]:
    """Each finger's displacement along its published opening direction from the reference opening."""
    if geometry.finger_opening == OPENING_SYMMETRIC:
        from .localization import measured_jaw_separation_m

        separation = measured_jaw_separation_m(effector)
        if separation is None:
            raise ValueError("measured jaw separation is unavailable")
        return [separation / len(geometry.fingers)] * len(geometry.fingers)
    values = effector.get("gripper_finger_qpos")
    if geometry.finger_opening == OPENING_SIGNED:
        if not isinstance(values, Sequence) or len(values) != len(geometry.finger_signs):
            raise ValueError("measured finger positions are unavailable")
        return [sign * (float(value) - reference) for sign, value, reference
                in zip(geometry.finger_signs, values, geometry.reference_finger_qpos, strict=True)]
    if not isinstance(values, Sequence) or len(values) != len(geometry.fingers):
        raise ValueError("measured finger positions are unavailable")
    return [float(value) - reference for value, reference in zip(values, geometry.reference_finger_qpos)]


def _views(observation: Mapping[str, Any], root: Path) -> tuple[list[oc.DepthView], dict[str, Any]]:
    views, camera_positions = [], {}
    resolved = Path(root).resolve()
    for name, record in sorted((observation.get("cameras") or {}).items()):
        files = record.get("files") or {}
        paths = {}
        for role in ("depth_numpy_m_f32", "calibration"):
            reference = files.get(role) or {}
            path = (resolved / str(reference.get("path", ""))).resolve()
            if (not path.is_file() or not path.is_relative_to(resolved)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != reference.get("sha256")):
                break
            paths[role] = path
        if len(paths) != 2:
            continue
        view = oc.view_from_files(name, paths["depth_numpy_m_f32"],
                                  json.loads(paths["calibration"].read_text(encoding="utf-8")))
        views.append(view)
        camera_positions[name] = np.asarray(view.camera_to_world, float)[:3, 3]
    return views, camera_positions


def annotate_observation(
    observation: Mapping[str, Any], root: Path, *, arm: str, target_pose: Sequence[float],
    geometry: Mapping[str, JawGeometry], held: Mapping[str, HeldExtent],
    views: tuple[list[oc.DepthView], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ``observed_clearance`` value for one arm's commanded target."""
    effectors = (observation.get("proprioception") or {}).get("end_effectors") or {}
    if arm not in geometry or arm not in effectors:
        return _unknown("geometry_unavailable")
    views, camera_positions = views if views is not None else _views(observation, root)
    robot, holding = [], {}
    for name, effector in sorted(effectors.items()):
        if name not in geometry:
            continue
        jaws = geometry[name]
        holding[name] = jaws_hold(effector, jaws, observation)
        extent = held.get(name) if holding[name] else None
        pose = [float(value) for value in effector["pose_xyz_xyzw"]]
        measured = oc.footprint_at(pose, finger_qpos=_qpos(effector, jaws), fingers=jaws.fingers,
                                   held_local_corners=None if extent is None else extent.local_corners)
        robot.append(oc.robot_volume(measured.solids, controlled_point=pose[:3], extra_points=[
            camera_positions[camera] for camera in jaws.wrist_cameras if camera in camera_positions],
            root_indices=jaws.root_face))
    extent = held.get(arm) if holding.get(arm) else None
    footprint = oc.footprint_at([float(value) for value in target_pose],
                                finger_qpos=_qpos(effectors[arm], geometry[arm]), fingers=geometry[arm].fingers,
                                held_local_corners=None if extent is None else extent.local_corners)
    result = oc.measure(views, observation_id=str(observation.get("observation_id", "")), footprint=footprint,
                        robot=robot)
    result["held_object"] = None if not holding.get(arm) else {
        "target_ref": None if extent is None else extent.target_ref,
        "extent_included": extent is not None and extent.local_corners is not None,
        "recentered_under_jaws": bool(extent is not None and extent.recentered),
    }
    return result


def _unknown(reason: str) -> dict[str, Any]:
    return {"status": "unknown", "reason": reason, "radius_m": oc.RADIUS_M, "claim_boundary": oc.CLAIM_BOUNDARY}


def held_local_corners(
    localization: Mapping[str, Any], attach_pose: Sequence[float], contact_center_local: Sequence[float],
) -> tuple[Any, bool] | None:
    """A localization's world box fixed to link6 at the close pose, or None when unmeasured."""
    low, high = localization.get("aabb_min_xyz_m"), localization.get("aabb_max_xyz_m")
    if localization.get("status", "localized") != "localized" or low is None or high is None:
        return None
    low, high = np.asarray(low, float), np.asarray(high, float)
    support = localization.get("support_z_m")
    if isinstance(support, (int, float)) and math.isfinite(float(support)):
        low[2] = min(low[2], float(support))
    turn, origin = oc.rotation(attach_pose[3:]), np.asarray(attach_pose[:3], float)
    jaw = origin + turn @ np.asarray(contact_center_local, float)
    centre, half = (low + high) / 2.0, (high - low) / 2.0
    recentered = bool((np.abs(jaw[:2] - centre[:2]) > half[:2] + RECENTER_MARGIN_M).any())
    if recentered:
        low[:2], high[:2] = low[:2] + (jaw[:2] - centre[:2]), high[:2] + (jaw[:2] - centre[:2])
    return (oc.box_corners(low, high) - origin) @ turn, recentered


# --------------------------------------------------------------------------
# Host integration: the manipulation Tool Runtime.
# --------------------------------------------------------------------------

def declared(host: Any) -> bool:
    surface = getattr(host, "surface", None)
    return bool(surface is not None and surface.supports(V2_CAPABILITY))


def _holds(host: Any) -> dict[str, dict[str, Any]]:
    holds = getattr(host, _HOLDS, None)
    if holds is None:
        holds = {}
        setattr(host, _HOLDS, holds)
    return holds


def note_gripper(host: Any, targets: Mapping[str, str], *, target_ref: str | None,
                 observation: Mapping[str, Any]) -> None:
    """Record which target the policy named when one arm's jaws closed; forget it on open.

    Runs after the action is submitted, so it never raises: a record it cannot
    make leaves that arm's held extent unavailable. A close without a name
    keeps an earlier record (a re-close on the same object).
    """
    if not declared(host):
        return
    holds = _holds(host)
    closing = [arm for arm, directive in targets.items() if directive == "close"]
    for arm, directive in targets.items():
        if directive == "open":
            holds.pop(arm, None)
    if not target_ref or len(closing) != 1:
        return
    try:
        pose = [float(value) for value in observation["proprioception"]["end_effectors"][closing[0]]["pose_xyz_xyzw"]]
    except (KeyError, TypeError, ValueError):
        holds.pop(closing[0], None)
        return
    holds[closing[0]] = {"target_ref": target_ref, "attach_pose": pose, "source": "set_gripper"}


def note_grasp_execute(host: Any) -> None:
    """Record the committed grasp's target and contact pose for its arm (never raises)."""
    if not declared(host):
        return
    from .skill_driver import load_commitment

    try:
        commitment = load_commitment(host) or {}
        approach = (commitment.get("poses") or {}).get("approach") or {}
        if commitment.get("skill") != "grasp_execute" or not commitment.get("source_ref") or not approach:
            return
        _holds(host)[str(commitment["arm"])] = {
            "target_ref": str(commitment["source_ref"]), "source": "grasp_execute",
            "attach_pose": [float(value) for value in (*approach["position_m"], *approach["orientation_xyzw"])]}
    except (OSError, ValueError, KeyError, TypeError):
        return


def after_skill(host: Any, skill: str, result: Any) -> Any:
    """Return a begun skill's result unchanged, after noting a ``grasp_execute`` commitment."""
    if skill == "grasp_execute":
        note_grasp_execute(host)
    return result


def _held_extent(host: Any, arm: str, geometry: JawGeometry) -> HeldExtent | None:
    record = _holds(host).get(arm)
    if record is None:
        return None
    if "local_corners" not in record:
        from .grasp import load_segmented_target

        try:
            manifest, _mask, _overlay = load_segmented_target(artifact_root=host.artifact_root,
                                                              target_ref=record["target_ref"])
            record["local_corners"] = held_local_corners(manifest.get("localization") or {},
                                                         record["attach_pose"], geometry.contact_center_local)
        except (OSError, ValueError, KeyError, TypeError):
            record["local_corners"] = None
    measured = record["local_corners"]
    return HeldExtent(target_ref=record["target_ref"], local_corners=None if measured is None else measured[0],
                      recentered=bool(measured and measured[1]))


def _geometry(host: Any) -> dict[str, JawGeometry]:
    view = host.surface.view_config or {}
    binding = view.get("public_geometry_descriptor") or {}
    convention = str(view.get("controlled_frame_convention"))
    key = (binding.get("sha256"), convention)
    cached = getattr(host, _GEOMETRY, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    path = (_ROOT / str(binding.get("path", ""))).resolve()
    if (not path.is_file() or not path.is_relative_to(_ROOT)
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")):
        raise ValueError("public geometry descriptor is missing or drifted")
    from .observed_clearance_geometry import load_jaw_geometry_v2

    geometry = load_jaw_geometry_v2(path, convention)
    setattr(host, _GEOMETRY, (key, geometry))
    return geometry


def _cached_views(host: Any, observation: Mapping[str, Any], root: Path) -> tuple[list[oc.DepthView], dict]:
    """One read of the observation's depth per observation, shared by every target of a call."""
    key = (str(observation.get("observation_id", "")), str(Path(root).resolve()))
    cached = getattr(host, _VIEWS, None)
    if cached is None or cached[0] != key:
        cached = (key, _views(observation, root))
        setattr(host, _VIEWS, cached)
    return cached[1]


def reference(host: Any, arm: str, target_pose_xyz_xyzw: Sequence[float], *, observation: Mapping[str, Any],
              root: Path) -> dict[str, Any]:
    """``{"observed_clearance": ...}`` for one compiled target on a declaring surface, else nothing."""
    if not declared(host):
        return {}
    try:
        geometry = _geometry(host)
        held = {name: _held_extent(host, name, jaws) for name, jaws in geometry.items()}
        value = annotate_observation(observation, root, arm=arm, target_pose=target_pose_xyz_xyzw,
                                     geometry=geometry, held={k: v for k, v in held.items() if v is not None},
                                     views=_cached_views(host, observation, root))
    except Exception:  # an observed reference never blocks a move
        value = _unknown("reference_failed")
    return {"observed_clearance": value}


__all__ = [
    "HOLD_ENCODER", "HOLD_SEPARATION", "HeldExtent", "JawGeometry", "OPENING_ENCODER", "OPENING_SIGNED",
    "OPENING_SYMMETRIC", "RECENTER_MARGIN_M", "V2_CAPABILITY", "after_skill", "annotate_observation", "declared",
    "held_local_corners",
    "jaws_hold", "load_jaw_geometry", "note_grasp_execute", "note_gripper", "reference",
]
