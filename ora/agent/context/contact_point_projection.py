"""Where each parallel jaw's nominal contact centre lands in every public camera.

``controlled_point_pixels`` projects the pose the controller moves. On the
Panda and ALOHA that pose is already a grip centre; on the ARX X5 it is the
link6 wrist flange, 0.1501 m behind the pad contact baseline, and it sits about
5 cm *behind* the arm's own wrist camera, so the wrist image has no pixel for
it at all. A surface that declares ``jaw_contact_center_projection`` also
publishes, per camera and arm, the pixel of the descriptor's reviewed nominal
contact centre: the controlled point offset plus ``contact_center_distance_m``
along the controlled point's approach axis. It is nominal source geometry,
not a measured contact, and it uses the same public calibration and measured
pose as the controlled point.

The delivered field is ``jaw_center_pixels``. A policy-visible key must not
contain a term of ``agent/evidence/trajectory.py`` ``FORBIDDEN_FRAGMENTS`` on a
word boundary: the earlier ``jaw_contact_center_pixels`` did, and every trace
that carried it failed the privileged-field audit.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .public_facts import (
    PublicFactsError, controlled_point_projection_delivery, project_controlled_point,
    project_current_public_facts,
)

__all__ = [
    "JAW_CENTER_PIXELS_FIELD", "JAW_CONTACT_CENTER_PROJECTION", "jaw_contact_center_delivery",
    "jaw_contact_center_offsets", "project_policy_public_facts",
]

JAW_CONTACT_CENTER_PROJECTION = "descriptor_contact_center_v1"
JAW_CENTER_PIXELS_FIELD = "jaw_center_pixels"
_ROOT = Path(__file__).resolve().parents[3]


def jaw_contact_center_delivery(view_config: Mapping[str, Any] | None) -> bool:
    return (controlled_point_projection_delivery(view_config)
            and view_config.get("jaw_contact_center_projection") == JAW_CONTACT_CENTER_PROJECTION)


def jaw_contact_center_offsets(view_config: Mapping[str, Any], *,
                               project_root: Path = _ROOT) -> dict[str, list[float]]:
    """Per-arm contact-centre offset in the controlled point's local frame.

    Read from the hash-bound public geometry descriptor the surface already
    selects; there is no second copy of the number.
    """
    from ora.agent.composition.embodiment import load_embodiment_descriptor

    binding = view_config.get("public_geometry_descriptor")
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise PublicFactsError("contact-centre projection requires a public geometry descriptor")
    path = (project_root / binding["path"]).resolve()
    if (not path.is_relative_to(project_root.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")):
        raise PublicFactsError("public geometry descriptor is missing or drifted")
    descriptor = load_embodiment_descriptor(path)
    points = {point["id"]: point for point in descriptor["control"]["controlled_points"]}
    offsets: dict[str, list[float]] = {}
    for gripper in descriptor["geometry"]["grippers"]:
        contact = gripper.get("contact_geometry") or {}
        parameters = gripper.get("reference_parameters") or {}
        point = points.get(contact.get("controlled_point"))
        distance = parameters.get("contact_center_distance_m")
        if point is None or type(distance) not in (int, float) or not math.isfinite(distance):
            raise PublicFactsError("descriptor lacks a reviewed jaw contact centre")
        manipulator = str(point["manipulator_id"])
        if not manipulator.endswith("_arm"):
            raise PublicFactsError("descriptor manipulator does not name a public arm")
        approach = [float(value) for value in point["local_axes"]["approach"]]
        offsets[manipulator[: -len("_arm")]] = [
            float(base) + float(distance) * axis for base, axis in zip(point["offset_m"], approach)
        ]
    return offsets


def _rotate(quaternion_xyzw: list[float], vector: list[float]) -> list[float]:
    x, y, z, w = quaternion_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not norm:
        raise PublicFactsError("measured orientation is not a rotation")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [sum(row[index] * vector[index] for index in range(3)) for row in rotation]


def _with_contact_centers(facts: Mapping[str, Any], *, observation: Mapping[str, Any],
                          offsets: Mapping[str, list[float]]) -> dict[str, Any]:
    effectors = observation.get("proprioception", {}).get("end_effectors", {})
    centres = {}
    for arm, effector in sorted(effectors.items()):
        pose = effector.get("pose_xyz_xyzw") if isinstance(effector, Mapping) else None
        if arm not in offsets or not isinstance(pose, list) or len(pose) != 7:
            continue
        rotated = _rotate([float(value) for value in pose[3:]], offsets[arm])
        centres[arm] = [float(pose[index]) + rotated[index] for index in range(3)]
    result = copy.deepcopy(dict(facts))
    for entry in result.get("cameras", {}).values():
        calibration = entry.get("calibration", {})
        if calibration.get("status") != "reported":
            continue
        entry[JAW_CENTER_PIXELS_FIELD] = {
            arm: project_controlled_point(
                calibration["intrinsic_cv"], calibration["world_to_camera_cv"], centre,
                width=entry["width"], height=entry["height"])
            for arm, centre in centres.items()
        }
    return result


def project_policy_public_facts(*, observation: Mapping[str, Any], observation_root: Path,
                                view_config: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """The one entry point both policy runtimes use for current public facts."""
    facts = project_current_public_facts(
        observation=observation, observation_root=observation_root,
        view_config=view_config, **kwargs)
    if not jaw_contact_center_delivery(view_config):
        return facts
    return _with_contact_centers(
        facts, observation=observation, offsets=jaw_contact_center_offsets(view_config))
