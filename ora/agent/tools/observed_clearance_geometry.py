"""Per-embodiment jaw models of ``observed_clearance_reference_v2`` (integration 0059, M4).

``observed_clearance_runtime.load_jaw_geometry`` reads the ARX X5 descriptor's
``reference_parameters`` (pad contact centre, empty-close encoder reference)
and two-finger encoder positions. The ALOHA and Panda descriptors publish
neither field and their observations carry other public facts, so each
embodiment's jaw is modelled from what it does publish, and nothing is
borrowed from another robot:

``arx_x5_link6``
    v1 unchanged (``load_jaw_geometry``).
``panda_grip_site``
    jaw centre = the grip site (``jaw_center``); each finger opens by its own
    measured ``gripper_finger_qpos`` along its published opening direction,
    with the descriptor's sign: the extractor
    (``libero_pro/robot_asset_geometry.panda_geometry_from_model``) publishes
    finger_1's direction as +joint axis and finger_2's as -joint axis
    (``finger_joint2`` travels 0 to -0.04 m), so the signs are (+1, -1).
``aloha_grip_center_tcp``
    jaw centre = the grip-centre TCP; the observation publishes the measured
    jaw separation ``gripper_opening_m`` (the sum of both prismatic finger
    travels, ``robotwin/gripper_state.JAW_OPENING_SEMANTICS``), and the descriptor
    declares ``opening_mode: symmetric`` (mimic multiplier 1.0,
    ``aloha_gripper_geometry``), so each finger opens by half of it.

On both grip-centre embodiments the jaws count as holding when the last
command was ``close`` and the measured jaw separation
(``localization.measured_jaw_separation_m``) exceeds
``localization.NEAR_CLOSED_M`` -- the bound below which the Harness already
treats closed jaws as empty (``closure_evidence``, the Panda aperture
projection). It is a measurement, not a grasp check: a jaw stopped short by
something other than the named target also reads as holding. Wrist cameras are
those the descriptor attaches to the arm or to its gripper; finger root faces
lie back along the published approach axis (``observed_clearance.root_face``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import observed_clearance as oc
from .jaw_center import grip_centre_offsets
from .localization import NEAR_CLOSED_M
from .observed_clearance_runtime import (
    HOLD_SEPARATION, OPENING_SIGNED, OPENING_SYMMETRIC, JawGeometry, load_jaw_geometry,
)

#: Controlled frame -> how its jaw opens and when it holds.
JAW_MODELS: dict[str, Mapping[str, Any]] = {
    "arx_x5_link6": {"loader": "v1"},
    "panda_grip_site": {"opening": OPENING_SIGNED, "finger_signs": (1.0, -1.0), "hold": HOLD_SEPARATION},
    "aloha_grip_center_tcp": {"opening": OPENING_SYMMETRIC, "finger_signs": (), "hold": HOLD_SEPARATION},
}


def load_jaw_geometry_v2(path: Path, convention: str) -> dict[str, JawGeometry]:
    """Per-arm jaw geometry from a public embodiment descriptor under one controlled-frame convention."""
    from ora.agent.composition.embodiment import load_embodiment_descriptor

    model = JAW_MODELS.get(convention)
    if model is None:
        raise ValueError(f"no observed-clearance jaw model for {convention!r}")
    if model.get("loader") == "v1":
        return load_jaw_geometry(path)
    descriptor = load_embodiment_descriptor(Path(path))
    centres = grip_centre_offsets(descriptor, convention)
    points = {point["id"]: point for point in descriptor["control"]["controlled_points"]}
    cameras = descriptor.get("sensing", {}).get("cameras", [])
    result: dict[str, JawGeometry] = {}
    for gripper in descriptor["geometry"]["grippers"]:
        jaws = gripper.get("contact_geometry") or {}
        point = points[jaws["controlled_point"]]
        manipulator = str(point["manipulator_id"])
        arm = manipulator.removesuffix("_arm")
        fingers = tuple(oc.FingerBounds(minimum=finger["collision_bounds_m"]["minimum"],
                                        maximum=finger["collision_bounds_m"]["maximum"],
                                        opening_direction=finger["opening_direction_local"])
                        for finger in jaws["fingers"])
        signs = tuple(model["finger_signs"])
        if signs and len(signs) != len(fingers):
            raise ValueError(f"{convention} finger signs do not match the descriptor's fingers")
        result[arm] = JawGeometry(
            fingers=fingers,
            reference_finger_qpos=tuple(float(value) for value in jaws["reference_finger_qpos_m"]),
            empty_close_reference=None,
            contact_center_local=tuple(float(value) for value in centres[arm]),
            wrist_cameras=tuple(str(camera["public_name"]) for camera in cameras
                                if camera.get("attached_to") in {manipulator, gripper["end_effector_id"]}),
            finger_opening=str(model["opening"]),
            finger_signs=signs,
            hold_rule=str(model["hold"]),
            near_closed_m=NEAR_CLOSED_M,
            root_face=oc.root_face(point["local_axes"]["approach"]),
        )
    return result


__all__ = ["JAW_MODELS", "load_jaw_geometry_v2"]
