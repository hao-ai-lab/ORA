"""Public RoboTwin gripper command and encoder evidence for the observation."""
from __future__ import annotations

import math

import numpy as np


# Source-pinned aloha-agilex parallel-jaw geometry. The control profile
# already fails closed on ``aloha_config_sha256``/``aloha_urdf_sha256`` drift,
# and in that pinned URDF both finger joints of an arm (``fl_joint7``/
# ``fl_joint8``, ``fr_joint7``/``fr_joint8``) are prismatic rails whose travel
# is metres: each is mounted 0.024493 m off the palm centreline, driven
# outward along its own axis, and its finger mesh reaches back inward by the
# same 0.024494 m. A pad face therefore sits at exactly its own joint travel
# from the centreline, the jaws touch at zero travel, and the measured jaw
# separation is the sum of the two measured travels. Nothing here is a
# calibration constant we invented; it is the pinned kinematic chain read at
# runtime.
JAW_OPENING_SEMANTICS = "jaw_separation_m_sum_of_measured_prismatic_finger_travel"


def _measured_jaw_separation_m(chain, qpos, active_joints):
    """Sum the measured prismatic finger travels into a jaw separation in metres.

    Fail closed on anything that is not a metre-valued prismatic finger rail.
    A wrong number here is worse than no number: the policy compares this
    against object extents, which a normalized 0..1 opening cannot support.
    """
    if not chain:
        raise ValueError("gripper has no finger joints")
    separation = 0.0
    for joint, _multiplier, _offset in chain:
        kind = getattr(joint, "type", None)
        if isinstance(kind, str) and kind != "prismatic":
            raise ValueError("gripper joint is not a prismatic finger rail")
        travel = float(qpos[active_joints.index(joint)])
        if not math.isfinite(travel):
            raise ValueError("nonfinite gripper finger encoder")
        separation += travel
    return separation


def public_gripper_state(robot):
    """Robot-only command and encoder evidence; never infer grasp or retention.

    RoboTwin's get_normal_real_gripper_val reads a drive target, not qpos.
    Read the finger encoders explicitly and invert the primary mimic binding
    before applying the embodiment's closed/open calibration. The normalized
    opening is intentionally not clipped and cannot be compared against an
    object extent, so the measured jaw separation is published beside it in
    metres. No derived grasp or retention claim is published: only measured
    quantities, because a constant flag teaches its reader to ignore it.
    """
    result = {}
    for arm in ("left", "right"):
        value = dict(commanded_opening_normalized=None, drive_target_opening_normalized=None,
            measured_opening_normalized=None, measured_primary_joint_position=None,
            gripper_opening_m=None,
            measurement_source="articulation_qpos_primary_gripper_joint",
            opening_semantics="normalized_closed_0_open_1_not_aperture_m",
            opening_m_semantics=JAW_OPENING_SEMANTICS)
        try:
            command = float(getattr(robot, arm + "_gripper_val"))
            if not math.isfinite(command):
                raise ValueError("nonfinite gripper command")
            value["commanded_opening_normalized"] = command
            entity = getattr(robot, arm + "_entity")
            chain = list(getattr(robot, arm + "_gripper"))
            joint, multiplier, offset = chain[0]
            low, high = map(float, getattr(robot, arm + "_gripper_scale"))
            multiplier, offset = float(multiplier), float(offset)
            if not all(math.isfinite(x) for x in (low, high, multiplier, offset)) or high == low or multiplier == 0:
                raise ValueError("invalid gripper calibration")
            qpos = entity.get_qpos()
            active_joints = list(entity.get_active_joints())
            position = float(qpos[active_joints.index(joint)])
            target = float(np.asarray(joint.get_drive_target()).reshape(-1)[0])
            if not math.isfinite(position) or not math.isfinite(target):
                raise ValueError("nonfinite gripper encoder or drive target")
            value.update(measured_primary_joint_position=position,
                measured_opening_normalized=((position - offset) / multiplier - low) / (high - low),
                drive_target_opening_normalized=((target - offset) / multiplier - low) / (high - low))
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            value["unavailable_reason"] = "gripper_command_encoder_or_calibration_unavailable"
            result[arm] = value
            continue
        try:
            value["gripper_opening_m"] = _measured_jaw_separation_m(
                chain, qpos, active_joints,
            )
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            value["jaw_separation_unavailable_reason"] = (
                "gripper_finger_travel_geometry_unavailable"
            )
        result[arm] = value
    return result
