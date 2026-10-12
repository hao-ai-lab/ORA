"""Dojo public controller feedback, shared by observations and receipts."""
from __future__ import annotations
from typing import Any
from ora.contracts.acceptance import gripper_acceptance_check, motion_acceptance_check
from ora.contracts.gripper_feedback import gripper_feedback
from ora.environment.controllers.robodojo_arx import public_feedback_contract
from . import sparse_motion
from .common import (
    finger_lateral_offset_m, two_finger_opening, GRIPPER_TOLERANCE, POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD,
)


def gripper_state(session: Any, arm: str, fingers: Any, applied: float) -> dict[str, Any]:
    """The one publisher of measured/commanded ARX gripper semantics.

    ``gripper`` is the two-finger mean on the command scale (E11G12-001), the
    value every consumer -- arrival and stall, ``jaw_hold``, the observation --
    reads; the typed feedback adds the finger pair's signed lateral offset.
    """
    measured = two_finger_opening(session.robots[arm], fingers)
    requested = session.gripper_directives.get(arm)
    effective = (None if requested is None else
                 max(requested, sparse_motion.gripper_closed_floor(session, arm)))
    feedback = gripper_feedback(
        requested_opening_normalized=requested,
        drive_target_opening_normalized=applied,
        measured_opening_normalized=measured,
        effective_target_opening_normalized=effective,
        tolerance_normalized=GRIPPER_TOLERANCE,
        drive_target_source="resolved_sdk_joint_position_target",
        measurement_source="two_finger_encoder_mean_v1",
        pose_hold_semantics="preserve_last_applied_drive_target",
        finger_lateral_offset_m=finger_lateral_offset_m(fingers),
    )
    return {
        "gripper": measured, "gripper_finger_qpos": fingers.tolist(),
        # Retain the released numeric read-back for raw evidence readers.
        # The typed feedback names it as an applied target, not intent.
        "gripper_command": applied,
        "gripper_command_target": feedback["requested_command"],
        "gripper_feedback": feedback,
    }


def add_receipt_feedback(receipt: dict[str, Any]) -> None:
    receipt["controller_feedback"] = public_feedback_contract()
    # Store the producer's actual comparison before any summary/projection.
    # A later reader must not backfill today's bounds into an old receipt.
    effects = receipt.get("arms", {})
    pose_errors = {arm: effect["pose_error"] for arm, effect in effects.items()
                   if effect.get("targeted") and effect.get("pose_error") is not None}
    if receipt.get("schema") == 'ora.robodojo-motion-receipt.v1' and receipt.get("action") == "move_pose":
        pose_errors = {arm: {"position_m": receipt["position_error_m"],
                             "orientation_rad": receipt["orientation_error_rad"]}
                       for arm in receipt["target_arms"]}
    checks = []
    if pose_errors:
        checks.append(motion_acceptance_check(stage="final_endpoint", errors=pose_errors,
            position_tolerance_m=POSITION_TOLERANCE_M, orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD))
    if receipt.get("action") == "set_gripper":
        arms = receipt.get("target_arms", [receipt.get("arm")])
        errors = {arm: receipt["after"][arm]["gripper_feedback"]["arrival_check"]["residual_normalized"]
                  for arm in arms if arm in receipt["after"]}
        if errors:
            checks.append(gripper_acceptance_check(errors=errors, tolerance_normalized=GRIPPER_TOLERANCE))
    receipt["acceptance_checks"] = checks
