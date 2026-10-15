"""Receipts and test-local capability sets for the 0059 delivery lessons.

The three delivery capabilities activate only on the next surface generation
(integration 0058 unit A0). Until a manifest declares them, tests add them
locally: ``with_capabilities`` for the runtime projections, ``composed_with``
when a composition binder must run. Neither edits a pinned asset.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Iterable

from ora.agent.composition import tool_surface as tool_surface_module
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
from ora.contracts.acceptance import gripper_acceptance_check, motion_acceptance_check
from ora.contracts.gripper_feedback import gripper_feedback
from ora.contracts.robodojo_arx import public_feedback_contract

SUMMARY = "last_action_summary_v1"
DEGREES = "policy_angles_degrees_v1"
DEDUP = "within_payload_deduplication_v1"
NEW_CAPABILITIES = (SUMMARY, DEGREES, DEDUP)

#: The surfaces the direct_v3 defaults resolved to (inventory section 1); frozen
#: since direct_v4 (integration 0059 unit A0), whose v2 surfaces declare the
#: delivery capabilities. These keep the legacy bytes. Only the RoboTwin one is
#: still registered (the runtime_v1 prepared-grasp qualification binds it); the
#: LIBERO-Pro and RoboDojo ones are not in this tree.
DEFAULT_PROFILES = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v1",
}


def with_capabilities(surface: Any, *extra: str) -> Any:
    """A resolved surface that also declares ``extra``, for runtime projection only."""
    return dataclasses.replace(surface, capabilities=(*surface.capabilities, *extra))


def composed_with(monkeypatch: Any, profile_id: str, *extra: str) -> Any:
    """Compose ``profile_id`` as if its manifest also declared ``extra``."""
    original = tool_surface_module._load_manifest

    def load(requested: str) -> Any:
        path, root, manifest, revision = original(requested)
        if requested == profile_id:
            manifest = copy.deepcopy(manifest)
            manifest["capabilities"] = [*manifest["capabilities"], *extra]
        return path, root, manifest, revision

    monkeypatch.setattr(tool_surface_module, "_load_manifest", load)
    return resolve_tool_surface(profile_id)


def project(record: dict[str, Any], capabilities: Iterable[str],
            *, public_name: str | None = None) -> dict[str, Any]:
    """Project one receipt through the registry the Native session uses."""
    projector = resolve_receipt_effect_projector(record["schema"])
    assert projector is not None
    declared = frozenset(capabilities)
    return projector(record, ReceiptProjectionContext(
        last_action_id=record.get("action_id"),
        last_action_public_name=public_name,
        surface_supports=declared.__contains__,
    ))


def dojo_gripper_feedback(measured: float = 0.99, requested: float = 1.0) -> dict[str, Any]:
    return gripper_feedback(
        requested_opening_normalized=requested, drive_target_opening_normalized=requested,
        measured_opening_normalized=measured, effective_target_opening_normalized=requested,
        tolerance_normalized=0.03, drive_target_source="resolved_sdk_joint_position_target",
        measurement_source="two_finger_encoder_mean_v1",
        pose_hold_semantics="preserve_last_applied_drive_target", finger_lateral_offset_m=-1.5e-07,
    )


POSE = [0.3, -0.35, 0.92, 0.0, 0.0, 0.7071, 0.7071]


def robodojo_receipt(*, action: str = "move_pose", arm: str = "left", status: str = "reached",
                     stage: str | None = None, code: str | None = None,
                     pose_error: tuple[float, float] | None = (0.0087, 0.0282),
                     native_actions: int = 9, feedback: dict[str, Any] | None = None,
                     gripper_check: float | None = None) -> dict[str, Any]:
    """One RoboDojo sparse receipt, built from the contract owners' own builders."""
    other = "right" if arm == "left" else "left"
    feedback = feedback or {side: dojo_gripper_feedback() for side in ("left", "right")}
    segment = {
        "segment": "move", "ik_status": "Success", "native_actions": native_actions,
        "joint_delta_rad": 1.7734, "joint_target": [0.0, 1.7734, 1.3675, -1.1648, 0.0, -1.3158],
        "orientation_error_rad": 0.0282, "position_error_m": 0.0087,
        "ik_selection": {"canonical_interval_rad": [-3.141592653589793, 3.141592653589793],
                         "joint_jump_limit_rad": 3.141592653589793, "limiting_joint": "joint2"},
    }
    arms = {
        arm: {"targeted": True, "interaction": "free_space", "gripper_error": None,
              "pose_error": (None if pose_error is None else
                             {"translation_m": pose_error[0], "orientation_rad": pose_error[1]}),
              "plan_validation": None, "plan_ref": None, "inactive_arm_drift": None,
              "segments": [segment] if action != "set_gripper" else []},
        other: {"targeted": False, "interaction": None, "gripper_error": None, "pose_error": None,
                "plan_validation": None, "plan_ref": None, "segments": [],
                "inactive_arm_drift": {"translation_m": 0.0, "orientation_rad": 2.98e-08}},
    }
    checks = []
    if pose_error is not None:
        checks.append(motion_acceptance_check(
            stage="final_endpoint", errors={arm: {"translation_m": pose_error[0],
                                                  "orientation_rad": pose_error[1]}},
            position_tolerance_m=0.01, orientation_tolerance_rad=0.1))
    if gripper_check is not None:
        checks.append(gripper_acceptance_check(errors={arm: gripper_check}, tolerance_normalized=0.03))
    return {
        "schema": "ora.robodojo-sparse-motion-receipt.v1",
        "action_id": "action-0007", "action": action, "execution_status": status,
        "executor": ["robodojo_sdk_ik_joint_interpolation"], "motion_strategy": "global",
        "failure_stage": stage, "failure_code": code, "terminal_after_action": False,
        "target_arms": [arm],
        "synchronization": {"executed": native_actions > 0, "mode": "single_arm", "requested": False},
        "arms": arms, "acceptance_checks": checks,
        "controller_feedback": public_feedback_contract(),
        "before": {side: {"pose_xyz_xyzw": list(POSE), "gripper_feedback": feedback[side]}
                   for side in ("left", "right")},
        "after": {side: {"pose_xyz_xyzw": list(POSE), "gripper_feedback": feedback[side]}
                  for side in ("left", "right")},
        "gripper_targets_effective": {}, "native_actions": native_actions,
        "native_action_budget": 20,
    }


def robotwin_receipt(*, arm: str = "left", status: str = "plan_rejected",
                     stage: str | None = "plan_validation", code: str | None = "inter_arm_collision",
                     residual: tuple[float, float] = (0.104, 2.0944), executed: bool = False,
                     pose_error: tuple[float, float] | None = None) -> dict[str, Any]:
    """One RoboTwin v7 receipt shaped like the B2 default produces."""
    other = "right" if arm == "left" else "left"
    planner = {"called": True, "status": "Success", "path_steps": 120,
               "topk_selection": {"candidate_outcomes": [{"ik_distance_rad": 0.37}]}}
    return {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action": {"action": "move_pose"}, "action_id": "action-0011",
        "execution_status": status, "target_arms": [arm],
        "synchronization": {"executed": executed, "requested": False},
        "arms": {
            arm: {"targeted": True, "planner": planner,
                  "pose_error": (None if pose_error is None else
                                 {"translation_m": pose_error[0], "orientation_rad": pose_error[1]})},
            other: {"targeted": False, "planner": {"called": False, "status": "not_called",
                                                   "path_steps": 0},
                    "inactive_arm_drift": {"translation_m": 0.0, "orientation_rad": 0.004}},
        },
        "controller_tracking": {"completed": executed, "arms": {
            arm: {"targeted": True, "maximum_joint_tracking_error_rad": 0.0059}}},
        "acceptance_checks": [motion_acceptance_check(
            stage="final_endpoint", errors={arm: {"translation_m": residual[0],
                                                  "orientation_rad": residual[1]}},
            position_tolerance_m=0.005, orientation_tolerance_rad=0.05)],
        "motion_strategy": "global", "failure_stage": stage, "failure_code": code,
        "terminal_after_action": False,
    }


def libero_receipt(*, status: str = "reached", translation: float = 0.0033,
                   orientation: float = 0.0064) -> dict[str, Any]:
    """One LIBERO sparse receipt with the endpoint contract's free-text summary."""
    summary = (f"Measured pose meets the endpoint bound; current residual "
               f"{translation * 1000:.1f} mm / {orientation:.4f} rad against 5.0 mm / 0.0500 rad.")
    start, end = [0.1, 0.0, 0.9, 0, 1, 0, 0], [0.1, 0.03, 0.86, 0, 1, 0, 0]
    return {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action_id": "action-0003", "action": "move_pose", "execution_status": status,
        "executor": ["libero_osc_local_pose"], "motion_strategy": "local",
        "failure_stage": None, "failure_code": None, "terminal_after_action": False,
        "target_arms": ["right"], "synchronization": {"executed": True, "mode": "single_arm"},
        "arms": {"right": {
            "targeted": True, "pose_error": {"translation_m": translation, "orientation_rad": orientation},
            "endpoint_contract": {"summary": summary, "within_tolerance": True,
                                  "judged_residual": {"position_m": translation,
                                                      "orientation_rad": orientation},
                                  "tolerance": {"position_m": 0.005, "orientation_rad": 0.05}},
        }},
        "acceptance_checks": [motion_acceptance_check(
            stage="final_endpoint", errors={"right": {"translation_m": translation,
                                                      "orientation_rad": orientation}},
            position_tolerance_m=0.005, orientation_tolerance_rad=0.05, comparison="<")],
        "segment": {"before_pose_xyz_xyzw": start, "reached_pose_xyz_xyzw": end,
                    "feedback": {"contact_estimated": False}},
    }


def skill_receipt() -> dict[str, Any]:
    """One closed grasp skill that stopped at its pregrasp stage."""
    return {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": "action-0021", "skill": "grasp_execute", "status": "aborted",
        "candidate_id": "grasp-ce4e0b8a", "stop_reason": "pregrasp_not_reached",
        "resumed_from": 0, "terminal_after_action": False,
        "committed_sequence": ["open", "pregrasp", "approach", "close", "lift", "verify"],
        "stage_receipts": [
            {"stage": "open", "status": "executed", "reason": "executed"},
            {"stage": "pregrasp", "status": "not_reached", "reason": "IK_FAIL", "measurements": [
                {"quantity": "endpoint_residual", "value": {
                    "errors_by_arm": {"left": {"orientation_rad": 2.616047, "translation_m": 0.18848}},
                    "reason": "IK_FAIL"}}]},
        ],
    }


def libero_hybrid_receipt() -> dict[str, Any]:
    """One LIBERO hybrid gripper receipt: single arm, no target-arm list."""
    return {
        "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action_id": "action-0004", "action": "set_gripper", "motion_status": "reached",
        "terminal_after_action": False, "feedback": {"contact_estimated": False},
        "acceptance_checks": [gripper_acceptance_check(errors={"right": 0.03}, tolerance_normalized=0.05)],
        "before_pose_xyz_xyzw": [0.1, 0.0, 0.9, 0, 1, 0, 0],
        "reached_pose_xyz_xyzw": [0.1, 0.0002, 0.9, 0, 1, 0, 0],
    }
