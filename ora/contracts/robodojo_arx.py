"""Source-bound ARX X5 facts shared by policy composition and controller code.

Robot facts: RoboDojo Assets a14409d7fae673c00499e01fd88b4457df6351b1,
Assets/Robots/x5/robot_config.yml (SHA-256
9de4f63b84c0f730623c09af7a2d530d84973981397280b6e4b626b50ba6f1c9).
The dual fixed-base composition is env_cfg/robot/dual_x5.yml in RoboDojo
ee67a1468510da7624a089164402359f2afc72c8. Geometry here never implies
collision-free motion, contact, retention, or a known physical jaw aperture.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


ARX_X5_CONTRACT = {
    "schema": "ora.robodojo-arx-controller.v1",
    "embodiment_id": "arx_x5", "robot_models": ["ARX_X5", "ARX_X5"],
    "base_mobility": "fixed", "arms": ["left", "right"], "joint_dof_per_arm": 6,
    "controlled_link": "link6", "position_frame": "episode_local_world",
    "position_unit": "metre", "policy_quaternion_order": "xyzw", "native_quaternion_order": "wxyz",
    "gripper_command": {"range": [0.0, 1.0], "closed": 0.0, "open": 1.0,
                        "primary_joint_range": [-0.01, 0.044], "mimic_multiplier": 1.0, "mimic_offset": 0.0,
                        "physical_jaw_aperture_calibrated": False},
    # Pinned SDK env/robot_manager/control_manager.py::MetaControl.get_action.
    # The bridge retains this rule for explicit gripper commands; it does not
    # limit the directive relative to the previous *command*.
    "gripper_drive_limit": {
        "kind": "measured_primary_relative",
        "maximum_normalized_offset": 0.2,
        "reference": "current_primary_finger_encoder",
        "coupling": "both_drive_targets_from_limited_primary_target",
        "applies_to": "explicit_gripper_commands",
        "pose_hold": "preserve_last_applied_drive_target",
    },
    "controller": "CUROBO_IK_JOINT_INTERPOLATION", "action_codec": "runtime_action_v3",
    "action_abstraction": "cartesian_target", "internal_controller_policy": "tool_surface_owned",
    "physics_dt_s": 0.004, "control_hz": 25, "physics_steps_per_native_action": 10,
    "native_actions_per_command": 20, "official_native_action_limit_owner": "environment_task",
    "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.10, "gripper_tolerance_normalized": 0.03,
    "pose_action_holds": "gripper_directive", "gripper_action_holds": "eef_pose_target",
    "other_arm_holds": "previous_joint_target", "failed_ik_executes_gripper": False,
    "workspace_bounds_owner": "runtime", "collision_model_claim": "bounded_motion_execution_evidence",
    "sensor_profile": "rgbd_metric_v1", "public_cameras": ["head", "left_wrist", "right_wrist"],
    "sensor_modalities": ["rgb", "metric_depth"], "calibration_owner": "per_observation",
    "state_fields": ["measured_eef_pose_xyz_xyzw", "measured_gripper", "gripper_command"],
}


def controller_binding() -> dict[str, Any]:
    value = json.loads(json.dumps(ARX_X5_CONTRACT))
    value["contract_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def public_feedback_contract() -> dict[str, Any]:
    """Snapshot the rule and the bounds the executor actually consumes."""
    return {
        "schema": "ora.controller-feedback.v1",
        "position_tolerance_m": ARX_X5_CONTRACT["position_tolerance_m"],
        "orientation_tolerance_rad": ARX_X5_CONTRACT["orientation_tolerance_rad"],
        "gripper_tolerance_normalized": ARX_X5_CONTRACT["gripper_tolerance_normalized"],
        "arrival_comparison": "<=",
        "native_actions_per_command": ARX_X5_CONTRACT["native_actions_per_command"],
        "gripper_drive_limit": dict(ARX_X5_CONTRACT["gripper_drive_limit"]),
    }
