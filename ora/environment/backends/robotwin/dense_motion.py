"""Action dispatch, the profile-less dense Runtime action and drive-target digests.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
A sparse motion action leaves here for ``_execute_sparse_motion``, whose
dense joint paths run through the motion transaction
(``motion_transaction._execute_motion_transaction_dense_action``).
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ora.contracts.action import (
    MOTION_ACTION_V3_SCHEMA,
    validate_runtime_action,
)

from .common import (
    RECEIPT_SCHEMA,
    RoboTwinLiveError,
    _atomic_json,
    _file_hash,
    _json_hash,
    _jsonable,
    _pose_error,
    _validate_workspace_target,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)


class RoboTwinDenseMotionMixin:
    """Action dispatch, the profile-less dense Runtime action and drive-target digests."""

    def _execute_without_capture(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one validated action and persist its public receipt."""

        import numpy as np

        environment = self._require_environment()
        if self.is_terminal:
            raise RoboTwinLiveError("episode is already terminal")
        action = validate_runtime_action(payload)
        receipt_path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        if receipt_path.exists():
            raise RoboTwinLiveError(f"refusing to overwrite receipt: {receipt_path}")
        started = time.time()
        if action["schema"] == MOTION_ACTION_V3_SCHEMA:
            return self._execute_sparse_motion(
                action=action, receipt_path=receipt_path, started=started,
            )
        if action["action"] in {
            "move_pose", "move_free_space", "servo_delta", "set_gripper",
        }:
            arguments = action["arguments"]
            is_pose_action = action["action"] in {
                "move_pose", "move_free_space", "servo_delta",
            }
            before = {
                arm: {
                    "pose_xyz_xyzw": _wxyz_to_xyzw(environment.get_arm_pose(arm)),
                    "gripper": float(
                        environment.robot.get_left_gripper_val()
                        if arm == "left"
                        else environment.robot.get_right_gripper_val()
                    ),
                }
                for arm in ("left", "right")
            }
            commanded = {
                arm: _xyzw_to_wxyz(
                    before[arm]["pose_xyz_xyzw"][:3],
                    before[arm]["pose_xyz_xyzw"][3:],
                )
                for arm in ("left", "right")
            }
            arm = arguments["arm"]
            grippers = {
                "left": before["left"]["gripper"],
                "right": before["right"]["gripper"],
            }
            if is_pose_action:
                pose_arguments = (
                    arguments
                    if action["action"] == "move_pose"
                    else arguments["target_pose"]
                )
                position = [
                    float(value) for value in pose_arguments["position_m"]
                ]
                _validate_workspace_target(position, self.workspace_bounds_m)
                commanded[arm] = _xyzw_to_wxyz(
                    position, pose_arguments["orientation_xyzw"]
                )
            else:
                position = [float(value) for value in before[arm]["pose_xyz_xyzw"][:3]]
                grippers[arm] = float(arguments["gripper"])
            ee_action = np.asarray(
                commanded["left"]
                + [grippers["left"]]
                + commanded["right"]
                + [grippers["right"]],
                dtype=np.float64,
            )

            planner: dict[str, Any] = {}
            originals = {}
            for planned_arm in ("left", "right"):
                name = f"{planned_arm}_plan_path"
                original = getattr(environment.robot, name)
                originals[name] = original

                def wrapped(*args: Any, _arm: str = planned_arm, _original: Any = original, **kwargs: Any) -> Any:
                    result = _original(*args, **kwargs)
                    status = result.get("status") if isinstance(result, Mapping) else None
                    positions = result.get("position") if isinstance(result, Mapping) else None
                    planner[_arm] = {
                        "status": str(status) if status is not None else "unknown",
                        "path_steps": int(getattr(positions, "shape", [0])[0])
                        if positions is not None
                        else 0,
                    }
                    return result

                setattr(environment.robot, name, wrapped)
            try:
                self.replay_recorder.record(
                    action_sequence=self.action_sequence,
                    action=action,
                    controller="robotwin_ee_action",
                    arrays={"ee_action": ee_action},
                    physics_steps=0,
                    metadata={"replay_route": "take_action(action_type=ee)"},
                )
                if self.operator_execution_video:
                    self._execute_legacy_action_with_operator_capture(
                        environment=environment, ee_action=ee_action, action=action,
                    )
                else:
                    environment.take_action(ee_action, action_type="ee")
            finally:
                for name, original in originals.items():
                    setattr(environment.robot, name, original)
            if bool(getattr(environment, "eval_success", False)):
                self._freeze_official_terminal(environment, action=action)
            self._capture_operator_frame(label=f"action-{self.action_sequence:04d}")
            reached = {
                selected_arm: {
                    "pose_xyz_xyzw": _wxyz_to_xyzw(environment.get_arm_pose(selected_arm)),
                    "gripper": float(
                        environment.robot.get_left_gripper_val()
                        if selected_arm == "left"
                        else environment.robot.get_right_gripper_val()
                    ),
                }
                for selected_arm in ("left", "right")
            }
            command_xyzw = (
                position + [
                    float(value) for value in pose_arguments["orientation_xyzw"]
                ]
                if is_pose_action
                else list(before[arm]["pose_xyz_xyzw"])
            )
            gripper_before = float(before[arm]["gripper"])
            gripper_after = float(reached[arm]["gripper"])
            preserved_pose_error = _pose_error(
                before[arm]["pose_xyz_xyzw"], reached[arm]["pose_xyz_xyzw"]
            )
            preserved_gripper_residual = abs(gripper_before - gripper_after)
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "action_id": action["action_id"],
                "action": action["action"],
                "arguments": action["arguments"],
                "consumed": True,
                "state_changed": True,
                "commanded_pose_xyz_xyzw": command_xyzw,
                "reached_pose_xyz_xyzw": reached[arm]["pose_xyz_xyzw"],
                "reached_gripper": reached[arm]["gripper"],
                "pose_error": _pose_error(command_xyzw, reached[arm]["pose_xyz_xyzw"]),
                "gripper_evidence": {
                    "measured_before": gripper_before,
                    "measured_after": gripper_after,
                    "closure_delta": gripper_before - gripper_after,
                    "closure_residual": (
                        abs(float(arguments["gripper"]) - gripper_after)
                        if action["action"] == "set_gripper" else None
                    ),
                    "grasp_success": None,
                },
                "preserved_dimension": (
                    {
                        "field": "gripper", "before": gripper_before,
                        "after": gripper_after,
                        "residual": round(preserved_gripper_residual, 6),
                        "verified": preserved_gripper_residual <= 1e-3,
                    }
                    if is_pose_action
                    else {
                        "field": "eef_pose",
                        "translation_residual_m": preserved_pose_error["translation_m"],
                        "orientation_residual_rad": preserved_pose_error["orientation_rad"],
                        "verified": preserved_pose_error["translation_m"] <= 0.005,
                    }
                ),
                "planner": planner,
                "before": before,
                "after": reached,
                "semantic_action_count": int(environment.take_action_cnt),
                "started_unix_s": started,
                "finished_unix_s": time.time(),
            }
        else:  # pragma: no cover - protected by validate_runtime_action
            raise RoboTwinLiveError("unsupported Runtime action")
        _atomic_json(receipt_path, receipt)
        record = {
            "path": receipt_path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(receipt_path),
            "receipt": receipt,
        }
        self.last_receipt = record
        self.replay_recorder.update(
            self.action_sequence, execution_status="executed",
        )
        self.action_sequence += 1
        return record

    @staticmethod
    def _drive_target_digest(environment: Any, arm: str) -> str | None:
        joints = getattr(environment.robot, f"{arm}_arm_joints", None)
        if not isinstance(joints, (list, tuple)):
            return None
        values: list[Any] = []
        for joint in joints:
            getter = getattr(joint, "get_drive_target", None)
            if not callable(getter):
                return None
            values.append(_jsonable(getter()))
        return _json_hash(values)

    @staticmethod
    def _gripper_drive_target_digest(environment: Any, arm: str) -> str | None:
        joints = getattr(environment.robot, f"{arm}_gripper", None)
        if not isinstance(joints, (list, tuple)):
            return None
        values: list[Any] = []
        for record in joints:
            joint = record[0] if isinstance(record, (list, tuple)) and record else None
            getter = getattr(joint, "get_drive_target", None)
            if not callable(getter):
                return None
            values.append(_jsonable(getter()))
        return _json_hash(values)
