"""Public RGB-D, calibration and proprioception observations.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

from .common import (
    OBSERVATION_SCHEMA,
    RoboTwinLiveError,
    _atomic_json,
    _file_hash,
    _jsonable,
    _wxyz_to_xyzw,
)


class RoboTwinObservationMixin:
    """Public RGB-D, calibration and proprioception observations."""

    def _require_environment(self) -> Any:
        if self.environment is None:
            raise RoboTwinLiveError("RoboTwin episode is not started")
        return self.environment

    def _read_observation_data(self, environment: Any) -> dict[str, Any]:
        """Read RGB-D and proprioception without advancing physics."""

        raw = dict(environment.get_obs())
        cameras = raw.get("observation")
        if not isinstance(cameras, Mapping) or not cameras:
            raise RoboTwinLiveError("RoboTwin observation contains no cameras")
        cameras = dict(cameras)
        raw["observation"] = cameras
        return copy.deepcopy(raw)

    def _public_proprioception(
        self, environment: Any, raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Assemble the policy-visible proprioception for one observation.

        Command and measurement are independent evidence and neither replaces
        the other: ``gripper_command_target`` is the last accepted directive,
        ``gripper_opening_m`` is the measured jaw separation in metres. A
        policy that can only see its own last command cannot tell an accepted
        close from jaws stopped early by an object, and a normalized 0..1
        opening cannot be compared against an object extent. The measurement
        is therefore published unconditionally.
        """

        endpose = raw.get("endpose")
        joints = raw.get("joint_action")
        if not isinstance(endpose, Mapping) or not isinstance(joints, Mapping):
            raise RoboTwinLiveError("RoboTwin observation lacks proprioception")
        if self.action_codec == "motion_action_v3":
            proprioception = {
                "end_effectors": {
                    arm: {
                        "pose_xyz_xyzw": _wxyz_to_xyzw(
                            (
                                self.terminal_public_poses[arm]
                                if self.terminal_public_poses is not None
                                else self._public_pose_wxyz(environment, arm)
                            ) if self._uses_grip_center_profile()
                            else endpose[f"{arm}_endpose"]
                        ),
                        "controlled_point_semantics": (
                            "parallel_jaw_grip_center_tcp"
                            if self._uses_grip_center_profile()
                            else "RoboTwin_planner_pose"
                        ),
                        "gripper_command_target": (
                            self.terminal_gripper_targets[arm]
                            if self.terminal_gripper_targets is not None
                            else self.gripper_command_targets[arm]
                        ),
                    }
                    for arm in ("left", "right")
                },
                "joints": {
                    "left_arm": _jsonable(joints.get("left_arm", [])),
                    "right_arm": _jsonable(joints.get("right_arm", [])),
                },
            }
        else:
            proprioception = {
                "end_effectors": {
                    "left": {
                        "pose_xyz_xyzw": _wxyz_to_xyzw(endpose["left_endpose"]),
                        "gripper": float(endpose["left_gripper"]),
                    },
                    "right": {
                        "pose_xyz_xyzw": _wxyz_to_xyzw(endpose["right_endpose"]),
                        "gripper": float(endpose["right_gripper"]),
                    },
                },
                "joints": {
                    "left_arm": _jsonable(joints.get("left_arm", [])),
                    "left_gripper": float(joints.get("left_gripper", endpose["left_gripper"])),
                    "right_arm": _jsonable(joints.get("right_arm", [])),
                    "right_gripper": float(joints.get("right_gripper", endpose["right_gripper"])),
                },
            }
        from .gripper_state import public_gripper_state

        grippers = raw.get("public_gripper_state")
        if not isinstance(grippers, Mapping):
            grippers = public_gripper_state(getattr(environment, "robot", None))
        for arm, state in grippers.items():
            record = proprioception["end_effectors"].get(arm)
            if not isinstance(record, dict):
                continue
            record["gripper_opening_m"] = state.get("gripper_opening_m")
            if self.public_robot_geometry is not None:
                record["gripper_state"] = copy.deepcopy(state)
        proprioception["gripper_evidence_semantics"] = (
            "command_target_and_measured_jaw_separation_m"
        )
        return proprioception

    def _verify_loaded_robot_geometry(self, environment: Any) -> None:
        self.robot_geometry_verification = None
        if self.public_robot_geometry is None:
            return
        from ora.contracts.robot_geometry import verify_loaded_robot_geometry
        from .robot_asset_geometry import collect_aloha_grippers

        profile = self.control_profile
        if not isinstance(profile, Mapping):
            raise RoboTwinLiveError("public robot geometry requires a bound control profile")
        try:
            raw_profile = profile.get("profile", profile)
            evidence = collect_aloha_grippers(
                self._robot_geometry_urdf,
                dict(raw_profile), loaded_robots={
                    arm: getattr(environment.robot, f"{arm}_entity") for arm in ("left", "right")})
            observed = {arm + "_gripper": evidence["geometry"][arm] for arm in ("left", "right")}
            result = verify_loaded_robot_geometry(self.public_robot_geometry, observed)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RoboTwinLiveError("loaded ALOHA geometry differs from the public descriptor") from exc
        self.robot_geometry_verification = {**result, "round_index": self.round_index}
        _atomic_json(self.output_root / f"robot-geometry/round-{self.round_index}.json", {
            **self.robot_geometry_verification,
            "expected_binding": dict(self.public_robot_geometry), "observed_geometry": observed,
            "robot_asset_evidence": evidence,
        })

    def observe(self, policy_observation_root: Path) -> tuple[dict[str, Any], list[Path]]:
        """Project a fresh immutable, privilege-clean observation."""

        import numpy as np
        from PIL import Image

        environment = self._require_environment()
        observation_id = (
            f"{self.task}:episode-{self.episode:03d}:observation-{self.observation_sequence:04d}"
        )
        target = policy_observation_root.resolve() / observation_id
        if target.exists():
            raise RoboTwinLiveError(f"refusing to overwrite observation: {target}")
        target.mkdir(parents=True)
        raw = (
            copy.deepcopy(self.terminal_observation)
            if self.terminal_observation is not None
            else self._read_observation_data(environment)
        )
        cameras = raw["observation"]
        camera_records: dict[str, Any] = {}
        image_paths: list[Path] = []
        for camera_name in sorted(cameras):
            camera = cameras[camera_name]
            if not isinstance(camera, Mapping) or "rgb" not in camera or "depth" not in camera:
                raise RoboTwinLiveError(f"camera lacks paired RGB-D: {camera_name}")
            rgb = np.asarray(camera["rgb"])
            depth = np.asarray(camera["depth"], dtype=np.float32)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
                raise RoboTwinLiveError(f"camera RGB-D shape mismatch: {camera_name}")
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            camera_root = target / camera_name
            camera_root.mkdir()
            rgb_png = camera_root / "rgb.png"
            depth_npy = camera_root / "depth-m.npy"
            calibration = camera_root / "calibration.json"
            Image.fromarray(rgb).save(rgb_png)
            np.save(depth_npy, (depth / 1000.0).astype(np.float32), allow_pickle=False)
            calibration_payload = {
                "intrinsic_cv": _jsonable(camera["intrinsic_cv"]),
                "extrinsic_cv": _jsonable(camera["extrinsic_cv"]),
                "cam2world_gl": _jsonable(camera["cam2world_gl"]),
                "depth_unit": "metres",
                "depth_array_order": "row_major_height_width",
                "rgb_array_order": "row_major_height_width_rgb",
                "policy_quaternion_order": "xyzw",
            }
            _atomic_json(calibration, calibration_payload)
            files = {}
            role_paths = [
                ("rgb_lossless_png", rgb_png),
                ("depth_numpy_m_f32", depth_npy),
                ("calibration", calibration),
            ]
            for role, path in role_paths:
                files[role] = {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": _file_hash(path),
                    "bytes": path.stat().st_size,
                }
            camera_records[camera_name] = {
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "mount": (
                    "left_wrist" if camera_name == "left_camera"
                    else "right_wrist" if camera_name == "right_camera"
                    else "fixed_world"
                ),
                "calibration_dynamic": camera_name in {"left_camera", "right_camera"},
                "modalities": ["rgb", "depth", "calibration"],
                "files": files,
            }
            image_paths.append(rgb_png)

        joints = raw.get("joint_action")
        if not isinstance(joints, Mapping):
            raise RoboTwinLiveError("RoboTwin observation lacks proprioception")
        proprioception = self._public_proprioception(environment, raw)
        _atomic_json(target / "proprioception.json", proprioception)
        previous_receipt = self.last_receipt
        payload = {
            "schema": OBSERVATION_SCHEMA,
            "observation_id": observation_id,
            "task": self.task,
            "episode": self.episode,
            "sequence": self.observation_sequence,
            "encoding": "metric_rgbd_v1",
            "main_camera": "head_camera",
            "captured_unix_s": (
                self.official_terminal["captured_unix_s"]
                if self.official_terminal is not None else time.time()
            ),
            "cameras": camera_records,
            "proprioception": proprioception,
            **({"robot_geometry_verification": self.robot_geometry_verification}
               if self.robot_geometry_verification is not None else {}),
            "previous_execution_receipt": (
                None
                if previous_receipt is None
                else {
                    "path": previous_receipt["path"],
                    "sha256": previous_receipt["sha256"],
                    "receipt": previous_receipt["receipt"],
                }
            ),
            "forbidden_modalities_absent": [
                "actor_state",
                "object_gt_pose",
                "segmentation",
                "contact",
                "success_predicate",
                "expert_program",
                "task_source",
            ],
        }
        manifest_path = target / "observation.json"
        _atomic_json(manifest_path, payload)
        for path in sorted(candidate for candidate in target.rglob("*") if candidate.is_file()):
            path.chmod(0o444)
        self.observation_sequence += 1
        return payload, image_paths
