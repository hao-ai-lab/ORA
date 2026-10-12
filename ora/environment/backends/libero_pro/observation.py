"""Public RGB-D, calibration, proprioception and worker state read-back.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from ora.environment.backends.libero_pro.motion_settle import (
    public_gripper_command,
)

from .common import (
    OBSERVATION_SCHEMA,
    LiberoProError,
    _atomic_json,
    _file_hash,
    _metric_depth,
    _normalized_gripper,
)


class LiberoObservationMixin:
    """Public RGB-D, calibration, proprioception and worker state read-back."""

    @staticmethod
    def _strip(value: Any) -> Any:
        if value is None:
            return None
        return value[0]

    def _raw_observation(self) -> dict[str, Any]:
        environment = self._require_environment()
        raw = getattr(environment, "current_raw_obs", None)
        if raw is None:
            raise LiberoProError("LIBERO environment has no current raw observation")
        if isinstance(raw, Mapping):
            selected = {key: self._strip(value) for key, value in raw.items()}
        else:
            try:
                selected = raw[0]
            except (IndexError, KeyError, TypeError) as exc:
                raise LiberoProError(
                    "LIBERO raw observation batch is not indexable"
                ) from exc
        if not isinstance(selected, Mapping):
            raise LiberoProError("LIBERO raw observation is not an object")
        return dict(selected)

    def _camera_meta(self, source_name: str, height: int, width: int) -> dict[str, Any]:
        environment = self._require_environment()
        get_camera_meta = getattr(environment, "get_camera_meta", None)
        if callable(get_camera_meta):
            meta = get_camera_meta(
                camera_name=source_name, height=height, width=width,
            )
        else:
            vector_environment = getattr(environment, "env", None)
            get_env_attr = getattr(vector_environment, "get_env_attr", None)
            if not callable(get_env_attr):
                raise LiberoProError(
                    f"camera calibration transport is unavailable: {source_name}"
                )
            values = get_env_attr("harness_camera_meta")
            meta = (
                values[0].get(source_name)
                if isinstance(values, list)
                and len(values) == 1
                and isinstance(values[0], Mapping)
                else None
            )
        if not isinstance(meta, Mapping):
            raise LiberoProError(f"camera calibration is unavailable: {source_name}")
        if meta.get("height") not in (None, height) or meta.get("width") not in (
            None, width,
        ):
            raise LiberoProError(f"camera calibration size drift: {source_name}")
        return dict(meta)

    def _grip_site_pose(self) -> tuple[Any, Any]:
        import numpy as np

        environment = self._require_environment()
        direct = getattr(environment, "get_grip_site_pose", None)
        if callable(direct):
            value = direct()
        else:
            vector_environment = getattr(environment, "env", None)
            get_env_attr = getattr(vector_environment, "get_env_attr", None)
            if not callable(get_env_attr):
                raise LiberoProError("grip-site pose transport is unavailable")
            values = get_env_attr("harness_grip_site_pose")
            value = values[0] if isinstance(values, list) and len(values) == 1 else None
        if not isinstance(value, Mapping) or value.get("frame") != "robosuite_grip_site":
            raise LiberoProError("controller-owned grip-site pose is unavailable")
        position = np.asarray(value.get("position_m"), dtype=np.float64)
        orientation = np.asarray(value.get("orientation_xyzw"), dtype=np.float64)
        if (
            position.shape != (3,) or orientation.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(orientation).all()
            or abs(float(np.linalg.norm(orientation)) - 1.0) > 1e-3
        ):
            raise LiberoProError("controller-owned grip-site pose is malformed")
        return position, orientation / np.linalg.norm(orientation)

    def public_gripper_command_target(self) -> str:
        """The last accepted jaw target, or ``unknown`` before there is one."""

        return getattr(self, "_gripper_command_target", "unknown")

    def _gripper_hold_directive(self) -> float:
        """Read the stateful Panda gripper target without rewriting it."""

        environment = self._require_environment()
        direct = getattr(environment, "get_gripper_hold_directive", None)
        if callable(direct):
            directive = float(direct())
            if math.isfinite(directive) and -1.0 <= directive <= 1.0:
                return directive
            raise LiberoProError("Panda gripper directive is invalid")
        vector_environment = getattr(environment, "env", None)
        get_env_attr = getattr(vector_environment, "get_env_attr", None)
        if not callable(get_env_attr):
            raise LiberoProError("gripper hold-directive transport is unavailable")
        observed = get_env_attr("harness_gripper_hold_directive")
        if not isinstance(observed, list) or len(observed) != 1:
            raise LiberoProError("gripper hold directive read-back is unavailable")
        directive = float(observed[0])
        if not math.isfinite(directive) or not -1.0 <= directive <= 1.0:
            raise LiberoProError("Panda gripper directive is invalid")
        return directive

    def _worker_attr(self, name: str) -> Any:
        environment = self._require_environment()
        vector_environment = getattr(environment, "env", None)
        getter = getattr(vector_environment, "get_env_attr", None)
        if not callable(getter):
            raise LiberoProError(f"worker attribute transport unavailable: {name}")
        values = getter(name)
        if not isinstance(values, list) or len(values) != 1:
            raise LiberoProError(f"worker attribute read-back unavailable: {name}")
        return values[0]

    def _motion_state(self) -> dict[str, Any]:
        state = self._worker_attr("harness_motion_state")
        if not isinstance(state, Mapping):
            raise LiberoProError("motion state is unavailable")
        if len(state.get("joint_positions", ())) != 7:
            raise LiberoProError("motion state has invalid Panda joints")
        return dict(state)

    def _verify_loaded_robot_geometry(self) -> None:
        self.robot_geometry_verification = None
        if self.public_robot_geometry is None:
            return
        from ora.contracts.robot_geometry import verify_loaded_robot_geometry

        try:
            observed = self._worker_attr("harness_robot_geometry")
            result = verify_loaded_robot_geometry(
                self.public_robot_geometry, observed)
        except (ValueError, TypeError) as exc:
            raise LiberoProError("loaded Panda geometry differs from the public descriptor") from exc
        self.robot_geometry_verification = {**result, "round_index": self.round_index}
        _atomic_json(self.output_root / f"robot-geometry/round-{self.round_index}.json", {
            **self.robot_geometry_verification,
            "expected_binding": dict(self.public_robot_geometry), "observed_geometry": observed,
        })

    def observe(self, policy_observation_root: Path) -> tuple[dict[str, Any], list[Path]]:
        import numpy as np
        from PIL import Image

        raw = self._raw_observation()
        self.last_raw_observation = raw
        observation_id = (
            f"{self.suite}:task-{self.task_id:02d}:seed-{self.seed}:"
            f"observation-{self.observation_sequence:04d}"
        )
        target = policy_observation_root.resolve() / observation_id
        if target.exists():
            raise LiberoProError(f"refusing to overwrite observation: {target}")
        target.mkdir(parents=True)
        camera_specs = {
            "agentview": ("agentview", "agentview_image", "agentview_depth"),
            "wrist": (
                "robot0_eye_in_hand", "robot0_eye_in_hand_image",
                "robot0_eye_in_hand_depth",
            ),
        }
        cameras: dict[str, Any] = {}
        image_paths: list[Path] = []
        for public_name, (source_name, image_key, depth_key) in camera_specs.items():
            if image_key not in raw or depth_key not in raw:
                raise LiberoProError(f"LIBERO observation lacks paired RGB-D: {source_name}")
            rgb = np.asarray(raw[image_key])[::-1]
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise LiberoProError(f"invalid RGB shape: {source_name}")
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            meta = self._camera_meta(source_name, int(rgb.shape[0]), int(rgb.shape[1]))
            depth = _metric_depth(raw[depth_key], meta)[::-1].astype(np.float32)
            if depth.shape != rgb.shape[:2]:
                raise LiberoProError(f"RGB-D shape mismatch: {source_name}")
            intrinsic = np.asarray(meta["intrinsic_K"], dtype=np.float64)
            cam2world = np.asarray(meta["extrinsic_cam2world"], dtype=np.float64)
            if intrinsic.shape != (3, 3) or cam2world.shape != (4, 4):
                raise LiberoProError(f"invalid calibration matrices: {source_name}")
            camera_root = target / public_name
            camera_root.mkdir()
            rgb_path = camera_root / "rgb.png"
            depth_path = camera_root / "depth-m.npy"
            calibration_path = camera_root / "calibration.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            _atomic_json(calibration_path, {
                "source_camera": source_name,
                "intrinsic_cv": intrinsic,
                "extrinsic_cv": np.linalg.inv(cam2world),
                "cam2world_cv": cam2world,
                "depth_unit": "metres",
                "valid_depth_range_m": {
                    "near": float(meta["depth_near"]),
                    "far": float(meta["depth_far"]),
                },
                "depth_array_order": "opencv_top_left_row_major",
                "policy_quaternion_order": "xyzw",
            })
            files = {}
            for role, path in (
                ("rgb_lossless_png", rgb_path),
                ("depth_numpy_m_f32", depth_path),
                ("calibration", calibration_path),
            ):
                files[role] = {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": _file_hash(path), "bytes": path.stat().st_size,
                }
            cameras[public_name] = {
                "source_camera": source_name,
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "files": files,
            }
            image_paths.append(rgb_path)
        grip_position, grip_orientation = self._grip_site_pose()
        position = grip_position.tolist()
        orientation = grip_orientation.tolist()
        gripper_qpos = [float(value) for value in raw["robot0_gripper_qpos"]]
        gripper = _normalized_gripper(gripper_qpos)
        proprioception = {
            "end_effectors": {
                "right": {
                    "pose_xyz_xyzw": position + orientation,
                    "pose_frame": "robosuite_grip_site",
                    "gripper": gripper,
                    "gripper_finger_qpos": gripper_qpos,
                    # Command beside measurement: the encoder alone cannot
                    # answer whether the last close took effect.
                    **public_gripper_command(
                        self._gripper_hold_directive(),
                        self.public_gripper_command_target(),
                    ),
                }
            },
            "single_arm": True,
        }
        _atomic_json(target / "proprioception.json", proprioception)
        payload = {
            "schema": OBSERVATION_SCHEMA,
            "observation_id": observation_id,
            "task": self.task,
            "sequence": self.observation_sequence,
            "encoding": "metric_rgbd_v1",
            "captured_unix_s": time.time(),
            "main_camera": "agentview",
            "cameras": cameras,
            "proprioception": proprioception,
            **({"robot_geometry_verification": self.robot_geometry_verification}
               if self.robot_geometry_verification is not None else {}),
            "previous_execution_receipt": self.last_receipt,
            "forbidden_modalities_absent": [
                "actor_state", "object_gt_pose", "segmentation", "contact",
                "success_predicate", "expert_program", "task_source",
            ],
        }
        manifest = target / "observation.json"
        _atomic_json(manifest, payload)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            path.chmod(0o444)
        self.observation_sequence += 1
        return payload, image_paths

    def _current_state(self) -> tuple[Any, Any, float]:
        raw = self._raw_observation()
        position, orientation = self._grip_site_pose()
        gripper = _normalized_gripper(raw["robot0_gripper_qpos"])
        return position, orientation, gripper
