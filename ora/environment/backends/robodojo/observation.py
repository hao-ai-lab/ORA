"""Read-only RoboDojo camera calibration and public observation projection."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
from .bridge import RobodojoError
from .common import array, file_hash, json_default, write_json

class RobodojoObservationMixin:
    def _calibration(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        from .camera_capture import physical_camera_pose, renderer_camera_pose, renderer_intrinsics

        manager = self.environment.camera_manager
        index = manager.camera_names[0].index(camera_name)
        camera = manager.cameras[0][index]
        view = self.environment.capture_manager.tiled_cameras[index]
        parameters = view._harness_camera_params.get_data()
        intrinsic = renderer_intrinsics(parameters, tuple(map(int, view.camera_resolution)))
        original_intrinsic = array(manager.get_camera_intrinsics(index, env_id=0))
        if not np.allclose(intrinsic, original_intrinsic, rtol=1e-5, atol=1e-4):
            raise RobodojoError("render projection differs from the original RoboDojo camera calibration")
        self.calibration_readbacks[camera_name] = {
            "source": "original_render_product_CameraParams", "render_product": view._render_product_path,
            "original_robodojo_intrinsic": original_intrinsic,
            "sdk_raw_aperture_intrinsic": array(camera.get_intrinsics_matrix()),
            "parameters": {key: parameters[key] for key in (
                "cameraProjection", "cameraViewTransform", "renderProductResolution",
                "cameraModel", "cameraAperture", "cameraApertureOffset")},
        }
        # The pinned SDK's Camera.get_world_pose defaults to USD, which may be
        # stale when GPU physics uses Fabric. Query the actual optical prim in
        # the active transform store, not RoboDojo's parent camera rig.
        cam2world = physical_camera_pose(camera, use_fabric=self.environment.use_fabric)
        origin = array(self.environment.sim.scene.env_origins[0])
        cam2world[:3, 3] -= origin
        rendered_pose = renderer_camera_pose(parameters, origin)
        if not np.allclose(rendered_pose, cam2world, rtol=0, atol=1e-4):
            raise RobodojoError("rendered camera pose differs from the current physical optical pose")
        self.calibration_readbacks[camera_name].update(
            cam2world_cv=rendered_pose,
            simulation_time_s=float(view._harness_simulation_time.get_data()["simulationTime"]),
            reference_time=view._harness_reference_time.get_data(),
            current_physical_pose_verified=True,
        )
        offset = getattr(self.environment, "harness_simulation_time_offset_s", None)
        if offset is not None:
            # Scheduled session: the annotator time spans every earlier epoch.
            readback = self.calibration_readbacks[camera_name]
            readback.update(simulation_time_offset_s=float(offset),
                            epoch_simulation_time_s=readback["simulation_time_s"] - float(offset))
        if (intrinsic.shape != (3, 3) or cam2world.shape != (4, 4)
                or not np.isfinite(intrinsic).all() or not np.isfinite(cam2world).all()
                or intrinsic[0, 0] <= 1 or intrinsic[1, 1] <= 1
                or abs(np.linalg.det(cam2world[:3, :3]) - 1) > 1e-3):
            raise RobodojoError("camera calibration is invalid")
        return intrinsic, cam2world

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        from .session import CAMERAS, FORBIDDEN_ABSENT
        from ora.environment.controllers.robodojo_arx import public_feedback_contract
        from PIL import Image

        self._require_started()
        before = self._clock()
        raw = self.environment.get_obs()
        after = self._clock()
        clock_record = {"observation_sequence": self.observation_sequence, "before_physics_steps": before,
                        "after_physics_steps": after, "read_only": before == after}
        with (self.output_root / "observation-clocks.jsonl").open("a") as stream:
            stream.write(json.dumps(clock_record, sort_keys=True) + "\n")
        if after != before:
            self._fault("observation_advanced_physics")
            raise RobodojoError("read-only observation advanced physics")
        # A scheduled session names the round, never the layout ordinal, so no
        # layout id reaches the policy; the unscheduled id is unchanged.
        scene = "layout-0" if self.layout_schedule is None else f"round-{self.round_index}"
        observation_id = f"robodojo:{self.task}:{scene}:observation-{self.observation_sequence:04d}"
        target = Path(destination).resolve() / observation_id
        if target.exists():
            raise RobodojoError("refusing to overwrite an observation")
        target.mkdir(parents=True)
        cameras, image_paths = {}, []
        for public, source in CAMERAS.items():
            data = raw["vision"][source]
            rgb = np.asarray(data["color"])
            depth = np.asarray(data["depth"], dtype=np.float32).squeeze()
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]:
                raise RobodojoError(f"invalid paired RGB-D sensor: {public}")
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                raise RobodojoError(f"camera has no finite positive depth: {public}")
            intrinsic, cam2world = self._calibration(source)
            camera_root = target / public
            camera_root.mkdir()
            rgb_path, depth_path = camera_root / "rgb.png", camera_root / "depth-m.npy"
            calibration_path = camera_root / "calibration.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            write_json(calibration_path, {
                "source_camera": source, "intrinsic_cv": intrinsic,
                "extrinsic_cv": np.linalg.inv(cam2world), "cam2world_cv": cam2world,
                "depth_unit": "metres", "depth_array_order": "opencv_top_left_row_major",
                "policy_quaternion_order": "xyzw", "world_frame": "robodojo_episode_local",
                "depth_kind": "distance_to_image_plane", "finite_depth_fraction": float(valid.mean()),
                "renderer_calibration": self.calibration_readbacks.get(source),
            })
            files = {
                role: {"path": path.relative_to(target).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
                for role, path in (("rgb_lossless_png", rgb_path), ("depth_numpy_m_f32", depth_path), ("calibration", calibration_path))
            }
            cameras[public] = {
                "source_camera": source, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "mount": "fixed_world" if public == "head" else public,
                "calibration_dynamic": public != "head", "modalities": ["rgb", "depth", "calibration"],
                "files": files,
            }
            image_paths.append(rgb_path)
        proprioception = self._proprioception()
        write_json(target / "proprioception.json", proprioception)
        payload = {
            "schema": "ora-observation.v1", "observation_id": observation_id,
            "task": self.task, "sequence": self.observation_sequence, "round_index": self.round_index,
            "encoding": "metric_rgbd_v1", "captured_unix_s": time.time(), "main_camera": "head",
            "cameras": cameras, "proprioception": proprioception,
            "controller_feedback": public_feedback_contract(),
            "previous_execution_receipt": self.last_receipt, "forbidden_modalities_absent": FORBIDDEN_ABSENT,
        }
        if getattr(self, "robot_geometry_verification", None) is not None:
            payload["robot_geometry_verification"] = deepcopy(self.robot_geometry_verification)
        write_json(target / "observation.json", payload)
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        self.observation_sequence += 1
        self.last_observation_id = observation_id
        return payload, image_paths
