"""Observation state, ``observe``, ``mark_point`` and derived views.

Current and historical observation state every tool reads, the
``observe`` and ``mark_point`` tools, and the overlay, point-cloud and
image helpers they render with.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.tools.dual_arm_surface import ROBODOJO_RGBD_CAPABILITY
from ora.contracts.workspace import workspace_state

from ora.agent.tools import mark_point as _mark_point, observe_arguments, saved_steps
from ora.agent.tools.geometry import materialize_two_view_pointcloud
from ora.agent.tools.runtime.common import ROBODOJO_POLICY_CAMERAS
from ora.agent.tools.robot import ToolError, _append
from ora.agent.tools.runtime_artifacts import _artifact_record, _png_base64


class ObserveRuntimeMixin:
    """Observation state, ``observe``, ``mark_point`` and derived views."""

    def _rgb_output_resolution(self, source: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(source) as image:
            native = image.size
        configured = self.view_config["rgb_resolution"]
        return native if configured == "native" else (
            int(configured[0]), int(configured[1])
        )

    def _derived_resolution(self) -> tuple[int, int]:
        configured = self.view_config["derived_resolution"]
        return int(configured[0]), int(configured[1])

    @staticmethod
    def _image_dimensions(path: Path) -> dict[str, int]:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return {"width": width, "height": height}

    def _current_grip_state(self) -> tuple[dict[str, Any], Path, dict[str, Any], list[float], float]:
        control, root, observation = self._current()
        end_effectors = observation.get("proprioception", {}).get("end_effectors", {})
        state = end_effectors.get("right") or end_effectors.get("left")
        if not isinstance(state, Mapping):
            raise ToolError("current grip-site state is unavailable", code="pose_state_unavailable")
        pose = state.get("pose_xyz_xyzw")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ToolError("current grip-site pose is unavailable", code="pose_state_unavailable")
        return control, root, observation, [float(item) for item in pose], float(state.get("gripper", 0))

    @staticmethod
    def _dual_arm_states(observation: Mapping[str, Any]) -> dict[str, list[float]]:
        effectors = observation.get("proprioception", {}).get("end_effectors", {})
        if not isinstance(effectors, Mapping) or set(effectors) != {"left", "right"}:
            raise ToolError(
                "dual-arm sparse surface requires left and right end effectors",
                code="embodiment_binding", retryable=False,
            )
        result: dict[str, list[float]] = {}
        for arm in ("left", "right"):
            state = effectors.get(arm)
            pose = state.get("pose_xyz_xyzw") if isinstance(state, Mapping) else None
            if not isinstance(pose, list) or len(pose) != 7:
                raise ToolError(
                    f"current {arm} end-effector pose is unavailable",
                    code="embodiment_binding", retryable=False,
                )
            result[arm] = [float(item) for item in pose]
        return result

    def _pointcloud_artifact_key(self, observation: Mapping[str, Any]) -> str:
        if self.surface.view_config is None:
            encoded_identity = str(observation["observation_id"]).encode()
        else:
            artifact_identity = {
                "observation_id": observation["observation_id"],
                "derived_resolution": list(self._derived_resolution()),
            }
            encoded_identity = json.dumps(
                artifact_identity, ensure_ascii=False, sort_keys=True,
            ).encode()
        return hashlib.sha256(encoded_identity).hexdigest()[:20]

    def _pointcloud(self, root: Path, observation: Mapping[str, Any]) -> dict[str, Any]:
        key = self._pointcloud_artifact_key(observation)
        target = self.artifact_root / key / "pointcloud"
        manifest_path = target / "manifest.json"
        if manifest_path.is_file():
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("observation_id") != observation.get("observation_id"):
                raise ToolError("pointcloud observation identity drift", code="artifact_hash_mismatch")
            references = [cached.get("points"), *[
                value.get("artifact") for value in cached.get("views", {}).values()
                if isinstance(value, Mapping)
            ]]
            for reference in references:
                relative = reference.get("path") if isinstance(reference, Mapping) else None
                artifact = (target / relative).resolve() if isinstance(relative, str) else None
                if (
                    artifact is None or not artifact.is_file()
                    or not artifact.is_relative_to(target.resolve())
                    or hashlib.sha256(artifact.read_bytes()).hexdigest() != reference.get("sha256")
                ):
                    raise ToolError("pointcloud artifact hash drift", code="artifact_hash_mismatch")
            _append(self.event_path, {
                "type": "pointcloud_requested", "observation_id": observation["observation_id"],
                "cache_hit": True, "time": time.time(),
            })
            return cached
        result = materialize_two_view_pointcloud(
            observation_root=root, observation=observation, output_root=target,
            workspace_bounds_m=self.workspace_bounds_m,
            resolution=self._derived_resolution(),
        )
        _append(self.event_path, {
            "type": "pointcloud_materialized", "record": result, "time": time.time(),
        })
        _append(self.event_path, {
            "type": "pointcloud_requested", "observation_id": observation["observation_id"],
            "cache_hit": False, "time": time.time(),
        })
        return result

    def _write_overlay(
        self, *, source: Path, observation_id: str, identity: Mapping[str, Any],
        points: list[tuple[int, int, str, tuple[int, int, int]]],
        output_resolution: tuple[int, int] | None = None,
    ) -> Path:
        from PIL import Image, ImageDraw

        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
        name = hashlib.sha256(encoded).hexdigest()[:24] + ".png"
        target = self.artifact_root / hashlib.sha256(
            observation_id.encode()
        ).hexdigest()[:20] / "overlays" / name
        image = Image.open(source).convert("RGB")
        if output_resolution is not None and image.size != output_resolution:
            image = image.resize(output_resolution, Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        for x, y, label, colour in points:
            radius = 3
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=colour, width=1,
            )
            draw.point((x, y), fill=colour)
            if label:
                draw.text((x + 4, y + 2), label, fill=colour)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        expected = stream.getvalue()
        if target.is_file():
            if target.read_bytes() != expected:
                raise ToolError(
                    "derived view artifact hash drift",
                    code="artifact_hash_mismatch", retryable=False,
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.png")
            temporary.write_bytes(expected)
            os.replace(temporary, target)
            target.chmod(0o444)
        _append(self.event_path, {
            "type": "openeta_view_derived",
            "observation_id": observation_id,
            "identity": dict(identity),
            "artifact": _artifact_record(target, self.control_root.parent),
            "time": time.time(),
        })
        return target

    def _annotated_rgb(
        self, *, root: Path, observation: Mapping[str, Any], camera: str,
        labelled_world_points: list[tuple[str, list[float], tuple[int, int, int]]],
        identity_kind: str,
    ) -> Path:
        import numpy as np

        _record, rgb, _depth, calibration_path = self._camera(
            root, observation, camera
        )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=float)
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=float)
        from PIL import Image

        with Image.open(rgb) as image:
            width, height = image.size
        output_width, output_height = self._rgb_output_resolution(rgb)
        projected: list[tuple[int, int, str, tuple[int, int, int]]] = []
        identity_points = []
        for label, xyz, colour in labelled_world_points:
            world = np.asarray([*xyz, 1.0], dtype=float)
            camera_point = extrinsic @ world
            if camera_point[2] <= 1e-6:
                continue
            pixel = intrinsic @ camera_point[:3]
            x = int(round(float(pixel[0] / pixel[2])))
            y = int(round(float(pixel[1] / pixel[2])))
            identity_points.append({"label": label, "xyz_m": xyz, "colour": colour})
            if 0 <= x < width and 0 <= y < height:
                projected.append((
                    int(round(x * (output_width - 1) / max(width - 1, 1))),
                    int(round(y * (output_height - 1) / max(height - 1, 1))),
                    label,
                    colour,
                ))
        return self._write_overlay(
            source=rgb,
            observation_id=str(observation["observation_id"]),
            identity={
                "kind": identity_kind, "camera": camera,
                "source_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
                "source_resolution": [width, height],
                "output_resolution": [output_width, output_height],
                "points": identity_points,
            },
            points=projected,
            output_resolution=(output_width, output_height),
        )

    def _annotated_pointcloud(
        self, *, observation: Mapping[str, Any], pointcloud: Mapping[str, Any],
        view: str,
        labelled_world_points: list[tuple[str, list[float], tuple[int, int, int]]],
        identity_kind: str,
    ) -> Path:
        view_record = pointcloud["views"][view]
        width, height = view_record["resolution"]
        h_axis = str(view_record["horizontal_axis"])
        v_axis = str(view_record["vertical_axis"])
        axis_index = {"x": 0, "y": 1, "z": 2}
        h_low, h_high = view_record["horizontal_bounds_m"]
        v_low, v_high = view_record["vertical_bounds_m"]
        projected = []
        identity_points = []
        for label, xyz, colour in labelled_world_points:
            x = int(round(
                (xyz[axis_index[h_axis]] - h_low) / (h_high - h_low) * (width - 1)
            ))
            y = int(round(
                (v_high - xyz[axis_index[v_axis]]) / (v_high - v_low) * (height - 1)
            ))
            identity_points.append({"label": label, "xyz_m": xyz, "colour": colour})
            if 0 <= x < width and 0 <= y < height:
                projected.append((x, y, label, colour))
        key = self._pointcloud_artifact_key(observation)
        source = self.artifact_root / key / "pointcloud" / view_record["artifact"]["path"]
        return self._write_overlay(
            source=source,
            observation_id=str(observation["observation_id"]),
            identity={
                "kind": identity_kind, "view": view,
                "source_sha256": view_record["artifact"]["sha256"],
                "points": identity_points,
            },
            points=projected,
        )

    def begin_read_only_feedback_batch(
        self, name: str, calls: Sequence[Mapping[str, Any]],
    ) -> bool:
        return _mark_point.begin_feedback_batch(self, name, calls)

    def finish_read_only_feedback_batch(
        self, name: str, results: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return _mark_point.finish_feedback_batch(self, name, results)

    def _attach_observe_public_facts(self, result: dict[str, Any], root: Path,
                                    observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self.surface.supports("current_public_robot_facts_v1"):
            return result
        from ora.agent.context.contact_point_projection import project_policy_public_facts
        from ora.agent.context.public_facts import (
            PublicFactsError, controlled_point_projection_delivery, gripper_command_delivery,
        )
        # Only physical camera panels have pinhole calibration. Point-cloud
        # projections remain explicitly named derived views, not virtual sensors.
        views = [view for view in result["views"] if view in observation["cameras"]]
        try:
            facts = project_policy_public_facts(
                observation=observation, observation_root=root, view_config=self.view_config,
                selected_views=views, snapshot_role=("historical" if result.get("status")
                    == "actual_historical_observation" else "current"),
                gripper_command_delivered=gripper_command_delivery(self.view_config),
                controlled_point_projection=controlled_point_projection_delivery(self.view_config),
            )
        except PublicFactsError as exc:
            # The Harness's own published robot facts failed to project: an
            # infrastructure code from the table, never retryable by the policy.
            raise ToolError(str(exc), code="embodiment_binding", retryable=False) from exc
        # Measured joint evidence replaces the old hard-coded 80 mm proxy and
        # command-only summary. Neither establishes attachment or grasp success.
        for key in ("grip_xyz_m", "aperture_mm", "aperture_limits_mm", "end_effectors"):
            result.pop(key, None)
        result["public_robot_facts"] = facts
        return result

    def _observe(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.tools.retention_recovery import enabled as recovery_enabled
        if "step" in arguments and recovery_enabled(self.surface):
            if saved_steps.route_current(self, arguments["step"]) is not None:
                return self._retention_observe(arguments)
            arguments = {key: value for key, value in arguments.items() if key != "step"}
        if self.surface.supports("robotwin_rgbd_v1") or self.surface.supports(ROBODOJO_RGBD_CAPABILITY):
            return self._dual_arm_observe(arguments, allowed=self._physical_policy_cameras(),
                backend="RoboTwin" if self.surface.supports("robotwin_rgbd_v1") else "RoboDojo")
        if set(arguments) - {"views", "history_point_ids"}:
            raise ToolError("observe accepts views and history_point_ids")
        requested = arguments.get("views") or list(
            self.view_config["default_observe_views"]
        )
        if not isinstance(requested, list) or not requested or len(requested) != len(set(requested)):
            raise ToolError("views must be a non-empty unique list")
        allowed = {"agentview", "wrist", "pointcloud_top", "pointcloud_front", "pointcloud_side"}
        if not set(requested).issubset(allowed):
            raise ToolError("unknown observe view", constraints={"enum": sorted(allowed)})
        control, root, observation, pose, gripper = self._current_grip_state()
        history_points = observe_arguments.solved_history(
            self._load_points(), arguments.get("history_point_ids") or [],
        )
        encoded: list[str] = []
        dimensions: dict[str, dict[str, int]] = {}
        pointcloud = None
        for view in requested:
            if view in {"agentview", "wrist"}:
                rendered = self._annotated_rgb(
                    root=root, observation=observation, camera=view,
                    labelled_world_points=[("", pose[:3], (255, 0, 255)), *history_points],
                    identity_kind="observe",
                )
                encoded.append(_png_base64(rendered))
                dimensions[view] = self._image_dimensions(rendered)
            else:
                pointcloud = pointcloud or self._pointcloud(root, observation)
                rendered = self._annotated_pointcloud(
                    observation=observation, pointcloud=pointcloud, view=view,
                    labelled_world_points=history_points,
                    identity_kind="observe",
                )
                encoded.append(_png_base64(rendered))
                width, height = pointcloud["views"][view]["resolution"]
                dimensions[view] = {"width": int(width), "height": int(height)}
        result = {
            "success": True,
            "observation_id": control["observation_id"],
            "views": requested,
            "grip_xyz_m": pose[:3],
            "aperture_mm": gripper * 80.0,
            "aperture_limits_mm": {"near_closed": 3.0, "open": 80.0},
            "images_base64": encoded,
        }
        if self.surface.view_config is not None:
            result["view_dimensions"] = dimensions
        return self._attach_observe_public_facts(result, root, observation)

    def _retention_observation(self, step: Any):
        """Select saved public evidence in the current round; never reset the world."""
        control, _root, _observation, _pose, _gripper = self._current_grip_state()
        states, round_start = saved_steps.round_timeline(self, control)
        refusal = saved_steps.round_step_refusal(step, states, round_start)
        if refusal is not None:
            raise refusal
        _step, state, root, observation = self._select_observation(step)
        effector = observation["proprioception"]["end_effectors"]
        effector = effector.get("right") or effector.get("left")
        selected = {**control, "observation_id": state["observation"]["id"],
                    "observation_sha256": state["observation"]["manifest"]["sha256"]}
        return selected, root, observation, effector["pose_xyz_xyzw"], float(effector.get("gripper", 0))

    def _retention_observe(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        observe_arguments.refuse_historical_arguments(arguments)
        control, root, observation, pose, gripper = self._retention_observation(arguments["step"])
        views = arguments.get("views") or list(self.view_config["default_observe_views"])
        allowed = self._physical_policy_cameras()
        if (not isinstance(views, list) or not views or any(not isinstance(view, str) for view in views)
                or len(views) != len(set(views)) or not set(views) <= allowed):
            raise ToolError("historical observe requires physical camera views",
                            code="invalid_arguments", constraints={"enum": sorted(allowed)})
        encoded, dimensions = [], {}
        for view in views:
            _record, rgb, _depth, _calibration = self._camera(root, observation, view)
            encoded.append(_png_base64(rgb))
            dimensions[view] = self._image_dimensions(rgb)
        result = {"success": True, "observation_id": control["observation_id"],
                "status": "actual_historical_observation",
                "observation_step": arguments["step"], "views": views,
                "view_dimensions": dimensions, "images_base64": encoded}
        if self.surface.supports("robotwin_rgbd_v1"):
            states = self._dual_arm_states(observation)
            result["end_effectors"] = {arm: {"pose_xyz_xyzw": states[arm],
                "gripper_command_target": effector.get("gripper_command_target", "unknown"),
                # Measured jaw separation in metres: the command target is an echo
                # of the last request and cannot be compared against object extents.
                **({"gripper_opening_m": effector.get("gripper_opening_m")}
                   if self.surface.supports("measured_jaw_separation_v1") else {})}
                for arm, effector in observation["proprioception"]["end_effectors"].items()}
        else:
            result.update(grip_xyz_m=pose[:3], aperture_mm=gripper * 80.0,
                          aperture_limits_mm={"near_closed": 3.0, "open": 80.0})
        return self._attach_observe_public_facts(result, root, observation)

    def _physical_policy_cameras(self) -> set[str]:
        """The cameras this surface's observe schema admits, for every backend."""
        return (self._robotwin_policy_cameras() if self.surface.supports("robotwin_rgbd_v1") else set(
            ROBODOJO_POLICY_CAMERAS) if self.surface.supports(ROBODOJO_RGBD_CAPABILITY) else {"agentview", "wrist"})

    def _robotwin_policy_cameras(self) -> set[str]:
        return {"head_camera", "front_camera", "left_camera", "right_camera"}

    @staticmethod
    def _dual_arm_gripper_target(effector: Mapping[str, Any]) -> str:
        """Read one arm's public categorical gripper command target.

        Both backends publish the category with their own proprioception --
        RoboTwin from its cached accepted directive, RoboDojo by projecting the
        normalized command it wrote to the SDK -- so this layer reads that one
        value and never re-derives it from a normalized number or a measured
        finger reading. An absent field stays `unknown`.
        """

        return effector.get("gripper_command_target", "unknown")

    def _dual_arm_observe(
        self, arguments: Mapping[str, Any], *, allowed: set[str], backend: str,
    ) -> dict[str, Any]:
        from ora.agent.tools.dual_arm_observation import _dual_arm_observe as invoke
        return invoke(self, arguments, allowed=allowed, backend=backend)

    def _load_points(self) -> dict[str, Any]:
        if not self.point_path.is_file():
            return {}
        value = json.loads(self.point_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _mark_point(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return _mark_point.execute(self, arguments)

    def _marked_point_workspace(self, xyz_m: Sequence[float]) -> dict[str, Any]:
        state = workspace_state(xyz_m, self.workspace_bounds_m)
        return {
            **state,
            "semantics": "measurement_not_reachability",
            "motion_use": (
                "candidate_requires_motion_preview"
                if state["inside"] else "measurement_only_outside_action_envelope"
            ),
        }
