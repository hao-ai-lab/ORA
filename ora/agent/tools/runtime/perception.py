"""``vision_segment``, ``vision_ground`` and target geometry.

The segmentation and text-grounding tools and the current-observation
geometry of a segmented target. The frozen-surface ``vision_offset`` runs
only at the commits that composed it.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ora.agent.tools.dual_arm_surface import (
    ROBODOJO_RGBD_CAPABILITY, conform_to_result_schema,
)

from ora.agent.tools import (
    grasp, grounding_preview, overhead_view, perception_arguments, saved_steps, service_refusals,
    target_refusals,
)
from ora.agent.tools.grasp import (
    DEPTH_INLIER_GEOMETRY_CAPABILITY, GROUNDING_MAX_CANDIDATES, grounding_public_result,
    load_segmented_target, materialize_grounded_targets,
    perception_selection_evidence,
    segmented_target_geometry,
)
from ora.agent.tools.runtime.common import ROBODOJO_POLICY_CAMERAS
from ora.agent.tools.robot import ToolError, _append
from ora.agent.tools.runtime_artifacts import _png_base64


class PerceptionRuntimeMixin:
    """``vision_segment``, ``vision_ground`` and target geometry."""

    def _vision_segment(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.tools import segmentation_batch
        contract = segmentation_batch.select_contract(self.surface)
        if contract is None:
            raise ToolError(
                "vision_segment requires the batch point-segmentation capability",
                code="tool_surface_capability_mismatch", retryable=False,
            )
        return self._vision_segment_batch(arguments, contract)

    def _overhead_view_poses(self, result: Any) -> Any:
        """On a surface declaring overhead_footprint_localization_v1, add the overhead wrist pose to oblique measurements.

        The pose comes from the current grip-site pose and the current wrist calibration; without a
        current wrist camera the measurement simply carries no pose.
        """
        if not self.surface.supports(overhead_view.CAPABILITY):
            return result
        import json

        try:
            _control, root, observation, pose, _gripper = self._current_grip_state()
            _record, _rgb, _depth, calibration_path = self._camera(root, observation, "wrist")
            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        except (ToolError, OSError, ValueError, KeyError):
            return result
        camera_to_world = calibration.get("cam2world_cv")
        if camera_to_world is None:
            return result
        return overhead_view.annotate(result, grip_pose_xyz_xyzw=pose, wrist_camera_to_world=camera_to_world)

    def _vision_segment_batch(self, arguments: Mapping[str, Any], contract: Any) -> dict[str, Any]:
        # `contract` is the published batch result the resolved surface declares;
        # a frozen surface must keep receiving the bytes it was sealed with.
        from ora.agent.tools import segmentation_batch

        if (not {"camera", "points_uv"} <= set(arguments)
                or set(arguments) - {"camera", "points_uv", "max_candidates", "step"}):
            raise ToolError("vision_segment requires camera and points_uv; optional max_candidates and step",
                            code="invalid_arguments")
        camera = arguments["camera"]
        if not isinstance(camera, str) or not camera:
            raise ToolError("camera must be a public camera name", field="camera")
        control, root, observation, _pose, _gripper = self._current_grip_state()
        if saved_steps.route_current(self, arguments.get("step")) is not None:
            control, root, observation, _pose, _gripper = self._retention_observation(arguments["step"])
        record, rgb_path, depth_path, calibration_path = self._camera(root, observation, camera)
        maximum = perception_arguments.batch_point_arguments(
            arguments["points_uv"], arguments.get("max_candidates", 3),
            width=int(record["width"]), height=int(record["height"]),
        )
        if not callable(getattr(self.segmentation_backend, "segment_points", None)):
            raise ToolError("selected segmentation backend does not support independent batch queries",
                            code="tool_capability_unavailable", retryable=False)
        try:
            manifest, paths, cached = contract.materialize_batch(
                artifact_root=self.artifact_root, observation_id=str(control["observation_id"]),
                camera=camera, rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path,
                points_uv=arguments["points_uv"], max_candidates=maximum, backend=self.segmentation_backend,
                **({"depth_profile": profile} if (profile := grounding_preview.options(self.surface)) else {}))
        except ValueError as exc:
            raise service_refusals.refusal(exc, code="segmentation_failed") from exc
        _append(self.event_path, {"type": "segmentation_batch_materialized", "time": time.time(),
            "observation_id": control["observation_id"], "camera": camera,
            "query_ref": manifest["query_ref"], "target_refs": [c["target_ref"] for c in manifest["candidates"]],
            "cache_reused": cached})
        result = grasp.project_target_frame(
            contract.public_result(manifest, cache_reused=cached),
            **grasp.target_support.projection(self.surface, grasp.TARGET_PRINCIPAL_FRAME_CAPABILITY),
            footprint_declared=self.surface.supports(overhead_view.CAPABILITY),
        )
        result = self._overhead_view_poses(result)
        result["images_base64"] = [_png_base64(path) for path in paths]
        return result

    def _vision_ground(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"camera", "text", "max_candidates"}
        if set(arguments) - allowed or not {"camera", "text"}.issubset(arguments):
            raise ToolError(
                "vision_ground requires camera and text; the Runtime binds the "
                "current observation automatically",
                code="invalid_arguments",
            )
        control, root, observation, _pose, _gripper = self._current_grip_state()
        observation_id = str(control["observation_id"])
        camera = arguments["camera"]
        allowed_cameras = (
            self._robotwin_policy_cameras()
            if self.surface.supports("robotwin_rgbd_v1")
            else set(ROBODOJO_POLICY_CAMERAS)
            if self.surface.supports(ROBODOJO_RGBD_CAPABILITY)
            else {"agentview", "wrist"}
        )
        if camera not in allowed_cameras:
            raise ToolError(
                "camera is not available to text grounding", field="camera",
                received=camera, constraints={"enum": sorted(allowed_cameras)},
            )
        text = arguments["text"]
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 120:
            raise ToolError(
                "text must contain 1..120 non-whitespace characters", field="text",
            )
        # Two different limits were being conflated. The palette's length bounds
        # what the overview can draw in distinguishable colours; the resident
        # service bounds what it will answer at all, and refuses anything past
        # three with a 422. Taking the palette as the request bound sent eleven
        # doomed calls in one cohort -- 4, 6 and 8 candidates, none answered,
        # against 291 successes at three or fewer -- and handed the policy an
        # HTTP status and a documentation link instead of the limit. The
        # smaller of the two is the one a caller has to satisfy, and the
        # constraint travels with the refusal so it can be read rather than
        # guessed.
        ceiling = min(GROUNDING_MAX_CANDIDATES, grasp.GROUNDING_REQUEST_MAX_CANDIDATES)
        max_candidates = arguments.get("max_candidates", 3)
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= ceiling
        ):
            raise ToolError(
                f"max_candidates must be an integer in [1,{ceiling}]; the "
                "grounding service answers at most that many ranked regions "
                "per call",
                field="max_candidates", received=max_candidates,
                constraints={"minimum": 1, "maximum": ceiling},
            )
        _record, rgb_path, depth_path, calibration_path = self._camera(
            root, observation, camera,
        )
        if self.grounding_backend is None:
            raise ToolError(
                "vision_ground backend is unavailable", code="grounding_unavailable",
                retryable=True,
            )
        try:
            manifest, policy_image_paths = materialize_grounded_targets(
                artifact_root=self.artifact_root,
                observation_id=observation_id,
                camera=camera,
                rgb_path=rgb_path,
                depth_path=depth_path,
                calibration_path=calibration_path,
                text=text.strip(),
                max_candidates=max_candidates,
                backend=self.grounding_backend,
                depth_profile=grounding_preview.options(self.surface),
            )
        except ValueError as exc:
            raise service_refusals.refusal(exc, code="grounding_failed") from exc
        _append(self.event_path, {
            "type": "grounding_query_materialized",
            "observation_id": observation_id,
            "camera": camera,
            "text": text.strip(),
            "grounding_ref": manifest["grounding_ref"],
            "backend": manifest["backend"],
            "candidate_count": len(manifest["candidates"]),
            "overlay_sha256": manifest["overlay"]["sha256"],
            "candidate_previews": manifest["visualization"]["candidate_previews"],
            "time": time.time(),
        })
        result = grasp.project_target_frame(
            grounding_public_result(manifest, policy_image_paths[0]),
            **grasp.target_support.projection(self.surface, grasp.TARGET_PRINCIPAL_FRAME_CAPABILITY),
            footprint_declared=self.surface.supports(overhead_view.CAPABILITY),
        )
        result = self._overhead_view_poses(result)
        if self.surface.supports("public_rgbd_perception_evidence_v1"):
            result["perception_evidence"] = perception_selection_evidence(
                result=result, artifact_root=self.artifact_root,
                source_size=(int(_record["width"]), int(_record["height"])),
            )
        if self.surface.supports(ROBODOJO_RGBD_CAPABILITY):
            result = conform_to_result_schema(
                self.surface.result_schemas.get("vision_ground", self.surface.result_schemas.get("vision.ground")),
                result,
            )
        result["images_base64"] = [
            _png_base64(path) for path in policy_image_paths
        ]
        return result

    def _current_target_geometry(
        self, *, control: Mapping[str, Any], root: Path,
        observation: Mapping[str, Any], target_ref: str,
    ) -> dict[str, Any]:
        try:
            target, _mask, _overlay = load_segmented_target(
                artifact_root=self.artifact_root, target_ref=target_ref,
            )
        except ValueError as exc:
            raise target_refusals.unavailable_target(
                exc, target_ref=target_ref, artifact_root=self.artifact_root,
                observation_id=str(control["observation_id"]),
            ) from exc
        identity = target.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("observation_id") != control["observation_id"]
            or not isinstance(identity.get("camera"), str)
        ):
            raise target_refusals.stale_target(
                target_ref=target_ref, identity=identity,
                current_observation_id=str(control["observation_id"]),
            )
        _record, _rgb, depth_path, calibration_path = self._camera(
            root, observation, str(identity["camera"]),
        )
        try:
            geometry = segmented_target_geometry(
                artifact_root=self.artifact_root,
                target_ref=target_ref,
                depth_path=depth_path,
                calibration_path=calibration_path,
                voxel_size_m=0.01,
                use_depth_inliers=self.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY),
                rgb_path=_rgb,
            )
            if geometry.get("status") == "unknown":
                raise target_refusals.GeometryQualityRefusal(geometry)
            return geometry
        except target_refusals.GeometryQualityRefusal as refusal:
            raise target_refusals.unmeasurable_geometry(
                target_ref=target_ref, geometry=refusal.geometry,
            ) from refusal
        except ValueError as exc:
            raise ToolError(
                str(exc), code="target_geometry_unavailable", field="target_ref", received=target_ref, retryable=True,
                retry_requires_changed_arguments=True,
            ) from exc
