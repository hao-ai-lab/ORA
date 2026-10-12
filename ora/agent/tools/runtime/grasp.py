"""``grasp_plan`` and the internal read-only motion probe.

Grasp proposals for a segmented target, the embodiment compiler
identity, and the reachability probe the closed grasp skill runs over
its committed pregrasp (``grasp_feasibility.check_pregrasp``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from ora.agent.tools.dual_arm_surface import (
    ARX_X5_GRASP_COMPILER_CAPABILITY, dual_arm_grasp_scene_cameras,
)
from ora.contracts.pose import PoseCompilationError, canonicalize_quaternion
from ora.contracts.refusal import EnvironmentRefusal
from ora.contracts.workspace import workspace_state

from ora.agent.tools import (
    motion_arguments, perception_feedback, reach_check, scene_extent_review,
    service_refusals, skill_retention, target_refusals, x5_topdown,
)
from ora.agent.tools.geometry import fuse_rgbd_pointcloud_sources
from ora.agent.tools.grasp import (
    GRASP_FUSION_MAX_DEPTH_M, GRASP_FUSION_VOXEL_SIZE_M,
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    grasp_plan_public_result, load_segmented_target,
    materialize_grasp_query,
)
from ora.agent.tools.mark_point import opencv_camera_point_to_world as _opencv_camera_point_to_world
from ora.agent.tools.motion_arguments import (
    project_copyable_approach_targets as _project_copyable_grasp_approach_targets,
)
from ora.agent.tools.robot import ToolError, _append, _finite_vector
from ora.agent.tools.runtime_artifacts import _artifact_record, _png_base64


class _GraspReviewSources(NamedTuple):
    """The review cameras ``grasp_plan`` could and could not use.

    ``binding`` is ``None`` unless the surface publishes RGB-D perception
    evidence. The manifest, the materialization event and the public result
    share these objects.
    """

    available: list[str]
    unavailable: list[str]
    binding: dict[str, Any] | None


def _grasp_request_fields(
    arguments: Mapping[str, Any], *, current_observation_id: str,
) -> tuple[str, str, int]:
    """Bind the observation, the target and the candidate count of one request."""
    observation_id = current_observation_id
    if (
        not isinstance(observation_id, str)
        or observation_id != current_observation_id
    ):
        raise ToolError(
            "grasp query requires the current observation_id",
            code="stale_observation", field="observation_id",
            received=observation_id,
            constraints={"current": current_observation_id},
        )
    target_ref = arguments["target_ref"]
    if not isinstance(target_ref, str) or not target_ref:
        raise ToolError("target_ref must be a non-empty string", field="target_ref")
    maximum = 3
    max_candidates = arguments.get("max_candidates", maximum)
    if (
        isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
        or not 1 <= max_candidates <= maximum
    ):
        raise ToolError(
            f"max_candidates must be an integer in [1,{maximum}]",
            field="max_candidates", received=max_candidates,
        )
    return observation_id, target_ref, max_candidates


class GraspRuntimeMixin:
    """``grasp_plan`` and the internal read-only motion probe."""

    def _grasp_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self._require_grasp_argument_names(arguments)
        control, root, observation, _pose, _gripper = self._current_grip_state()
        observation_id, target_ref, max_candidates = _grasp_request_fields(
            arguments,
            current_observation_id=str(control.get("observation_id")),
        )
        camera = self._grasp_target_camera(target_ref, observation_id)
        _record, rgb_path, depth_path, calibration_path = self._camera(
            root, observation, camera,
        )
        arm = arguments.get("arm")
        scene_cameras = self._grasp_scene_cameras(arm, root, observation)
        review_sources = self._grasp_review_sources(
            root=root, observation=observation, camera=camera, target_ref=target_ref,
            scene_cameras=scene_cameras,
        )
        manifest, overlay_path = self._materialize_grasp_manifest(
            arm=arm, observation_id=observation_id,
            target_ref=target_ref, root=root, observation=observation,
            rgb_path=rgb_path, depth_path=depth_path,
            calibration_path=calibration_path, max_candidates=max_candidates,
            scene_cameras=scene_cameras, review_sources=review_sources,
        )
        review = self._record_grasp_materialization(
            observation_id=observation_id, target_ref=target_ref,
            manifest=manifest, overlay_path=overlay_path,
            review_source_binding=review_sources.binding,
        )
        result = grasp_plan_public_result(manifest, overlay_path)
        if self.surface.supports("public_rgbd_perception_evidence_v1"):
            self._attach_grasp_review_evidence(
                result, manifest, review=review, camera=camera,
                scene_cameras=scene_cameras, review_sources=review_sources,
            )
        self._qualify_grasp_plan_candidates(result, manifest, arm)
        result["images_base64"] = [_png_base64(overlay_path)]
        if perception_feedback.enabled(self.surface):
            perception_feedback.grasp_review(result, manifest, overlay_path)
        if self.surface.supports("dual_arm_grasp_plan_v1"):
            result["arm"] = arm
        return result

    def _require_grasp_argument_names(self, arguments: Mapping[str, Any]) -> None:
        """Refuse an argument set the surface's grasp tool does not accept."""
        allowed = (
            {"target_ref", "max_candidates", "arm"}
            if self.surface.supports("dual_arm_grasp_plan_v1")
            else {"target_ref", "max_candidates"}
        )
        required = (
            {"target_ref", "arm"}
            if self.surface.supports("dual_arm_grasp_plan_v1")
            else {"target_ref"}
        )
        if set(arguments) - allowed or not required.issubset(arguments):
            raise ToolError(
                "grasp_plan requires target_ref; the Runtime binds the current "
                "observation automatically",
                code="invalid_arguments",
            )

    def _grasp_target_camera(self, target_ref: str, observation_id: str) -> str:
        """Load the segmented target and return the camera it was segmented in."""
        try:
            target, _mask_path, _target_overlay = load_segmented_target(
                artifact_root=self.artifact_root, target_ref=target_ref,
            )
        except ValueError as exc:
            raise target_refusals.unavailable_target(
                exc, target_ref=target_ref, artifact_root=self.artifact_root,
                observation_id=observation_id,
            ) from exc
        identity = target.get("identity", {})
        if identity.get("observation_id") != observation_id:
            raise target_refusals.stale_target(
                target_ref=target_ref, identity=identity,
                current_observation_id=observation_id,
            )
        camera = identity.get("camera")
        if not isinstance(camera, str):
            raise ToolError(
                "target_ref lacks a source camera",
                code="artifact_hash_mismatch", retryable=False,
            )
        return camera

    def _grasp_scene_cameras(
        self, arm: Any, root: Path, observation: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return the review cameras; a dual-arm plan also binds its arm."""
        if self.surface.supports("dual_arm_grasp_plan_v1"):
            if arm not in {"left", "right"}:
                raise ToolError(
                    "grasp_plan arm must be left or right",
                    code="invalid_arguments", field="arm",
                )
            scene_cameras = dual_arm_grasp_scene_cameras(self.surface, arm)
            if not self.surface.supports("public_rgbd_perception_evidence_v1"):
                for scene_camera in scene_cameras:
                    self._camera(root, observation, scene_camera)
        else:
            scene_cameras = ("agentview", "wrist")
        return scene_cameras

    def _grasp_review_sources(
        self, *, root: Path, observation: Mapping[str, Any],
        camera: str, scene_cameras: tuple[str, ...], target_ref: str = "",
    ) -> _GraspReviewSources:
        """Resolve which review cameras carry a usable public RGB-D source."""
        # Correctness surfaces keep the target source strict and make review
        # cameras optional on both embodiments, including head-only scenes.
        review_available: list[str] = []
        review_unavailable: list[str] = []
        review_source_binding: dict[str, Any] | None = None
        if self.surface.supports("public_rgbd_perception_evidence_v1"):
            # The same decoding and geometry checks stay strict for the target
            # source, even when that camera is also one of the review views.
            try:
                primary_review_source = self._usable_grasp_review_source(
                    root, observation, camera,
                )
            except (ToolError, ValueError, TypeError, KeyError, OSError, EOFError) as exc:
                # A published calibration or artifact fault keeps its own
                # infrastructure code; the rest stays this tool's failure, with
                # its detail. Whether a different target_ref can help is not
                # known here (a corrupt depth file and an empty view look alike).
                if isinstance(exc, ToolError) and exc.as_dict()["infrastructure"]:
                    raise
                raise ToolError(
                    f"grasp primary RGB-D source is unusable: camera {camera} of "
                    f"target_ref {target_ref}: {exc}",
                    code="grasp_proposal_failed", retryable=True,
                ) from exc
            review_source_binding = {"requested_cameras": list(scene_cameras),
                                     "available_sources": [], "unavailable_sources": []}
            for scene_camera in scene_cameras:
                try:
                    source = (primary_review_source if scene_camera == camera
                              else self._usable_grasp_review_source(
                                  root, observation, scene_camera,
                              ))
                except (ToolError, ValueError, TypeError, KeyError, OSError, EOFError):
                    review_unavailable.append(scene_camera)
                    review_source_binding["unavailable_sources"].append({
                        "camera": scene_camera, "reason": "unusable_public_rgbd_review_source",
                    })
                else:
                    review_available.append(scene_camera)
                    review_source_binding["available_sources"].append(source)
        return _GraspReviewSources(
            review_available, review_unavailable, review_source_binding,
        )

    def _usable_grasp_review_source(
        self, root: Path, observation: Mapping[str, Any], name: str,
    ) -> dict[str, Any]:
        """Decode one public RGB-D camera; return its fused-source record."""
        _, _, _, source_calibration = self._camera(root, observation, name)
        calibration = json.loads(source_calibration.read_text(encoding="utf-8"))
        if not isinstance(calibration, Mapping):
            raise ToolError("camera calibration must be an object",
                            code="invalid_camera_calibration")
        _opencv_camera_point_to_world([0.0, 0.0, 1.0, 1.0],
                                      calibration=calibration)
        fused = fuse_rgbd_pointcloud_sources(
            observation_root=root, observation=observation,
            workspace_bounds_m=self.workspace_bounds_m, camera_names=(name,),
            voxel_size_m=GRASP_FUSION_VOXEL_SIZE_M,
            max_depth_m=GRASP_FUSION_MAX_DEPTH_M,
        )
        return dict(fused["sources"][0])

    def _materialize_grasp_manifest(
        self, *, arm: Any, observation_id: str, target_ref: str,
        root: Path, observation: Mapping[str, Any], rgb_path: Path,
        depth_path: Path, calibration_path: Path, max_candidates: int,
        scene_cameras: tuple[str, ...], review_sources: _GraspReviewSources,
    ) -> tuple[dict[str, Any], Path]:
        """Propose and compile the candidates; write the query manifest."""
        review_available = review_sources.available
        review_source_binding = review_sources.binding
        try:
            manifest, overlay_path = materialize_grasp_query(
                artifact_root=self.artifact_root,
                observation_id=observation_id,
                target_ref=target_ref,
                observation_root=root,
                observation=observation,
                rgb_path=rgb_path,
                depth_path=depth_path,
                calibration_path=calibration_path,
                workspace_bounds_m=self.workspace_bounds_m,
                max_candidates=max_candidates,
                backend=self.grasp_backend,
                compile_for_execution=True,
                scene_cameras=(tuple(review_available) if review_source_binding is not None
                               else scene_cameras),
                # The proposer receives the same fused scene the review canvas
                # shows. Withholding it made self-occluded geometry invisible to
                # the network while it was drawn for the policy; a second camera
                # adds a median 53% more points inside the crop the network
                # actually reads. Surfaces without the capability keep the
                # single-view identity exactly.
                backend_scene_cameras=(
                    tuple(review_available)
                    if review_source_binding is not None
                    and self.surface.supports("fused_backend_grasp_input_v1")
                    else None
                ),
                review_source_binding=review_source_binding,
                compiler_id=self._grasp_compiler_id(arm),
                execution_tool=(
                    "move_pose"
                    if self.surface.supports("enhanced_move_pose_v1")
                    else "move_free_space"
                ),
                public_contact_pose=self.surface.supports(
                    "public_grasp_contact_pose_v1"
                ),
                candidate_selection=reach_check.grasp_candidate_selection(self, arm, observation),
                observed_extent=self.surface.supports(scene_extent_review.CAPABILITY),
                review_policy_visible=perception_feedback.enabled(self.surface),
                lift_m=(skill_retention.LIFT_M
                        if self.surface.supports(skill_retention.CAPABILITY) else None),
            )
        except ValueError as exc:
            raise service_refusals.refusal(
                exc, code="grasp_proposal_failed", target_ref=target_ref,
            ) from exc
        return manifest, overlay_path

    def _record_grasp_materialization(
        self, *, observation_id: str, target_ref: str,
        manifest: Mapping[str, Any], overlay_path: Path,
        review_source_binding: dict[str, Any] | None,
    ) -> Any:
        """Append the materialization event; return the review visualization."""
        input_pointcloud_path = (
            overlay_path.parent / manifest["input_pointcloud"]["artifact"]["path"]
        )
        event = {
            "type": "grasp_plan_materialized",
            "observation_id": observation_id,
            "target_ref": target_ref,
            "query_ref": manifest["query_ref"],
            "backend": manifest["backend"],
            "candidate_count": len(manifest["candidates"]),
            "overlay_sha256": manifest["overlay"]["sha256"],
            "policy_overlay_artifact": _artifact_record(
                overlay_path, self.control_root.parent,
            ),
            "input_pointcloud_artifact": _artifact_record(
                input_pointcloud_path, self.control_root.parent,
            ),
            "manifest_artifact": _artifact_record(
                overlay_path.parent / "manifest.json", self.control_root.parent,
            ),
            "visualization": manifest.get("visualization"),
            "review_visualization": manifest.get("review_visualization"),
            "time": time.time(),
        }
        if review_source_binding is not None:
            event["review_source_binding"] = review_source_binding
        review = manifest.get("review_visualization")
        if isinstance(review, Mapping) and review.get("status") == "available":
            review_path = overlay_path.parent / review["artifact"]["path"]
            event["artifact"] = _artifact_record(
                review_path, self.control_root.parent,
            )
        review_cloud = manifest.get("review_pointcloud")
        if isinstance(review_cloud, Mapping):
            review_cloud_path = overlay_path.parent / review_cloud["artifact"]["path"]
            event["review_pointcloud_artifact"] = _artifact_record(
                review_cloud_path, self.control_root.parent,
            )
        _append(self.event_path, event)
        return review

    def _attach_grasp_review_evidence(
        self, result: dict[str, Any], manifest: Mapping[str, Any], *,
        review: Any, camera: str, scene_cameras: tuple[str, ...],
        review_sources: _GraspReviewSources,
    ) -> None:
        """Add the plan's review-camera evidence and its claim boundary."""
        review_available = review_sources.available
        review_unavailable = review_sources.unavailable
        result["review_evidence"] = {
            "status": "available" if isinstance(review, Mapping)
            and review.get("status") == "available" else "unavailable",
            "primary_camera": camera, "requested_cameras": list(scene_cameras),
            "available_cameras": review_available, "unavailable_cameras": review_unavailable,
            "primary_input_required": True,
            "claim_boundary": "optional_review_only;primary_rgbd_input_remains_required;not_grasp_success",
        }
        if self.surface.supports("fused_backend_grasp_input_v1"):
            backend_input = manifest.get("input_pointcloud", {}).get("backend_input_profile", {})
            result["review_evidence"]["claim_boundary"] = (
                "primary_rgbd_input_remains_required;backend_input_scope="
                + str(backend_input.get("scope", "unavailable"))
                + ";backend_cameras="
                + ",".join(backend_input.get("contributing_cameras", []))
                + ";review_is_not_grasp_success"
            )

    def _qualify_grasp_plan_candidates(
        self, result: dict[str, Any], manifest: Mapping[str, Any], arm: Any,
    ) -> None:
        """Publish the selection, reachability and copyable approach targets."""
        reach_check.publish_grasp_selection(self, result, manifest)
        if self.surface.supports("enhanced_move_pose_v1"):
            for candidate in result["candidates"]:
                candidate["compiler_checks"]["reachability"] = (
                    "delegated_to_move_pose"
                )
        if self.surface.supports("copyable_grasp_approach_target_v1"):
            _project_copyable_grasp_approach_targets(
                result,
                move_pose_schema=motion_arguments.surface_move_pose_schema(self.surface),
                semantic_interaction=any(
                    self.surface.supports(capability)
                    for capability in (
                        "motion_intent_router_v1", "motion_intent_router_v2",
                        # The single-arm flat move_pose form also carries the
                        # semantic interaction field.
                        "single_arm_flat_targets_v1",
                    )
                ),
                allow_grip_site_alias=self.surface.supports(
                    "panda_grip_site_axes_v1"
                ),
            )

    def _grasp_compiler_id(self, arm: str) -> str:
        if self.surface.supports(x5_topdown.CAPABILITY):
            return x5_topdown.compiler_id(arm)
        if self.surface.supports(ARX_X5_GRASP_COMPILER_CAPABILITY):
            return f"contact_graspnet_to_robodojo_x5_{arm}_link6_v1"
        if self.surface.supports(
            "contact_graspnet_robotwin_aloha_grip_center_compiler_v3"
        ):
            return f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v3"
        if self.surface.supports("dual_arm_grasp_plan_v1"):
            return f"contact_graspnet_to_robotwin_aloha_{arm}_end_effector_v1"
        return str(PANDA_GRIP_SITE_COMPILER_IDENTITY["id"])

    def _motion_probe(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.motion_probe_callback is None:
            raise ToolError(
                "motion.probe requires the Native environment read-only bridge",
                code="tool_unavailable", retryable=False,
            )
        if set(arguments) != {"queries", "variant_policy"}:
            raise ToolError(
                "motion.probe requires queries and variant_policy",
                code="invalid_arguments",
            )
        queries = arguments.get("queries")
        policy = arguments.get("variant_policy")
        if (
            not isinstance(queries, list) or not 1 <= len(queries) <= 12
            or policy not in {"exact", "parallel_jaw_equivalents"}
        ):
            raise ToolError("motion.probe request is invalid", code="invalid_arguments")
        control, root, observation, _pose, _gripper = self._current_grip_state()
        normalized: list[dict[str, Any]] = []
        query_ids: set[str] = set()
        for query in queries:
            if not isinstance(query, Mapping) or set(query) - {
                "query_id", "arm", "pose",
            }:
                raise ToolError("motion.probe query fields are invalid")
            query_id = query.get("query_id")
            arm = query.get("arm")
            if arm is None and self.surface.supports(
                "single_arm_flat_targets_v1"
            ):
                arm = "right"
                query = {**query, "arm": "right"}
            if (
                not isinstance(query_id, str) or not query_id
                or query_id in query_ids or arm not in {"left", "right"}
                or "pose" not in query
            ):
                raise ToolError("motion.probe query identity is invalid")
            pose = query.get("pose")
            if not isinstance(pose, Mapping) or set(pose) != {
                "position_m", "orientation_xyzw",
            }:
                raise ToolError("motion.probe pose is invalid")
            position = _finite_vector(pose["position_m"], 3, "position_m")
            workspace = workspace_state(position, self.workspace_bounds_m)
            if not workspace["inside"]:
                raise ToolError(
                    "motion.probe target is outside the action workspace; "
                    "select a position within the declared bounds",
                    code="invalid_arguments",
                    field=f"queries.{query_id}.pose.position_m", received=position,
                    constraints={
                        "allowed": dict(self.workspace_bounds_m),
                        "violations": workspace["violations"],
                    },
                )
            try:
                orientation = canonicalize_quaternion(
                    pose["orientation_xyzw"], "orientation_xyzw",
                )
            except PoseCompilationError as exc:
                raise ToolError(
                    str(exc), code="invalid_arguments",
                    field="orientation_xyzw",
                    received=pose.get("orientation_xyzw"),
                    constraints={"norm_min": 0.98, "norm_max": 1.02},
                ) from exc
            normalized.append({
                "query_id": query_id, "arm": arm,
                "pose": {
                    "position_m": position,
                    "orientation_xyzw": orientation,
                },
            })
            query_ids.add(query_id)
        from ora.agent.tools.motion_probe_feedback import probe_request_fields
        try:
            result = dict(self.motion_probe_callback({
                **probe_request_fields(self.surface),
                "observation_id": control["observation_id"],
                "observation_sha256": control["observation_sha256"],
                "queries": normalized,
                "variant_policy": policy,
            }))
        except EnvironmentRefusal as error:
            raise ToolError(
                str(error), code=error.code, retryable=error.retryable,
                field=error.field, received=error.received,
                constraints={"allowed": error.allowed},
            ) from error
        if result.get("schema") != "ora.motion-probe-result.v1":
            raise ToolError(
                "environment motion probe result schema drifted",
                code="environment_contract", retryable=False,
            )
        query_results = result.get("results")
        if not isinstance(query_results, list) or len(query_results) != len(normalized):
            raise ToolError("motion probe query result count differs",
                            code="environment_contract", retryable=False)
        registrations: dict[str, dict[str, Any]] = {}
        for query, query_result in zip(normalized, query_results):
            if (not isinstance(query_result, Mapping)
                    or query_result.get("query_id") != query["query_id"]
                    or query_result.get("arm") != query["arm"]):
                raise ToolError("motion probe query result identity differs",
                                code="environment_contract", retryable=False)
            variants = query_result.get("variants", [])
            if not isinstance(variants, list) or any(not isinstance(v, Mapping) for v in variants):
                raise ToolError("motion probe variant results are invalid",
                                code="environment_contract", retryable=False)
            handles = [(query_result.get("plan_ref"),
                        query_result.get("selected_variant_id"),
                        query_result.get("selected_target_pose_xyzw"))]
            handles.extend((v.get("plan_ref"), v.get("variant_id"),
                            v.get("target_pose_xyzw")) for v in variants
                           if v.get("reachable") is True)
            for plan_ref, selected_variant, target_pose in handles:
                if plan_ref is None:
                    continue
                if (not isinstance(plan_ref, str) or not plan_ref
                        or not isinstance(selected_variant, str) or not selected_variant
                        or not isinstance(target_pose, list) or len(target_pose) != 7
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in target_pose)):
                    raise ToolError("motion probe plan handle target is invalid",
                                    code="environment_contract", retryable=False)
                binding = {
                    "observation_id": control["observation_id"],
                    "observation_sha256": control["observation_sha256"],
                    "arm": query["arm"],
                    "variant_id": selected_variant,
                    "target_pose_xyzw": list(target_pose),
                    "submitted_mutation_index": len(self._submitted_mutations),
                }
                if plan_ref in registrations and registrations[plan_ref] != binding:
                    raise ToolError("motion probe plan handle has conflicting targets",
                                    code="environment_contract", retryable=False)
                registrations[plan_ref] = binding
        self._known_plan_refs.update(registrations)
        return result
