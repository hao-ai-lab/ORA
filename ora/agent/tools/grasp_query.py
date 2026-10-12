"""Steps of ``grasp.materialize_grasp_query``.

``materialize_grasp_query`` in ``grasp.py`` binds the target and runs these
steps in order: admit the proposer's cameras, hash the query identity, reuse
a verified manifest or call the backend, then write the input cloud, the
policy-visible contact sheet, the best-effort global review, the backend
arrays and the manifest. This module imports ``grasp.py``'s artifact,
geometry and rendering helpers, so ``grasp.py`` imports it when a query runs
rather than at module import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from . import scene_extent_review, skill_geometry
from .geometry import fuse_rgbd_pointcloud_sources
from .grasp import (
    GRASP_BACKEND_EXCLUSION_REASONS,
    GRASP_BACKEND_FUSED_INPUT_SCOPE,
    GRASP_BACKEND_FUSED_SCOPE,
    GRASP_BACKEND_INPUT_FUSION_SCHEMA,
    GRASP_BACKEND_MINIMUM_CAMERA_POINTS,
    GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE,
    GRASP_BACKEND_SINGLE_VIEW_SCOPE,
    GRASP_CANDIDATE_COLOURS,
    GRASP_COMPILER_IDENTITIES,
    GRASP_FUSION_MAX_DEPTH_M,
    GRASP_FUSION_VOXEL_SIZE_M,
    GRASP_INPUT_POINTCLOUD_SCHEMA,
    GRASP_PLAN_REVIEW_ID,
    GRASP_PLAN_VISUALIZATION_ID,
    GRASP_QUERY_SCHEMA,
    GraspProposalBackend,
    _admit_backend_scene_cameras,
    _artifact,
    _atomic_npz,
    _atomic_png,
    _canonical_hash,
    _compiled_grasp_geometry,
    _prepare_compiled_grasp_candidates,
    _render_grasp_global_review,
    _render_grasp_overlay,
    _scene_and_target_points_camera,
    _verify_artifact,
    _world_points_to_camera,
)
from .grasp_compiler import PANDA_GRIP_SITE_COMPILER_IDENTITY
from .grasp_selection import apply_candidate_selection


class _GraspBackendInput(NamedTuple):
    """The admitted proposer input cloud and the profile its identity binds."""

    admission: dict[str, Any]
    fused: dict[str, Any]
    contributing_cameras: list[str]
    multiview: bool
    profile: dict[str, Any]


class _GraspQueryPoints(NamedTuple):
    """Target points and the proposer scene in the target source camera frame."""

    target_camera: Any
    camera_to_world: Any
    backend_scene_world: Any
    backend_scene_camera: Any


class _GraspProposals(NamedTuple):
    """Compiled, selected candidates and what the backend reported beside them."""

    candidates: list[dict[str, Any]]
    compiled_candidates: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    response_service_identity: Any
    backend_output: Any
    input_regime: Any
    selection_record: Any


def _grasp_backend_record(backend: GraspProposalBackend) -> dict[str, Any]:
    return {
        "id": backend.backend_id,
        "version": backend.backend_version,
        "config_sha256": _canonical_hash(dict(backend.configuration())),
    }


def _registered_grasp_compiler(compiler_id: str | None) -> Mapping[str, Any]:
    compiler = GRASP_COMPILER_IDENTITIES.get(
        compiler_id or PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
    )
    if compiler is None:
        raise ValueError("grasp compiler is unregistered")
    return compiler


def _grasp_backend_input(
    *, observation_root: Path, observation: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]], target_source_camera: str,
    backend_scene_cameras: Sequence[str] | None,
) -> _GraspBackendInput:
    """Admit the proposer's cameras and describe the input its identity binds."""

    # Default: the historical single-view input, whose query identity stays
    # byte-identical.  An explicit request opts into the multi-view fusion the
    # review canvas has always computed and then thrown away.
    requested_backend_cameras = (
        (target_source_camera,) if backend_scene_cameras is None
        else tuple(str(name) for name in backend_scene_cameras)
    )
    admission = _admit_backend_scene_cameras(
        observation_root=observation_root,
        observation=observation,
        workspace_bounds_m=workspace_bounds_m,
        target_source_camera=target_source_camera,
        requested_cameras=requested_backend_cameras,
        voxel_size_m=GRASP_FUSION_VOXEL_SIZE_M,
        max_depth_m=GRASP_FUSION_MAX_DEPTH_M,
    )
    backend_input = admission["fused"]
    contributing_cameras = list(admission["contributing_cameras"])
    multiview_backend_input = len(contributing_cameras) > 1
    backend_input_profile = {
        "scope": (
            GRASP_BACKEND_FUSED_SCOPE if multiview_backend_input
            else GRASP_BACKEND_SINGLE_VIEW_SCOPE
        ),
        "camera": target_source_camera,
        "voxel_size_m": GRASP_FUSION_VOXEL_SIZE_M,
        "max_depth_m": GRASP_FUSION_MAX_DEPTH_M,
        "workspace_bounds_m": dict(workspace_bounds_m),
        "robot_self_filter": False,
        "multiview_fusion": multiview_backend_input,
    }
    if backend_scene_cameras is not None:
        # Only an opt-in request extends the identity, so frozen single-view
        # query refs keep their exact hashes.  Every camera whose RGB-D
        # actually determines the proposer input is bound by content here.
        backend_input_profile.update({
            "fusion_schema": GRASP_BACKEND_INPUT_FUSION_SCHEMA,
            "requested_cameras": list(admission["requested_cameras"]),
            "contributing_cameras": contributing_cameras,
            "excluded_cameras": [dict(item) for item in admission["excluded_cameras"]],
            "contributing_sources": [
                {
                    "camera": name,
                    "files": admission["source_files"][name],
                    "calibration_sha256": admission["calibrations"][name]["sha256"],
                }
                for name in contributing_cameras
            ],
            "deduplication": backend_input["deduplication"],
            "point_frame": "target_source_camera_opencv",
        })
    return _GraspBackendInput(
        admission, backend_input, contributing_cameras,
        multiview_backend_input, backend_input_profile,
    )


def _grasp_query_identity(
    *, observation_id: str, target_ref: str, identity: Mapping[str, Any],
    actual_source: Mapping[str, Any], backend_input_profile: dict[str, Any],
    workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
    backend_record: dict[str, Any], compiler: Mapping[str, Any],
    compile_for_execution: bool, execution_tool: str,
    target: Mapping[str, Any], public_contact_pose: bool,
    review_source_binding: Mapping[str, Any] | None,
    candidate_selection: Mapping[str, Any] | None, observed_extent: bool,
) -> dict[str, Any]:
    """The hashed identity whose digest names the query directory."""

    query_identity = {
        "observation_id": observation_id,
        "target_ref": target_ref,
        "camera": identity["camera"],
        "source": actual_source,
        "backend_input_profile": backend_input_profile,
        "workspace_bounds_m": dict(workspace_bounds_m),
        "max_candidates": max_candidates,
        "backend": backend_record,
        "compiler": dict(compiler),
        "compiler_mode": (
            "execution_plan" if compile_for_execution else "review_geometry_only"
        ),
        "execution_tool": execution_tool,
        "target_mask_sha256": target["private_mask"]["sha256"],
        "visualization_renderer": GRASP_PLAN_VISUALIZATION_ID,
        "input_pointcloud_schema": GRASP_INPUT_POINTCLOUD_SCHEMA,
    }
    if public_contact_pose:
        # Conditional so frozen pre-EXP08A query identities keep their exact
        # hashes; surfaces without the capability never set the flag.
        query_identity["public_contact_pose"] = True
    if review_source_binding is not None:
        # Correctness surfaces bind optional review availability and source
        # hashes, so a degraded view cannot reuse a previously fused artifact.
        query_identity["review_source_binding"] = dict(review_source_binding)
    if candidate_selection is not None:  # released query refs keep their hashes
        query_identity["candidate_selection"] = dict(candidate_selection["identity"])
    query_identity.update(scene_extent_review.identity(observed_extent))
    return query_identity


def _verified_grasp_query(
    query_root: Path, manifest_path: Path, query_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Reuse a materialized query after re-verifying every artifact it names."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != GRASP_QUERY_SCHEMA or manifest.get("identity") != query_identity:
        raise ValueError("grasp query identity drift")
    overlay = _verify_artifact(query_root, manifest["overlay"])
    _verify_artifact(query_root, manifest["input_pointcloud"]["artifact"])
    review = manifest.get("review_visualization")
    if isinstance(review, Mapping) and review.get("status") == "available":
        _verify_artifact(query_root, review["artifact"])
    review_cloud = manifest.get("review_pointcloud")
    if isinstance(review_cloud, Mapping):
        _verify_artifact(query_root, review_cloud["artifact"])
    backend_output_record = manifest.get("backend_output")
    if isinstance(backend_output_record, Mapping):
        _verify_artifact(query_root, backend_output_record["artifact"])
    return manifest, overlay


def _grasp_query_points(
    *, mask_path: Path, depth_path: Path, calibration_path: Path,
    backend_input: Mapping[str, Any],
) -> _GraspQueryPoints:
    _target_scene_camera, target_camera, camera_to_world = _scene_and_target_points_camera(
        mask_path=mask_path, depth_path=depth_path, calibration_path=calibration_path,
    )
    backend_scene_world = backend_input["xyz_world_m"]
    backend_scene_camera = _world_points_to_camera(
        backend_scene_world, camera_to_world,
    )
    return _GraspQueryPoints(
        target_camera, camera_to_world, backend_scene_world, backend_scene_camera,
    )


def _propose_grasp_candidates(
    *, backend: GraspProposalBackend, points: _GraspQueryPoints, target_ref: str,
    workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
    candidate_selection: Mapping[str, Any] | None, compiler: Mapping[str, Any],
    compile_for_execution: bool, public_contact_pose: bool, lift_m: float | None = None,
) -> _GraspProposals:
    """Call the backend, compile its candidates and apply the selection.

    ``lift_m`` is the closed grasp's lift above the contact on a surface that
    executes one; a candidate whose lift leaves the workspace is rejected here.
    """

    candidates, raw_diagnostics = backend.propose(
        scene_points_camera_m=points.backend_scene_camera,
        target_points_camera_m=points.target_camera,
        camera_to_world=points.camera_to_world,
        target_ref=target_ref,
        workspace_bounds_m=workspace_bounds_m,
        max_candidates=max_candidates if candidate_selection is None else candidate_selection["identity"]["pool_limit"],
        compiler_identity=compiler,
    )
    diagnostics = dict(raw_diagnostics)
    response_service_identity = diagnostics.pop("_service_identity", None)
    backend_output = diagnostics.pop("_canonical_backend_output", None)
    input_regime = diagnostics.pop("_backend_input_regime", None)
    candidates, compiled_candidates, compiler_rejected = (
        _prepare_compiled_grasp_candidates(
            candidates, compile_for_execution=compile_for_execution,
            compiler_identity=compiler,
            public_contact_pose=public_contact_pose,
        )
    )
    if lift_m is not None and compile_for_execution:
        candidates, compiled_candidates, unliftable = skill_geometry.split_liftable(
            candidates, compiled_candidates, lift_m, workspace_bounds_m)
        compiler_rejected = [*compiler_rejected, *unliftable]
    diagnostics["compiler_rejected"] = compiler_rejected
    candidates, compiled_candidates, selection_record = apply_candidate_selection(
        candidates, compiled_candidates, candidate_selection, max_candidates)
    diagnostics["compiler_returned_candidate_count"] = len(compiled_candidates)
    return _GraspProposals(
        candidates, compiled_candidates, diagnostics, response_service_identity,
        backend_output, input_regime, selection_record,
    )


def _write_grasp_input_pointcloud(
    query_root: Path, points: _GraspQueryPoints, target_world: Any,
) -> Path:
    input_pointcloud_path = query_root / "input-pointcloud.npz"
    _atomic_npz(
        input_pointcloud_path,
        backend_scene_points_camera_m=points.backend_scene_camera,
        backend_scene_points_world_m=points.backend_scene_world,
        target_points_camera_m=points.target_camera,
        target_camera_to_world=points.camera_to_world,
        target_points_world_m=target_world,
    )
    input_pointcloud_path.chmod(0o444)
    return input_pointcloud_path


def _write_grasp_overlay(
    *, query_root: Path, rgb_path: Path, depth_path: Path, mask_path: Path,
    calibration_path: Path, observation_id: str, target_ref: str,
    candidates: list[Mapping[str, Any]], compiler: Mapping[str, Any],
    execution_tool: str,
) -> tuple[Path, Any, dict[str, Any]]:
    """Render and write the policy-visible contact sheet."""

    overlay_path = query_root / "overlay.png"
    overlay, visualization = _render_grasp_overlay(
        rgb_path=rgb_path, depth_path=depth_path, mask_path=mask_path,
        calibration_path=calibration_path,
        observation_id=observation_id, target_ref=target_ref,
        candidates=candidates,
        compiler_identity=compiler,
        execution_tool=execution_tool,
    )
    _atomic_png(overlay_path, overlay)
    overlay_path.chmod(0o444)
    return overlay_path, overlay, visualization


def _write_grasp_global_review(
    *, query_root: Path, observation_root: Path, observation: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]], scene_cameras: Sequence[str],
    camera_to_world: Any, target_world: Any, overlay: Any,
    observation_id: str, target_ref: str, target_source_camera: str,
    candidates: list[Mapping[str, Any]], compiler: Mapping[str, Any],
    proposer_input: _GraspBackendInput, observed_extent: bool,
    review_policy_visible: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Best-effort fused review cloud and image; a failure is recorded, not raised."""

    review_visualization: dict[str, Any] = {
        "renderer": GRASP_PLAN_REVIEW_ID,
        "status": "unavailable",
        "policy_visible": False,
        "claim_boundary": "best_effort_review_only_not_required_for_grasp_plan",
    }
    review_pointcloud = None
    try:
        fused = fuse_rgbd_pointcloud_sources(
            observation_root=observation_root,
            observation=observation,
            workspace_bounds_m=workspace_bounds_m,
            camera_names=tuple(scene_cameras),
            voxel_size_m=GRASP_FUSION_VOXEL_SIZE_M,
            max_depth_m=GRASP_FUSION_MAX_DEPTH_M,
        )
        review_scene_world, review_pointcloud_path = _write_grasp_review_pointcloud(
            query_root, fused, camera_to_world,
        )
        review_visualization = _write_grasp_review_image(
            query_root=query_root, observation_root=observation_root,
            observation=observation, workspace_bounds_m=workspace_bounds_m,
            scene_cameras=scene_cameras, target_world=target_world,
            overlay=overlay, fused=fused, review_scene_world=review_scene_world,
            observation_id=observation_id, target_ref=target_ref,
            target_source_camera=target_source_camera, candidates=candidates,
            compiler=compiler, proposer_input=proposer_input,
            observed_extent=observed_extent,
            review_policy_visible=review_policy_visible,
        )
        review_pointcloud = _grasp_review_pointcloud_record(
            fused, scene_cameras, review_pointcloud_path, query_root,
        )
    except Exception as exc:  # Review evidence must never block policy planning.
        review_visualization["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:512],
        }
    return review_visualization, review_pointcloud


def _write_grasp_review_pointcloud(
    query_root: Path, fused: Mapping[str, Any], camera_to_world: Any,
) -> tuple[Any, Path]:
    review_scene_world = fused["xyz_world_m"]
    review_scene_camera = _world_points_to_camera(
        review_scene_world, camera_to_world,
    )
    review_pointcloud_path = query_root / "review-pointcloud.npz"
    _atomic_npz(
        review_pointcloud_path,
        review_scene_points_camera_m=review_scene_camera,
        review_scene_points_world_m=review_scene_world,
        review_scene_rgb_u8=fused["rgb_u8"],
        review_scene_source_index=fused["source_index"],
    )
    review_pointcloud_path.chmod(0o444)
    return review_scene_world, review_pointcloud_path


def _write_grasp_review_image(
    *, query_root: Path, observation_root: Path, observation: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]], scene_cameras: Sequence[str],
    target_world: Any, overlay: Any, fused: Mapping[str, Any],
    review_scene_world: Any, observation_id: str, target_ref: str,
    target_source_camera: str, candidates: list[Mapping[str, Any]],
    compiler: Mapping[str, Any], proposer_input: _GraspBackendInput,
    observed_extent: bool, review_policy_visible: bool,
) -> dict[str, Any]:
    """Render and write the global review; return its available record."""

    review_path = query_root / "global-review.png"
    extent = scene_extent_review.cell(
        observation_root=observation_root, observation=observation,
        camera_names=tuple(scene_cameras), max_depth_m=GRASP_FUSION_MAX_DEPTH_M,
        workspace_bounds_m=workspace_bounds_m, target_world=target_world,
        grip_sites=[(item["rank"], colour, _compiled_grasp_geometry(
            item, compiler_identity=compiler)["grip_site"]) for item, colour in zip(
            candidates, GRASP_CANDIDATE_COLOURS, strict=False)],
    ) if observed_extent else None
    review_image, available_review = _render_grasp_global_review(
        camera_overlay=overlay,
        camera_panel_resolution=[overlay.width, overlay.height],
        scene_world=review_scene_world,
        target_world=target_world,
        scene_source_index=fused["source_index"],
        scene_sources=fused["sources"],
        workspace_bounds_m=workspace_bounds_m,
        observation_id=observation_id,
        target_ref=target_ref,
        camera=target_source_camera,
        candidates=candidates,
        compiler_identity=compiler,
        backend_input_scope=proposer_input.profile["scope"],
        backend_cameras=proposer_input.contributing_cameras,
        backend_points_by_camera={
            str(source["camera"]): int(source.get("retained_voxel_points", 0))
            for source in proposer_input.fused["sources"]
        },
        backend_excluded_cameras=[
            dict(item) for item in proposer_input.admission["excluded_cameras"]
        ],
        observed_extent=extent, policy_visible=review_policy_visible,
    )
    _atomic_png(review_path, review_image)
    review_path.chmod(0o444)
    available_review["status"] = "available"
    available_review["artifact"] = _artifact(review_path, query_root)
    return available_review


def _grasp_review_pointcloud_record(
    fused: Mapping[str, Any], scene_cameras: Sequence[str],
    review_pointcloud_path: Path, query_root: Path,
) -> dict[str, Any]:
    return {
        "schema": "ora.grasp-review-pointcloud.v1",
        "sources": fused["sources"],
        "camera_order": list(scene_cameras),
        "voxel_size_m": GRASP_FUSION_VOXEL_SIZE_M,
        "max_depth_m": GRASP_FUSION_MAX_DEPTH_M,
        "input_point_count": fused["input_point_count"],
        "workspace_point_count": fused["workspace_point_count"],
        "filtered_voxel_count": fused["filtered_voxel_count"],
        "artifact": _artifact(review_pointcloud_path, query_root),
        "policy_visible": False,
    }


def _write_grasp_backend_output(
    query_root: Path, backend_output: Mapping[str, Any] | None, input_regime: Any,
) -> dict[str, Any] | None:
    """Write the canonical Contact-GraspNet arrays when the backend returned them."""

    backend_output_record = None
    if backend_output is not None:
        backend_output_path = query_root / "backend-output.npz"
        _atomic_npz(backend_output_path, **backend_output)
        backend_output_path.chmod(0o444)
        backend_output_record = {
            "schema": "ora.contact-graspnet-backend-output.v1",
            "arrays": {
                name: {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                }
                for name, value in backend_output.items()
            },
            "artifact": _artifact(backend_output_path, query_root),
            "input_regime": input_regime,
            "policy_visible": False,
        }
    return backend_output_record


def _grasp_input_pointcloud_record(
    *, query_root: Path, input_pointcloud_path: Path, identity: Mapping[str, Any],
    target_source_camera: str, proposer_input: _GraspBackendInput,
    points: _GraspQueryPoints, target_world: Any,
) -> dict[str, Any]:
    """The manifest record of the proposer input cloud and its fusion."""

    admission = proposer_input.admission
    backend_input = proposer_input.fused
    contributing_cameras = proposer_input.contributing_cameras
    target_camera, camera_to_world = points.target_camera, points.camera_to_world
    backend_scene_camera = points.backend_scene_camera
    backend_scene_world = points.backend_scene_world
    return {
        "schema": GRASP_INPUT_POINTCLOUD_SCHEMA,
        "input_scope": (
            GRASP_BACKEND_FUSED_INPUT_SCOPE if proposer_input.multiview
            else GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE
        ),
        "target_source_camera": identity["camera"],
        "backend_input_profile": proposer_input.profile,
        "backend_source": backend_input["sources"][0],
        "backend_input_sources": [
            dict(source) for source in backend_input["sources"]
        ],
        "backend_input_fusion": {
            "schema": GRASP_BACKEND_INPUT_FUSION_SCHEMA,
            "requested_cameras": list(admission["requested_cameras"]),
            "contributing_cameras": contributing_cameras,
            "excluded_cameras": [
                dict(item) for item in admission["excluded_cameras"]
            ],
            "exclusion_vocabulary": list(GRASP_BACKEND_EXCLUSION_REASONS),
            "minimum_camera_points": GRASP_BACKEND_MINIMUM_CAMERA_POINTS,
            "deduplication": backend_input["deduplication"],
            # The fused world cloud is expressed in the target camera's
            # OpenCV frame before it reaches the proposer; both directions
            # of that transform are recorded so a replay can undo it.
            "backend_point_frame": {
                "camera": target_source_camera,
                "convention": "opencv_camera_optical",
                "world_to_backend_camera": admission["calibrations"][
                    target_source_camera
                ]["extrinsic_cv_world_to_camera"],
                "backend_camera_to_world": [
                    [float(value) for value in row]
                    for row in camera_to_world.tolist()
                ],
            },
            "cameras": [
                {
                    "camera": str(source["camera"]),
                    "source_index": int(source["source_index"]),
                    "source_resolution": list(source["source_resolution"]),
                    "valid_depth_points": int(source["valid_depth_points"]),
                    "workspace_points": int(source["workspace_points"]),
                    "retained_voxel_points": int(source["retained_voxel_points"]),
                    "calibration": admission["calibrations"][
                        str(source["camera"])
                    ],
                }
                for source in backend_input["sources"]
            ],
        },
        "input_point_count": backend_input["input_point_count"],
        "workspace_point_count": backend_input["workspace_point_count"],
        "filtered_voxel_count": backend_input["filtered_voxel_count"],
        "arrays": {
            "backend_scene_points_camera_m": {
                "shape": list(backend_scene_camera.shape),
                "dtype": str(backend_scene_camera.dtype),
            },
            "backend_scene_points_world_m": {
                "shape": list(backend_scene_world.shape),
                "dtype": str(backend_scene_world.dtype),
            },
            "target_points_camera_m": {
                "shape": list(target_camera.shape), "dtype": str(target_camera.dtype),
            },
            "target_camera_to_world": {
                "shape": list(camera_to_world.shape), "dtype": str(camera_to_world.dtype),
            },
            "target_points_world_m": {
                "shape": list(target_world.shape), "dtype": str(target_world.dtype),
            },
        },
        "artifact": _artifact(input_pointcloud_path, query_root),
    }


def _grasp_query_manifest(
    *, query_ref: str, query_identity: dict[str, Any],
    backend_record: dict[str, Any], proposals: _GraspProposals,
    backend_output_record: dict[str, Any] | None,
    visualization: dict[str, Any], review_visualization: dict[str, Any],
    review_pointcloud: dict[str, Any] | None,
    input_pointcloud: dict[str, Any], overlay: dict[str, Any],
) -> dict[str, Any]:
    candidates = proposals.candidates
    selection_record = proposals.selection_record
    return {
        "schema": GRASP_QUERY_SCHEMA,
        "query_ref": query_ref,
        "identity": query_identity,
        "backend": backend_record,
        "status": "candidates" if candidates else "no_candidate",
        "candidates": candidates,
        "compiled_candidates": proposals.compiled_candidates,
        "diagnostics": dict(proposals.diagnostics),
        "response_service_identity": proposals.response_service_identity,
        "backend_output": backend_output_record,
        "visualization": visualization,
        "review_visualization": review_visualization,
        "review_pointcloud": review_pointcloud,
        "input_pointcloud": input_pointcloud,
        "overlay": overlay,
        **({"candidate_selection": selection_record} if selection_record is not None else {}),
    }
