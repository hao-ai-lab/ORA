"""Observation-bound segmentation and read-only grasp-proposal helpers.

The experimental runtime uses resident, loopback-only SAM3 and
Contact-GraspNet services behind strict model-identity and array contracts.
Dependency-light deterministic implementations remain available only as test
doubles while sharing the same policy-visible artifact boundary.
"""

from __future__ import annotations

from ora.agent.tools.grasp_compilation import (
    _rotation_matrix_to_quaternion_xyzw, _public_grasp_compiler_identity,
)

import base64
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .arx_x5_gripper import ARX_X5_LINK6_COMPILER_IDENTITIES
from .x5_topdown import X5_TOPDOWN_COMPILER_IDENTITIES, topdown_contact_height
from .geometry import fuse_rgbd_pointcloud_sources
from . import artifact_errors as _errors, overhead_view, target_support
from .grasp_diagnostics import hybrid_grasp_diagnostics
from .grasp_selection import apply_candidate_selection  # re-exported; grasp_query.py calls it
from .grounding_palette import (
    GROUNDING_PREVIEW_RENDER_LIMIT,
    GROUNDING_CANDIDATE_COLOURS,
    GROUNDING_MAX_CANDIDATES,
    grounding_candidate_colour,
)
from . import grounding_preview, scene_extent_review
from .grounding_preview import (  # the candidate card moved out with its depth strip
    GROUNDING_VISUALIZATION_ID, render_candidate_preview as _render_grounding_candidate_preview,
)
from .grasp_overlay import (  # contact-sheet drawing steps of _render_grasp_overlay
    _draw_candidate_card, _draw_contact_sheet_legend, _draw_target_panel,
    _gripper_wireframe, _new_contact_sheet, _overlay_camera_matrices,
    _overlay_fonts, _overlay_target, _project_world, _tinted_target_source,
)
from .grasp_compiler import (
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    compile_grasp_candidate as _compile_grasp_candidate_impl,
    compiled_grasp_geometry as _compiled_grasp_geometry_impl,
    prepare_compiled_grasp_candidates as _prepare_compiled_grasp_candidates_impl,
)


TARGET_SCHEMA = "ora.segmented-target.v2"
TARGET_INDEX_SCHEMA = "ora.segmented-target-index.v1"
GRASP_QUERY_SCHEMA = "ora.grasp-query.v7"
GROUNDING_QUERY_SCHEMA = "ora.grounding-query.v2"
TARGET_REF_PATTERN = re.compile(r"target-[0-9a-f]{24}")
LOCALIZATION_METHOD = "mask_depth_primary_component_robust_pca_v1"
DEPTH_INLIER_GEOMETRY_CAPABILITY = "public_target_geometry_depth_inliers_v2"
DEPTH_INLIER_SELECTION_METHOD = "valid_depth_primary_component_mad_v1"
TARGET_GEOMETRY_QUALITY_MINIMUMS = {
    "depth_valid_fraction": 0.7,
    "depth_inlier_fraction": 0.7,
    "primary_component_fraction": 0.6,
}
SAM3_MODEL_IDENTITY = {
    "family": "sam3",
    "source_commit": "6fe87d64a5beb9084923d7a9e002741178635b09",
    "checkpoint_sha256": (
        "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
    ),
}
CONTACT_GRASPNET_MODEL_IDENTITY = {
    "family": "contact_graspnet_pytorch",
    "source_commit": "da3dcfb2f53e43b186083ee4a9d1e232f73efc98",
    "checkpoint_sha256": (
        "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
    ),
    "config_sha256": (
        "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5"
    ),
}
# The jaw geometry these identities are built from lives beside its source
# binding, because the published result schema is pinned to the same number.
from .aloha_gripper_geometry import (  # noqa: E402
    ALOHA_GRIPPER_GEOMETRY,
    ALOHA_NOMINAL_MAXIMUM_OPENING_M,
    ALOHA_SOURCE_BINDING,
)

ALOHA_END_EFFECTOR_COMPILER_IDENTITIES = {
    arm: {
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_end_effector_v1",
        "embodiment_descriptor_id": "aloha_agilex",
        "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "controlled_point": "end_effector",
        "controlled_point_semantics": "RoboTwin_planner_pose",
        "contact_graspnet_gripper_depth_m": 0.1034,
        "source_origin_to_controlled_point_m": 0.0,
        "controlled_point_to_contact_baseline_m": 0.1034,
        "maximum_opening_m": ALOHA_NOMINAL_MAXIMUM_OPENING_M,
        "axis_mapping": {
            "model_approach_z": "end_effector_local_positive_x",
            "model_jaw_x": "end_effector_local_positive_y",
        },
        "arm": arm,
        "source_binding": ALOHA_SOURCE_BINDING,
    }
    for arm in ("left", "right")
}
ALOHA_GRIP_CENTER_COMPILER_IDENTITIES = {
    arm: {
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v2",
        "embodiment_descriptor_id": "aloha_agilex",
        "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "controlled_point": "end_effector",
        "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
        "contact_graspnet_gripper_depth_m": 0.1034,
        "source_origin_to_controlled_point_m": 0.12,
        "controlled_point_to_contact_baseline_m": -0.0166,
        "maximum_opening_m": ALOHA_NOMINAL_MAXIMUM_OPENING_M,
        "axis_mapping": {
            "model_approach_z": "grip_center_local_positive_x",
            "model_jaw_x": "grip_center_local_positive_y",
        },
        "arm": arm,
        "source_binding": ALOHA_SOURCE_BINDING,
    }
    for arm in ("left", "right")
}
ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES = {
    arm: {
        **identity,
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v3",
        # Contact-GraspNet reports the origin of its Panda gripper model.
        # Its own source-bound finger/contact depth is 103.4 mm.  The 120 mm
        # ALOHA link6 -> public-TCP transform is applied later by the RoboTwin
        # motion adapter and must not be reused here.
        "source_origin_to_controlled_point_m": 0.1034,
        "controlled_point_to_contact_baseline_m": 0.0,
        "maximum_approach_m": 0.10,
        # A deterministic public-RGB-D candidate family complements learned
        # oblique proposals.  The TCP target is a small qualified clearance
        # above the robust visible top surface; pre-grasp remains 100 mm back
        # along world-down approach.  These values are ALOHA compiler facts,
        # not task/object GT.
        "public_rgbd_topdown_profile": {
            "surface_percentile": 95.0,
            "contact_clearance_m": 0.008,
            "pregrasp_standoff_m": 0.10,
            "maximum_candidates": 2,
        },
    }
    for arm, identity in ALOHA_GRIP_CENTER_COMPILER_IDENTITIES.items()
}
GRASP_COMPILER_IDENTITIES = {
    PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]: PANDA_GRIP_SITE_COMPILER_IDENTITY,
    **{
        identity["id"]: identity
        for identity in ALOHA_END_EFFECTOR_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ALOHA_GRIP_CENTER_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ARX_X5_LINK6_COMPILER_IDENTITIES.values()
    },
    **{identity["id"]: identity for identity in X5_TOPDOWN_COMPILER_IDENTITIES.values()},
}
GRASP_PLAN_VISUALIZATION_ID = "grasp-plan-target-local-contact-sheet.v2"
GRASP_PLAN_REVIEW_ID = "grasp-plan-global-pointcloud-review.v3"
GRASP_INPUT_POINTCLOUD_SCHEMA = "ora.grasp-input-pointcloud.v4"
GRASP_SCENE_CAMERAS = ("agentview", "wrist")
#: One colour per drawable grasp candidate, shared by the camera overlay and
#: the global review so the two renderings of one candidate list cannot
#: disagree. The number of colours is the number of candidates that can be
#: attributed, so extending this table is what raises that bound.
GRASP_CANDIDATE_COLOURS = ((255, 82, 82), (80, 210, 255), (255, 205, 64), (96, 255, 96))
GRASP_MAX_DRAWABLE_CANDIDATES = len(GRASP_CANDIDATE_COLOURS)
GRASP_FUSION_VOXEL_SIZE_M = 0.005
GRASP_FUSION_MAX_DEPTH_M = 2.5
GRASP_BACKEND_INPUT_FUSION_SCHEMA = "ora.grasp-backend-input-fusion.v1"
GRASP_BACKEND_SINGLE_VIEW_SCOPE = "target_source_camera_rgbd_only"
GRASP_BACKEND_FUSED_SCOPE = "multiview_fused_rgbd"
GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE = "target_source_camera_for_backend_only"
GRASP_BACKEND_FUSED_INPUT_SCOPE = "multiview_fused_rgbd_for_backend"
# One camera must place at least this many deduplicated points inside the
# workspace before it is allowed to alter the proposer's input geometry.
GRASP_BACKEND_MINIMUM_CAMERA_POINTS = 16
# Closed vocabulary: a camera that cannot enter the fused proposer input is
# always recorded with exactly one of these reasons, never silently dropped.
GRASP_BACKEND_EXCLUSION_REASONS = (
    "camera_absent_from_observation",
    "rgbd_source_reference_missing",
    "rgbd_source_file_unavailable",
    "rgbd_source_hash_drift",
    "rgbd_calibration_invalid",
    "rgbd_depth_unusable",
    "insufficient_workspace_depth_points",
    "duplicate_camera_request",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_png(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _file_hash(path),
        "bytes": path.stat().st_size,
    }


def _verify_artifact(root: Path, reference: Mapping[str, Any]) -> Path:
    relative = reference.get("path")
    if not isinstance(relative, str):
        raise _errors.ArtifactIntegrityError("artifact reference lacks a path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise _errors.ArtifactIntegrityError("artifact is unavailable")
    if _file_hash(path) != reference.get("sha256"):
        raise _errors.ArtifactHashMismatch("artifact hash drift")
    return path


def _safe_path_component(value: str, *, field_name: str) -> str:
    if (
        not value or value in {".", ".."}
        or Path(value).name != value
        or "/" in value or "\\" in value
    ):
        raise ValueError(f"{field_name} is not a safe artifact path component")
    return value


def _segmentation_source_identity(
    *, observation_id: str, camera: str, rgb_path: Path,
    depth_path: Path, calibration_path: Path,
) -> dict[str, Any]:
    observation_id = _safe_path_component(
        observation_id, field_name="observation_id",
    )
    camera = _safe_path_component(camera, field_name="camera")
    source_root = rgb_path.resolve().parent
    expected = {
        "rgb": (rgb_path.resolve(), "rgb.png"),
        "depth": (depth_path.resolve(), "depth-m.npy"),
        "calibration": (calibration_path.resolve(), "calibration.json"),
    }
    files: dict[str, Any] = {}
    for role, (path, expected_name) in expected.items():
        if path.parent != source_root or path.name != expected_name or not path.is_file():
            raise ValueError(
                "segmentation source files must use rgb.png, depth-m.npy and "
                "calibration.json in one camera directory"
            )
        files[role] = {
            "name": expected_name,
            "sha256": _file_hash(path),
            "bytes": path.stat().st_size,
        }
    return {
        "policy_observation_path": (
            Path("policy-observations") / observation_id / camera
        ).as_posix(),
        "files": files,
    }


def _segmented_target_root(
    *, artifact_root: Path, observation_id: str, camera: str, target_ref: str,
) -> Path:
    return (
        artifact_root / "policy-observations" / observation_id / camera
        / "segmentation-queries" / target_ref
    )


def _target_index_path(artifact_root: Path, target_ref: str) -> Path:
    return artifact_root / "segmented-target-index" / f"{target_ref}.json"


def _materialize_target_index(
    *, artifact_root: Path, target_root: Path, manifest_path: Path,
    observation_id: str, camera: str, target_ref: str,
) -> None:
    index_path = _target_index_path(artifact_root, target_ref)
    record = {
        "schema": TARGET_INDEX_SCHEMA,
        "target_ref": target_ref,
        "observation_id": observation_id,
        "camera": camera,
        "artifact_path": target_root.resolve().relative_to(
            artifact_root.resolve(),
        ).as_posix(),
        "manifest_sha256": _file_hash(manifest_path),
    }
    if index_path.is_file():
        if json.loads(index_path.read_text(encoding="utf-8")) != record:
            raise _errors.ArtifactHashMismatch("segmented target index drift")
        return
    _atomic_json(index_path, record)
    index_path.chmod(0o444)


def _resolve_segmented_target_root(
    *, artifact_root: Path, target_ref: str,
) -> Path:
    index_path = _target_index_path(artifact_root, target_ref)
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("schema") != TARGET_INDEX_SCHEMA
            or index.get("target_ref") != target_ref
        ):
            raise _errors.ArtifactIntegrityError("segmented target index is invalid")
        relative = index.get("artifact_path")
        if not isinstance(relative, str):
            raise _errors.ArtifactIntegrityError("segmented target index lacks an artifact path")
        target_root = (artifact_root / relative).resolve()
        if not target_root.is_relative_to(artifact_root.resolve()):
            raise _errors.ArtifactIntegrityError("segmented target index escapes the artifact root")
        manifest_path = target_root / "manifest.json"
        if (
            not manifest_path.is_file()
            or _file_hash(manifest_path) != index.get("manifest_sha256")
        ):
            raise _errors.ArtifactHashMismatch("segmented target index manifest hash drift")
        return target_root

    # Read-only compatibility for pre-normalization qualification artifacts.
    legacy_root = artifact_root / "segmented-targets" / target_ref
    if legacy_root.joinpath("manifest.json").is_file():
        return legacy_root
    raise _errors.TargetRefUnknown("target_ref is unavailable: no segmented target in this run carries it")


class PointSegmentationBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]: ...


class BatchPointSegmentationBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def segment_points(
        self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


class TextGroundingBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def ground(
        self, *, rgb_u8: Any, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


class GraspProposalBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


def _loopback_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint.rstrip("/"))
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model-service endpoint must be an unauthenticated loopback URL")
    return endpoint.rstrip("/")


def _encode_npy(value: Any) -> str:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_npy(value: Any) -> Any:
    import numpy as np

    if not isinstance(value, str):
        raise ValueError("model service returned a non-string array payload")
    try:
        raw = base64.b64decode(value, validate=True)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ValueError("model service returned an invalid array payload") from exc


def _verify_model_identity(
    received: Any, expected: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(received, Mapping):
        raise _errors.service_identity_drift("model service omitted model identity", service="model")
    for key, value in expected.items():
        if received.get(key) != value:
            raise _errors.service_identity_drift(f"model service {key} differs from the qualified identity", service="model")
    return dict(expected)


def _require_ready_service(
    *, client: Any, endpoint: str, service: str,
    expected_model: Mapping[str, str],
) -> dict[str, Any]:
    import httpx

    try:
        response = client.get(f"{endpoint}/healthz")
        response.raise_for_status()
        health = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise _errors.http_failure(exc, service=service, what="service readiness") from exc
    if not isinstance(health, Mapping):
        raise _errors.service_identity_drift(f"{service} service returned malformed readiness", service=service)
    if (
        health.get("schema") != "ora.model-service.health.v1"
        or health.get("ready") is not True
        or health.get("service") != service
        or health.get("warmup_complete") is not True
        or health.get("load_count") != 1
    ):
        raise _errors.service_identity_drift(f"{service} service is not warm and singly loaded", service=service)
    _verify_model_identity(health.get("model"), expected_model)
    if not isinstance(health.get("instance_id"), str) or not health["instance_id"]:
        raise _errors.service_identity_drift(f"{service} service omitted its model instance identity", service=service)
    if not isinstance(health.get("adapter_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", health["adapter_sha256"],
    ):
        raise _errors.service_identity_drift(f"{service} service omitted its adapter hash", service=service)
    started = health.get("started_unix_s")
    if not isinstance(started, (int, float)) or not math.isfinite(float(started)):
        raise _errors.service_identity_drift(f"{service} service omitted its start time", service=service)
    return dict(health)


def _normalize_expected_service_binding(
    binding: Mapping[str, Any], *, schema: str, endpoint: str,
) -> dict[str, Any]:
    value = dict(binding)
    required = {
        "schema", "tool_surface", "endpoint", "instance_id", "pid",
        "started_unix_s", "adapter_sha256", "model", "load_count",
        "warmup_complete", "qualification",
    }
    qualification = value.get("qualification")
    if (
        set(value) != required
        or value.get("schema") != schema
        or value.get("endpoint") != endpoint
        or not isinstance(value.get("tool_surface"), str)
        or not value["tool_surface"]
        or not isinstance(value.get("instance_id"), str)
        or not value["instance_id"]
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("started_unix_s"), bool)
        or not isinstance(value.get("started_unix_s"), (int, float))
        or not math.isfinite(float(value["started_unix_s"]))
        or not isinstance(value.get("adapter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["adapter_sha256"]) is None
        or not isinstance(value.get("model"), Mapping)
        or value.get("load_count") != 1
        or value.get("warmup_complete") is not True
        or not isinstance(qualification, Mapping)
        or qualification.get("status") != "passed"
        or not isinstance(qualification.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", qualification["sha256"]) is None
        or isinstance(qualification.get("created_unix_s"), bool)
        or not isinstance(qualification.get("created_unix_s"), (int, float))
        or not math.isfinite(float(qualification["created_unix_s"]))
    ):
        raise ValueError("expected model-service binding is invalid")
    return value


def _require_bound_service_identity(
    *, health: Mapping[str, Any], binding: Mapping[str, Any], service: str,
) -> None:
    for field in (
        "instance_id", "started_unix_s", "adapter_sha256", "model",
        "load_count", "warmup_complete", "pid",
    ):
        if health.get(field) != binding.get(field):
            raise _errors.service_identity_drift(
                f"{service} runtime service differs from bound {field}", service=service,
            )


def _require_response_service_identity(
    *, result: Mapping[str, Any], binding: Mapping[str, Any] | None,
    service: str,
) -> dict[str, Any]:
    identity = result.get("service_identity")
    required = {
        "schema", "instance_id", "pid", "started_unix_s", "adapter_sha256",
    }
    if (
        not isinstance(identity, Mapping)
        or set(identity) != required
        or identity.get("schema")
        != "ora.model-service-response-identity.v1"
        or not isinstance(identity.get("instance_id"), str)
        or not identity["instance_id"]
        or not isinstance(identity.get("pid"), int)
        or isinstance(identity.get("started_unix_s"), bool)
        or not isinstance(identity.get("started_unix_s"), (int, float))
        or not math.isfinite(float(identity["started_unix_s"]))
        or not isinstance(identity.get("adapter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["adapter_sha256"]) is None
    ):
        raise _errors.service_identity_drift(f"{service} response omitted its service instance identity", service=service)
    if binding is not None:
        for field in ("instance_id", "pid", "started_unix_s", "adapter_sha256"):
            if identity.get(field) != binding.get(field):
                raise _errors.service_identity_drift(
                    f"{service} response differs from bound {field}", service=service,
                )
    return dict(identity)


#: What the resident SAM3 text-grounding service accepts, which is not what the
#: overlay palette can draw. ``infer_ground`` refuses anything outside [1,3] with
#: a 422, and the palette's 32 was being used as the request bound: eleven calls
#: in one cohort asked for 4, 6 or 8 and every one came back as an HTTP status
#: the policy could not act on, while all 291 calls at three or fewer succeeded.
#: The two limits are different questions and this is the one the wire answers.
GROUNDING_REQUEST_MAX_CANDIDATES = 3


@dataclass(frozen=True)
class Sam3ApiSegmenter:
    """Point-prompt SAM3 client for one already-warm loopback service."""

    endpoint: str = "http://127.0.0.1:8114"
    timeout_s: float = 30.0
    backend_id: str = "sam3_point_api"
    backend_version: str = "v1"
    expected_binding: Mapping[str, Any] | None = None
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import httpx

        object.__setattr__(self, "endpoint", _loopback_endpoint(self.endpoint))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("SAM3 service timeout must be positive")
        if self.expected_binding is not None:
            object.__setattr__(
                self, "expected_binding",
                _normalize_expected_service_binding(
                    self.expected_binding,
                    schema="ora.segmentation-service-binding.v1",
                    endpoint=self.endpoint,
                ),
            )
        object.__setattr__(
            self, "_client", httpx.Client(timeout=self.timeout_s),
        )

    def configuration(self) -> Mapping[str, Any]:
        return {
            "endpoint": self.endpoint,
            "timeout_s": self.timeout_s,
            "expected_model": dict(SAM3_MODEL_IDENTITY),
            "api_schema": "ora.model-service.segment-point.v1",
            "expected_binding_sha256": (
                None if self.expected_binding is None
                else _canonical_hash(self.expected_binding)
            ),
        }

    def close(self) -> None:
        self._client.close()

    def require_ready(self) -> dict[str, Any]:
        health = _require_ready_service(
            client=self._client, endpoint=self.endpoint,
            service="segmentation", expected_model=SAM3_MODEL_IDENTITY,
        )
        if self.expected_binding is not None:
            _require_bound_service_identity(
                health=health, binding=self.expected_binding, service="SAM3",
            )
        return health

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]:
        import cv2
        import httpx
        import numpy as np
        from PIL import Image

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        depth = np.asarray(depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("SAM3 requires aligned HxWx3 RGB and HxW depth")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {
            "schema": "ora.model-service.segment-point-request.v1",
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "point_uv": list(point_uv),
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/segment-point", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _errors.http_failure(exc, service="SAM3", what="service request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.segment-point.v1":
            raise _errors.service_identity_drift("SAM3 service response schema mismatch", service="SAM3")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding, service="SAM3",
        )
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        mask = _decode_npy(result.get("mask_npy_base64")).astype(bool)
        if mask.shape != depth.shape:
            raise ValueError("SAM3 service mask shape differs from the source image")
        u, v = point_uv
        if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
            raise ValueError("SAM3 positive prompt is outside the source image")
        if not bool(mask[v, u]):
            raise ValueError("SAM3 service mask does not contain the positive prompt")
        component_count, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8,
        )
        prompt_label = int(labels[v, u])
        if component_count <= 1 or prompt_label == 0:
            raise ValueError("SAM3 service mask has no prompted foreground component")
        raw_pixel_count = int(np.count_nonzero(mask))
        raw_component_count = component_count - 1
        mask = labels == prompt_label
        prompt_component_pixel_count = int(np.count_nonzero(mask))
        service_metrics = result.get("metrics")
        if not isinstance(service_metrics, Mapping):
            raise ValueError("SAM3 service response lacks timing metrics")
        return mask, {
            "model": model,
            "_service_identity": service_identity,
            "model_score": float(result["score"]),
            "request_id": str(result["request_id"]),
            "mask_normalization": {
                "method": "positive_prompt_8_connected_component_v1",
                "raw_component_count": raw_component_count,
                "raw_pixel_count": raw_pixel_count,
                "prompt_component_pixel_count": prompt_component_pixel_count,
                "retained_fraction": (
                    prompt_component_pixel_count / raw_pixel_count
                ),
            },
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(service_metrics["queue_ms"]),
                "service_inference": float(service_metrics["inference_ms"]),
                "service_total": float(service_metrics["total_ms"]),
            },
        }

    def segment_points(
        self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        """Fetch independent alternatives; host-side selection owns the total K."""
        import httpx
        import numpy as np
        from PIL import Image
        from .segmentation_batch import validate_points

        rgb, depth = np.asarray(rgb_u8, dtype=np.uint8), np.asarray(depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("SAM3 requires aligned HxWx3 RGB and HxW depth")
        validate_points(points_uv, width=rgb.shape[1], height=rgb.shape[0])
        if self.expected_binding is not None:
            health = self.require_ready()
            if "segment_points_v1" not in health.get("capabilities", []):
                raise _errors.service_identity_drift("bound SAM3 service does not support batch point queries", service="SAM3")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {"schema": "ora.model-service.segment-points-request.v1",
                   "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
                   "points_uv": points_uv}
        started = time.perf_counter()
        try:
            response = self._client.post(f"{self.endpoint}/v1/segment-points", json=payload)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _errors.http_failure(exc, service="SAM3", what="batch request") from exc
        if result.get("schema") != "ora.model-service.segment-points.v1":
            raise _errors.service_identity_drift("SAM3 batch response schema mismatch", service="SAM3")
        identity = _require_response_service_identity(result=result, binding=self.expected_binding, service="SAM3")
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        queries = result.get("queries")
        if not isinstance(queries, list) or len(queries) != len(points_uv):
            raise ValueError("SAM3 batch response lost query identities")
        raw_candidates, query_results = [], []
        for index, (query, point) in enumerate(zip(queries, points_uv)):
            if not isinstance(query, Mapping) or query.get("query_index") != index or query.get("point_uv") != point:
                raise ValueError("SAM3 batch response query identity drift")
            candidates, error = query.get("candidates"), query.get("error")
            if (not isinstance(candidates, list) or len(candidates) > 3
                    or error is not None and (not isinstance(error, Mapping) or candidates)):
                raise ValueError("SAM3 batch response has malformed candidates or failure")
            seen = set()
            for candidate in candidates:
                source_index, score = candidate.get("source_index"), candidate.get("score")
                if (type(source_index) is not int or source_index < 0 or source_index in seen
                        or type(score) not in (int, float) or not math.isfinite(score)):
                    raise ValueError("SAM3 batch candidate ordering/score is invalid")
                seen.add(source_index)
                mask = _decode_npy(candidate.get("mask_npy_base64"))
                if mask.shape != depth.shape or not np.isfinite(mask).all():
                    raise ValueError("SAM3 batch mask differs from the source image")
                raw_candidates.append({"query_index": index, "source_index": source_index,
                    "point_uv": point, "mask": mask.astype(bool), "backend_score": float(score)})
            query_results.append({"query_index": index, "point_uv": point,
                                  "raw_candidate_count": len(candidates),
                                  "error_code": "point_inference_failed" if error else None})
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping) or result.get("image_encodings") != 1:
            raise ValueError("SAM3 batch response lacks timing/encoding evidence")
        return raw_candidates, {"model": model, "_service_identity": identity,
            "request_id": str(result["request_id"]), "query_results": query_results,
            "image_encodings": 1,
            "latency_ms": {"roundtrip": (time.perf_counter() - started) * 1000,
                "service_queue": float(metrics["queue_ms"]),
                "service_inference": float(metrics["inference_ms"]),
                "service_total": float(metrics["total_ms"])}}

    def ground(
        self, *, rgb_u8: Any, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        import httpx
        import numpy as np
        from PIL import Image

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM3 text grounding requires one HxWx3 RGB image")
        prompt = text.strip() if isinstance(text, str) else ""
        if not 1 <= len(prompt) <= 120:
            raise ValueError("SAM3 text prompt must contain 1..120 characters")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= GROUNDING_REQUEST_MAX_CANDIDATES
        ):
            raise ValueError(
                "SAM3 text grounding max_candidates must be in "
                f"[1,{GROUNDING_REQUEST_MAX_CANDIDATES}]"
            )
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {
            "schema": "ora.model-service.ground-text-request.v1",
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "text": prompt,
            "max_candidates": max_candidates,
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/ground-text", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _errors.http_failure(exc, service="SAM3", what="text-grounding request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.ground-text.v1":
            raise _errors.service_identity_drift("SAM3 text-grounding response schema mismatch", service="SAM3")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding, service="SAM3",
        )
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        masks = _decode_npy(result.get("masks_npy_base64")).astype(bool)
        scores = _decode_npy(result.get("scores_npy_base64")).astype(
            np.float64,
        ).reshape(-1)
        boxes = _decode_npy(result.get("boxes_xyxy_npy_base64")).astype(
            np.float64,
        )
        if masks.size == 0:
            masks = np.empty((0, rgb.shape[0], rgb.shape[1]), dtype=bool)
            boxes = np.empty((0, 4), dtype=np.float64)
        if (
            masks.ndim != 3
            or masks.shape[1:] != rgb.shape[:2]
            or boxes.ndim != 2
            or boxes.shape[1:] != (4,)
            or not len(masks) == len(scores) == len(boxes)
            or len(masks) > max_candidates
        ):
            raise ValueError("SAM3 text grounding returned unaligned candidates")
        candidates: list[dict[str, Any]] = []
        for source_index, (mask, score, box) in enumerate(
            zip(masks, scores, boxes, strict=True)
        ):
            if not np.all(np.isfinite(box)) or not math.isfinite(float(score)):
                raise ValueError("SAM3 text grounding returned non-finite values")
            if int(np.count_nonzero(mask)) < 1:
                continue
            clipped = [
                max(0, min(rgb.shape[1], int(round(float(box[0]))))),
                max(0, min(rgb.shape[0], int(round(float(box[1]))))),
                max(0, min(rgb.shape[1], int(round(float(box[2]))))),
                max(0, min(rgb.shape[0], int(round(float(box[3]))))),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError("SAM3 text grounding returned an invalid box")
            candidates.append({
                "source_index": source_index,
                "mask": mask,
                "backend_score": float(score),
                "bbox_xyxy": clipped,
            })
        candidates.sort(
            key=lambda item: (-float(item["backend_score"]), item["source_index"])
        )
        for rank, candidate in enumerate(candidates):
            candidate["rank"] = rank
        service_metrics = result.get("metrics")
        if not isinstance(service_metrics, Mapping):
            raise ValueError("SAM3 text-grounding response lacks timing metrics")
        return candidates, {
            "model": model,
            "_service_identity": service_identity,
            "request_id": str(result["request_id"]),
            "returned_candidate_count": len(candidates),
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(service_metrics["queue_ms"]),
                "service_inference": float(service_metrics["inference_ms"]),
                "service_total": float(service_metrics["total_ms"]),
            },
        }


@dataclass(frozen=True)
class ContactGraspNetApiProposer:
    """Contact-GraspNet client for one already-warm loopback service."""

    endpoint: str = "http://127.0.0.1:8115"
    timeout_s: float = 45.0
    forward_passes: int = 1
    hover_offset_m: float = 0.08
    maximum_aperture_m: float = 0.08
    raw_candidate_limit: int = 32
    backend_id: str = "contact_graspnet_api"
    backend_version: str = "v1"
    expected_binding: Mapping[str, Any] | None = None
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import httpx

        object.__setattr__(self, "endpoint", _loopback_endpoint(self.endpoint))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Contact-GraspNet service timeout must be positive")
        if self.forward_passes != 1:
            raise ValueError("the first Contact-GraspNet treatment fixes one forward pass")
        if (
            isinstance(self.raw_candidate_limit, bool)
            or not isinstance(self.raw_candidate_limit, int)
            or not 1 <= self.raw_candidate_limit <= 32
        ):
            raise ValueError("Contact-GraspNet raw candidate limit must be in [1,32]")
        if self.expected_binding is not None:
            object.__setattr__(
                self, "expected_binding",
                _normalize_expected_service_binding(
                    self.expected_binding,
                    schema="ora.grasp-proposal-service-binding.v1",
                    endpoint=self.endpoint,
                ),
            )
        object.__setattr__(
            self, "_client", httpx.Client(timeout=self.timeout_s),
        )

    def configuration(self) -> Mapping[str, Any]:
        return {
            "endpoint": self.endpoint,
            "timeout_s": self.timeout_s,
            "forward_passes": self.forward_passes,
            "hover_offset_m": self.hover_offset_m,
            "maximum_aperture_m": self.maximum_aperture_m,
            "raw_candidate_limit": self.raw_candidate_limit,
            "expected_model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            "api_schema": "ora.model-service.grasp-proposals.v1",
            "seed_policy": "canonical_scene_target_pointcloud_v1",
            "expected_binding_sha256": (
                None if self.expected_binding is None
                else _canonical_hash(self.expected_binding)
            ),
        }

    def close(self) -> None:
        self._client.close()

    def require_ready(self) -> dict[str, Any]:
        health = _require_ready_service(
            client=self._client, endpoint=self.endpoint,
            service="grasp_proposals",
            expected_model=CONTACT_GRASPNET_MODEL_IDENTITY,
        )
        if self.expected_binding is not None:
            _require_bound_service_identity(
                health=health, binding=self.expected_binding,
                service="Contact-GraspNet",
            )
        return health

    @staticmethod
    def _inside(point: Any, bounds: Mapping[str, list[float]]) -> bool:
        return all(
            float(bounds[axis][0]) <= float(point[index]) <= float(bounds[axis][1])
            for index, axis in enumerate(("x", "y", "z"))
        )

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        import httpx
        import numpy as np

        scene = np.ascontiguousarray(
            np.asarray(scene_points_camera_m, dtype=np.float32)
        )
        target = np.ascontiguousarray(
            np.asarray(target_points_camera_m, dtype=np.float32)
        )
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) < 16:
            raise ValueError("Contact-GraspNet requires an Nx3 scene point cloud")
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 16:
            raise _errors.TargetInsufficient("Contact-GraspNet requires an Nx3 target point cloud of at least 16 points")
        if transform.shape != (4, 4):
            raise ValueError("Contact-GraspNet requires one 4x4 camera-to-world transform")
        seed_identity = {
            "schema": "ora.grasp-inference-seed.v1",
            "scene": {
                "shape": list(scene.shape),
                "dtype": str(scene.dtype),
                "sha256": hashlib.sha256(scene.tobytes(order="C")).hexdigest(),
            },
            "target": {
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "sha256": hashlib.sha256(target.tobytes(order="C")).hexdigest(),
            },
        }
        inference_seed = int(_canonical_hash(seed_identity)[:8], 16)
        payload = {
            "schema": "ora.model-service.grasp-proposals-request.v1",
            "scene_points_npy_base64": _encode_npy(scene),
            "target_points_npy_base64": _encode_npy(target),
            # The policy-visible limit is applied only after embodiment
            # compilation.  Request a wider bounded backend pool so invalid
            # high-score proposals cannot hide a valid lower-ranked grasp.
            "max_candidates": self.raw_candidate_limit,
            "forward_passes": self.forward_passes,
            "seed": inference_seed,
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/grasp-proposals", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _errors.http_failure(exc, service="Contact-GraspNet", what="service request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.grasp-proposals.v1":
            raise _errors.service_identity_drift("Contact-GraspNet service response schema mismatch", service="Contact-GraspNet")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding,
            service="Contact-GraspNet",
        )
        model = _verify_model_identity(
            result.get("model"), CONTACT_GRASPNET_MODEL_IDENTITY,
        )
        poses = _decode_npy(result.get("poses_camera_npy_base64")).astype(np.float64)
        scores = _decode_npy(result.get("scores_npy_base64")).astype(np.float64).reshape(-1)
        openings = _decode_npy(
            result.get("openings_m_npy_base64"),
        ).astype(np.float64).reshape(-1)
        contacts = _decode_npy(
            result.get("contact_points_camera_npy_base64"),
        ).astype(np.float64)
        if poses.size == 0:
            poses = np.empty((0, 4, 4), dtype=np.float64)
            contacts = np.empty((0, 3), dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("Contact-GraspNet returned malformed poses")
        if contacts.ndim != 2 or contacts.shape[1:] != (3,):
            raise ValueError("Contact-GraspNet returned malformed contact points")
        if not (len(poses) == len(scores) == len(openings) == len(contacts)):
            raise ValueError("Contact-GraspNet returned unaligned candidate arrays")
        if not all(
            np.isfinite(array).all()
            for array in (poses, scores, openings, contacts)
        ):
            raise ValueError("Contact-GraspNet returned non-finite candidate arrays")
        compiler = dict(
            compiler_identity or PANDA_GRIP_SITE_COMPILER_IDENTITY
        )
        if compiler.get("id") not in GRASP_COMPILER_IDENTITIES:
            raise ValueError("Contact-GraspNet compiler identity is unregistered")
        controlled_offset = float(
            compiler["source_origin_to_controlled_point_m"]
        )
        maximum_opening = float(
            compiler.get("maximum_opening_m", self.maximum_aperture_m)
        )
        candidates = []
        rejected = []
        for source_index, (pose_camera, score, opening) in enumerate(
            zip(poses, scores, openings, strict=True)
        ):
            pose_world = transform @ pose_camera
            source_origin = pose_world[:3, 3]
            jaw = pose_world[:3, 0]
            approach = pose_world[:3, 2]
            identity = {
                "target_ref": target_ref,
                "pose_world": np.round(pose_world, 9).tolist(),
                "opening_m": round(float(opening), 9),
                "backend": self.backend_id,
                "version": self.backend_version,
                "compiler": compiler["id"],
            }
            candidate_id = "grasp-" + _canonical_hash(identity)[:24]
            reasons = []
            jaw_norm = float(np.linalg.norm(jaw))
            approach_norm = float(np.linalg.norm(approach))
            if jaw_norm <= 1e-6 or approach_norm <= 1e-6:
                reasons.append("degenerate_gripper_axis")
            else:
                jaw = jaw / jaw_norm
                approach = approach / approach_norm
                if abs(float(np.dot(jaw, approach))) > 0.05:
                    reasons.append("non_orthogonal_gripper_axes")
            if reasons:
                rejected.append({"candidate_id": candidate_id, "reasons": reasons})
                continue
            controlled_point = source_origin + approach * controlled_offset
            source_pregrasp = source_origin - approach * self.hover_offset_m
            controlled_pregrasp = (
                controlled_point - approach * self.hover_offset_m
            )
            reasons = []
            if not math.isfinite(float(opening)) or not 0 < opening <= maximum_opening:
                reasons.append(
                    "aperture_exceeds_panda_limit"
                    if compiler["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
                    else "aperture_exceeds_embodiment_limit"
                )
            if not self._inside(controlled_point, workspace_bounds_m):
                reasons.append("compiled_controlled_point_outside_workspace")
            if not self._inside(controlled_pregrasp, workspace_bounds_m):
                reasons.append(
                    "compiled_pregrasp_controlled_point_outside_workspace"
                )
            if reasons:
                rejected.append({"candidate_id": candidate_id, "reasons": reasons,
                                 "predicted_opening_m": float(opening) if math.isfinite(float(opening)) else None,
                                 "maximum_opening_m": maximum_opening,
                                 "limit_semantics": "compiler_admission_not_measured_physical_aperture"})
                continue
            candidates.append({
                "candidate_id": candidate_id,
                "source_index": source_index,
                "source_gripper_origin_xyz_m": [
                    float(item) for item in source_origin
                ],
                "source_pregrasp_origin_xyz_m": [
                    float(item) for item in source_pregrasp
                ],
                # Legacy public aliases retained for grasp.query consumers.
                # grasp.plan compiles only the explicit source-origin fields.
                "grasp_center_xyz_m": [float(item) for item in source_origin],
                "pregrasp_center_xyz_m": [
                    float(item) for item in source_pregrasp
                ],
                "approach_world": [float(item) for item in approach],
                "jaw_world": [float(item) for item in jaw],
                "gripper_opening_m": float(opening),
                "backend_score": float(score),
                "pose_convention": (
                    "contact_graspnet_panda_gripper_origin_opencv_v2"
                ),
                "checks": {
                    "workspace": "passed",
                    "aperture": "passed",
                    "ik": "not_checked",
                    "collision": "not_checked",
                },
                "execution_ready": False,
            })
        candidates.sort(
            key=lambda item: (-float(item["backend_score"]), item["candidate_id"])
        )
        candidates = candidates[:max_candidates]
        for rank, candidate in enumerate(candidates):
            candidate["rank"] = rank
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("Contact-GraspNet response lacks timing metrics")
        return candidates, {
            "model": model,
            "_service_identity": service_identity,
            "request_id": str(result["request_id"]),
            "scene_point_count": int(len(scene)),
            "target_point_count": int(len(target)),
            "returned_candidate_count": int(len(poses)),
            "raw_candidate_limit": self.raw_candidate_limit,
            "compiled_candidate_count": len(candidates),
            "inference_seed": inference_seed,
            "seed_policy": "canonical_scene_target_pointcloud_v1",
            "seed_identity_sha256": _canonical_hash(seed_identity),
            "_backend_input_regime": result.get("input_regime") if isinstance(result.get("input_regime"), Mapping) else None,
            "rejected": rejected,
            # Consumed by materialize_grasp_query and removed before the
            # policy-visible diagnostics projection.  These exact aligned
            # arrays make the learned proposal reproducible without exposing
            # an additional policy authority.
            "_canonical_backend_output": {
                "poses_camera": poses,
                "scores": scores,
                "contact_points_camera": contacts,
                "openings_m": openings,
            },
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(metrics["queue_ms"]),
                "service_inference": float(metrics["inference_ms"]),
                "service_total": float(metrics["total_ms"]),
            },
            "warning": (
                "aperture passed means only that the predicted opening is positive "
                "and within the compiler's admission limit; it does not establish "
                "object-width fit or measured jaw clearance. Model candidates "
                "require embodiment compilation, IK and collision checks"
            ),
        }


@dataclass(frozen=True)
class DepthPromptSegmenter:
    """Conservative point-prompt RGB-D segmentation baseline.

    Depth selects a bounded probable foreground region; GrabCut uses RGB only
    to refine its boundary.  The connected component containing the exact
    prompt is retained.  This is a qualification baseline, not a claim of
    open-vocabulary semantic segmentation.
    """

    backend_id: str = "depth_prompt_grabcut"
    backend_version: str = "v1"
    depth_band_min_m: float = 0.035
    depth_band_fraction: float = 0.08
    depth_band_max_m: float = 0.15
    grabcut_iterations: int = 3

    def configuration(self) -> Mapping[str, Any]:
        return {
            "depth_band_min_m": self.depth_band_min_m,
            "depth_band_fraction": self.depth_band_fraction,
            "depth_band_max_m": self.depth_band_max_m,
            "grabcut_iterations": self.grabcut_iterations,
        }

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]:
        import cv2
        import numpy as np

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        depth = np.asarray(depth_m, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("segmentation requires aligned HxWx3 RGB and HxW depth")
        u, v = point_uv
        height, width = depth.shape
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError("segmentation point is outside the source image")
        radius = 2
        seed_patch = depth[
            max(0, v - radius):min(height, v + radius + 1),
            max(0, u - radius):min(width, u + radius + 1),
        ]
        valid_seed = seed_patch[np.isfinite(seed_patch) & (seed_patch > 1e-6)]
        if not valid_seed.size:
            raise ValueError("segmentation point has no valid metric depth")
        seed_depth = float(np.median(valid_seed))
        depth_band = max(
            self.depth_band_min_m,
            min(self.depth_band_max_m, self.depth_band_fraction * seed_depth),
        )
        valid = np.isfinite(depth) & (depth > 1e-6)
        probable = valid & (np.abs(depth - seed_depth) <= depth_band)
        if int(np.count_nonzero(probable)) < 4:
            raise ValueError("segmentation prompt has insufficient nearby geometry")

        if min(height, width) >= 16:
            state = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
            state[probable] = cv2.GC_PR_FGD
            seed_radius = max(1, min(4, min(height, width) // 48))
            cv2.circle(state, (u, v), seed_radius, cv2.GC_FGD, thickness=-1)
            state[0, :] = cv2.GC_BGD
            state[-1, :] = cv2.GC_BGD
            state[:, 0] = cv2.GC_BGD
            state[:, -1] = cv2.GC_BGD
            background = np.zeros((1, 65), dtype=np.float64)
            foreground = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(
                    rgb, state, None, background, foreground,
                    self.grabcut_iterations, cv2.GC_INIT_WITH_MASK,
                )
                foreground_mask = (
                    (state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD)
                ) & probable
            except cv2.error:
                foreground_mask = probable
        else:
            foreground_mask = probable

        components, labels = cv2.connectedComponents(
            foreground_mask.astype(np.uint8), connectivity=8,
        )
        seed_label = int(labels[v, u])
        if components <= 1 or seed_label == 0:
            components, labels = cv2.connectedComponents(
                probable.astype(np.uint8), connectivity=8,
            )
            seed_label = int(labels[v, u])
        if components <= 1 or seed_label == 0:
            raise ValueError("segmentation could not isolate the prompted component")
        mask = labels == seed_label
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1,
        ).astype(bool)
        components, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8,
        )
        seed_label = int(labels[v, u])
        mask = labels == seed_label if components > 1 and seed_label else mask
        pixel_count = int(np.count_nonzero(mask))
        area_fraction = pixel_count / float(height * width)
        if pixel_count < 4:
            raise ValueError("segmentation result is too small")
        if area_fraction > 0.80:
            raise ValueError("segmentation result is implausibly broad")
        return mask, {
            "seed_depth_m": seed_depth,
            "depth_band_m": depth_band,
            "probable_pixel_count": int(np.count_nonzero(probable)),
        }


@dataclass(frozen=True)
class PcaTopDownGraspProposer:
    """Simple Panda-width top-down candidates from one segmented point cloud."""

    backend_id: str = "pca_topdown_parallel_jaw"
    backend_version: str = "v1"
    maximum_aperture_m: float = 0.08
    aperture_clearance_m: float = 0.006
    hover_offset_m: float = 0.08

    def configuration(self) -> Mapping[str, Any]:
        return {
            "maximum_aperture_m": self.maximum_aperture_m,
            "aperture_clearance_m": self.aperture_clearance_m,
            "hover_offset_m": self.hover_offset_m,
            "proposal_family": "top_down_only",
        }

    @staticmethod
    def _canonical_axis(axis: Any) -> Any:
        import numpy as np

        value = np.asarray(axis, dtype=np.float64)
        value = value / np.linalg.norm(value)
        if value[0] < -1e-9 or (abs(value[0]) <= 1e-9 and value[1] < 0):
            value = -value
        return value

    @staticmethod
    def _inside(
        point: Any, bounds: Mapping[str, list[float]],
    ) -> bool:
        return all(
            float(bounds[axis][0]) <= float(point[index]) <= float(bounds[axis][1])
            for index, axis in enumerate(("x", "y", "z"))
        )

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        import numpy as np

        points_camera = np.asarray(target_points_camera_m, dtype=np.float64)
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if points_camera.ndim != 2 or points_camera.shape[1] != 3:
            raise ValueError("grasp proposal requires Nx3 target points")
        if transform.shape != (4, 4):
            raise ValueError("grasp proposal requires one 4x4 camera-to-world transform")
        points_h = np.column_stack([points_camera, np.ones(len(points_camera))])
        points = (transform @ points_h.T).T[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 16:
            raise _errors.TargetInsufficient("grasp proposal has too few valid target points")
        low = np.percentile(points, 2, axis=0)
        high = np.percentile(points, 98, axis=0)
        retained = np.all((points >= low) & (points <= high), axis=1)
        filtered = points[retained]
        if len(filtered) < 16:
            filtered = points
        xy = filtered[:, :2]
        covariance = np.cov(xy - np.mean(xy, axis=0), rowvar=False)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise _errors.TargetInsufficient("target geometry has invalid horizontal covariance")
        values, vectors = np.linalg.eigh(covariance)
        minor = self._canonical_axis(vectors[:, int(np.argmin(values))])
        rotations = (0.0, math.pi / 4, -math.pi / 4, math.pi / 2)
        axis_options = []
        for angle in rotations:
            c, s = math.cos(angle), math.sin(angle)
            rotated = np.asarray([
                c * minor[0] - s * minor[1],
                s * minor[0] + c * minor[1],
            ])
            axis_options.append(self._canonical_axis(rotated))
        center = np.median(filtered, axis=0)
        approach = np.asarray([0.0, 0.0, -1.0])
        compiler = dict(
            compiler_identity or PANDA_GRIP_SITE_COMPILER_IDENTITY
        )
        if compiler.get("id") not in GRASP_COMPILER_IDENTITIES:
            raise ValueError("grasp compiler identity is unregistered")
        topdown_profile = compiler.get("public_rgbd_topdown_profile")
        height = None  # the X5 support-aware rule (x5_topdown.py); ALOHA below is unchanged
        if isinstance(topdown_profile, Mapping) and "height_rule" in topdown_profile:
            height = topdown_contact_height(filtered, scene_points_camera_m, transform, topdown_profile)
            center[2], hover_offset = height["contact_centre_z_m"], float(topdown_profile["pregrasp_standoff_m"])
            max_candidates = min(max_candidates, int(topdown_profile["maximum_candidates"]))
            center = center - approach * float(compiler["controlled_point_to_contact_baseline_m"])
        elif topdown_profile is None:
            center[2] = float(
                (
                    np.percentile(filtered[:, 2], 5)
                    + np.percentile(filtered[:, 2], 95)
                )
                / 2
            )
            hover_offset = self.hover_offset_m
        else:
            if (
                not isinstance(topdown_profile, Mapping)
                or set(topdown_profile) != {
                    "surface_percentile", "contact_clearance_m",
                    "pregrasp_standoff_m", "maximum_candidates",
                }
            ):
                raise ValueError("ALOHA public RGB-D top-down profile is invalid")
            percentile = float(topdown_profile["surface_percentile"])
            clearance = float(topdown_profile["contact_clearance_m"])
            hover_offset = float(topdown_profile["pregrasp_standoff_m"])
            maximum_topdown = int(topdown_profile["maximum_candidates"])
            if (
                not 50.0 <= percentile <= 99.0
                or not 0.0 <= clearance <= 0.02
                or not 0.02 <= hover_offset <= 0.10
                or not 1 <= maximum_topdown <= 4
            ):
                raise ValueError("ALOHA public RGB-D top-down profile is out of bounds")
            center[2] = float(
                np.percentile(filtered[:, 2], percentile) + clearance
            )
            max_candidates = min(max_candidates, maximum_topdown)
        hover = center - approach * hover_offset
        maximum_opening = float(
            compiler.get("maximum_opening_m", self.maximum_aperture_m)
        )
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for option_index, axis_xy in enumerate(axis_options):
            projection = xy @ axis_xy
            required = float(
                np.percentile(projection, 98) - np.percentile(projection, 2)
                + self.aperture_clearance_m
            )
            jaw = np.asarray([axis_xy[0], axis_xy[1], 0.0])
            reasons = []
            if required > maximum_opening:
                reasons.append(
                    "aperture_exceeds_panda_limit"
                    if compiler["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
                    else "aperture_exceeds_embodiment_limit"
                )
            if not self._inside(center, workspace_bounds_m):
                reasons.append("contact_pose_outside_workspace")
            if not self._inside(hover, workspace_bounds_m):
                reasons.append("hover_pose_outside_workspace")
            if height is not None and not height["reaches_object"]:
                reasons.append("fingertips_above_robust_top")
            identity = {
                "target_ref": target_ref,
                "center": [round(float(item), 9) for item in center],
                "jaw": [round(float(item), 9) for item in jaw],
                "approach": approach.tolist(),
                "required_aperture_m": round(required, 9),
                "backend": self.backend_id,
                "version": self.backend_version,
                "compiler": compiler["id"],
            }
            candidate_id = "grasp-" + _canonical_hash(identity)[:24]
            if reasons:
                rejected.append({"candidate_id": candidate_id, "reasons": reasons,
                                 "required_opening_m": required,
                                 "maximum_opening_m": maximum_opening,
                                 "limit_semantics": "compiler_admission_not_measured_physical_aperture"})
                continue
            score = max(0.0, 1.0 - required / maximum_opening)
            candidates.append({
                "candidate_id": candidate_id,
                "source_index": option_index,
                "grasp_center_xyz_m": [float(item) for item in center],
                "pregrasp_center_xyz_m": [float(item) for item in hover],
                "approach_world": approach.tolist(),
                "jaw_world": [float(item) for item in jaw],
                "gripper_opening_m": required,
                "backend_score": score,
                "pose_convention": "pca_topdown_parallel_jaw_center_v1",
                **({"topdown_height": dict(height)} if height is not None else {}),
                "checks": {
                    "workspace": "passed",
                    "aperture": "passed",
                    "ik": "not_checked",
                    "collision": "not_checked",
                },
                "execution_ready": False,
            })
        candidates.sort(
            key=lambda item: (-float(item["backend_score"]), item["candidate_id"])
        )
        candidates = candidates[:max_candidates]
        for rank, candidate in enumerate(candidates):
            candidate["rank"] = rank
        return candidates, {
            "model": {
                "family": "deterministic_pca_test_double",
                "source_commit": "repository-source",
                "checkpoint_sha256": "not_applicable",
            },
            "target_point_count": int(len(points)),
            "filtered_point_count": int(len(filtered)),
            "rejected": rejected,
            "warning": "geometric candidates are not IK or collision qualified",
        }


@dataclass(frozen=True)
class AlohaHybridGraspProposer:
    """Fuse learned CGN candidates with public-RGB-D ALOHA top-down poses.

    The learned service remains source-bound and mandatory.  The deterministic
    family adds a geometry-first candidate when the target mask supplies a
    closure-compatible top surface.  Neither family receives simulator GT.
    """

    contact_backend: GraspProposalBackend
    topdown_backend: PcaTopDownGraspProposer = field(
        default_factory=PcaTopDownGraspProposer,
    )
    backend_id: str = "contact_graspnet_plus_aloha_rgbd_topdown"
    backend_version: str = "v1"

    def configuration(self) -> Mapping[str, Any]:
        return {
            "contact_backend": {
                "id": self.contact_backend.backend_id,
                "version": self.contact_backend.backend_version,
                "configuration": dict(self.contact_backend.configuration()),
            },
            "topdown_backend": {
                "id": self.topdown_backend.backend_id,
                "version": self.topdown_backend.backend_version,
                "configuration": dict(self.topdown_backend.configuration()),
            },
            "selection": "up_to_two_topdown_then_contact_fill_v1",
        }

    def close(self) -> None:
        closer = getattr(self.contact_backend, "close", None)
        if callable(closer):
            closer()

    def require_ready(self) -> Mapping[str, Any]:
        readiness = getattr(self.contact_backend, "require_ready", None)
        if not callable(readiness):
            raise ValueError("hybrid grasp proposer lacks learned-service readiness")
        return readiness()

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        learned, learned_raw_diagnostics = self.contact_backend.propose(
            scene_points_camera_m=scene_points_camera_m,
            target_points_camera_m=target_points_camera_m,
            camera_to_world=camera_to_world,
            target_ref=target_ref,
            workspace_bounds_m=workspace_bounds_m,
            max_candidates=max_candidates,
            compiler_identity=compiler_identity,
        )
        # The learned service's private response identity and canonical
        # backend arrays must surface at the top level so the shared
        # materializer can strip them before the policy-visible projection.
        learned_diagnostics = dict(learned_raw_diagnostics)
        service_identity = learned_diagnostics.pop("_service_identity", None)
        backend_output = learned_diagnostics.pop("_canonical_backend_output", None)
        input_regime = learned_diagnostics.pop("_backend_input_regime", None)
        geometric, geometric_diagnostics = self.topdown_backend.propose(
            scene_points_camera_m=scene_points_camera_m,
            target_points_camera_m=target_points_camera_m,
            camera_to_world=camera_to_world,
            target_ref=target_ref,
            workspace_bounds_m=workspace_bounds_m,
            max_candidates=max_candidates,
            compiler_identity=compiler_identity,
        )
        selected_geometric_count = min(2, max_candidates, len(geometric))
        selected = [dict(item) for item in geometric[:selected_geometric_count]]
        selected.extend(
            dict(item) for item in learned[: max_candidates - len(selected)]
        )
        if len(selected) < max_candidates:
            selected.extend(
                dict(item)
                for item in geometric[
                    selected_geometric_count:
                    selected_geometric_count + max_candidates - len(selected)
                ]
            )
        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for candidate in selected:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidate["rank"] = len(deduplicated)
            deduplicated.append(candidate)
        diagnostics: dict[str, Any] = hybrid_grasp_diagnostics(
            learned_diagnostics=learned_diagnostics,
            geometric_diagnostics=geometric_diagnostics,
            learned=learned, geometric=geometric, selected=deduplicated,
        )
        if service_identity is not None:
            diagnostics["_service_identity"] = service_identity
        if backend_output is not None:
            diagnostics["_canonical_backend_output"] = backend_output
        if input_regime is not None:
            diagnostics["_backend_input_regime"] = input_regime
        return deduplicated, diagnostics


def _mask_summary(mask: Any, depth_m: Any) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    rows, columns = np.nonzero(value)
    if not len(rows):
        raise ValueError("segmentation mask is empty")
    selected = depth[value]
    valid = selected[
        np.isfinite(selected) & (selected > 1e-6) & (selected <= 2.5)
    ]
    summary: dict[str, Any] = {
        "pixel_count": int(len(rows)),
        "area_fraction": float(len(rows) / value.size),
        "bbox_xyxy": [
            int(columns.min()), int(rows.min()),
            int(columns.max()) + 1, int(rows.max()) + 1,
        ],
        "centroid_uv": [float(np.mean(columns)), float(np.mean(rows))],
        "depth_valid_fraction": float(len(valid) / len(selected)),
        "depth_m": None,
    }
    if len(valid):
        summary["depth_m"] = {
            "min": float(np.min(valid)),
            "median": float(np.median(valid)),
            "max": float(np.max(valid)),
        }
    return summary


def _mask_depth_inlier_selection(
    *, mask: Any, depth_m: Any,
) -> tuple[Any, dict[str, Any]]:
    """Share the existing localization input selection without trimming XYZ.

    A connected handle within the depth window remains part of the source.
    This is measurement selection, never object identity or completed shape.
    """

    import cv2
    import numpy as np

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if selected.ndim != 2 or not selected.size or selected.shape != depth.shape:
        raise ValueError("segmentation mask and depth must have matching image dimensions")
    valid = selected & np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5)
    valid_count = int(np.count_nonzero(valid))
    mask_count = int(np.count_nonzero(selected))
    component_count, labels = cv2.connectedComponents(
        valid.astype(np.uint8), connectivity=8,
    )
    component_sizes = np.bincount(labels.ravel(), minlength=component_count)
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        primary_label = int(np.argmax(component_sizes))
        primary = labels == primary_label
        primary_count = int(component_sizes[primary_label])
    else:
        primary = np.zeros_like(valid)
        primary_count = 0
    median_depth = mad_depth = depth_half_width = None
    inlier_mask = np.zeros_like(valid)
    if primary_count >= 16:
        primary_depth = depth[primary]
        median_depth = float(np.median(primary_depth))
        mad_depth = float(np.median(np.abs(primary_depth - median_depth)))
        depth_half_width = max(0.03, 4.0 * 1.4826 * mad_depth)
        inlier_mask = primary & (np.abs(depth - median_depth) <= depth_half_width)
    retained_count = int(np.count_nonzero(inlier_mask))
    return inlier_mask, {
        "method": DEPTH_INLIER_SELECTION_METHOD,
        "mask_pixel_count": mask_count,
        "valid_depth_point_count": valid_count,
        "valid_depth_component_count": max(0, component_count - 1),
        "primary_component_point_count": primary_count,
        "inlier_point_count": retained_count,
        "depth_valid_fraction": float(valid_count / mask_count) if mask_count else 0.0,
        "primary_component_fraction": float(primary_count / valid_count) if valid_count else 0.0,
        "depth_inlier_fraction": float(retained_count / primary_count) if primary_count else 0.0,
        "median_depth_m": median_depth,
        "median_absolute_deviation_m": mad_depth,
        "depth_half_width_m": depth_half_width,
        "valid_depth_range_m": [1e-6, 2.5],
        "connectivity": 8,
        "minimum_component_points": 16,
        "minimum_depth_half_width_m": 0.03,
        "mad_scale": 4.0 * 1.4826,
        "touches_image_boundary": bool(selected[0].any() or selected[-1].any()
                                       or selected[:, 0].any() or selected[:, -1].any()),
    }


def _mask_depth_localization(
    *, mask: Any, depth_m: Any, calibration_path: Path,
    extended_frame: bool = False,
) -> dict[str, Any]:
    """Estimate bounded object geometry from public RGB-D measurements."""

    import numpy as np

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if selected.shape != depth.shape or intrinsic.shape != (3, 3):
        raise ValueError("segmentation localization RGB-D calibration shape mismatch")
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if extrinsic.shape != (4, 4) or not (
        np.isfinite(intrinsic).all() and np.isfinite(extrinsic).all()
    ):
        raise ValueError("segmentation localization calibration is invalid")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("segmentation localization focal lengths must be positive")

    inlier_mask, selection = _mask_depth_inlier_selection(mask=selected, depth_m=depth)
    valid_count = selection["valid_depth_point_count"]
    mask_count = selection["mask_pixel_count"]
    primary_count = selection["primary_component_point_count"]

    base = {
        "status": "insufficient_depth",
        "method": LOCALIZATION_METHOD,
        "frame": "world",
        "scope": "visible_mask_depth",
        "mask_pixel_count": mask_count,
        "valid_depth_point_count": valid_count,
        "valid_depth_component_count": selection["valid_depth_component_count"],
        "inlier_point_count": 0,
        "depth_valid_fraction": float(valid_count / mask_count) if mask_count else 0.0,
        "primary_component_fraction": (
            float(primary_count / valid_count) if valid_count else 0.0
        ),
        "depth_inlier_fraction": 0.0,
        "center_xyz_m": None,
        "aabb_min_xyz_m": None,
        "aabb_max_xyz_m": None,
        "extent_xyz_m": None,
        "principal_axis_world": None,
    }
    if extended_frame:
        base.update(dict.fromkeys(_EXTENDED_FRAME_KEYS), **target_support.empty(), **overhead_view.empty())
    if primary_count < 16:
        return base

    rows, columns = np.nonzero(inlier_mask)
    if len(rows) < 16:
        return base
    z = depth[rows, columns]
    camera_points = np.stack([
        (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
        np.ones_like(z),
    ], axis=1)
    try:
        camera_to_world = np.linalg.inv(extrinsic)
    except np.linalg.LinAlgError as exc:
        raise ValueError("segmentation localization extrinsic is singular") from exc
    world_points = (camera_to_world @ camera_points.T).T[:, :3]
    if not np.isfinite(world_points).all():
        raise ValueError("segmentation localization produced non-finite points")
    center = np.median(world_points, axis=0)
    low, high = np.percentile(world_points, [2.5, 97.5], axis=0)
    centered = world_points - center
    base.update({
        "status": "localized",
        "inlier_point_count": int(len(world_points)),
        "depth_inlier_fraction": float(len(world_points) / primary_count),
        "center_xyz_m": [float(value) for value in center],
        "aabb_min_xyz_m": [float(value) for value in low],
        "aabb_max_xyz_m": [float(value) for value in high],
        "extent_xyz_m": [float(value) for value in high - low],
        **_principal_frame(centered, extended=extended_frame),
        **(target_support.estimate(mask=selected, depth_m=depth, intrinsic=intrinsic, object_points=world_points,
                                   camera_to_world=camera_to_world) if extended_frame else {}),
        **(_footprint(inlier_mask, depth, intrinsic, camera_to_world, world_points) if extended_frame else {}),
    })
    return base


def _footprint(inlier_mask: Any, depth: Any, intrinsic: Any, camera_to_world: Any, world_points: Any) -> dict[str, Any]:
    """``overhead_view.footprint`` of the inlier mask eroded by one pixel (its edge carries background depth)."""
    import cv2
    import numpy as np

    core = cv2.erode(np.asarray(inlier_mask, dtype=np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rows, columns = np.nonzero(core)
    points = world_points
    if len(rows) >= 16:
        z = depth[rows, columns]
        camera_points = np.stack([
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones_like(z),
        ], axis=1)
        points = (camera_to_world @ camera_points.T).T[:, :3]
    return overhead_view.footprint(points, camera_to_world[:3, 3])


def _principal_frame(centered_points: Any, *, extended: bool) -> dict[str, Any]:
    """The target's measured principal frame, with each axis beside its extent.

    The eigendecomposition was already being run and two of its three vectors
    thrown away, so the policy could read a target's position and size but
    never its orientation. Three tasks turn on exactly that and stand at zero
    across every cohort: a QR board faces the way its least-extent axis points,
    a fan's facing needs two axes to disambiguate, and an upright bottle is its
    greatest-extent axis.

    ``principal_extents_m`` keeps eigenvalue order, because that is what pairs
    each extent with ``principal_axes_world[i]``. Eigenvalue order is variance
    order and variance order is not size order -- a real target measured
    [0.1034, 0.0704, 0.0751] -- so the named largest/middle/smallest come from
    the sizes and each carries the axis that goes with it. Extents are the same
    2.5/97.5 percentile spread the centre and AABB already use, measured along
    each axis rather than along the world axes the AABB happens to use.

    Each axis is sign-normalised by its dominant component, so the same target
    measured twice reports the same frame rather than an arbitrary flip.
    """

    import numpy as np

    covariance = centered_points.T @ centered_points / max(1, len(centered_points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Keep the released single-axis meaning: maximum variance, not maximum
    # percentile extent. Sparse point clouds can order these differently.
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    if principal[int(np.argmax(np.abs(principal)))] < 0:
        principal = -principal
    legacy_axis = [float(value) for value in principal]
    if not extended:
        return {"principal_axis_world": legacy_axis}
    order = np.argsort(eigenvalues)[::-1]
    axes, extents = [], []
    for index in order:
        axis = eigenvectors[:, index]
        sign_index = int(np.argmax(np.abs(axis)))
        if axis[sign_index] < 0:
            axis = -axis
        projection = centered_points @ axis
        low, high = np.percentile(projection, [2.5, 97.5])
        axes.append([float(value) for value in axis])
        extents.append(float(high - low))
    ranked = sorted(range(3), key=lambda i: extents[i], reverse=True)
    largest, middle, smallest = ranked
    return {
        "principal_axes_world": axes,
        "principal_extents_m": extents,
        "largest_extent_m": extents[largest],
        "middle_extent_m": extents[middle],
        "smallest_extent_m": extents[smallest],
        "largest_extent_axis_world": axes[largest],
        "middle_extent_axis_world": axes[middle],
        "smallest_extent_axis_world": axes[smallest],
        # The single axis older consumers read, unchanged in meaning.
        "principal_axis_world": legacy_axis,
    }


def _render_mask_overlay(
    *, rgb_u8: Any, mask: Any, point_uv: tuple[int, int], summary: Mapping[str, Any],
) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    selected = np.asarray(mask, dtype=bool)
    overlay = rgb.copy()
    tint = np.asarray([48, 220, 96], dtype=np.float64)
    overlay[selected] = np.clip(
        0.55 * overlay[selected].astype(np.float64) + 0.45 * tint, 0, 255,
    ).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    x0, y0, x1, y1 = summary["bbox_xyxy"]
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 64), 1)
    cv2.drawMarker(
        overlay, point_uv, (255, 64, 64), cv2.MARKER_CROSS,
        markerSize=max(7, min(rgb.shape[:2]) // 20), thickness=1,
    )
    return Image.fromarray(overlay, mode="RGB")


def _render_grounding_overlay(
    *, rgb_u8: Any, candidates: list[Mapping[str, Any]],
) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    overlay = rgb.astype(np.float64)
    colours = tuple(
        np.asarray(grounding_candidate_colour(index), dtype=np.float64)
        for index in range(len(candidates))
    )
    for candidate, colour in zip(candidates, colours, strict=True):
        selected = np.asarray(candidate["mask"], dtype=bool)
        overlay[selected] = np.clip(
            0.60 * overlay[selected] + 0.40 * colour, 0, 255,
        )
    rendered = overlay.astype(np.uint8)
    for candidate, colour in zip(candidates, colours, strict=True):
        selected = np.asarray(candidate["mask"], dtype=bool)
        display_colour = tuple(int(item) for item in colour)
        contours, _hierarchy = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(rendered, contours, -1, display_colour, 1)
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        cv2.rectangle(
            rendered, (x0, y0), (x1 - 1, y1 - 1), display_colour, 2,
        )
        cv2.putText(
            rendered, f"T{candidate['rank']}", (x0 + 2, max(12, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, display_colour, 1, cv2.LINE_AA,
        )
    return Image.fromarray(rendered, mode="RGB")


def materialize_grounded_targets(
    *, artifact_root: Path, observation_id: str, camera: str,
    rgb_path: Path, depth_path: Path, calibration_path: Path,
    text: str, max_candidates: int, backend: TextGroundingBackend,
    depth_profile: Mapping[str, Any] | None = None,  # grounding_preview.options(surface)
) -> tuple[dict[str, Any], list[Path]]:
    import numpy as np
    from PIL import Image

    source = _segmentation_source_identity(
        observation_id=observation_id, camera=camera,
        rgb_path=rgb_path, depth_path=depth_path,
        calibration_path=calibration_path,
    )
    backend_record = {
        "id": backend.backend_id,
        "version": backend.backend_version,
        "config_sha256": _canonical_hash(dict(backend.configuration())),
    }
    query_identity = {
        "observation_id": observation_id,
        "camera": camera,
        "text": text,
        "max_candidates": max_candidates,
        "source": source,
        "backend": backend_record,
        "visualization": {"renderer": GROUNDING_VISUALIZATION_ID,
                          **grounding_preview.identity(depth_profile)},
    }
    query_ref = "grounding-query-" + _canonical_hash(query_identity)[:24]
    query_root = artifact_root / "grounding-queries" / query_ref
    manifest_path = query_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != GROUNDING_QUERY_SCHEMA
            or manifest.get("identity") != query_identity
        ):
            raise ValueError("grounding query identity drift")
        overlay_path = _verify_artifact(query_root, manifest["overlay"])
        visualization = manifest.get("visualization")
        if (
            not isinstance(visualization, Mapping)
            or visualization.get("renderer") != GROUNDING_VISUALIZATION_ID
        ):
            raise ValueError("grounding visualization identity drift")
        preview_paths = [
            _verify_artifact(query_root, item["artifact"])
            for item in visualization.get("candidate_previews", [])
        ]
        for candidate in manifest.get("candidates", []):
            load_segmented_target(
                artifact_root=artifact_root, target_ref=candidate["target_ref"],
            )
        return manifest, [overlay_path, *preview_paths]

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
    raw_candidates, raw_diagnostics = backend.ground(
        rgb_u8=rgb, text=text, max_candidates=max_candidates,
    )
    diagnostics = dict(raw_diagnostics)
    response_service_identity = diagnostics.pop("_service_identity", None)
    candidates: list[dict[str, Any]] = []
    rendered_candidates: list[dict[str, Any]] = []
    for raw in raw_candidates[:max_candidates]:
        mask = np.asarray(raw["mask"], dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("grounded mask shape differs from aligned depth")
        summary = _mask_summary(mask, depth)
        localization = _mask_depth_localization(
            mask=mask, depth_m=depth, calibration_path=calibration_path,
            extended_frame=True,
        )
        rank = int(raw["rank"])
        source_index = int(raw["source_index"])
        box = [int(item) for item in raw["bbox_xyxy"]]
        mask_sha256 = hashlib.sha256(
            np.asarray(mask, dtype=np.uint8).tobytes(order="C")
        ).hexdigest()
        identity = {
            "observation_id": observation_id,
            "camera": camera,
            "prompt_kind": "text",
            "text": text,
            "source_index": source_index,
            "mask_sha256": mask_sha256,
            "source": source,
            "backend": backend_record,
            "localization_method": LOCALIZATION_METHOD,
        }
        target_ref = "target-" + _canonical_hash(identity)[:24]
        target_root = _segmented_target_root(
            artifact_root=artifact_root, observation_id=observation_id,
            camera=camera, target_ref=target_ref,
        )
        target_root.mkdir(parents=True, exist_ok=True)
        mask_path = target_root / "mask.npy"
        target_overlay_path = target_root / "overlay.png"
        target_manifest_path = target_root / "manifest.json"
        rendered = {
            "mask": mask,
            "bbox_xyxy": box,
            "rank": rank,
        }
        if not target_manifest_path.is_file():
            _atomic_npy(mask_path, mask.astype(np.uint8))
            target_overlay = _render_grounding_overlay(
                rgb_u8=rgb, candidates=[rendered],
            )
            _atomic_png(target_overlay_path, target_overlay)
            mask_path.chmod(0o444)
            target_overlay_path.chmod(0o444)
            target_manifest = {
                "schema": TARGET_SCHEMA,
                "target_ref": target_ref,
                "identity": identity,
                "prompt": {
                    "kind": "text", "text": text,
                    "source_index": source_index,
                },
                "backend": backend_record,
                "summary": summary,
                "localization": localization,
                "diagnostics": {
                    "candidate_metadata_scope": "grounding_query_manifest",
                },
                "private_mask": _artifact(mask_path, target_root),
                "overlay": _artifact(target_overlay_path, target_root),
            }
            _atomic_json(target_manifest_path, target_manifest)
            target_manifest_path.chmod(0o444)
        else:
            target_manifest = json.loads(
                target_manifest_path.read_text(encoding="utf-8")
            )
            if (
                target_manifest.get("schema") != TARGET_SCHEMA
                or target_manifest.get("identity") != identity
            ):
                raise ValueError("grounded target identity drift")
            _verify_artifact(target_root, target_manifest["private_mask"])
            _verify_artifact(target_root, target_manifest["overlay"])
        _materialize_target_index(
            artifact_root=artifact_root, target_root=target_root,
            manifest_path=target_manifest_path, observation_id=observation_id,
            camera=camera, target_ref=target_ref,
        )
        candidates.append({
            "rank": rank,
            "target_ref": target_ref,
            "backend_score": float(raw["backend_score"]),
            "bbox_xyxy": box,
            "mask_summary": summary,
            "localization": localization,
            "overlay_sha256": target_manifest["overlay"]["sha256"],
        })
        rendered_candidates.append(rendered)

    query_root.mkdir(parents=True, exist_ok=True)
    overlay_path = query_root / "overlay.png"
    overlay = _render_grounding_overlay(
        rgb_u8=rgb, candidates=rendered_candidates,
    )
    _atomic_png(overlay_path, overlay)
    overlay_path.chmod(0o444)
    candidate_previews: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    # One preview per candidate is unbounded against the result-size ceiling:
    # every one of these reaches images_base64 and adds a 340-byte record, and
    # at 32 the result passes 50,000 bytes and is replaced by a pointer.
    for rendered in rendered_candidates[:GROUNDING_PREVIEW_RENDER_LIMIT]:
        target_ref = next(
            item["target_ref"] for item in candidates
            if item["rank"] == rendered["rank"]
        )
        preview_input = dict(rendered)
        preview_input["target_ref"] = target_ref
        preview_input["backend_score"] = next(
            item["backend_score"] for item in candidates
            if item["rank"] == rendered["rank"]
        )
        preview, preview_record = _render_grounding_candidate_preview(
            rgb_u8=rgb, candidate=preview_input, text=text,
            depth_profile=grounding_preview.profile_input(
                depth_profile, depth_m=depth, calibration_path=calibration_path,
                localization=next(item["localization"] for item in candidates
                                  if item["rank"] == rendered["rank"])),
        )
        preview_path = query_root / f"candidate-T{rendered['rank']}.png"
        _atomic_png(preview_path, preview)
        preview_path.chmod(0o444)
        preview_record["artifact"] = _artifact(preview_path, query_root)
        candidate_previews.append(preview_record)
        preview_paths.append(preview_path)
    overview_artifact = _artifact(overlay_path, query_root)
    manifest = {
        "schema": GROUNDING_QUERY_SCHEMA,
        "grounding_ref": query_ref,
        "identity": query_identity,
        "backend": backend_record,
        "status": "candidates" if candidates else "no_candidate",
        "candidates": candidates,
        "diagnostics": dict(diagnostics),
        "response_service_identity": response_service_identity,
        "overlay": overview_artifact,
        "visualization": {
            "renderer": GROUNDING_VISUALIZATION_ID,
            "policy_visible": True,
            "image_order": [
                "overview",
                *[f"T{item['rank']}" for item in candidate_previews],
            ],
            "overview": overview_artifact,
            "candidate_previews": candidate_previews,
            "claim_boundary": (
                "overview plus candidate-local enlargements from exact public "
                "RGB pixels; no new sensor detail or semantic ground truth"
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return manifest, [overlay_path, *preview_paths]


def load_segmented_target(
    *, artifact_root: Path, target_ref: str,
) -> tuple[dict[str, Any], Path, Path]:
    if TARGET_REF_PATTERN.fullmatch(target_ref) is None:
        raise _errors.TargetRefMalformed("target_ref is malformed: not a segmented-target reference")
    target_root = _resolve_segmented_target_root(
        artifact_root=artifact_root, target_ref=target_ref,
    )
    manifest_path = target_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TARGET_SCHEMA or manifest.get("target_ref") != target_ref:
        raise _errors.ArtifactIntegrityError("target_ref manifest is invalid")
    identity = manifest.get("identity")
    if (
        not isinstance(identity, Mapping)
        or "target-" + _canonical_hash(identity)[:24] != target_ref
    ):
        raise _errors.ArtifactHashMismatch("target_ref does not match its content identity")
    mask = _verify_artifact(target_root, manifest["private_mask"])
    overlay = _verify_artifact(target_root, manifest["overlay"])
    return manifest, mask, overlay


def validate_segmented_target_source(
    target: Mapping[str, Any], *, observation_id: str, rgb_path: Path,
    depth_path: Path, calibration_path: Path,
) -> None:
    """Bind derived target geometry to the exact source RGB, depth and calibration."""
    identity = target.get("identity", {})
    if identity.get("observation_id") != observation_id:
        raise ValueError("target_ref is stale for the selected observation")
    source = _segmentation_source_identity(
        observation_id=observation_id, camera=identity.get("camera", ""),
        rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path,
    )
    legacy = {f"{name}_sha256": source["files"][name]["sha256"]
              for name in ("rgb", "depth", "calibration")}
    if identity.get("source") not in (source, legacy):
        raise ValueError("target_ref source artifacts do not match the selected observation")


def perception_selection_evidence(
    *, result: Mapping[str, Any], artifact_root: Path,
    source_size: tuple[int, int], point_uv: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Expose selection/depth limitations without declaring task identity."""
    import numpy as np

    width, height = source_size
    references = [result["target_ref"]] if point_uv is not None else [
        candidate["target_ref"] for candidate in result["candidates"]
    ]
    records = []
    point_inside = None
    for reference in references:
        target, mask_path, _ = load_segmented_target(artifact_root=artifact_root, target_ref=reference)
        mask = np.load(mask_path, allow_pickle=False).astype(bool)
        if mask.shape != (height, width):
            raise ValueError("selection evidence dimensions differ from source image")
        if point_uv is not None:
            point_inside = bool(mask[point_uv[1], point_uv[0]])
        summary, localization = target["summary"], target.get("localization", {})
        def number(value: Any) -> float | None:
            return float(value) if type(value) in (int, float) and math.isfinite(value) else None
        extent = localization.get("extent_xyz_m")
        if not (isinstance(extent, list) and len(extent) == 3
                and all(number(value) is not None for value in extent)):
            extent = None
        records.append({
            "target_ref": reference,
            "mask_touches_image_boundary": bool(mask[0].any() or mask[-1].any()
                                                or mask[:, 0].any() or mask[:, -1].any()),
            "mask_pixel_count": int(np.count_nonzero(mask)),
            "depth_valid_fraction": number(summary.get("depth_valid_fraction")),
            "depth_inlier_fraction": number(localization.get("depth_inlier_fraction")),
            "primary_component_fraction": number(localization.get("primary_component_fraction")),
            "visible_extent_xyz_m": extent,
        })
    return {
        "schema": "ora.perception-selection-evidence.v1",
        "prompt_kind": "point" if point_uv is not None else "text",
        "source_image_size": [width, height],
        "point_uv": None if point_uv is None else list(point_uv),
        "point_inside_returned_mask": point_inside,
        "semantic_identity_verified": False,
        "targets": records,
    }


def segmented_target_geometry(
    *, artifact_root: Path, target_ref: str,
    depth_path: Path, calibration_path: Path, voxel_size_m: float = 0.01,
    use_depth_inliers: bool = False, rgb_path: Path | None = None,
) -> dict[str, Any]:
    """Derive policy-safe target geometry only from its bound RGB-D mask."""

    import numpy as np

    manifest, mask_path, _overlay = load_segmented_target(
        artifact_root=artifact_root, target_ref=target_ref,
    )
    selection = None
    selected_pixels = None
    source_binding = None
    if use_depth_inliers:
        if rgb_path is None:
            raise ValueError("depth-inlier target geometry requires its bound RGB source")
        validate_segmented_target_source(
            manifest, observation_id=manifest["identity"]["observation_id"],
            rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path,
        )
        raw_mask = np.load(mask_path, allow_pickle=False).astype(bool)
        depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
        inlier_mask, selection = _mask_depth_inlier_selection(mask=raw_mask, depth_m=depth)
        valid = raw_mask & np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5)
        selected_pixels = inlier_mask[valid]
        source_binding = {
            "observation_id": manifest["identity"]["observation_id"],
            "target_ref": target_ref,
            "target_manifest_sha256": _file_hash(mask_path.parent / "manifest.json"),
            "mask_sha256": _file_hash(mask_path),
            "rgb_sha256": _file_hash(rgb_path),
            "depth_sha256": _file_hash(depth_path),
            "calibration_sha256": _file_hash(calibration_path),
            "retained_pixel_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(inlier_mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "retained_pixel_mask_shape": list(inlier_mask.shape),
            "retained_pixel_mask_encoding": "uint8_opencv_top_left_row_major",
        }
    _scene_camera, target_camera, camera_to_world = (
        _scene_and_target_points_camera(
            mask_path=mask_path,
            depth_path=depth_path,
            calibration_path=calibration_path,
        )
    )
    if selected_pixels is not None:
        target_camera = target_camera[selected_pixels]
    target_world = np.asarray(
        _camera_points_to_world(target_camera, camera_to_world),
        dtype=np.float64,
    )
    if target_world.ndim != 2 or target_world.shape[1] != 3 or len(target_world) < 16:
        raise ValueError("segmented target world geometry is invalid")
    centroid = np.mean(target_world, axis=0)
    covariance = np.cov((target_world - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order].T
    for axis in axes:
        largest = int(np.argmax(np.abs(axis)))
        if axis[largest] < 0.0:
            axis *= -1.0
    projections = (target_world - centroid) @ axes.T
    extents = np.max(projections, axis=0) - np.min(projections, axis=0)
    quantized = np.round(target_world / float(voxel_size_m)).astype(np.int64)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    voxel_points = target_world[np.sort(indices)]
    result = {
        "schema": "ora.observation-bound-target-geometry.v1",
        "observation_id": manifest["identity"]["observation_id"],
        "target_ref": target_ref,
        "source_camera": manifest["identity"]["camera"],
        "centroid_world_m": centroid.tolist(),
        "principal_axes_world": axes.tolist(),
        "principal_variances_m2": eigenvalues.tolist(),
        "principal_extents_m": extents.tolist(),
        "visible_point_count": int(target_world.shape[0]),
        "voxel_size_m": float(voxel_size_m),
        "voxel_points_world_m": voxel_points.tolist(),
        "claim_boundary": (
            "current visible RGB-D mask geometry only; no hidden surface, object "
            "identity, contact, retention, simulator ground truth or expert data"
        ),
    }
    if selection is not None:
        source_quality = {key: selection[key] for key in (
            "depth_valid_fraction", "depth_inlier_fraction",
            "primary_component_fraction", "touches_image_boundary",
        )}
        failed = [key for key, minimum in TARGET_GEOMETRY_QUALITY_MINIMUMS.items()
                  if source_quality[key] < minimum]
        result.update({
            "schema": "ora.observation-bound-target-geometry.v2",
            "status": "unknown" if failed else "measured",
            "reason": "mask_depth_quality_insufficient" if failed else "selected_visible_depth_measured",
            "source_quality": source_quality,
            "quality_minimums": dict(TARGET_GEOMETRY_QUALITY_MINIMUMS),
            "quality_failures": failed,
            "input_selection": {**selection, "retained_voxel_point_count": len(voxel_points)},
            "source_binding": source_binding,
            "semantic_identity_verified": False,
        })
    return result


def _scene_and_target_points_camera(
    *, mask_path: Path, depth_path: Path, calibration_path: Path,
) -> tuple[Any, Any, Any]:
    import numpy as np

    mask = np.load(mask_path, allow_pickle=False).astype(bool)
    depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if mask.shape != depth.shape or intrinsic.shape != (3, 3):
        raise ValueError("segmented target RGB-D calibration shape mismatch")
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if extrinsic.shape != (4, 4):
        raise ValueError("segmented target extrinsic must be 4x4")
    scene_valid = np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5)
    rows, columns = np.nonzero(scene_valid)
    if len(rows) < 16 or int(np.count_nonzero(mask & scene_valid)) < 16:
        raise _errors.TargetInsufficient("segmented target has too few valid depth pixels")
    z = depth[rows, columns]
    scene_camera = np.stack([
        (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
    ], axis=1)
    target_camera = scene_camera[mask[rows, columns]]
    return scene_camera, target_camera, np.linalg.inv(extrinsic)


def _render_grasp_overlay(
    *, rgb_path: Path, depth_path: Path, mask_path: Path, calibration_path: Path,
    observation_id: str, target_ref: str,
    candidates: list[Mapping[str, Any]],
    compiler_identity: Mapping[str, Any] = PANDA_GRIP_SITE_COMPILER_IDENTITY,
    execution_tool: str = "move_free_space",
) -> tuple[Any, dict[str, Any]]:
    """Render the target-local contact sheet; the steps are in grasp_overlay.py."""

    source, mask, source_array = _tinted_target_source(rgb_path, mask_path)
    intrinsic, extrinsic = _overlay_camera_matrices(calibration_path)
    _scene_camera, target_camera, camera_to_world = _scene_and_target_points_camera(
        mask_path=mask_path,
        depth_path=depth_path,
        calibration_path=calibration_path,
    )
    target_world = _camera_points_to_world(target_camera, camera_to_world)
    colours = GRASP_CANDIDATE_COLOURS
    fonts = _overlay_fonts()
    target = _overlay_target(source, mask, source_array)
    compiled_records, projection_records = _grasp_overlay_candidate_records(
        candidates, colours, compiler_identity=compiler_identity,
        intrinsic=intrinsic, extrinsic=extrinsic,
    )
    image, draw = _new_contact_sheet(
        observation_id=observation_id, target_ref=target_ref, fonts=fonts,
    )
    target_crop = _draw_target_panel(image, draw, target, fonts)
    _draw_contact_sheet_legend(
        draw, compiler_identity=compiler_identity,
        execution_tool=execution_tool, fonts=fonts,
    )
    for index in range(3):
        _draw_candidate_card(
            image, draw, index=index, compiled_records=compiled_records,
            target=target, intrinsic=intrinsic, extrinsic=extrinsic,
            target_world=target_world, fonts=fonts,
        )

    return image, {
        "renderer": GRASP_PLAN_VISUALIZATION_ID,
        "source_resolution": [source.width, source.height],
        "target_crop_xyxy": list(target_crop),
        "render_resolution": [image.width, image.height],
        "target_mask_sha256": _file_hash(mask_path),
        "candidate_projections": projection_records,
        "candidate_panel_count": len(compiled_records),
        "policy_visible": True,
        "claim_boundary": (
            "target_local_geometry_preview_only; deterministic public RGB-D "
            "derivation; IK, reachability, contact, retention and task success "
            "are not established"
        ),
    }


def _grasp_overlay_candidate_records(
    candidates: list[Mapping[str, Any]], colours: Any, *,
    compiler_identity: Mapping[str, Any], intrinsic: Any, extrinsic: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile each drawable candidate and project its wireframe points."""

    projection_records: list[dict[str, Any]] = []
    compiled_records: list[dict[str, Any]] = []
    for candidate, colour in zip(candidates, colours, strict=False):
        geometry = _compiled_grasp_geometry(
            candidate, compiler_identity=compiler_identity,
        )
        shape = _gripper_wireframe(geometry, float(candidate["gripper_opening_m"]))
        world_points = list(shape["points"])
        source_pixels = [
            _project_world(point, intrinsic=intrinsic, extrinsic=extrinsic)
            for point in world_points
        ]
        compiled_records.append({
            "candidate": candidate,
            "colour": colour,
            "geometry": geometry,
            "wireframe": shape,
            "source_pixels": source_pixels,
        })
        projection_records.append({
            "candidate_id": candidate["candidate_id"],
            "rank": int(candidate["rank"]),
            "raw_origin_uv": list(source_pixels[4]) if source_pixels[4] else None,
            "contact_baseline_uv": list(source_pixels[5]) if source_pixels[5] else None,
            "grip_site_uv": list(source_pixels[6]) if source_pixels[6] else None,
            "pregrasp_uv": list(source_pixels[7]) if source_pixels[7] else None,
            "compiler_status": "returned",
        })
    return compiled_records, projection_records


def _camera_points_to_world(points: Any, camera_to_world: Any) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("grasp input point-cloud transform is malformed")
    return values @ transform[:3, :3].T + transform[:3, 3]


def _world_points_to_camera(points: Any, camera_to_world: Any) -> Any:
    import numpy as np

    transform = np.asarray(camera_to_world, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("grasp input point-cloud transform is malformed")
    return _camera_points_to_world(points, np.linalg.inv(transform))


def _render_grasp_global_review(
    *, camera_overlay: Any, camera_panel_resolution: list[int],
    scene_world: Any, target_world: Any, scene_source_index: Any,
    scene_sources: list[Mapping[str, Any]],
    workspace_bounds_m: Mapping[str, list[float]],
    observation_id: str, target_ref: str, camera: str,
    candidates: list[Mapping[str, Any]],
    compiler_identity: Mapping[str, Any] = PANDA_GRIP_SITE_COMPILER_IDENTITY,
    backend_input_scope: str = GRASP_BACKEND_SINGLE_VIEW_SCOPE,
    backend_cameras: Sequence[str] | None = None,
    backend_points_by_camera: Mapping[str, int] | None = None,
    backend_excluded_cameras: Sequence[Mapping[str, Any]] | None = None,
    observed_extent: Mapping[str, Any] | None = None,  # scene_extent_review.cell(...)
    policy_visible: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw

    scene = np.asarray(scene_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    source_index = np.asarray(scene_source_index, dtype=np.int16)
    if source_index.shape != (len(scene),):
        raise ValueError("grasp review source attribution is malformed")
    axes = {"x": 0, "y": 1, "z": 2}
    for axis in axes:
        bounds = workspace_bounds_m.get(axis)
        if (
            not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError("grasp review requires finite xyz workspace bounds")

    colours = GRASP_CANDIDATE_COLOURS
    panel_size = 360
    panel_margin = 22
    view_specs = (
        ("TOP", "x", "y"),
        ("FRONT", "x", "z"),
        ("SIDE", "y", "z"),
    )
    view_records: list[dict[str, Any]] = []

    def make_panel(title: str, horizontal: str, vertical: str) -> Any:
        horizontal_index = axes[horizontal]
        vertical_index = axes[vertical]
        horizontal_bounds = [float(value) for value in workspace_bounds_m[horizontal]]
        vertical_bounds = [float(value) for value in workspace_bounds_m[vertical]]
        lower_h, upper_h = horizontal_bounds
        lower_v, upper_v = vertical_bounds
        raster = np.full((panel_size, panel_size, 3), [28, 32, 40], dtype=np.uint8)

        def pixels(points: Any) -> tuple[Any, Any, Any]:
            values = np.asarray(points, dtype=np.float64)
            valid = (
                np.isfinite(values).all(axis=1)
                & (values[:, horizontal_index] >= lower_h)
                & (values[:, horizontal_index] <= upper_h)
                & (values[:, vertical_index] >= lower_v)
                & (values[:, vertical_index] <= upper_v)
            )
            selected = values[valid]
            if not len(selected):
                return np.empty(0, dtype=int), np.empty(0, dtype=int), valid
            x = np.rint(
                (selected[:, horizontal_index] - lower_h)
                / (upper_h - lower_h) * (panel_size - 1)
            ).astype(int)
            y = np.rint(
                (upper_v - selected[:, vertical_index])
                / (upper_v - lower_v) * (panel_size - 1)
            ).astype(int)
            return x, y, valid

        scene_x, scene_y, scene_valid = pixels(scene)
        retained_sources = source_index[scene_valid]
        source_colours = ((102, 110, 124), (151, 122, 232))
        for index, source_colour in enumerate(source_colours):
            selected = retained_sources == index
            raster[scene_y[selected], scene_x[selected]] = source_colour
        target_x, target_y, target_valid = pixels(target)
        for delta_x, delta_y in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            x = np.clip(target_x + delta_x, 0, panel_size - 1)
            y = np.clip(target_y + delta_y, 0, panel_size - 1)
            raster[y, x] = [32, 210, 190]
        image = Image.fromarray(raster, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(82, 90, 104))
        draw.text((10, 9), scene_extent_review.panel_title(observed_extent is not None, f"{title}  {horizontal.upper()}–{vertical.upper()}"), fill=(242, 244, 248))
        draw.text(
            (10, panel_size - 20),
            f"{horizontal} [{lower_h:+.2f},{upper_h:+.2f}]  "
            f"{vertical} [{lower_v:+.2f},{upper_v:+.2f}] m",
            fill=(178, 187, 200),
        )

        def one_point(point: Any) -> tuple[int, int] | None:
            x, y, valid = pixels(np.asarray(point, dtype=np.float64).reshape(1, 3))
            return (int(x[0]), int(y[0])) if bool(valid[0]) else None

        for candidate, colour in zip(candidates, colours, strict=False):
            geometry = _compiled_grasp_geometry(
                candidate, compiler_identity=compiler_identity,
            )
            center = np.asarray(geometry["grip_site"], dtype=np.float64)
            pregrasp = np.asarray(
                geometry["pregrasp_grip_site"], dtype=np.float64,
            )
            jaw = np.asarray(geometry["jaw_axis"], dtype=np.float64)
            # Draw the jaw at the compiled opening, capped by *this*
            # embodiment's admission limit. The literal 0.04 that used to sit
            # here is half the Panda maximum and was a wrong-embodiment
            # constant on every non-Panda panel.
            half = min(
                float(compiler_identity.get("maximum_opening_m", 0.08)),
                float(candidate["gripper_opening_m"]),
            ) / 2.0
            center_pixel = one_point(center)
            pregrasp_pixel = one_point(pregrasp)
            jaw_pixels = (one_point(center - jaw * half), one_point(center + jaw * half))
            if jaw_pixels[0] is not None and jaw_pixels[1] is not None:
                draw.line(jaw_pixels, fill=colour, width=4)
            if center_pixel is not None:
                x, y = center_pixel
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=colour, width=2)
                draw.text((x + 7, y + 2), f"G{candidate['rank']}", fill=colour)
            if pregrasp_pixel is not None:
                x, y = pregrasp_pixel
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=colour, width=2)
            if center_pixel is not None and pregrasp_pixel is not None:
                draw.line((pregrasp_pixel, center_pixel), fill=colour, width=2)

        view_records.append({
            "view": title.lower(),
            "horizontal_axis": horizontal,
            "vertical_axis": vertical,
            "horizontal_bounds_m": horizontal_bounds,
            "vertical_bounds_m": vertical_bounds,
            "scene_points_in_bounds": int(np.count_nonzero(scene_valid)),
            "target_points_in_bounds": int(np.count_nonzero(target_valid)),
            "scene_points_by_source": {
                str(source["camera"]): int(np.count_nonzero(
                    retained_sources == int(source["source_index"])
                ))
                for source in scene_sources
            },
        })
        return image

    cell_size = 384
    left_width = cell_size * 2
    legend_width = 400
    canvas = Image.new("RGB", (left_width + legend_width, cell_size * 2), (16, 19, 25))
    draw = ImageDraw.Draw(canvas)
    source_width, source_height = camera_panel_resolution
    source_panel = camera_overlay.crop((0, 0, source_width, source_height)).convert("RGB")
    source_panel.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    camera_cell = Image.new("RGB", (panel_size, panel_size), (28, 32, 40))
    camera_cell.paste(
        source_panel,
        ((panel_size - source_panel.width) // 2, (panel_size - source_panel.height) // 2),
    )
    camera_draw = ImageDraw.Draw(camera_cell)
    camera_draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(82, 90, 104))
    camera_draw.text(
        (10, 9), f"POLICY CONTACT SHEET  {camera}", fill=(242, 244, 248),
    )
    panels = [camera_cell if observed_extent is None else observed_extent["image"]] + [
        make_panel(*spec) for spec in view_specs]
    for index, panel in enumerate(panels):
        column = index % 2
        row = index // 2
        canvas.paste(
            panel,
            (column * cell_size + panel_margin // 2,
             row * cell_size + panel_margin // 2),
        )

    left = left_width + 18
    draw.text((left, 18), "GRASP BACKEND + GLOBAL REVIEW", fill=(242, 244, 248))
    draw.text((left, 44), f"observation  {observation_id[-34:]}", fill=(165, 174, 190))
    draw.text((left, 62), f"target       {target_ref}", fill=(165, 174, 190))
    draw.text((left, 92), "REVIEW SCOPE", fill=(255, 176, 64))
    review_caption, backend_caption = scene_extent_review.captions(
        observed_extent is not None, scene_sources=scene_sources,
        backend_cameras=backend_cameras, camera=camera)
    draw.text((left, 112), review_caption, fill=(225, 229, 236))
    draw.text((left, 132), backend_caption, fill=(255, 176, 64))
    draw.text((left, 158), f"scene points   {len(scene):,}", fill=(192, 200, 212))
    draw.text((left, 178), f"target points  {len(target):,}", fill=(32, 210, 190))
    source_top = 198
    source_colours = ((102, 110, 124), (151, 122, 232))
    for source, source_colour in zip(scene_sources, source_colours, strict=False):
        retained_count = int(source.get("retained_voxel_points", 0))
        draw.text(
            (left, source_top),
            f"{source['camera']:<10} {retained_count:,} retained",
            fill=source_colour,
        )
        source_top += 18
    draw.text((left, source_top + 12), "CANDIDATES", fill=(242, 244, 248))
    top = source_top + 38
    for candidate, colour in zip(candidates, colours, strict=False):
        draw.rectangle((left, top + 2, left + 10, top + 12), fill=colour)
        draw.text(
            (left + 18, top),
            f"G{candidate['rank']}  score {float(candidate['backend_score']):.3f}  "
            f"opening {float(candidate['gripper_opening_m']) * 1000:.1f} mm",
            fill=(235, 238, 243),
        )
        top += 28
    warning_top = canvas.height - 122
    draw.line((left, warning_top, canvas.width - 18, warning_top), fill=(68, 75, 89))
    draw.text((left, warning_top + 12), "REVIEW-ONLY DERIVATION", fill=(80, 210, 255))
    legend_names = [str(source["camera"]) for source in scene_sources]
    draw.text(
        (left, warning_top + 34),
        " · ".join(
            f"{shade} {name} review input"
            for shade, name in zip(("gray", "purple"), legend_names, strict=False)
        ) or "no review scene input",
        fill=(192, 200, 212),
    )
    draw.text((left, warning_top + 54), "cyan: target-mask input points", fill=(32, 210, 190))
    draw.text((left, warning_top + 78), "missing surfaces remain missing", fill=(255, 176, 64))
    draw.text((left, warning_top + 98), "not IK, collision, contact or retention proof", fill=(255, 176, 64))
    resolved_backend_cameras = (
        [str(source["camera"]) for source in scene_sources]
        if backend_cameras is None else [str(name) for name in backend_cameras]
    )
    return canvas, {
        "renderer": GRASP_PLAN_REVIEW_ID,
        "review_scope": "synchronized_agentview_wrist_rgbd_fusion",
        # Derived, never asserted: this field named a single view even after
        # the caller began fusing, so it now restates what the proposer got.
        "backend_input_scope": backend_input_scope,
        "backend_cameras": resolved_backend_cameras,
        "backend_points_by_camera": (
            {} if backend_points_by_camera is None
            else {str(key): int(value) for key, value in backend_points_by_camera.items()}
        ),
        "backend_excluded_cameras": [
            dict(item) for item in (backend_excluded_cameras or ())
        ],
        "target_source_camera": camera,
        "scene_cameras": [str(source["camera"]) for source in scene_sources],
        "scene_points_by_source": {
            str(source["camera"]): int(source.get("retained_voxel_points", 0))
            for source in scene_sources
        },
        "scene_point_count": int(len(scene)),
        "target_point_count": int(len(target)),
        "views": view_records,
        **({} if observed_extent is None else {"observed_extent": observed_extent["record"]}),
        "render_resolution": [canvas.width, canvas.height],
        # True where perception_feedback appends this image to the result.
        "policy_visible": policy_visible,
        "claim_boundary": (
            "review_only fused public-camera projection; the proposer received "
            + ", ".join(resolved_backend_cameras)
            + "; the target-mask subset stays single-view from "
            + camera
            + "; surfaces missing from every contributing camera remain missing; "
            "not IK, collision, contact or retention evidence"
        ),
    }


def _camera_rgbd_admission(
    *, observation_root: Path, observation: Mapping[str, Any], camera_name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Classify one camera's RGB-D source for the fused proposer input.

    Returns ``(reason, record)``.  A non-``None`` reason is drawn from
    ``GRASP_BACKEND_EXCLUSION_REASONS`` so an excluded camera is always named
    in evidence instead of disappearing from the fusion.
    """

    import numpy as np

    record = observation.get("cameras", {}).get(camera_name)
    if not isinstance(record, Mapping):
        return "camera_absent_from_observation", None
    files = record.get("files")
    if not isinstance(files, Mapping):
        return "rgbd_source_reference_missing", None
    resolved_root = observation_root.resolve()
    references: dict[str, dict[str, Any]] = {}
    for role in ("rgb_lossless_png", "depth_numpy_m_f32", "calibration"):
        reference = files.get(role)
        relative = reference.get("path") if isinstance(reference, Mapping) else None
        if not isinstance(relative, str):
            return "rgbd_source_reference_missing", None
        path = (observation_root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(resolved_root):
            return "rgbd_source_file_unavailable", None
        digest = _file_hash(path)
        if digest != reference.get("sha256"):
            return "rgbd_source_hash_drift", None
        references[role] = {"path": relative, "sha256": digest, "resolved": path}
    try:
        calibration = json.loads(
            references["calibration"]["resolved"].read_text(encoding="utf-8")
        )
        intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
        if (
            intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or abs(float(np.linalg.det(extrinsic))) < 1e-9
        ):
            return "rgbd_calibration_invalid", None
        camera_to_world = np.linalg.inv(extrinsic)
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return "rgbd_calibration_invalid", None
    return None, {
        "camera": camera_name,
        # Per-camera calibration travels with the fused cloud: a reader can
        # reproject any fused point without reopening the observation.
        "calibration": {
            "path": references["calibration"]["path"],
            "sha256": references["calibration"]["sha256"],
            "convention": "opencv_camera_optical",
            "intrinsic_cv": [[float(value) for value in row] for row in intrinsic],
            "extrinsic_cv_world_to_camera": [
                [float(value) for value in row] for row in extrinsic
            ],
            "camera_to_world": [
                [float(value) for value in row] for row in camera_to_world
            ],
        },
        "files": {
            role: {"path": value["path"], "sha256": value["sha256"]}
            for role, value in references.items()
        },
    }


def _admit_backend_scene_cameras(
    *, observation_root: Path, observation: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]],
    target_source_camera: str, requested_cameras: Sequence[str],
    voxel_size_m: float, max_depth_m: float,
) -> dict[str, Any]:
    """Fuse every usable requested camera into one proposer input cloud.

    The target source camera is always first, so it wins voxel ties and the
    fused cloud stays consistent with the single-camera target mask.  Every
    other camera either contributes or is excluded with a named reason; the
    target source camera failing is a hard error, exactly as before.
    """

    seen: set[str] = set()
    candidates: list[str] = []
    excluded: list[dict[str, Any]] = []
    for name in [target_source_camera, *requested_cameras]:
        text = str(name)
        if text in seen:
            if text != target_source_camera:
                excluded.append({
                    "camera": text, "reason": "duplicate_camera_request",
                })
            continue
        seen.add(text)
        candidates.append(text)

    calibrations: dict[str, Any] = {}
    source_files: dict[str, Any] = {}
    admitted: list[str] = []
    for name in candidates:
        reason, record = _camera_rgbd_admission(
            observation_root=observation_root, observation=observation,
            camera_name=name,
        )
        if reason is None and len(candidates) > 1:
            # A camera only earns a place in the fused geometry once its own
            # depth actually lands inside the workspace.
            try:
                probe = fuse_rgbd_pointcloud_sources(
                    observation_root=observation_root, observation=observation,
                    workspace_bounds_m=workspace_bounds_m, camera_names=(name,),
                    voxel_size_m=voxel_size_m, max_depth_m=max_depth_m,
                )
            except (ValueError, TypeError, KeyError, OSError, EOFError):
                reason = "rgbd_depth_unusable"
            else:
                if int(probe["sources"][0].get("workspace_points", 0)) < (
                    GRASP_BACKEND_MINIMUM_CAMERA_POINTS
                ):
                    reason = "insufficient_workspace_depth_points"
        if reason is not None:
            if name == target_source_camera:
                raise ValueError(
                    "grasp backend input requires the target source camera "
                    f"{name}: {reason}"
                )
            excluded.append({"camera": name, "reason": reason})
            continue
        admitted.append(name)
        calibrations[name] = record["calibration"]
        source_files[name] = record["files"]

    fused = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root, observation=observation,
        workspace_bounds_m=workspace_bounds_m, camera_names=tuple(admitted),
        voxel_size_m=voxel_size_m, max_depth_m=max_depth_m,
    )
    return {
        "fused": fused,
        "requested_cameras": [str(name) for name in requested_cameras],
        "contributing_cameras": admitted,
        "excluded_cameras": excluded,
        "calibrations": calibrations,
        "source_files": source_files,
    }


def materialize_grasp_query(
    *, artifact_root: Path, observation_id: str, target_ref: str,
    observation_root: Path, observation: Mapping[str, Any],
    rgb_path: Path, depth_path: Path, calibration_path: Path,
    workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
    backend: GraspProposalBackend, compile_for_execution: bool = False,
    scene_cameras: Sequence[str] = GRASP_SCENE_CAMERAS,
    backend_scene_cameras: Sequence[str] | None = None,
    compiler_id: str | None = None,
    execution_tool: str = "move_free_space",
    public_contact_pose: bool = False,
    review_source_binding: Mapping[str, Any] | None = None,
    candidate_selection: Mapping[str, Any] | None = None,  # see grasp_selection.py
    observed_extent: bool = False,  # scene_extent_review.CAPABILITY
    review_policy_visible: bool = False,  # the surface delivers the review image
    lift_m: float | None = None,  # skill_geometry.split_liftable
) -> tuple[dict[str, Any], Path]:
    """Bind, reuse or materialize one content-addressed grasp query.

    The steps live in grasp_query.py, which imports this module's helpers.
    """

    from .grasp_query import (
        _grasp_backend_input, _grasp_backend_record,
        _grasp_input_pointcloud_record, _grasp_query_identity,
        _grasp_query_manifest, _grasp_query_points, _propose_grasp_candidates,
        _registered_grasp_compiler, _verified_grasp_query,
        _write_grasp_backend_output, _write_grasp_global_review,
        _write_grasp_input_pointcloud, _write_grasp_overlay,
    )

    target, mask_path, identity, actual_source = _current_grasp_target(
        artifact_root=artifact_root, observation_id=observation_id,
        target_ref=target_ref, rgb_path=rgb_path, depth_path=depth_path,
        calibration_path=calibration_path,
    )
    backend_record = _grasp_backend_record(backend)
    compiler = _registered_grasp_compiler(compiler_id)
    target_source_camera = str(identity["camera"])
    proposer_input = _grasp_backend_input(
        observation_root=observation_root, observation=observation,
        workspace_bounds_m=workspace_bounds_m,
        target_source_camera=target_source_camera,
        backend_scene_cameras=backend_scene_cameras,
    )
    query_identity = _grasp_query_identity(
        observation_id=observation_id, target_ref=target_ref, identity=identity,
        actual_source=actual_source, backend_input_profile=proposer_input.profile,
        workspace_bounds_m=workspace_bounds_m, max_candidates=max_candidates,
        backend_record=backend_record, compiler=compiler,
        compile_for_execution=compile_for_execution,
        execution_tool=execution_tool, target=target,
        public_contact_pose=public_contact_pose,
        review_source_binding=review_source_binding,
        candidate_selection=candidate_selection, observed_extent=observed_extent,
    )
    query_ref = "grasp-query-" + _canonical_hash(query_identity)[:24]
    query_root = artifact_root / "grasp-queries" / query_ref
    manifest_path = query_root / "manifest.json"
    if manifest_path.is_file():
        return _verified_grasp_query(query_root, manifest_path, query_identity)

    points = _grasp_query_points(
        mask_path=mask_path, depth_path=depth_path,
        calibration_path=calibration_path, backend_input=proposer_input.fused,
    )
    proposals = _propose_grasp_candidates(
        backend=backend, points=points, target_ref=target_ref,
        workspace_bounds_m=workspace_bounds_m, max_candidates=max_candidates,
        candidate_selection=candidate_selection, compiler=compiler,
        compile_for_execution=compile_for_execution,
        public_contact_pose=public_contact_pose, lift_m=lift_m,
    )
    target_world = _camera_points_to_world(points.target_camera, points.camera_to_world)
    query_root.mkdir(parents=True, exist_ok=True)
    input_pointcloud_path = _write_grasp_input_pointcloud(query_root, points, target_world)
    overlay_path, overlay, visualization = _write_grasp_overlay(
        query_root=query_root, rgb_path=rgb_path, depth_path=depth_path,
        mask_path=mask_path, calibration_path=calibration_path,
        observation_id=observation_id, target_ref=target_ref,
        candidates=proposals.candidates, compiler=compiler,
        execution_tool=execution_tool,
    )
    review_visualization, review_pointcloud = _write_grasp_global_review(
        query_root=query_root, observation_root=observation_root,
        observation=observation, workspace_bounds_m=workspace_bounds_m,
        scene_cameras=scene_cameras, camera_to_world=points.camera_to_world,
        target_world=target_world, overlay=overlay,
        observation_id=observation_id, target_ref=target_ref,
        target_source_camera=target_source_camera,
        candidates=proposals.candidates, compiler=compiler,
        proposer_input=proposer_input, observed_extent=observed_extent,
        review_policy_visible=review_policy_visible,
    )
    backend_output_record = _write_grasp_backend_output(
        query_root, proposals.backend_output, proposals.input_regime,
    )
    manifest = _grasp_query_manifest(
        query_ref=query_ref, query_identity=query_identity,
        backend_record=backend_record, proposals=proposals,
        backend_output_record=backend_output_record,
        visualization=visualization,
        review_visualization=review_visualization,
        review_pointcloud=review_pointcloud,
        input_pointcloud=_grasp_input_pointcloud_record(
            query_root=query_root, input_pointcloud_path=input_pointcloud_path,
            identity=identity, target_source_camera=target_source_camera,
            proposer_input=proposer_input, points=points,
            target_world=target_world,
        ),
        overlay=_artifact(overlay_path, query_root),
    )
    _atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return manifest, overlay_path


def _current_grasp_target(
    *, artifact_root: Path, observation_id: str, target_ref: str,
    rgb_path: Path, depth_path: Path, calibration_path: Path,
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any], dict[str, Any]]:
    """Load the target and refuse it unless it was segmented from this RGB-D."""

    target, mask_path, _target_overlay = load_segmented_target(
        artifact_root=artifact_root, target_ref=target_ref,
    )
    identity = target.get("identity", {})
    if identity.get("observation_id") != observation_id:
        raise ValueError("target_ref is stale for the current observation")
    source = identity.get("source", {})
    actual_source = _segmentation_source_identity(
        observation_id=observation_id, camera=identity.get("camera", ""),
        rgb_path=rgb_path, depth_path=depth_path,
        calibration_path=calibration_path,
    )
    legacy_source = {
        "rgb_sha256": actual_source["files"]["rgb"]["sha256"],
        "depth_sha256": actual_source["files"]["depth"]["sha256"],
        "calibration_sha256": actual_source["files"]["calibration"]["sha256"],
    }
    if source != actual_source and source != legacy_source:
        raise ValueError("target_ref source artifacts do not match the current observation")
    return target, mask_path, identity, actual_source


def grasp_public_result(
    manifest: Mapping[str, Any], overlay_path: Path,
) -> dict[str, Any]:
    from PIL import Image

    with Image.open(overlay_path) as image:
        width, height = image.size
    identity = manifest["identity"]
    return {
        "success": True,
        "status": manifest["status"],
        "observation_id": identity["observation_id"],
        "target_ref": identity["target_ref"],
        "camera": identity["camera"],
        "query_ref": manifest["query_ref"],
        "backend": manifest["backend"],
        "candidates": manifest["candidates"],
        "diagnostics": manifest["diagnostics"],
        "overlay": {
            "sha256": manifest["overlay"]["sha256"],
            "width": width,
            "height": height,
        },
    }


#: Surfaces declaring this receive the target's full measured principal frame.
#: Older and frozen surfaces keep the single axis they were written against;
#: their result schemas forbid unknown properties, and the producer is shared,
#: so the extra measurements are removed on the way out rather than never taken.
TARGET_PRINCIPAL_FRAME_CAPABILITY = "target_principal_frame_v1"

#: Measured by :func:`_principal_frame`, delivered only under the capability.
_EXTENDED_FRAME_KEYS = (
    "principal_axes_world", "principal_extents_m", "largest_extent_m", "middle_extent_m",
    "smallest_extent_m", "largest_extent_axis_world", "middle_extent_axis_world", "smallest_extent_axis_world",
)


def project_target_frame(result: Any, *, declared: bool, support_declared: bool = False,
                         footprint_declared: bool = False) -> Any:
    """Keep or drop the extended principal frame, the support height and the footprint on a policy result.

    The live producers measure all three into the artifact; each surface receives only what it
    declares (``target_principal_frame_v1``, ``target_support_height_v1``,
    ``overhead_footprint_localization_v1``), so one producer serves surfaces written before any
    existed. The frozen v1 batch producer measures none.
    """

    if not isinstance(result, Mapping):
        return result
    keys = ((() if declared else _EXTENDED_FRAME_KEYS) + (() if support_declared else target_support.KEYS)
            + (() if footprint_declared else (*overhead_view.KEYS, overhead_view.POSE_KEY)))
    candidates = result.get("candidates") or ()
    holders = [result, *(item for item in candidates if isinstance(item, Mapping))]
    for holder in holders:
        localization = holder.get("localization")
        if isinstance(localization, dict):
            for key in keys:
                localization.pop(key, None)
    return result


def grounding_public_result(
    manifest: Mapping[str, Any], overlay_path: Path,
) -> dict[str, Any]:
    from PIL import Image

    with Image.open(overlay_path) as image:
        width, height = image.size
    identity = manifest["identity"]
    candidate_previews = []
    for image_index, preview in enumerate(
        manifest["visualization"]["candidate_previews"], start=1,
    ):
        candidate_previews.append({
            "image_index": image_index,
            "rank": preview["rank"],
            "target_ref": preview["target_ref"],
            "sha256": preview["artifact"]["sha256"],
            "source_dimensions": preview["source_dimensions"],
            "source_bbox_xyxy": preview["source_bbox_xyxy"],
            "crop_xyxy": preview["crop_xyxy"],
            "render_dimensions": preview["render_dimensions"],
            "resampling": preview["resampling"],
        })
    return {
        "success": True,
        "status": manifest["status"],
        "observation_id": identity["observation_id"],
        "camera": identity["camera"],
        "text": identity["text"],
        "grounding_ref": manifest["grounding_ref"],
        "backend": manifest["backend"],
        "candidates": manifest["candidates"],
        "diagnostics": manifest["diagnostics"],
        "candidate_previews": candidate_previews,
        "overlay": {
            "sha256": manifest["overlay"]["sha256"],
            "width": width,
            "height": height,
            "image_index": 0,
        },
    }


def _compiled_grasp_geometry(
    candidate: Mapping[str, Any], *,
    compiler_identity: Mapping[str, Any] = PANDA_GRIP_SITE_COMPILER_IDENTITY,
) -> dict[str, Any]:
    """Resolve one Contact-GraspNet source frame into one controlled point.

    The Panda grip-site mapping stays single-sourced in the shared grasp
    compiler; registered non-Panda embodiments compile through the
    identity-parametrized mapping below.
    """

    import numpy as np

    if compiler_identity["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]:
        geometry = dict(_compiled_grasp_geometry_impl(candidate))
        # The shared Panda compiler names the jaw axis x; renderers consume
        # the embodiment-neutral jaw_axis alias.
        geometry["jaw_axis"] = geometry["x_axis"]
        return geometry

    convention = candidate.get("pose_convention")
    if convention == compiler_identity["source_pose_convention"]:
        source_origin = np.asarray(
            candidate["source_gripper_origin_xyz_m"], dtype=np.float64,
        )
        source_pregrasp = np.asarray(
            candidate["source_pregrasp_origin_xyz_m"], dtype=np.float64,
        )
        direct_grip_site = direct_pregrasp_grip_site = None
    elif convention == "pca_topdown_parallel_jaw_center_v1":
        direct_grip_site = np.asarray(
            candidate["grasp_center_xyz_m"], dtype=np.float64,
        )
        direct_pregrasp_grip_site = np.asarray(
            candidate["pregrasp_center_xyz_m"], dtype=np.float64,
        )
        source_origin = source_pregrasp = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError("grasp candidate pose convention is not compiler-qualified")
    approach = np.asarray(candidate["approach_world"], dtype=np.float64)
    jaw = np.asarray(candidate["jaw_world"], dtype=np.float64)
    if any(
        value.shape != (3,)
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate vectors must each contain three values")
    if not all(
        np.isfinite(value).all()
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate contains non-finite geometry")
    approach_norm = float(np.linalg.norm(approach))
    if not math.isfinite(approach_norm) or approach_norm <= 1e-6:
        raise ValueError("grasp approach axis is degenerate")
    approach_axis = approach / approach_norm
    jaw_axis = jaw - approach_axis * float(np.dot(jaw, approach_axis))
    jaw_norm = float(np.linalg.norm(jaw_axis))
    if not math.isfinite(jaw_norm) or jaw_norm <= 1e-6:
        raise ValueError("grasp jaw and approach axes are degenerate")
    jaw_axis /= jaw_norm
    lateral_axis = np.cross(approach_axis, jaw_axis)
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if not math.isfinite(lateral_norm) or lateral_norm <= 1e-6:
        raise ValueError("grasp axes cannot define a lateral direction")
    lateral_axis /= lateral_norm
    aloha = compiler_identity.get("controlled_point") == "end_effector"
    if aloha:
        x_axis, y_axis, z_axis = approach_axis, jaw_axis, lateral_axis
    else:
        x_axis, y_axis, z_axis = jaw_axis, lateral_axis, approach_axis
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(rotation)) < 0.999:
        raise ValueError("grasp axes do not form a right-handed orientation")
    source_to_grip_site = float(
        compiler_identity["source_origin_to_controlled_point_m"]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        source_origin = direct_grip_site - approach_axis * source_to_grip_site
        source_pregrasp = (
            direct_pregrasp_grip_site - approach_axis * source_to_grip_site
        )
    approach_distance = float(np.dot(
        source_origin - source_pregrasp, approach_axis,
    ))
    lateral_error = float(np.linalg.norm(
        (source_origin - source_pregrasp) - approach_axis * approach_distance,
    ))
    if (
        not math.isfinite(approach_distance)
        or not 0.005 - 1e-9 <= approach_distance <= 0.10 + 1e-9
        or lateral_error > 1e-4
    ):
        raise ValueError("grasp pregrasp is not one bounded axial servo away")
    gripper_depth = float(
        compiler_identity["contact_graspnet_gripper_depth_m"]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        grip_site = direct_grip_site
        pregrasp_grip_site = direct_pregrasp_grip_site
    else:
        grip_site = source_origin + approach_axis * source_to_grip_site
        pregrasp_grip_site = grip_site - approach_axis * approach_distance
    contact_baseline = source_origin + approach_axis * gripper_depth
    opening = float(candidate["gripper_opening_m"])
    contact_negative_jaw = contact_baseline - jaw_axis * opening / 2.0
    contact_positive_jaw = contact_baseline + jaw_axis * opening / 2.0
    return {
        "source_gripper_origin": source_origin,
        "source_pregrasp_origin": source_pregrasp,
        "grip_site": grip_site,
        "pregrasp_grip_site": pregrasp_grip_site,
        "contact_baseline": contact_baseline,
        "contact_negative_jaw": contact_negative_jaw,
        "contact_positive_jaw": contact_positive_jaw,
        "approach_distance": approach_distance,
        "rotation": rotation,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "jaw_axis": jaw_axis,
    }


def _compile_grasp_candidate(
    candidate: Mapping[str, Any], *,
    compiler_identity: Mapping[str, Any] = PANDA_GRIP_SITE_COMPILER_IDENTITY,
    public_contact_pose: bool = False,
) -> dict[str, Any]:
    if compiler_identity["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]:
        return _compile_grasp_candidate_impl(candidate)
    geometry = _compiled_grasp_geometry(
        candidate, compiler_identity=compiler_identity,
    )
    maximum_approach = float(
        compiler_identity.get("maximum_approach_m", 0.05)
    )
    if (
        not math.isfinite(maximum_approach)
        or not 0.005 <= maximum_approach <= 0.10
    ):
        raise ValueError("grasp compiler approach limit is invalid")
    if float(geometry["approach_distance"]) > maximum_approach + 1e-9:
        raise ValueError("grasp approach exceeds bounded servo distance")
    opening = float(candidate["gripper_opening_m"])
    maximum_opening = float(compiler_identity.get("maximum_opening_m", 0.08))
    if opening > maximum_opening:
        raise ValueError("grasp opening exceeds embodiment limit")
    aloha = compiler_identity.get("controlled_point") == "end_effector"
    orientation_xyzw = _rotation_matrix_to_quaternion_xyzw(
        geometry["rotation"],
    )
    contact_pose = (
        # Same convention as the Panda compiler: the controlled point's
        # world pose at contact, closing the residual check against the
        # measured end-effector after the approach.
        {
            "contact_pose": {
                "position_m": [
                    float(item) for item in geometry["grip_site"]
                ],
                "orientation_xyzw": orientation_xyzw,
            },
        }
        if public_contact_pose else {}
    )
    # Only identities that declare the flag add the field, so frozen ALOHA
    # candidate records keep their exact shape.
    aperture_calibration = (
        {}
        if "physical_jaw_aperture_calibrated" not in compiler_identity
        else {
            "aperture_calibration": (
                "calibrated"
                if compiler_identity["physical_jaw_aperture_calibrated"]
                else str(compiler_identity["aperture_caveat"])
            ),
        }
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": int(candidate["rank"]),
        "pregrasp_pose": {
            "position_m": [
                float(item) for item in geometry["pregrasp_grip_site"]
            ],
            "orientation_xyzw": orientation_xyzw,
        },
        **contact_pose,
        "approach_delta": {
            "frame": "end_effector" if aloha else "grip_site",
            "translation_m": [
                float(geometry["approach_distance"]) if aloha else 0.0,
                0.0,
                0.0 if aloha else float(geometry["approach_distance"]),
            ],
            "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        },
        "gripper_opening_m": opening,
        "backend_score": float(candidate["backend_score"]),
        "compiler_checks": {
            "finite_geometry": "passed",
            "right_handed_controlled_point_axes": "passed",
            "bounded_local_approach": "passed",
            "source_origin_to_controlled_point": "passed",
            "embodiment_mapping": compiler_identity["id"],
            "workspace": candidate["checks"]["workspace"],
            "aperture": candidate["checks"]["aperture"],
            **aperture_calibration,
            "reachability": "delegated_to_move_free_space",
        },
    }


def _prepare_compiled_grasp_candidates(
    candidates: list[dict[str, Any]], *, compile_for_execution: bool,
    compiler_identity: Mapping[str, Any] = PANDA_GRIP_SITE_COMPILER_IDENTITY,
    public_contact_pose: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if compiler_identity["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]:
        return _prepare_compiled_grasp_candidates_impl(
            candidates, compile_for_execution=compile_for_execution,
        )
    return _prepare_compiled_grasp_candidates_impl(
        candidates, compile_for_execution=compile_for_execution,
        compile_candidate=lambda raw: _compile_grasp_candidate(
            raw, compiler_identity=compiler_identity,
            public_contact_pose=public_contact_pose,
        ),
        check_geometry=lambda raw: _compiled_grasp_geometry(
            raw, compiler_identity=compiler_identity,
        ),
    )


def grasp_plan_public_result(
    manifest: Mapping[str, Any], overlay_path: Path,
) -> dict[str, Any]:
    base = grasp_public_result(manifest, overlay_path)
    compiler_identity = manifest.get("identity", {}).get("compiler")
    if (
        not isinstance(compiler_identity, Mapping)
        or compiler_identity.get("id") not in GRASP_COMPILER_IDENTITIES
    ):
        raise ValueError("grasp plan compiler identity is unavailable")
    compiled = [dict(item) for item in manifest["compiled_candidates"]]
    diagnostics = dict(base["diagnostics"])
    diagnostics["compiler_returned_candidate_count"] = len(compiled)
    return {
        "success": True,
        "status": "candidates" if compiled else "no_candidate",
        "observation_id": base["observation_id"],
        "target_ref": base["target_ref"],
        "camera": base["camera"],
        "grasp_query_ref": base["query_ref"],
        "backend": base["backend"],
        # The Panda grasp_plan_v1 result schema publishes the complete
        # registered identity; the dual-arm surfaces' frozen result schemas
        # pin the restricted projection without reconstruction constants.
        "compiler": (
            dict(compiler_identity)
            if compiler_identity["id"] == PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
            else _public_grasp_compiler_identity(compiler_identity)
        ),
        "candidates": compiled,
        "diagnostics": diagnostics,
        "overlay": base["overlay"],
    }
