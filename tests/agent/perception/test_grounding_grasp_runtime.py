"""Shared grounding/grasp Runtime contracts."""

from __future__ import annotations

import hashlib
import base64
import copy
import io
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ora.agent.composition import resolve_tool_surface
from ora.agent.context import ContextProjector
from ora.agent.model_services.identity import grounding_grasp_adapter_sha256
from ora.agent.tools.runtime import core as runtime_core
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.grasp import (
    CONTACT_GRASPNET_MODEL_IDENTITY,
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    SAM3_MODEL_IDENTITY,
    ContactGraspNetApiProposer,
    Sam3ApiSegmenter,
    _compile_grasp_candidate,
    _compiled_grasp_geometry,
)
from ora.agent.tools.geometry import fuse_rgbd_pointcloud_sources
from ora.agent.tools.robot import ToolError
from ora.environment import SyntheticSession
from viewer.trajectory import compile_trajectory


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
# The maintained single-arm LIBERO-Pro and dual-arm RoboTwin defaults.
LIBERO_SURFACE = "libero_placement_direct_overhead_v1"
ROBOTWIN_SURFACE = "robotwin_toolchain_b2_closed_noservo_v2"


def test_model_services_are_shared_and_experiment_free() -> None:
    shared = ROOT / "ora/agent/model_services"
    assert shared.is_dir()
    assert "experiments." not in (
        shared / "resident.py"
    ).read_text(encoding="utf-8")
    assert "experiments." not in (
        shared / "grounding_grasp.py"
    ).read_text(encoding="utf-8")
    identity_source = (shared / "identity.py").read_text(encoding="utf-8")
    for name in ("identity.py", "resident.py", "grounding_grasp.py"):
        assert f'"{name}"' in identity_source
    assert len(grounding_grasp_adapter_sha256()) == 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_object_observation(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "environment", "grasp-plan", 0, max_actions=4,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    observation_root = source / observation["observation_id"]
    camera_root = observation_root / "agentview"
    rgb_path = camera_root / "rgb.png"
    depth_path = camera_root / "depth-m.npy"
    calibration_path = camera_root / "calibration.json"
    rgb = np.full((64, 64, 3), [48, 52, 60], dtype=np.uint8)
    rgb[18:44, 22:40] = [210, 48, 32]
    rgb[22:42, 44:58] = [48, 80, 220]
    Image.fromarray(rgb).save(rgb_path, format="PNG")
    depth = np.full((64, 64), 1.2, dtype=np.float32)
    depth[18:44, 22:40] = 0.8
    depth[22:42, 44:58] = 0.85
    np.save(depth_path, depth, allow_pickle=False)
    calibration_path.write_text(json.dumps({
        "intrinsic_cv": [
            [400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0],
        ],
        "extrinsic_cv": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "cam2world_gl": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "depth_unit": "metres",
    }, sort_keys=True), encoding="utf-8")
    record = observation["cameras"]["agentview"]
    record["width"] = 64
    record["height"] = 64
    for role, path in {
        "rgb_lossless_png": rgb_path,
        "depth_numpy_m_f32": depth_path,
        "calibration": calibration_path,
    }.items():
        record["files"][role] = {
            "path": path.relative_to(observation_root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    wrist_root = observation_root / "wrist"
    wrist_rgb_path = wrist_root / "rgb.png"
    wrist_depth_path = wrist_root / "depth-m.npy"
    wrist_calibration_path = wrist_root / "calibration.json"
    wrist_rgb = np.full((64, 64, 3), [76, 58, 112], dtype=np.uint8)
    Image.fromarray(wrist_rgb).save(wrist_rgb_path, format="PNG")
    wrist_depth = np.full((64, 64), 1.2, dtype=np.float32)
    wrist_depth[6:16, 6:16] = 0.95
    np.save(wrist_depth_path, wrist_depth, allow_pickle=False)
    wrist_calibration_path.write_text(
        calibration_path.read_text(encoding="utf-8"), encoding="utf-8",
    )
    wrist_record = observation["cameras"]["wrist"]
    wrist_record["width"] = 64
    wrist_record["height"] = 64
    for role, path in {
        "rgb_lossless_png": wrist_rgb_path,
        "depth_numpy_m_f32": wrist_depth_path,
        "calibration": wrist_calibration_path,
    }.items():
        wrist_record["files"][role] = {
            "path": path.relative_to(observation_root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    (observation_root / "observation.json").write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    control = tmp_path / "control"
    ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, PROJECT,
        expose_manual=False, main_camera="agentview",
    ).publish(
        observation=observation,
        observation_root=observation_root,
        task_instruction="pick the red object",
        remaining_budget={"robot_decisions": 4, "model_calls": 12},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    return control, observation["observation_id"]


class FakeSam3:
    backend_id = "sam3_text_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"fixture": "two_text_candidates"}

    def require_ready(self) -> dict[str, object]:
        return {"ready": True, "load_count": 1}

    def segment(self, **_kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        mask = np.zeros((64, 64), dtype=bool)
        mask[18:44, 22:40] = True
        return mask, {"fixture": "point"}

    def ground(
        self, *, rgb_u8: object, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert np.asarray(rgb_u8).shape == (64, 64, 3)
        assert text == "red object"
        first = np.zeros((64, 64), dtype=bool)
        first[18:44, 22:40] = True
        second = np.zeros((64, 64), dtype=bool)
        second[22:42, 44:58] = True
        values = [
            {"source_index": 0, "rank": 0, "mask": first,
             "backend_score": 0.82, "bbox_xyxy": [22, 18, 40, 44]},
            {"source_index": 1, "rank": 1, "mask": second,
             "backend_score": 0.41, "bbox_xyxy": [44, 22, 58, 42]},
        ][:max_candidates]
        return values, {
            "request_id": "ground-test",
            "returned_candidate_count": len(values),
        }


class FakeContactGraspNet:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"pregrasp_offset_m": 0.04}

    def require_ready(self) -> dict[str, object]:
        return {"ready": True, "load_count": 1}

    def propose(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        # A surface declaring grasp_plan_ik_reachability_annotation_v1 asks for
        # a pool of up to 32 and applies the display cap after IK selection.
        assert kwargs["max_candidates"] <= 32
        scene = np.asarray(kwargs["scene_points_camera_m"])
        target = np.asarray(kwargs["target_points_camera_m"])
        assert scene.ndim == 2 and scene.shape[1] == 3
        assert target.shape == (26 * 18, 3)
        # The single-view boundary used to be asserted here, which encoded a
        # transport decision as a test-double invariant: the moment the proposer
        # received the fused scene the canvas already showed, this double would
        # have rejected it. Record the fact instead of forbidding it.
        self.saw_second_camera_depth = bool(np.any(np.isclose(scene[:, 2], 0.95)))
        self.last_scene_points_camera_m = scene.copy()
        candidate = {
            "candidate_id": "grasp-0123456789abcdef01234567",
            "source_index": 0,
            "rank": 0,
            "source_gripper_origin_xyz_m": [0.0, 0.0, 0.9034],
            "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.9434],
            "approach_world": [0.0, 0.0, -1.0],
            "jaw_world": [1.0, 0.0, 0.0],
            "gripper_opening_m": 0.04,
            "backend_score": 0.75,
            "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
            "checks": {
                "workspace": "passed", "aperture": "passed",
                "ik": "not_checked", "collision": "not_checked",
            },
            "execution_ready": False,
        }
        pose = np.eye(4, dtype=np.float64)[None]
        pose[0, :3, 3] = [0.0, 0.0, 0.9034]
        return [candidate], {
            "request_id": "grasp-test",
            "rejected": [],
            "_backend_input_regime": {
                "schema": "ora.contact-graspnet-input-regime.v1",
                "policy": "distinct_shuffled_crop_points_v1", "status": "region",
                "crop_points": 2029, "unique_points": 2029,
                "configured_input_points": 20000, "input_points": 2029,
                "padding_factor": 1.0, "upstream_padding_factor": 9.8571,
            },
            "_canonical_backend_output": {
                "poses_camera": pose,
                "scores": np.asarray([0.75], dtype=np.float64),
                "contact_points_camera": np.asarray(
                    [[0.0, 0.0, 1.0068]], dtype=np.float64,
                ),
                "openings_m": np.asarray([0.04], dtype=np.float64),
            },
        }


def _select_correctness_surface(runtime, profile=LIBERO_SURFACE):
    runtime.surface = resolve_tool_surface(profile)
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    runtime._motion_probe = lambda arguments: {"results": [
        {"query_id": query["query_id"], "reachable": False, "variants": [], "selected_variant_id": None, "plan_ref": None}
        for query in arguments["queries"]
    ]}
    return runtime


def test_correctness_perception_reports_thin_surface_without_identity_claim(tmp_path):
    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object", "max_candidates": 2})
    evidence = grounded["perception_evidence"]
    assert evidence["source_image_size"] == [64, 64]
    assert evidence["semantic_identity_verified"] is False
    assert evidence["point_uv"] is None
    assert [item["target_ref"] for item in evidence["targets"]] == [item["target_ref"] for item in grounded["candidates"]]
    assert evidence["targets"][0]["visible_extent_xyz_m"][2] == 0
    assert evidence["targets"][0]["depth_valid_fraction"] == 1
    assert evidence["targets"][0]["mask_touches_image_boundary"] is False
    with pytest.raises(ToolError, match="source image"):
        runtime.invoke("vision_segment", {"camera": "agentview", "points_uv": [[1023, 575]]})


@pytest.mark.parametrize("damage", ["missing", "bad_hash", "bad_calibration",
    "missing_intrinsic_valid_hash", "wrong_shape_valid_hash", "singular_valid_hash",
    "nonfinite_valid_hash", "zero_focal_valid_hash", "bad_rgb_valid_hash",
    "bad_depth_valid_hash"])
def test_correctness_optional_wrist_camera_failure_preserves_primary_grasp(tmp_path, damage):
    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime)
    control, root, observation, pose, gripper = runtime._current_grip_state()
    runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object"})
    target_ref = grounded["candidates"][0]["target_ref"]
    baseline = runtime.invoke("grasp_plan", {"target_ref": target_ref, "max_candidates": 3})
    if damage == "missing":
        observation["cameras"].pop("wrist")
    elif damage == "bad_hash":
        observation["cameras"]["wrist"]["files"]["rgb_lossless_png"]["sha256"] = "0" * 64
    elif damage == "bad_calibration":
        path = root / observation["cameras"]["wrist"]["files"]["calibration"]["path"]
        path.chmod(0o644)
        path.write_text("{}")
    else:
        files = observation["cameras"]["wrist"]["files"]
        role = ("rgb_lossless_png" if damage == "bad_rgb_valid_hash" else
                "depth_numpy_m_f32" if damage == "bad_depth_valid_hash" else "calibration")
        path = root / files[role]["path"]
        path.chmod(0o644)
        if role != "calibration":
            path.write_bytes(b"corrupt")
        else:
            calibration = json.loads(path.read_text())
            if damage == "missing_intrinsic_valid_hash":
                calibration.pop("intrinsic_cv")
            elif damage == "wrong_shape_valid_hash":
                calibration["extrinsic_cv"] = [1, 2, 3]
            elif damage == "singular_valid_hash":
                calibration["extrinsic_cv"] = np.zeros((4, 4)).tolist()
            elif damage == "nonfinite_valid_hash":
                calibration["intrinsic_cv"][0][0] = float("nan")
            elif damage == "zero_focal_valid_hash":
                calibration["intrinsic_cv"][0][0] = 0
            path.write_text(json.dumps(calibration))
        files[role]["sha256"] = _sha256(path)
    result = runtime.invoke("grasp_plan", {"target_ref": target_ref, "max_candidates": 3})
    assert result["candidates"] == baseline["candidates"]
    assert result["review_evidence"]["unavailable_cameras"] == ["wrist"]
    assert result["review_evidence"]["available_cameras"] == ["agentview"]
    assert result["review_evidence"]["status"] == "available"
    assert result["grasp_query_ref"] != baseline["grasp_query_ref"]
    events = [json.loads(line) for line in runtime.event_path.read_text().splitlines()]
    event = next(item for item in reversed(events) if item["type"] == "grasp_plan_materialized")
    assert event["review_source_binding"]["unavailable_sources"] == [{
        "camera": "wrist", "reason": "unusable_public_rgbd_review_source"}]
    manifest = json.loads((runtime.artifact_root / "grasp-queries" /
                           result["grasp_query_ref"] / "manifest.json").read_text())
    assert manifest["review_visualization"]["scene_cameras"] == ["agentview"]
    assert result["review_evidence"]["primary_input_required"] is True
    primary = root / observation["cameras"]["agentview"]["files"]["depth_numpy_m_f32"]["path"]
    primary.chmod(0o644)
    primary.write_bytes(b"corrupt")
    with pytest.raises(ToolError):
        runtime.invoke("grasp_plan", {"target_ref": target_ref, "max_candidates": 3})


def test_correctness_dual_arm_head_only_grasp_does_not_require_wrist(tmp_path):
    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime, ROBOTWIN_SURFACE)
    control, root, observation, pose, gripper = runtime._current_grip_state()
    observation["cameras"] = {"head_camera": observation["cameras"]["agentview"]}
    runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
    grounded = runtime.invoke("vision_ground", {"camera": "head_camera", "text": "red object"})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"],
                                           "arm": "right", "max_candidates": 3})
    assert result["review_evidence"]["primary_camera"] == "head_camera"
    assert result["review_evidence"]["available_cameras"] == ["head_camera"]
    assert result["review_evidence"]["unavailable_cameras"] == ["right_camera"]
    assert result["status"] == "candidates"


@pytest.mark.parametrize("damage", ["missing_intrinsic", "singular_extrinsic", "bad_depth"])
def test_correctness_primary_rgbd_stays_required_with_hash_valid_invalid_geometry(tmp_path, damage):
    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime)
    control, root, observation, pose, gripper = runtime._current_grip_state()
    runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object"})
    files = observation["cameras"]["agentview"]["files"]
    role = "depth_numpy_m_f32" if damage == "bad_depth" else "calibration"
    path = root / files[role]["path"]
    path.chmod(0o644)
    if damage == "bad_depth":
        path.write_bytes(b"corrupt")
    else:
        calibration = json.loads(path.read_text())
        if damage == "missing_intrinsic":
            calibration.pop("intrinsic_cv")
        else:
            calibration["extrinsic_cv"] = np.zeros((4, 4)).tolist()
        path.write_text(json.dumps(calibration))
    files[role]["sha256"] = _sha256(path)
    with pytest.raises(ToolError) as refused:
        runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"]})
    # An invalid published calibration keeps its infrastructure code instead of
    # being masked as a retryable proposal failure (EXP11B census C3/D4).
    if damage == "bad_depth":
        assert "primary RGB-D source is unusable" in str(refused.value)
    else:
        assert refused.value.code == "invalid_camera_calibration"
        assert refused.value.as_dict()["infrastructure"] is True


def _runtime(
    tmp_path: Path, *, grasp_backend: object | None = None,
    profile: str = LIBERO_SURFACE,
) -> tuple[ManipulationToolRuntime, str]:
    control, observation_id = _publish_object_observation(tmp_path)
    sam = FakeSam3()
    return ManipulationToolRuntime(
        control_root=control,
        manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=BOUNDS,
        state_root=None,
        tool_surface_profile=profile,
        controlled_end_effector="right",
        segmentation_backend=sam,
        grounding_backend=sam,
        grasp_backend=grasp_backend or FakeContactGraspNet(),
    ), observation_id


def test_text_ground_then_grasp_plan_compiles_existing_motion_arguments(
    tmp_path: Path,
) -> None:
    runtime, observation_id = _runtime(tmp_path)
    assert runtime.classify_call_effect("vision_ground", {}) == "read_only"
    assert runtime.classify_call_effect("grasp_plan", {}) == "read_only"
    grounded = runtime.invoke("vision_ground", {
        "camera": "agentview", "text": "red object", "max_candidates": 2,
    })
    assert grounded["observation_id"] == observation_id
    assert grounded["status"] == "candidates"
    assert len(grounded["candidates"]) == 2
    assert len({item["target_ref"] for item in grounded["candidates"]}) == 2
    assert len(grounded["images_base64"]) == 3
    assert [item["image_index"] for item in grounded["candidate_previews"]] == [1, 2]
    assert [item["rank"] for item in grounded["candidate_previews"]] == [0, 1]
    assert [item["target_ref"] for item in grounded["candidate_previews"]] == [
        item["target_ref"] for item in grounded["candidates"]
    ]
    rendered_sizes = []
    for encoded in grounded["images_base64"]:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            rendered_sizes.append(image.size)
    # Cards carry the depth profile on this surface, hence the taller canvas.
    assert rendered_sizes == [(64, 64), (1024, 800), (1024, 800)]
    grounding_manifests = list(
        runtime.artifact_root.glob("grounding-queries/*/manifest.json")
    )
    assert len(grounding_manifests) == 1
    grounding_manifest = json.loads(
        grounding_manifests[0].read_text(encoding="utf-8")
    )
    assert grounding_manifest["schema"] == "ora.grounding-query.v2"
    visualization = grounding_manifest["visualization"]
    assert visualization["renderer"] == "grounding-candidate-inspection.v2"
    assert visualization["image_order"] == ["overview", "T0", "T1"]
    assert visualization["policy_visible"] is True
    assert all(
        item["resampling"] == "bicubic_visual_enlargement_no_new_source_detail"
        for item in visualization["candidate_previews"]
    )
    assert all(
        "not higher-resolution sensing" in item["claim_boundary"]
        for item in visualization["candidate_previews"]
    )

    planned = runtime.invoke("grasp_plan", {
        "target_ref": grounded["candidates"][0]["target_ref"],
        "max_candidates": 3,
    })
    assert planned["observation_id"] == observation_id
    assert planned["status"] == "candidates"
    candidate = planned["candidates"][0]
    assert candidate["pregrasp_pose"]["position_m"] == pytest.approx(
        [0.0, 0.0, 0.8464]
    )
    assert np.linalg.norm(candidate["pregrasp_pose"]["orientation_xyzw"]) == pytest.approx(1.0)
    assert candidate["contact_pose"] == {
        "position_m": pytest.approx([0.0, 0.0, 0.8064]),
        "orientation_xyzw": pytest.approx(
            candidate["pregrasp_pose"]["orientation_xyzw"],
        ),
    }
    # The approach is the grip-site +Z 40 mm delta, spelled as a target the
    # surface's own move_pose accepts verbatim.
    assert candidate["approach_delta"] == {
        "position_mode": "delta",
        "orientation_mode": "hold_current",
        "translation_m": pytest.approx([0.0, 0.0, 0.04]),
        "position_frame": "end_effector",
        "interaction": "contact_approach",
    }
    assert candidate["compiler_checks"]["reachability"] == (
        "delegated_to_move_pose"
    )
    assert planned["compiler"]["controlled_point"] == "robosuite_grip_site"
    assert planned["compiler"]["controlled_point_semantics"] == "grip_center"
    assert planned["compiler"]["source_origin_to_controlled_point_m"] == 0.097
    assert "plan_id" not in json.dumps(planned)
    assert "pointcloud" not in json.dumps(planned)
    query_manifests = list(runtime.artifact_root.glob("grasp-queries/*/manifest.json"))
    assert len(query_manifests) == 1
    query_manifest = json.loads(query_manifests[0].read_text(encoding="utf-8"))
    assert query_manifest["schema"] == "ora.grasp-query.v7"
    visualization = query_manifest["visualization"]
    assert visualization["renderer"] == "grasp-plan-target-local-contact-sheet.v2"
    assert visualization["target_mask_sha256"] == query_manifest["identity"][
        "target_mask_sha256"
    ]
    assert visualization["candidate_projections"][0] == {
        "candidate_id": candidate["candidate_id"],
        "rank": 0,
        "raw_origin_uv": [32, 32],
        "contact_baseline_uv": [32, 32],
        "grip_site_uv": [32, 32],
        "pregrasp_uv": [32, 32],
        "compiler_status": "returned",
    }
    assert visualization["candidate_panel_count"] == 1
    assert visualization["policy_visible"] is True
    overlay_path = query_manifests[0].parent / query_manifest["overlay"]["path"]
    with Image.open(overlay_path) as overlay:
        assert overlay.size == (1440, 1080)
    review = query_manifest["review_visualization"]
    assert review["status"] == "available"
    assert review["renderer"] == "grasp-plan-global-pointcloud-review.v3"
    assert review["review_scope"] == "synchronized_agentview_wrist_rgbd_fusion"
    assert review["backend_input_scope"] == "multiview_fused_rgbd"
    assert review["scene_cameras"] == ["agentview", "wrist"]
    assert set(review["scene_points_by_source"]) == {"agentview", "wrist"}
    assert all(value > 0 for value in review["scene_points_by_source"].values())
    # This surface delivers the review image beside the contact sheet.
    assert review["policy_visible"] is True and len(planned["images_base64"]) == 2
    assert [item["view"] for item in review["views"]] == ["top", "front", "side"]
    review_path = query_manifests[0].parent / review["artifact"]["path"]
    with Image.open(review_path) as review_image:
        assert review_image.size == (1168, 768)
    pointcloud = query_manifest["input_pointcloud"]
    assert pointcloud["schema"] == "ora.grasp-input-pointcloud.v4"
    assert pointcloud["input_scope"] == "multiview_fused_rgbd_for_backend"
    profile = pointcloud["backend_input_profile"]
    assert {key: profile[key] for key in (
        "camera", "contributing_cameras", "excluded_cameras", "max_depth_m",
        "multiview_fusion", "robot_self_filter", "scope", "voxel_size_m",
        "workspace_bounds_m",
    )} == {
        "camera": "agentview",
        "contributing_cameras": ["agentview", "wrist"],
        "excluded_cameras": [],
        "max_depth_m": 2.5,
        "multiview_fusion": True,
        "robot_self_filter": False,
        "scope": "multiview_fused_rgbd",
        "voxel_size_m": 0.005,
        "workspace_bounds_m": BOUNDS,
    }
    assert pointcloud["backend_source"]["camera"] == "agentview"
    pointcloud_path = query_manifests[0].parent / pointcloud["artifact"]["path"]
    with np.load(pointcloud_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "backend_scene_points_camera_m", "backend_scene_points_world_m",
            "target_points_camera_m", "target_camera_to_world",
            "target_points_world_m",
        }
        assert arrays["backend_scene_points_camera_m"].shape[1] == 3
        assert arrays["target_points_camera_m"].shape == (26 * 18, 3)
        assert arrays["target_camera_to_world"].shape == (4, 4)
        assert np.array_equal(
            arrays["backend_scene_points_camera_m"],
            runtime.grasp_backend.last_scene_points_camera_m,
        )
    review_pointcloud = query_manifest["review_pointcloud"]
    assert [item["camera"] for item in review_pointcloud["sources"]] == [
        "agentview", "wrist",
    ]
    with np.load(
        query_manifests[0].parent / review_pointcloud["artifact"]["path"],
        allow_pickle=False,
    ) as arrays:
        assert set(arrays["review_scene_source_index"].tolist()) == {0, 1}
        assert arrays["review_scene_rgb_u8"].shape == arrays[
            "review_scene_points_world_m"
        ].shape
    backend_output = query_manifest["backend_output"]
    assert backend_output["schema"] == (
        "ora.contact-graspnet-backend-output.v1"
    )
    assert backend_output["policy_visible"] is False
    # The service's input regime is private evidence beside the raw arrays.
    assert backend_output["input_regime"]["input_points"] == 2029
    assert "_backend_input_regime" not in query_manifest["diagnostics"]
    with np.load(
        query_manifests[0].parent / backend_output["artifact"]["path"],
        allow_pickle=False,
    ) as arrays:
        assert arrays["contact_points_camera"].shape == (1, 3)
    with pytest.raises(ToolError) as caller_bound_observation:
        runtime.invoke("grasp_plan", {
            "observation_id": observation_id,
            "target_ref": grounded["candidates"][0]["target_ref"],
        })
    assert caller_bound_observation.value.code == "invalid_arguments"

    masks = list(runtime.artifact_root.rglob("mask.npy"))
    assert len(masks) == 2
    assert not list((tmp_path / "workspace").rglob("mask.npy"))
    events = runtime.event_path.read_text(encoding="utf-8")
    assert "grounding_query_materialized" in events
    assert "grasp_plan_materialized" in events
    assert "mask.npy" not in events
    materialized = next(
        json.loads(line) for line in events.splitlines()
        if json.loads(line).get("type") == "grasp_plan_materialized"
    )
    assert materialized["artifact"]["path"].endswith("/global-review.png")
    assert materialized["policy_overlay_artifact"]["path"].endswith("/overlay.png")
    assert materialized["input_pointcloud_artifact"]["path"].endswith(
        "/input-pointcloud.npz"
    )
    assert materialized["review_pointcloud_artifact"]["path"].endswith(
        "/review-pointcloud.npz"
    )
    assert materialized["manifest_artifact"]["path"].endswith("/manifest.json")
    assert materialized["visualization"]["claim_boundary"].startswith(
        "target_local_geometry_preview_only"
    )
    assert materialized["review_visualization"]["policy_visible"] is True


def test_surfaces_without_the_perception_review_capabilities_keep_released_identities(
    tmp_path: Path,
) -> None:
    """EXP11A g009 perception review fixes are capability-gated, not global.

    The RoboTwin prepared-grasp qualification surface declares neither
    grounding_candidate_depth_profile_v1 nor observed_extent_scene_review_v1,
    so the grounding identity is the released structure -- the query ref
    recomputes from it alone -- and every card keeps the released 1024x576 size.
    """

    from ora.agent.tools import grasp

    runtime, _observation_id = _runtime(tmp_path)
    _select_correctness_surface(runtime, "robotwin_toolchain_b2_closed_noservo_v1")
    control, root, observation, pose, gripper = runtime._current_grip_state()
    observation["cameras"] = {"head_camera": observation["cameras"]["agentview"]}
    runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
    assert not runtime.surface.supports("grounding_candidate_depth_profile_v1")
    grounded = runtime.invoke("vision_ground", {
        "camera": "head_camera", "text": "red object", "max_candidates": 2,
    })
    manifest = json.loads(next(
        runtime.artifact_root.glob("grounding-queries/*/manifest.json")
    ).read_text(encoding="utf-8"))
    identity = manifest["identity"]
    assert identity["visualization"] == {"renderer": "grounding-candidate-inspection.v2"}
    assert set(identity) == {"observation_id", "camera", "text", "max_candidates",
                             "source", "backend", "visualization"}
    assert grounded["grounding_ref"] == "grounding-query-" + grasp._canonical_hash(identity)[:24]
    assert all("depth_profile" not in item
               for item in manifest["visualization"]["candidate_previews"])
    assert [item["render_dimensions"] for item in grounded["candidate_previews"]] == [
        [1024, 576], [1024, 576]]
    assert not runtime.surface.supports("observed_extent_scene_review_v1")
    planned = runtime.invoke("grasp_plan", {
        "target_ref": grounded["candidates"][0]["target_ref"], "arm": "right",
        "max_candidates": 3,
    })
    query = json.loads(next(
        runtime.artifact_root.glob("grasp-queries/*/manifest.json")
    ).read_text(encoding="utf-8"))
    assert "review_renderer" not in query["identity"]
    assert planned["grasp_query_ref"] == "grasp-query-" + grasp._canonical_hash(query["identity"])[:24]
    review = query["review_visualization"]
    assert review["status"] == "available" and "observed_extent" not in review
    # Perception feedback delivers the review image, and the manifest says so.
    assert review["policy_visible"] is True and len(planned["images_base64"]) == 2


def test_grasp_plan_survives_review_fusion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing optional review camera is dropped before fusion on the maintained
    # surfaces, so force the best-effort review fusion itself to fail.
    from ora.agent.tools import grasp_query

    def failing_fusion(**_kwargs: object) -> object:
        raise ValueError("review fusion failed: " + "x" * 1024)

    runtime, _observation_id = _runtime(tmp_path)
    grounded = runtime.invoke("vision_ground", {
        "camera": "agentview", "text": "red object", "max_candidates": 1,
    })
    monkeypatch.setattr(grasp_query, "fuse_rgbd_pointcloud_sources", failing_fusion)

    planned = runtime.invoke("grasp_plan", {
        "target_ref": grounded["candidates"][0]["target_ref"],
        "max_candidates": 1,
    })
    assert planned["status"] == "candidates"
    manifest_path = next(runtime.artifact_root.glob("grasp-queries/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_visualization"]["status"] == "unavailable"
    assert manifest["review_visualization"]["failure"]["type"] == "ValueError"
    assert len(manifest["review_visualization"]["failure"]["message"]) <= 512
    assert manifest["review_pointcloud"] is None


def test_grasp_plan_drops_one_degenerate_candidate_before_render(
    tmp_path: Path,
) -> None:
    class MixedBackend(FakeContactGraspNet):
        def propose(self, **kwargs: object):  # type: ignore[no-untyped-def]
            candidates, diagnostics = super().propose(**kwargs)
            bad = dict(candidates[0])
            bad["candidate_id"] = "grasp-fedcba9876543210fedcba98"
            bad["rank"] = 0
            bad["approach_world"] = [0.0, 0.0, 0.0]
            good = dict(candidates[0])
            good["rank"] = 1
            return [bad, good], diagnostics

    runtime, _observation_id = _runtime(tmp_path, grasp_backend=MixedBackend())
    grounded = runtime.invoke("vision_ground", {
        "camera": "agentview", "text": "red object", "max_candidates": 1,
    })
    planned = runtime.invoke("grasp_plan", {
        "target_ref": grounded["candidates"][0]["target_ref"],
        "max_candidates": 2,
    })
    assert [item["rank"] for item in planned["candidates"]] == [0]
    assert planned["diagnostics"]["compiler_rejected"] == [{
        "candidate_id": "grasp-fedcba9876543210fedcba98",
        "reason": "grasp approach axis is degenerate",
    }]


def test_contact_graspnet_origin_compiles_to_panda_grip_site() -> None:
    approach = np.asarray([0.437330037083, 0.099535647523, -0.893775751258])
    approach /= np.linalg.norm(approach)
    jaw = np.asarray([-0.875623586418, -0.179426567165, -0.448429974359])
    source_origin = np.asarray([0.075219177673, 0.004896843306, 0.167504583204])
    candidate = {
        "candidate_id": "grasp-0123456789abcdef01234567",
        "source_index": 0,
        "rank": 0,
        "source_gripper_origin_xyz_m": source_origin.tolist(),
        "source_pregrasp_origin_xyz_m": (source_origin - approach * 0.04).tolist(),
        "approach_world": approach.tolist(),
        "jaw_world": jaw.tolist(),
        "gripper_opening_m": 0.008972176351,
        "backend_score": 0.2892,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed"},
        "execution_ready": False,
    }
    geometry = _compiled_grasp_geometry(candidate)
    assert geometry["contact_baseline"] == pytest.approx(
        source_origin + approach * 0.1034,
    )
    assert geometry["grip_site"] == pytest.approx(
        source_origin + approach * 0.097,
    )
    assert np.linalg.norm(
        geometry["contact_baseline"] - geometry["grip_site"],
    ) == pytest.approx(0.0064)
    compiled = _compile_grasp_candidate(candidate)
    assert compiled["pregrasp_pose"]["position_m"] == pytest.approx(
        source_origin + approach * 0.057,
    )
    assert compiled["approach_delta"]["translation_m"] == pytest.approx(
        [0.0, 0.0, 0.04],
    )
    assert compiled["compiler_checks"]["embodiment_mapping"] == (
        PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
    )


def test_grasp_scene_fusion_is_stable_and_requires_both_public_cameras(
    tmp_path: Path,
) -> None:
    runtime, _observation_id = _runtime(tmp_path)
    _control, observation_root, observation, _pose, _gripper = runtime._current_grip_state()
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root,
        observation=observation,
        workspace_bounds_m=BOUNDS,
        camera_names=("agentview", "wrist"),
        voxel_size_m=0.005,
        max_depth_m=2.5,
    )
    assert set(fused["source_index"].tolist()) == {0, 1}
    assert fused["deduplication"] == "stable_first_by_camera_order"

    duplicate = copy.deepcopy(observation)
    duplicate["cameras"]["wrist"] = copy.deepcopy(
        duplicate["cameras"]["agentview"]
    )
    duplicate_fusion = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root,
        observation=duplicate,
        workspace_bounds_m=BOUNDS,
        camera_names=("agentview", "wrist"),
        voxel_size_m=0.005,
        max_depth_m=2.5,
    )
    assert set(duplicate_fusion["source_index"].tolist()) == {0}
    assert duplicate_fusion["sources"][1]["retained_voxel_points"] == 0

    missing = copy.deepcopy(observation)
    del missing["cameras"]["wrist"]
    with pytest.raises(ValueError, match="requires camera wrist"):
        fuse_rgbd_pointcloud_sources(
            observation_root=observation_root,
            observation=missing,
            workspace_bounds_m=BOUNDS,
            camera_names=("agentview", "wrist"),
            voxel_size_m=0.005,
            max_depth_m=2.5,
        )


def test_viewer_projects_grasp_plan_preview_and_supports_hash_only_runs(
    tmp_path: Path,
) -> None:
    run = tmp_path / "grasp-plan-run"
    state_root = run / "policy-state/steps"
    state_root.mkdir(parents=True)
    (state_root / "step-0000.json").write_text(json.dumps({
        "step": 0,
        "observation": {"id": "obs-0"},
        "cameras": {},
    }), encoding="utf-8")
    (run / "policy-state/manifest.json").write_text(json.dumps({
        "steps": [{"step": 0, "path": "policy-state/steps/step-0000.json"}],
    }), encoding="utf-8")
    query_ref = "grasp-query-0123456789abcdef01234567"
    overlay = run / "openeta-tool-artifacts/grasp-queries" / query_ref / "overlay.png"
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (10, 20, 30)).save(overlay)
    records = [
        {"type": "observation", "observation_id": "obs-0",
         "policy_state": {"step": 0}},
        {"type": "grasp_plan_materialized", "observation_id": "obs-0",
         "query_ref": query_ref, "candidate_count": 2,
         "overlay_sha256": _sha256(overlay)},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    trajectory = compile_trajectory(run)
    preview = next(
        image for image in trajectory["steps"][0]["images"]
        if image.get("kind") == "grasp_plan"
    )
    assert preview["label"] == "grasp.plan policy contact sheet · 2 candidates"
    assert preview["query_ref"] == query_ref
    assert preview["path"].endswith(f"{query_ref}/overlay.png")

    records[1]["overlay_sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert not any(
        image.get("kind") == "grasp_plan"
        for image in compile_trajectory(run)["steps"][0]["images"]
    )

    review = overlay.parent / "global-review.png"
    Image.new("RGB", (48, 36), (30, 40, 50)).save(review)
    records[1]["artifact"] = {
        "path": review.relative_to(run).as_posix(),
        "sha256": _sha256(review),
        "bytes": review.stat().st_size,
    }
    records[1]["policy_overlay_artifact"] = {
        "path": overlay.relative_to(run).as_posix(),
        "sha256": _sha256(overlay),
        "bytes": overlay.stat().st_size,
    }
    records[1].pop("overlay_sha256", None)
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    projected = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") == "grasp_plan"
    ]
    assert [image["visibility"] for image in projected] == [
        "policy_visible", "review_only",
    ]
    assert projected[0]["label"] == "grasp.plan policy contact sheet · 2 candidates"
    assert projected[0]["path"].endswith("/overlay.png")
    assert projected[1]["label"] == "grasp.plan global input review · 2 candidates"
    assert projected[1]["path"].endswith("/global-review.png")


def test_viewer_projects_hash_bound_grounding_overlay(tmp_path: Path) -> None:
    run = tmp_path / "grounding-run"
    state_root = run / "policy-state/steps"
    state_root.mkdir(parents=True)
    (state_root / "step-0000.json").write_text(json.dumps({
        "step": 0,
        "observation": {"id": "obs-0"},
        "cameras": {},
    }), encoding="utf-8")
    (run / "policy-state/manifest.json").write_text(json.dumps({
        "steps": [{"step": 0, "path": "policy-state/steps/step-0000.json"}],
    }), encoding="utf-8")
    grounding_ref = "grounding-query-0123456789abcdef01234567"
    overlay = (
        run / "openeta-tool-artifacts/grounding-queries"
        / grounding_ref / "overlay.png"
    )
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (20, 30, 40)).save(overlay)
    previews = []
    for rank in range(3):
        preview = overlay.parent / f"candidate-T{rank}.png"
        Image.new("RGB", (64, 48), (30 + rank, 40, 50)).save(preview)
        previews.append({
            "rank": rank,
            "target_ref": f"target-{rank:024x}",
            "artifact": {
                "path": preview.name,
                "sha256": _sha256(preview),
                "bytes": preview.stat().st_size,
            },
        })
    records = [
        {"type": "observation", "observation_id": "obs-0",
         "policy_state": {"step": 0}},
        {"type": "grounding_query_materialized", "observation_id": "obs-0",
         "grounding_ref": grounding_ref, "candidate_count": 3,
         "camera": "agentview", "overlay_sha256": _sha256(overlay),
         "candidate_previews": previews},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    grounding_images = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") in {"grounding", "grounding_candidate"}
    ]
    assert len(grounding_images) == 4
    projected = grounding_images[0]
    assert projected["label"] == "vision.ground · agentview · 3 targets"
    assert projected["visibility"] == "policy_visible"
    assert projected["grounding_ref"] == grounding_ref
    assert [image["rank"] for image in grounding_images[1:]] == [0, 1, 2]
    assert grounding_images[1]["label"] == (
        "vision.ground T0 candidate enlargement · agentview"
    )

    records[1]["candidate_previews"][1]["artifact"]["sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    grounding_images = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") in {"grounding", "grounding_candidate"}
    ]
    assert [image.get("rank") for image in grounding_images[1:]] == [0, 2]

    records[1]["overlay_sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert not any(
        image.get("kind") == "grounding"
        for image in compile_trajectory(run)["steps"][0]["images"]
    )


def test_sam3_text_grounding_client_validates_ranked_arrays() -> None:
    def encoded(value: np.ndarray) -> str:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "segmentation",
                "warmup_complete": True,
                "load_count": 1,
                "instance_id": "grounding-test-instance",
                "adapter_sha256": "a" * 64,
                "started_unix_s": 1.0,
                "model": dict(SAM3_MODEL_IDENTITY),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            assert self.path == "/v1/ground-text"
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            assert request["schema"] == (
                "ora.model-service.ground-text-request.v1"
            )
            assert request["text"] == "red object"
            masks = np.zeros((2, 8, 8), dtype=np.uint8)
            masks[0, 1:5, 1:4] = 1
            masks[1, 2:7, 5:7] = 1
            response = {
                "schema": "ora.model-service.ground-text.v1",
                "request_id": "grounding-test-request",
                "model": dict(SAM3_MODEL_IDENTITY),
                "service_identity": {
                    "schema": "ora.model-service-response-identity.v1",
                    "instance_id": "grounding-test-instance",
                    "pid": 123,
                    "started_unix_s": 1.0,
                    "adapter_sha256": "a" * 64,
                },
                "masks_npy_base64": encoded(masks),
                "scores_npy_base64": encoded(
                    np.asarray([0.8, 0.3], dtype=np.float32),
                ),
                "boxes_xyxy_npy_base64": encoded(
                    np.asarray([[1, 1, 4, 5], [5, 2, 7, 7]], dtype=np.float32),
                ),
                "metrics": {
                    "queue_ms": 0.1, "inference_ms": 9.0, "total_ms": 9.1,
                },
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Sam3ApiSegmenter(
        endpoint=f"http://127.0.0.1:{server.server_port}",
    )
    try:
        assert client.require_ready()["load_count"] == 1
        candidates, diagnostics = client.ground(
            rgb_u8=np.zeros((8, 8, 3), dtype=np.uint8),
            text="red object",
            max_candidates=2,
        )
        assert [item["rank"] for item in candidates] == [0, 1]
        assert candidates[0]["bbox_xyxy"] == [1, 1, 4, 5]
        assert candidates[0]["backend_score"] == pytest.approx(0.8)
        assert diagnostics["model"] == SAM3_MODEL_IDENTITY
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_contact_graspnet_binding_requires_the_exact_live_instance() -> None:
    class Handler(BaseHTTPRequestHandler):
        instance_id = "grasp-test-instance"

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "grasp_proposals",
                "warmup_complete": True,
                "load_count": 1,
                "instance_id": type(self).instance_id,
                "pid": 456,
                "adapter_sha256": "c" * 64,
                "started_unix_s": 2.0,
                "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
        "endpoint": endpoint,
        "instance_id": "grasp-test-instance",
        "pid": 456,
        "started_unix_s": 2.0,
        "adapter_sha256": "c" * 64,
        "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
        "load_count": 1,
        "warmup_complete": True,
        "qualification": {
            "sha256": "d" * 64,
            "status": "passed",
            "created_unix_s": 2.0,
        },
    }
    client = ContactGraspNetApiProposer(
        endpoint=endpoint, expected_binding=binding,
    )
    try:
        assert client.require_ready()["instance_id"] == "grasp-test-instance"
        Handler.instance_id = "replacement-instance"
        with pytest.raises(ValueError, match="bound instance_id"):
            client.require_ready()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_grasp_plan_runtime_splits_the_bound_service_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class BoundSam:
        def __init__(self, *, endpoint: str, expected_binding: object) -> None:
            seen["sam_endpoint"] = endpoint
            seen["sam_binding"] = expected_binding

        def require_ready(self) -> dict[str, object]:
            return {"service": "segmentation", "instance_id": "sam"}

        def ground(self, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
            return [], {}

    class BoundGrasp:
        def __init__(
            self, *, endpoint: str, hover_offset_m: float,
            expected_binding: object,
        ) -> None:
            seen["grasp_endpoint"] = endpoint
            seen["hover_offset_m"] = hover_offset_m
            seen["grasp_binding"] = expected_binding

        def require_ready(self) -> dict[str, object]:
            return {"service": "grasp_proposals", "instance_id": "grasp"}

    monkeypatch.setattr(runtime_core, "Sam3ApiSegmenter", BoundSam)
    monkeypatch.setattr(runtime_core, "ContactGraspNetApiProposer", BoundGrasp)
    control, _observation_id = _publish_object_observation(tmp_path)
    segmentation_binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
    }
    grasp_binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
    }
    ManipulationToolRuntime(
        control_root=control,
        manual_path=MANUAL,
        event_path=tmp_path / "events-bound.jsonl",
        workspace_bounds_m=BOUNDS,
        state_root=None,
        tool_surface_profile=LIBERO_SURFACE,
        controlled_end_effector="right",
        prerequisite_binding={
            "schema": "ora.model-service-set-binding.v1",
            "tool_surface": LIBERO_SURFACE,
            "services": {
                "segmentation": segmentation_binding,
                "grasp_proposals": grasp_binding,
            },
        },
    )
    assert seen == {
        "sam_endpoint": "http://127.0.0.1:8114",
        "sam_binding": segmentation_binding,
        "grasp_endpoint": "http://127.0.0.1:8115",
        "hover_offset_m": 0.04,
        "grasp_binding": grasp_binding,
    }


def test_capability_scoped_service_binding_replaces_the_rename_hack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 set binding is valid for any surface supporting its scope."""

    seen: dict[str, object] = {}

    class BoundSam:
        def __init__(self, *, endpoint: str, expected_binding: object) -> None:
            seen["sam_binding"] = expected_binding

        def ground(self, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
            return [], {}

    class BoundGrasp:
        def __init__(
            self, *, endpoint: str, hover_offset_m: float,
            expected_binding: object,
        ) -> None:
            seen["grasp_binding"] = expected_binding

    monkeypatch.setattr(runtime_core, "Sam3ApiSegmenter", BoundSam)
    monkeypatch.setattr(runtime_core, "ContactGraspNetApiProposer", BoundGrasp)
    control, _observation_id = _publish_object_observation(tmp_path)
    # Issued under a different profile name: previously this forced the
    # experiment-side _rebind_service_surface hack.
    segmentation_binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": ROBOTWIN_SURFACE,
    }
    grasp_binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": ROBOTWIN_SURFACE,
    }

    def build(capability_scope: list[str]) -> ManipulationToolRuntime:
        return ManipulationToolRuntime(
            control_root=control,
            manual_path=MANUAL,
            event_path=tmp_path / "events-scoped.jsonl",
            workspace_bounds_m=BOUNDS,
            state_root=None,
            tool_surface_profile=LIBERO_SURFACE,
            controlled_end_effector="right",
            prerequisite_binding={
                "schema": "ora.model-service-set-binding.v2",
                "capability_scope": capability_scope,
                "services": {
                    "segmentation": segmentation_binding,
                    "grasp_proposals": grasp_binding,
                },
            },
        )

    build(["contact_graspnet_panda_grip_site_compiler_v2"])
    assert seen["sam_binding"] == segmentation_binding
    assert seen["grasp_binding"] == grasp_binding

    with pytest.raises(ToolError, match="capability scope is not supported"):
        build(["capability_the_surface_does_not_declare_v1"])
    with pytest.raises(ValueError, match="set binding is invalid"):
        build([])


def test_the_grounding_request_bound_is_the_service_limit_not_the_palette():
    """Two ceilings were conflated and the wrong one bounded the request.

    The palette's length says how many regions the overview can draw in
    distinguishable colours; the resident service says how many it will answer
    at all, and `infer_ground` refuses anything outside [1,3] with a 422. One
    cohort asked for 4, 6 and 8 across eleven calls and every one came back as
    an HTTP status with a documentation link -- retryable, but naming nothing
    the policy could change -- while all 291 calls at three or fewer succeeded.
    """

    from ora.agent.model_services import grounding_grasp as service
    from ora.agent.tools import grasp

    source = inspect.getsource(service.GroundingSegmentService.infer_ground)
    assert f"1 <= maximum <= {grasp.GROUNDING_REQUEST_MAX_CANDIDATES}" in source, (
        "the request bound must be the range the service actually validates"
    )
    for profile in ("robotwin_toolchain_b2_closed_noservo_v1", ROBOTWIN_SURFACE,
                    LIBERO_SURFACE):
        for item in resolve_tool_surface(profile).definitions:
            if item.get("name") != "vision_ground":
                continue
            schema = (item.get("inputSchema") or {})["properties"]["max_candidates"]
            assert schema["maximum"] == grasp.GROUNDING_REQUEST_MAX_CANDIDATES
            assert "1 to 3" in schema["description"]
