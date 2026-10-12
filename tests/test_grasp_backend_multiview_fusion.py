"""The grasp proposer receives the fused cloud, and says which cameras made it.

The multi-view fusion already existed and was spent only on the review canvas:
`review_scope` said `synchronized_agentview_wrist_rgbd_fusion` while
`backend_input_scope` said `target_source_camera_rgbd_only`, so self-occluded
surfaces were invisible to the network that proposes the grasp.  These tests
pin the fused proposer input and the evidence that must travel with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.tools.grasp import materialize_grasp_query
from test_grounding_grasp_runtime import BOUNDS, _runtime


# Contract strings are pinned here as literals rather than imported, so these
# tests fail on the behaviour rather than on a missing symbol.
GRASP_BACKEND_SINGLE_VIEW_SCOPE = "target_source_camera_rgbd_only"
GRASP_BACKEND_FUSED_SCOPE = "multiview_fused_rgbd"
GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE = "target_source_camera_for_backend_only"
GRASP_BACKEND_FUSED_INPUT_SCOPE = "multiview_fused_rgbd_for_backend"
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
# The wrist fixture camera carries one depth patch at 0.95 m that appears in no
# other view, so its presence in the proposer input is unambiguous evidence
# that a second camera reached the network.
WRIST_ONLY_DEPTH_M = 0.95


class RecordingProposer:
    """A proposer double that keeps every scene cloud it was handed."""

    backend_id = "recording_grasp_proposer_test_double"
    backend_version = "v1"

    def __init__(self) -> None:
        self.scenes: list[np.ndarray] = []

    def configuration(self) -> dict[str, object]:
        return {"fixture": "records_scene_input"}

    def require_ready(self) -> dict[str, object]:
        return {"ready": True, "load_count": 1}

    def propose(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        scene = np.asarray(kwargs["scene_points_camera_m"], dtype=np.float64)
        assert scene.ndim == 2 and scene.shape[1] == 3
        self.scenes.append(scene.copy())
        return [{
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
        }], {"request_id": "recording-test", "rejected": []}


def _grounded_query_inputs(tmp_path: Path, proposer: RecordingProposer) -> dict[str, object]:
    runtime, observation_id = _runtime(tmp_path, grasp_backend=proposer)
    grounded = runtime.invoke("vision_ground", {
        "camera": "agentview", "text": "red object", "max_candidates": 1,
    })
    _, root, observation = runtime._current()
    _, rgb, depth, calibration = runtime._camera(root, observation, "agentview")
    return {
        "artifact_root": runtime.artifact_root,
        "observation_id": observation_id,
        "target_ref": grounded["candidates"][0]["target_ref"],
        "observation_root": root,
        "observation": observation,
        "rgb_path": rgb,
        "depth_path": depth,
        "calibration_path": calibration,
        "workspace_bounds_m": BOUNDS,
        "max_candidates": 1,
        "backend": proposer,
    }


def _query_root(manifest: dict, artifact_root: Path) -> Path:
    return artifact_root / "grasp-queries" / manifest["query_ref"]


def test_fused_scene_reaches_the_proposer_and_the_single_view_path_stays_labelled(
    tmp_path: Path,
) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)

    single_view, _ = materialize_grasp_query(**inputs)
    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", "wrist"),
    )

    single_scene, fused_scene = proposer.scenes
    # The wrist-only surface is invisible to the single-view proposer and
    # present for the fused one; that is the whole point of the change.
    assert not np.any(np.isclose(single_scene[:, 2], WRIST_ONLY_DEPTH_M))
    assert np.any(np.isclose(fused_scene[:, 2], WRIST_ONLY_DEPTH_M))
    assert len(fused_scene) > len(single_scene)

    single_profile = single_view["input_pointcloud"]["backend_input_profile"]
    assert single_profile["scope"] == GRASP_BACKEND_SINGLE_VIEW_SCOPE
    assert single_profile["multiview_fusion"] is False
    assert (
        single_view["input_pointcloud"]["input_scope"]
        == GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE
    )

    fused_profile = fused["input_pointcloud"]["backend_input_profile"]
    assert fused_profile["scope"] == GRASP_BACKEND_FUSED_SCOPE
    assert fused_profile["multiview_fusion"] is True
    assert fused_profile["contributing_cameras"] == ["agentview", "wrist"]
    assert fused_profile["excluded_cameras"] == []
    assert (
        fused["input_pointcloud"]["input_scope"] == GRASP_BACKEND_FUSED_INPUT_SCOPE
    )
    # A different proposer input must be a different query identity.
    assert fused["query_ref"] != single_view["query_ref"]


def test_fused_input_reports_every_contributing_camera_and_its_point_count(
    tmp_path: Path,
) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", "wrist"),
    )
    ledger = fused["input_pointcloud"]["backend_input_fusion"]

    assert [item["camera"] for item in ledger["cameras"]] == ["agentview", "wrist"]
    contributions = {
        item["camera"]: item["retained_voxel_points"] for item in ledger["cameras"]
    }
    assert all(count > 0 for count in contributions.values())
    # The reported per-camera contributions must account for the whole cloud
    # the proposer received, not just describe it.
    assert sum(contributions.values()) == fused["input_pointcloud"]["filtered_voxel_count"]
    assert sum(contributions.values()) == len(proposer.scenes[-1])

    for record in ledger["cameras"]:
        calibration = record["calibration"]
        assert np.asarray(calibration["intrinsic_cv"]).shape == (3, 3)
        assert np.asarray(calibration["extrinsic_cv_world_to_camera"]).shape == (4, 4)
        assert np.asarray(calibration["camera_to_world"]).shape == (4, 4)
        assert np.allclose(
            np.asarray(calibration["extrinsic_cv_world_to_camera"])
            @ np.asarray(calibration["camera_to_world"]),
            np.eye(4), atol=1e-9,
        )
        assert calibration["sha256"]


def test_recorded_fusion_transform_reproduces_the_exact_proposer_cloud(
    tmp_path: Path,
) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", "wrist"),
    )
    pointcloud = fused["input_pointcloud"]
    frame = pointcloud["backend_input_fusion"]["backend_point_frame"]
    assert frame["camera"] == "agentview"

    arrays = np.load(
        _query_root(fused, inputs["artifact_root"]) / pointcloud["artifact"]["path"],
        allow_pickle=False,
    )
    world = arrays["backend_scene_points_world_m"]
    world_to_camera = np.asarray(frame["world_to_backend_camera"], dtype=np.float64)
    reprojected = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    assert np.allclose(reprojected, arrays["backend_scene_points_camera_m"], atol=1e-9)
    assert np.allclose(reprojected, proposer.scenes[-1], atol=1e-9)
    assert np.allclose(
        world_to_camera @ np.asarray(frame["backend_camera_to_world"], dtype=np.float64),
        np.eye(4), atol=1e-9,
    )


@pytest.mark.parametrize(
    "camera,damage,reason",
    [
        ("wrist", "hash_drift", "rgbd_source_hash_drift"),
        ("wrist", "missing_file", "rgbd_source_file_unavailable"),
        ("wrist", "out_of_range_depth", "rgbd_depth_unusable"),
        ("wrist", "sparse_depth", "insufficient_workspace_depth_points"),
        ("nonexistent_camera", None, "camera_absent_from_observation"),
    ],
)
def test_an_unusable_camera_is_excluded_with_a_named_reason(
    tmp_path: Path, camera: str, damage: str | None, reason: str,
) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    if damage is not None:
        record = inputs["observation"]["cameras"][camera]
        depth_path = (
            inputs["observation_root"] / record["files"]["depth_numpy_m_f32"]["path"]
        )
        depth_path.chmod(0o600)
        if damage == "missing_file":
            depth_path.unlink()
        else:
            depth = np.load(depth_path, allow_pickle=False)
            if damage == "hash_drift":
                # Only the recorded content changes; the manifest hash goes stale.
                depth[:] = 1.25
            elif damage == "out_of_range_depth":
                # Beyond the fusion depth ceiling: nothing survives at all.
                depth[:] = 40.0
            else:
                # Valid, in-range depth that lands almost entirely outside the
                # workspace, leaving too few points to move the geometry.
                depth[:] = 2.4
                depth[30:33, 30:33] = 1.0
            np.save(depth_path, depth, allow_pickle=False)
            if damage != "hash_drift":
                record["files"]["depth_numpy_m_f32"]["sha256"] = hashlib.sha256(
                    depth_path.read_bytes()
                ).hexdigest()

    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", camera),
    )
    profile = fused["input_pointcloud"]["backend_input_profile"]
    assert profile["contributing_cameras"] == ["agentview"]
    assert profile["excluded_cameras"] == [{"camera": camera, "reason": reason}]
    assert all(
        item["reason"] in GRASP_BACKEND_EXCLUSION_REASONS
        for item in profile["excluded_cameras"]
    )
    # One unusable camera degrades the scope truthfully; it never leaves the
    # result claiming a fusion that did not happen.
    assert profile["scope"] == GRASP_BACKEND_SINGLE_VIEW_SCOPE
    assert profile["multiview_fusion"] is False
    assert not np.any(np.isclose(proposer.scenes[-1][:, 2], WRIST_ONLY_DEPTH_M))


def test_the_review_record_stops_asserting_a_single_view_backend_input(
    tmp_path: Path,
) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)

    single_view, _ = materialize_grasp_query(**inputs)
    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", "wrist"),
    )

    single_review = single_view["review_visualization"]
    assert single_review["status"] == "available", single_review.get("failure")
    assert single_review["backend_input_scope"] == GRASP_BACKEND_SINGLE_VIEW_SCOPE
    assert single_review["backend_cameras"] == ["agentview"]

    fused_review = fused["review_visualization"]
    assert fused_review["status"] == "available", fused_review.get("failure")
    assert fused_review["backend_input_scope"] == GRASP_BACKEND_FUSED_SCOPE
    assert fused_review["backend_cameras"] == ["agentview", "wrist"]
    assert fused_review["backend_points_by_camera"]["wrist"] > 0
    assert "only the target source camera" not in fused_review["claim_boundary"]


def test_the_target_source_camera_stays_mandatory(tmp_path: Path) -> None:
    """The target camera can never be excluded: the mask subset comes from it."""

    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    # The explicit rgb/depth/calibration arguments still bind the target
    # source, so only the fusion's own view of the observation is broken here.
    del inputs["observation"]["cameras"]["agentview"]["files"]["depth_numpy_m_f32"]

    with pytest.raises(ValueError, match="target source camera"):
        materialize_grasp_query(
            **inputs, backend_scene_cameras=("agentview", "wrist"),
        )
    assert proposer.scenes == []


def test_frozen_single_view_query_identity_is_unchanged(tmp_path: Path) -> None:
    """A caller that asks for nothing must hash exactly as it did before."""

    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    manifest, _ = materialize_grasp_query(**inputs)
    profile = manifest["identity"]["backend_input_profile"]
    assert profile == {
        "camera": "agentview",
        "max_depth_m": 2.5,
        "multiview_fusion": False,
        "robot_self_filter": False,
        "scope": GRASP_BACKEND_SINGLE_VIEW_SCOPE,
        "voxel_size_m": 0.005,
        "workspace_bounds_m": dict(BOUNDS),
    }


def test_opting_in_binds_the_contributing_sources_by_content(tmp_path: Path) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    implicit, _ = materialize_grasp_query(**inputs)
    explicit, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview",),
    )
    # Opting in binds every camera whose RGB-D determines the proposer input,
    # so the identity legitimately differs even when the camera set matches.
    sources = explicit["identity"]["backend_input_profile"]["contributing_sources"]
    assert [item["camera"] for item in sources] == ["agentview"]
    assert set(sources[0]["files"]) == {
        "rgb_lossless_png", "depth_numpy_m_f32", "calibration",
    }
    assert explicit["query_ref"] != implicit["query_ref"]
    assert np.allclose(proposer.scenes[0], proposer.scenes[1])


def test_json_serialisable_manifest_keeps_the_fusion_ledger(tmp_path: Path) -> None:
    proposer = RecordingProposer()
    inputs = _grounded_query_inputs(tmp_path, proposer)
    fused, _ = materialize_grasp_query(
        **inputs, backend_scene_cameras=("agentview", "wrist"),
    )
    stored = json.loads(
        (_query_root(fused, inputs["artifact_root"]) / "manifest.json").read_text()
    )
    assert (
        stored["input_pointcloud"]["backend_input_fusion"]
        == fused["input_pointcloud"]["backend_input_fusion"]
    )
