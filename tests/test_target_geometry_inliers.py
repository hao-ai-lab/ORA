"""Bound RGB-D selection reaches consumers without erasing observed handles.

Only segmentation inference is replaced. Artifact production, hashes, geometry,
policy-state binding and placement dispatch run through their real code paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from ora.agent.tools.grasp import load_segmented_target, segmented_target_geometry
from ora.agent.tools.robot import ToolError
from placement_fixtures import SelectedMask, materialize_segmented_target


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path, mask, depth):
    camera = tmp_path / "source" / "agentview"
    camera.mkdir(parents=True)
    Image.fromarray(np.full((*mask.shape, 3), 120, dtype=np.uint8)).save(camera / "rgb.png")
    np.save(camera / "depth-m.npy", depth, allow_pickle=False)
    (camera / "calibration.json").write_text(json.dumps({
        "intrinsic_cv": [[200., 0., 48.], [0., 200., 48.], [0., 0., 1.]],
        "extrinsic_cv": np.eye(4).tolist(), "depth_unit": "metres",
    }))
    artifact_root = tmp_path / "artifacts"
    y, x = np.argwhere(mask)[0]
    manifest, _ = materialize_segmented_target(
        artifact_root=artifact_root, observation_id="obs-0000", camera="agentview",
        rgb_path=camera / "rgb.png", depth_path=camera / "depth-m.npy",
        calibration_path=camera / "calibration.json", point_uv=(int(x), int(y)),
        label="selected region", backend=SelectedMask(mask),
    )
    return manifest, dict(artifact_root=artifact_root, target_ref=manifest["target_ref"],
        rgb_path=camera / "rgb.png", depth_path=camera / "depth-m.npy",
        calibration_path=camera / "calibration.json", voxel_size_m=.001)


def _body_handle_spill():
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:60, 20:60] = True
    mask[38:42, 60:86] = True
    depth = np.full(mask.shape, .8, dtype=np.float32)
    depth[38:42, 60:84] = .81  # A narrow connected handle is real observed extent.
    depth[38:42, 84:86] = 1.3  # Connected background-depth spill at its tip.
    retained = mask.copy()
    retained[38:42, 84:86] = False
    return mask, depth, retained


def test_bound_inliers_remove_depth_spill_and_preserve_connected_handle(tmp_path):
    mask, depth, retained = _body_handle_spill()
    manifest, kwargs = _source(tmp_path, mask, depth)
    files = {p: _hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    legacy = segmented_target_geometry(**kwargs)
    geometry = segmented_target_geometry(**kwargs, use_depth_inliers=True)
    assert legacy["schema"].endswith(".v1") and "input_selection" not in legacy
    assert legacy["visible_point_count"] == mask.sum()
    assert geometry["status"] == "measured"
    assert geometry["visible_point_count"] == retained.sum()
    assert geometry["visible_point_count"] == manifest["localization"]["inlier_point_count"]
    rows, columns = np.nonzero(retained)
    z = depth[rows, columns].astype(float)
    expected = np.column_stack(((columns - 48) * z / 200, (rows - 48) * z / 200, z))
    np.testing.assert_allclose(geometry["centroid_world_m"], expected.mean(axis=0))
    points = np.asarray(geometry["voxel_points_world_m"])
    np.testing.assert_allclose(points, expected)
    assert np.max(points[:, 0]) > .14  # The far end of the thin handle survives.
    assert np.max(np.asarray(legacy["voxel_points_world_m"])[:, 2]) > 1.2
    assert np.max(points[:, 2]) < .82
    binding = geometry["source_binding"]
    assert binding["retained_pixel_mask_sha256"] == hashlib.sha256(retained.astype(np.uint8).tobytes()).hexdigest()
    assert binding["retained_pixel_mask_shape"] == [96, 96]
    assert binding["observation_id"] == "obs-0000"
    assert binding["target_ref"] == manifest["target_ref"]
    for role in ("rgb", "depth", "calibration"):
        assert binding[f"{role}_sha256"] == _hash(kwargs[f"{role}_path"])
    assert {p: _hash(p) for p in files} == files


@pytest.mark.parametrize("condition,failure", [
    ("invalid_depth", "depth_valid_fraction"),
    ("depth_spill", "depth_inlier_fraction"),
    ("disconnected", "primary_component_fraction"),
])
def test_poor_selected_depth_is_unknown_without_altering_thresholds(tmp_path, condition, failure):
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:60, 20:60] = True
    depth = np.full(mask.shape, .8, dtype=np.float32)
    if condition == "invalid_depth":
        depth[20:40, 20:60] = 0
    elif condition == "depth_spill":
        depth[20:36, 20:60] = 1.3
    else:
        mask[:] = False
        mask[10:30, 10:30] = mask[60:80, 60:80] = True
    _, kwargs = _source(tmp_path, mask, depth)
    geometry = segmented_target_geometry(**kwargs, use_depth_inliers=True)
    assert geometry["status"] == "unknown"
    assert failure in geometry["quality_failures"]
    assert geometry["quality_minimums"] == {
        "depth_valid_fraction": .7, "depth_inlier_fraction": .7,
        "primary_component_fraction": .6,
    }


@pytest.mark.parametrize("role", ["rgb", "depth", "calibration"])
def test_geometry_rejects_unbound_source_bytes(tmp_path, role):
    mask, depth, _ = _body_handle_spill()
    _, kwargs = _source(tmp_path, mask, depth)
    replacement = tmp_path / "changed-camera"
    shutil.copytree(kwargs["rgb_path"].parent, replacement)
    for selected in ("rgb", "depth", "calibration"):
        kwargs[f"{selected}_path"] = replacement / kwargs[f"{selected}_path"].name
    path = kwargs[f"{role}_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source artifacts do not match"):
        segmented_target_geometry(**kwargs, use_depth_inliers=True)


# Placement consumers run on the kept surface that declares place_plan: the
# RoboTwin prepared-grasp v1 surface over its public dual-arm timeline.
CAMERA = "head_camera"


def _select(run, mask, label):
    frame = run.frames[-1]
    camera = frame["root"] / CAMERA
    y, x = np.argwhere(mask)[0]
    manifest, _ = materialize_segmented_target(
        artifact_root=run.runtime.artifact_root,
        observation_id=frame["root"].name, camera=CAMERA,
        rgb_path=camera / "rgb.png", depth_path=camera / "depth-m.npy",
        calibration_path=camera / "calibration.json", point_uv=(int(x), int(y)),
        label=label, backend=SelectedMask(mask),
    )
    return manifest["target_ref"]


def _runtime(tmp_path):
    from test_robotwin_placement_adapter import RoboTwinPublicRun

    run = RoboTwinPublicRun(tmp_path)
    run.publish({"left": "close"})
    return run


def _place_plan(run):
    return run.runtime.invoke("place_plan", run.args("left", camera=CAMERA))


def test_runtime_current_historical_and_placement_share_filtered_geometry(tmp_path):
    run = _runtime(tmp_path)
    original = run.frames[-1]["regions"][CAMERA]["object"]
    _, path, _ = load_segmented_target(artifact_root=run.runtime.artifact_root, target_ref=original)
    mask = np.load(path, allow_pickle=False).astype(bool)
    mask[46:48, 8:10] = True  # A distant secondary selected patch is not the object.
    selected = _select(run, mask, "object with secondary mask spill")
    run.frames[-1]["regions"][CAMERA]["object"] = selected
    control, root, observation, _, _ = run.runtime._current_grip_state()
    current = run.runtime._current_target_geometry(control=control, root=root,
        observation=observation, target_ref=selected)
    historical, state = run.runtime._target_geometry_and_state(target_ref=selected)
    assert current["visible_point_count"] == historical["visible_point_count"] == 253
    assert current["source_binding"] == historical["source_binding"]
    assert state["step"] == 1
    result = _place_plan(run)
    assert result["status"] == "candidates"
    record = json.loads((run.root / result["evidence"]["path"]).read_text())
    # The placement record keeps every selected point and names the filtered
    # geometry it judged quality on as its quality source.
    geometry = record["object_geometry"]
    assert "source_binding" not in geometry
    assert geometry["quality_source_binding"] == current["source_binding"]
    assert geometry["visible_point_count"] == int(mask.sum()) == 257
    assert len(record["geometry_inputs"]) == 2
    assert run.runtime._submitted_mutations == []


@pytest.mark.parametrize("role", ["object", "support"])
def test_placement_unknown_inputs_cannot_produce_candidates_or_contradictions(tmp_path, role):
    run = _runtime(tmp_path)
    mask = np.zeros((96, 96), dtype=bool)
    mask[16:32, 16:32] = mask[64:80, 64:80] = True
    selected = _select(run, mask, "ambiguous disconnected selected region")
    run.frames[-1]["regions"][CAMERA][role] = selected
    result = _place_plan(run)
    assert result["status"] == "unknown"
    assert all(check["status"] == "unknown" for check in result["checks"].values())
    assert result["reason"] == "selected_mask_depth_or_object_comparability_insufficient"
    assert result["candidates"] == []
    control, root, observation, _, _ = run.runtime._current_grip_state()
    with pytest.raises(ToolError) as error:
        run.runtime._current_target_geometry(control=control, root=root,
            observation=observation, target_ref=selected)
    assert error.value.code == "target_geometry_unavailable"
    assert run.runtime._submitted_mutations == []


@pytest.mark.parametrize("role", ["object", "support"])
def test_partial_object_is_unknown_but_observed_support_patch_is_usable(tmp_path, role):
    run = _runtime(tmp_path)
    mask = np.zeros((96, 96), dtype=bool)
    mask[0:40, 20:76] = True
    run.frames[-1]["regions"][CAMERA][role] = _select(run, mask, "image edge selected patch")
    result = _place_plan(run)
    record = json.loads((run.root / result["evidence"]["path"]).read_text())
    geometry = next(item["geometry"] for item in record["geometry_inputs"] if item["role"] == role)
    assert geometry["source_quality"]["touches_image_boundary"] is True
    assert geometry["status"] == "measured"
    if role == "object":
        assert result["status"] == "unknown" and result["candidates"] == []
    else:
        assert result["status"] == "candidates" and result["candidates"]
