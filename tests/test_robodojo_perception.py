"""RoboDojo three-camera perception path and ARX X5 grasp geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ora.agent.composition.tool_surface import (
    ToolSurfaceError, _resolve_view_config,
)
from ora.agent.native.observation_projection import (
    AUXILIARY_CONTACT_SHEET_LAYOUTS, ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)
from ora.agent.tools.runtime import core as runtime_core
from ora.agent.tools.arx_x5_gripper import (
    ARX_X5_APERTURE_CAVEAT, ARX_X5_GRIPPER_GEOMETRY,
    ARX_X5_LINK6_COMPILER_IDENTITIES, ARX_X5_SOURCE_BINDING,
    ARX_X5_SOURCE_FILES, nominal_jaw_aperture_m,
)
from ora.agent.tools.geometry import fuse_rgbd_pointcloud_sources
from ora.agent.tools.grasp import (
    GRASP_COMPILER_IDENTITIES, grasp_plan_public_result, load_segmented_target,
    materialize_grasp_query, materialize_grounded_targets,
)
from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT


ROBODOJO_CAMERAS = ("head", "left_wrist", "right_wrist")
# Exact key set RobodojoSession.observe writes per camera; the agreement test
# below re-derives it from the real writer through the fake plant.
ROBODOJO_CALIBRATION_KEYS = frozenset({
    "source_camera", "intrinsic_cv", "extrinsic_cv", "cam2world_cv",
    "depth_unit", "depth_array_order", "policy_quaternion_order",
    "world_frame", "depth_kind", "finite_depth_fraction",
    "renderer_calibration",
})
ROBODOJO_CAMERA_RECORD_KEYS = frozenset({
    "source_camera", "width", "height", "mount", "calibration_dynamic",
    "modalities", "files",
})
BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
X5_ASSET_ROOT = (Path.home() / "robodojo-runtime/source/RoboDojo") / (
    ARX_X5_SOURCE_BINDING["asset_root"]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer_calibration(source: str, intrinsic: np.ndarray) -> dict[str, object]:
    """Mirror the nested numpy read-backs the session serialises per camera."""

    return {
        "source": "original_render_product_CameraParams",
        "render_product": f"/Render/RenderProduct_{source}",
        "original_robodojo_intrinsic": intrinsic.tolist(),
        "sdk_raw_aperture_intrinsic": intrinsic.tolist(),
        "parameters": {
            "cameraProjection": np.eye(4).ravel().tolist(),
            "cameraViewTransform": np.eye(4).ravel().tolist(),
            "renderProductResolution": [64, 48],
            "cameraModel": "pinhole",
            "cameraAperture": [20.955, 15.716],
            "cameraApertureOffset": [0.0, 0.0],
        },
        "cam2world_cv": np.eye(4).tolist(),
        "simulation_time_s": 2.0,
        "reference_time": {"referenceTimeNumerator": 120, "referenceTimeDenominator": 60},
        "current_physical_pose_verified": True,
    }


def _wrist_cam2world(sign: float) -> np.ndarray:
    """A wrist camera displaced sideways so its points do not dedupe with head."""

    pose = np.eye(4)
    pose[:3, 3] = [sign * 0.30, 0.05, 0.02]
    return pose


def _synthetic_robodojo_observation(
    root: Path, *, head_depth_patch: tuple[slice, slice, float] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Write the RoboDojo per-camera layout with the session's exact keys.

    ``head_depth_patch`` (rows, columns, depth) overwrites part of the head
    background, e.g. to place points below the action workspace.
    """

    observation_id = "robodojo:hang_mugs:layout-0:observation-0003"
    target = root / observation_id
    target.mkdir(parents=True)
    intrinsic = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    cameras: dict[str, object] = {}
    for public, sign in (("head", 0.0), ("left_wrist", 1.0), ("right_wrist", -1.0)):
        source = f"cam_{public}"
        rgb = np.full((48, 64, 3), [52, 58, 66], dtype=np.uint8)
        depth = np.full((48, 64), 1.2, dtype=np.float32)
        if public == "head":
            rgb[14:34, 20:40] = [200, 60, 40]
            depth[14:34, 20:40] = 0.8
            if head_depth_patch is not None:
                rows, columns, value = head_depth_patch
                depth[rows, columns] = value
        cam2world = np.eye(4) if public == "head" else _wrist_cam2world(sign)
        camera_root = target / public
        camera_root.mkdir()
        rgb_path, depth_path = camera_root / "rgb.png", camera_root / "depth-m.npy"
        calibration_path = camera_root / "calibration.json"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth, allow_pickle=False)
        calibration_path.write_text(json.dumps({
            "source_camera": source, "intrinsic_cv": intrinsic.tolist(),
            "extrinsic_cv": np.linalg.inv(cam2world).tolist(),
            "cam2world_cv": cam2world.tolist(),
            "depth_unit": "metres", "depth_array_order": "opencv_top_left_row_major",
            "policy_quaternion_order": "xyzw", "world_frame": "robodojo_episode_local",
            "depth_kind": "distance_to_image_plane", "finite_depth_fraction": 1.0,
            "renderer_calibration": _renderer_calibration(source, intrinsic),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        cameras[public] = {
            "source_camera": source, "width": 64, "height": 48,
            "mount": "fixed_world" if public == "head" else public,
            "calibration_dynamic": public != "head",
            "modalities": ["rgb", "depth", "calibration"],
            "files": {
                role: {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": _sha256(path), "bytes": path.stat().st_size,
                }
                for role, path in (
                    ("rgb_lossless_png", rgb_path),
                    ("depth_numpy_m_f32", depth_path),
                    ("calibration", calibration_path),
                )
            },
        }
    payload = {
        "schema": "ora-observation.v1", "observation_id": observation_id,
        "task": "hang_mugs", "sequence": 3, "encoding": "metric_rgbd_v1",
        "captured_unix_s": 0.0, "main_camera": "head", "cameras": cameras,
        "proprioception": {
            "end_effectors": {
                arm: {
                    "pose_xyz_xyzw": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                    "pose_frame": "robodojo_episode_local_ee_link",
                    "gripper": 1.0, "gripper_finger_qpos": [0.044, 0.044],
                    "arm_joint_qpos": [0.0] * 6, "gripper_command": 1.0,
                    "gripper_command_target": "open",
                }
                for arm in ("left", "right")
            },
            "single_arm": False,
        },
        "previous_execution_receipt": None,
        "forbidden_modalities_absent": True,
    }
    (target / "observation.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return target, payload


class FakeHeadGrounding:
    backend_id = "sam3_text_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"fixture": "robodojo_head"}

    def ground(
        self, *, rgb_u8: object, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert np.asarray(rgb_u8).shape == (48, 64, 3)
        assert text == "red mug"
        mask = np.zeros((48, 64), dtype=bool)
        mask[14:34, 20:40] = True
        return [{
            "source_index": 0, "rank": 0, "mask": mask,
            "backend_score": 0.9, "bbox_xyxy": [20, 14, 40, 34],
        }][:max_candidates], {"request_id": "ground-x5", "returned_candidate_count": 1}


class FakeX5Proposer:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert kwargs["compiler_identity"]["id"] == (
            "contact_graspnet_to_robodojo_x5_left_link6_v1"
        )
        scene = np.asarray(kwargs["scene_points_camera_m"])
        target = np.asarray(kwargs["target_points_camera_m"])
        assert scene.ndim == 2 and scene.shape[1] == 3 and len(target) == 20 * 20
        # Contact-GraspNet's Panda origin sits 103.4 mm behind the visible
        # 0.8 m surface; approach runs along camera +z into the scene.
        return [{
            "candidate_id": "grasp-0123456789abcdef01234567",
            "source_index": 0, "rank": 0,
            "source_gripper_origin_xyz_m": [0.0, 0.0, 0.6966],
            "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.6566],
            "approach_world": [0.0, 0.0, 1.0], "jaw_world": [1.0, 0.0, 0.0],
            "gripper_opening_m": 0.05, "backend_score": 0.7,
            "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
            "checks": {"workspace": "passed", "aperture": "passed",
                       "ik": "not_checked", "collision": "not_checked"},
            "execution_ready": False,
        }], {"request_id": "grasp-x5", "rejected": []}


def test_synthetic_layout_matches_the_real_writer(tmp_path: Path) -> None:
    """The synthetic key set equals what RobodojoSession.observe writes."""

    # Pytest puts tests/ on sys.path, so the backend module's fake plant is
    # importable without duplicating it.
    import test_robodojo_backend as backend_fixture
    from ora.environment.backends.robodojo.session import RobodojoSession

    plant = backend_fixture.Plant()
    session = RobodojoSession(
        config={"workspace_bounds_m": BOUNDS}, output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None), {
            "schema": "ora.robodojo-source-manifest.v1",
            "source_revisions": {"robodojo": "1" * 40},
        }),
    )
    intrinsic = np.array([[50.0, 0.0, 5.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]])
    session._calibration = lambda name: (intrinsic, np.eye(4))
    for source in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        session.calibration_readbacks[source] = {
            "original_robodojo_intrinsic": intrinsic, "cam2world_cv": np.eye(4),
            "parameters": {"renderProductResolution": np.array([10, 8])},
        }
    session.start()
    real, _ = session.observe(tmp_path / "observations")
    real_root = tmp_path / "observations" / real["observation_id"]
    synthetic_root, synthetic = _synthetic_robodojo_observation(tmp_path / "synthetic")
    for public in ROBODOJO_CAMERAS:
        real_keys = set(json.loads((real_root / public / "calibration.json").read_text()))
        synthetic_keys = set(json.loads((synthetic_root / public / "calibration.json").read_text()))
        assert real_keys == synthetic_keys == ROBODOJO_CALIBRATION_KEYS
        assert set(real["cameras"][public]) == set(synthetic["cameras"][public])
        assert set(real["cameras"][public]) == ROBODOJO_CAMERA_RECORD_KEYS
        assert set(real["cameras"][public]["files"]) == {
            "rgb_lossless_png", "depth_numpy_m_f32", "calibration",
        }
        assert np.load(real_root / public / "depth-m.npy").dtype == np.float32
    assert set(real["cameras"]) == set(ROBODOJO_CAMERAS) == set(
        ARX_X5_CONTRACT["public_cameras"]
    )
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=real_root, observation=real,
        workspace_bounds_m=BOUNDS, camera_names=ROBODOJO_CAMERAS,
        voxel_size_m=0.005, max_depth_m=2.5,
    )
    assert [item["camera"] for item in fused["sources"]] == list(ROBODOJO_CAMERAS)
    assert fused["workspace_point_count"] == 3 * 8 * 10


def test_three_camera_fusion_and_head_grounding_accept_the_layout(tmp_path: Path) -> None:
    observation_root, observation = _synthetic_robodojo_observation(tmp_path / "obs")
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root, observation=observation,
        workspace_bounds_m=BOUNDS, camera_names=ROBODOJO_CAMERAS,
        voxel_size_m=0.005, max_depth_m=2.5,
    )
    assert set(fused["source_index"].tolist()) == {0, 1, 2}
    assert all(item["retained_voxel_points"] > 0 for item in fused["sources"])
    head = observation_root / "head"
    manifest, images = materialize_grounded_targets(
        artifact_root=tmp_path / "artifacts",
        observation_id=str(observation["observation_id"]), camera="head",
        rgb_path=head / "rgb.png", depth_path=head / "depth-m.npy",
        calibration_path=head / "calibration.json",
        text="red mug", max_candidates=1, backend=FakeHeadGrounding(),
    )
    assert manifest["identity"]["camera"] == "head"
    assert manifest["identity"]["source"]["policy_observation_path"].endswith("/head")
    assert len(manifest["candidates"]) == 1 and len(images) == 2
    target, _mask, _overlay = load_segmented_target(
        artifact_root=tmp_path / "artifacts",
        target_ref=manifest["candidates"][0]["target_ref"],
    )
    assert target["localization"]["status"] == "localized"
    assert target["localization"]["center_xyz_m"][2] == pytest.approx(0.8, abs=1e-6)


def test_x5_grasp_plan_compiles_link6_and_carries_the_aperture_caveat(tmp_path: Path) -> None:
    observation_root, observation = _synthetic_robodojo_observation(tmp_path / "obs")
    head = observation_root / "head"
    common = {
        "artifact_root": tmp_path / "artifacts",
        "observation_id": str(observation["observation_id"]),
        "rgb_path": head / "rgb.png", "depth_path": head / "depth-m.npy",
        "calibration_path": head / "calibration.json",
    }
    grounded, _ = materialize_grounded_targets(
        camera="head", text="red mug", max_candidates=1,
        backend=FakeHeadGrounding(), **common,
    )
    manifest, overlay_path = materialize_grasp_query(
        target_ref=grounded["candidates"][0]["target_ref"],
        observation_root=observation_root, observation=observation,
        workspace_bounds_m=BOUNDS, max_candidates=1, backend=FakeX5Proposer(),
        compile_for_execution=True, scene_cameras=("head", "left_wrist"),
        compiler_id="contact_graspnet_to_robodojo_x5_left_link6_v1",
        execution_tool="move_pose", public_contact_pose=True, **common,
    )
    identity = manifest["identity"]["compiler"]
    assert identity["physical_jaw_aperture_calibrated"] is False
    assert identity["aperture_caveat"] == ARX_X5_APERTURE_CAVEAT
    assert manifest["review_visualization"]["status"] == "available"
    assert manifest["review_pointcloud"]["camera_order"] == ["head", "left_wrist"]
    compiled = manifest["compiled_candidates"]
    assert len(compiled) == 1 and manifest["diagnostics"]["compiler_rejected"] == []
    candidate = compiled[0]
    # link6 sits 46.7 mm behind the model origin; contact lands on the 0.8 m
    # surface (0.6966 + 0.1034), pregrasp one 40 mm axial servo back.
    assert candidate["contact_pose"]["position_m"][2] == pytest.approx(0.6499, abs=1e-6)
    assert candidate["pregrasp_pose"]["position_m"][2] == pytest.approx(0.6099, abs=1e-6)
    assert candidate["approach_delta"]["frame"] == "end_effector"
    assert candidate["approach_delta"]["translation_m"] == pytest.approx([0.04, 0.0, 0.0])
    assert candidate["approach_delta"]["rotation_rotvec_rad"] == [0.0, 0.0, 0.0]
    assert candidate["compiler_checks"]["embodiment_mapping"] == identity["id"]
    assert candidate["compiler_checks"]["aperture_calibration"] == ARX_X5_APERTURE_CAVEAT
    public = grasp_plan_public_result(manifest, overlay_path)
    assert public["compiler"] == {
        "id": identity["id"], "embodiment_descriptor_id": "arx_x5_robodojo_v1",
        "controlled_point": "end_effector",
        "controlled_point_semantics": "robodojo_link6_pose", "arm": "left",
        "physical_jaw_aperture_calibrated": False,
        "aperture_caveat": ARX_X5_APERTURE_CAVEAT,
        # The admission ceiling and the per-candidate predicted width have
        # different provenances; the caveat is about the first one only.
        "maximum_opening_source": "nominal_urdf_mesh_geometry",
        "predicted_opening_source": "contact_graspnet_panda_width_head_bin",
    }
    assert "source_binding" not in public["compiler"]
    assert public["candidates"][0]["compiler_checks"]["aperture_calibration"] == (
        ARX_X5_APERTURE_CAVEAT
    )


def test_aloha_compiled_candidates_keep_their_frozen_shape() -> None:
    """The caveat field appears only for identities that declare the flag."""

    from ora.agent.tools.grasp import (
        _compile_grasp_candidate, _public_grasp_compiler_identity,
    )

    raw = {
        "candidate_id": "grasp-0123456789abcdef01234567", "rank": 0,
        "source_gripper_origin_xyz_m": [0.0, 0.0, 0.6966],
        "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.6566],
        "approach_world": [0.0, 0.0, 1.0], "jaw_world": [1.0, 0.0, 0.0],
        "gripper_opening_m": 0.05, "backend_score": 0.7,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    aloha = GRASP_COMPILER_IDENTITIES["contact_graspnet_to_robotwin_aloha_left_grip_center_v3"]
    aloha_compiled = _compile_grasp_candidate(raw, compiler_identity=aloha)
    assert "aperture_calibration" not in aloha_compiled["compiler_checks"]
    assert set(_public_grasp_compiler_identity(aloha)) == {
        "id", "embodiment_descriptor_id", "controlled_point",
        "controlled_point_semantics", "arm",
    }
    x5 = GRASP_COMPILER_IDENTITIES["contact_graspnet_to_robodojo_x5_right_link6_v1"]
    x5_compiled = _compile_grasp_candidate(raw, compiler_identity=x5)
    assert x5_compiled["compiler_checks"]["aperture_calibration"] == ARX_X5_APERTURE_CAVEAT
    with pytest.raises(ValueError, match="exceeds embodiment limit"):
        _compile_grasp_candidate({**raw, "gripper_opening_m": 0.085}, compiler_identity=x5)


def test_x5_gripper_geometry_is_internally_consistent() -> None:
    geometry = ARX_X5_GRIPPER_GEOMETRY
    command = ARX_X5_CONTRACT["gripper_command"]
    assert geometry["command_joint_range_m"] == command["primary_joint_range"]
    assert geometry["mimic_multiplier"] == command["mimic_multiplier"]
    assert geometry["physical_jaw_aperture_calibrated"] is (
        command["physical_jaw_aperture_calibrated"]
    ) is False
    low, high = geometry["pad_inner_face_span_link6_x_m"]
    assert low < geometry["source_declared_gripper_bias_m"] < geometry[
        "contact_baseline_link6_x_m"
    ] < high
    assert nominal_jaw_aperture_m(1.0) == pytest.approx(
        geometry["nominal_maximum_aperture_m"], abs=1e-6,
    )
    assert nominal_jaw_aperture_m(0.0) == 0.0
    samples = [nominal_jaw_aperture_m(value / 10) for value in range(11)]
    assert samples == sorted(samples)
    with pytest.raises(ValueError):
        nominal_jaw_aperture_m(1.5)
    for arm, identity in ARX_X5_LINK6_COMPILER_IDENTITIES.items():
        assert GRASP_COMPILER_IDENTITIES[identity["id"]] is identity
        assert identity["arm"] == arm
        assert identity["source_origin_to_controlled_point_m"] + identity[
            "controlled_point_to_contact_baseline_m"
        ] == pytest.approx(identity["contact_graspnet_gripper_depth_m"], abs=1e-9)
        assert identity["maximum_opening_m"] < geometry["nominal_maximum_aperture_m"]
        assert identity["source_binding"] is ARX_X5_SOURCE_BINDING


def test_x5_geometry_provenance_matches_the_pinned_assets() -> None:
    if not X5_ASSET_ROOT.is_dir():
        pytest.skip("pinned RoboDojo x5 assets are not present on this host")
    for key, relative in ARX_X5_SOURCE_FILES.items():
        assert _sha256(X5_ASSET_ROOT / relative) == ARX_X5_SOURCE_BINDING[key], key
    urdf = (X5_ASSET_ROOT / "X5A.urdf").read_text(encoding="utf-8")
    origin = ARX_X5_GRIPPER_GEOMETRY["finger_joint_origin_link6_m"]
    assert f'<origin xyz="{origin["x"]} 0.024896 -0.0002436"' in urdf
    assert '<limit lower="0" upper="0.044"' in urdf
    config = (X5_ASSET_ROOT / "robot_config.yml").read_text(encoding="utf-8")
    assert "gripper_bias: 0.145" in config and "gripper_scale: [-0.01, 0.044]" in config


def test_three_camera_contact_sheet_is_one_labeled_row(tmp_path: Path) -> None:
    camera_paths: dict[str, Path] = {}
    for index, name in enumerate(ROBODOJO_CAMERAS):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (64, 48), (10 * index, 40, 60)).save(path)
        camera_paths[name] = path
    derived = materialize_auxiliary_contact_sheet(
        run_root=tmp_path, observation_step=2, camera_paths=camera_paths,
        cameras=ROBODOJO_CAMERAS,
    )
    assert (derived["width"], derived["height"]) == (64 * 3, 24 + 48)
    manifest = json.loads(Path(derived["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["identity"]["layout"] == "one_by_three_native_rgb_with_label_header"
    assert manifest["identity"]["camera_order"] == list(ROBODOJO_CAMERAS)
    with Image.open(derived["path"]) as sheet:
        # Each cell keeps its native pixels under the label header.
        assert sheet.getpixel((0, 24)) == (0, 40, 60)
        assert sheet.getpixel((64 * 2, 24)) == (20, 40, 60)
    assert AUXILIARY_CONTACT_SHEET_LAYOUTS[4][0] == "two_by_two_native_rgb_with_label_header"
    with pytest.raises(ObservationProjectionError, match="three or four unique"):
        materialize_auxiliary_contact_sheet(
            run_root=tmp_path, observation_step=3,
            camera_paths={name: camera_paths[name] for name in ROBODOJO_CAMERAS[:2]},
            cameras=ROBODOJO_CAMERAS[:2],
        )


def test_view_config_admits_three_sheet_cameras_or_no_sheet() -> None:
    base = {
        "schema": "ora.policy-view-config.v1",
        "rgb_resolution": "native", "derived_resolution": [256, 256],
        "default_observe_views": list(ROBODOJO_CAMERAS),
        "coordinate_convention": "opencv_top_left_xy",
    }
    capabilities = ["automatic_current_auxiliary_contact_sheet_v1"]
    with_sheet = _resolve_view_config({
        "view_config": {
            **base, "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
            "auxiliary_contact_sheet_cameras": list(ROBODOJO_CAMERAS),
        },
        "capabilities": capabilities,
    })
    assert with_sheet["auxiliary_contact_sheet_cameras"] == list(ROBODOJO_CAMERAS)
    without_sheet = _resolve_view_config({
        "view_config": {**base, "automatic_actual_views": ["head"]},
        "capabilities": [],
    })
    assert "auxiliary_contact_sheet_cameras" not in without_sheet
    with pytest.raises(ToolSurfaceError, match="three or four unique"):
        _resolve_view_config({
            "view_config": {
                **base, "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
                "auxiliary_contact_sheet_cameras": ["head", "left_wrist"],
            },
            "capabilities": capabilities,
        })


def test_tool_runtime_selects_robodojo_cameras_and_x5_compiler_by_capability() -> None:
    def surface(*capabilities: str) -> SimpleNamespace:
        return SimpleNamespace(supports=lambda name: name in capabilities)

    robodojo = surface(
        "dual_arm_grasp_plan_v1", runtime_core.ROBODOJO_RGBD_CAPABILITY,
        runtime_core.ARX_X5_GRASP_COMPILER_CAPABILITY,
    )
    assert runtime_core.dual_arm_grasp_scene_cameras(robodojo, "right") == ("head", "right_wrist")
    assert runtime_core.dual_arm_grasp_compiler_id(robodojo, "right") == (
        "contact_graspnet_to_robodojo_x5_right_link6_v1"
    )
    robotwin = surface(
        "dual_arm_grasp_plan_v1", "robotwin_rgbd_v1",
        "contact_graspnet_robotwin_aloha_grip_center_compiler_v3",
    )
    assert runtime_core.dual_arm_grasp_scene_cameras(robotwin, "left") == ("head_camera", "left_camera")
    assert runtime_core.dual_arm_grasp_compiler_id(robotwin, "left") == (
        "contact_graspnet_to_robotwin_aloha_left_grip_center_v3"
    )
    assert runtime_core.dual_arm_grasp_compiler_id(surface("dual_arm_grasp_plan_v1"), "left") == (
        "contact_graspnet_to_robotwin_aloha_left_end_effector_v1"
    )
    assert runtime_core.dual_arm_grasp_compiler_id(surface(), "left") is None
    assert set(ROBODOJO_CAMERAS) == runtime_core.ROBODOJO_POLICY_CAMERAS
    assert all(name in runtime_core.RETENTION_REVIEW_CAMERA_ORDER for name in ROBODOJO_CAMERAS)


# --------------------------------------------------------------------------
# grounding_candidate_depth_profile_v1: the candidate card's depth strip.
# --------------------------------------------------------------------------

def _ground_head(tmp_path: Path, artifacts: str, depth_profile: object) -> tuple[dict, list[Path]]:
    observation_root, observation = _synthetic_robodojo_observation(tmp_path / artifacts / "obs")
    head = observation_root / "head"
    return materialize_grounded_targets(
        artifact_root=tmp_path / artifacts / "artifacts",
        observation_id=str(observation["observation_id"]), camera="head",
        rgb_path=head / "rgb.png", depth_path=head / "depth-m.npy",
        calibration_path=head / "calibration.json",
        text="red mug", max_candidates=1, backend=FakeHeadGrounding(),
        depth_profile=depth_profile,
    )


def test_the_depth_profile_card_adds_a_strip_under_unchanged_crops(tmp_path: Path) -> None:
    from ora.agent.tools import grounding_preview
    from ora.agent.tools.grasp import grounding_public_result
    from ora.agent.tools.grounding_palette import grounding_candidate_colour

    released, released_paths = _ground_head(tmp_path, "released", None)
    options = {"renderer": grounding_preview.PROFILE_RENDERER, "support_line": False}
    profiled, profiled_paths = _ground_head(tmp_path, "profiled", options)
    # Target refs never carry the renderer; the grounding ref does, and only
    # when the profile is drawn.
    assert [c["target_ref"] for c in profiled["candidates"]] == [
        c["target_ref"] for c in released["candidates"]]
    assert released["identity"]["visualization"] == {
        "renderer": "grounding-candidate-inspection.v2"}
    assert profiled["identity"]["visualization"] == {
        "renderer": "grounding-candidate-inspection.v2", "depth_profile": options}
    assert profiled["grounding_ref"] != released["grounding_ref"]
    with Image.open(released_paths[1]) as image:
        released_card = np.asarray(image.convert("RGB"))
    with Image.open(profiled_paths[1]) as image:
        profiled_card = np.asarray(image.convert("RGB"))
    assert released_card.shape == (576, 1024, 3) and profiled_card.shape == (800, 1024, 3)
    # Title, both 480 px crops and their labels are the released pixels.
    assert np.array_equal(profiled_card[:576], released_card)
    colour = np.asarray(grounding_candidate_colour(0), dtype=np.uint8)
    for left in (16, 528):
        strip = profiled_card[600:776, left + 46:left + 472]
        assert np.all(strip == colour, axis=-1).sum() > 0, "the candidate's points are drawn"
    record = profiled["visualization"]["candidate_previews"][0]["depth_profile"]
    localization = profiled["candidates"][0]["localization"]
    assert record["status"] == "rendered"
    assert record["candidate_depth_points"] == localization["inlier_point_count"] == 20 * 20
    assert record["window_center_xyz_m"] == pytest.approx(localization["center_xyz_m"])
    largest = max(localization["extent_xyz_m"] + [localization["largest_extent_m"]])
    assert record["vertical_span_m"] == pytest.approx(max(0.10, 2.5 * largest))
    assert all(view["candidate_points_in_window"] == 400 for view in record["views"])
    assert record["support_z_m"] is None and record["options"] == options
    # The policy-visible preview record keeps its keys; only the size moves.
    released_public = grounding_public_result(released, released_paths[0])["candidate_previews"]
    profiled_public = grounding_public_result(profiled, profiled_paths[0])["candidate_previews"]
    assert [set(item) for item in profiled_public] == [set(item) for item in released_public]
    assert profiled_public[0]["render_dimensions"] == [1024, 800]
    assert profiled_public[0]["resampling"] == released_public[0]["resampling"]
    assert profiled_public[0]["crop_xyxy"] == released_public[0]["crop_xyxy"]


def _profile_card(*, support_line: bool, localization: dict) -> tuple[np.ndarray, dict]:
    from ora.agent.tools import grounding_preview

    rgb = np.full((48, 64, 3), 60, dtype=np.uint8)
    depth = np.full((48, 64), 1.2, dtype=np.float64)
    depth[14:34, 20:40] = 0.8
    mask = np.zeros((48, 64), dtype=bool)
    mask[14:34, 20:40] = True
    calibration = Path(localization.pop("_calibration"))
    image, metadata = grounding_preview.render_candidate_preview(
        rgb_u8=rgb, text="red mug",
        candidate={"rank": 0, "mask": mask, "bbox_xyxy": [20, 14, 40, 34],
                   "target_ref": "target-" + "0" * 24, "backend_score": 0.9},
        depth_profile=grounding_preview.profile_input(
            {"renderer": grounding_preview.PROFILE_RENDERER, "support_line": support_line},
            depth_m=depth, calibration_path=calibration, localization=localization),
    )
    return np.asarray(image), metadata["depth_profile"]


def test_the_support_line_follows_the_support_capability_and_the_window(tmp_path: Path) -> None:
    observation_root, _observation = _synthetic_robodojo_observation(tmp_path / "obs")
    calibration = str(observation_root / "head" / "calibration.json")
    localized = {"status": "localized", "center_xyz_m": [0.0, 0.0, 0.8],
                 "extent_xyz_m": [0.2, 0.2, 0.0], "support_z_m": 0.75}
    card, record = _profile_card(support_line=True, localization={**localized, "_calibration": calibration})
    assert record["support_z_m"] == 0.75 and record["views"][0]["support_line_drawn"] is True
    assert record["views"][1]["support_line_drawn"] is False, "the top view has no height axis"
    support_row = 580 + 22 + int((0.8 + 0.25 - 0.75) / record["metres_per_pixel"])
    assert np.all(card[support_row, 62:500] == (255, 176, 64), axis=-1).sum() > 100
    # A surface without target_support_height_v1 never draws the withheld number.
    _card, withheld = _profile_card(support_line=False, localization={**localized, "_calibration": calibration})
    assert withheld["support_z_m"] is None
    assert not any(view["support_line_drawn"] for view in withheld["views"])
    # Out of the window: recorded, not drawn.
    _card, outside = _profile_card(support_line=True, localization={
        **localized, "support_z_m": 0.2, "_calibration": calibration})
    assert outside["support_z_m"] == 0.2 and not outside["views"][0]["support_line_drawn"]
    # No localized depth: the strip says so instead of drawing an empty window.
    _card, missing = _profile_card(support_line=True, localization={
        "status": "insufficient_depth", "center_xyz_m": None, "_calibration": calibration})
    assert missing == {"renderer": "grounding-candidate-depth-profile.v1", "status": "unavailable",
                       "options": {"renderer": "grounding-candidate-depth-profile.v1",
                                   "support_line": True},
                       "reason": "candidate_not_localized"}


def test_only_a_surface_declaring_the_depth_profile_draws_it(tmp_path: Path) -> None:
    import base64
    import io
    from dataclasses import replace

    import test_robodojo_b2_parity as parity
    from ora.agent.composition.tool_surface import resolve_tool_surface
    from ora.agent.tools import grounding_preview

    live = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    assert grounding_preview.options(live) == {
        "renderer": "grounding-candidate-depth-profile.v1", "support_line": True}
    # The same Dojo surface without the capability keeps the released card.
    released = replace(live, capabilities=tuple(
        item for item in live.capabilities if item != grounding_preview.CAPABILITY))
    assert grounding_preview.options(released) is None
    assert grounding_preview.options(resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")) is None
    sizes = {}
    for name, surface in (("released", released), ("live", live)):
        runtime = parity.tool_runtime(tmp_path / name)
        runtime.surface = surface
        runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
        result = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
        with Image.open(io.BytesIO(base64.b64decode(result["images_base64"][1]))) as image:
            sizes[name] = image.size
        assert result["candidate_previews"][0]["render_dimensions"] == list(sizes[name])
    assert sizes == {"released": (1024, 576), "live": (1024, 800)}



# --------------------------------------------------------------------------
# observed_extent_scene_review_v1: the global review's observed-extent cell.
# --------------------------------------------------------------------------

#: Tighter than BOUNDS so the synthetic scene has points beside and below it:
#: the head background reaches x +/-0.48 at 1.2 m and the patch below sits at
#: z 0.5, under the 0.72 m tabletop bound the live RoboDojo workspace uses.
REVIEW_WORKSPACE = {"x": [-0.4, 0.4], "y": [-0.4, 0.4], "z": [0.72, 1.4]}
BELOW_PATCH = (slice(0, 12), slice(0, 18), 0.5)  # well above the 0.5 % percentile trim


def _plan_review(tmp_path: Path, name: str, **flags: object) -> tuple[dict, Path]:
    observation_root, observation = _synthetic_robodojo_observation(
        tmp_path / name / "obs", head_depth_patch=BELOW_PATCH)
    head = observation_root / "head"
    common = {
        "artifact_root": tmp_path / name / "artifacts",
        "observation_id": str(observation["observation_id"]),
        "rgb_path": head / "rgb.png", "depth_path": head / "depth-m.npy",
        "calibration_path": head / "calibration.json",
    }
    grounded, _ = materialize_grounded_targets(
        camera="head", text="red mug", max_candidates=1, backend=FakeHeadGrounding(), **common)
    manifest, overlay_path = materialize_grasp_query(
        target_ref=grounded["candidates"][0]["target_ref"],
        observation_root=observation_root, observation=observation,
        workspace_bounds_m=REVIEW_WORKSPACE, max_candidates=1, backend=FakeX5Proposer(),
        compile_for_execution=True, scene_cameras=("head", "left_wrist"),
        backend_scene_cameras=("head", "left_wrist"),
        compiler_id="contact_graspnet_to_robodojo_x5_left_link6_v1",
        execution_tool="move_pose", public_contact_pose=True, **flags, **common,
    )
    review = manifest["review_visualization"]
    assert review["status"] == "available"
    return manifest, overlay_path.parent / review["artifact"]["path"]


def test_the_observed_extent_cell_shows_what_the_workspace_panels_clip(tmp_path: Path) -> None:
    released, released_path = _plan_review(tmp_path, "released")
    extended, extended_path = _plan_review(
        tmp_path, "extended", observed_extent=True, review_policy_visible=True)
    assert extended["identity"] == {
        **released["identity"], "review_renderer": "grasp-review-observed-extent.v1"}
    assert extended["query_ref"] != released["query_ref"]
    released_review = released["review_visualization"]
    review = extended["review_visualization"]
    # The three workspace panels keep their bounds, points and resolution.
    assert review["views"] == released_review["views"]
    assert review["scene_point_count"] == released_review["scene_point_count"]
    assert all(view["horizontal_bounds_m"] == REVIEW_WORKSPACE[view["horizontal_axis"]]
               for view in review["views"])
    assert review["render_resolution"] == released_review["render_resolution"] == [1168, 768]
    assert "observed_extent" not in released_review
    assert (released_review["policy_visible"], review["policy_visible"]) == (False, True)
    extent = review["observed_extent"]
    assert extent["status"] == "available"
    assert extent["voxel_size_m"] == 0.01 and extent["max_depth_m"] == 2.5
    assert extent["camera_order"] == ["head", "left_wrist"]
    # The patch at z 0.5 is counted and drawn, and widens the view below 0.72.
    assert extent["points_below_workspace"]["z"] > 0
    assert extent["view_bounds_m"]["z"][0] <= 0.5 + 0.01
    assert extent["points_above_workspace"]["x"] + extent["points_below_workspace"]["x"] > 0
    assert extent["points_outside_workspace"] >= extent["points_below_workspace"]["z"]
    for axis, (low, high) in REVIEW_WORKSPACE.items():
        assert extent["view_bounds_m"][axis][0] <= low and extent["view_bounds_m"][axis][1] >= high
        for corner, centre in ((0, -1), (1, 1)):
            # Every depth-capped point of either camera lies inside the fusion box.
            assert (extent["fusion_box_m"][axis][corner] - extent["percentile_bounds_m"][axis][corner]) * centre >= 0
    assert extent["fused_voxel_count"] == sum(extent["points_by_source"].values())
    assert extent["top_view_points_drawn"] > 0 and extent["front_view_points_drawn"] > 0
    with Image.open(released_path) as image:
        released_canvas = np.asarray(image.convert("RGB"))
    with Image.open(extended_path) as image:
        canvas = np.asarray(image.convert("RGB"))
    assert canvas.shape == released_canvas.shape == (768, 1168, 3)
    # Panel pixels below each title row are the released ones; the cell is new.
    for left, top in ((395, 11), (11, 395), (395, 395)):
        assert np.array_equal(canvas[top + 24:top + 360, left:left + 360],
                              released_canvas[top + 24:top + 360, left:left + 360])
    assert not np.array_equal(canvas[11:371, 11:371], released_canvas[11:371, 11:371])


def test_an_unusable_extent_source_leaves_an_unavailable_cell(tmp_path: Path) -> None:
    from ora.agent.tools import scene_extent_review

    observation_root, observation = _synthetic_robodojo_observation(tmp_path / "obs")
    result = scene_extent_review.cell(
        observation_root=observation_root, observation=observation,
        camera_names=("head", "missing_camera"), max_depth_m=2.5,
        workspace_bounds_m=REVIEW_WORKSPACE, target_world=np.zeros((0, 3)), grip_sites=[])
    assert result["image"].size == (360, 360)
    assert result["record"]["status"] == "unavailable"
    assert result["record"]["failure"]["type"] == "KeyError"
    assert scene_extent_review.identity(False) == {}
    assert scene_extent_review.captions(
        False, scene_sources=[{"camera": "head"}], backend_cameras=["head", "left_wrist"],
        camera="head") == ("fused RGB-D · agentview + wrist", "backend input · head only")
    assert scene_extent_review.captions(
        True, scene_sources=[{"camera": "head"}, {"camera": "left_wrist"}],
        backend_cameras=["head", "left_wrist"], camera="head") == (
        "fused RGB-D · head + left_wrist", "backend input · head + left_wrist")
