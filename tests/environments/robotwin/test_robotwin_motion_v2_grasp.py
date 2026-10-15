"""Robotwin motion v2 grasp contract regressions."""

from __future__ import annotations
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.evidence.trajectory import _forbidden_path
from ora.agent.tools.runtime.core import _grasp_hover_offset_m
from ora.agent.tools.grasp import AlohaHybridGraspProposer, ALOHA_END_EFFECTOR_COMPILER_IDENTITIES, ALOHA_GRIP_CENTER_COMPILER_IDENTITIES, ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES, CONTACT_GRASPNET_MODEL_IDENTITY, PcaTopDownGraspProposer, SAM3_MODEL_IDENTITY, ContactGraspNetApiProposer, Sam3ApiSegmenter, _compile_grasp_candidate, _public_grasp_compiler_identity, grasp_plan_public_result
from ora.contracts.pose import quaternion_matrix
from ora.environment.backends.robotwin.session import RoboTwinLiveSession


def test_parallel_jaw_variants_rotate_about_aloha_approach_axis() -> None:
    raw = np.asarray([0.2, -0.3, 0.1, 0.9], dtype=np.float64)
    raw /= np.linalg.norm(raw)
    base = np.asarray(quaternion_matrix(raw.tolist()))
    variants = RoboTwinLiveSession._orientation_variants_xyzw(
        raw.tolist(), policy="parallel_jaw_equivalents",
    )
    assert len(variants) == 6
    rotations = {}
    for identifier, xyzw in variants:
        rotation = np.asarray(quaternion_matrix(xyzw))
        assert rotation[:, 0] == pytest.approx(base[:, 0])
        rotations[identifier] = rotation
    assert rotations["variant-03-approach-x-+180deg"][:, 1] == pytest.approx(
        -base[:, 1]
    )


def test_view_config_contact_sheet_cameras_are_declared_not_hardcoded(
    tmp_path: Path,
) -> None:
    """The sheet's source cameras come from the manifest, not shared code."""

    from ora.agent.composition.tool_surface import _resolve_view_config
    from ora.agent.native.observation_projection import (
        ObservationProjectionError,
        materialize_auxiliary_contact_sheet,
    )
    from PIL import Image

    base = {
        "schema": "ora.policy-view-config.v1",
        "rgb_resolution": "native",
        "derived_resolution": [256, 256],
        "automatic_actual_views": ["main_camera", "auxiliary_contact_sheet"],
        "default_observe_views": ["main_camera"],
        "coordinate_convention": "opencv_top_left_xy",
    }
    capabilities = ["automatic_current_auxiliary_contact_sheet_v1"]
    custom = ["cam_a", "cam_b", "cam_c", "cam_d"]
    resolved = _resolve_view_config({
        "view_config": {
            **base, "auxiliary_contact_sheet_cameras": custom,
        },
        "capabilities": capabilities,
    })
    assert resolved["auxiliary_contact_sheet_cameras"] == custom

    three = _resolve_view_config({
        "view_config": {**base, "auxiliary_contact_sheet_cameras": custom[:3]},
        "capabilities": capabilities,
    })
    assert three["auxiliary_contact_sheet_cameras"] == custom[:3]
    for invalid in (custom[:2], custom + ["cam_e"], ["cam_a"] * 3):
        with pytest.raises(ToolSurfaceError, match="three or four unique"):
            _resolve_view_config({
                "view_config": {**base, "auxiliary_contact_sheet_cameras": invalid},
                "capabilities": capabilities,
            })
    with pytest.raises(ToolSurfaceError, match="requires the sheet"):
        _resolve_view_config({
            "view_config": {
                **base,
                "automatic_actual_views": ["main_camera"],
                "auxiliary_contact_sheet_cameras": custom,
            },
            "capabilities": capabilities,
        })

    # The projection composes whatever four cameras the surface declares.
    run_root = tmp_path
    camera_paths: dict[str, Path] = {}
    for index, name in enumerate(custom):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (32, 24), (10 * index, 20, 30)).save(path)
        camera_paths[name] = path
    derived = materialize_auxiliary_contact_sheet(
        run_root=run_root,
        observation_step=0,
        camera_paths=camera_paths,
        cameras=tuple(custom),
    )
    manifest = json.loads(
        Path(derived["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["identity"]["camera_order"] == custom

    with pytest.raises(ObservationProjectionError, match="four unique"):
        materialize_auxiliary_contact_sheet(
            run_root=run_root, observation_step=1,
            camera_paths=camera_paths, cameras=("cam_a", "cam_a", "cam_b", "cam_c"),
        )


def test_grip_center_grasp_compiler_targets_public_tcp() -> None:
    compiler = ALOHA_GRIP_CENTER_COMPILER_IDENTITIES["left"]
    candidate = {
        "candidate_id": "candidate-0",
        "rank": 0,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "source_gripper_origin_xyz_m": [0.0, 0.0, 1.0],
        "source_pregrasp_origin_xyz_m": [-0.04, 0.0, 1.0],
        "approach_world": [1.0, 0.0, 0.0],
        "jaw_world": [0.0, 1.0, 0.0],
        "gripper_opening_m": 0.04,
        "backend_score": 0.8,
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    compiled = _compile_grasp_candidate(candidate, compiler_identity=compiler)
    assert compiled["pregrasp_pose"]["position_m"][0] == pytest.approx(0.08)
    # Under the promoted v7 lineage the complete compiler identity is
    # published; assert the semantic fact on the identity itself.
    assert compiler["controlled_point_semantics"] == (
        "parallel_jaw_grip_center_tcp"
    )


def test_aloha_rgbd_grasp_staging_follows_its_capability() -> None:
    from dataclasses import replace

    staged = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")
    assert staged.supports("robotwin_aloha_public_rgbd_topdown_grasp_v1")
    released = replace(staged, capabilities=tuple(
        item for item in staged.capabilities
        if item != "robotwin_aloha_public_rgbd_topdown_grasp_v1"
    ))
    assert _grasp_hover_offset_m(released, grasp_plan_enabled=True) == 0.04
    assert _grasp_hover_offset_m(staged, grasp_plan_enabled=True) == 0.10


def test_aloha_compiler_emits_contact_pose_only_with_public_capability() -> None:
    candidate = {
        "candidate_id": "candidate-contact-pose",
        "rank": 0,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "source_gripper_origin_xyz_m": [0.0, 0.0, 1.0],
        "source_pregrasp_origin_xyz_m": [-0.10, 0.0, 1.0],
        "approach_world": [1.0, 0.0, 0.0],
        "jaw_world": [0.0, 1.0, 0.0],
        "gripper_opening_m": 0.04,
        "backend_score": 0.8,
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    identity = ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES["left"]
    default = _compile_grasp_candidate(candidate, compiler_identity=identity)
    assert "contact_pose" not in default
    compiled = _compile_grasp_candidate(
        candidate, compiler_identity=identity, public_contact_pose=True,
    )
    contact = compiled["contact_pose"]
    assert contact["position_m"][0] == pytest.approx(
        compiled["pregrasp_pose"]["position_m"][0] + 0.10
    )
    assert contact["position_m"][1] == pytest.approx(0.0)
    assert contact["position_m"][2] == pytest.approx(1.0)
    assert contact["orientation_xyzw"] == (
        compiled["pregrasp_pose"]["orientation_xyzw"]
    )


def test_v3_aloha_topdown_candidate_uses_visible_surface_and_clearance() -> None:
    x, y = np.meshgrid(
        np.linspace(-0.035, 0.035, 30),
        np.linspace(-0.018, 0.018, 20),
    )
    z = np.linspace(0.74, 0.77, x.size)
    points = np.column_stack([x.ravel(), y.ravel(), z])
    candidates, diagnostics = PcaTopDownGraspProposer().propose(
        scene_points_camera_m=points,
        target_points_camera_m=points,
        camera_to_world=np.eye(4),
        target_ref="target-0123456789abcdef01234567",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45],
        },
        max_candidates=4,
        compiler_identity=ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES[
            "left"
        ],
    )
    assert candidates
    candidate = candidates[0]
    low = np.percentile(points, 2, axis=0)
    high = np.percentile(points, 98, axis=0)
    filtered = points[np.all((points >= low) & (points <= high), axis=1)]
    expected_contact_z = np.percentile(filtered[:, 2], 95) + 0.008
    assert candidate["grasp_center_xyz_m"][2] == pytest.approx(
        expected_contact_z, abs=5e-4,
    )
    assert candidate["pregrasp_center_xyz_m"][2] == pytest.approx(
        expected_contact_z + 0.10, abs=5e-4,
    )
    compiled = _compile_grasp_candidate(
        candidate,
        compiler_identity=ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES[
            "left"
        ],
    )
    assert compiled["approach_delta"]["translation_m"] == pytest.approx(
        [0.10, 0.0, 0.0]
    )
    assert diagnostics["warning"].startswith("geometric candidates")


def test_aloha_hybrid_grasp_keeps_public_topdown_and_learned_families() -> None:
    class LearnedBackend:
        backend_id = "learned-test"
        backend_version = "v1"

        @staticmethod
        def configuration() -> dict[str, str]:
            return {"fixture": "learned"}

        @staticmethod
        def propose(**_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return [{
                "candidate_id": "grasp-learned",
                "rank": 0,
                "pose_convention": (
                    "contact_graspnet_panda_gripper_origin_opencv_v2"
                ),
                "source_gripper_origin_xyz_m": [0.0, 0.0, 0.80],
                "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.90],
                "approach_world": [0.0, 0.0, -1.0],
                "jaw_world": [0.0, 1.0, 0.0],
                "gripper_opening_m": 0.04,
                "backend_score": 0.9,
                "checks": {"workspace": "passed", "aperture": "passed"},
            }], {
                # The identity and ledger a conforming learned backend returns;
                # the hybrid has to carry both to the top level or its own
                # result schema rejects the call.
                "model": {
                    "family": "learned-test",
                    "source_commit": "0" * 40,
                    "checkpoint_sha256": "1" * 64,
                    "config_sha256": "2" * 64,
                },
                "request_id": "grasp-request-0001",
                "rejected": [{"candidate_id": "grasp-learned-2",
                              "reasons": ["aperture_exceeds_gripper"]}],
            }

    # Wider than the compiler's admission bound, which is derived from the
    # pinned source rather than fixed at 55 mm, so the ledger this test reads
    # still has something in it.
    x, y = np.meshgrid(
        np.linspace(-0.06, 0.06, 30),
        np.linspace(-0.018, 0.018, 20),
    )
    points = np.column_stack([
        x.ravel(), y.ravel(), np.linspace(0.74, 0.77, x.size),
    ])
    hybrid = AlohaHybridGraspProposer(contact_backend=LearnedBackend())
    candidates, diagnostics = hybrid.propose(
        scene_points_camera_m=points,
        target_points_camera_m=points,
        camera_to_world=np.eye(4),
        target_ref="target-0123456789abcdef01234567",
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45],
        },
        max_candidates=3,
        compiler_identity=ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES[
            "left"
        ],
    )
    assert candidates[0]["pose_convention"] == (
        "pca_topdown_parallel_jaw_center_v1"
    )
    assert any(item["candidate_id"] == "grasp-learned" for item in candidates)
    assert diagnostics["candidate_family_counts"]["learned_available"] == 1
    assert diagnostics["candidate_family_counts"][
        "public_rgbd_topdown_available"
    ] >= 1
    # The contract this backend is judged against, checked against what it
    # actually emits. Without this the two were never compared: candidates
    # were produced and then discarded at the format gate, three calls in a
    # row, with a non-retryable internal_tool_error and no way for a policy
    # to tell that the geometry had been found.
    from ora.agent.composition.tool_surface import (
        resolve_tool_surface, validate_schema_instance,
    )

    schema = resolve_tool_surface(
        "robotwin_toolchain_b2_closed_noservo_v1"
    ).result_schemas["grasp_plan"]
    # The compiler fields are added by the caller after the backend returns.
    emitted = dict(diagnostics)
    emitted.setdefault("compiler_rejected", [])
    emitted.setdefault("compiler_returned_candidate_count", len(candidates))
    validate_schema_instance(
        emitted, {**schema["$defs"]["diagnostics"], "$defs": schema["$defs"]},
    )
    # The ledger is the union: a candidate either family rejected is one this
    # call considered and did not return.
    assert emitted["request_id"] == "grasp-request-0001"
    assert any(
        item["candidate_id"] == "grasp-learned-2" for item in emitted["rejected"]
    )
    pca_rejected = [item for item in emitted["rejected"] if "required_opening_m" in item]
    assert pca_rejected
    assert all(item["required_opening_m"] > item["maximum_opening_m"] for item in pca_rejected)
    assert all(item["limit_semantics"] == "compiler_admission_not_measured_physical_aperture"
               for item in pca_rejected)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_aloha_grasp_compiler_is_arm_bound_and_public_end_effector(arm: str) -> None:
    candidate = {
        "candidate_id": "candidate-0",
        "rank": 0,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "source_gripper_origin_xyz_m": [0.0, 0.0, 1.0],
        "source_pregrasp_origin_xyz_m": [-0.04, 0.0, 1.0],
        "approach_world": [1.0, 0.0, 0.0],
        "jaw_world": [0.0, 1.0, 0.0],
        "gripper_opening_m": 0.04,
        "backend_score": 0.8,
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    compiled = _compile_grasp_candidate(
        candidate,
        compiler_identity=ALOHA_END_EFFECTOR_COMPILER_IDENTITIES[arm],
    )
    assert compiled["approach_delta"]["frame"] == "end_effector"
    assert compiled["approach_delta"]["translation_m"] == [0.04, 0.0, 0.0]
    assert compiled["compiler_checks"]["embodiment_mapping"].endswith(
        f"_{arm}_end_effector_v1"
    )
    assert _forbidden_path(compiled) is None


def test_complete_grasp_plan_result_is_policy_safe(tmp_path: Path) -> None:
    from PIL import Image

    compiler = ALOHA_END_EFFECTOR_COMPILER_IDENTITIES["left"]
    public_compiler = _public_grasp_compiler_identity(compiler)
    assert set(public_compiler) == {
        "id",
        "embodiment_descriptor_id",
        "controlled_point",
        "controlled_point_semantics",
        "arm",
    }
    overlay = tmp_path / "overlay.png"
    Image.new("RGB", (8, 6)).save(overlay)
    candidate = {
        "candidate_id": "candidate-0",
        "rank": 0,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "source_gripper_origin_xyz_m": [0.0, 0.0, 1.0],
        "source_pregrasp_origin_xyz_m": [-0.04, 0.0, 1.0],
        "approach_world": [1.0, 0.0, 0.0],
        "jaw_world": [0.0, 1.0, 0.0],
        "gripper_opening_m": 0.04,
        "backend_score": 0.8,
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    result = grasp_plan_public_result({
        "status": "candidates",
        "query_ref": "grasp-query-" + "a" * 24,
        "identity": {
            "observation_id": "observation-0",
            "target_ref": "target-" + "b" * 24,
            "camera": "head_camera",
            "compiler": compiler,
        },
        "backend": {"id": "grasp_api", "version": "v1", "config_sha256": "c" * 64},
        "candidates": [candidate],
        "compiled_candidates": [
            _compile_grasp_candidate(candidate, compiler_identity=compiler),
        ],
        "diagnostics": {
            "returned_candidate_count": 1,
            "rejected": [],
            "warning": "requires motion qualification",
        },
        "overlay": {"sha256": "d" * 64},
    }, overlay)
    result["arm"] = "left"
    assert _forbidden_path(result) is None
    assert result["compiler"] == public_compiler


def test_contact_graspnet_filters_with_selected_aloha_compiler() -> None:
    def encoded(value: np.ndarray) -> str:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "grasp_proposals",
                "warmup_complete": True,
                "load_count": 1,
                "instance_id": "aloha-cgn-test",
                "pid": 456,
                "adapter_sha256": "b" * 64,
                "started_unix_s": 1.0,
                "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        last_request: dict | None = None

        def do_POST(self) -> None:  # noqa: N802
            assert self.path == "/v1/grasp-proposals"
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            type(self).last_request = request
            pose_camera = np.eye(4, dtype=np.float32)
            pose_camera[:3, :3] = np.asarray([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
            pose_camera[:3, 3] = [0.0, 0.0, 1.0]
            body = json.dumps({
                "schema": "ora.model-service.grasp-proposals.v1",
                "request_id": "aloha-compiler-request",
                "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
                "service_identity": {
                    "schema": "ora.model-service-response-identity.v1",
                    "instance_id": "aloha-cgn-test",
                    "pid": 456,
                    "started_unix_s": 1.0,
                    "adapter_sha256": "b" * 64,
                },
                "poses_camera_npy_base64": encoded(pose_camera[None]),
                "scores_npy_base64": encoded(np.asarray([0.9], dtype=np.float32)),
                "openings_m_npy_base64": encoded(
                    np.asarray([0.05], dtype=np.float32),
                ),
                "contact_points_camera_npy_base64": encoded(
                    np.zeros((1, 3), dtype=np.float32),
                ),
                "metrics": {"queue_ms": 0.1, "inference_ms": 1.0, "total_ms": 1.1},
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
    proposer = ContactGraspNetApiProposer(
        endpoint=f"http://127.0.0.1:{server.server_port}", hover_offset_m=0.04,
    )
    try:
        candidates, _diagnostics = proposer.propose(
            scene_points_camera_m=np.zeros((16, 3), dtype=np.float32),
            target_points_camera_m=np.zeros((16, 3), dtype=np.float32),
            camera_to_world=np.eye(4, dtype=np.float32),
            target_ref="target-" + "a" * 24,
            workspace_bounds_m={
                "x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.9, 1.1],
            },
            max_candidates=1,
            compiler_identity=ALOHA_END_EFFECTOR_COMPILER_IDENTITIES["left"],
        )
        assert len(candidates) == 1
        assert candidates[0]["checks"]["workspace"] == "passed"
        compiled = _compile_grasp_candidate(
            candidates[0],
            compiler_identity=ALOHA_END_EFFECTOR_COMPILER_IDENTITIES["left"],
        )
        assert compiled["pregrasp_pose"]["position_m"][0] == pytest.approx(-0.04)
    finally:
        proposer.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sam3_text_grounding_service_validates_ranked_masks() -> None:
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
                "instance_id": "robotwin-sam3-test",
                "pid": 123,
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
            assert request["text"] == "medium block"
            masks = np.zeros((2, 8, 8), dtype=np.uint8)
            masks[0, 1:5, 1:4] = 1
            masks[1, 2:7, 5:7] = 1
            body = json.dumps({
                "schema": "ora.model-service.ground-text.v1",
                "request_id": "grounding-request",
                "model": dict(SAM3_MODEL_IDENTITY),
                "service_identity": {
                    "schema": "ora.model-service-response-identity.v1",
                    "instance_id": "robotwin-sam3-test",
                    "pid": 123,
                    "started_unix_s": 1.0,
                    "adapter_sha256": "a" * 64,
                },
                "masks_npy_base64": encoded(masks),
                "scores_npy_base64": encoded(np.asarray([0.8, 0.3], dtype=np.float32)),
                "boxes_xyxy_npy_base64": encoded(np.asarray(
                    [[1, 1, 4, 5], [5, 2, 7, 7]], dtype=np.float32,
                )),
                "metrics": {"queue_ms": 0.1, "inference_ms": 1.0, "total_ms": 1.1},
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
    client = Sam3ApiSegmenter(endpoint=f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.require_ready()["instance_id"] == "robotwin-sam3-test"
        candidates, diagnostics = client.ground(
            rgb_u8=np.zeros((8, 8, 3), dtype=np.uint8),
            text="medium block",
            max_candidates=2,
        )
        assert [candidate["rank"] for candidate in candidates] == [0, 1]
        assert candidates[0]["bbox_xyxy"] == [1, 1, 4, 5]
        assert diagnostics["model"] == SAM3_MODEL_IDENTITY
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_contact_graspnet_service_binding_detects_instance_replacement() -> None:
    class Handler(BaseHTTPRequestHandler):
        instance_id = "robotwin-grasp-test"

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
        "tool_surface": "robotwin_grasp_plan_dual_arm_v1",
        "endpoint": endpoint,
        "instance_id": "robotwin-grasp-test",
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
        assert client.require_ready()["instance_id"] == "robotwin-grasp-test"
        Handler.instance_id = "replacement-instance"
        with pytest.raises(ValueError, match="bound instance_id"):
            client.require_ready()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
