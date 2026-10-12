"""Private segmentation and read-only geometric grasp-query contracts."""

from __future__ import annotations

import hashlib
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
pytestmark = pytest.mark.usefixtures("ready_model_services")


from ora.agent.composition import resolve_tool_surface
from ora.agent.context import ContextProjector
from ora.agent.native.session import NativePlanner
from ora.agent.tools.grasp import (
    CONTACT_GRASPNET_MODEL_IDENTITY,
    SAM3_MODEL_IDENTITY,
    ContactGraspNetApiProposer,
    DepthPromptSegmenter,
    PcaTopDownGraspProposer,
    Sam3ApiSegmenter,
)
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
#: The maintained LIBERO-Pro default: resident segmentation and declared result schemas.
DEFAULT_SURFACE = "libero_placement_direct_overhead_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_agentview_with_object(
    observation: dict, observation_root: Path,
) -> None:
    camera_root = observation_root / "agentview"
    rgb_path = camera_root / "rgb.png"
    depth_path = camera_root / "depth-m.npy"
    calibration_path = camera_root / "calibration.json"

    rgb = np.full((64, 64, 3), [48, 52, 60], dtype=np.uint8)
    rgb[20:44, 24:40] = [210, 48, 32]
    Image.fromarray(rgb).save(rgb_path, format="PNG")
    depth = np.full((64, 64), 1.2, dtype=np.float32)
    depth[20:44, 24:40] = 0.8
    np.save(depth_path, depth, allow_pickle=False)
    calibration_path.write_text(json.dumps({
        "intrinsic_cv": [[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "cam2world_gl": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
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
    (observation_root / "observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime(
    tmp_path: Path, *, tool_surface_profile: str = DEFAULT_SURFACE,
    segmentation_backend: object | None = None,
) -> tuple[ManipulationToolRuntime, str]:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "environment", "grasp-segmentation", 0, max_actions=4,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    observation_root = source / observation["observation_id"]
    _replace_agentview_with_object(observation, observation_root)

    control = tmp_path / "control"
    projector = ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, PROJECT,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=observation_root,
        task_instruction="segment the red object and inspect possible grasps",
        remaining_budget={"robot_decisions": 4, "model_calls": 12},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile=tool_surface_profile,
        controlled_end_effector="right",
        segmentation_backend=segmentation_backend or DepthPromptSegmenter(),
        grasp_backend=PcaTopDownGraspProposer(),
    )
    return runtime, observation["observation_id"]


def test_segmentation_help_successor_validates_real_read_and_lifecycle_results(
    tmp_path: Path,
) -> None:
    # The default surface's observe also projects descriptor-verified robot
    # facts that this Synthetic fixture does not carry; its declared observe
    # result is exercised with real facts in test_public_robot_facts.py.
    runtime, _observation_id = _runtime(tmp_path)
    assert runtime.invoke("report_issue", {
        "message": "The fixture records one concrete interface diagnostic.",
    }) == {"success": True}
    assert runtime.invoke("check_task", {}) == {
        "success": False,
        "reason": "task_checker_unavailable",
        "retryable": True,
    }


def test_native_causal_projection_follows_declared_public_result_schema() -> None:
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=resolve_tool_surface(DEFAULT_SURFACE))
    visible = planner._causal_tool_result("vision_segment", {
        "status": "candidates",
        "observation_id": "obs-1",
        "camera": "agentview",
        "candidates": [{
            "target_ref": "target-1",
            "mask_summary": {"pixel_count": 123},
            "localization": {
                "status": "localized", "center_xyz_m": [0.1, 0.2, 0.3],
            },
            "overlay": {"sha256": "abc", "width": 64, "height": 64},
        }],
        "backend": {"id": "sam3"},
        "_runtime": {"private": "not projected"},
    })
    candidate = visible["candidates"][0]
    assert candidate["target_ref"] == "target-1"
    assert candidate["localization"]["center_xyz_m"] == [0.1, 0.2, 0.3]
    assert candidate["mask_summary"] == {"pixel_count": 123}
    assert candidate["overlay"] == {"width": 64, "height": 64}
    assert visible["status"] == "candidates"
    assert visible["observation_id"] == "obs-1"
    # backend is provenance the result schema does not declare, so it stays out.
    assert "backend" not in visible
    assert "sha256" not in json.dumps(visible)
    assert "_runtime" not in visible

    planner.tools = SimpleNamespace(
        surface=resolve_tool_surface("native_atomic_causal_v1")
    )
    task_check = planner._causal_tool_result("check_task", {
        "success": True,
        "task_success": False,
    })
    # This surface has no declared result schema; its task-check fallback
    # still reports only the official task verdict.
    assert task_check == {"task_success": False}


def test_faithful_task_check_and_finish_gate_follow_surface_capability(
    tmp_path: Path,
) -> None:
    runtime, _observation_id = _runtime(
        tmp_path, tool_surface_profile="native_atomic_causal_v1",
    )
    unavailable = runtime.invoke("check_task", {})
    assert unavailable == {
        "success": False,
        "reason": "task_checker_unavailable",
        "retryable": True,
    }
    with pytest.raises(ToolError, match="success requires check_task=true"):
        runtime.invoke("finish_episode", {"outcome": "success"})
    (runtime.control_root / "policy-task-check.json").write_text(
        json.dumps({"task_success": True}), encoding="utf-8",
    )
    assert runtime.invoke("check_task", {}) == {
        "success": True, "task_success": True,
    }


def test_resident_sam_binding_accepts_the_sam_only_pose_v2_surface() -> None:
    binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": "segmentation_localize_v2",
        "endpoint": "http://127.0.0.1:8114",
        "instance_id": "model-instance-test",
        "pid": 123,
        "started_unix_s": 1.0,
        "adapter_sha256": "a" * 64,
        "model": dict(SAM3_MODEL_IDENTITY),
        "load_count": 1,
        "warmup_complete": True,
        "qualification": {
            "sha256": "b" * 64,
            "status": "passed",
            "created_unix_s": 1.0,
        },
    }
    segmenter = Sam3ApiSegmenter(expected_binding=binding)
    try:
        assert segmenter.expected_binding["tool_surface"] == (
            "segmentation_localize_v2"
        )
    finally:
        segmenter.close()


def test_segmentation_runtime_binds_one_ready_resident_model(
    tmp_path: Path,
) -> None:
    class ReadySegmenter:
        backend_id = "ready_segmenter_test_double"
        backend_version = "v1"

        def __init__(self) -> None:
            self.readiness_calls = 0

        def configuration(self) -> dict[str, str]:
            return {"fixture": "resident_ready"}

        def require_ready(self) -> dict[str, object]:
            self.readiness_calls += 1
            return {
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "segmentation",
                "instance_id": "model-instance-test",
                "load_count": 1,
                "warmup_complete": True,
            }

        def segment(
            self, *, rgb_u8: object, depth_m: object,
            point_uv: tuple[int, int],
        ) -> tuple[np.ndarray, dict[str, object]]:
            del rgb_u8, depth_m, point_uv
            mask = np.zeros((64, 64), dtype=bool)
            mask[20:44, 24:40] = True
            return mask, {"fixture": True}

    backend = ReadySegmenter()
    runtime, _observation_id = _runtime(
        tmp_path, segmentation_backend=backend,
    )
    assert backend.readiness_calls == 1
    events = [
        json.loads(line) for line in runtime.event_path.read_text().splitlines()
    ]
    binding = next(
        event for event in events
        if event["type"] == "resident_model_services_bound"
    )
    assert binding["tool_surface"] == DEFAULT_SURFACE
    assert binding["services"]["segmentation"]["load_count"] == 1


def test_segmentation_runtime_refuses_an_unready_resident_model(
    tmp_path: Path,
) -> None:
    class ColdSegmenter:
        backend_id = "cold_segmenter_test_double"
        backend_version = "v1"

        def configuration(self) -> dict[str, str]:
            return {"fixture": "resident_cold"}

        def require_ready(self) -> dict[str, object]:
            raise ValueError("segmentation service is not warm")

    with pytest.raises(ValueError, match="not warm"):
        _runtime(
            tmp_path, segmentation_backend=ColdSegmenter(),
        )


def test_panda_aperture_gate_returns_no_candidate_for_oversize_target() -> None:
    x, y = np.meshgrid(np.linspace(-0.06, 0.06, 12), np.linspace(-0.06, 0.06, 12))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.8)])
    candidates, diagnostics = PcaTopDownGraspProposer().propose(
        scene_points_camera_m=points,
        target_points_camera_m=points,
        camera_to_world=np.eye(4),
        target_ref="target-0123456789abcdef01234567",
        workspace_bounds_m=BOUNDS,
        max_candidates=4,
    )
    assert candidates == []
    assert len(diagnostics["rejected"]) == 4
    assert all(
        "aperture_exceeds_panda_limit" in rejected["reasons"]
        for rejected in diagnostics["rejected"]
    )


def test_resident_api_clients_validate_identity_mask_and_grasp_arrays() -> None:
    def encoded(value: np.ndarray) -> str:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    class Handler(BaseHTTPRequestHandler):
        health_service = "segmentation"
        instance_id = "model-instance-test"
        replace_after_health = False
        degenerate_grasp = False
        grasp_requests: list[dict[str, object]] = []

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path != "/healthz":
                self.send_error(404)
                return
            service = type(self).health_service
            model = (
                SAM3_MODEL_IDENTITY if service == "segmentation"
                else CONTACT_GRASPNET_MODEL_IDENTITY
            )
            response = {
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": service,
                "model": dict(model),
                "warmup_complete": True,
                "adapter_sha256": "a" * 64,
                "instance_id": type(self).instance_id,
                "pid": 123,
                "load_count": 1,
                "started_unix_s": 1.0,
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # Freeze the health body first, then switch identity before its
            # bytes reach the client. ThreadingHTTPServer may serve the next
            # POST concurrently as soon as the complete body has been sent.
            if type(self).replace_after_health:
                type(self).instance_id = "replacement-after-health"
                type(self).replace_after_health = False
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            if self.path == "/v1/segment-point":
                assert request["schema"].endswith("segment-point-request.v1")
                mask = np.zeros((6, 6), dtype=np.uint8)
                mask[1, 1:3] = 1
                mask[4:6, 3:6] = 1
                response = {
                    "schema": "ora.model-service.segment-point.v1",
                    "request_id": "perception-segment-test",
                    "model": dict(SAM3_MODEL_IDENTITY),
                    "service_identity": {
                        "schema": (
                            "ora.model-service-response-identity.v1"
                        ),
                        "instance_id": type(self).instance_id,
                        "pid": 123,
                        "started_unix_s": 1.0,
                        "adapter_sha256": "a" * 64,
                    },
                    "score": 0.95,
                    "mask_npy_base64": encoded(mask),
                    "metrics": {
                        "queue_ms": 0.1, "inference_ms": 10.0, "total_ms": 10.1,
                    },
                }
            elif self.path == "/v1/grasp-proposals":
                assert request["schema"].endswith("grasp-proposals-request.v1")
                type(self).grasp_requests.append(request)
                pose = np.eye(4, dtype=np.float32)[None]
                pose[0, :3, 3] = [0.0, 0.0, 0.8]
                if type(self).degenerate_grasp:
                    pose[0, :3, 2] = 0.0
                response = {
                    "schema": "ora.model-service.grasp-proposals.v1",
                    "request_id": "perception-grasp-test",
                    "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
                    "service_identity": {
                        "schema": (
                            "ora.model-service-response-identity.v1"
                        ),
                        "instance_id": type(self).instance_id,
                        "pid": 123,
                        "started_unix_s": 1.0,
                        "adapter_sha256": "a" * 64,
                    },
                    "poses_camera_npy_base64": encoded(pose),
                    "scores_npy_base64": encoded(np.asarray([0.75], dtype=np.float32)),
                    "openings_m_npy_base64": encoded(
                        np.asarray([0.04], dtype=np.float32),
                    ),
                    "contact_points_camera_npy_base64": encoded(
                        np.asarray([[0.0, 0.0, 0.9034]], dtype=np.float32),
                    ),
                    "metrics": {
                        "queue_ms": 0.2, "inference_ms": 20.0, "total_ms": 20.2,
                    },
                }
            else:
                self.send_error(404)
                return
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
    endpoint = f"http://127.0.0.1:{server.server_port}"
    segmenter = Sam3ApiSegmenter(endpoint=endpoint)
    proposer = ContactGraspNetApiProposer(endpoint=endpoint)
    try:
        Handler.health_service = "segmentation"
        segment_health = segmenter.require_ready()
        assert segment_health["warmup_complete"] is True
        mask, diagnostics = segmenter.segment(
            rgb_u8=np.zeros((6, 6, 3), dtype=np.uint8),
            depth_m=np.ones((6, 6), dtype=np.float32),
            point_uv=(1, 1),
        )
        assert int(mask.sum()) == 2
        assert diagnostics["model_score"] == pytest.approx(0.95)
        assert diagnostics["mask_normalization"] == {
            "method": "positive_prompt_8_connected_component_v1",
            "raw_component_count": 2,
            "raw_pixel_count": 8,
            "prompt_component_pixel_count": 2,
            "retained_fraction": pytest.approx(0.25),
        }
        binding = {
            "schema": "ora.segmentation-service-binding.v1",
            "tool_surface": "segmentation_localize_v1",
            "endpoint": endpoint,
            "instance_id": "model-instance-test",
            "pid": 123,
            "started_unix_s": 1.0,
            "adapter_sha256": "a" * 64,
            "model": dict(SAM3_MODEL_IDENTITY),
            "load_count": 1,
            "warmup_complete": True,
            "qualification": {
                "sha256": "b" * 64,
                "status": "passed",
                "created_unix_s": 1.0,
            },
        }
        bound_segmenter = Sam3ApiSegmenter(
            endpoint=endpoint, expected_binding=binding,
        )
        try:
            assert bound_segmenter.require_ready()["instance_id"] == (
                "model-instance-test"
            )
            Handler.instance_id = "replacement-instance"
            with pytest.raises(ValueError, match="bound instance_id"):
                bound_segmenter.segment(
                    rgb_u8=np.zeros((6, 6, 3), dtype=np.uint8),
                    depth_m=np.ones((6, 6), dtype=np.float32),
                    point_uv=(1, 1),
                )
        finally:
            Handler.instance_id = "model-instance-test"
            bound_segmenter.close()
        points = np.column_stack([
            np.linspace(-0.02, 0.02, 16),
            np.zeros(16),
            np.full(16, 0.8),
        ])
        transform = np.eye(4)
        transform[0, 3] = 0.1
        Handler.health_service = "grasp_proposals"
        grasp_health = proposer.require_ready()
        assert grasp_health["load_count"] == 1
        candidates, diagnostics = proposer.propose(
            scene_points_camera_m=points,
            target_points_camera_m=points,
            camera_to_world=transform,
            target_ref="target-0123456789abcdef01234567",
            workspace_bounds_m=BOUNDS,
            max_candidates=1,
        )
        assert diagnostics["model"] == CONTACT_GRASPNET_MODEL_IDENTITY
        assert candidates[0]["source_gripper_origin_xyz_m"] == pytest.approx(
            [0.1, 0.0, 0.8],
        )
        assert candidates[0]["source_pregrasp_origin_xyz_m"] == pytest.approx(
            [0.1, 0.0, 0.72],
        )
        assert candidates[0]["pose_convention"] == (
            "contact_graspnet_panda_gripper_origin_opencv_v2"
        )
        assert candidates[0]["gripper_opening_m"] == pytest.approx(0.04)
        assert candidates[0]["execution_ready"] is False
        assert diagnostics["_canonical_backend_output"][
            "contact_points_camera"
        ][0] == pytest.approx([0.0, 0.0, 0.9034])
        assert diagnostics["seed_policy"] == (
            "canonical_scene_target_pointcloud_v1"
        )
        assert Handler.grasp_requests[-1]["max_candidates"] == 32
        assert diagnostics["raw_candidate_limit"] == 32
        first_seed = Handler.grasp_requests[-1]["seed"]
        _same_candidates, same_diagnostics = proposer.propose(
            scene_points_camera_m=points,
            target_points_camera_m=points,
            camera_to_world=transform,
            target_ref="target-fedcba9876543210fedcba98",
            workspace_bounds_m=BOUNDS,
            max_candidates=1,
        )
        assert Handler.grasp_requests[-1]["seed"] == first_seed
        assert same_diagnostics["inference_seed"] == first_seed
        changed_points = points.copy()
        changed_points[0, 0] += 0.001
        proposer.propose(
            scene_points_camera_m=changed_points,
            target_points_camera_m=points,
            camera_to_world=transform,
            target_ref="target-fedcba9876543210fedcba98",
            workspace_bounds_m=BOUNDS,
            max_candidates=1,
        )
        assert Handler.grasp_requests[-1]["seed"] != first_seed

        Handler.degenerate_grasp = True
        degenerate_candidates, degenerate_diagnostics = proposer.propose(
            scene_points_camera_m=points,
            target_points_camera_m=points,
            camera_to_world=transform,
            target_ref="target-fedcba9876543210fedcba98",
            workspace_bounds_m=BOUNDS,
            max_candidates=1,
        )
        Handler.degenerate_grasp = False
        assert degenerate_candidates == []
        assert degenerate_diagnostics["rejected"][0]["reasons"] == [
            "degenerate_gripper_axis",
        ]

        bound_grasp_binding = {
            "schema": "ora.grasp-proposal-service-binding.v1",
            "tool_surface": "grasp_plan_v1",
            "endpoint": endpoint,
            "instance_id": "model-instance-test",
            "pid": 123,
            "started_unix_s": 1.0,
            "adapter_sha256": "a" * 64,
            "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            "load_count": 1,
            "warmup_complete": True,
            "qualification": {
                "sha256": "b" * 64,
                "status": "passed",
                "created_unix_s": 1.0,
            },
        }
        Handler.instance_id = "model-instance-test"
        bound_proposer = ContactGraspNetApiProposer(
            endpoint=endpoint, expected_binding=bound_grasp_binding,
        )
        try:
            Handler.replace_after_health = True
            with pytest.raises(ValueError, match="response differs from bound"):
                bound_proposer.propose(
                    scene_points_camera_m=points,
                    target_points_camera_m=points,
                    camera_to_world=transform,
                    target_ref="target-fedcba9876543210fedcba98",
                    workspace_bounds_m=BOUNDS,
                    max_candidates=1,
                )
        finally:
            Handler.instance_id = "model-instance-test"
            Handler.replace_after_health = False
            bound_proposer.close()
    finally:
        segmenter.close()
        proposer.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    with pytest.raises(ValueError, match="loopback"):
        Sam3ApiSegmenter(endpoint="https://models.example.com")
