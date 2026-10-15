"""Control plane viewer contract regressions."""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
import pytest
from ora.environment.replay_worker import ReplayWorkerError, _FrameWriter, _apply_robotwin_control_target, _capture_steps, _needs_final_capture, _uniform_capture_steps, _libero_qualified_controller_binding, _recorded_control_step_count, _robotwin_replay_needs_planner, _robotwin_replay_route
from experiments.runtime_v1 import run as experiment
from viewer import app as viewer
from viewer.libero_pro_catalog import CATALOG_SCHEMA as LIBERO_PRO_CATALOG_SCHEMA, LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION, MIGRATED_FROM_COMMIT as LIBERO_MIGRATED_FROM_COMMIT, ROBUSTNESS_CASES as LIBERO_ROBUSTNESS_CASES, TASKS as LIBERO_PRO_TASKS
from viewer.robotwin_catalog import CATALOG_SCHEMA, MIGRATED_FROM_COMMIT, ROBOTWIN_REVISION, TASKS
from viewer.registry import discover_run_roots, register_run, visible_runs
from viewer.trajectory import ANALYSIS_SCHEMA, _controller_replay_source, _continuous_execution_source, _load_run_analysis, _render_continuous_execution_frame, _render_trajectory_frame, compile_raw_trajectory, compile_trajectory, compile_trajectory_video, raw_trajectory_event
from viewer.replay import ViewerReplayError, ensure_controller_replay
from test_control_plane import (
    ROOT,
)


def test_viewer_start_page_is_allowlisted_and_opens_the_selected_catalog() -> None:
    viewer_args = viewer.build_parser().parse_args([])
    assert viewer_args.page == "runs"
    assert viewer.build_parser().parse_args([
        "--page", "libero-pro",
    ]).page == "libero-pro"
    assert viewer.build_parser().parse_args([
        "--page", "libero",
    ]).page == "libero"
    assert viewer.browser_start_url(
        host="127.0.0.1", port=5012, page="libero",
    ) == "http://127.0.0.1:5012/gallery/libero"
    assert viewer.browser_start_url(
        host="127.0.0.1", port=5012, page="libero-pro",
    ) == "http://127.0.0.1:5012/gallery/libero-pro"
    assert viewer.browser_start_url(
        host="0.0.0.0", port=5012, page="robotwin",
    ) == "http://127.0.0.1:5012/gallery/robotwin"
    assert viewer.browser_start_url(
        host="::1", port=5012, page="runs",
    ) == "http://[::1]:5012/"


def test_viewer_lists_resolved_runs_and_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    run = runs / "example"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({
        "experiment": "viewer-test", "config": "synthetic-preflight",
        "milestone": "M0", "task": "synthetic", "seed": 0,
        "policy": {"kind": "scripted"}, "policy_context": {},
        "source": {"commit": "deadbeef1234567890"},
    }))
    (run / "launch.json").write_text(json.dumps({"created_unix_s": 1786149350.0}))
    (run / "status.json").write_text(json.dumps({"state": "completed", "official_success": None}))
    compiled = run / "compiled-policy"
    compiled.mkdir()
    (compiled / "episode-prefix.md").write_text("compiled episode prefix v1")
    state_steps = run / "policy-state/steps"
    state_steps.mkdir(parents=True)
    (run / "policy-state/manifest.json").write_text(json.dumps({
        "steps": [{"step": 0, "path": "policy-state/steps/step-0000.json"}]
    }))
    (state_steps / "step-0000.json").write_text(json.dumps({
        "step": 0, "cause": "reset", "main_camera": "head_camera", "cameras": {}
    }))
    bundles = run / "context-bundles"
    bundles.mkdir()
    (bundles / "call-0000.json").write_text(json.dumps({
        "call": 0, "message": {"kind": "initial_user_message"}
    }))
    (run / "context-manifest.json").write_text(json.dumps({
        "memory_system": {"active": False, "provider": "NoOpMemoryProvider"}
    }))
    native = run / "native"
    native.mkdir()
    (native / "agent-session.json").write_text(json.dumps({
        "backend": "codex_app_server_v2", "process_invocations": 1,
        "process_restarts": 0, "thread_id": "thread-1", "calls": [],
        "compaction_events": [], "rollovers": [],
    }))
    (run / "runtime-metrics.json").write_text(json.dumps({
        "context": {
            "peak_request_input_tokens": 83481,
            "model_context_window_tokens": 258400,
            "peak_request_input_utilization": 0.323069,
            "observed_input_headroom_tokens": 174919,
        },
        "workload": {
            "input_tokens": 7141287, "cached_input_tokens": 6912768,
            "uncached_input_tokens": 228519, "output_tokens": 15731,
        },
        "harness_input": {
            "text": {"later_turn_utf8_bytes": {"p50": 1541, "p95": 1545}},
            "images": {"count": 60},
        },
        "latency": {
            "episode_wall_s": 896.973,
            "agent_turn_s": {"p50": 11.646, "p95": 20.788},
        },
    }))

    def libero_run(
        run_id: str, *, experiment: str, task_id: int, seed: int,
        official_success: bool,
    ) -> Path:
        root = runs / run_id
        root.mkdir()
        (root / "plan.json").write_text(json.dumps({
            "experiment": experiment, "config": f"condition-seed-{seed}",
            "milestone": "LIBERO", "task": f"libero_object_swap-t{task_id:02d}",
            "seed": seed, "evidence_tier": "behavioral",
            "runtime": {
                "kind": "libero_pro", "suite": "libero_object_swap",
                "task_id": task_id,
            },
            "policy": {"kind": "native_api", "model": "model-test"},
            "source": {"commit": f"source-{run_id}"},
        }))
        (root / "status.json").write_text(json.dumps({
            "state": "completed", "official_success": official_success,
        }))
        (root / "evaluation.json").write_text(json.dumps({
            "status": "valid", "official_success": official_success,
        }))
        (root / "seal.json").write_text("{}")
        return root

    task_success = libero_run(
        "task-success", experiment="context-treatment", task_id=0, seed=0,
        official_success=True,
    )
    (task_success / "runtime-metrics.json").write_text(json.dumps({
        "schema": "ora.runtime-metrics.v1",
        "run_id": "task-success", "measurement_status": "complete",
        "measurement_quality": {
            "request_context": "provider_reported_with_window",
            "image_inventory": "exact",
            "provider_cache_telemetry": "reported",
        },
        "context": {
            "exchange_count": 4, "peak_request_input_tokens": 16000,
            "model_context_window_tokens": 32000,
            "peak_request_input_utilization": 0.5,
            "measurement": "provider_reported_exchange_input",
            "request_input_tokens": {"mean": 12000, "p95": 15500},
            "projection": {
                "context_retention": "test_context_v1",
                "epoch_transition_count": 1,
            },
        },
        "harness_input": {"images": {
            "count": 10, "measurement": "exact_inline_data_url_inventory",
            "per_request": {"mean": 2.5, "p95": 4, "max": 4},
        }},
        "cache": {
            "token_cache_read_ratio": 0.75,
            "call_cache_read_hit_rate": 0.5,
            "cache_read_input_tokens": 36000,
            "input_tokens_with_cache_read_telemetry": 48000,
            "measurement": "provider_reported_cache_telemetry",
        },
    }))
    task_failure = libero_run(
        "task-failure", experiment="atomic-treatment", task_id=0, seed=1,
        official_success=False,
    )
    other_task = libero_run(
        "other-task", experiment="other-treatment", task_id=2, seed=0,
        official_success=False,
    )
    indexed = [run, task_success, task_failure, other_task]
    monkeypatch.setattr(viewer, "iter_run_roots", lambda: iter(indexed))

    def resolve(run_id: str) -> Path:
        if run_id != "example":
            raise RuntimeError("not found")
        return run

    monkeypatch.setattr(viewer, "resolve_run_root", resolve)
    derived_video = tmp_path / "trajectory.mp4"
    derived_video.write_bytes(b"derived-video")
    monkeypatch.setattr(
        viewer, "compile_trajectory_video",
        lambda _root, *, replay_mode="review", allow_replay_compile=True: (
            derived_video,
            {"schema": "ora.viewer-trajectory-video.v1"},
        ),
    )
    client = viewer.app.test_client()
    assert b"example" not in client.get("/").data
    registry_page = client.get("/?view=all&experiment=viewer-test")
    assert b"example" in registry_page.data
    assert b"viewer-test" in registry_page.data
    assert b"2026-08-08" in registry_page.data
    assert b"synthetic-preflight" in registry_page.data
    assert b"Raw trace" in registry_page.data
    assert b"Composition" in registry_page.data
    assert b"Diagnostics" in registry_page.data
    task_registry = client.get(
        "/?view=attempts&backend=libero_pro&task="
        "libero_pro%3Alibero_object_swap%3At00"
    )
    assert b"task-success" in task_registry.data
    assert b"task-failure" in task_registry.data
    assert b"other-task" not in task_registry.data
    assert b"/task/libero_pro/libero_object_swap/0/runs" in task_registry.data
    successful_registry = client.get(
        "/?view=all&backend=libero_pro&state=completed&outcome=success"
    )
    assert b"All experiments" in successful_registry.data
    assert b"LIBERO / LIBERO-Pro" in successful_registry.data
    assert b"Official success" in successful_registry.data
    assert b"task-success" in successful_registry.data
    assert b"task-failure" not in successful_registry.data
    failed_registry = client.get("/?view=all&outcome=failure")
    assert b"task-failure" in failed_registry.data
    assert b"task-success" not in failed_registry.data
    detail = client.get("/run/example")
    assert detail.status_code == 200
    assert b"Episode notebook" in detail.data
    assert b"Episode outline" in detail.data
    assert b"Execution details" in detail.data
    assert b"Raw trajectory" in detail.data
    assert b"Composition" in detail.data
    composition = client.get("/run/example/composition")
    assert composition.status_code == 200
    assert b"Exact recorded configuration" in composition.data
    assert b"compiled episode prefix v1" in composition.data
    assert client.get("/api/run/example/composition").json["schema"] == (
        "ora.policy-config-inspection.v1"
    )
    trajectory = client.get("/api/run/example/trajectory")
    assert trajectory.status_code == 200
    assert trajectory.json["schema"] == "ora.viewer-trajectory.v1"
    assert len(trajectory.json["steps"]) == 1
    assert client.get("/run/example/trajectory.mp4").data == b"derived-video"
    assert client.get(
        "/run/example/trajectory.mp4?replay_mode=unknown"
    ).status_code == 400
    assert client.get("/api/run/example/trajectory-video").json["schema"].endswith("trajectory-video.v1")
    raw_detail = client.get("/run/example/raw")
    assert b"compiled episode prefix v1" in raw_detail.data
    assert b"Policy-state timeline" in raw_detail.data
    assert b"initial_user_message" in raw_detail.data
    assert b"NoOpMemoryProvider" in raw_detail.data
    assert b"Unified causal timeline" in raw_detail.data
    assert b"Context occupancy" in raw_detail.data
    assert b"83,481" in raw_detail.data
    assert b"896.973 s" in raw_detail.data
    raw_trajectory = client.get("/run/example/trajectory/raw")
    assert raw_trajectory.status_code == 200
    assert b"Raw trajectory" in raw_trajectory.data
    assert b"Experiment ID" in raw_trajectory.data
    assert b"viewer-test" in raw_trajectory.data
    assert b"2026-08-08" in raw_trajectory.data
    assert b"deadbeef1234" in raw_trajectory.data
    assert b"Exact JSONL line" in raw_trajectory.data
    assert client.get("/api/run/example/trajectory/raw").json["schema"] == (
        "ora.viewer-raw-trajectory.v1"
    )
    assert client.get("/api/run/example/trajectory/raw/unknown/1").status_code == 404
    experiment_page = client.get("/experiment/viewer-test/runs")
    assert experiment_page.status_code == 200
    assert b"synthetic-preflight" in experiment_page.data
    assert b"2026-08-08" in experiment_page.data
    assert b"Raw trace" in experiment_page.data
    assert b"Configuration catalog" in experiment_page.data
    assert b"All experiments / runs" in experiment_page.data
    declaration = ROOT / "experiments/runtime_v1/experiment.yaml"
    monkeypatch.setattr(viewer, "experiment_definition", lambda experiment_id: (
        (declaration, {"configs": {"synthetic-config": {"policy": "scripted"}}})
        if experiment_id == "viewer-config-test" else (None, {})
    ))
    config_page = client.get("/experiment/viewer-config-test/configs")
    assert config_page.status_code == 200
    assert b"Tracked experiment declaration" in config_page.data
    assert b"synthetic-config" in config_page.data
    assert b"experiments/runtime_v1/experiment.yaml" in config_page.data
    task_cohort = client.get("/task/libero_pro/libero_object_swap/0/runs")
    assert task_cohort.status_code == 200
    assert b"Cross-experiment task cohort" in task_cohort.data
    assert b"task-success" in task_cohort.data
    assert b"task-failure" in task_cohort.data
    assert b"context-treatment" in task_cohort.data
    assert b"atomic-treatment" in task_cohort.data
    assert b"other-task" not in task_cohort.data
    assert b"not one matched treatment" in task_cohort.data
    assert b"Context workload comparison" in task_cohort.data
    assert b"12.0k" in task_cohort.data
    assert b"2.5" in task_cohort.data
    assert b"75.0%" in task_cohort.data
    assert b"test_context_v1" in task_cohort.data
    seed_zero = client.get(
        "/task/libero_pro/libero_object_swap/0/runs?seed=0"
    )
    assert b"task-success" in seed_zero.data
    assert b"task-failure" not in seed_zero.data
    failures = client.get(
        "/task/libero_pro/libero_object_swap/0/runs?outcome=failure"
    )
    assert b"task-success" not in failures.data
    assert b"task-failure" in failures.data
    assert client.get(
        "/task/libero_pro/libero_object_swap/9/runs"
    ).status_code == 404
    assert client.get("/run/../secret").status_code == 404


def test_viewer_compiles_public_state_chain_into_decision_trajectory(tmp_path: Path) -> None:
    run = tmp_path / "trajectory-run"
    steps = run / "policy-state/steps"
    image = run / "policy-observations/obs-0/agentview/rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    steps.mkdir(parents=True)
    action = {
        "action": "target_state", "action_id": "action-1",
        "arguments": {"arm": "right", "pose": {"position_m": [0.1, 0.2, 0.3]}},
    }
    receipt = {
        "action_id": "action-1", "consumed": True, "pose_reached": True,
        "phase_receipts": [{"phase": "pose", "status": "reached"}],
        "control_health": {
            "status": "degraded", "codes": ["servo_target_not_reached"],
            "target_error_before_m": 0.05, "target_error_after_m": 0.03,
        },
    }
    common_camera = {"agentview": {"files": {"rgb_lossless_png": {
        "path": "policy-observations/obs-0/agentview/rgb.png",
    }}}}
    (steps / "step-0000.json").write_text(json.dumps({
        "step": 0, "cause": "reset", "main_camera": "agentview",
        "observation": {"id": "obs-0"}, "cameras": common_camera,
        "remaining_budget": {"robot_decisions": 2},
    }))
    (steps / "step-0001.json").write_text(json.dumps({
        "step": 1, "cause": "after_action", "main_camera": "agentview",
        "observation": {"id": "obs-1"}, "cameras": common_camera,
        "previous_public_action": action,
        "previous_execution_receipt": {
            "path": "receipts/action-0000.json",
            "sha256": "receipt-sha256",
            "receipt": receipt,
        },
    }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
        {"step": 1, "path": "policy-state/steps/step-0001.json"},
    ]}))
    (run / "plan.json").write_text(json.dumps({
        "experiment": "test", "task": "pick",
        "source": {"commit": "source-review"},
    }))
    (run / "result.json").write_text(json.dumps({"rounds": [{
        "round_index": 0, "official_success": True, "decisions": 1,
        "model_calls": 1, "policy_state_steps": {"start": 0, "end_exclusive": 2},
    }]}))
    trajectory = compile_trajectory(run)
    assert trajectory["projection"] == "read_only_derived"
    assert len(trajectory["steps"]) == 2
    assert trajectory["steps"][0]["action"]["position_m"] == [0.1, 0.2, 0.3]
    assert trajectory["steps"][0]["receipt"]["reached"] is True
    assert trajectory["steps"][0]["receipt"]["control_health"]["codes"] == [
        "servo_target_not_reached"
    ]
    assert trajectory["steps"][0]["round_index"] == 0
    assert trajectory["rounds"][0]["official_success"] is True
    assert trajectory["steps"][0]["images"][0]["path"].endswith("rgb.png")
    assert _render_trajectory_frame(run, trajectory, 0).size == (1280, 720)

    analysis_path = tmp_path / "viewer-analysis.json"
    analysis_path.write_text(json.dumps({
        "schema": ANALYSIS_SCHEMA, "run_id": run.name,
        "source_commit": "source-review", "title": "Exact review",
        "incidents": [{"id": "I1", "steps": [0]}],
    }))
    analysis = _load_run_analysis(
        run, {"source": {"commit": "source-review"}},
        candidates=[analysis_path],
    )
    assert analysis is not None
    assert analysis["title"] == "Exact review"
    assert analysis["projection"] == "curated_navigation_only"
    assert _load_run_analysis(
        run, {"source": {"commit": "different"}}, candidates=[analysis_path],
    ) is None


def test_viewer_prefers_same_session_robotwin_control_frames_over_snapshots(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs/continuous-robotwin"
    steps = run / "policy-state/steps"
    steps.mkdir(parents=True)
    action = {
        "action": "move_pose", "action_id": "action-1",
        "arguments": {"targets": {"left": {
            "position_m": [-0.1, -0.2, 0.8],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }}},
    }
    receipt = {
        "action_id": "action-1", "consumed": True,
        "execution_status": "not_reached", "target_arms": ["left"],
        "settle_steps": 4, "terminal_after_action": False,
        "arms": {
            "left": {
                "targeted": True,
                "planner": {"called": True, "status": "Success", "path_steps": 12},
                "pose_error": {"translation_m": 0.021, "orientation_rad": 0.1},
            },
            "right": {
                "targeted": False,
                "planner": {"called": False, "status": "not_called", "path_steps": 0},
                "inactive_arm_drift": {"translation_m": 0.000004, "orientation_rad": 0.0},
            },
        },
    }
    (steps / "step-0000.json").write_text(json.dumps({
        "step": 0, "cause": "reset", "observation": {"id": "obs-0"},
        "cameras": {}, "remaining_budget": {"robot_decisions": 1},
    }))
    (steps / "step-0001.json").write_text(json.dumps({
        "step": 1, "cause": "after_action", "observation": {"id": "obs-1"},
        "cameras": {}, "previous_public_action": action,
        "previous_execution_receipt": receipt,
    }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
        {"step": 1, "path": "policy-state/steps/step-0001.json"},
    ]}))
    (run / "plan.json").write_text(json.dumps({
        "experiment": "test", "task": "place_mouse_pad",
        "runtime": {"kind": "robotwin"}, "source": {"commit": "source"},
    }))
    frame_root = run / "native/operator-execution-frames"
    frame_root.mkdir(parents=True)
    frame = frame_root / "frame-000000.jpg"
    from PIL import Image
    Image.new("RGB", (1280, 960), "#33443f").save(frame)
    frame_sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    timeline_record = {
        "schema": "ora.robotwin-operator-execution-frame.v1",
        "frame_index": 0,
        # RoboTwin records paths relative to its native output root; the
        # Viewer resolves both this canonical form and historical run-relative
        # paths.
        "path": "operator-execution-frames/frame-000000.jpg",
        "sha256": frame_sha,
        "action_sequence": 0,
        "action_id": "action-1",
        "action": "move_pose",
        "phase": "control",
        "control_step_index": 5,
        "cameras": [
            "observer_camera", "head_camera", "front_camera",
            "left_camera", "right_camera",
        ],
        "capture_semantics": (
            "same_environment_session_after_actual_physics_control_step"
        ),
    }
    (frame_root / "timeline.jsonl").write_text(json.dumps(timeline_record) + "\n")

    trajectory = compile_trajectory(run)
    assert trajectory["steps"][0]["receipt"]["execution_status"] == "not_reached"
    assert trajectory["steps"][0]["receipt"]["reached"] is False
    source = _continuous_execution_source(run)
    assert source is not None
    assert source["frame_count"] == 1
    action_steps = {"action-1": trajectory["steps"][0]}
    rendered = _render_continuous_execution_frame(
        run, trajectory, source["frames"][0],
        source_root=run,
        source_kind="same_session_robotwin_control_frames",
        action_steps=action_steps, action_count=1,
    )
    assert rendered.size == (1920, 960)
    video, manifest = compile_trajectory_video(run)
    assert video.is_file()
    assert manifest["schema"] == "ora.viewer-trajectory-video.v2"
    assert manifest["continuous_execution"] is True
    assert manifest["frame_count"] == 1
    assert manifest["camera_panels"] == timeline_record["cameras"]


def test_viewer_controller_replay_source_stays_in_derived_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "runs/replay-run"
    (run / "native/controller-replay").mkdir(parents=True)
    (run / "native/controller-replay/manifest.json").write_text("{}")
    derived = tmp_path / "viewer-derived/replay-run/replay-test"
    (derived / "frames").mkdir(parents=True)
    from PIL import Image
    frame = derived / "frames/frame-000000.jpg"
    Image.new("RGB", (1280, 720), "#33443f").save(frame)
    timeline = derived / "timeline.jsonl"
    timeline.write_text(json.dumps({
        "schema": "ora.viewer-controller-replay-frame.v2",
        "frame_index": 0,
        "path": "frames/frame-000000.jpg",
        "action_sequence": 0,
        "action_id": "action-0",
        "action": "move_pose",
        "phase": "control",
        "control_step_index": 0,
        "sampling": {
            "mode": "action_keyframes_v1",
            "max_frames_per_action": 4,
            "hold_steps_per_sample": 8,
        },
        "cameras": ["agentview", "wrist"],
        "recorded_execution": {
            "status": "executed", "settle_steps": 0, "fault": None,
        },
        "capture_semantics": "derived_controller_replay_action_keyframes",
    }) + "\n")
    result = {
        "schema": "ora.viewer-controller-replay.v2",
        "camera_panels": ["agentview", "wrist"],
        "sampling": {
            "mode": "action_keyframes_v1",
            "max_frames_per_action": 4,
            "hold_steps_per_sample": 8,
        },
        "timeline": {"path": "timeline.jsonl"},
        "capture_semantics": (
            "derived_same_source_seed_controller_replay_action_keyframes"
        ),
    }
    monkeypatch.setattr(
        "viewer.trajectory.ensure_controller_replay",
        lambda _root, *, replay_mode="review": (derived, result),
    )
    source = _controller_replay_source(run)
    assert source is not None
    assert source["root"] == derived
    assert source["kind"] == "viewer_derived_controller_replay"
    assert source["sampling"] == {
        "mode": "action_keyframes_v1",
        "max_frames_per_action": 4,
        "hold_steps_per_sample": 8,
    }
    assert source["frame_count"] == 1
    assert source["frames"][0]["recorded_execution"]["status"] == "executed"
    drifted_record = dict(source["frames"][0])
    drifted_record["sampling"] = {
        "mode": "action_keyframes_v1",
        "max_frames_per_action": 3,
        "hold_steps_per_sample": 8,
    }
    timeline.write_text(json.dumps(drifted_record) + "\n")
    with pytest.raises(RuntimeError, match="sampling metadata drifted"):
        _controller_replay_source(run)


def test_viewer_replay_builders_recheck_cache_after_exclusive_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "runs/concurrent-replay"
    run.mkdir(parents=True)
    controller = tmp_path / "viewer-derived/concurrent-replay/controller"

    class CompletingBuilder:
        def __init__(self, output: Path, manifest: dict[str, object]) -> None:
            self.output = output
            self.manifest = manifest

        def __enter__(self) -> None:
            self.output.mkdir(parents=True)
            (self.output / "manifest.json").write_text(
                json.dumps(self.manifest) + "\n", encoding="utf-8",
            )

        def __exit__(self, *_args: object) -> None:
            return None

    compile_calls: list[str] = []
    monkeypatch.setattr(
        "viewer.replay.replay_identity",
        lambda _root, *, replay_mode: ("controller-identity", controller),
    )
    monkeypatch.setattr(
        "viewer.replay._exclusive_replay_lock",
        lambda _output: CompletingBuilder(controller, {
            "schema": "ora.viewer-controller-replay.v2",
            "identity": "controller-identity",
        }),
    )
    monkeypatch.setattr(
        "viewer.replay._compile_controller_replay",
        lambda *_args: compile_calls.append("controller"),
    )
    assert ensure_controller_replay(run)[0] == controller
    assert compile_calls == []


def test_controller_replay_video_removes_transient_frames_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "runs/replay-cleanup"
    steps = run / "policy-state/steps"
    replay_manifest = run / "native/controller-replay/manifest.json"
    steps.mkdir(parents=True)
    replay_manifest.parent.mkdir(parents=True)
    replay_manifest.write_text("{}")
    action = {
        "action": "move_pose",
        "action_id": "action-0",
        "arguments": {"targets": {"left": {
            "position_m": [-0.1, -0.2, 0.8],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }}},
    }
    receipt = {
        "action_id": "action-0",
        "consumed": True,
        "execution_status": "reached",
        "target_arms": ["left"],
        "arms": {},
    }
    (steps / "step-0000.json").write_text(json.dumps({
        "step": 0,
        "cause": "reset",
        "observation": {"id": "obs-0"},
        "cameras": {},
    }))
    (steps / "step-0001.json").write_text(json.dumps({
        "step": 1,
        "cause": "after_action",
        "observation": {"id": "obs-1"},
        "cameras": {},
        "previous_public_action": action,
        "previous_execution_receipt": receipt,
    }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
        {"step": 1, "path": "policy-state/steps/step-0001.json"},
    ]}))
    (run / "plan.json").write_text(json.dumps({
        "experiment": "test",
        "task": "place_mouse_pad",
        "runtime": {"kind": "robotwin"},
        "source": {"commit": "source"},
    }))
    derived = tmp_path / "viewer-derived/replay-cleanup/replay-test"
    frames = derived / "frames"
    frames.mkdir(parents=True)
    from PIL import Image
    frame = frames / "frame-000000.jpg"
    Image.new("RGB", (1280, 960), "#33443f").save(frame)
    sampling = {
        "mode": "action_keyframes_v1",
        "max_frames_per_action": 4,
        "hold_steps_per_sample": 8,
    }
    timeline = derived / "timeline.jsonl"
    timeline.write_text(json.dumps({
        "schema": "ora.viewer-controller-replay-frame.v2",
        "frame_index": 0,
        "path": "frames/frame-000000.jpg",
        "action_sequence": 0,
        "action_id": "action-0",
        "action": "move_pose",
        "phase": "control_final",
        "control_step_index": 9,
        "sampling": sampling,
        "cameras": [
            "observer_camera", "head_camera", "front_camera",
            "left_camera", "right_camera",
        ],
        "recorded_execution": {
            "status": "reached", "settle_steps": 0, "fault": None,
        },
        "capture_semantics": "derived_controller_replay_action_keyframes",
    }) + "\n")
    result = {
        "schema": "ora.viewer-controller-replay.v2",
        "camera_panels": [
            "observer_camera", "head_camera", "front_camera",
            "left_camera", "right_camera",
        ],
        "sampling": sampling,
        "timeline": {"path": "timeline.jsonl"},
        "capture_semantics": (
            "derived_same_source_seed_controller_replay_action_keyframes"
        ),
    }
    calls = 0

    def ensure(_root: Path, *, replay_mode: str = "review") -> tuple[Path, dict]:
        nonlocal calls
        calls += 1
        return derived, result

    monkeypatch.setattr("viewer.trajectory.ensure_controller_replay", ensure)
    video, video_manifest = compile_trajectory_video(run, replay_mode="forensic")
    assert video.is_file()
    assert calls == 1
    assert not derived.exists()
    assert video.with_suffix(".timeline.jsonl").is_file()
    assert video_manifest["sampling"] == sampling
    assert video_manifest["temporary_replay_frames_retained"] is False
    cached_video, cached_manifest = compile_trajectory_video(run, replay_mode="forensic")
    assert cached_video == video
    assert cached_manifest == video_manifest
    assert calls == 1


@pytest.mark.parametrize(
    "controller",
    [
        "robotwin_dense_joint_targets",
        "robotwin_local_cartesian_dls",
        "robotwin_global_then_local_settle",
    ],
)
def test_robotwin_strategy_labels_share_dense_target_viewer_replay(
    controller: str,
) -> None:
    assert _robotwin_replay_route(controller) == "dense_targets"


def test_robotwin_legacy_ee_label_keeps_separate_viewer_replay_route() -> None:
    assert _robotwin_replay_route("robotwin_ee_action") == "ee_action"


def test_robotwin_dense_replay_bypasses_unused_planner_initialization() -> None:
    dense = {"entries": [
        {"replayable": True, "controller": "robotwin_dense_joint_targets"},
        {"replayable": False, "controller": "robotwin_ee_action"},
    ]}
    legacy = {"entries": [
        {"replayable": True, "controller": "robotwin_ee_action"},
    ]}
    assert _robotwin_replay_needs_planner(dense) is False
    assert _robotwin_replay_needs_planner(legacy) is True


def test_robotwin_viewer_replay_action_keyframes_are_bounded_and_keep_endpoints() -> None:
    assert _uniform_capture_steps(8, 1) == frozenset({7})
    assert _uniform_capture_steps(8, 4) == frozenset({0, 2, 5, 7})
    assert _capture_steps(300, {
        "mode": "action_keyframes_v1", "max_frames_per_action": 4,
        "hold_steps_per_sample": 8,
    }) == frozenset({0, 100, 199, 299})
    assert _capture_steps(8, {
        "mode": "physics_stride_v1", "capture_stride": 3,
    }) == frozenset({0, 3, 6})
    assert _needs_final_capture(8, 6) is True
    assert _needs_final_capture(7, 6) is False
    assert _needs_final_capture(0, None) is False


def test_robotwin_viewer_replay_recovers_legacy_dense_step_count() -> None:
    import numpy as np

    entry = {"metadata": {"control_sequence": {
        "left_arm": {
            "position": {"array": "left_position"},
            "velocity": {"array": "left_velocity"},
        },
        "left_gripper": {"num_step": 300},
    }}}
    arrays = {
        "left_position": np.zeros((37, 6), dtype=np.float32),
        "left_velocity": np.zeros((37, 6), dtype=np.float32),
    }
    assert _recorded_control_step_count(entry, arrays) == 300
    entry["metadata"]["controller_tracking"] = {"physics_step_attempts": 41}
    assert _recorded_control_step_count(entry, arrays) == 41


def test_robotwin_viewer_review_applies_only_selected_recorded_target() -> None:
    import numpy as np

    calls: list[tuple] = []

    class Robot:
        def set_arm_joints(self, position, velocity, arm):
            calls.append(("arm", arm, position.tolist(), velocity.tolist()))

        def set_gripper(self, result, arm, per_step):
            calls.append(("gripper", arm, float(result), float(per_step)))

    environment = type("Environment", (), {"robot": Robot()})()
    sequence = {
        "left_arm": {
            "position": np.asarray([[0.0], [0.5], [1.0]]),
            "velocity": np.asarray([[0.0], [0.1], [0.0]]),
        },
        "left_gripper": {
            "result": np.asarray([1.0, 0.5, 0.0]),
            "num_step": 3,
            "per_step": 0.1,
        },
        "right_arm": None,
        "right_gripper": None,
    }
    _apply_robotwin_control_target(environment, sequence, 2)
    assert calls == [
        ("arm", "left", [1.0], [0.0]),
        ("gripper", "left", 0.0, 0.1),
    ]


def test_viewer_replay_frame_writer_binds_sampling_profile(
    tmp_path: Path,
) -> None:
    writer = _FrameWriter(
        tmp_path,
        cameras=[],
        sampling={
            "mode": "action_keyframes_v1",
            "max_frames_per_action": 4,
            "hold_steps_per_sample": 8,
        },
    )
    assert writer.sampling == {
        "mode": "action_keyframes_v1",
        "max_frames_per_action": 4,
        "hold_steps_per_sample": 8,
    }


@pytest.mark.parametrize(
    "controller", ["robotwin_unknown_controller", {"malformed": True}, None],
)
def test_robotwin_viewer_replay_rejects_unknown_controller_label(
    controller: object,
) -> None:
    with pytest.raises(ReplayWorkerError, match="unsupported RoboTwin"):
        _robotwin_replay_route(controller)


def test_libero_viewer_replay_reuses_sealed_controller_qualification(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "native/controller-replay/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    binding = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": "qualified-contract",
        "binding": {"robosuite_module": "/qualified/robosuite/__init__.py"},
    }
    (tmp_path / "native/reset.json").write_text(json.dumps({
        "schema": "ora.libero-pro-reset.v1",
        "qualified_controller_binding": binding,
    }))

    assert _libero_qualified_controller_binding(manifest) == binding

    (tmp_path / "native/reset.json").write_text(json.dumps({
        "schema": "ora.libero-pro-reset.v1",
        "qualified_controller_binding": None,
    }))
    with pytest.raises(ReplayWorkerError, match="qualified LIBERO controller"):
        _libero_qualified_controller_binding(manifest)


def test_viewer_indexes_exact_raw_events_and_synchronizes_observation_cycles(
    tmp_path: Path,
) -> None:
    run = tmp_path / "raw-trajectory-run"
    steps = run / "policy-state/steps"
    steps.mkdir(parents=True)
    for step, observation_id in enumerate(("obs-0", "obs-1")):
        image = run / f"policy-observations/{observation_id}/agentview/rgb.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(f"image-{step}".encode())
        (steps / f"step-{step:04d}.json").write_text(json.dumps({
            "step": step,
            "cause": "reset" if step == 0 else "after_action",
            "task_instruction": "Move the object into the basket.",
            "main_camera": "agentview",
            "observation": {"id": observation_id},
            "cameras": {"agentview": {"files": {"rgb_lossless_png": {
                "path": f"policy-observations/{observation_id}/agentview/rgb.png",
            }}}},
        }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
        {"step": 1, "path": "policy-state/steps/step-0001.json"},
    ]}))
    (run / "plan.json").write_text(json.dumps({
        "experiment": "raw-test", "config": "native-v2", "task": "pick-place",
    }))
    (run / "result.json").write_text(json.dumps({
        "state": "completed", "termination": "budget_exhausted",
        "official_success": False,
        "rounds": [{
            "round_index": 0, "official_success": False,
            "policy_state_steps": {"start": 0, "end_exclusive": 2},
        }],
    }))
    runtime_records = [
        {"type": "observation", "time": 1.0, "call": 0,
         "observation_id": "obs-0", "policy_state": {"step": 0}},
        {"type": "action_submitted", "time": 4.0,
         "action": {"action": "target_state", "action_id": "action-1"}},
        {"type": "motion_receipt", "time": 5.0,
         "record": {"action_id": "action-1", "pose_reached": True}},
        {"type": "observation", "time": 6.0, "call": 1,
         "observation_id": "obs-1", "policy_state": {"step": 1}},
    ]
    runtime_text = "".join(json.dumps(record) + "\n" for record in runtime_records)
    (run / "events.jsonl").write_text(runtime_text)
    artifact = run / "openeta/views/marked.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"marked-image")
    agent_records = [
        {"type": "provider_response", "unix_s": 2.0, "observation_step": 0,
         "model_call_id": "model-1", "payload": {"message": {
             "content": "Inspect the red object before moving.",
             "tool_calls": [{"function": {"name": "mark_point"}}],
         }}},
        {"type": "tool_result", "unix_s": 3.0, "observation_step": 0,
         "tool_call_id": "tool-1", "tool": "mark_point",
         "result": {"artifact": {"path": "openeta/views/marked.png"}}},
    ]
    agent_lines = [json.dumps(record) for record in agent_records]
    (run / "agent-events.jsonl").write_bytes(
        (agent_lines[0] + "\r\n" + agent_lines[1] + "\r\n").encode()
    )
    transcript_line = json.dumps({
        "call": 0, "started": 2.5, "finished": 2.7, "stdout": "exact transcript",
    })
    (run / "agent-transcript.jsonl").write_text(transcript_line + "\n")

    trace = compile_raw_trajectory(run)
    assert trace["schema"] == "ora.viewer-raw-trajectory.v1"
    assert trace["summary"]["event_count"] == 7
    assert trace["summary"]["observation_cycles"] == 2
    assert [event["timestamp_unix_s"] for event in trace["events"]] == [
        1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
    ]
    assert len({event["key"] for event in trace["events"]}) == 7
    agent_index = next(event for event in trace["events"] if event["key"] == "agent:1")
    assert agent_index["observation_step"] == 0
    assert "Inspect the red object" in agent_index["search_text"]
    assert trace["cycles"][0]["images"][0]["path"].endswith("obs-0/agentview/rgb.png")
    assert trace["cycles"][0]["after_images"][0]["path"].endswith(
        "obs-1/agentview/rgb.png"
    )
    assert trace["events"][0]["round_index"] == 0
    assert trace["cycles"][0]["round_index"] == 0

    detail = raw_trajectory_event(run, "agent", 2)
    expected_line = (agent_lines[1] + "\r\n").encode()
    assert detail["schema"] == "ora.viewer-raw-trajectory-event.v1"
    assert detail["raw_json"] == agent_lines[1]
    assert detail["record"] == agent_records[1]
    assert detail["line_sha256"] == hashlib.sha256(expected_line).hexdigest()
    assert detail["artifacts"][0]["path"] == "openeta/views/marked.png"
    with pytest.raises(ValueError, match="identity"):
        raw_trajectory_event(run, "unknown", 1)
    with pytest.raises(ValueError, match="does not exist"):
        raw_trajectory_event(run, "runtime", 99)


def test_viewer_discovers_and_classifies_configured_historical_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "collected-outputs"
    formal = outputs / "libero-study/runs/formal-run"
    diagnostic = outputs / "runtime/runs/synthetic-run"
    for root in (formal, diagnostic):
        root.mkdir(parents=True)
    (formal / "plan.json").write_text(json.dumps({
        "experiment": "libero-study", "config": "policy-a",
        "milestone": "LIBERO", "task": "libero_object_swap-t02", "seed": 0,
        "evidence_tier": "behavioral", "runtime": {
            "kind": "libero_pro", "suite": "libero_object_swap", "task_id": 2,
        },
        "policy": {"kind": "codex", "model": "gpt-test"},
        "source": {"commit": "deadbeef"},
    }))
    (formal / "status.json").write_text(json.dumps({"state": "completed"}))
    (formal / "evaluation.json").write_text(json.dumps({
        "status": "valid", "official_success": False,
    }))
    (formal / "seal.json").write_text("{}")
    (diagnostic / "plan.json").write_text(json.dumps({
        "experiment": "runtime", "task": "synthetic_tool_boundary",
        "evidence_tier": "diagnostic", "runtime": {"kind": "synthetic"},
        "policy": {"kind": "scripted"},
    }))
    roots = list(discover_run_roots([outputs]))
    registrations = [register_run(root) for root in roots]
    assert {run.run_id for run in registrations} == {"formal-run", "synthetic-run"}
    assert next(run for run in registrations if run.run_id == "formal-run").category == "formal"
    formal_registration = next(
        run for run in registrations if run.run_id == "formal-run"
    )
    assert formal_registration.task_key == "libero_pro:libero_object_swap:t02"
    assert next(run for run in registrations if run.run_id == "synthetic-run").category == "diagnostic"
    assert [run.run_id for run in visible_runs(
        registrations, view="diagnostics",
    )] == ["synthetic-run"]
    assert [run.run_id for run in visible_runs(
        registrations, view="formal", experiment="libero-study",
    )] == ["formal-run"]
    assert visible_runs(
        registrations, view="formal", experiment="different-study",
    ) == []
    assert [run.run_id for run in visible_runs(registrations, view="formal")] == ["formal-run"]
    assert [run.run_id for run in visible_runs(
        registrations, view="formal", backend="libero_pro"
    )] == ["formal-run"]
    assert [run.run_id for run in visible_runs(
        registrations, view="formal",
        task_key="libero_pro:libero_object_swap:t02", seed="0",
    )] == ["formal-run"]
    assert [run.run_id for run in visible_runs(
        registrations, view="all", state="completed", outcome="failure",
    )] == ["formal-run"]
    assert visible_runs(
        registrations, view="all", state="completed", outcome="success",
    ) == []


def test_viewer_recursively_discovers_legacy_worktree_snapshots(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run = outputs / "worktree-snapshots/feature/experiment/runs/legacy-run"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({
        "experiment": "experiment", "task": "legacy", "seed": 4,
        "evidence_tier": "behavioral", "runtime": {"kind": "libero_pro"},
        "policy": {"kind": "native_api", "model": "model"},
    }))
    assert list(discover_run_roots([outputs])) == [run.resolve()]


def test_viewer_recovers_pre_manifest_images_and_agent_turns(tmp_path: Path) -> None:
    run = tmp_path / "historical-run"
    image = run / "policy-observations/obs-0/head_camera/rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"historical-image")
    operator = run / "native/operator-world-frames/frame-0000-reset.png"
    operator.parent.mkdir(parents=True)
    operator.write_bytes(b"operator-image")
    run.mkdir(exist_ok=True)
    (run / "plan.json").write_text(json.dumps({
        "experiment": "historical", "task": "adjust_bottle",
    }))
    (run / "events.jsonl").write_text(json.dumps({
        "type": "observation", "observation_id": "obs-0",
        "context": {"main_camera": "head_camera"},
    }) + "\n")
    agent_event = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "inspect before moving"},
    }
    (run / "agent-transcript.jsonl").write_text(json.dumps({
        "call": 0, "started": 1.0, "finished": 2.0,
        "stdout": json.dumps(agent_event),
    }) + "\n")
    trajectory = compile_trajectory(run)
    assert [item["label"] for item in trajectory["steps"][0]["images"]] == [
        "head_camera", "operator world",
    ]
    assert trajectory["steps"][0]["images"][1]["visibility"] == "control_only"
    assert trajectory["steps"][0]["model_calls"][0]["messages"] == [
        "inspect before moving",
    ]


def test_viewer_video_compiler_script_is_reusable() -> None:
    script = ROOT / "scripts/compile-viewer-videos"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    completed = subprocess.run(
        [str(script), "--help"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "--outputs-root OUTPUTS_ROOT" in completed.stdout
    assert "--video-scope {formal,libero,robotwin}" in completed.stdout
    assert "--replay-mode {forensic,review}" in completed.stdout
    assert "--compile-videos" in completed.stdout


def test_robotwin_gallery_migrates_complete_pinned_catalog() -> None:
    assert len(TASKS) == 50
    assert len({task.id for task in TASKS}) == 50
    assert MIGRATED_FROM_COMMIT == "a762defefdecff6286ba3079cb02a77cd5327141"
    assert ROBOTWIN_REVISION == "c3ddfa8b97d5519efa828b075999bd0006778e5e"

    client = viewer.app.test_client()
    response = client.get(
        "/gallery/robotwin/adjust_bottle?embodiment=franka-panda&viewpoint=head"
    )
    assert response.status_code == 200
    assert b"RoboTwin Tasks" in response.data
    assert b"franka-panda_head.mp4" in response.data
    assert b"Generator acceptance rates, not policy success rates" in response.data

    filtered = client.get("/gallery/robotwin?q=microwave&family=articulation")
    assert b"Open Microwave" in filtered.data
    assert b"Open Laptop" not in filtered.data
    assert client.get("/gallery/robotwin/not-a-task").status_code == 404

    payload = client.get("/api/robotwin/tasks").get_json()
    assert payload["schema"] == CATALOG_SCHEMA
    assert payload["robotwin_revision"] == ROBOTWIN_REVISION
    assert len(payload["tasks"]) == 50
    adjust_bottle = next(task for task in payload["tasks"] if task["id"] == "adjust_bottle")
    assert adjust_bottle["average_steps"] == 147
    assert adjust_bottle["generation_success_percent"]["aloha-agilex"] == 93


def test_libero_pro_gallery_preserves_task_order_and_registration_boundary() -> None:
    assert len(LIBERO_PRO_TASKS) == 130
    assert LIBERO_MIGRATED_FROM_COMMIT == "a762defefdecff6286ba3079cb02a77cd5327141"
    assert LIBERO_PRO_CODE_REVISION == "2b910b5b5f53016bef9907632f6f840f1ce2229c"
    assert LIBERO_PRO_DATA_REVISION == "c86fc3b8293185a6f373677018ff3e37f8391602"
    for family, count in {
        "spatial": 10,
        "object": 10,
        "goal": 10,
        "libero_90": 90,
        "libero_10": 10,
    }.items():
        assert [
            task.task_id for task in LIBERO_PRO_TASKS if task.family == family
        ] == list(range(count))
    assert sum(task.pro_anchor for task in LIBERO_PRO_TASKS) == 40
    assert all(task.base_runtime_supported for task in LIBERO_PRO_TASKS)

    object_zero = next(
        task for task in LIBERO_PRO_TASKS
        if task.family == "object" and task.task_id == 0
    )
    assert object_zero.name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    assert object_zero.variant("position").suite == "libero_object_swap"
    assert object_zero.variant("position").instruction == (
        "Pick the alphabet soup and place it in the basket"
    )
    assert object_zero.variant("language").instruction == (
        "grab alphabet soup and put it into basket"
    )
    assert object_zero.variant("task").instruction == (
        "Pick the cream cheese and place it in the basket"
    )
    libero_ten_zero = next(
        task for task in LIBERO_PRO_TASKS
        if task.family == "libero_10" and task.task_id == 0
    )
    assert libero_ten_zero.name.startswith("LIVING_ROOM_SCENE2_")
    assert libero_ten_zero.key == "libero_10:task-00"
    assert libero_ten_zero.variant("position").suite == "libero_10_swap"

    client = viewer.app.test_client()
    response = client.get("/gallery/libero-pro/object/0?perturbation=task")
    assert response.status_code == 200
    assert b"LIBERO tasks" in response.data
    assert b"libero_object_swap" in response.data
    assert b"Pick the cream cheese and place it in the basket" in response.data
    assert b"Runtime \xc2\xb7 BDDL-driven" in response.data
    assert b"seed mod 50" in response.data
    assert b"Published results" not in response.data

    libero_90 = client.get("/gallery/libero/libero_90/89")
    assert libero_90.status_code == 200
    assert b"Original LIBERO runtime" in libero_90.data
    assert b"no Pro variants" in libero_90.data
    assert b"LIBERO-90" in libero_90.data
    assert b"Runtime \xc2\xb7 BDDL-driven" not in libero_90.data
    assert client.get("/gallery/libero").status_code == 200

    filtered = client.get("/gallery/libero-pro?q=microwave&family=libero_10")
    assert filtered.status_code == 200
    assert b"yellow and white mug in the microwave" in filtered.data
    assert b"put_both_moka_pots_on_the_stove" not in filtered.data
    assert client.get("/gallery/libero-pro/not-a-family/0").status_code == 404
    assert client.get("/gallery/libero-pro/object/10").status_code == 404

    payload = client.get("/api/libero-pro/tasks").get_json()
    assert payload["schema"] == LIBERO_PRO_CATALOG_SCHEMA
    assert payload["counts"] == {
        "base_tasks": 130,
        "controlled_transfer_tasks": 30,
        "libero_100_tasks": 100,
        "libero_90_tasks": 90,
        "libero_long_tasks": 10,
        "libero_pro_anchor_tasks": 40,
        "core_configurations": 160,
        "core_suites": 16,
        "init_states_per_task": 50,
        "robustness_cases": 7,
        "robustness_configurations": 280,
    }
    assert len(payload["tasks"]) == 130
    assert client.get("/api/libero/tasks").get_json() == payload
    assert len(LIBERO_ROBUSTNESS_CASES) == 7
    assert sum(
        len(task["robustness_assets"]) for task in payload["tasks"]
    ) == 280
    assert payload["evidence_boundary"] == {
        "contains_demonstrations": False,
        "contains_policy_rollouts": False,
        "kind": "read_only_catalog_with_runtime_identity",
        "robustness_extension_runtime_registered": True,
        "runtime_registered_base_suites": 5,
        "runtime_registered_core_suites": 16,
        "runtime_registered_robustness_suites": 28,
    }
    assert all(
        asset["runtime_registered"]
        for task in payload["tasks"] for asset in task["robustness_assets"]
    )

    assert client.get("/api/libero-pro/results").status_code == 404
