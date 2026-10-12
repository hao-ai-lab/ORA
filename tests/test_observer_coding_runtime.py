"""Observer-first coding Runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

from ora.orchestrator.context_audit import build_context_audit, verify_context_audit, write_context_audit
from ora.orchestrator.metrics import write_runtime_metrics
from ora.environment import SyntheticSession
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.tools.robot import TOOL_DEFINITIONS, RobotToolRuntime


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "tests/data/prompts/startup-prompt-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENTS = ROOT / "ora/policy/planners/codex/workspace-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
CAMERAS = ("observer_camera", "head_camera", "front_camera", "left_camera", "right_camera")


def _observer_workspace(tmp_path: Path):
    source = tmp_path / "policy-observations"
    session = SyntheticSession(
        tmp_path / "native", "synthetic_observer", 0,
        main_camera="observer_camera", camera_names=CAMERAS,
    )
    session.start()
    observation, _ = session.observe(source)
    state = PolicyStateStore(tmp_path, "exp00c-test", "plan-test")
    snapshot = state.publish(
        observation=observation,
        observation_root=source / observation["observation_id"], cause="reset",
        task_instruction="Analyze Observer evidence.", previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 2, "wall_time_s": 10},
    )
    workspace = tmp_path / "sandbox/workspace"
    control = tmp_path / "control"
    projector = ContextProjector(
        workspace, control, PROMPT, MANUAL, AGENTS, main_camera="observer_camera"
    )
    pointer = projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="Analyze Observer evidence.",
        remaining_budget={"robot_decisions": 1, "model_calls": 2, "wall_time_s": 10},
        policy_state={
            "step": 0, "path": state.steps[-1]["path"],
            "sha256": state.steps[-1]["sha256"],
            "state_sha256": snapshot["state_sha256"],
        },
    )
    tools = RobotToolRuntime(
        control, MANUAL, tmp_path / "events.jsonl", BOUNDS, state_root=state.root,
    )
    return session, observation, workspace, pointer, tools


def test_observer_is_the_only_automatic_image_and_auxiliary_rgbd_is_on_demand(tmp_path: Path) -> None:
    _session, observation, workspace, pointer, tools = _observer_workspace(tmp_path)
    assert observation["main_camera"] == "observer_camera"
    assert set(observation["cameras"]) == set(CAMERAS)
    assert pointer["main_image"] == "inputs/current/observer_camera/rgb-thumbnail.png"
    thumbnails = sorted(workspace.glob("inputs/current/*/rgb-thumbnail.png"))
    assert [path.parent.name for path in thumbnails] == ["observer_camera"]
    for camera in CAMERAS:
        assert (workspace / f"inputs/current/{camera}/rgb.png").is_file()
        assert (workspace / f"inputs/current/{camera}/depth-m.npy").is_file()
        assert (workspace / f"inputs/current/{camera}/calibration.json").is_file()
    depth = tools.invoke("vision.inspect", {"camera": "left_camera", "modality": "depth"})
    assert depth["depth_stats"]["valid_fraction"] == 1.0
    assert depth["source_artifacts"]["depth"]["sha256"]


def test_observer_projection_round_trip_and_eef_projection_are_hash_bound(tmp_path: Path) -> None:
    _session, _observation, _workspace, _pointer, tools = _observer_workspace(tmp_path)
    back = tools.invoke("geometry.query", {
        "camera": "observer_camera", "points": [[1, 1]], "window_px": 0,
    })
    assert back["step"] == 0
    assert back["source_state_sha256"]
    world = back["points"][0]["world_xyz_m"]
    projected = tools.invoke("geometry.query", {
        "camera": "observer_camera", "world_points_m": [world, [0.0, 0.0, 1.0]],
    })
    assert projected["calibration_sha256"]
    assert projected["world_points"][0]["pixel_uv"] == [1.0, 1.0]
    assert projected["world_points"][0]["in_frame"] is True
    assert projected["world_points"][1]["in_frame"] is True
    geometry_schema = next(
        item["inputSchema"] for item in TOOL_DEFINITIONS
        if item["name"] == "geometry.query"
    )
    assert {tuple(item["required"]) for item in geometry_schema["oneOf"]} == {
        ("points",), ("region",), ("world_points_m",),
    }


def test_context_audit_separates_provider_harness_images_coding_and_cache(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "native").mkdir(parents=True)
    (run / "context-bundles").mkdir()
    plan = {
        "plan_sha256": "plan", "runtime": {"main_camera": "observer_camera"},
        "policy_context": {},
    }
    result = {
        "run_id": "audit-run", "plan_sha256": "plan", "wall_duration_s": 2.0,
    }
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "result.json").write_text(json.dumps(result))
    (run / "events.jsonl").write_text("")
    message = "<robot_turn_update>delta</robot_turn_update>"
    bundle = {
        "call": 0, "message": {"sha256": "message-hash", "utf8_bytes": len(message)},
        "image_attachment": {
            "path": "context-bundles/image-call-0000.png", "bytes": 10,
            "sha256": "image-hash", "source": {"sha256": "source-hash"},
            "derivation": {"maximum_size_px": [512, 512]},
        },
    }
    (run / "context-bundles/call-0000.json").write_text(json.dumps(bundle))
    session = {
        "backend": "codex_app_server_v2", "session_id": "agent", "process_pid": 1,
        "process_invocations": 1, "process_restarts": 0, "thread_id": "thread",
        "rollovers": [], "compaction_events": [],
        "calls": [{
            "call": 0, "turn_id": "turn", "message_kind": "resume_user_message",
            "project_message_chars": len(message), "project_message_bytes": len(message),
            "latency_s": 1.0, "usage": {
                "input_tokens": 100, "cached_input_tokens": 80,
                "output_tokens": 10, "reasoning_output_tokens": 4, "total_tokens": 110,
            },
        }],
    }
    (run / "native/agent-session.json").write_text(json.dumps(session))
    (run / "native/coding-audit.jsonl").write_text(json.dumps({
        "turn_id": "turn", "cwd": "scratch", "commands": [],
        "scratch": {"created": [], "modified": [], "reserved_paths_changed": []},
        "immutable_inputs_unchanged": True, "outside_workspace_path_violations": [],
    }) + "\n")
    write_runtime_metrics(run)
    audit = write_context_audit(run)
    assert audit["turns"][0]["provider"]["cached_input_tokens"] == 80
    assert audit["turns"][0]["harness_text"]["token_status"].startswith("estimated")
    assert audit["camera_disclosure"]["automatic_policy_camera"] == "observer_camera"
    assert audit["coding"]["integrity_passed"] is True
    assert audit["verifiable_redundancy"]["provider_cached_ratio_classification"] == "cache_reuse_not_redundant_new_input"
    assert build_context_audit(run) == audit
    assert verify_context_audit(run)["status"] == "valid"
