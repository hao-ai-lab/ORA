"""Native harness sft contract regressions."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from ora.orchestrator.evaluation import evaluate, seal
from ora.orchestrator.coordinator import run_episode
from ora.agent.providers.native import FakeProvider, translate_anthropic_request
from ora.agent.evidence.trajectory import ArtifactStore, TrajectoryError, TrajectoryRecorder, canonical_bytes, export_sft, verify_agent_trace
from test_native_harness import (
    assistant_tool,
    native_plan,
    usage,
)


def test_native_synthetic_trace_evaluate_seal_and_sft_export(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    plan = native_plan()
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    provider = FakeProvider([
        {
            "message": assistant_tool("read-state", "observation.dump_state", {}),
            "usage": usage(300, 20),
        },
        {
            "message": assistant_tool("submit-action", "robot.submit_action", {
                "action": "move_pose", "arm": "left", "position_m": [0, 0, 1.02],
                "orientation_xyzw": [0, 0, 0, 1],
            }, reasoning="provider-visible reasoning"),
            "usage": usage(500, 30),
        },
    ])
    result = run_episode(
        run_root=run, plan=plan, run_id="native-m0", provider_override=provider
    )
    assert result["state"] == "completed"
    assert result["termination"] == "trusted_runtime_boundary"
    assert result["decisions"] == 1
    assert result["model_calls"] == 2
    trace = verify_agent_trace(run)
    assert trace["status"] == "valid"
    context_manifest = json.loads((run / "context-manifest.json").read_text())
    assert context_manifest["native_context"] == {
        **context_manifest["native_context"],
        "canonical_message_ir": "openai_chatml",
        "provider_wire_protocol": plan["policy"]["provider"],
        "automatic_observation_image": True,
    }
    assert context_manifest["planner"]["supports_images"] is True
    assert "protocol" not in context_manifest["native_context"]
    system_layer = context_manifest["context_order"][0]
    expected_system = (
        (run / "sandbox/workspace/AGENTS.md").read_text().strip()
        + "\n\n"
        + (run / "sandbox/workspace/contract/episode-prefix.md").read_text().strip()
    )
    assert system_layer["exact_text"] == expected_system
    assert system_layer["exact_sha256"] == hashlib.sha256(
        expected_system.encode()
    ).hexdigest()
    tool_layer = next(
        item for item in context_manifest["context_order"]
        if item["kind"] == "openai_function_tools"
    )
    manual_tool = next(
        item for item in tool_layer["definitions"]
        if item["name"] == "robot.read_manual"
    )
    assert manual_tool["inputSchema"]["properties"]["section"]["enum"] == [
        "Embodiment", "Cameras", "Saved state", "Frames", "Actions", "Safety",
    ]
    event_types = [
        json.loads(line)["type"]
        for line in (run / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for required in (
        "session_started", "context_projected", "provider_request", "provider_response",
        "tool_call_proposed", "tool_result", "action_proposed", "environment_receipt",
        "fresh_observation_published", "official_verdict", "run_terminated", "session_closed",
    ):
        assert required in event_types
    evaluation = evaluate(run)
    assert evaluation["status"] == "valid"
    assert evaluation["agent_trace_final_event_sha256"] == trace["final_event_sha256"]
    assert seal(run)["evaluation_status"] == "valid"
    destination = export_sft(run, tmp_path / "sft")
    records = [
        json.loads(line)
        for line in (destination / "conversations.openai-tools.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[-1]["messages"][-1]["tool_calls"][0]["id"] == "submit-action"
    assert "reasoning_content" not in records[-1]["messages"][-1]
    assert "data:image/png;base64," in json.dumps(records[0])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["official_success"] is None
    assert manifest["termination"] == "trusted_runtime_boundary"


def test_sft_export_uses_canonical_request_instead_of_anthropic_wire(
    tmp_path: Path,
) -> None:
    run = tmp_path / "anthropic-export-run"
    store = ArtifactStore(run)
    recorder = TrajectoryRecorder(run, "run", "session")
    canonical = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "vision.inspect", "description": "inspect",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    }
    wire = translate_anthropic_request({
        **canonical, "max_tokens": 64, "stream": True,
    })
    canonical_ref = store.put_json(canonical, kind="native_canonical_provider_request")
    wire_ref = store.put_json(wire, kind="native_provider_wire_request")
    recorder.append(
        "provider_request", model_call_id="call", artifact_ref=wire_ref,
        payload={"model": "glm-5.2", "canonical_request_ref": canonical_ref},
    )
    recorder.append(
        "provider_response", model_call_id="call",
        payload={"message": {
            "role": "assistant", "content": "done",
            "reasoning_content": "trace only",
            "_provider_continuation_state": {"opaque": "provider only"},
        }},
    )
    (run / "plan.json").write_text(json.dumps({
        "source": {"commit": "test"},
        "policy": {"context_profile": "native_minimal_v1"},
    }))
    (run / "result.json").write_text(json.dumps({
        "run_id": "run", "official_success": None, "termination": "test",
    }))
    (run / "evaluation.json").write_text(json.dumps({
        "status": "valid",
        "learning_eligibility": {"eligible": True, "reasons": []},
    }))
    destination = export_sft(run, tmp_path / "anthropic-sft")
    record = json.loads(
        (destination / "conversations.openai-tools.jsonl").read_text().strip()
    )
    assert record["messages"][0] == {"role": "user", "content": "hello"}
    assert record["messages"][-1] == {"role": "assistant", "content": "done"}
    assert "provider only" not in json.dumps(record)
    assert record["tools"][0]["type"] == "function"
    assert "input_schema" not in json.dumps(record["tools"])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["exact_request_sha256"] == [canonical_ref["sha256"]]


def test_sft_export_attributes_decisions(tmp_path: Path) -> None:
    """EXP08A training readiness: each exported decision names its kind."""

    run = tmp_path / "attribution-export-run"
    store = ArtifactStore(run)
    recorder = TrajectoryRecorder(run, "run", "session")
    exchanges = [
        ("call-move", {
            "role": "assistant", "content": "move the arm",
            "tool_calls": [{
                "id": "move-0", "type": "function",
                "function": {"name": "move_pose", "arguments": "{}"},
            }],
        }),
        ("call-final", {"role": "assistant", "content": "done"}),
    ]
    for call_id, assistant in exchanges:
        canonical = {
            "model": "k3",
            "messages": [{"role": "user", "content": "observe"}],
            "tools": [],
        }
        canonical_ref = store.put_json(
            canonical, kind="native_canonical_provider_request",
        )
        recorder.append(
            "provider_request", model_call_id=call_id,
            artifact_ref=canonical_ref,
            payload={"model": "k3", "canonical_request_ref": canonical_ref},
        )
        recorder.append(
            "provider_response", model_call_id=call_id,
            payload={"message": assistant},
        )
    (run / "plan.json").write_text(json.dumps({
        "source": {"commit": "test"},
        "policy": {"context_profile": "native_minimal_v1"},
        "policy_composition": {"tool_surface": {
            "profile_id": "test_surface",
            "mutation_tools": ["move_pose", "set_gripper"],
            "lifecycle_tools": ["finish_episode"],
        }},
    }))
    (run / "result.json").write_text(json.dumps({
        "run_id": "run", "official_success": None, "termination": "test",
    }))
    (run / "evaluation.json").write_text(json.dumps({
        "status": "valid",
        "learning_eligibility": {"eligible": True, "reasons": []},
    }))
    destination = export_sft(run, tmp_path / "attribution-sft")
    attribution = [
        json.loads(line)
        for line in (destination / "attribution.jsonl").read_text().splitlines()
    ]
    assert attribution == [
        {
            "decision_kind": "robot_action",
            "delegated_executor": None,
            "index": 0,
            "tool_call_names": ["move_pose"],
        },
        {
            "decision_kind": "read_only_or_reflection",
            "delegated_executor": None,
            "index": 1,
            "tool_call_names": [],
        },
    ]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["attribution_counts"] == {
        "read_only_or_reflection": 1, "robot_action": 1,
    }
    assert {item["path"] for item in manifest["files"]} == {
        "conversations.openai-tools.jsonl", "attribution.jsonl", "loss-mask.jsonl",
    }
    masks = [json.loads(line) for line in
             (destination / "loss-mask.jsonl").read_text().splitlines()]
    assert len(masks) == 2
    for index, mask in enumerate(masks):
        assert mask == dict(index=index, target_author="language_model",
                            target_message_index=1, message_loss_weights=[0, 1])
    assert manifest["loss_mask_contract"]["tokenizer_compilation_required"] is True


def test_sft_export_masks_history(tmp_path):
    run = tmp_path / "run"
    store = ArtifactStore(run)
    recorder = TrajectoryRecorder(run, "run", "session")
    canonical = {"messages": [
        {"role": "user", "content": "align"},
        {"role": "assistant", "content": "move to the alignment pose"},
        {"role": "tool", "content": "motion stopped", "tool_call_id": "move-0"},
    ], "tools": []}
    reference = store.put_json(canonical, kind="native_canonical_provider_request")
    recorder.append("provider_request", model_call_id="call", artifact_ref=reference,
                    payload={"canonical_request_ref": reference})
    recorder.append("provider_response", model_call_id="call",
                    payload={"message": {"role": "assistant", "content": "inspect the result"}})
    (run / "plan.json").write_text(json.dumps({"source": {}, "policy": {}}))
    (run / "result.json").write_text(json.dumps({"run_id": "run"}))
    (run / "evaluation.json").write_text(json.dumps({"status": "valid",
        "learning_eligibility": {"eligible": True, "reasons": []}}))
    destination = export_sft(run, tmp_path / "export")
    mask = json.loads((destination / "loss-mask.jsonl").read_text())
    assert mask["message_loss_weights"] == [0, 0, 0, 1]


def test_trace_validator_rejects_hash_and_artifact_drift(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    recorder = TrajectoryRecorder(tmp_path, "run", "session")
    ref = store.put_json({"safe": True}, kind="provider_request")
    recorder.append(
        "provider_request", model_call_id="call", artifact_ref=ref,
        payload={"model": "fake"},
    )
    recorder.append(
        "provider_response", model_call_id="call", payload={"message": {"role": "assistant", "content": "done"}},
    )
    assert verify_agent_trace(tmp_path)["status"] == "valid"
    artifact = tmp_path / ref["path"]
    artifact.write_text("tampered", encoding="utf-8")
    check = verify_agent_trace(tmp_path)
    assert check["status"] == "invalid"
    assert any("artifact_hash" in error for error in check["errors"])
    artifact.write_bytes(canonical_bytes({"safe": True}))
    trace_path = tmp_path / "agent-events.jsonl"
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events[0]["payload"]["model"] = "tampered-model"
    trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    assert any("hash" in error for error in verify_agent_trace(tmp_path)["errors"])

    pairing_root = tmp_path / "pairing"
    pairing = TrajectoryRecorder(pairing_root, "run", "session")
    pairing.append(
        "tool_call_proposed", observation_step=0, model_call_id="call",
        tool_call_id="tool", payload={"name": "vision.inspect", "batch_valid": True},
    )
    pairing.append(
        "tool_result", observation_step=1, model_call_id="call",
        tool_call_id="tool", payload={"ok": True},
    )
    pairing.append(
        "tool_call_proposed", observation_step=1, model_call_id="call-2",
        tool_call_id="tool", payload={"name": "vision.inspect", "batch_valid": True},
    )
    pairing.append(
        "tool_call_proposed", observation_step=1, model_call_id="call-2",
        tool_call_id="tool-missing", payload={"name": "geometry.query", "batch_valid": True},
    )
    pairing_check = verify_agent_trace(pairing_root)
    assert any("cross_observation_tool_result" in error for error in pairing_check["errors"])
    assert any("tool_call_id" in error for error in pairing_check["errors"])
    assert any("missing_tool_result" in error for error in pairing_check["errors"])

    privileged_root = tmp_path / "privileged"
    privileged_store = ArtifactStore(privileged_root)
    privileged = TrajectoryRecorder(privileged_root, "run", "session")
    request_ref = privileged_store.put_json({
        "messages": [{
            "role": "tool", "tool_call_id": "tool",
            "content": json.dumps({"object_gt_pose": [0, 0, 0]}),
        }],
        "tools": [],
    }, kind="native_provider_request")
    privileged.append(
        "provider_request", model_call_id="call", artifact_ref=request_ref,
        payload={"model": "fake"},
    )
    privileged.append(
        "provider_response", model_call_id="call",
        payload={"message": {"role": "assistant", "content": "done"}},
    )
    (privileged_root / "plan.json").write_text(json.dumps({"source": {}, "policy": {}}))
    (privileged_root / "result.json").write_text(json.dumps({"run_id": "run"}))
    with pytest.raises(TrajectoryError, match="privileged"):
        export_sft(privileged_root, tmp_path / "privileged-export")


def test_trace_validator_allows_only_surface_bound_public_grasp_contact_pose(
    tmp_path: Path,
) -> None:
    contact_pose = {
        "position_m": [-0.2, -0.06, 0.08],
        "orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
    }

    def build_trace(root: Path, value: object, *, capability: bool) -> dict[str, object]:
        root.mkdir(parents=True)
        surface = {
            "capabilities": ["public_grasp_contact_pose_v1"] if capability else [],
        }
        (root / "plan.json").write_text(json.dumps({
            "policy_composition": {"tool_surface": surface},
        }))
        store = ArtifactStore(root)
        recorder = TrajectoryRecorder(root, "run", "session")
        request_ref = store.put_json({
            "messages": [{
                "role": "tool",
                "tool_call_id": "tool",
                "content": json.dumps({"candidates": [{"contact_pose": value}]}),
            }],
            "tools": [],
        }, kind="native_provider_request")
        recorder.append(
            "provider_request",
            model_call_id="call",
            artifact_ref=request_ref,
            payload={"model": "fake"},
        )
        recorder.append(
            "provider_response",
            model_call_id="call",
            payload={"message": {"role": "assistant", "content": "done"}},
        )
        return verify_agent_trace(root)

    assert build_trace(tmp_path / "allowed", contact_pose, capability=True)["status"] == "valid"
    assert build_trace(tmp_path / "unbound", contact_pose, capability=False)["status"] == "invalid"
    assert build_trace(
        tmp_path / "malformed",
        {**contact_pose, "object_id": "hidden"},
        capability=True,
    )["status"] == "invalid"
    assert build_trace(
        tmp_path / "gt",
        {"position_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
         "object_gt_pose": [0, 0, 0]},
        capability=True,
    )["status"] == "invalid"
