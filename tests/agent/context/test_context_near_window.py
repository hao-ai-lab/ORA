"""Near-window protection, fixed Episode media tails and body recovery."""
import copy
import json

import pytest

from ora.agent.context.historical_observation import invariant_block
from ora.agent.contracts import PolicySessionError
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import FakeProvider
from tests.agent.context.test_lossless_context_runtime import MANAGEMENT, TOOLS, _image, _messages, _pipeline


def history(completed, images=4):
    messages = _messages(completed)
    for message in messages:
        if "_observation" in message:
            step = message["_observation"]["step"]
            message["content"][1:] = [_image(f"{step}-{i}") for i in range(images)]
    return messages


def pipeline_at(path, **settings):
    return _pipeline(path, management={**MANAGEMENT, "trigger_inline_images": 24, **settings})


def test_fixed_media_tail_externalizes_old_steps_below_pressure_threshold(tmp_path):
    pipeline = pipeline_at(tmp_path)
    assembler = NativeRequestAssembler("test", "provider_default", 2048, False, FakeProvider([]))
    below = pipeline.project(messages=history(5), tools=TOOLS, observation_step=5)
    assembler.prepare(below)
    assert below["manifest"]["active_inline_images"] == 12
    assert below["manifest"]["lifecycle_media_externalized_through_action_step"] == 2
    assert below["manifest"]["protected_inline_media_action_steps"] == [3, 4, 5]
    # The fixed tail advances on the next completed ActionStep independently
    # of the still-unreached 24-image pressure threshold.
    above = pipeline.project(messages=history(6), tools=TOOLS, observation_step=6)
    assembler.prepare(above)
    assert above["manifest"]["active_inline_images"] == 12
    assert above["manifest"]["lifecycle_media_externalized_through_action_step"] == 3
    assert above["manifest"]["request_relation"] == "projection_epoch"
    again = pipeline.project(messages=history(6), tools=TOOLS, observation_step=6)
    assembler.prepare(again)
    assert again["messages"] == above["messages"]
    assert again["manifest"]["projection_transitions"] == []


def test_fixed_media_tail_may_exceed_soft_image_pressure_threshold(tmp_path):
    projection = pipeline_at(tmp_path, trigger_inline_images=4).project(
        messages=history(5, 8), tools=TOOLS, observation_step=5,
    )
    assert projection["manifest"]["active_inline_images"] == 24
    assert projection["manifest"]["lifecycle_media_externalized_through_action_step"] == 2
    assert projection["manifest"]["protected_inline_media_action_steps"] == [3, 4, 5]


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin", "robodojo"])
def test_historical_observation_recovers_exact_public_body_and_preserves_causal_effect(tmp_path, backend):
    pipeline = pipeline_at(tmp_path, trigger_estimated_tokens=1)
    messages = history(5)
    expected = {}
    for message in messages:
        if "_observation" not in message:
            continue
        step = message["_observation"]["step"]
        body = {
            "current_actual": {
                "step": step, "actual_sequence": step + 1,
                "public_robot_facts": {"layout": backend, "measured": [0.5] * 9000},
                "camera_geometry": {"calibration": [1.0] * 9000},
            },
            "last_action_effect": {"status": "applied", "feedback": "pose held"},
            "comparison": {"available": True, "action_result_actual_sequence": step + 1},
            "terminal_lifecycle": False,
        }
        original = json.dumps(body)
        message["content"][0]["text"] = original
        message["_private_worker"] = {"task_source": "must not enter Artifact"}
        expected[step] = original
    projection = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    assert projection["manifest"]["lossless_invariants_before_sha256"] == projection["manifest"]["lossless_invariants_after_sha256"]
    bodies = [json.loads(m["content"][0]["text"]) for m in projection["messages"]
              if m["role"] == "user" and isinstance(m["content"], list)]
    for step, body in enumerate(bodies):
        if step < 2:
            _, recovered = pipeline.context_artifacts.resolve(body["recoverable_observation"]["ref"])
            assert recovered.decode() == expected[step]
            assert b"must not enter Artifact" not in recovered
            for key in ("last_action_effect", "comparison", "terminal_lifecycle"):
                assert body[key] == json.loads(expected[step])[key]
        else:
            assert body == json.loads(expected[step])
    assert projection["manifest"]["active_inline_images"] == 12


def test_observation_invariant_rejects_tampered_recovery_body(tmp_path):
    from ora.agent.context.historical_observation import materialize, externalize
    pipeline = pipeline_at(tmp_path)
    message = history(0)[-1]
    message["content"][0]["text"] = json.dumps({"current_actual": {"measured": [1] * 4000}})
    materialize(message, catalog=pipeline.context_artifacts, round_index=1, action_step=0)
    projected, changed, _ = externalize([message], eligible_indexes={0})
    assert changed == [0]
    assert invariant_block(projected[0]) == message["content"][0]
    tampered = copy.deepcopy(projected[0])
    tampered["content"][0]["text"] = tampered["content"][0]["text"].replace('"observation_externalized":true', '"observation_externalized":false')
    with pytest.raises(PolicySessionError, match="differs from recoverable"):
        invariant_block(tampered)


def test_resolved_l5_threshold_reports_ratio_dominance_without_tuning(tmp_path):
    pipeline = pipeline_at(tmp_path, episode_checkpoint_trigger_tokens=150000,
                           episode_checkpoint_trigger_ratio=0.6)
    pipeline.context_window_tokens = 1_048_576
    projected = pipeline.project(messages=history(0), tools=TOOLS, observation_step=0)
    manifest = projected["manifest"]
    assert manifest["resolved_episode_checkpoint_trigger_tokens"] == max(150000, int(pipeline.hard_input_tokens * 0.6))
    assert manifest["resolved_episode_checkpoint_trigger_tokens"] > 600000
    assert manifest["context_management"]["episode_checkpoint_trigger_tokens"] == 150000


def test_seal_coverage_accepts_target_retention_and_rejects_protected_eviction(tmp_path):
    from ora.orchestrator.context_coverage import coverage_violations
    manifest = pipeline_at(tmp_path).project(messages=history(6), tools=TOOLS,
                                            observation_step=6)["manifest"]
    assert coverage_violations(manifest, manifest["action_step_inventory"]) == []
    tampered = {**manifest, "lifecycle_media_externalized_through_action_step": 4}
    assert "lossless_context_lifecycle_coverage_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
    tampered = {**manifest, "active_inline_images": 4}
    assert "lossless_context_image_target_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
    tampered = {**manifest, "resolved_episode_checkpoint_trigger_tokens": 150000}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
