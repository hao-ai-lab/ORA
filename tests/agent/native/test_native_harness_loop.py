"""Native harness loop contract regressions."""

from __future__ import annotations
import json
from pathlib import Path
import httpx
import pytest
from ora.agent.context.native import NativeContextPipeline
from ora.agent.contracts import PolicySessionError
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import FakeProvider, OpenAICompatibleProvider
from ora.agent.evidence.trajectory import ArtifactStore, verify_agent_trace
from tests.agent.native.test_native_harness import (
    AGENT_RULES,
    PROMPT,
    assistant_tool,
    publish_native_boundary,
    usage,
)


def test_context_pipeline_spills_large_results_and_refuses_an_exhausted_window(
    tmp_path: Path,
) -> None:
    messages = [{"role": "system", "content": "stable"}]
    for step in range(14):
        messages.extend([
            {
                "role": "user", "content": f"observation-{step}",
                "_observation": {"step": step, "observation_id": f"obs-{step}", "keyframe": True},
            },
            {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": f"tool-{step}", "type": "function",
                    "function": {"name": "vision.inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "tool", "tool_call_id": f"tool-{step}", "name": "vision.inspect",
                "content": ("x" * 17000 if step == 13 else f"result-{step}"),
            },
        ])
    pipeline = NativeContextPipeline(
        "native_minimal_v1", ArtifactStore(tmp_path), 200_000, 4096
    )
    projected = pipeline.project(messages=messages, tools=[], observation_step=13)
    manifest = projected["manifest"]
    assert manifest["passes"][1]["name"] == "l1_spill"
    assert len(manifest["passes"][1]["artifact_refs"]) == 1
    spill_ref = manifest["passes"][1]["artifact_refs"][0]
    assert (tmp_path / spill_ref["path"]).is_file()
    assert manifest["cache_epoch"] == 0
    rendered = json.dumps(projected["messages"])
    # The rewrite retention keeps every observation group; only the large
    # result is spilled.
    for step in range(14):
        assert f"observation-{step}" in rendered
        assert f"tool-{step}" in rendered
    # The spill stub records that the complete result is retained without
    # leaking the artifact store path or content hash into provider context.
    assert "complete_result_recorded" in rendered
    assert "artifact_ref" not in rendered
    assert len(manifest["stable_prefix_hash"]) == 64

    constrained = NativeContextPipeline(
        "native_minimal_v1", ArtifactStore(tmp_path / "small"), 4097, 1024
    )
    with pytest.raises(Exception, match="context_exhausted"):
        constrained.project(messages=messages[-3:], tools=[], observation_step=13)


def test_mutation_batch_is_rejected_with_paired_results_before_single_proposal(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    mixed = {
        "role": "assistant", "content": "", "tool_calls": [
            {
                "id": "read-1", "type": "function",
                "function": {"name": "robot.read_manual", "arguments": '{"section":"safety"}'},
            },
            assistant_tool("mutate-bad", "robot.submit_action", {
                "action": "set_gripper", "arm": "left", "gripper": 0.5,
            })["tool_calls"][0],
        ],
    }
    multiple_reads = {
        "role": "assistant", "content": "", "tool_calls": [
            assistant_tool("read-state", "observation.dump_state", {})["tool_calls"][0],
            assistant_tool("read-image", "vision.inspect", {
                "camera": "head_camera", "resolution": "thumbnail",
            })["tool_calls"][0],
        ],
    }
    invalid_arguments = {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "bad-json", "type": "function",
            "function": {"name": "robot.read_manual", "arguments": "{not-json"},
        }],
    }
    provider = FakeProvider([
        {"message": {"role": "assistant", "content": "I need another turn."}, "usage": usage(80)},
        {"message": multiple_reads, "usage": usage(90)},
        {"message": invalid_arguments, "usage": usage(95)},
        {"message": mixed, "usage": usage(100)},
        {"message": assistant_tool("mutate-good", "robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.5,
        }), "usage": usage(120)},
    ])
    planner = NativePlanner(
        tmp_path, "native-boundary", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )
    planner.record_environment_receipt({
        "path": "receipts/action-0000.json",
        "receipt": {
            "action": "move_free_space",
            "action_id": outcome["action"]["action_id"],
            "state_changed": True,
        },
        "sha256": "receipt-hash",
    })
    planner.record_fresh_observation({
        **pointer, "policy_state": {**pointer["policy_state"], "step": 1}
    })
    planner.close()
    assert outcome["submitted"] is True
    assert len([
        message for message in provider.requests[2]["messages"]
        if message.get("role") == "tool"
    ]) == 2
    read_assistant_index = next(
        index for index, message in enumerate(provider.requests[2]["messages"])
        if message.get("role") == "assistant"
        and len(message.get("tool_calls", [])) == 2
    )
    assert [
        message["role"]
        for message in provider.requests[2]["messages"][
            read_assistant_index + 1:read_assistant_index + 5
        ]
    ] == ["tool", "tool", "user", "user"]
    assert any(
        "valid JSON" in message["content"]
        for message in provider.requests[3]["messages"]
        if message.get("role") == "tool"
    )
    mixed_results = [
        message for message in provider.requests[4]["messages"]
        if message.get("role") == "tool"
    ][-2:]
    assert len(mixed_results) == 2
    # Only the mutation of a mixed batch is refused; the read-only call ran.
    read, mutation = mixed_results
    assert "mutation_batch_rejected" not in read["content"] and "safety" in read["content"]
    refusal = json.loads(mutation["content"])["error"]
    assert refusal["code"] == "mutation_batch_rejected"
    assert refusal["constraints"] == {"other_calls_executed": 1}
    assert verify_agent_trace(tmp_path)["status"] == "valid"
    trace_events = [
        json.loads(line)
        for line in (tmp_path / "agent-events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    receipt_event = next(
        event for event in trace_events if event["type"] == "environment_receipt"
    )
    assert receipt_event["payload"]["action_id"] == outcome["action"]["action_id"]

    interrupt_root = tmp_path / "interrupt"
    interrupt_root.mkdir()
    pointer, workspace, tools = publish_native_boundary(interrupt_root)
    interrupted_provider = FakeProvider([{"message": {"role": "assistant", "content": "unused"}}])
    interrupted = NativePlanner(
        interrupt_root, "interrupted", tools, interrupted_provider,
        AGENT_RULES, PROMPT, "fake", "native_minimal_v1", 32768,
        max_output_tokens=2048,
    )
    interrupted.interrupt("test_interrupt")
    interrupted.close()
    assert interrupted_provider.closed is True
    assert verify_agent_trace(interrupt_root)["status"] == "valid"

    failure_root = tmp_path / "trace-failure"
    failure_root.mkdir()
    pointer, workspace, tools = publish_native_boundary(failure_root)
    failure_provider = FakeProvider([{
        "message": assistant_tool("must-not-submit", "robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.2,
        }),
        "usage": usage(100),
    }])
    failed = NativePlanner(
        failure_root, "trace-failure", tools, failure_provider,
        AGENT_RULES, PROMPT, "fake", "native_minimal_v1", 32768,
        max_output_tokens=2048,
    )

    class FailingRecorder:
        def __init__(self, delegate): self.delegate = delegate
        def append(self, event_type, **fields):
            if event_type == "tool_call_proposed":
                raise OSError("simulated trace writer failure")
            return self.delegate.append(event_type, **fields)

    failed.recorder = FailingRecorder(failed.recorder)
    with pytest.raises(OSError, match="trace writer failure"):
        failed.run_observation(
            task_instruction="Move once.", context=pointer,
            image_path=workspace / pointer["main_image"],
        )
    assert not (failure_root / "control/submitted-action.json").exists()
    failed.close()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_loop_keeps_signed_tool_arguments_exact_across_dispatch(tmp_path, batch_size):
    from ora.agent.providers.tool_names import tool_alias
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    requests = []
    arguments = '{ "section" : "overview" }'

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {"role":"assistant", "content":"", "reasoning_content":"Read the manual.",
                "tool_calls":[{"id":f"read-{i}","type":"function","function":{
                    "name":tool_alias("robot.read_manual"),"arguments":arguments}}
                    for i in range(batch_size)]}
            finish = "tool_calls"
        else:
            prior = next(m for m in body["messages"] if m["role"] == "assistant")
            assert all(call["function"]["arguments"] == arguments for call in prior["tool_calls"])
            assert prior["reasoning_content"] == "Read the manual."
            assert len([m for m in body["messages"] if m["role"] == "tool"]) == batch_size
            message, finish = {"role":"assistant","content":"done"}, "stop"
        return httpx.Response(200, json={"choices":[{"message":message,"finish_reason":finish}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider("https://provider.invalid", "secret", client=client)
    planner = NativePlanner(tmp_path, "signed-dispatch", tools, provider, AGENT_RULES, PROMPT,
        "model", "native_minimal_v1", 32768, max_output_tokens=2048, stream=False,
        max_model_calls_per_observation=2)
    try:
        result = planner.run_observation(task_instruction="Read only.", context=pointer,
            image_path=workspace/pointer["main_image"])
        assert result["submitted"] is False
        assert len(requests) == 2
        assert not (tmp_path/"control/submitted-action.json").exists()
    finally:
        planner.close()
        client.close()


def test_truncated_tool_call_never_reaches_the_tool_gateway(tmp_path: Path) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([{
        "message": assistant_tool("truncated", "robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.2,
        }),
        "finish_reason": "length",
        "usage": usage(100),
    }])
    planner = NativePlanner(
        tmp_path, "truncated-tool", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    with pytest.raises(PolicySessionError, match="truncated"):
        planner.run_observation(
            task_instruction="Move once.", context=pointer,
            image_path=workspace / pointer["main_image"],
        )
    assert not (tmp_path / "control/submitted-action.json").exists()
    planner.close()


def test_invalid_provider_tool_json_retries_without_world_mutation(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([
        {
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "invalid_tool_input",
            "output_rejection": {
                "code": "invalid_tool_input_json",
                "provider_stop_reason": "max_tokens",
                "policy_outcome": "continue_same_observation",
                "world_state_changed": False,
            },
            "usage": usage(100),
        },
        {
            "message": assistant_tool(
                "valid-after-rejection", "robot.submit_action",
                {"action": "set_gripper", "arm": "left", "gripper": 0.2},
            ),
            "usage": usage(120),
        },
    ])
    planner = NativePlanner(
        tmp_path, "invalid-provider-tool-json", tools, provider,
        AGENT_RULES, PROMPT, "fake", "native_minimal_v1", 32768,
        max_output_tokens=2048,
    )
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )
    assert outcome["submitted"] is True
    assert outcome["attempts"] == 2
    assert len(provider.requests) == 2
    assert len(planner.calls) == 2
    assert planner.calls[0]["provider_output_status"] == "rejected"
    assert planner.calls[1]["provider_output_status"] == "accepted"
    events = [
        json.loads(line)
        for line in (tmp_path / "agent-events.jsonl").read_text().splitlines()
    ]
    assert sum(event["type"] == "provider_response" for event in events) == 2
    rejected = next(
        event for event in events
        if event["type"] == "provider_output_rejected"
    )
    assert rejected["payload"] == {
        "code": "invalid_tool_input_json",
        "provider_stop_reason": "max_tokens",
        "policy_outcome": "continue_same_observation",
        "world_state_changed": False,
    }
    assert sum(event["type"] == "action_proposed" for event in events) == 1
    continuation = next(
        message for message in planner.messages
        if message.get("role") == "user"
        and "invalid tool-input JSON" in str(message.get("content"))
    )
    assert continuation["_source_event_ids"] == [rejected["event_id"]]
    assert (tmp_path / "control/submitted-action.json").is_file()
    verification = verify_agent_trace(tmp_path)
    assert verification["errors"] == ["action_without_fresh_observation"]
    assert not any(
        "request_without_response" in error
        for error in verification["errors"]
    )
    planner.close()


def test_truncated_reasoning_without_tool_continues_same_observation(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([
        {
            "message": {
                "role": "assistant", "content": "",
                "reasoning_content": "unfinished but provider-visible reasoning",
            },
            "finish_reason": "length",
            "usage": usage(100),
        },
        {
            "message": assistant_tool(
                "continued-action", "robot.submit_action",
                {"action": "set_gripper", "arm": "left", "gripper": 0.2},
            ),
            "usage": usage(120),
        },
    ])
    planner = NativePlanner(
        tmp_path, "truncated-reasoning", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )
    assert outcome["submitted"] is True
    assert outcome["attempts"] == 2
    assert (tmp_path / "control/submitted-action.json").is_file()
    assert len(provider.requests) == 2
    continuation_messages = provider.requests[1]["messages"]
    assert any(
        message.get("role") == "assistant"
        and message.get("reasoning_content")
        == "unfinished but provider-visible reasoning"
        for message in planner.messages
    )
    assert any(
        message.get("role") == "user"
        and "same observation" in str(message.get("content"))
        for message in continuation_messages
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "agent-events.jsonl").read_text().splitlines()
    ]
    assert sum(event["type"] == "provider_output_truncated" for event in events) == 1
    truncated = next(
        event for event in events if event["type"] == "provider_output_truncated"
    )
    assert truncated["payload"]["policy_outcome"] == "continue_same_observation"
    continuation = next(
        message for message in planner.messages
        if message.get("role") == "user"
        and "same observation" in str(message.get("content"))
    )
    assert continuation["_source_event_ids"] == [truncated["event_id"]]
    planner.close()


def test_ordinary_no_action_continues_same_observation_until_action(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([
        {
            "message": {
                "role": "assistant",
                "content": "I should inspect this before acting.",
            },
            "finish_reason": "stop",
            "usage": usage(100),
        },
        {
            "message": assistant_tool(
                "continued-action", "robot.submit_action",
                {"action": "set_gripper", "arm": "left", "gripper": 0.2},
            ),
            "usage": usage(120),
        },
    ])
    planner = NativePlanner(
        tmp_path, "ordinary-no-action", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )

    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )

    assert outcome["submitted"] is True
    assert outcome["attempts"] == 2
    assert len(provider.requests) == 2
    assert any(
        message.get("role") == "user"
        and "No action was proposed for this observation" in str(message.get("content"))
        for message in provider.requests[1]["messages"]
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "agent-events.jsonl").read_text().splitlines()
    ]
    continuation_event = next(
        event for event in events
        if event["type"] == "provider_no_action_continuation"
    )
    continuation_message = next(
        message for message in planner.messages
        if message.get("role") == "user"
        and "No action was proposed for this observation" in str(message.get("content"))
    )
    assert continuation_event["payload"]["policy_outcome"] == "continue_same_observation"
    assert continuation_message["_source_event_ids"] == [
        continuation_event["event_id"]
    ]
    planner.close()


def test_repeated_tool_free_truncation_stops_at_observation_call_limit(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([
        {
            "message": {"role": "assistant", "content": "first partial"},
            "finish_reason": "length",
            "usage": usage(100),
        },
        {
            "message": {"role": "assistant", "content": "second partial"},
            "finish_reason": "length",
            "usage": usage(120),
        },
    ])
    planner = NativePlanner(
        tmp_path, "bounded-truncation", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
        max_model_calls_per_observation=2,
    )

    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )

    assert outcome == {
        "submitted": False, "termination": "model_call_limit", "attempts": 2,
    }
    assert len(provider.requests) == 2
    assert not (tmp_path / "control/submitted-action.json").exists()
    planner.close()
