"""Provider continuation stays bound to the assistant message it accompanies.

The Runtime binds opaque provider state (OpenAI ``reasoning_content``, Anthropic
signed thinking blocks) to a content address over the canonical assistant
message.  Every Harness stage that handles that message afterwards -- the
provider/tool dispatch loop, the lossless Context projection and the wire
request assembly -- must leave its canonical ``role``/``content``/``tool_calls``
untouched, or the next request replays provider-native reasoning against a
message the provider never produced.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ora.agent.context import ContextProjector
from ora.agent.context.native import NativeContextPipeline
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.evidence.trajectory import ArtifactStore
from ora.agent.memory.artifacts import ContextArtifactCatalog
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.native.session import NativePlanner
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
    canonical_assistant_sha256,
)
from ora.agent.providers.native import (
    OpenAICompatibleProvider, translate_anthropic_request,
)
from ora.agent.tools.robot import RobotToolRuntime
from ora.environment.synthetic import SyntheticSession


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "ora/policy/core/thin-react-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENT_RULES = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
MODEL = "continuation-binding-model"
OPENAI_CONTINUATION_KIND = "openai_chat_reasoning_content_v1"
ANTHROPIC_CONTINUATION_KIND = "anthropic_assistant_content_blocks.v1"

# The exact shape the GPT-6 RoboTwin cell died on: an empty assistant content
# string, provider reasoning that must be replayed, a dotted tool name that the
# OpenAI wire path aliases, and nested tool arguments whose key order is not the
# Harness canonical sorted order.
NON_CANONICAL_ARGUMENTS = (
    '{"queries":[{"arm":"left","query_id":"red_top","pose":'
    '{"position_m":[-0.138,-0.018,0.895],'
    '"orientation_xyzw":[0,0.7071068,0,0.7071068]}}],'
    '"variant_policy":"parallel_jaw_equivalents"}'
)
MANAGEMENT = {
    "schema": "ora.native-context-management.v4",
    "causal_exact_tail_action_steps": 3,
    "inline_media_tail_action_steps": 2,
    "eager_large_result_bytes": 50_000,
    "eager_preview_bytes": 2_048,
    "eager_exact_deduplication": True,
    "trigger_inline_images": 24,
    "trigger_estimated_tokens": 1_000_000,
    "trigger_provider_input_ratio": 0.70,
    "episode_checkpoint_trigger_tokens": 150_000,
    "episode_checkpoint_trigger_ratio": 0.60,
    "emergency_fallback_ratio": 0.80,
    "emergency_target_ratio": 0.65,
    "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600,
}


def _openai_assistant(
    call_id: str, name: str, arguments: str, *, reasoning: str, content: str = "",
) -> dict[str, Any]:
    """Build the message an OpenAI-compatible adapter returns, verbatim."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    message["reasoning_content"] = reasoning
    message["tool_calls"] = [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": arguments},
    }]
    message[PROVIDER_CONTINUATION_MESSAGE_KEY] = ProviderContinuationState.create(
        provider_protocol="openai_chat_completions",
        model=MODEL,
        native_kind=OPENAI_CONTINUATION_KIND,
        native_payload={"reasoning_content": reasoning},
        canonical_message=message,
    ).to_dict()
    return message


def _anthropic_assistant(
    call_id: str, name: str, arguments: Mapping[str, Any], *, thinking: str,
) -> dict[str, Any]:
    """Build the canonical message and signed blocks an Anthropic turn returns."""

    message: dict[str, Any] = {"role": "assistant", "content": ""}
    message["reasoning_content"] = thinking
    message["tool_calls"] = [{
        "id": call_id, "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                dict(arguments), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }]
    blocks = [
        {"type": "thinking", "thinking": thinking, "signature": "signed-abc"},
        {
            "type": "tool_use", "id": call_id,
            "name": name.replace(".", "_") if "." in name else name,
            "input": dict(arguments),
        },
    ]
    message[PROVIDER_CONTINUATION_MESSAGE_KEY] = ProviderContinuationState.create(
        provider_protocol="anthropic_messages",
        model=MODEL,
        native_kind=ANTHROPIC_CONTINUATION_KIND,
        native_payload={"content": blocks},
        canonical_message=message,
        response_id="msg-continuation",
    ).to_dict()
    return message


class _ScriptedProvider:
    """Replay recorded assistant messages through a real wire translator."""

    def __init__(
        self, responses: Sequence[Mapping[str, Any]], *, protocol: str,
        prepare: Any,
    ) -> None:
        self._responses = [copy.deepcopy(dict(item)) for item in responses]
        self.protocol = protocol
        self.prepare_request = prepare
        self.requests: list[dict[str, Any]] = []
        self.wire_requests: list[dict[str, Any]] = []
        self.index = 0
        self.closed = False

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str,
    ) -> dict[str, Any]:
        if self.index >= len(self._responses):
            raise AssertionError("scripted provider is exhausted")
        self.requests.append(copy.deepcopy(dict(request)))
        self.wire_requests.append(self.prepare_request(request))
        message = self._responses[self.index]
        self.index += 1
        return {
            "message": copy.deepcopy(message),
            "usage": None,
            "finish_reason": (
                "tool_calls" if message.get("tool_calls") else "stop"
            ),
            "raw_events": [{"scripted": True}],
            "model_call_id": model_call_id,
            "started_unix_s": 0.0,
            "finished_unix_s": 0.0,
            "attempts": 1,
        }

    def close(self) -> None:
        self.closed = True


def _native_boundary(tmp_path: Path) -> tuple[dict[str, Any], Path, RobotToolRuntime]:
    source = tmp_path / "policy-observations"
    environment = SyntheticSession(
        tmp_path / "native/environment", "continuation-binding", 0,
    )
    environment.start()
    observation, _ = environment.observe(source)
    store = PolicyStateStore(tmp_path, "continuation-binding", "continuation-plan")
    state = store.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        cause="reset",
        task_instruction="Move once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 8},
    )
    record = store.steps[-1]
    workspace = tmp_path / "sandbox/workspace"
    control = tmp_path / "control"
    projector = ContextProjector(workspace, control, PROMPT, MANUAL, AGENT_RULES)
    pointer = projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="Move once.",
        remaining_budget={"robot_decisions": 1, "model_calls": 8},
        policy_state={
            "step": state["step"],
            "path": record["path"],
            "sha256": record["sha256"],
            "state_sha256": state["state_sha256"],
        },
    )
    tools = RobotToolRuntime(
        control, MANUAL, tmp_path / "events.jsonl", BOUNDS,
        state_root=store.root,
    )
    return pointer, workspace, tools


def _run_native_turn(
    tmp_path: Path, provider: _ScriptedProvider,
) -> tuple[NativePlanner, dict[str, Any]]:
    pointer, workspace, tools = _native_boundary(tmp_path)
    planner = NativePlanner(
        tmp_path, "continuation-binding", tools, provider, AGENT_RULES, PROMPT,
        MODEL, "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )
    planner.close()
    return planner, outcome


def _history_assistant(planner: NativePlanner) -> dict[str, Any]:
    return next(
        message for message in planner.messages
        if message.get("role") == "assistant"
        and PROVIDER_CONTINUATION_MESSAGE_KEY in message
    )


def test_openai_dispatch_keeps_the_bound_assistant_message_verbatim(
    tmp_path: Path,
) -> None:
    """Dispatch normalization must not re-serialize history tool arguments."""

    first = _openai_assistant(
        "call-probe", "vision.inspect",
        '{"resolution":"thumbnail","camera":"head_camera"}',
        reasoning="**Preparing a top-down view**",
    )
    second = {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": "call-move", "type": "function",
            "function": {
                "name": "robot.submit_action",
                "arguments": json.dumps({
                    "action": "set_gripper", "arm": "left", "gripper": 0.5,
                }),
            },
        }],
    }
    provider = _ScriptedProvider(
        [first, second], protocol="openai_chat_completions",
        prepare=OpenAICompatibleProvider.prepare_request,
    )
    planner, outcome = _run_native_turn(tmp_path, provider)

    assert outcome["submitted"] is True
    assistant = _history_assistant(planner)
    # The provider's exact bytes survive dispatch: unsorted key order intact.
    assert assistant["tool_calls"][0]["function"]["arguments"] == (
        '{"resolution":"thumbnail","camera":"head_camera"}'
    )
    state = assistant[PROVIDER_CONTINUATION_MESSAGE_KEY]
    assert canonical_assistant_sha256(assistant) == (
        state["canonical_message_sha256"]
    )
    # The follow-up wire request replayed the provider reasoning against that
    # same message, under the alias the dotted tool name receives.
    replayed = [
        message for message in provider.wire_requests[1]["messages"]
        if message.get("role") == "assistant"
    ]
    assert len(replayed) == 1
    assert replayed[0]["reasoning_content"] == "**Preparing a top-down view**"
    assert replayed[0]["tool_calls"][0]["function"]["name"].startswith(
        "vision_inspect_"
    )
    assert PROVIDER_CONTINUATION_MESSAGE_KEY not in replayed[0]


def test_anthropic_dispatch_keeps_the_bound_assistant_message_verbatim(
    tmp_path: Path,
) -> None:
    """The shared dispatch loop must not disturb an Anthropic continuation."""

    first = _anthropic_assistant(
        "call-probe", "vision.inspect",
        {"camera": "head_camera", "resolution": "thumbnail"},
        thinking="private anthropic plan",
    )
    second = {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": "call-move", "type": "function",
            "function": {
                "name": "robot.submit_action",
                "arguments": json.dumps({
                    "action": "set_gripper", "arm": "left", "gripper": 0.5,
                }),
            },
        }],
    }
    provider = _ScriptedProvider(
        [first, second], protocol="anthropic_messages",
        prepare=translate_anthropic_request,
    )
    planner, outcome = _run_native_turn(tmp_path, provider)

    assert outcome["submitted"] is True
    assistant = _history_assistant(planner)
    assert canonical_assistant_sha256(assistant) == (
        assistant[PROVIDER_CONTINUATION_MESSAGE_KEY]["canonical_message_sha256"]
    )
    blocks = [
        block
        for message in provider.wire_requests[1]["messages"]
        if message.get("role") == "assistant"
        for block in message["content"]
    ]
    assert [block["type"] for block in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["signature"] == "signed-abc"


def test_continuation_over_an_id_less_tool_call_fails_closed(
    tmp_path: Path,
) -> None:
    """Repairing a missing call id would silently invalidate the binding."""

    message: dict[str, Any] = {
        "role": "assistant", "content": "",
        "reasoning_content": "unstable provider",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "robot.read_manual", "arguments": "{}"},
        }],
    }
    message[PROVIDER_CONTINUATION_MESSAGE_KEY] = ProviderContinuationState.create(
        provider_protocol="openai_chat_completions",
        model=MODEL,
        native_kind=OPENAI_CONTINUATION_KIND,
        native_payload={"reasoning_content": "unstable provider"},
        canonical_message=message,
    ).to_dict()
    provider = _ScriptedProvider(
        [message], protocol="openai_chat_completions",
        prepare=OpenAICompatibleProvider.prepare_request,
    )
    with pytest.raises(PolicySessionError, match="without a stable id"):
        _run_native_turn(tmp_path, provider)


class _RejectingSummary:
    @staticmethod
    def summarize(
        *, messages: Sequence[Mapping[str, Any]],
        source_refs: Sequence[Mapping[str, Any]],
    ) -> Any:  # pragma: no cover - no semantic layer is expected here
        raise AssertionError("no semantic Context layer applies to this fixture")


def _lossless_messages(assistant: Mapping[str, Any]) -> list[dict[str, Any]]:
    assistant = copy.deepcopy(dict(assistant))
    assistant["_source_event_ids"] = ["agent-event-000005"]
    call = assistant["tool_calls"][0]
    return [
        {
            "role": "system", "content": "harness system",
            "_protected_context": True, "_context_source": "harness_system",
        },
        {
            "role": "user", "content": "project instructions",
            "_protected_context": True,
            "_context_source": "project_instructions",
        },
        {
            "role": "user", "content": "episode task",
            "_protected_context": True, "_context_source": "episode_task",
        },
        {
            "role": "user", "content": "ACTUAL observation",
            "_observation": {"step": 1, "receipt": None, "keyframe": True},
            "_session_round": 1,
            "_source_event_ids": ["agent-event-000004"],
        },
        assistant,
        {
            "role": "tool", "tool_call_id": call["id"],
            "name": call["function"]["name"],
            "content": json.dumps({"ok": True, "reachable": True}, sort_keys=True),
            "_source_event_ids": ["agent-event-000006"],
        },
    ]


class _LosslessProvider:
    protocol = "openai_chat_completions"
    prepare_request = staticmethod(OpenAICompatibleProvider.prepare_request)


def _project_and_assemble(
    root: Path, assistant: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one assistant turn through lossless v4 and the OpenAI wire path."""

    messages = _lossless_messages(assistant)
    name = messages[4]["tool_calls"][0]["function"]["name"]
    artifacts = ArtifactStore(root)
    pipeline = NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=262_144, max_output_tokens=16_384,
        retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1",
        summary_provider=_RejectingSummary(),
        context_management=MANAGEMENT,
        context_artifacts=ContextArtifactCatalog(
            root, "policy-session-binding", artifacts,
        ),
        mutation_tools=("robot.submit_action",),
    )
    projected = pipeline.project(
        messages=messages, tools=[{
            "type": "function",
            "function": {
                "name": name, "description": "probe",
                "parameters": {"type": "object"},
            },
        }], observation_step=1,
    )
    prepared = NativeRequestAssembler(
        model=MODEL, reasoning_effort="medium", max_output_tokens=16_384,
        stream=True, provider=_LosslessProvider(),
    ).prepare(projected)
    return (
        next(
            message for message in projected["messages"]
            if message.get("role") == "assistant"
        ),
        next(
            message for message in prepared.wire["messages"]
            if message.get("role") == "assistant"
        ),
    )


def test_lossless_v4_projection_and_openai_assembly_preserve_the_binding(
    tmp_path: Path,
) -> None:
    """L1-L6 must not rewrite a continuation-bearing assistant message.

    The first case is the message the dispatch loop actually leaves in session
    history, so the whole capture → dispatch → projection → assembly chain is
    covered; the second is the exact dead-run shape, with a dotted tool name and
    a nested object whose keys are not in canonical order.
    """

    first = _openai_assistant(
        "call-probe", "vision.inspect",
        '{"resolution":"thumbnail","camera":"head_camera"}',
        reasoning="**Preparing a top-down view**",
    )
    provider = _ScriptedProvider(
        [first, {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "call-move", "type": "function",
                "function": {
                    "name": "robot.submit_action",
                    "arguments": json.dumps({
                        "action": "set_gripper", "arm": "left", "gripper": 0.5,
                    }),
                },
            }],
        }], protocol="openai_chat_completions",
        prepare=OpenAICompatibleProvider.prepare_request,
    )
    planner, _outcome = _run_native_turn(tmp_path / "session", provider)
    dispatched = _history_assistant(planner)

    for label, assistant, arguments, alias in (
        (
            "dispatched", dispatched,
            '{"resolution":"thumbnail","camera":"head_camera"}',
            "vision_inspect_",
        ),
        (
            "dead-run", _openai_assistant(
                "call-probe", "motion.probe", NON_CANONICAL_ARGUMENTS,
                reasoning="**Preparing topdown quaternion**",
            ), NON_CANONICAL_ARGUMENTS, "motion_probe_",
        ),
    ):
        bound = assistant[PROVIDER_CONTINUATION_MESSAGE_KEY][
            "canonical_message_sha256"
        ]
        projected, wire = _project_and_assemble(
            tmp_path / f"projection-{label}", assistant,
        )
        assert projected["content"] == "", label
        assert projected["tool_calls"][0]["function"]["arguments"] == (
            arguments
        ), label
        assert canonical_assistant_sha256(projected) == bound, label
        assert wire["reasoning_content"] == assistant["reasoning_content"], label
        assert wire["tool_calls"][0]["function"]["name"].startswith(alias), label
        assert wire["tool_calls"][0]["function"]["arguments"] == arguments, label
