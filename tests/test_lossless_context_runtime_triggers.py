"""Lossless context runtime triggers contract regressions."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Mapping
import pytest
from ora.agent.contracts import PolicySessionError
from ora.agent.native.tool_loop import _provider_estimated_input_tokens
from ora.agent.memory.artifacts import RecoverableArtifactToolRuntime
from test_lossless_context_runtime import (
    CHECKPOINT,
    MANAGEMENT,
    Summary,
    TOOLS,
    _BaseRuntime,
    _exchange_for_test,
    _image,
    _messages,
    _pipeline,
    _project_with_input_usage,
    _single_round_messages,
    _tool_content,
    _two_round_messages,
)


def test_l1_retains_current_result_then_externalizes_outside_near_window(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "run")
    messages = _messages(0)
    payload = json.dumps({"status": "actual", "diagnostic": "L" * 60_000})
    messages.extend(_exchange_for_test(
        "observe", "large-current", payload,
        decision="Inspect the current long diagnostic.",
    ))
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=0,
    )
    l1 = next(
        item for item in projected["manifest"]["passes"]
        if item["name"] == "l1_large_result_reference"
    )
    assert l1["changed_messages"] == []
    messages.extend(_messages(4)[6:])
    projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    assert any(t["kind"] == "lifecycle_read_only_externalization"
               for t in projected["manifest"]["projection_transitions"])
    visible = _tool_content(projected["messages"], "large-current")
    pointer = json.loads(visible)["recoverable_artifact"]
    _entry, recovered = pipeline.context_artifacts.resolve(pointer["ref"])
    assert recovered.decode("utf-8") == payload


def test_l2_externalizes_only_later_exact_duplicate_without_pressure(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "run", management={**MANAGEMENT, "trigger_inline_images": 24})
    messages = _messages(5)
    duplicate = json.dumps({
        "status": "ok", "diagnostic": "same-read-only-value-" * 240,
    })
    first = next(
        item for item in messages
        if item.get("tool_call_id") == "observe-0"
    )
    first["content"] = duplicate
    second_read = next(item for item in messages if item.get("tool_call_id") == "observe-1")
    second_read["content"] = duplicate
    messages.extend(_exchange_for_test(
        "observe", "duplicate-current", duplicate,
        decision="Read the same immutable value again.",
    ))
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    l2 = next(
        item for item in projected["manifest"]["passes"]
        if item["name"] == "l2_exact_duplicate_reference"
    )
    assert len(l2["changed_messages"]) == 1
    assert _tool_content(projected["messages"], "observe-0") == duplicate
    later = json.loads(_tool_content(
        projected["messages"], "observe-1",
    ))
    assert later["externalized"] is True
    assert later["recoverable_artifact"]["open"] == "artifact.read"
    assert _tool_content(projected["messages"], "duplicate-current") == duplicate


def test_l3_protects_all_open_media_and_configured_recent_completed_tail(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "run")
    messages = _messages(1)
    for call_id in ("derived-old", "derived-latest"):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"READ-ONLY feedback {call_id}"},
                _image(call_id),
            ],
            "_tool_image_call_id": call_id,
            "_source_event_ids": [f"agent-event-{call_id}"],
        })
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=1,
    )
    by_call = {
        call_id: next(
            message for message in projected["messages"]
            if isinstance(message.get("content"), list)
            and any(
                isinstance(block, Mapping)
                and block.get("type") == "text"
                and str(block.get("text", "")).endswith(call_id)
                for block in message["content"]
            )
        )
        for call_id in ("derived-old", "derived-latest")
    }
    assert any(
        block.get("type") == "image_url"
        for block in by_call["derived-old"]["content"]
        if isinstance(block, Mapping)
    )
    assert any(
        block.get("type") == "image_url"
        for block in by_call["derived-latest"]["content"]
        if isinstance(block, Mapping)
    )
    assert projected["manifest"]["active_inline_images"] == 4
    assert projected["manifest"][
        "lifecycle_media_externalized_through_action_step"
    ] is None
    assert projected["manifest"]["protected_inline_media_action_steps"] == [0, 1]
    assert projected["manifest"]["projection_transitions"] == []

    no_completed_tail = _pipeline(
        tmp_path / "tail-zero", management={
            **MANAGEMENT, "inline_media_tail_action_steps": 0,
        },
    ).project(messages=messages, tools=TOOLS, observation_step=1)
    assert no_completed_tail["manifest"]["active_inline_images"] == 3
    assert no_completed_tail["manifest"][
        "lifecycle_media_externalized_through_action_step"
    ] == 0
    assert no_completed_tail["manifest"][
        "protected_inline_media_action_steps"
    ] == [1]


def test_small_tool_payload_stays_exact_but_backing_ref_excludes_private_trace(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "run")
    messages = _messages(5)
    target = next(
        message for message in messages
        if message.get("role") == "tool"
        and message.get("name") == "observe"
        and message.get("tool_call_id") == "observe-3"
    )
    exact_visible = str(target["content"])
    target["_artifact_ref"] = pipeline.artifacts.put_json({
        "status": "actual",
        "private_mask": "must never become policy readable",
        "path": "/private/worker/path",
    }, kind="canonical_private_tool_result")
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=4,
    )
    visible = next(
        message for message in projected["messages"]
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "observe-3"
    )
    assert visible["content"] == exact_visible
    assert "recoverable_artifact" not in json.loads(visible["content"])
    assert "/private/worker/path" not in visible["content"]
    materialization = next(
        item for item in projected["manifest"]["passes"]
        if item["name"] == "recoverable_payload_materialization"
    )
    expected_sha = hashlib.sha256(exact_visible.encode("utf-8")).hexdigest()
    reference = next(
        item for item in materialization["artifact_refs"]
        if item["sha256"] == expected_sha
    )
    _entry, recovered = pipeline.context_artifacts.resolve(
        reference["ref"],
    )
    assert recovered.decode("utf-8") == exact_visible
    assert b"private_mask" not in recovered
    duplicate_pass = next(
        item for item in projected["manifest"]["passes"]
        if item["name"] == "l2_exact_duplicate_reference"
    )
    assert duplicate_pass["changed_messages"] == []


def test_l4_externalizes_mid_sized_and_second_order_read_results(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path / "run",
        management={**MANAGEMENT, "trigger_estimated_tokens": 1},
    )
    messages = _messages(5)
    expected: dict[str, str] = {}
    for tool_call_id, name, payload in (
        (
            "observe-0", "observe",
            json.dumps({"status": "actual", "diagnostic": "M" * 12_000}),
        ),
        (
            "observe-1", "artifact.read",
            json.dumps({
                "status": "ok", "ref": "artifact://source/" + "a" * 64,
                "offset": 0, "next_offset": 8192, "eof": False,
                "content": "R" * 8_000,
            }),
        ),
    ):
        message = next(
            item for item in messages
            if item.get("role") == "tool"
            and item.get("tool_call_id") == tool_call_id
        )
        message["name"] = name
        message["content"] = payload
        call = next(
            call
            for item in messages
            if item.get("role") == "assistant"
            for call in item.get("tool_calls", [])
            if call.get("id") == tool_call_id
        )
        call["function"]["name"] = name
        expected[tool_call_id] = payload
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    l4 = next(
        item for item in projected["manifest"]["passes"]
        if item["name"] == "l4_read_only_result_reference"
    )
    assert len(l4["changed_messages"]) >= 2
    assert any(
        item["kind"] == "pressure_externalization"
        for item in projected["manifest"]["projection_transitions"]
    )
    runtime = RecoverableArtifactToolRuntime(
        base=_BaseRuntime(), run_root=tmp_path / "run",
        event_path=tmp_path / "run/unused.jsonl",
    )
    runtime.bind_context_artifacts(
        pipeline.context_artifacts, event_sink=lambda *_args, **_kwargs: None,
    )
    for tool_call_id, original in expected.items():
        visible = next(
            item for item in projected["messages"]
            if item.get("role") == "tool"
            and item.get("tool_call_id") == tool_call_id
        )
        payload = json.loads(visible["content"])
        pointer = payload["recoverable_artifact"]
        assert pointer["open"] == "artifact.read"
        arguments = {"ref": pointer["ref"]}
        chunks: list[str] = []
        offset = 0
        while True:
            result = runtime.invoke_with_context(
                pointer["open"], {**arguments, "offset": offset},
                call_context={},
            )
            chunks.append(result["content"])
            if result["eof"]:
                break
            offset = result["next_offset"]
        assert "".join(chunks) == original


def test_l5_capsules_one_closed_episode_after_exact_provider_high_watermark(
    tmp_path: Path,
) -> None:
    management = {
        **MANAGEMENT,
        "trigger_inline_images": 2,
        "trigger_estimated_tokens": 1000,
    }
    summary = Summary(CHECKPOINT)
    pipeline = _pipeline(
        tmp_path / "semantic", management=management,
        context_window=55_000, summary=summary,
    )
    first = pipeline.project(
        messages=_messages(5, large_decisions={0, 1}),
        tools=TOOLS, observation_step=5,
    )
    first_manifest = first["manifest"]
    assert first_manifest["episode_checkpoint_count"] == 0
    assert first_manifest["token_estimate_status"] == "provider_usage_unavailable"
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first_manifest),
        input_tokens=11_000,
        projection_epoch=first_manifest["projection_epoch"],
        provider_wire_request_sha256="a" * 64,
    )
    projected = pipeline.project(
        messages=_messages(5, large_decisions={0, 1}),
        tools=TOOLS, observation_step=5,
    )
    transitions = projected["manifest"]["projection_transitions"]
    semantic = next(item for item in transitions if item["kind"] == "episode_capsule")
    assert semantic["semantic_admission_provider_input_tokens"] >= semantic[
        "semantic_high_watermark_tokens"
    ]
    assert projected["manifest"]["episode_checkpoint_count"] == 1
    assert projected["manifest"]["episode_checkpointed_rounds"] == [1]
    assert semantic["exact_transcript"]["open"]["tool"] == "artifact.read"
    _entry, transcript = pipeline.context_artifacts.resolve(
        semantic["exact_transcript"]["ref"]
    )
    assert b"decision and reflection 0" in transcript
    assert any(
        str(message.get("content", "")).startswith("<episode_context_capsule>")
        for message in projected["messages"]
    )
    assert len(summary.calls) == 1


@pytest.mark.parametrize("closed_episode", [False, True])
def test_continuous_actions_do_not_erase_reported_compaction_pressure(
    tmp_path: Path, closed_episode: bool,
) -> None:
    """A new observation must not require a spare read-only policy call."""
    pipeline = _pipeline(tmp_path, context_window=55_000)
    messages = _messages if closed_episode else _single_round_messages
    first = pipeline.project(messages=messages(5), tools=TOOLS, observation_step=5)
    source_epoch = first["manifest"]["projection_epoch"]
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=11_000 if closed_episode else 41_000,
        projection_epoch=source_epoch,
        provider_wire_request_sha256="a" * 64,
    )
    # An action and new camera frame arrive before the next policy request.
    projected = pipeline.project(messages=messages(6), tools=TOOLS, observation_step=6)
    manifest = projected["manifest"]
    kind = "episode_capsule" if closed_episode else "emergency_compaction"
    transition = next(t for t in manifest["projection_transitions"] if t["kind"] == kind)
    assert transition["semantic_admission_provider_projection_epoch"] == source_epoch
    assert transition["semantic_admission_provider_wire_request_sha256"] == "a" * 64
    assert manifest["projection_epoch"] == source_epoch + 1
    assert manifest["causal_forbidden_field_scan"] == "passed"
    assert len(pipeline.summary_provider.calls) == 1
    pipeline.project(messages=messages(7), tools=TOOLS, observation_step=7)
    assert len(pipeline.summary_provider.calls) == 1


@pytest.mark.parametrize("epoch_offset,input_tokens", [(1, 41_000), (0, 1_000)])
def test_continuous_actions_require_current_above_threshold_provider_usage(
    tmp_path: Path, epoch_offset: int, input_tokens: int,
) -> None:
    pipeline = _pipeline(tmp_path, context_window=55_000)
    first = pipeline.project(messages=_messages(5), tools=TOOLS, observation_step=5)
    if epoch_offset:
        with pytest.raises(PolicySessionError, match="projection epoch is stale"):
            pipeline.observe_provider_usage(
                estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
                input_tokens=input_tokens,
                projection_epoch=first["manifest"]["projection_epoch"] + epoch_offset,
                provider_wire_request_sha256="b" * 64,
            )
    else:
        pipeline.observe_provider_usage(
            estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
            input_tokens=input_tokens,
            projection_epoch=first["manifest"]["projection_epoch"] + epoch_offset,
            provider_wire_request_sha256="b" * 64,
        )
    pipeline.project(messages=_messages(6), tools=TOOLS, observation_step=6)
    assert len(pipeline.summary_provider.calls) == 0


def test_rough_overestimate_cannot_admit_summary_or_context_exhaustion(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path / "rough", management={
            **MANAGEMENT,
            "trigger_inline_images": 2,
            "trigger_estimated_tokens": 1000,
        }, context_window=55_000,
    )
    messages = _messages(5, large_decisions={0, 1})
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    assert first["manifest"]["estimated_input_tokens"] > first["manifest"][
        "hard_input_tokens"
    ]
    assert first["manifest"]["episode_checkpoint_count"] == 0
    assert first["manifest"]["episode_checkpoint_admission"]["reason"] == (
        "provider_usage_unavailable"
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=9_000,
        projection_epoch=first["manifest"]["projection_epoch"],
        provider_wire_request_sha256="b" * 64,
    )
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    manifest = projected["manifest"]
    assert manifest["estimated_input_tokens"] == 9_000
    assert manifest["episode_checkpoint_count"] == 0
    assert manifest["episode_checkpoint_admission"]["reason"] == (
        "below_high_watermark"
    )
    assert manifest["episode_checkpoint_admission"]["provider_input_tokens"] == 9_000


def test_invalid_l5_summary_is_rejected_and_original_episode_remains(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path / "invalid", management={
            **MANAGEMENT, "trigger_estimated_tokens": 1000,
        }, context_window=55_000, summary=Summary({}),
    )
    messages = _messages(5, large_decisions={0, 1})
    first = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=11_000,
        projection_epoch=first["manifest"]["projection_epoch"],
        provider_wire_request_sha256="c" * 64,
    )
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    assert projected["manifest"]["episode_checkpoint_admission"]["status"] == (
        "rejected"
    )
    assert projected["manifest"]["episode_checkpoint_count"] == 0
    assert projected["manifest"]["semantic_rejection_count"] == 1
    assert any(
        "round observation 0" in str(message.get("content"))
        for message in projected["messages"]
    )


def test_l6_target_miss_is_telemetry_until_actual_hard_limit(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path / "target", management={
            **MANAGEMENT,
            "trigger_inline_images": 2,
            "trigger_estimated_tokens": 1000,
        }, context_window=55_000,
    )
    messages = _single_round_messages(5, large_decisions={0, 1})
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=41_000,
        projection_epoch=first["manifest"]["projection_epoch"],
        provider_wire_request_sha256="e" * 64,
    )
    semantic = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=5,
    )
    transition = next(
        item for item in semantic["manifest"]["projection_transitions"]
        if item["kind"] == "emergency_compaction"
    )
    assert transition["exact_transcript"]["ref"].startswith(
        "artifact://policy-session-1/"
    )
    _entry, exact_history = pipeline.context_artifacts.resolve(
        transition["exact_transcript"]["ref"]
    )
    assert b"decision and reflection 0" in exact_history
    assert transition["exact_transcript"]["ref"] in str(semantic["messages"])
    target = transition[
        "target_input_tokens"
    ]
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(semantic["manifest"]),
        input_tokens=target + 1,
        projection_epoch=semantic["manifest"]["projection_epoch"],
        provider_wire_request_sha256="f" * 64,
    )
    projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    assert projected["manifest"]["emergency_target_validation"]["status"] == (
        "failed"
    )
    assert projected["manifest"]["emergency_compaction_count"] == 1


@pytest.mark.parametrize("completed", [1, 5])
def test_exact_provider_hard_limit_is_the_terminal_context_fault(
    tmp_path: Path, completed: int,
) -> None:
    pipeline = _pipeline(
        tmp_path / "hard-limit", management={
            **MANAGEMENT,
            "trigger_inline_images": 2,
            "trigger_estimated_tokens": 1000,
        }, context_window=55_000,
    )
    messages = _single_round_messages(completed, large_decisions={0, 1})
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=first["manifest"]["hard_input_tokens"] + 1,
        projection_epoch=first["manifest"]["projection_epoch"],
        provider_wire_request_sha256="6" * 64,
    )

    if completed == 1:
        with pytest.raises(PolicySessionError, match="context_exhausted"):
            pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    else:
        recovered = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
        assert recovered["manifest"]["emergency_compaction_count"] == 1


def test_closed_episode_capsule_supersedes_overlapping_open_episode_l6(
    tmp_path: Path,
) -> None:
    """A partial L6 frontier must never strand L5 inside an Episode."""

    pipeline = _pipeline(
        tmp_path / "l6-to-l5-takeover", management={
            **MANAGEMENT,
            "trigger_inline_images": 2,
            "trigger_estimated_tokens": 1000,
        }, context_window=55_000,
    )
    open_episode = _single_round_messages(5, large_decisions={0, 1})
    first = pipeline.project(
        messages=open_episode, tools=TOOLS, observation_step=5,
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(first["manifest"]),
        input_tokens=43_000,
        projection_epoch=first["manifest"]["projection_epoch"],
        provider_wire_request_sha256="7" * 64,
    )
    emergency = pipeline.project(
        messages=open_episode, tools=TOOLS, observation_step=5,
    )
    emergency_transition = next(
        item for item in emergency["manifest"]["projection_transitions"]
        if item["kind"] == "emergency_compaction"
    )
    emergency_ref = emergency_transition["exact_transcript"]["ref"]
    assert emergency["manifest"]["episode_checkpoint_count"] == 0
    assert emergency["manifest"]["emergency_compaction_count"] == 1
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(emergency["manifest"]),
        input_tokens=41_000,
        projection_epoch=emergency["manifest"]["projection_epoch"],
        provider_wire_request_sha256="8" * 64,
    )

    closed_episode = _two_round_messages(
        first_round_action_steps=6,
        second_round_completed=3,
        large_decisions={0, 1},
    )
    takeover = pipeline.project(
        messages=closed_episode, tools=TOOLS, observation_step=9,
    )
    capsule_transition = next(
        item for item in takeover["manifest"]["projection_transitions"]
        if item["kind"] == "episode_capsule"
    )
    superseded = capsule_transition["superseded_emergency_checkpoint"]
    assert capsule_transition["checkpointed_action_steps_before"] == 0
    assert capsule_transition["checkpointed_action_steps_after"] == 6
    assert "superseded_overlapping_emergency_checkpoint" in (
        capsule_transition["reasons"]
    )
    assert superseded["exact_transcript"]["ref"] == emergency_ref
    assert takeover["manifest"]["episode_checkpointed_rounds"] == [1]
    assert takeover["manifest"]["emergency_compaction_count"] == 1
    assert takeover["manifest"]["emergency_checkpoint_sha256"] is None
    assert takeover["manifest"]["emergency_exact_transcript_ref"] is None
    assert takeover["manifest"]["checkpoint_slot_count"] == 1
    assert sum(
        str(message.get("content", "")).startswith(
            "<episode_context_capsule>"
        )
        for message in takeover["messages"]
    ) == 1
    assert not any(
        str(message.get("content", "")).startswith(
            "<emergency_context_checkpoint>"
        )
        for message in takeover["messages"]
    )
    _entry, full_episode = pipeline.context_artifacts.resolve(
        capsule_transition["exact_transcript"]["ref"]
    )
    assert b"decision and reflection 0" in full_episode
    assert b"decision and reflection 5" in full_episode

    # A later L6 starts after the full Episode frontier and may replace the
    # visible Capsule, while its cumulative exact transcript remains complete.
    extended = _two_round_messages(
        first_round_action_steps=6,
        second_round_completed=6,
        large_decisions={0, 1},
    )
    extended_projection = pipeline.project(
        messages=extended, tools=TOOLS, observation_step=12,
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(
            extended_projection["manifest"]
        ),
        input_tokens=43_000,
        projection_epoch=extended_projection["manifest"]["projection_epoch"],
        provider_wire_request_sha256="9" * 64,
    )
    future_emergency = pipeline.project(
        messages=extended, tools=TOOLS, observation_step=12,
    )
    future_transition = next(
        item for item in future_emergency["manifest"]["projection_transitions"]
        if item["kind"] == "emergency_compaction"
    )
    assert future_transition[
        "emergency_summarized_action_steps_before"
    ] == 6
    _entry, cumulative = pipeline.context_artifacts.resolve(
        future_transition["exact_transcript"]["ref"]
    )
    assert b"decision and reflection 0" in cumulative
    assert b"decision and reflection 8" in cumulative
    assert future_emergency["manifest"]["checkpoint_slot_count"] == 1


def test_the_estimate_trigger_scales_with_the_model_it_is_bound_to(
    tmp_path: Path,
) -> None:
    """L4's ceiling is a share of the window, as L5's already was.

    The absolute 48000 every shipped baseline carries was sized for a 200K
    window, where it is a quarter of the context. Bound to a 1M model it is
    4.7%, and the first B2 run rotated its projection epoch on 76 of 100
    calls, pushing inline images from 50 down to 6 -- while the whole
    un-externalized context would still have been 783K against a 1,013,760
    ceiling. Nothing needed to leave.

    L5 eighteen lines below already took max(absolute, ceiling x ratio). One
    layer scaled with the model and the layer above it did not.
    """

    # The figure every shipped baseline carries, not this module's fixture.
    shipped = {**MANAGEMENT, "trigger_estimated_tokens": 48_000,
               "episode_checkpoint_trigger_tokens": 150_000,
               "episode_checkpoint_trigger_ratio": 0.60}
    small = _pipeline(tmp_path / "small", context_window=200_000,
                      management=shipped)
    large = _pipeline(tmp_path / "large", context_window=1_048_576,
                      management=shipped)
    from ora.agent.context.native import ESTIMATE_PRESSURE_WINDOW_SHARE as ratio
    absolute = int(shipped["trigger_estimated_tokens"])

    # On a small window the absolute floor still governs: a share of 200K is
    # below it, and lowering the bar for a small model is not the point.
    assert small.estimate_pressure_trigger() == max(
        absolute, int(small.hard_input_tokens * ratio),
    )
    # On a large window the share governs, and it is far above the floor.
    assert large.estimate_pressure_trigger() == int(
        large.hard_input_tokens * ratio,
    )
    assert large.estimate_pressure_trigger() > absolute
    # Both layers now answer the same question the same way.
    assert large.estimate_pressure_trigger() < large.episode_checkpoint_trigger()


def test_l5_same_episode_rejection_has_backoff_and_finite_attempt_budget(tmp_path):
    summary = Summary({})
    pipeline = _pipeline(tmp_path, context_window=55_000, summary=summary)
    source_key = None
    for completed, attempts, reason in (
        (5, 1, "summary_rejected_original_retained"),
        (6, 1, "same_episode_retry_backoff"),
        (9, 2, "summary_rejected_original_retained"),
        (16, 2, "same_episode_retry_backoff"),
        (17, 3, "summary_rejected_original_retained"),
        (32, 3, "same_episode_retry_backoff"),
        (33, 4, "summary_rejected_original_retained"),
        (60, 4, "same_episode_retry_exhausted"),
    ):
        result = _project_with_input_usage(pipeline, _messages(completed), completed, 11_000)
        admission = result["manifest"]["episode_checkpoint_admission"]
        assert admission["reason"] == reason
        assert admission["attempts"] == attempts == len(summary.calls)
        assert admission["max_attempts"] == 4
        assert source_key is None or admission["retry_source_sha256"] == source_key
        source_key = admission["retry_source_sha256"]
        assert result["manifest"]["episode_checkpoint_count"] == 0
        assert any("round observation 0" in str(message.get("content")) for message in result["messages"])


def test_l5_can_recover_on_a_later_eligible_attempt(tmp_path):
    summary = Summary({})
    pipeline = _pipeline(tmp_path, context_window=55_000, summary=summary)
    _project_with_input_usage(pipeline, _messages(5), 5, 11_000)
    summary.output = CHECKPOINT
    deferred = _project_with_input_usage(pipeline, _messages(6), 6, 11_000)
    assert deferred["manifest"]["episode_checkpoint_count"] == 0
    admitted = _project_with_input_usage(pipeline, _messages(9), 9, 11_000)
    assert len(summary.calls) == 2
    assert admitted["manifest"]["episode_checkpoint_count"] == 1
    assert summary.calls[0]["source_refs"] == summary.calls[1]["source_refs"]


def test_deferred_l5_does_not_block_l6_at_emergency_watermark(tmp_path):
    summary = Summary({})
    pipeline = _pipeline(tmp_path, context_window=55_000, summary=summary)
    _project_with_input_usage(pipeline, _messages(5), 5, 11_000)
    summary.output = CHECKPOINT
    result = _project_with_input_usage(pipeline, _messages(6), 6, 43_000)
    assert result["manifest"]["episode_checkpoint_admission"]["reason"] == "same_episode_retry_backoff"
    assert result["manifest"]["emergency_compaction_count"] == 1
    assert result["manifest"]["episode_checkpoint_count"] == 0
    assert len(summary.calls) == 2
    transition = next(item for item in result["manifest"]["projection_transitions"]
                      if item["kind"] == "emergency_compaction")
    _, original = pipeline.context_artifacts.resolve(transition["exact_transcript"]["ref"])
    assert b"decision and reflection 0" in original


def _externalized_marker(body: str, preview_bytes: int) -> dict:
    """One externalized tool result, as the policy receives it."""

    import json as _json
    from ora.agent.context.projection import _recoverable_tool_marker

    digest = "0" * 64
    scope = "native-policy-" + "a" * 32
    reference = {
        "ref": f"artifact://{scope}/{digest}", "sha256": digest,
        "policy_scope": scope, "media_type": "application/json",
        "bytes": len(body.encode("utf-8")), "round": 1, "action_step": 0,
    }
    return _json.loads(_recoverable_tool_marker(
        {"content": body}, reference, preview_bytes=preview_bytes))


def test_a_cut_preview_says_it_was_cut_and_how_many_items_exist():
    """The shortfall, named, because the cut does not look like one.

    A three-candidate grasp_plan result arrives as one whole candidate, half of
    the second and no sign of a third: the slice ends wherever the byte budget
    ends, inside an object, under a status record that reads complete. Across
    619 runs and 63,954 tool calls the tool that opens the full result was
    called twice, both in one run and both correctly, so the format was never
    the obstacle -- nothing said the preview was partial.
    """

    import json as _json

    body = _json.dumps({
        "status": "candidates", "success": True,
        "candidates": [{"id": index, "pad": "x" * 200} for index in range(3)],
    })
    marker = _externalized_marker(body, 200)
    assert marker["preview_truncated"] is True
    assert marker["item_counts"] == {"candidates": 3}
    assert marker["preview_bytes"] == 200
    assert marker["result_bytes"] == len(body.encode("utf-8"))
    assert marker["preview_bytes"] < marker["result_bytes"]
    # The pointer and the tool that opens it stay exactly where they were.
    assert marker["recoverable_artifact"]["open"]
    assert marker["externalized"] is True


def test_a_preview_that_holds_everything_does_not_claim_a_shortfall():
    import json as _json

    body = _json.dumps({"status": "ok", "views": [1, 2]})
    marker = _externalized_marker(body, 10_000)
    assert marker["preview_truncated"] is False
    assert marker["preview_bytes"] == marker["result_bytes"]


def test_a_tool_summary_is_reported_as_a_summary_not_as_a_cut():
    """A declared summary is complete for what it covers; say which it is."""

    import json as _json

    body = _json.dumps({
        "status": "ok", "summary": {"queries": [{"q": 1}, {"q": 2}]},
        "results": [{"big": "y" * 500}] * 2,
    })
    marker = _externalized_marker(body, 200)
    assert marker["preview_is_summary"] is True
    assert marker["preview_truncated"] is False
    assert marker["item_counts"] == {"results": 2}
