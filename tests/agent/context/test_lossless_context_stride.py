"""Cache-stable stride aging of the lossless Native Context tails (v5).

Context management v5 keeps the v4 tails as minimums and ages older
ActionSteps only in batches of ``tail_aging_stride_action_steps``. Between
batches every Provider request must be a strict extension of the previous one,
so a prefix cache that reuses only an exactly extended request keeps hitting.
These tests drive one synthetic multi-step episode through the real pipeline
and request assembler and count the history rewrites the assembler reports.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ora.agent.context.management import (
    CONTEXT_MANAGEMENT_V4_SCHEMA,
    CONTEXT_MANAGEMENT_V5_SCHEMA,
    ContextManagementConfigError,
    validate_context_management,
    validate_context_management_v4,
    validate_context_management_v5,
)
from ora.agent.context.native import NativeContextPipeline
from ora.agent.contracts import PolicySessionError, SummaryResult
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from ora.agent.memory.artifacts import (
    ContextArtifactCatalog, RECOVERABLE_ARTIFACT_TAG,
    decode_recoverable_artifact_marker,
)
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import AnthropicMessagesProvider
from ora.orchestrator.context_coverage import coverage_violations

CAMERAS = ("head", "left_wrist", "right_wrist")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "move_pose", "description": "move",
        "parameters": {"type": "object"},
    },
}]
V4 = {
    "schema": CONTEXT_MANAGEMENT_V4_SCHEMA,
    "causal_exact_tail_action_steps": 3,
    "inline_media_tail_action_steps": 2,
    "eager_large_result_bytes": 50_000,
    "eager_preview_bytes": 2_048,
    "eager_exact_deduplication": True,
    # Every request carries more than two images, so L4 is always triggered.
    "trigger_inline_images": 2,
    "trigger_estimated_tokens": 48_000,
    "trigger_provider_input_ratio": 0.70,
    "episode_checkpoint_trigger_tokens": 150_000,
    "episode_checkpoint_trigger_ratio": 0.60,
    "emergency_fallback_ratio": 0.80,
    "emergency_target_ratio": 0.65,
    "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600,
}


def _v5(stride: int, **overrides: Any) -> dict[str, Any]:
    return {
        **V4, "schema": CONTEXT_MANAGEMENT_V5_SCHEMA,
        "tail_aging_stride_action_steps": stride, **overrides,
    }


CHECKPOINT = {
    "task_progress": ["The object has not been confirmed as transported."],
    "confirmed_facts": ["The retained actual view is authoritative."],
    "invalidated_beliefs": ["An older alignment belief may be stale."],
    "failed_attempts": [{
        "failure": "The earlier attempt did not establish transport.",
        "evidence": "The retained evidence did not confirm attachment.",
        "correction": "Re-ground before the next grasp attempt.",
    }],
    "current_constraints": ["Use qualitative state only."],
    "do_not_repeat": ["Do not infer attachment from closure."],
    "re_ground_next": ["Inspect the current actual view."],
}


class _Summary:
    """A valid compactor; the tests count its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, *, messages: Sequence[Mapping[str, Any]],
                  source_refs: Sequence[Mapping[str, Any]]) -> SummaryResult:
        self.calls += 1
        refs = copy.deepcopy(list(source_refs))
        return SummaryResult(
            content=json.dumps(CHECKPOINT), source_refs=refs,
            source_set_sha256=hashlib.sha256(canonical_bytes(refs)).hexdigest(),
            metadata={"call_id": f"summary-{self.calls}", "attempts": 1},
        )


def _image(label: str) -> dict[str, Any]:
    payload = base64.b64encode(label.encode("utf-8")).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + payload}}


def _observation(step: int, session_round: int) -> dict[str, Any]:
    body = {
        "current_actual": {
            "step": step, "actual_sequence": step,
            "observation_id": f"observation-{step:04d}",
            # The bulky measured body L4 replaces with a recoverable pointer.
            "camera_geometry": {
                camera: {"intrinsic_cv": [[600.0 + step, 0.0, 320.0]] * 24}
                for camera in CAMERAS
            },
        },
        "comparison": {"available": step > 0},
        "terminal_lifecycle": False,
    }
    content: list[dict[str, Any]] = [{
        "type": "text", "text": json.dumps(body, sort_keys=True),
    }]
    for camera in CAMERAS:
        content.extend([
            {"type": "text", "text": f"CURRENT ACTUAL · {camera}"},
            _image(f"{camera}-step-{step}"),
        ])
    return {
        "role": "user", "content": content,
        "_observation": {
            "step": step,
            "receipt": (
                {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
                if step else None
            ),
        },
        "_session_round": session_round,
        "_source_event_ids": [f"agent-event-{step * 10:06d}"],
    }


def _read_exchange(step: int) -> list[dict[str, Any]]:
    call_id = f"observe-{step}"
    return [{
        "role": "assistant", "content": f"inspect before acting {step}",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "observe", "arguments": "{}"}}],
        "_source_event_ids": [f"agent-event-{step * 10 + 1:06d}"],
    }, {
        "role": "tool", "tool_call_id": call_id, "name": "observe",
        "content": json.dumps({
            "status": "actual", "views": list(CAMERAS),
            "detail": f"read-only body {step} " + "x" * 4_000,
        }, sort_keys=True),
        "_source_event_ids": [f"agent-event-{step * 10 + 2:06d}"],
    }]


def _mutation_exchange(step: int) -> list[dict[str, Any]]:
    call_id = f"mutation-{step}"
    return [{
        "role": "assistant", "content": f"act {step}",
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "move_pose", "arguments": json.dumps({"step": step}),
        }}],
        "_source_event_ids": [f"agent-event-{step * 10 + 3:06d}"],
    }, {
        "role": "tool", "tool_call_id": call_id, "name": "move_pose",
        "content": json.dumps({"status": "submitted", "step": step}),
        "_source_event_ids": [f"agent-event-{step * 10 + 4:06d}"],
    }]


def _episode(
    completed: int, *, second_round_from: int | None = None,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Return every request snapshot of an append-only synthetic episode.

    Each ActionStep is two Provider calls, as the measured EXP11A g015 cadence
    (1.85 calls per ActionStep): one on the fresh observation and one after a
    read-only tool result. The mutation then closes the step.
    """

    history: list[dict[str, Any]] = [
        {"role": "system", "content": "system", "_protected_context": True,
         "_context_source": "harness_system"},
        {"role": "user", "content": "embodiment", "_protected_context": True,
         "_context_source": "project_instructions"},
        {"role": "user", "content": "task", "_protected_context": True,
         "_context_source": "episode_task"},
    ]
    snapshots: list[tuple[int, list[dict[str, Any]]]] = []
    for step in range(completed + 1):
        session_round = (
            2 if second_round_from is not None and step >= second_round_from
            else 1
        )
        if step == second_round_from:
            history.append({
                "role": "user",
                "content": (
                    '<official_round_result>\n{"instruction":"reflect",'
                    '"passed":false,"round":1}\n</official_round_result>'
                ),
                "_protected_context": True,
                "_context_source": "official_round_result_1",
                "_source_event_ids": ["agent-event-900000"],
            })
        history.append(_observation(step, session_round))
        snapshots.append((step, list(history)))
        history.extend(_read_exchange(step))
        snapshots.append((step, list(history)))
        if step < completed:
            history.extend(_mutation_exchange(step))
    return snapshots


def _pipeline(root: Path, management: Mapping[str, Any]) -> NativeContextPipeline:
    artifacts = ArtifactStore(root)
    return NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=200_000, max_output_tokens=2048,
        retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1", summary_provider=_Summary(),
        context_management=management,
        context_artifacts=ContextArtifactCatalog(root, "policy-session-1", artifacts),
        mutation_tools=("move_pose", "close_gripper", "open_gripper"),
    )


def _assembler(wire: str) -> NativeRequestAssembler:
    provider: Any = (
        AnthropicMessagesProvider(
            base_url="https://provider.invalid", api_key="test", client=object(),
        )
        if wire == "anthropic" else object()
    )
    return NativeRequestAssembler(
        model="m", reasoning_effort="medium", max_output_tokens=2048,
        stream=False, provider=provider,
    )


def _run(
    root: Path, management: Mapping[str, Any], *, completed: int = 18,
    second_round_from: int | None = None, wire: str = "openai",
) -> tuple[NativeContextPipeline, list[dict[str, Any]]]:
    pipeline = _pipeline(root, management)
    assembler = _assembler(wire)
    calls: list[dict[str, Any]] = []
    for step, snapshot in _episode(
        completed, second_round_from=second_round_from,
    ):
        projected = pipeline.project(
            messages=snapshot, tools=TOOLS, observation_step=step,
        )
        prepared = assembler.prepare(projected)
        calls.append({
            "step": step, "manifest": projected["manifest"],
            "canonical": canonical_bytes(prepared.canonical),
            "wire": canonical_bytes(prepared.wire),
            "messages": prepared.canonical["messages"],
        })
    return pipeline, calls


def _rewrites(calls: Sequence[Mapping[str, Any]]) -> list[int]:
    """The open ActionStep of every request that rewrote delivered history."""

    return [
        call["step"] for call in calls
        if call["manifest"]["request_relation"] == "projection_epoch"
    ]


def _inline_images(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        isinstance(block, Mapping) and block.get("type") == "image_url"
        for message in messages if isinstance(message.get("content"), list)
        for block in message["content"]
    )


# --- prefix stability ------------------------------------------------------


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_v4_rewrites_history_on_every_new_action_step(tmp_path: Path, wire: str) -> None:
    _pipeline_v4, calls = _run(tmp_path, V4, wire=wire)
    # Step 0 loses its media when step 3 arrives; from then on every fresh
    # observation ages one more step, so every ActionStep costs one rewrite.
    assert _rewrites(calls) == list(range(3, 19))


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_v5_stride_four_rewrites_once_per_four_action_steps(tmp_path: Path, wire: str) -> None:
    _pipeline_v5, calls = _run(tmp_path, _v5(4), wire=wire)
    assert _rewrites(calls) == [6, 10, 14, 18]
    for previous, call in zip(calls, calls[1:]):
        relation = call["manifest"]["request_relation"]
        canonical = call["manifest"]["canonical_prefix_relation"]
        wire_relation = call["manifest"]["provider_wire_prefix_relation"]
        if call["step"] in {6, 10, 14, 18} and call["step"] != previous["step"]:
            assert relation == "projection_epoch"
            assert call["manifest"]["projection_epoch"] == (
                previous["manifest"]["projection_epoch"] + 1
            )
            continue
        # Every other request strictly extends the previous one.
        assert relation == "append_only", call["step"]
        assert canonical["status"] == "exact_prefix"
        assert wire_relation["status"] == "exact_prefix"
        assert call["wire"] != previous["wire"]


def test_one_aging_event_carries_media_text_and_pressure_together(tmp_path: Path) -> None:
    _pipeline_v5, calls = _run(tmp_path, _v5(4))
    boundary = next(
        call for call in calls
        if call["manifest"]["request_relation"] == "projection_epoch"
    )
    manifest = boundary["manifest"]
    kinds = [item["kind"] for item in manifest["projection_transitions"]]
    assert kinds == ["lifecycle_media_externalization", "pressure_externalization"]
    assert {item["to_epoch"] for item in manifest["projection_transitions"]} == {
        manifest["projection_epoch"],
    }
    # Every frontier lands on its exact minimum tail at the same event.
    assert boundary["step"] == 6
    assert manifest["lifecycle_media_externalized_through_action_step"] == 3
    assert manifest["pressure_externalized_through_action_step"] == 2
    assert manifest["tail_aging"]["aged"] is True
    assert manifest["tail_aging"]["pending_media_action_steps"] == 0
    assert manifest["tail_aging"]["pending_read_only_action_steps"] == 0


def test_v5_stride_one_is_byte_identical_to_v4(tmp_path: Path) -> None:
    v5_only = {
        "context_management", "resolved_media_retention_policy",
        "resolved_tail_aging_stride_action_steps", "tail_aging",
    }
    for wire in ("openai", "anthropic"):
        _a, v4_calls = _run(
            tmp_path / f"v4-{wire}", V4, second_round_from=9, wire=wire,
        )
        _b, v5_calls = _run(
            tmp_path / f"v5-{wire}", _v5(1), second_round_from=9, wire=wire,
        )
        assert len(v4_calls) == len(v5_calls) == 38
        for old, new in zip(v4_calls, v5_calls):
            assert new["canonical"] == old["canonical"], old["step"]
            assert new["wire"] == old["wire"], old["step"]
            assert {
                key: value for key, value in new["manifest"].items()
                if key not in v5_only
            } == {
                key: value for key, value in old["manifest"].items()
                if key not in v5_only
            }


def test_v4_manifest_keeps_its_exact_evidence_fields(tmp_path: Path) -> None:
    _pipeline_v4, calls = _run(tmp_path, V4, completed=6)
    manifest = calls[-1]["manifest"]
    assert manifest["resolved_media_retention_policy"] == (
        "current_round_action_step_tail_v1"
    )
    assert "tail_aging" not in manifest
    assert "resolved_tail_aging_stride_action_steps" not in manifest


# --- limits -----------------------------------------------------------------


def test_inline_media_stays_within_minimum_tail_plus_stride_minus_one(tmp_path: Path) -> None:
    _pipeline_v5, calls = _run(tmp_path, _v5(4))
    completed_with_media = [
        len(call["manifest"]["protected_inline_media_action_steps"]) - 1
        for call in calls
    ]
    # min tail 2 + stride 4 - 1 = 5 completed ActionSteps, plus the open one.
    assert max(completed_with_media) == 5
    for call in calls:
        assert _inline_images(call["messages"]) <= 6 * len(CAMERAS)
        assert call["manifest"]["active_inline_images"] == _inline_images(
            call["messages"],
        )


def test_aged_steps_keep_recoverable_references(tmp_path: Path) -> None:
    pipeline, calls = _run(tmp_path, _v5(4))
    final = calls[-1]
    aged_through = final["manifest"]["lifecycle_media_externalized_through_action_step"]
    assert aged_through == 15
    messages = final["messages"]
    observations = [
        message for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
    ]
    catalog = pipeline.context_artifacts
    assert catalog is not None
    for step, message in enumerate(observations):
        blocks = message["content"]
        images = [block for block in blocks if block.get("type") == "image_url"]
        markers = [
            decode_recoverable_artifact_marker(block["text"])
            for block in blocks
            if block.get("type") == "text"
            and block["text"].startswith(f"<{RECOVERABLE_ARTIFACT_TAG}>")
        ]
        assert len(markers) == len(CAMERAS), step
        assert len(images) == (0 if step <= aged_through else len(CAMERAS)), step
        for camera, pointer in zip(CAMERAS, markers):
            _entry, recovered = catalog.resolve(pointer["ref"])
            assert recovered == f"{camera}-step-{step}".encode("utf-8")
    pressure_through = final["manifest"]["pressure_externalized_through_action_step"]
    assert pressure_through == 14
    read_results = [
        json.loads(message["content"]) for message in messages
        if message.get("role") == "tool" and message.get("name") == "observe"
    ]
    for step, result in enumerate(read_results):
        if step <= pressure_through:
            assert result["externalized"] is True
            _entry, recovered = catalog.resolve(result["recoverable_artifact"]["ref"])
            assert json.loads(recovered)["detail"].startswith(f"read-only body {step} ")
        else:
            assert "externalized" not in result
    for step, message in enumerate(observations):
        body = json.loads(message["content"][0]["text"])
        if step <= pressure_through:
            assert "camera_geometry" not in body["current_actual"]
            pointer = body["recoverable_observation"]
            _entry, recovered = catalog.resolve(pointer["ref"])
            assert "camera_geometry" in json.loads(recovered)["current_actual"]
        else:
            assert "camera_geometry" in body["current_actual"]


def test_stride_slack_never_counts_as_l4_pressure(tmp_path: Path) -> None:
    # The minimum-tail view carries three visual steps (9 images); the slack
    # carries up to six (18). A trigger between the two must not turn the
    # stride itself into a second rewrite inside each stride.
    management = _v5(
        4, trigger_inline_images=12, trigger_estimated_tokens=1_000_000,
    )
    _pipeline_v5, calls = _run(tmp_path, management)
    assert _rewrites(calls) == [6, 10, 14, 18]
    assert max(_inline_images(call["messages"]) for call in calls) > 12
    for call in calls:
        kinds = {item["kind"] for item in call["manifest"]["projection_transitions"]}
        assert "pressure_externalization" not in kinds
        if call["manifest"]["tail_aging"]["pending_media_action_steps"]:
            assert call["manifest"]["tail_aging"]["pressure_admission"] == (
                "deferred_to_tail_aging"
            )


def test_a_new_episode_releases_prior_episode_media_at_once(tmp_path: Path) -> None:
    # Round 2 starts one step after an aging event, so only the reset (not
    # the stride) is due: v4 releases prior-Episode media at reset and so does
    # v5, instead of carrying old-scene images into the new layout.
    _pipeline_v5, calls = _run(tmp_path, _v5(4), second_round_from=7)
    first_of_round_two = next(call for call in calls if call["step"] == 7)
    manifest = first_of_round_two["manifest"]
    assert manifest["request_relation"] == "projection_epoch"
    assert manifest["tail_aging"]["reasons"] == ["prior_episode_media_released"]
    assert manifest["lifecycle_media_externalized_through_action_step"] == 6
    assert manifest["protected_inline_media_action_steps"] == [7]
    # Within the new Episode the stride resumes: one rewrite per four steps.
    assert _rewrites(calls) == [6, 7, 11, 15]


def test_provider_anchored_episode_capsule_is_not_deferred_by_the_stride(
    tmp_path: Path,
) -> None:
    from ora.agent.native.tool_loop import _provider_estimated_input_tokens

    pipeline = _pipeline(tmp_path, _v5(4))
    snapshots = _episode(10, second_round_from=7)
    for step, snapshot in snapshots[:-1]:
        pipeline.project(messages=snapshot, tools=TOOLS, observation_step=step)
    step, snapshot = snapshots[-1]
    # A frozen measurement: the same projection admitting no transition.
    measured = pipeline._project_action_step_lossless_v4(
        messages=[copy.deepcopy(dict(message)) for message in snapshot],
        tools=TOOLS, observation_step=step,
        allow_transition=False,
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(measured["manifest"]),
        input_tokens=160_000,
        projection_epoch=measured["manifest"]["projection_epoch"],
        provider_wire_request_sha256="c" * 64,
    )
    projected = pipeline.project(
        messages=snapshot, tools=TOOLS, observation_step=step,
    )
    manifest = projected["manifest"]
    # Mid-stride: aging is pending, so L4 waits, but L5 is Provider-anchored
    # and closes the finished Episode exactly as v4 would.
    assert manifest["tail_aging"]["aged"] is False
    assert manifest["tail_aging"]["pressure_admission"] == "deferred_to_tail_aging"
    assert [item["kind"] for item in manifest["projection_transitions"]] == [
        "episode_capsule",
    ]
    assert manifest["episode_checkpointed_rounds"] == [1]


# --- evidence ---------------------------------------------------------------


def test_coverage_evidence_accepts_stride_lag_and_rejects_overdue_aging(tmp_path: Path) -> None:
    _pipeline_v5, calls = _run(tmp_path, _v5(4), second_round_from=9)
    for call in calls:
        manifest = call["manifest"]
        assert manifest["resolved_media_retention_policy"] == (
            "current_round_action_step_tail_stride_v1"
        )
        assert manifest["resolved_tail_aging_stride_action_steps"] == 4
        assert coverage_violations(manifest, manifest["action_step_inventory"]) == []
    lagging = next(
        call["manifest"] for call in calls
        if call["manifest"]["tail_aging"]["pending_media_action_steps"] == 3
    )
    overdue = copy.deepcopy(lagging)
    overdue["context_management"]["tail_aging_stride_action_steps"] = 3
    assert "lossless_context_lifecycle_coverage_invalid" in coverage_violations(
        overdue, overdue["action_step_inventory"],
    )
    prior = next(
        call["manifest"] for call in calls if call["step"] == 10
    )
    leaked = copy.deepcopy(prior)
    leaked["lifecycle_media_externalized_through_action_step"] = 7
    leaked["protected_inline_media_action_steps"] = [8, 9, 10]
    assert "lossless_context_lifecycle_coverage_invalid" in coverage_violations(
        leaked, leaked["action_step_inventory"],
    )


def test_lossless_v5_sessions_carry_an_explicit_round_identity() -> None:
    from ora.orchestrator.coordinator import _session_round_budget_fields

    for management in (V4, _v5(4)):
        assert _session_round_budget_fields(
            {"context_management": management},
            round_index=0, round_limit=1, multi_round=False,
        ) == {"session_round": 1, "session_rounds": 1}


# --- validation -------------------------------------------------------------


def test_v5_validates_the_stride_as_a_positive_integer() -> None:
    assert validate_context_management_v5(_v5(4))[
        "tail_aging_stride_action_steps"
    ] == 4
    assert validate_context_management(_v5(1)) == validate_context_management_v5(_v5(1))
    for invalid in (0, -1, True, 1.5, "4", None):
        with pytest.raises(ContextManagementConfigError, match="stride"):
            validate_context_management_v5(_v5(4) | {
                "tail_aging_stride_action_steps": invalid,
            })
    missing = _v5(4)
    del missing["tail_aging_stride_action_steps"]
    with pytest.raises(ContextManagementConfigError, match="missing"):
        validate_context_management_v5(missing)
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management_v5({**_v5(4), "schema": "unknown"})


def test_v4_rejects_the_stride_and_stays_frozen(tmp_path: Path) -> None:
    assert validate_context_management_v4(V4) == validate_context_management(V4)
    with pytest.raises(ContextManagementConfigError, match="extra"):
        validate_context_management_v4(V4 | {"tail_aging_stride_action_steps": 4})
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management_v4({**V4, "schema": CONTEXT_MANAGEMENT_V5_SCHEMA})
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management({**V4, "schema": "unknown"})
    with pytest.raises(PolicySessionError, match="extra"):
        _pipeline(tmp_path, V4 | {
            "tail_aging_stride_action_steps": 4,
        })


def test_the_pipeline_rejects_an_invalid_v5_stride(tmp_path: Path) -> None:
    with pytest.raises(PolicySessionError, match="stride"):
        _pipeline(tmp_path, _v5(0))
