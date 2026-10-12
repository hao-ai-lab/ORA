"""D3b in the lossless Context: the current round's slot is never aged.

Drives the synthetic stride episode of ``test_lossless_context_stride.py``
through the real pipeline and request assembler, with one
``round_static_facts`` slot appended after each round's first observation as
``native_slots_v4`` does. The slot must stay byte-exact through every L1-L4
aging batch of its round without adding a history rewrite, age to a
recoverable pointer when its round's media are released at reset, and be
carried out of a span an emergency checkpoint (L6) absorbs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context import round_static_facts as slot
from ora.agent.context.projection import _tiered_semantic_projection
from ora.agent.memory.artifacts import ContextArtifactCatalog
from ora.agent.native.constants import TOOL_IMAGE_CALL_LABELS
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from test_lossless_context_stride import (
    CHECKPOINT, TOOLS, _assembler, _episode, _pipeline, _rewrites, _run, _v5,
)


def _slot(step: int, session_round: int) -> dict[str, Any]:
    message = slot.slot_message(round_index=session_round, actual_sequence=step, fields={
        "camera_geometry": {"head": {"intrinsic_cv": [[600.0, 0.0, 320.0]] * 24}},
        "public_robot_facts": {"controller_feedback": {"position_tolerance_m": 0.01}},
    })
    message["_source_event_ids"] = [f"agent-event-{step * 10:06d}"]
    return message


def _with_slots(snapshots):
    """Insert each round's slot right after that round's first observation."""
    result = []
    for step, history in snapshots:
        updated, seen = [], set()
        for message in history:
            updated.append(message)
            session_round = message.get("_session_round")
            if isinstance(message.get("_observation"), dict) and session_round not in seen:
                seen.add(session_round)
                updated.append(_slot(message["_observation"]["step"], session_round))
        result.append((step, updated))
    return result


def _run_with_slots(root: Path, management, *, completed=18, second_round_from=None,
                    wire="openai"):
    pipeline = _pipeline(root, management)
    assembler = _assembler(wire)
    calls = []
    for step, snapshot in _with_slots(_episode(completed, second_round_from=second_round_from)):
        projected = pipeline.project(messages=snapshot, tools=TOOLS, observation_step=step)
        prepared = assembler.prepare(projected)
        calls.append({"step": step, "manifest": projected["manifest"],
                      "messages": prepared.canonical["messages"]})
    return pipeline, calls


def _slot_texts(messages):
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
            text = content[0].get("text", "")
            if text.startswith('{"round_static_facts"') or '"round_static_facts_externalized"' in text:
                texts.append(text)
    return texts


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_the_current_round_slot_survives_every_aging_batch(tmp_path: Path, wire: str) -> None:
    _, without = _run(tmp_path / "without", _v5(4), wire=wire)
    _, calls = _run_with_slots(tmp_path / "with", _v5(4), wire=wire)
    # The slot adds no history rewrite: aging happens at the same steps.
    assert _rewrites(calls) == _rewrites(without) == [6, 10, 14, 18]
    expected = slot.slot_message(round_index=1, actual_sequence=0, fields=json.loads(
        _slot(0, 1)["content"][0]["text"])["round_static_facts"]["current_actual"])
    for call in calls:
        assert _slot_texts(call["messages"]) == [expected["content"][0]["text"]]
    # L4 aged the observation bodies around it.
    assert any('"observation_externalized":true' in block.get("text", "")
               for call in calls for message in call["messages"]
               if isinstance(message.get("content"), list)
               for block in message["content"] if isinstance(block, dict))
    for previous, call in zip(calls, calls[1:]):
        if call["manifest"]["request_relation"] == "append_only":
            assert call["manifest"]["canonical_prefix_relation"]["status"] == "exact_prefix"


def test_reset_ages_the_previous_round_slot_with_its_media(tmp_path: Path) -> None:
    _, without = _run(tmp_path / "without", _v5(4), second_round_from=10)
    pipeline, calls = _run_with_slots(tmp_path / "with", _v5(4), second_round_from=10)
    assert _rewrites(calls) == _rewrites(without)
    reset = next(call for call in calls if call["step"] == 10)
    assert reset["manifest"]["request_relation"] == "projection_epoch"
    texts = _slot_texts(reset["messages"])
    marker = json.loads(texts[0])
    assert marker["round_static_facts"] == {"round": 1}
    assert marker["round_static_facts_externalized"] is True
    assert json.loads(texts[1])["round_static_facts"]["round"] == 2
    # The aged slot resolves to its exact original bytes.
    ref = marker["recoverable_round_static_facts"]["ref"]
    _entry, data = pipeline.context_artifacts.resolve(ref)
    assert data.decode() == _slot(0, 1)["content"][0]["text"]
    lifecycle = next(item for item in reset["manifest"]["passes"]
                     if item["name"] == "l3_lifecycle_media_reference")
    assert lifecycle["changed_messages"]
    # Before the reset round 1's slot is inline; after it, only round 2's is.
    before = next(call for call in calls if call["step"] == 9)
    assert json.loads(_slot_texts(before["messages"])[0])["round_static_facts"]["round"] == 1
    for call in calls:
        if call["step"] > 10:
            assert [json.loads(text).get("round_static_facts_externalized", False)
                    for text in _slot_texts(call["messages"])] == [True, False]


def test_an_emergency_checkpoint_carries_the_current_round_slot(tmp_path: Path) -> None:
    snapshots = _with_slots(_episode(8, second_round_from=4))
    _, history = snapshots[-1]
    catalog = ContextArtifactCatalog(tmp_path, "policy-session-1", ArtifactStore(tmp_path))
    transcript = catalog.put_bytes(canonical_bytes([]), kind="context_transcript",
                                   suffix=".json", media_type="application/json",
                                   round_index=None, action_step=None)
    projected = _tiered_semantic_projection(
        history, episode_checkpointed_action_steps=0, episode_capsules=(),
        emergency_summarized_action_steps=6, emergency_checkpoint=CHECKPOINT,
        emergency_transcript_ref=transcript,
    )
    rounds = [message["_round_static_facts"]["round"] for message in projected
              if slot.is_slot(message)]
    # Round 1's slot left with its summarized round; round 2's first step was
    # absorbed, and its slot is carried just ahead of the exact tail.
    assert rounds == [2]
    carried = next(index for index, message in enumerate(projected) if slot.is_slot(message))
    assert projected[carried - 1].get("_emergency_context_checkpoint") is True
    assert isinstance(projected[carried + 1].get("_observation"), dict)
    unchanged = _tiered_semantic_projection(
        history, episode_checkpointed_action_steps=0, episode_capsules=(),
        emergency_summarized_action_steps=2, emergency_checkpoint=CHECKPOINT,
        emergency_transcript_ref=transcript,
    )
    assert [message["_round_static_facts"]["round"] for message in unchanged
            if slot.is_slot(message)] == [2]


# -- One synthetic episode through the Supervisor --------------------------------


def _tool(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)},
    }]}


def _episode_run(root: Path, composition: str):
    from native_synthetic_spec import synthetic_spec
    from ora.agent.providers.native import FakeProvider
    from ora.orchestrator.coordinator import run_episode
    from ora.orchestrator.evaluation import evaluate
    from ora.orchestrator.spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["policy"]["context_retention"] = "action_step_lossless_v4"
    spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
    spec["policy"]["context_management"] = _v5(4)
    plan = resolve_run_spec(spec, root=Path(__file__).resolve().parents[1])
    # RunSpec registration of native_slots_v4 is integration 0058 unit A0's
    # (orchestrator/spec_composition.py); the session and Supervisor run it now.
    plan["policy"]["context_composition"] = composition
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = FakeProvider(responses=[
        _tool("c0", "close_gripper", {}), _tool("o0", "open_gripper", {}),
        _tool("c1", "close_gripper", {}),
        _tool("f0", "finish_episode", {"outcome": "failure", "reason": "done",
                                       "attempts": ["three steps"]}),
    ])
    result = run_episode(run_root=root, plan=plan, run_id=f"d3b-{composition}",
                         provider_override=provider)
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    requests = [json.loads((root / event["payload"]["canonical_request_ref"]["path"]).read_text())
                for event in events if event["type"] == "provider_request"]
    return result, events, requests, evaluate(root)


def test_a_synthetic_episode_carries_one_slot_and_stays_append_only(tmp_path: Path) -> None:
    result, events, requests, evaluation = _episode_run(tmp_path / "v4", slot.COMPOSITION)
    assert result["state"] == "completed" and evaluation["status"] == "valid"
    started = next(event for event in events if event["type"] == "session_started")
    assert started["payload"]["context_capabilities"] == [slot.CAPABILITY, TOOL_IMAGE_CALL_LABELS]
    for request in requests:
        slots = [index for index, message in enumerate(request["messages"])
                 if _slot_texts([message])]
        assert len(slots) == 1
        # The slot follows the round's first observation.
        assert "INITIAL ACTUAL" in json.dumps(request["messages"][slots[0] - 1])
    last = requests[-1]["messages"]
    observation = json.loads(next(
        message["content"][0]["text"] for message in reversed(last)
        if message["role"] == "user" and '"current_actual"' in message["content"][0]["text"]))
    fields = json.loads(_slot_texts(last)[0])["round_static_facts"]["current_actual"]
    assert observation["current_actual"]["static_facts"]["round_slot"] == 1
    assert slot.restore(observation["current_actual"], fields)["actual_sequence"] == 3
    projections = [event["payload"] for event in events if event["type"] == "context_projected"]
    assert [item["request_relation"] for item in projections[1:]] == ["append_only"] * 3
    assert any(record["kind"] == "round_static_facts" for record in projections[-1][
        "context_sources"])

    legacy, legacy_events, legacy_requests, legacy_evaluation = _episode_run(
        tmp_path / "v3", "native_slots_v3")
    assert legacy_evaluation["status"] == "valid"
    assert not any(_slot_texts(request["messages"]) for request in legacy_requests)
    assert "context_capabilities" not in next(
        event for event in legacy_events if event["type"] == "session_started")["payload"]


def test_runspec_policy_rules_for_native_slots_v4() -> None:
    from ora.orchestrator.spec_native_policy import _resolve_native_observation
    from ora.orchestrator.spec_values import RunSpecError

    policy = {"context_composition": slot.COMPOSITION, "context_profile": "native_causal_context_v1"}
    with pytest.raises(RunSpecError, match="native_slots_v4 requires context_retention"):
        _resolve_native_observation(dict(policy), "append_only_epoch_v1")
    with pytest.raises(RunSpecError, match="native_slots_v3 or native_slots_v4"):
        _resolve_native_observation({**policy, "context_composition": "native_slots_v5"}, None)


def test_each_supervisor_round_gets_its_slot_and_the_previous_one_ages(tmp_path: Path) -> None:
    from ora.orchestrator.coordinator import run_episode
    from ora.orchestrator.evaluation import evaluate
    from ora.orchestrator.spec import resolve_run_spec
    from test_multi_round_session import ROOT, _RoundProvider, _spec

    spec = _spec()
    spec["session"] = {"cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
                       "round_seeds": [0, 1, 2]}
    spec["policy"]["context_retention"] = "action_step_lossless_v4"
    spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
    spec["policy"]["context_management"] = _v5(4)
    plan = resolve_run_spec(spec, root=ROOT)
    # Multi-round registration of native_slots_v4 (orchestrator/spec_runtime.py) is A0's.
    plan["policy"]["context_composition"] = slot.COMPOSITION
    root = tmp_path / "run"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = _RoundProvider()
    result = run_episode(run_root=root, plan=plan, run_id="d3b-rounds", provider_override=provider)
    assert result["state"] == "completed" and result["round_count"] == 3
    assert evaluate(root)["status"] == "valid"
    for index, request in enumerate(provider.requests):
        texts = _slot_texts(request["messages"])
        assert [json.loads(text).get("round_static_facts_externalized", False)
                for text in texts] == [True] * index + [False]
        assert json.loads(texts[-1])["round_static_facts"]["round"] == index + 1
