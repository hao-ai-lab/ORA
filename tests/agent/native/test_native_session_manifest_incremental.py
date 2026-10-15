"""The Native Session manifest is byte-identical but no longer quadratic.

``native/agent-session.json`` embeds every call's projection manifest and was
re-encoded from scratch (pure-Python ``indent=2``) after every Provider
response. The incremental text must equal the reference serialization exactly;
the closed manifest always uses the reference serializer.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ora.agent.native.session import NativePlanner
from ora.agent.native.session_manifest import (
    IncrementalManifestText, dump_manifest_text,
)
from ora.agent.providers.native import FakeProvider
from ora.agent.evidence.trajectory import verify_agent_trace
from tests.agent.native.test_native_harness import (
    AGENT_RULES, PROMPT, assistant_tool, publish_native_boundary, usage,
)


def _random_value(rng: random.Random, depth: int = 0) -> object:
    choice = rng.randrange(9 if depth < 4 else 5)
    if choice == 0:
        return rng.randrange(-10**6, 10**6)
    if choice == 1:
        return rng.random() * 1e3
    if choice == 2:
        return rng.choice([True, False, None])
    if choice in (3, 4):
        return rng.choice([
            "", "plain", "line\nbreak", "tab\tquote\"", "ünïcødé 中文",
            "@@ora-splice-looks-like-a-marker@@", " ", "{\n}",
        ])
    if choice in (5, 6):
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {
        rng.choice(["b", "a", "ü", "calls", "z\n", ""]) + str(index):
        _random_value(rng, depth + 1)
        for index in range(rng.randrange(4))
    }


def _body(rng: random.Random, calls: list, usage_events: list) -> dict:
    return {
        "schema": "ora.native-policy-session.v1",
        "calls": calls,
        "usage_events": usage_events,
        "context_compactor_calls": [],
        "reported_totals": {"input_tokens": rng.randrange(10**6)},
        "exit_reason": None,
        "memory_projection": _random_value(rng),
    }


def test_incremental_text_equals_reference_serialization() -> None:
    rng = random.Random(20260927)
    renderer = IncrementalManifestText()
    calls: list = []
    usage_events: list = []
    for step in range(60):
        if rng.random() < 0.8:
            calls.append({"turn_id": f"model-call-{step:06d}", "x": _random_value(rng)})
        if rng.random() < 0.6:
            usage_events.append({"sequence": len(usage_events), "last": _random_value(rng)})
        if calls and rng.random() < 0.1:
            # A replaced record (new object) is re-serialized, never reused.
            calls[rng.randrange(len(calls))] = {"replaced": _random_value(rng)}
        if calls and rng.random() < 0.05:
            del calls[rng.randrange(len(calls)):]
        body = _body(rng, calls, usage_events)
        assert renderer.render(body) == dump_manifest_text(body)


def test_empty_and_absent_lists_match_reference() -> None:
    renderer = IncrementalManifestText()
    for body in (
        {"calls": [], "usage_events": []},
        {"calls": [{}], "usage_events": [[]]},
        {"other": 1},
        {"calls": "not-a-list", "usage_events": None},
    ):
        assert renderer.render(body) == dump_manifest_text(body)


def test_session_snapshots_and_closed_manifest_are_reference_bytes(
    tmp_path: Path,
) -> None:
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    provider = FakeProvider([
        {"message": {"role": "assistant", "content": "Thinking."}, "usage": usage(80)},
        {"message": assistant_tool("read", "robot.read_manual", {"section": "safety"}),
         "usage": usage(90)},
        {"message": assistant_tool("mutate", "robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.5,
        }), "usage": usage(120)},
    ])
    planner = NativePlanner(
        tmp_path, "native-manifest", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer,
        image_path=workspace / pointer["main_image"],
    )
    assert outcome["submitted"] is True
    planner.durability.barrier()
    snapshot = (tmp_path / "native/agent-session.json").read_text(encoding="utf-8")
    snapshot_value = json.loads(snapshot)
    assert snapshot_value["exit_reason"] is None
    # The session-start snapshot is written; the three post-response
    # snapshots fall inside the minimum interval and are skipped (the event
    # log keeps every call).
    assert snapshot_value["calls"] == []
    assert planner.manifest_writer.skipped_snapshots == 3
    assert snapshot == dump_manifest_text(snapshot_value)
    planner.close()
    closed = (tmp_path / "native/agent-session.json").read_text(encoding="utf-8")
    closed_value = json.loads(closed)
    assert closed_value["exit_reason"] == "closed"
    assert len(closed_value["calls"]) == 3
    assert closed == dump_manifest_text(closed_value)
    assert planner.durability.pending == 0
    assert not list((tmp_path / "native").glob(".agent-session.json.*.tmp"))
    verify_agent_trace(tmp_path)


class _Durability:
    def __init__(self) -> None:
        self.snapshots: list[str] = []

    def replace_text(self, path: Path, text: str) -> None:
        self.snapshots.append(text)


def test_snapshots_respect_the_minimum_interval(tmp_path: Path) -> None:
    from ora.agent.native.session_manifest import SessionManifestWriter

    now = [0.0]
    durability = _Durability()
    writer = SessionManifestWriter(
        tmp_path / "agent-session.json", durability, min_interval_s=120.0,
        clock=lambda: now[0],
    )
    calls: list = []
    for second in (0.0, 30.0, 119.0, 120.0, 200.0, 241.0):
        now[0] = second
        calls.append({"at": second})
        writer.write({"calls": calls, "usage_events": []}, final=False)
    assert [json.loads(text)["calls"][-1]["at"] for text in durability.snapshots] == [
        0.0, 120.0, 241.0,
    ]
    assert writer.skipped_snapshots == 3
