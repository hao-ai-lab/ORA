"""Every tool-returned image message names its tool and call (census A4).

Under ``native_slots_v3`` an image a tool returns is labelled only by its kind,
for example ``READ-ONLY tool feedback: 1 image(s).``, so the images of two
parallel calls in one turn cannot be told apart. The other compositions always
named the call. ``native_slots_v4`` declares ``tool_image_call_labels_v1``: the
label keeps its kind and adds ``returned by {name} ({call_id})``.
``native_slots_v3`` keeps its recorded bytes.

One synthetic episode per composition through the Supervisor: a single turn
calls ``mark_point``, ``preview_move`` and ``observe`` twice each, all of
which return images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context import round_static_facts
from ora.agent.native.constants import (
    COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_SLOTS_V3, NATIVE_SLOTS_V4, TOOL_IMAGE_CALL_LABELS,
)
from tests.agent.context.test_round_static_facts_context import _v5

PARALLEL = (
    ("m0", "mark_point", {"point_id": "P0", "view": "wrist", "x": 2, "y": 1}),
    ("m1", "mark_point", {"point_id": "P1", "view": "wrist", "x": 1, "y": 1}),
    ("p0", "preview_move", {"delta_mm": [1.0, 0.0, 0.0]}),
    ("p1", "preview_move", {"delta_mm": [0.0, 1.0, 0.0]}),
    ("o0", "observe", {}),
    ("o1", "observe", {}),
)
KINDS = {"mark_point": ("READ-ONLY tool feedback", 1), "preview_move": ("CANDIDATE preview", 1),
         "observe": ("CURRENT ACTUAL tool observation", 2)}


def _turn(*calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)},
    } for call_id, name, arguments in calls]}


def _episode(root: Path, composition: str) -> tuple[list[dict], list[str]]:
    from tests.support.native_synthetic_spec import synthetic_spec
    from ora.agent.providers.native import FakeProvider
    from ora.orchestrator.coordinator import run_episode
    from ora.orchestrator.spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["policy"]["context_retention"] = "action_step_lossless_v4"
    spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
    spec["policy"]["context_management"] = _v5(4)
    plan = resolve_run_spec(spec, root=Path(__file__).resolve().parents[3])
    # RunSpec registration of native_slots_v4 is integration 0058 unit A0's.
    plan["policy"]["context_composition"] = composition
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = FakeProvider(responses=[
        _turn(*PARALLEL),
        _turn(("f0", "finish_episode", {"outcome": "failure", "reason": "done",
                                        "attempts": ["parallel image tools"]})),
    ])
    result = run_episode(run_root=root, plan=plan, run_id=f"a4-{composition}",
                         provider_override=provider)
    assert result["state"] == "completed"
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    request = json.loads((root / [event for event in events if event["type"] == "provider_request"][1][
        "payload"]["canonical_request_ref"]["path"]).read_text())
    labels = []
    for message in request["messages"]:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        if any(isinstance(block, dict) and block.get("type") == "image_url" for block in content):
            text = content[0].get("text", "")
            if " image(s)" in text:
                labels.append(text)
    return events, labels


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, tuple[list[dict], list[str]]]:
    root = tmp_path_factory.mktemp("a4")
    return {composition: _episode(root / composition, composition)
            for composition in (NATIVE_SLOTS_V3, NATIVE_SLOTS_V4)}


def test_native_slots_v4_names_the_tool_and_call_of_every_image(runs) -> None:
    events, labels = runs[NATIVE_SLOTS_V4]
    assert labels == [f"{KINDS[name][0]}: {KINDS[name][1]} image(s) returned by {name} ({call_id})."
                      for call_id, name, _ in PARALLEL]
    started = next(event for event in events if event["type"] == "session_started")
    assert TOOL_IMAGE_CALL_LABELS in started["payload"]["context_capabilities"]


def test_native_slots_v3_keeps_its_recorded_labels(runs) -> None:
    events, labels = runs[NATIVE_SLOTS_V3]
    assert labels == [f"{KINDS[name][0]}: {KINDS[name][1]} image(s)." for _, name, _ in PARALLEL]
    started = next(event for event in events if event["type"] == "session_started")
    assert "context_capabilities" not in started["payload"]


def test_native_slots_v4_declares_both_context_capabilities() -> None:
    assert COMPOSITION_CONTEXT_CAPABILITIES == {
        NATIVE_SLOTS_V4: (round_static_facts.CAPABILITY, TOOL_IMAGE_CALL_LABELS),
    }
