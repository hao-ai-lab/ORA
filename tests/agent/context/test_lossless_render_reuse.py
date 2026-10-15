"""Lossless v4 renders reuse unchanged pass stages and copy only edited messages.

One projection may render L1-L4 up to four times (current frontiers, then the
L1/L2, L3 and L4 transitions). The golden digests in
``test_lossless_context_runtime_cost.py`` prove the output bytes are unchanged;
these tests bind the work: unchanged L1/L2 and L3 stages are not recomputed,
passes are copy-on-write, and canonical sizing of shared messages is exact.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context import historical_observation, lossless_render
from ora.agent.context import projection as projection_module
from ora.agent.context.request_size import CanonicalSizer
from ora.agent.evidence.trajectory import canonical_bytes
from tests.agent.context.test_lossless_context_runtime_cost import (
    TOOLS, _history, _observation, _pipeline,
)


def _count_passes(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"tool_results": 0, "media": 0, "observation": 0}
    tool_results = lossless_render._externalize_tool_results
    media = lossless_render._externalize_historical_media
    observation = historical_observation.externalize

    def count(name: str, function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return function(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(lossless_render, "_externalize_tool_results",
                        count("tool_results", tool_results))
    monkeypatch.setattr(lossless_render, "_externalize_historical_media",
                        count("media", media))
    monkeypatch.setattr(historical_observation, "externalize",
                        count("observation", observation))
    return counts


def test_transition_renders_resume_from_unchanged_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(tmp_path / "run")
    messages = _history(pipeline, steps=12, first_round=12, size=4_000,
                        project=lambda history, step: pipeline.project(
                            messages=history, tools=TOOLS, observation_step=step,
                        ))
    messages.append(_observation(12, 1, 4_000))
    counts = _count_passes(monkeypatch)
    projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=12)
    kinds = [item["kind"] for item in projected["manifest"]["projection_transitions"]]
    assert kinds == [
        "lifecycle_read_only_externalization", "lifecycle_media_externalization",
        "pressure_externalization",
    ]
    # Current frontiers plus the L1/L2, L3 and L4 transitions: four renders.
    assert counts["observation"] == 4
    # Without reuse: 4 x 3 tool-result passes and 4 media passes. L1/L2 run
    # only for the two distinct eager frontiers; the L4 render resumes from
    # the stored post-L3 state.
    assert counts["tool_results"] == 2 * 2 + 4
    assert counts["media"] == 3


def test_passes_copy_only_edited_messages() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "{}"}],
         "_observation": {"step": 0}, "_session_round": 1},
        {"role": "tool", "name": "observation.dump_state", "tool_call_id": "a",
         "content": "x" * 600, "_context_artifact_ref": {
             "ref": "artifact://s/" + "a" * 64, "sha256": "a" * 64, "policy_scope": "s",
             "media_type": "application/json", "bytes": 600,
         }},
        {"role": "user", "content": [{"type": "text", "text": "{}"}],
         "_observation": {"step": 1}, "_session_round": 1},
    ]
    before = copy.deepcopy(messages)
    output, changed, _refs = projection_module._externalize_tool_results(
        messages, through_action_steps=1, mutation_tools=(), selector="remaining",
        threshold_bytes=50_000, preview_bytes=16,
    )
    assert changed == [1]
    assert messages == before  # the input is never edited
    assert output[0] is messages[0] and output[2] is messages[2]
    assert output[1] is not messages[1]
    assert output[1]["_lossless_externalized"] is True


def test_message_list_sizing_is_exact_and_reuses_shared_messages() -> None:
    rng = random.Random(7)
    sizer = CanonicalSizer()

    def message(index: int) -> dict[str, Any]:
        return {
            "role": rng.choice(["user", "tool", "assistant"]),
            "content": [{"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + "A" * rng.randrange(1024, 4096),
            }}, {"type": "text", "text": "ü" * rng.randrange(5)}],
            "_source_event_ids": [f"agent-event-{index:06d}"],
            "nested": {"_source_event_ids": [f"nested-{index}"], "v": [1, None, 2.5]},
        }

    history = [message(index) for index in range(40)]
    for _ in range(5):
        edited = list(history)
        for index in rng.sample(range(len(edited)), 5):
            replacement = dict(edited[index])
            replacement["content"] = "replaced"
            edited[index] = replacement
        for value in (history, edited, [], [history[0]]):
            size, ids = sizer.measure(value)
            assert size == len(canonical_bytes(value))
            expected = {
                item for element in value for item in (
                    *element.get("_source_event_ids", []),
                    *element.get("nested", {}).get("_source_event_ids", []),
                )
            }
            assert ids == expected
        sizer.rotate()
