"""native_minimal_v1 Context keeps every read result inline and offers no Artifact tools.

Reached only through tests/test_robodojo_compat.py (the robodojo-compat launcher)
before the RoboDojo builder relocation; it runs here against product code.
"""
from __future__ import annotations

import json

from ora.agent.context.native import NativeContextPipeline
from ora.agent.evidence.trajectory import ArtifactStore


def test_minimal_context_retains_old_read_results_without_unavailable_artifact_tools(tmp_path):
    pipeline = NativeContextPipeline(
        profile="native_minimal_v1", artifacts=ArtifactStore(tmp_path / "context"),
        context_window_tokens=262144, max_output_tokens=8192, summary_policy="disabled",
    )
    messages = [{"role": "system", "content": "Two public tools are available."}]
    for index in range(7):
        messages.extend([
            {"role": "user", "content": f"Observation {index}", "_observation": {"step": index}},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{index}", "type": "function",
                "function": {"name": "observation.dump_state", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": f"call-{index}",
             "content": json.dumps({"public_state": f"unique-measurement-{index}"})},
        ])
    projection = pipeline.project(messages=messages, tools=[], observation_step=6)
    encoded = json.dumps(projection["messages"])
    assert all(f"unique-measurement-{index}" in encoded for index in range(7))
    assert "artifact_ref" not in encoded and "reference_only" not in encoded
