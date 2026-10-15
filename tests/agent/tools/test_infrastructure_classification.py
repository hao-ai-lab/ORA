"""One table says which tool errors are infrastructure; envelopes and the abort gate both read it.

EXP11B census E1/A1: the abort evidence gate kept its own code list while
infrastructure faults raised as plain ``invalid_arguments`` (a missing
grip-site pose, an observation manifest or camera artifact drifting or escaping
its root, invalid calibration matrices) never reached it, and the policy could
not see which refusals counted. Every envelope now reports
``infrastructure: true|false`` from ``agent/error_classes.py``, the gate counts
that table, those faults carry codes from it, and ``finish_episode``'s class
list is generated from the gate's own tables. The abort rule is unchanged.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.abort_evidence_schema import (
    TOOL_ERROR_CODES, counted_classes, infrastructure_classes_text,
)
from ora.agent.error_classes import INFRASTRUCTURE_CODES
from ora.agent.tools.abort_evidence import round_signals
from ora.agent.tools.tool_errors import ARGUMENT_REFUSAL_CODES, ToolError
from tests.orchestrator.test_runtime_v1 import publish_synthetic


def _event(code: str, tool: str = "geometry_query") -> str:
    return json.dumps({"type": "tool_error", "tool": tool, "error": ToolError("x", code=code).as_dict()})


def test_the_gate_and_the_envelope_read_the_same_table():
    assert TOOL_ERROR_CODES is INFRASTRUCTURE_CODES
    for code in sorted(INFRASTRUCTURE_CODES):
        assert ToolError("x", code=code).as_dict()["infrastructure"] is True
        assert round_signals([_event(code)]) == [{"source": "tool_error", "tool": "geometry_query", "code": code}]


@pytest.mark.parametrize("code", sorted(ARGUMENT_REFUSAL_CODES | {"stale_observation", "target_ref_unavailable"}))
def test_an_argument_or_state_refusal_is_not_infrastructure_and_never_counts(code):
    assert ToolError("x", code=code).as_dict()["infrastructure"] is False
    assert round_signals([_event(code)]) == []


def test_a_drifted_observation_manifest_is_infrastructure_and_counts(tmp_path):
    _observation, pointer, _workspace, control, tools, _store = publish_synthetic(tmp_path)
    source = json.loads((control / "current-observation.json").read_text())["source_root"]
    manifest = __import__("pathlib").Path(source) / "observation.json"
    manifest.chmod(0o644)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ToolError) as caught:
        tools.invoke("geometry.query", {"camera": "head_camera", "points": [[1, 1]]})
    error = caught.value.as_dict()
    assert (error["code"], error["infrastructure"]) == ("artifact_hash_mismatch", True)
    assert round_signals(tools.event_path.read_text().splitlines())[-1]["code"] == "artifact_hash_mismatch"


def test_a_missing_grip_site_pose_is_infrastructure(tmp_path):
    from tests.agent.motion.test_observed_clearance_runtime import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    pointer, source, observation = runtime._current()
    stripped = {**observation, "proprioception": {**observation["proprioception"], "end_effectors": {}}}
    runtime._current = lambda: (pointer, source, stripped)
    with pytest.raises(ToolError) as caught:
        runtime._current_grip_state()
    assert caught.value.as_dict()["code"] == "pose_state_unavailable"
    assert caught.value.as_dict()["infrastructure"] is True


def test_finish_episode_publishes_the_class_list_generated_from_the_gate():
    surface = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    description = next(item for item in surface.definitions if item["name"] == "finish_episode")["description"]
    assert description.endswith(infrastructure_classes_text())
    for codes in counted_classes().values():
        for code in codes:
            assert code in description, code
