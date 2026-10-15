"""A mutation call that fails after submission leaves no action behind, and is logged.

Census item 10 / E6 (359a18b9): the manipulation tool runtime validated a result against
its schema outside the error path, so a failure there -- or any raise after
``_atomic_submit`` -- left ``submitted-action.json`` persisted (closing the
observation for the rest of the turn) and wrote no ``tool_error`` event.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.tools.robot import ToolError


def _events(runtime):
    return [json.loads(line) for line in runtime.event_path.read_text().splitlines()]


def test_a_result_schema_failure_after_submission_unsubmits_and_is_logged(tmp_path):
    from dataclasses import replace

    from tests.agent.motion.test_observed_clearance_runtime import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    released = runtime.surface
    runtime.surface = replace(released, result_schemas={
        **released.result_schemas,
        "set_gripper": {"type": "object", "properties": {"never": {"type": "string"}}, "required": ["never"]},
    })
    action = runtime.control_root / "submitted-action.json"
    with pytest.raises(Exception):
        runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    assert not action.exists()
    assert (_events(runtime)[-1]["type"], _events(runtime)[-1]["tool"]) == ("tool_error", "set_gripper")
    runtime.surface = released
    runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    assert action.exists()


def test_a_refused_second_mutation_keeps_the_pending_action(tmp_path):
    from tests.agent.motion.test_observed_clearance_runtime import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    action = runtime.control_root / "submitted-action.json"
    pending = action.read_bytes()
    with pytest.raises(ToolError):
        runtime.invoke("set_gripper", {"targets": {"right": "close"}})
    assert action.read_bytes() == pending
