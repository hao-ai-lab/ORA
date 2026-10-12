"""Layer 3: the composed grasp_execute c1 is the native prepared grasp, stage for stage.

Both sides start from the same published candidate on the same synthetic
control state and run against identical synthetic backends behind the same
motion v3 shim. The native side commits through ``begin_skill`` and runs
``execute_skill_sequence``; the composed side calls the same tool name on a
surface that binds ``grasp_execute/c1``, and runs ``execute_composed_tool``
on the runtime's ``start_composed_tool``. Every scripted fault -- a pose stage
that does not arrive, a gripper the backend could not execute, an episode
that ends mid-sequence -- must give the same child actions, the same stage
records and the same receipt, apart from the composed identity and paths.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
pytestmark = pytest.mark.usefixtures("ready_model_services")


from ora.agent.tools import grasp_feasibility, skill_retention
from ora.agent.tools.composed.library import LIBRARY_ROOT, load_tool
from ora.contracts import (
    COMPOSED_TOOL_ACTION_SCHEMA, MOTION_ACTION_V3_SCHEMA, SKILL_SEQUENCE_ACTION_SCHEMA, validate_runtime_action,
)
from ora.orchestrator.action_executor import execute_skill_sequence
from ora.orchestrator.composed_tool_job import execute_composed_tool
from test_manipulation_skill_driver import _observation_id, _publish_grasp_candidate, _runtime, _submitted
from test_skill_sequence_executor import _session

RATIONALE = "chosen target; opening and short lift are appropriate"
GRASP = load_tool(LIBRARY_ROOT / "grasp_execute" / "c1")


def _host(tmp_path, *, composed: bool):
    host = _runtime(tmp_path)
    capabilities = [*host.surface.capabilities, skill_retention.CAPABILITY, grasp_feasibility.CAPABILITY,
                    "motion_intent_router_v1"]
    bindings = {}
    if composed:
        capabilities.append("composed_tools_v1")
        bindings = {"grasp_execute": {
            "tool": "grasp_execute", "version": "c1", "arm_profile": "single_arm",
            "compiled_sha256": GRASP.compiled("single_arm")["compiled_sha256"],
        }}
    host.surface = replace(host.surface, capabilities=tuple(capabilities), composed_tools=bindings)
    _publish_grasp_candidate(host, observation_id=_observation_id(host))
    host.motion_probe_callback = lambda request: {
        "schema": "ora.motion-probe-result.v1",
        "results": [{"query_id": query["query_id"], "arm": query["arm"], "reachable": True, "variants": []}
                    for query in request["queries"]],
    }
    return host


def _backend(tmp_path, *, faults, max_actions=32):
    """A synthetic session behind a motion v3 shim, with scripted per-stage faults."""

    backend = _session(tmp_path, max_actions=max_actions)
    original = backend.execute

    def execute(payload):
        action = validate_runtime_action(payload)
        stage = action["action_id"].rsplit("-", 1)[-1].split(".")[0]
        if action["schema"] == MOTION_ACTION_V3_SCHEMA:
            ((arm, target),) = action["arguments"]["targets"].items()
            if action["action"] == "set_gripper":
                thin = {"action": "set_gripper", "arguments": {"arm": arm, "gripper": 1.0 if target == "open" else 0.0}}
            else:
                thin = {"action": "move_pose", "arguments": {"arm": arm, **target}}
            payload = {
                "schema": "ora-action.v3", "observation_id": action["observation_id"],
                "action_id": action["action_id"], "bound_observation_sha256": action["bound_observation_sha256"],
                **thin,
            }
        result = original(payload)
        fault = faults.get(stage)
        if fault == "not_reached":
            result["receipt"].pop("commanded_pose_xyz_xyzw", None)
        elif fault == "unavailable":
            result["receipt"]["execution_status"] = "planner_failed"
            result["receipt"]["failure_code"] = "planner_failed"
        return result

    backend.execute = execute
    return backend


def _observe(backend, root):
    observation, _images = backend.observe(root)
    return observation, root / observation["observation_id"]


def _native(tmp_path, faults, max_actions):
    host = _host(tmp_path / "native-host", composed=False)
    result = host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": RATIONALE})
    action = _submitted(host)
    assert action["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA
    backend = _backend(tmp_path / "native-backend", faults=faults, max_actions=max_actions)
    root = tmp_path / "native-observations"
    initial, initial_root = _observe(backend, root)
    children = []
    executed = backend.execute

    def recording(payload):
        children.append(validate_runtime_action(payload))
        return executed(payload)

    backend.execute = recording
    outer, _latest = execute_skill_sequence(
        backend, action, initial_observation=initial, initial_observation_root=initial_root,
        drive_stage=host.drive_skill_stage, build_receipt=host.build_skill_receipt,
        capture_observation=lambda: _observe(backend, root),
    )
    return result, outer, children


def _composed(tmp_path, faults, max_actions):
    host = _host(tmp_path / "composed-host", composed=True)
    result = host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": RATIONALE})
    action = _submitted(host)
    assert action["schema"] == COMPOSED_TOOL_ACTION_SCHEMA
    backend = _backend(tmp_path / "composed-backend", faults=faults, max_actions=max_actions)
    root = tmp_path / "composed-observations"
    box = {"latest": _observe(backend, root), "captured": False}
    initial, initial_root = box["latest"]
    children = []
    executed = backend.execute

    def recording(payload):
        children.append(validate_runtime_action(payload))
        return executed(payload)

    backend.execute = recording

    def capture():
        box["latest"] = _observe(backend, root)
        box["captured"] = True
        return box["latest"]

    execution = host.start_composed_tool(action, capture=capture, latest=lambda: box["latest"])
    outer, _latest = execute_composed_tool(
        backend, action, execution=execution, initial_observation=initial, initial_observation_root=initial_root,
        capture_observation=capture, latest_box=box,
    )
    return result, outer, children


def _children(children):
    return [(item["action"], item["arguments"], item["action_id"].split(":")[-1]) for item in children]


def _stages(outer):
    return [
        {key: item[key] for key in ("stage", "status", "reason", "measurements") if key in item}
        | {"action_id": (item.get("action_id") or "").split(":")[-1]}
        for item in outer["stage_receipts"]
    ]


def _receipt(outer):
    receipt = copy.deepcopy(outer["skill_receipt"])
    receipt.pop("composed_tool", None)
    # c1 adds the fused check.hold verdict beside the native evidence blocks (spec 4.6).
    if isinstance(receipt.get("verification"), dict):
        receipt["verification"].pop("hold", None)
    receipt["delivery"].pop("body_bytes")
    for item in receipt.get("observation_evidence", []):
        item.pop("observation_root")
    for item in receipt["stages"]:
        item["action_id"] = (item.get("action_id") or "").split(":")[-1]
    return receipt


SCENARIOS = {
    "nominal": ({}, 32),
    "pregrasp_not_reached": ({"pregrasp": "not_reached"}, 32),
    "approach_not_reached": ({"approach": "not_reached"}, 32),
    "lift_not_reached": ({"lift": "not_reached"}, 32),
    "open_unavailable": ({"open": "unavailable"}, 32),
    "close_unavailable": ({"close": "unavailable"}, 32),
    "terminal_after_approach": ({}, 3),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_composed_grasp_equals_the_native_prepared_grasp(tmp_path, scenario):
    faults, max_actions = SCENARIOS[scenario]
    native_result, native, native_children = _native(tmp_path, faults, max_actions)
    composed_result, composed, composed_children = _composed(tmp_path, faults, max_actions)

    assert _children(composed_children) == _children(native_children)
    for key in ("status", "stop_reason", "robot_decisions", "dispatched_stage_actions", "committed_sequence",
                "terminal_after_action", "candidate_id", "consumed"):
        assert composed[key] == native[key], key
    assert _stages(composed) == _stages(native)
    assert _receipt(composed) == _receipt(native)
    assert composed["skill_receipt"]["composed_tool"]["tool"] == "grasp_execute"

    for key in ("committed_sequence", "candidate_id", "precommit_motion_check", "claim_boundary"):
        assert composed_result[key] == native_result[key], key
    assert composed_result["cost"]["robot_decisions"] == native_result["cost"]["robot_decisions"] == 1


def test_the_nominal_composed_grasp_dispatches_five_stages_and_verifies(tmp_path):
    _result, composed, children = _composed(tmp_path, {}, 32)
    assert [item["action_id"].split(":")[-1] for item in children] == [
        "00-open", "01-pregrasp", "02-approach", "03-close", "04-lift",
    ]
    assert composed["status"] == "completed" and composed["stop_reason"] == "sequence_completed"
    verify = composed["stage_receipts"][-1]
    assert verify["stage"] == "verify" and verify["action_id"] is None
    receipt = composed["skill_receipt"]
    assert receipt["verification"]["lift_probe"]["requested_world_lift_m"] == pytest.approx(0.03)
    assert receipt["observation_evidence"][0]["role"] == "post_close_before_lift"
    assert receipt["precommit_motion_check"]["probed_stages"] == ["pregrasp", "approach"]
    hold = receipt["verification"]["hold"]
    assert hold["schema"] == "ora.hold-verdict.v1" and hold["hold"] in {"supported", "contradicted", "unknown"}
    assert [item["source"] for item in hold["sources"]][:2] == ["closure_extent", "lift_surface_motion"]
