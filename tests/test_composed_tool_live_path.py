"""Composed tools on the live path: surface binding, action contract, dispatch and delivery.

The equivalence of the composed grasp with the native one is in
``test_composed_grasp_equivalence``; this module covers the joins around it.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.profile_catalog import COMPOSED_SUCCESSORS
from ora.agent.composition.tool_surface import ToolSurfaceError, bind_composed_tools
from ora.agent.tools import skill_driver, skill_ledger
from ora.agent.tools.composed.library import LIBRARY_ROOT, load_tool
from ora.agent.tools.robot import ToolError
from ora.contracts import (
    COMPOSED_TOOL_ACTION_SCHEMA, MOTION_ACTION_V3_SCHEMA, validate_composed_tool_action, validate_runtime_action,
)
from ora.contracts.action import ActionValidationError
from ora.orchestrator.composed_tool_job import run_composed_tool
from test_composed_grasp_equivalence import RATIONALE, _composed, _host
from test_manipulation_skill_driver import _submitted

GRASP = load_tool(LIBRARY_ROOT / "grasp_execute" / "c1")


# ---- surface binding ------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", sorted(COMPOSED_SUCCESSORS))
def test_the_composed_successors_serve_grasp_execute_from_its_compiled_document(profile):
    surface = resolve_tool_surface(profile)
    source = resolve_tool_surface(COMPOSED_SUCCESSORS[profile])
    compiled = GRASP.compiled("dual_arm")
    assert dict(surface.composed_tools) == {"grasp_execute": {
        "tool": "grasp_execute", "version": "c1", "arm_profile": "dual_arm",
        "compiled_sha256": compiled["compiled_sha256"],
    }}
    definition = next(item for item in surface.definitions if item["name"] == "grasp_execute")
    assert definition["description"].startswith(compiled["definition"]["description"])
    assert set(definition["inputSchema"]["properties"]) == {"candidate_id", "rationale"}
    assert surface.result_schemas["grasp_execute"] == compiled["result_schema"]
    assert "grasp_execute" in surface.mutation_tools
    assert {item["slot"] for item in surface.components} >= {"composed:grasp_execute"}
    assert surface.snapshot()["composed_tools"] == dict(surface.composed_tools)
    # Every other tool is the source surface's, byte for byte.
    others = {item["name"]: item for item in surface.definitions if item["name"] != "grasp_execute"}
    assert others == {item["name"]: item for item in source.definitions if item["name"] != "grasp_execute"}
    assert source.composed_tools == {} and "composed_tools" not in source.snapshot()
    assert surface.composition_sha256 != source.composition_sha256


def _binding_inputs(**manifest):
    definitions = [{"name": "grasp_execute", "description": "native", "inputSchema": {"type": "object"}},
                   {"name": "move_pose", "description": "native", "inputSchema": {"type": "object"}}]
    results = {"grasp_execute": {"type": "object"}, "move_pose": {"type": "object"}}
    return manifest, definitions, results


def test_binding_replaces_only_the_bound_tool():
    manifest, definitions, results = _binding_inputs(composed_tools={"grasp_execute": "grasp_execute/c1"})
    bindings, descriptors = bind_composed_tools(manifest, definitions, results, ["composed_tools_v1"])
    compiled = GRASP.compiled("single_arm")
    assert bindings["grasp_execute"]["arm_profile"] == "single_arm"
    assert definitions[0] == compiled["definition"] and definitions[1]["description"] == "native"
    assert results["grasp_execute"] == compiled["result_schema"]
    assert [item["ref"].rsplit("/", 1)[-1] for item in descriptors] == ["tool.star", "single_arm.json"]
    assert bind_composed_tools({}, definitions, results, []) == ({}, [])


@pytest.mark.parametrize(
    ("composed", "capabilities", "message"),
    [
        ({"grasp_execute": "grasp_execute/c1"}, [], "composed_tools_v1"),
        ({"place_execute": "grasp_execute/c1"}, ["composed_tools_v1"], "does not declare"),
        ({"move_pose": "grasp_execute/c1"}, ["composed_tools_v1"], "same name"),
        ({"grasp_execute": "grasp_execute"}, ["composed_tools_v1"], "<tool>/<version>"),
        ({"grasp_execute": "grasp_execute/c9"}, ["composed_tools_v1"], "composed_tools"),
        ({}, ["composed_tools_v1"], "must map"),
    ],
)
def test_binding_refusals_name_the_problem(composed, capabilities, message):
    manifest, definitions, results = _binding_inputs(composed_tools=composed)
    with pytest.raises(ToolSurfaceError, match=message):
        bind_composed_tools(manifest, definitions, results, capabilities)


# ---- the action contract --------------------------------------------------------------------------


def _action(**arguments):
    return {
        "schema": COMPOSED_TOOL_ACTION_SCHEMA, "observation_id": "obs-1", "action_id": "action-7",
        "action": "execute_composed_tool",
        "arguments": {
            "tool": "grasp_execute", "version": "c1", "arm_profile": "dual_arm",
            "compiled_sha256": "sha256:" + "a" * 64, "rationale": " why ",
            "committed_sequence": ["open", "close", "verify"], **arguments,
        },
        "bound_observation_sha256": "b" * 64,
    }


def test_the_composed_action_contract_normalizes_and_routes():
    normalized = validate_runtime_action(_action())
    assert normalized == validate_composed_tool_action(_action())
    assert normalized["arguments"]["rationale"] == "why"


@pytest.mark.parametrize(
    "change",
    [
        {"tool": "Grasp"}, {"version": "v1"}, {"arm_profile": "tri_arm"}, {"compiled_sha256": "abc"},
        {"rationale": ""}, {"rationale": "x" * 201}, {"committed_sequence": []},
        {"committed_sequence": ["open", "open"]}, {"committed_sequence": ["s"] * 7}, {"extra": 1},
    ],
)
def test_malformed_composed_actions_are_refused(change):
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action(**change))


# ---- the tool call --------------------------------------------------------------------------------


def test_a_native_sequence_in_flight_refuses_a_composed_call(tmp_path):
    host = _host(tmp_path, composed=True)
    control = json.loads((host.control_root / "current-observation.json").read_text())
    state = skill_ledger.begin(
        skill="place_execute", candidate_id="P0", rationale="placing", observation_id=control["observation_id"],
        observation_sha256=control["observation_sha256"], stop_before_commit=False, prepared_grasp=False,
    )
    skill_driver.store_ledger(host, state)
    with pytest.raises(ToolError) as raised:
        host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": RATIONALE})
    assert raised.value.code == "skill_sequence_active"
    assert not (host.control_root / "submitted-action.json").exists()


def test_a_stale_candidate_is_refused_by_the_composed_plan(tmp_path):
    host = _host(tmp_path, composed=True)
    path = host.artifact_root / "grasp-queries" / "grasp-query-fixture" / "manifest.json"
    body = json.loads(path.read_text())
    body["identity"]["observation_id"] = "an-earlier-observation"
    path.write_text(json.dumps(body))
    with pytest.raises(ToolError) as raised:
        host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": RATIONALE})
    assert raised.value.code == "stale_candidate_observation"
    assert not (host.control_root / "submitted-action.json").exists()


def test_the_composed_call_persists_its_commitment_and_one_action(tmp_path):
    host = _host(tmp_path, composed=True)
    result = host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": RATIONALE})
    action = _submitted(host)
    assert action["arguments"]["committed_sequence"] == ["open", "pregrasp", "approach", "close", "lift", "verify"]
    stored = json.loads((host.control_root / "composed-tool-commitment.json").read_text())
    assert stored["call_id"] == action["action_id"] and stored["candidate_id"] == "grasp-a1"
    assert set(stored["host_state"]["grasp_target"]) >= {"poses", "localization", "semantics", "precommit_motion_check"}
    assert result["precommit_motion_check"] == stored["host_state"]["grasp_target"]["precommit_motion_check"]
    drifted = copy.deepcopy(action)
    drifted["arguments"]["compiled_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="drifted"):
        host.start_composed_tool(drifted, capture=lambda: None, latest=lambda: None)


# ---- the coordinator join and delivery ------------------------------------------------------------


def test_the_job_refuses_a_runtime_without_the_composed_protocol(tmp_path):
    with pytest.raises(ValueError, match="start_composed_tool"):
        run_composed_tool(SimpleNamespace(), _action(), tools=SimpleNamespace(), current={},
                          observation_root=tmp_path, main_camera="head", record=lambda event: None)


def test_the_composed_receipt_reaches_the_policy_as_its_stages(tmp_path):
    from ora.agent.native.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )

    _result, outer, _children = _composed(tmp_path, {"approach": "not_reached"}, 32)
    effect = resolve_receipt_effect_projector(outer["schema"])(outer, ReceiptProjectionContext(
        last_action_id=outer["action_id"], last_action_public_name="grasp_execute",
    ))
    assert effect["action"] == "grasp_execute" and effect["status"] == "aborted"
    assert effect["failure_stage"] == "approach"
    assert [item["stage"] for item in effect["detail"]["stages"]] == ["open", "pregrasp", "approach"]
    assert effect["detail"]["stages_not_reached"] == ["close", "lift", "verify"]


# ---- RoboDojo admits composed children by stage, in order -----------------------------------------


def _child(action_id, action="set_gripper"):
    arguments = {"targets": {"right": "open"}} if action == "set_gripper" else {
        "targets": {"right": {"position_m": [0.2, 0.1, 0.9], "orientation_xyzw": [0, 0, 0, 1]}},
        "interactions": {"right": "free_space"},
    }
    return validate_runtime_action({
        "schema": MOTION_ACTION_V3_SCHEMA, "observation_id": "obs-1", "action_id": action_id, "action": action,
        "arguments": arguments, "bound_observation_sha256": "b" * 64,
    })


def _admit(sequence_ids, *, codec="motion_action_v3"):
    from ora.environment.backends.robodojo.sequence_admission import admit_sequence_child

    parent = validate_runtime_action(_action(committed_sequence=["open", "pregrasp", "close", "verify"]))
    frame = {"parent": parent, "children": [], "exact": {}, "next": 0, "composed": (-1, 0)}
    for action_id, kind in sequence_ids:
        admit_sequence_child(frame, _child(action_id, kind), action_codec=codec)


def test_robodojo_admits_ordered_stages_and_counted_retries():
    _admit([("action-7:00-open", "set_gripper"), ("action-7:01-pregrasp", "move_pose"),
            ("action-7:01-pregrasp.2", "move_pose"), ("action-7:02-close", "set_gripper")])
    _admit([("action-7:00-open", "set_gripper"), ("action-7:02-close", "set_gripper")])


@pytest.mark.parametrize(
    "sequence",
    [
        [("action-7:01-pregrasp", "move_pose"), ("action-7:00-open", "set_gripper")],
        [("action-7:00-open", "set_gripper"), ("action-7:00-open", "set_gripper")],
        [("action-7:01-pregrasp", "move_pose"), ("action-7:01-pregrasp.3", "move_pose")],
        [("action-7:02-pregrasp", "move_pose")],
        [("action-7:09-lift", "move_pose")],
        [("action-8:00-open", "set_gripper")],
        [("action-7:01-pregrasp.2", "move_pose")],
    ],
)
def test_robodojo_refuses_reordered_skipped_or_foreign_children(sequence):
    from ora.environment.backends.robodojo.bridge import RobodojoError

    with pytest.raises(RobodojoError):
        _admit(sequence)


def test_robodojo_refuses_composed_children_outside_sparse_motion_v3():
    from ora.environment.backends.robodojo.bridge import RobodojoError

    with pytest.raises(RobodojoError, match="sparse motion v3"):
        _admit([("action-7:00-open", "set_gripper")], codec="target_state_v4")


# ---- evidence readers join every composed child ---------------------------------------------------


def _leaf(action_id, directive):
    action = validate_runtime_action({
        "schema": MOTION_ACTION_V3_SCHEMA, "observation_id": "obs-1", "action_id": action_id, "action": "set_gripper",
        "arguments": {"targets": {"right": directive}}, "bound_observation_sha256": "b" * 64,
    })
    return action, {"schema": "fixture-leaf", "action_id": action_id, "consumed": True,
                    "action": "set_gripper", "arguments": action["arguments"]}


def _composed_outer(**change):
    parent = validate_runtime_action(_action(committed_sequence=["open", "pregrasp", "close", "verify"]))
    first, first_receipt = _leaf("action-7:00-open", "open")
    second, second_receipt = _leaf("action-7:00-open.2", "open")
    close, close_receipt = _leaf("action-7:02-close", "close")
    outer = {
        "schema": "ora.manipulation-skill-sequence-receipt.v1", "action_id": "action-7",
        "action": parent, "committed_sequence": parent["arguments"]["committed_sequence"], "consumed": True,
        "dispatched_stage_actions": 3, "status": "completed",
        "stage_receipts": [
            {"stage": "open", "status": "executed", "action_id": second["action_id"], "action": second,
             "receipt": second_receipt,
             "attempts": [{"action_id": first["action_id"], "action": first, "receipt": first_receipt}]},
            {"stage": "close", "status": "executed", "action_id": close["action_id"], "action": close,
             "receipt": close_receipt},
            {"stage": "verify", "status": "executed", "action_id": None},
        ],
    }
    outer.update(change)
    return parent, outer


def test_the_command_tree_joins_retries_and_skipped_stages_of_a_composed_tool():
    from ora.agent.evidence.command_tree import command_leaves

    parent, outer = _composed_outer()
    leaves = list(command_leaves(parent, outer, lambda receipt: receipt))
    assert [action["action_id"] for action, _receipt in leaves] == [
        "action-7:00-open", "action-7:00-open.2", "action-7:02-close",
    ]


@pytest.mark.parametrize(
    "break_it",
    [
        lambda outer: outer.update(dispatched_stage_actions=2),
        lambda outer: outer["stage_receipts"].reverse(),
        lambda outer: outer["stage_receipts"][0]["attempts"][0].update(action_id="action-7:00-open.3"),
        lambda outer: outer["stage_receipts"][1].update(receipt=None),
        lambda outer: outer.update(committed_sequence=["open"]),
    ],
)
def test_the_command_tree_refuses_a_composed_receipt_that_does_not_join(break_it):
    from ora.agent.evidence.command_tree import command_leaves

    parent, outer = _composed_outer()
    break_it(outer)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        list(command_leaves(parent, outer, lambda receipt: receipt))


def test_a_real_composed_receipt_records_one_child_per_dispatch(tmp_path):
    _result, outer, children = _composed(tmp_path, {}, 32)
    joined = [record["action_id"] for record in outer["stage_receipts"] if record.get("action_id")]
    assert joined == [child["action_id"] for child in children]
    assert all("attempts" not in record for record in outer["stage_receipts"])
    assert outer["dispatched_stage_actions"] == len(children)


def test_the_placement_reader_expands_a_composed_state_into_its_children():
    from ora.agent.tools.placement_receipts import _command_states

    parent, outer = _composed_outer()
    expanded = _command_states([{"previous_public_action": parent, "previous_execution_receipt": outer,
                                 "observation": {"observation_id": "obs-1"}}], None)
    assert [state["previous_public_action"]["action_id"] for state in expanded] == [
        "action-7:00-open", "action-7:00-open.2", "action-7:02-close",
    ]
    assert all(state["_skill_command"] for state in expanded)
