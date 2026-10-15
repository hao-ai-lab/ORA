"""Argument refusals say what was wrong, and identical repeats are counted.

The EXP11A g016 ``classify_objects`` cell sent ``geometry_query`` the same
arguments fourteen times in five minutes -- ``points``, ``region`` and
``world_points_m`` together -- and every result was the same bare refusal:
``invalid_arguments``, ``retryable: true``, ``field``/``received`` null and
``constraints`` empty. Three things were missing, and each is checked here for
every tool rather than for that one call:

* an argument refusal states that the unchanged request is refused again
  (``retry_requires_changed_arguments``) while ``retryable`` keeps its single
  meaning -- the policy can still recover through a later call;
* a mutually exclusive argument rule names the rule and the argument names
  that were received, never their values;
* the gateway counts byte-identical consecutive failures per policy session
  and appends one factual note to the new result, never to history.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.native.session import NativePlanner
from ora.agent.native.tool_loop import _tool_error
from ora.agent.providers.native import FakeProvider
from ora.agent.tools import robot as robot_tools
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.repeated_failures import RepeatedFailureTracker
from ora.agent.tools.tool_errors import (
    ARGUMENT_REFUSAL_CODES,
    ToolError,
    require_at_most_one_of,
    require_exactly_one_of,
)
from tests.agent.native.test_native_harness import AGENT_RULES, PROMPT, assistant_tool, publish_native_boundary, usage
from tests.orchestrator.test_runtime_v1 import publish_synthetic

#: The arguments the g016 cell sent, verbatim.
G016_ARGUMENTS = {
    "camera": "head", "points": [[224, 177], [148, 184], [246, 146]],
    "region": [0, 0, 1, 1], "step": 48, "window_px": 2,
    "world_points_m": [[0, 0, 0]],
}
OPERATIONS = ["points", "region", "world_points_m"]
NOTE = (
    "this identical request has failed {count} times in a row with {code}; "
    "sending it unchanged will fail again"
)


# --- the envelope -----------------------------------------------------------

def test_robot_module_still_exports_the_one_tool_error_class():
    assert robot_tools.ToolError is ToolError


@pytest.mark.parametrize("code", sorted(ARGUMENT_REFUSAL_CODES))
def test_an_argument_refusal_says_the_unchanged_request_is_refused_again(code):
    reported = ToolError("bad request", code=code).as_dict()
    assert reported["retry_requires_changed_arguments"] is True
    # retryable keeps its one meaning across every consumer: a later call
    # through this tool can still succeed. It is not a resend instruction.
    assert reported["retryable"] is True


@pytest.mark.parametrize("code,retryable", [
    ("stale_observation", True), ("target_geometry_unavailable", True),
    ("internal_tool_error", False), ("episode_terminal", False),
])
def test_other_codes_make_no_claim_about_an_unchanged_repeat(code, retryable):
    reported = ToolError("refused", code=code, retryable=retryable).as_dict()
    assert "retry_requires_changed_arguments" not in reported
    assert reported["retryable"] is retryable


def test_a_raising_site_can_state_the_flag_explicitly():
    assert ToolError(
        "x", code="invalid_arguments", retry_requires_changed_arguments=False,
    ).as_dict()["retry_requires_changed_arguments"] is False
    assert ToolError(
        "x", code="target_ref_unavailable", retry_requires_changed_arguments=True,
    ).as_dict()["retry_requires_changed_arguments"] is True


def test_a_non_tool_exception_becomes_an_internal_error_without_the_flag():
    error = _tool_error(RuntimeError("boom"))["error"]
    assert error["code"] == "internal_tool_error" and error["retryable"] is False
    assert "retry_requires_changed_arguments" not in error


# --- mutually exclusive argument rules --------------------------------------

def _refusal(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


def test_exactly_one_of_names_the_rule_and_the_received_names_only():
    error = _refusal(
        require_exactly_one_of, G016_ARGUMENTS, OPERATIONS,
        message="provide exactly one of points, region or world_points_m",
    )
    assert error == {
        "code": "invalid_arguments",
        "message": "provide exactly one of points, region or world_points_m",
        "retryable": True,
        "retry_requires_changed_arguments": True,
        "field": None,
        "received": ["camera", "points", "region", "step", "window_px", "world_points_m"],
        "constraints": {"exactly_one_of": OPERATIONS, "received": OPERATIONS},
        "infrastructure": False,
    }
    none = _refusal(require_exactly_one_of, {"camera": "head"}, OPERATIONS, message="m")
    assert none["constraints"] == {"exactly_one_of": OPERATIONS, "received": []}
    assert require_exactly_one_of({"camera": "head", "region": [0, 0, 1, 1]}, OPERATIONS, message="m") == "region"


def test_at_most_one_of_treats_an_explicit_null_as_omitted():
    assert require_at_most_one_of({"xyz_m": None, "delta_mm": [1, 0, 0]}, ("xyz_m", "delta_mm"), message="m") == "delta_mm"
    assert require_at_most_one_of({}, ("xyz_m", "delta_mm"), message="m") is None
    error = _refusal(
        require_at_most_one_of, {"xyz_m": [0.1, 0, 0.2], "delta_mm": [5, 0, 0], "gripper": "open"},
        ("xyz_m", "delta_mm"), message="provide xyz_m or delta_mm, not both",
    )
    assert error["message"] == "provide xyz_m or delta_mm, not both"
    assert error["received"] == ["delta_mm", "gripper", "xyz_m"]
    assert error["constraints"] == {"at_most_one_of": ["xyz_m", "delta_mm"], "received": ["xyz_m", "delta_mm"]}


def test_refusals_carry_argument_names_never_argument_values():
    error = _refusal(require_exactly_one_of, G016_ARGUMENTS, OPERATIONS, message="m")
    encoded = json.dumps(error)
    for value in ("224", "177", "[0, 0, 1, 1]", "48", "head"):
        assert value not in encoded, value


def test_geometry_query_reports_the_g016_violation(tmp_path):
    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    error = _refusal(tools._geometry, {**G016_ARGUMENTS, "camera": "head_camera"})
    assert error["message"] == "provide exactly one of points, region or world_points_m"
    assert error["retry_requires_changed_arguments"] is True
    assert error["received"] == ["camera", "points", "region", "step", "window_px", "world_points_m"]
    assert error["constraints"] == {"exactly_one_of": OPERATIONS, "received": OPERATIONS}
    unknown = _refusal(tools._geometry, {"camera": "head_camera", "points": [[1, 1]], "scale": 2})
    assert unknown["received"] == ["camera", "points", "scale"]
    assert unknown["constraints"]["unknown"] == ["scale"]
    assert unknown["constraints"]["required"] == ["camera"]


def test_the_pose_argument_exclusion_reports_through_the_shared_rule():
    error = _refusal(
        ManipulationToolRuntime._atomic_pose_arguments, SimpleNamespace(),
        {"xyz_m": [0.1, 0.0, 0.2], "delta_mm": [5, 0, 0]},
        current_pose=[0.0] * 7, arm="right",
    )
    assert error["constraints"] == {"at_most_one_of": ["xyz_m", "delta_mm"], "received": ["xyz_m", "delta_mm"]}


# --- report_issue (EXP11B census E7) ----------------------------------------

def _report_issue_runtime(tmp_path, *, advisory: bool):
    """The Dojo default surface; without the advisory capability only the handler checks."""
    from dataclasses import replace

    from tests.environments.robodojo.test_robodojo_perception import BOUNDS, FakeHeadGrounding, FakeX5Proposer

    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control,
        manual_path=Path(__file__).resolve().parents[3]
        / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile="robodojo_placement_closed_skills_v2", controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=FakeX5Proposer(),
    )
    if not advisory:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            name for name in runtime.surface.capabilities if name != "public_rgbd_retention_advisory_v1"))
    return runtime


@pytest.mark.parametrize("advisory", [True, False])
@pytest.mark.parametrize("message", ["", " ", "\n\t  "])
def test_a_blank_report_issue_message_names_the_field_and_rule(tmp_path, advisory, message):
    runtime = _report_issue_runtime(tmp_path, advisory=advisory)
    error = _refusal(runtime.invoke, "report_issue", {"message": message})
    assert error["code"] == "invalid_arguments" and error["field"] == "message"
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True
    assert error["received"] == {"type": "string", "length": len(message)}
    if message:
        # Whitespace passes the declared minLength; the handler states the rule.
        assert error["constraints"] == {"minLength": 1, "rule": "non_whitespace"}
    else:
        # Every surface refuses the empty string with the declared schema's rule.
        assert error["constraints"] == {"violated": "minLength", "minLength": 1}
    assert not (runtime.control_root / "openeta-reported-issues.jsonl").exists()


@pytest.mark.parametrize("arguments,field", [
    ({}, None),
    ({"message": 3}, "message"),
    ({"message": ["broken"]}, "message"),
    ({"message": "grasp_plan returns no candidates", "severity": "high"}, None),
])
def test_a_malformed_report_issue_names_what_was_wrong_without_values(tmp_path, arguments, field):
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    error = _refusal(runtime.invoke, "report_issue", arguments)
    assert error["code"] == "invalid_arguments" and error["field"] == field
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True
    assert error["constraints"]["violated"] in {"required", "type", "additionalProperties"}
    assert "broken" not in json.dumps(error) and "high" not in json.dumps(error)


@pytest.mark.parametrize("arguments", [["message"], "message", None])
def test_a_non_object_report_issue_is_a_policy_refusal_never_an_internal_error(tmp_path, arguments):
    """An internal_tool_error would count as infrastructure evidence for abort."""
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    error = _refusal(runtime._report_issue, arguments)
    assert error["code"] == "invalid_arguments" and error["constraints"]["violated"] == "type"
    assert error["infrastructure"] is False


def test_a_report_issue_with_text_is_recorded_stripped(tmp_path):
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    assert runtime.invoke("report_issue", {"message": "  grasp_plan returns no candidates \n"}) == {"success": True}
    recorded = json.loads((runtime.control_root / "openeta-reported-issues.jsonl").read_text())
    assert recorded["message"] == "grasp_plan returns no candidates"


# --- repeated identical failures --------------------------------------------

def _failure(code="invalid_arguments", **kwargs):
    return {"ok": False, "error": ToolError("refused", code=code, **kwargs).as_dict()}


def test_identical_consecutive_failures_are_counted_on_the_new_result_only():
    tracker = RepeatedFailureTracker()
    first = _failure()
    assert tracker.record("geometry_query", G016_ARGUMENTS, first) is first
    second = tracker.record("geometry_query", dict(reversed(list(G016_ARGUMENTS.items()))), _failure())
    third_input = _failure()
    third = tracker.record("geometry_query", G016_ARGUMENTS, third_input)
    assert second["error"]["repetition"] == {
        "identical_failures_in_a_row": 2,
        "note": NOTE.format(count=2, code="invalid_arguments"),
    }
    assert third["error"]["repetition"] == {
        "identical_failures_in_a_row": 3,
        "note": NOTE.format(count=3, code="invalid_arguments"),
    }
    assert "repetition" not in first["error"] and "repetition" not in third_input["error"]


@pytest.mark.parametrize("interruption", [
    ("geometry_query", {**G016_ARGUMENTS, "step": 49}, _failure()),
    ("observe", {}, _failure()),
    ("geometry_query", G016_ARGUMENTS, {"ok": True, "points": []}),
    ("geometry_query", G016_ARGUMENTS, _failure("stale_observation")),
    ("geometry_query", None, _failure()),
])
def test_any_different_call_or_outcome_resets_the_count(interruption):
    tracker = RepeatedFailureTracker()
    tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    name, arguments, result = interruption
    tracker.record(name, arguments, result)
    after = tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    assert "repetition" not in after["error"]


def test_the_note_claims_a_certain_repeat_failure_only_for_argument_refusals():
    tracker = RepeatedFailureTracker()
    for _ in range(2):
        repeated = tracker.record("place_execute", {"plan_ref": "p"}, _failure("stale_plan_ref"))
    assert repeated["error"]["repetition"] == {
        "identical_failures_in_a_row": 2,
        "note": "this identical request has failed 2 times in a row with stale_plan_ref",
    }


def test_two_policy_sessions_never_share_a_count():
    left, right = RepeatedFailureTracker(), RepeatedFailureTracker()
    left.record("geometry_query", G016_ARGUMENTS, _failure())
    assert "repetition" not in right.record("geometry_query", G016_ARGUMENTS, _failure())["error"]


# --- the v3 causal projection carries the new public fields -----------------

def test_native_causal_projection_keeps_the_refusal_and_its_count():
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=resolve_tool_surface("robodojo_placement_closed_skills_v2"))
    tracker = RepeatedFailureTracker()
    for _ in range(3):
        try:
            require_exactly_one_of(G016_ARGUMENTS, OPERATIONS, message="provide exactly one of points, region or world_points_m")
        except ToolError as exc:
            result = tracker.record("geometry_query", G016_ARGUMENTS, _tool_error(exc))
    visible = planner._causal_tool_result("geometry_query", result)["error"]
    assert visible["retry_requires_changed_arguments"] is True
    assert visible["constraints"] == {"exactly_one_of": OPERATIONS, "received": OPERATIONS}
    assert visible["received"] == sorted(G016_ARGUMENTS)
    assert visible["repetition"]["note"] == NOTE.format(count=3, code="invalid_arguments")


# --- the Native gateway: one count per policy session, history untouched ----

def test_native_loop_appends_the_count_without_editing_history(tmp_path: Path):
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    doubled = {"camera": "head_camera", "points": [[1, 1]], "region": [0, 0, 2, 2]}
    reordered = {"region": [0, 0, 2, 2], "points": [[1, 1]], "camera": "head_camera"}
    script = [
        assistant_tool("g1", "geometry.query", doubled),
        assistant_tool("g2", "geometry.query", reordered),
        assistant_tool("g3", "geometry.query", doubled),
        assistant_tool("manual", "robot.read_manual", {"section": "not-a-section"}),
        assistant_tool("g4", "geometry.query", doubled),
        assistant_tool("act", "robot.submit_action", {"action": "set_gripper", "arm": "left", "gripper": 0.5}),
    ]
    provider = FakeProvider([{"message": message, "usage": usage(100 + index)} for index, message in enumerate(script)])
    planner = NativePlanner(
        tmp_path, "repeated-failure", tools, provider, AGENT_RULES, PROMPT,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
        max_model_calls_per_observation=len(script),
    )
    snapshots: list[list[str]] = []
    original = planner._record_tool_result

    def recording(**kwargs):
        snapshots.append([m["content"] for m in planner.messages if m.get("role") == "tool"])
        return original(**kwargs)

    planner._record_tool_result = recording
    outcome = planner.run_observation(
        task_instruction="Move once.", context=pointer, image_path=workspace / pointer["main_image"],
    )
    assert outcome["submitted"] is True
    results = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in planner.messages if message.get("role") == "tool"
    }
    repetition = {call: result.get("error", {}).get("repetition") for call, result in results.items()}
    assert repetition["g1"] is None and repetition["manual"] is None and repetition["g4"] is None
    assert repetition["g2"]["identical_failures_in_a_row"] == 2
    assert repetition["g3"] == {
        "identical_failures_in_a_row": 3, "note": NOTE.format(count=3, code="invalid_arguments"),
    }
    assert results["g1"]["error"]["constraints"] == {"exactly_one_of": OPERATIONS, "received": ["points", "region"]}
    # Each result is appended once; nothing already delivered is rewritten.
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later[: len(earlier)] == earlier
    for earlier, later in zip(provider.requests, provider.requests[1:]):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
    planner.close()
