"""finish_episode v2: success or give_up with a reason, abort only on recorded infrastructure.

User decision 2026-10-04 (integration 0059 F1/F2): giving up is
``finish_episode(outcome="give_up")`` with a reason and a category; success
needs a reason that cites observable evidence; ``failure`` and ``attempts``
are retired; abort is admitted only when this round's Harness record holds an
infrastructure failure, on RoboTwin and LIBERO as on RoboDojo. Every refusal
of a reason or category is structured, retryable and names its field, and is
raised before any lifecycle state is written.

The capabilities are declared by the v2 surface generations (A0), which the
three current defaults are; these tests use test-local copies of them
(``finish_v2_surfaces``), and the kept RoboTwin noservo_v1 surface stands for
the released v1 contract. Offline: the fake Dojo plant runtime and the real
dispatcher with result-schema admission; finish_episode reads only the Control
root and the run's event log, so the same runtime serves every backend's
surface.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

import finish_v2_surfaces as v2
from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.abort_evidence_schema import infrastructure_classes_text
from ora.agent.composition.finish_outcomes_schema import (
    ABORT_CAPABILITY, ABORT_REFUSAL_CODE, CAPABILITY, CATEGORIES, CATEGORY_REFUSAL_CODE,
    MIN_REASON_CHARACTERS, MIN_REASON_WORDS, OUTCOMES, REASON_REFUSAL_CODE,
    bind_abort_round, bind_finish_outcomes,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.tools import finish_outcomes
from ora.agent.tools.robot import ToolError

BACKENDS = tuple(v2.DEFAULTS)
#: The one kept surface still on the released v1 finish contract.
RELEASED = "robotwin_toolchain_b2_closed_noservo_v1"
GOOD_GIVE_UP = "the mug slid off the table edge after the second grasp and is not in any camera"
GOOD_SUCCESS = "check_task returned true and the head camera image shows the mug resting on the plate"
GOOD_ABORT = "vision_ground failed with internal_tool_error, the grounding service is down"
RESET = {"type": "reset", "time": 1.0, "record": {"round_index": 0}}


def _finish(surface):
    return next(item for item in surface.definitions if item["name"] == "finish_episode")


def _runtime(tmp_path, surface, *, task_success=False):
    import test_robodojo_b2_parity as parity

    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = surface
    runtime.allowed_tools = tuple(item["name"] for item in surface.definitions)
    pointer, source, observation = runtime._current()
    pointer = {**pointer, "generation": 3, "observation_sha256": hashlib.sha256(
        json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    (runtime.control_root / "policy-task-check.json").write_text(
        json.dumps({"terminal": False, "task_success": task_success}), encoding="utf-8")
    return runtime


def _log(runtime, *events):
    with runtime.event_path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


def _refused(runtime, arguments):
    with pytest.raises(ToolError) as refused:
        runtime.invoke("finish_episode", arguments)
    assert not (runtime.control_root / "policy-finish.json").exists()
    return refused.value


# --------------------------------------------------------------------------
# The declared contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_a_v2_surface_declares_success_give_up_and_evidence_gated_abort(monkeypatch, tmp_path, backend):
    surface = v2.resolve(monkeypatch, tmp_path, backend)
    assert surface.supports(CAPABILITY) and surface.supports(ABORT_CAPABILITY)
    definition = _finish(surface)
    schema = definition["inputSchema"]
    assert schema["properties"]["outcome"]["enum"] == [*OUTCOMES, "abort"]
    assert schema["required"] == ["outcome", "reason"]
    assert set(schema["properties"]) == {"outcome", "reason", "category"}
    assert [item for item in schema["properties"]["category"]["enum"] if item is not None] == list(CATEGORIES)
    assert surface.result_schemas["finish_episode"]["properties"]["outcome"]["enum"] == [*OUTCOMES, "abort"]
    text = definition["description"]
    assert text.endswith(infrastructure_classes_text())
    assert f"at least {MIN_REASON_CHARACTERS} characters and {MIN_REASON_WORDS} words" in text
    for category in CATEGORIES:
        assert category in text
    # The rule is stated; nothing tells the policy what to try instead.
    for advice in ("try", "instead", "consider", "should"):
        assert advice not in text.lower().split()


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_help_on_a_v2_surface_names_the_retired_failure_exit(monkeypatch, tmp_path, backend):
    surface = v2.resolve(monkeypatch, tmp_path, backend)
    for definition in surface.definitions:
        text = json.dumps(definition)
        assert "finish with failure" not in text, definition["name"]
        assert "exhausted attempts" not in text, definition["name"]
    recover = next(item for item in surface.definitions if item["name"] == "recover_pose")
    assert "finish_episode names the accepted outcomes" in recover["description"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_without_the_abort_capability_the_outcomes_are_exactly_success_and_give_up(monkeypatch, tmp_path, backend):
    surface = v2.resolve(monkeypatch, tmp_path, backend, abort=False, drop=(ABORT_CAPABILITY,))
    definition = _finish(surface)
    assert definition["inputSchema"]["properties"]["outcome"]["enum"] == list(OUTCOMES)
    assert surface.result_schemas["finish_episode"]["properties"]["outcome"]["enum"] == list(OUTCOMES)
    assert "abort" not in definition["description"]


def test_a_released_surface_keeps_its_released_contract():
    surface = resolve_tool_surface(RELEASED)
    assert not surface.supports(CAPABILITY) and not surface.supports(ABORT_CAPABILITY)
    outcomes = _finish(surface)["inputSchema"]["properties"]["outcome"]["enum"]
    assert outcomes == ["success", "failure", "abort"]


def test_the_binders_refuse_a_surface_they_cannot_describe():
    definitions = [{"name": "finish_episode", "description": "x", "inputSchema": {}}]
    results = {"finish_episode": {"type": "object", "properties": {"outcome": {"enum": ["success"]}}}}
    with pytest.raises(ToolSurfaceError, match="faithful_check_task"):
        bind_finish_outcomes(copy.deepcopy(definitions), copy.deepcopy(results), [CAPABILITY])
    with pytest.raises(ToolSurfaceError, match="one declared finish_episode"):
        bind_finish_outcomes([], copy.deepcopy(results), [CAPABILITY, "faithful_check_task"])
    with pytest.raises(ToolSurfaceError, match=CAPABILITY):
        bind_abort_round(copy.deepcopy(definitions), copy.deepcopy(results), [ABORT_CAPABILITY])
    leftover = [*copy.deepcopy(definitions), {"name": "recover_pose", "inputSchema": {},
                                              "description": "On error: finish with failure."}]
    with pytest.raises(ToolSurfaceError, match="recover_pose"):
        bind_finish_outcomes(leftover, copy.deepcopy(results), [CAPABILITY, "faithful_check_task"])


def test_a_released_surface_admission_is_unchanged():
    from types import SimpleNamespace

    released = SimpleNamespace(surface=resolve_tool_surface(RELEASED))
    for outcome in ("success", "failure", "abort"):
        assert finish_outcomes.admit(released, {"outcome": outcome}) is None
    assert finish_outcomes.accepted_outcomes(released.surface) is None
    assert finish_outcomes.record_fields(released, {"outcome": "failure"}) == {}


# --------------------------------------------------------------------------
# The real dispatcher
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reason", [None, "", "   ", "n/a", "give up", "TODO", "placeholder reason",
                                    "x" * 40, "give_up give_up give_up give_up give_up",
                                    "too short to say much"])
def test_an_empty_token_or_placeholder_reason_is_refused_and_names_the_field(
        monkeypatch, tmp_path, backend, reason):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    arguments = {"outcome": "give_up", "category": "object_lost"}
    if reason is not None:
        arguments["reason"] = reason
    error = _refused(runtime, arguments)
    assert error.code == REASON_REFUSAL_CODE and error.retryable is True and error.field == "reason"
    assert error.constraints["minimum_characters"] == MIN_REASON_CHARACTERS
    assert error.constraints["minimum_words"] == MIN_REASON_WORDS
    assert error.as_dict()["infrastructure"] is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_give_up_needs_a_category_from_the_enum(monkeypatch, tmp_path, backend):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    for category in (None, "", "impossible"):
        arguments = {"outcome": "give_up", "reason": GOOD_GIVE_UP}
        if category is not None:
            arguments["category"] = category
        error = _refused(runtime, arguments)
        assert error.code == CATEGORY_REFUSAL_CODE and error.retryable is True
        assert error.field == "category" and error.constraints["allowed"] == list(CATEGORIES)
    error = _refused(runtime, {"outcome": "abort", "reason": GOOD_ABORT, "category": "other"})
    assert error.field == "category" and error.retryable is True


@pytest.mark.parametrize("backend", BACKENDS)
def test_failure_and_attempts_are_retired(monkeypatch, tmp_path, backend):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    error = _refused(runtime, {"outcome": "failure", "reason": GOOD_GIVE_UP})
    assert error.field == "outcome" and error.constraints["allowed"] == [*OUTCOMES, "abort"]
    error = _refused(runtime, {"outcome": "give_up", "reason": GOOD_GIVE_UP, "category": "object_lost",
                               "attempts": ["grasped twice"]})
    assert error.field == "attempts"


@pytest.mark.parametrize("backend", BACKENDS)
def test_give_up_is_accepted_and_recorded(monkeypatch, tmp_path, backend):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    accepted = runtime.invoke("finish_episode", {
        "outcome": "give_up", "reason": GOOD_GIVE_UP, "category": "object_lost"})
    assert accepted["outcome"] == "give_up" and accepted["success"] is True
    record = json.loads((runtime.control_root / "policy-finish.json").read_text())
    assert record["finish_contract"] == CAPABILITY
    assert record["outcome"] == "give_up" and record["category"] == "object_lost"
    assert record["reason"] == GOOD_GIVE_UP
    assert "abort_infrastructure_evidence" not in record and "abort_admission" not in record
    assert isinstance(record["untried_candidates"], dict)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_success_reason_must_cite_observable_evidence(monkeypatch, tmp_path, backend):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend), task_success=True)
    error = _refused(runtime, {"outcome": "success", "reason": "the requested arrangement is complete now"})
    assert error.code == REASON_REFUSAL_CODE and error.field == "reason" and error.retryable is True
    assert error.constraints["problem"] == "no_evidence_source"
    assert "check_task" in error.constraints["evidence_terms"]
    # check_task true still latches success: give_up is refused.
    with pytest.raises(ToolError) as latched:
        runtime.invoke("finish_episode", {"outcome": "give_up", "reason": GOOD_GIVE_UP,
                                          "category": "object_lost"})
    assert latched.value.code == "task_success_latched"
    accepted = runtime.invoke("finish_episode", {"outcome": "success", "reason": GOOD_SUCCESS})
    assert accepted["outcome"] == "success"
    record = json.loads((runtime.control_root / "policy-finish.json").read_text())
    assert record["finish_contract"] == CAPABILITY and "untried_candidates" not in record


@pytest.mark.parametrize("reason,problem", [
    ("check task returned true with the mug on the plate", None),
    ("task check true, the mug sits on the plate", None),
    ("checkTask returned True for the stacked blocks", None),
    ("the wrist image shows the drawer closed flush", None),
    ("it is done, the task is complete, ok", "placeholder"),
    ("the requested arrangement is complete now", "no_evidence_source"),
    ("done done done done done done done", "too_few_words"),
    # A CJK character counts as two characters and half a word.
    ("头部相机画面显示杯子已经放在盘子上", None),
    ("check_task 返回 true，杯子在盘子上", None),
    ("杯子掉到桌子下面了，再也找不到了", "no_evidence_source"),
    ("放弃本轮", "too_short"),
])
def test_the_reason_rule(reason, problem):
    assert finish_outcomes.reason_problem("success", reason) == problem
    # Only success needs a named evidence source.
    expected = None if problem == "no_evidence_source" else problem
    assert finish_outcomes.reason_problem("give_up", reason) == expected


@pytest.mark.parametrize("arguments", [
    {"outcome": "give_up", "reason": GOOD_GIVE_UP, "category": ["object_lost"]},
    {"outcome": "give_up", "reason": GOOD_GIVE_UP, "category": {"name": "object_lost"}},
    {"outcome": "give_up", "reason": GOOD_GIVE_UP, "category": 3},
    {"outcome": "give_up", "reason": [GOOD_GIVE_UP], "category": "object_lost"},
    {"outcome": "give_up", "reason": {"text": GOOD_GIVE_UP}, "category": "object_lost"},
    {"outcome": ["give_up"], "reason": GOOD_GIVE_UP, "category": "object_lost"},
    {"outcome": {"value": "give_up"}, "reason": GOOD_GIVE_UP},
    {"outcome": None, "reason": GOOD_GIVE_UP},
    {"reason": GOOD_GIVE_UP},
])
def test_malformed_arguments_are_a_policy_refusal_never_infrastructure_evidence(monkeypatch, tmp_path, arguments):
    """A malformed finish must not become the evidence an abort is admitted on."""
    from ora.agent.tools import abort_evidence

    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, "robotwin"))
    _log(runtime, RESET)
    error = _refused(runtime, arguments)
    assert error.retryable is True and error.as_dict()["infrastructure"] is False
    logged = [json.loads(line) for line in runtime.event_path.read_text().splitlines()]
    assert logged[-1]["type"] == "tool_error" and logged[-1]["error"]["code"] != "internal_tool_error"
    assert abort_evidence.recorded_signals(runtime.event_path) == []
    assert _refused(runtime, {"outcome": "abort", "reason": GOOD_ABORT}).code == ABORT_REFUSAL_CODE


@pytest.mark.parametrize("backend", BACKENDS)
def test_success_still_needs_check_task(monkeypatch, tmp_path, backend):
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    with pytest.raises(ToolError, match="check_task=true"):
        runtime.invoke("finish_episode", {"outcome": "success", "reason": GOOD_SUCCESS})


@pytest.mark.parametrize("backend", BACKENDS)
def test_abort_without_a_recorded_infrastructure_failure_is_refused(monkeypatch, tmp_path, backend):
    import test_policy_abort_evidence as v1

    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    _log(runtime, RESET, *v1.G012_RECORD, *v1.INTER_ARM_REFUSALS.values())
    error = _refused(runtime, {"outcome": "abort", "reason": "the target is outside the reachable workspace"})
    assert error.code == ABORT_REFUSAL_CODE and error.field == "outcome"
    assert error.constraints["allowed"] == [*OUTCOMES, "abort"]
    assert error.constraints["recorded_infrastructure_signals"] == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_abort_refusal_is_retryable_because_the_same_abort_is_admitted_once_a_fault_is_recorded(
        monkeypatch, tmp_path, backend):
    """EXP11B census E2: the refusal depends on the round's record, not on the arguments."""
    from ora.agent.composition.abort_evidence_schema import counted_classes
    import test_policy_abort_evidence as v1

    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    _log(runtime, RESET, *v1.G012_RECORD)
    arguments = {"outcome": "abort", "reason": GOOD_ABORT}
    error = _refused(runtime, arguments)
    reported = error.as_dict()
    assert reported["retryable"] is True and reported["infrastructure"] is False
    # The unchanged request can succeed after the record changes, so the
    # refusal makes no changed-arguments claim.
    assert "retry_requires_changed_arguments" not in reported
    # The counted classes come from the gate's own table; the recorded signals
    # are what the gate read, not a literal.
    assert error.constraints["counted"] == counted_classes()
    assert error.constraints["recorded_infrastructure_signals"] == []
    assert "admitted once" in str(error)
    _log(runtime, v1.INFRASTRUCTURE_RECEIPTS["execution_fault"])
    assert runtime.invoke("finish_episode", arguments)["outcome"] == "abort"


def _task_check(runtime, *, terminal, task_success):
    (runtime.control_root / "policy-task-check.json").write_text(
        json.dumps({"terminal": terminal, "task_success": task_success}), encoding="utf-8")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("arguments", [
    {"outcome": "abort", "reason": GOOD_ABORT},
    {"outcome": "abort", "reason": "abort"},
    {"outcome": "give_up", "reason": "n/a"},
    {"outcome": "give_up", "reason": GOOD_GIVE_UP, "category": "object_lost"},
    {"outcome": "failure", "reason": GOOD_GIVE_UP},
])
def test_a_latched_success_is_refused_first_and_is_retryable_with_outcome_success(
        monkeypatch, tmp_path, backend, arguments):
    """EXP11B census E4: once check_task confirms success, the latch is the refusal that helps."""
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend), task_success=True)
    _log(runtime, RESET)
    error = _refused(runtime, arguments)
    reported = error.as_dict()
    assert reported["code"] == "task_success_latched"
    assert reported["retryable"] is True and reported["retry_requires_changed_arguments"] is True
    assert reported["field"] == "outcome" and reported["received"] == arguments["outcome"]
    assert reported["constraints"]["allowed"] == ["success"]
    assert reported["infrastructure"] is False
    assert runtime.invoke("finish_episode", {"outcome": "success", "reason": GOOD_SUCCESS})["outcome"] == "success"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("terminal", [False, True])
def test_an_unconfirmed_success_is_retryable_only_while_the_round_is_not_terminal(
        monkeypatch, tmp_path, backend, terminal):
    """EXP11B census E4: in the terminal turn check_task can no longer become true."""
    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    _task_check(runtime, terminal=terminal, task_success=False)
    error = _refused(runtime, {"outcome": "success", "reason": GOOD_SUCCESS})
    assert error.code == "task_success_not_confirmed" and error.field == "outcome"
    assert error.retryable is (not terminal)
    assert error.constraints["terminal"] is terminal and error.constraints["task_success"] is False
    assert "check_task=true" in str(error)
    assert ("ended" in str(error)) is terminal


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fault", ["planner_error", "execution_fault", "skill_fault"])
def test_abort_on_a_recorded_failure_is_admitted_and_names_its_evidence(monkeypatch, tmp_path, backend, fault):
    import test_policy_abort_evidence as v1

    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, backend))
    _log(runtime, v1.INFRASTRUCTURE_RECEIPTS["execution_fault"], RESET, v1.INFRASTRUCTURE_RECEIPTS[fault])
    error = _refused(runtime, {"outcome": "abort", "reason": "abort"})
    assert error.code == REASON_REFUSAL_CODE and error.field == "reason"
    accepted = runtime.invoke("finish_episode", {"outcome": "abort", "reason": GOOD_ABORT})
    assert accepted["outcome"] == "abort"
    record = json.loads((runtime.control_root / "policy-finish.json").read_text())
    assert record["abort_admission"] == "infrastructure_evidence"
    # Only this round's record counts: the fault before the reset is not evidence.
    assert len(record["abort_infrastructure_evidence"]) == 1
    assert "untried_candidates" not in record


def test_abort_is_not_an_outcome_without_the_abort_capability(monkeypatch, tmp_path):
    import test_policy_abort_evidence as v1

    runtime = _runtime(tmp_path, v2.resolve(monkeypatch, tmp_path, "robotwin", abort=False, drop=(ABORT_CAPABILITY,)))
    _log(runtime, RESET, v1.INFRASTRUCTURE_RECEIPTS["execution_fault"])
    error = _refused(runtime, {"outcome": "abort", "reason": GOOD_ABORT})
    assert error.field == "outcome" and error.constraints["allowed"] == list(OUTCOMES)
