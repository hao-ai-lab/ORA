"""check.hold fusion, the shared question bank and sdk.ask (composed tools M2b)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from ora.agent.tools.composed import contract, hold, questions
from ora.agent.tools.composed.checks import check_tool, known_tool_names
from ora.agent.tools.composed.host import ExecutePort
from ora.agent.tools.composed.tunables import load_tunables
from test_composed_tool_execution import RETRYING, _arguments, _fixture

TUNABLES = load_tunables()


# ---- check.hold -----------------------------------------------------------------------------------


def _verification(strength="not_contradictory", motion="unknown", jaw=None):
    block = {"schema": "ora.closure-extent-evidence.v1", "strength": strength, "reason": "r",
             "lift_probe": {"surface_motion": {"status": motion, "reason": "m"}}}
    if jaw is not None:
        block["jaw_hold"] = {"schema": "ora.jaw-stall-hold-evidence.v1", "held": jaw, "reason": "j"}
    return block


@pytest.mark.parametrize(
    ("verification", "judgment", "expected"),
    [
        (_verification(), None, "unknown"),
        (_verification(motion="rigid_motion_supported"), None, "supported"),
        (_verification(strength="strong_negative", motion="rigid_motion_supported"), None, "contradicted"),
        (_verification(motion="stationary_surface_supported"), None, "contradicted"),
        (_verification(strength="weak_negative"), None, "unknown"),
        (_verification(jaw="likely"), None, "supported"),
        (_verification(jaw="unlikely", motion="rigid_motion_supported"), None, "contradicted"),
        (_verification(), {"status": "answered", "question": "q@v1", "answer": "held", "reason": "x"}, "supported"),
        (_verification(), {"status": "answered", "question": "q@v1", "answer": "empty", "reason": "x"}, "contradicted"),
        (_verification(), {"status": "unavailable", "question": "q@v1", "answer": None, "reason": "x"}, "unknown"),
        (None, None, "unknown"),
    ],
)
def test_hold_fuses_verdicts_by_the_rule_table(verification, judgment, expected):
    verdict = hold.hold_verdict(verification, judgment)
    assert verdict["hold"] == expected
    assert verdict["schema"] == hold.SCHEMA and verdict["rules"] == hold.RULES
    assert verdict["claim_boundary"] == hold.CLAIM_BOUNDARY
    if judgment is not None:
        assert verdict["sources"][-1]["source"] == "model_judgment" and verdict["sources"][-1]["model_judgment"]


def test_every_rule_value_is_a_reading():
    assert {reading for table in hold.RULE_TABLE.values() for reading in table.values()} == {
        "supports", "contradicts", "neutral",
    }


def test_check_hold_is_a_pure_sdk_function_a_body_can_branch_on(tmp_path):
    source = RETRYING.replace(
        'return sdk.complete(measurements = [sdk.measure("verify", "close_attempts", attempt + 1)])',
        'verdict = sdk.check.hold({"strength": "strong_negative"})\n'
        '            return sdk.complete(measurements = [sdk.measure("verify", "hold", verdict["hold"])])',
    )
    tool = _fixture(tmp_path, source)
    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES)
    assert done.finished.status == "completed"
    assert {"quantity": "hold", "value": "contradicted"} in done.finished.stages[-1]["measurements"]


# ---- the question bank ----------------------------------------------------------------------------


def test_the_first_question_is_registered_but_not_admitted():
    bank = questions.load_questions()
    assert list(bank) == ["gripper_holds_object@v1"]
    asked = bank["gripper_holds_object@v1"]
    assert asked.answers == ("held", "empty", "unclear") and asked.images == {"wrist": 1}
    assert not asked.admitted and asked.threshold == 0.9
    assert "[image: wrist]" in asked.render({}, ["wrist"])


def _write(root, **change):
    document = yaml.safe_load((questions.QUESTIONS_ROOT / "gripper_holds_object" / "v1.yaml").read_text())
    document.update(change)
    path = root / document["id"] / f"{document['version']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document))
    return root


def test_an_evaluated_question_is_admitted_and_checks_its_answers(tmp_path):
    root = _write(tmp_path, eval={"set": "sealed/wrist-hold-v1", "accuracy": 0.93, "threshold": 0.9})
    asked = questions.load_question("gripper_holds_object@v1", root)
    assert asked.admitted
    assert asked.answer({"answer": "held", "reason": "both jaws touch the can"})["status"] == "answered"
    for bad in ({"answer": "maybe", "reason": "x"}, {"answer": "held", "reason": "x" * 201}, "held", None):
        assert asked.answer(bad)["status"] == "unavailable"
    with pytest.raises(questions.QuestionError, match="not declared"):
        asked.render({}, ["head"])
    with pytest.raises(questions.QuestionError, match="more than 1"):
        asked.render({}, ["wrist", "wrist"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"prompt": "look at {head}"}, "undeclared slots"),
        ({"answer": {"type": "enum", "values": ["held"], "reason_max_chars": 200}}, "at least two"),
        ({"eval": {"set": None, "accuracy": 0.95, "threshold": 0.9}}, "evaluation set"),
        ({"eval": {"set": "s", "accuracy": 0.95, "threshold": 1.5}}, "threshold"),
        ({"version": "1"}, "<id>@v<n>"),
    ],
)
def test_malformed_questions_are_refused(tmp_path, change, message):
    document = yaml.safe_load((questions.QUESTIONS_ROOT / "gripper_holds_object" / "v1.yaml").read_text())
    document.update(change)
    with pytest.raises(questions.QuestionError, match=message):
        questions.parse_question(document, path="fixture.yaml")


# ---- sdk.ask --------------------------------------------------------------------------------------


def _port(host, monkeypatch, root=None):
    if root is not None:
        monkeypatch.setattr(questions, "QUESTIONS_ROOT", root)
    latest = ({"observation_id": "obs-9"}, None)
    return ExecutePort(host, host_state={}, capture=lambda: latest, latest=lambda: latest, completed=lambda: [])


def test_ask_is_unavailable_until_the_question_is_admitted_and_a_model_is_bound(tmp_path, monkeypatch):
    calls = []

    def judge(question, prompt, images, values):
        calls.append((question.ref, prompt, images, values))
        return {"answer": "held", "reason": "both jaws touch the can"}

    host = SimpleNamespace(model_judge=judge)
    assert _port(host, monkeypatch).ask("gripper_holds_object@v1", ["wrist"])["reason"] == "question_not_admitted"
    root = _write(tmp_path, eval={"set": "sealed/wrist-hold-v1", "accuracy": 0.93, "threshold": 0.9})
    unbound = _port(SimpleNamespace(), monkeypatch, root).ask("gripper_holds_object@v1", ["wrist"])
    assert unbound["status"] == "unavailable" and unbound["reason"] == "no_runtime_model_binding"
    answered = _port(host, monkeypatch, root).ask("gripper_holds_object@v1", ["wrist"])
    assert answered == {"status": "answered", "question": "gripper_holds_object@v1", "answer": "held",
                        "reason": "both jaws touch the can"}
    assert calls[0][2] == [{"role": "wrist", "observation_id": "obs-9"}]
    with pytest.raises(ValueError, match="not declared"):
        _port(host, monkeypatch, root).ask("gripper_holds_object@v1", ["head"])


def test_a_tool_may_only_ask_an_admitted_question(tmp_path):
    source = RETRYING.replace(
        'return sdk.complete(measurements = [sdk.measure("verify", "close_attempts", attempt + 1)])',
        'sdk.ask("gripper_holds_object@v1", images = ["wrist"])\n            return sdk.complete()',
    )
    tool = _fixture(tmp_path, source.replace("queries = 1", "queries = 2"))
    findings = check_tool(tool, tunables=TUNABLES, known_names=known_tool_names())
    assert [item.check for item in findings] == ["ask_question"]
    assert "evaluation threshold" in findings[0].message
    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES)
    assert done.finished.status == "completed"
