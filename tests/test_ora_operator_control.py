"""Offline ORA input timing, evidence boundaries and human-assisted accounting."""

from __future__ import annotations

import copy
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ora.agent.evidence.operator_messages import operator_trace_errors
from ora.agent.evidence.trajectory import verify_agent_trace
from ora.agent.providers.native import translate_anthropic_request
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.evaluation import assess_learning_eligibility
from ora.orchestrator.operator_control import (
    OperatorControlError, command_status, initialize_control, intervention_reasons,
    normalize_operator_control, submit_command,
)
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from ora.orchestrator.termination import TerminationAuthority

from test_multi_round_session import _SuccessProvider, _SuccessSyntheticSession, _spec


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def control_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from ora.orchestrator import paths

    state = tmp_path / "runtime-state"
    monkeypatch.setattr(paths, "STATE_ROOT", state)
    return state


def _interactive_spec(*, rounds: int = 2) -> dict[str, Any]:
    spec = _spec()
    spec.update({
        "experiment": "ora-interactive", "output_workspace": "outputs/ora-interactive",
        "operator_control": {"mode": "file_inbox_v1"},
        "session": {"cadence": "persistent_agent_k_rounds_v1", "rounds": rounds},
    })
    spec["budgets"]["max_model_calls"] = 4
    return spec


def _run_root(tmp_path: Path, plan: Mapping[str, Any], name: str = "ora-test") -> Path:
    root = tmp_path / name
    root.mkdir()
    _write(root / "plan.json", plan)
    return root


def test_inbox_is_atomic_idempotent_bound_and_outside_run(tmp_path: Path, control_state: Path) -> None:
    plan = resolve_run_spec(_interactive_spec())
    root = _run_root(tmp_path, plan)
    _write(root / "status.json", {"run_id": root.name, "state": "running"})
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    control = initialize_control(root, plan, root.name)
    assert control.root == control_state / "ora-interactive" / root.name
    command = {"id": "steer-1", "type": "steer", "message": "Check the lid before lifting."}
    first = submit_command(root, command)
    assert submit_command(root, command) == first
    assert control.poll() == [first]
    # A writer's unfinished temporary file cannot become a model message.
    (control.root / "inbox" / ".partial.tmp").write_text('{"id":', encoding="utf-8")
    assert control.poll() == [first]
    with pytest.raises(OperatorControlError, match="different contents"):
        submit_command(root, {**command, "message": "A different command."})
    control.acknowledge(first, "applied", before_model_call_id="model-call-000001")
    assert control.poll() == []
    status = command_status(root)
    assert status["pending"] == []
    assert status["acknowledgements"][0]["status"] == "applied"
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after
    drifted = {**plan, "plan_sha256": "b" * 64}
    _write(root / "plan.json", drifted)
    with pytest.raises(OperatorControlError, match="exact run and plan"):
        submit_command(root, {**command, "id": "steer-2"})


def test_terminal_sealed_and_wrong_node_commands_are_refused(tmp_path: Path, control_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = resolve_run_spec(_interactive_spec())
    root = _run_root(tmp_path, plan)
    control = initialize_control(root, plan, root.name)
    command = {"id": "steer-1", "type": "steer", "message": "Inspect again."}
    import ora.orchestrator.operator_control as controls

    with monkeypatch.context() as context:
        context.setattr(controls.socket, "gethostname", lambda: "another-node")
        with pytest.raises(OperatorControlError, match="exact run and plan"):
            submit_command(root, command)
    _write(root / "status.json", {"run_id": root.name, "state": "completed"})
    with pytest.raises(OperatorControlError, match="terminal"):
        submit_command(root, command)
    _write(root / "seal.json", {})
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(OperatorControlError, match="closed or sealed"):
        submit_command(root, command)
    assert not list((control.root / "inbox").glob("*.json"))
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_initial_followups_are_plan_bound_and_ready_before_first_decision(tmp_path: Path, control_state: Path) -> None:
    spec = _interactive_spec(rounds=1)
    initial = {"id": "follow-up-1", "type": "follow_up", "message": "Use the side view to check alignment."}
    spec["operator_control"]["initial_messages"] = [initial]
    plan = resolve_run_spec(spec)
    root = _run_root(tmp_path, plan)
    control = initialize_control(root, plan, root.name)
    assert [item["id"] for item in control.poll()] == [initial["id"]]
    assert initialize_control(root, plan, root.name).poll() == control.poll()
    with pytest.raises(OperatorControlError, match="follow_up requires a new plan"):
        submit_command(root, {**initial, "id": "not-in-plan"})
    assert resolve_run_spec(_interactive_spec(rounds=1))["plan_sha256"] != plan["plan_sha256"]


def test_operator_config_preserves_absent_and_normalizes_empty_initial_list() -> None:
    normal = _spec()
    assert "operator_control" not in resolve_run_spec(normal)
    empty = _interactive_spec()
    explicit_empty = copy.deepcopy(empty)
    explicit_empty["operator_control"]["initial_messages"] = []
    assert resolve_run_spec(empty)["plan_sha256"] == resolve_run_spec(explicit_empty)["plan_sha256"]
    with pytest.raises(OperatorControlError, match="experiment=ora-interactive"):
        normalize_operator_control({"mode": "file_inbox_v1"}, experiment="official-benchmark", policy_kind="native_api")
    with pytest.raises(RunSpecError, match="mode=file_inbox_v1"):
        resolve_run_spec({**empty, "operator_control": {"mode": "unsupported"}})
    with pytest.raises(OperatorControlError, match="unique"):
        normalize_operator_control({"mode": "file_inbox_v1", "initial_messages": [
            {"id": "x", "type": "follow_up", "message": "a"},
            {"id": "x", "type": "follow_up", "message": "b"},
        ]}, experiment="ora-interactive", policy_kind="native_api")


class _SteeredProvider:
    protocol = "anthropic_messages"
    prepare_request = staticmethod(translate_anthropic_request)

    def __init__(self, root: Path, *, steer: bool = True, stop: bool = False) -> None:
        self.root, self.steer, self.stop = root, steer, stop
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def complete(self, *, request: Mapping[str, Any], model_call_id: str) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(dict(request)))
        index = len(self.requests) - 1
        if index == 0 and self.steer:
            submit_command(self.root, {"id": "steer-live", "type": "steer", "message": "Check both viewpoints before continuing."})
        if index == 0 and self.stop:
            signal.raise_signal(signal.SIGTERM)
        calls = (
            [{"id": f"read-{number}", "type": "function", "function": {"name": "check_task", "arguments": "{}"}} for number in range(2)]
            if index == 0 and self.steer else [{
                "id": f"finish-{index}", "type": "function",
                "function": {"name": "finish_episode", "arguments": json.dumps({
                    "outcome": "failure", "reason": "offline qualification complete",
                    "attempts": ["inspected the reset observation"],
                })},
            }]
        )
        now = time.time()
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": calls},
            "finish_reason": "tool_calls", "usage": {
                "input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 20,
                "reasoning_output_tokens": 0, "total_tokens": 1020,
            },
            "raw_events": [{"qualification": True}], "model_call_id": model_call_id,
            "started_unix_s": now, "finished_unix_s": now, "attempts": 1,
        }

    def close(self) -> None:
        self.closed = True


def _execute(root: Path, plan: Mapping[str, Any], provider: _SteeredProvider) -> dict[str, Any]:
    return run_episode(
        run_root=root, plan=plan, run_id=root.name, provider_override=provider,
        environment_session_factory=lambda **kwargs: _SuccessSyntheticSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=kwargs["runtime_config"]["terminal_actions"],
            main_camera="agentview", camera_names=("agentview", "wrist"),
        ),
    )


def test_steer_enters_next_model_decision_after_entire_tool_batch(tmp_path: Path, control_state: Path) -> None:
    plan = resolve_run_spec(_interactive_spec())
    root = _run_root(tmp_path, plan)
    provider = _SteeredProvider(root)
    result = _execute(root, plan, provider)
    assert result["state"] == "completed", result
    assert result["human_intervention"] is True
    assert result["behavioral_sample_valid"] is False
    assert result["operator_message_count"] == 1
    assert all(record["human_intervention"] and not record["behavioral_sample_valid"] for record in result["rounds"])
    assert [record["official_success"] for record in result["rounds"]] == [False, False]
    assert "Check both viewpoints" not in json.dumps(provider.requests[0])
    assert all("Check both viewpoints" in json.dumps(request) for request in provider.requests[1:])
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    delivered = [event for event in events if event["type"] == "operator_message"]
    assert len(delivered) == 1
    delivery = delivered[0]
    assert delivery["payload"]["before_model_call_id"] == "model-call-000001"
    assert sum(event["type"] == "tool_result" for event in events[:delivery["seq"]]) == 2
    projections = [event for event in events if event["type"] == "context_projected"]
    assert any(source["kind"] == "operator_message" for source in projections[1]["payload"]["context_sources"])
    assert verify_agent_trace(root)["status"] == "valid", verify_agent_trace(root)
    outer = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert intervention_reasons(plan, result, outer) == []
    eligibility = assess_learning_eligibility(plan, result, outer)
    assert eligibility["eligible"] is False
    assert "human_intervention" in eligibility["reasons"]
    assert intervention_reasons(plan, {**result, "behavioral_sample_valid": True}, outer) == ["operator_intervention_metadata_mismatch"]
    status = command_status(root)
    assert status["closed"] is True
    assert status["acknowledgements"][0]["status"] == "applied"


def test_initial_followup_reaches_first_request_and_marks_run(tmp_path: Path, control_state: Path) -> None:
    spec = _interactive_spec(rounds=1)
    spec["operator_control"]["initial_messages"] = [{"id": "initial", "type": "follow_up", "message": "Ground the target from both cameras."}]
    plan = resolve_run_spec(spec)
    root = _run_root(tmp_path, plan)
    provider = _SteeredProvider(root, steer=False)
    result = _execute(root, plan, provider)
    assert result["state"] == "completed", result
    assert result["human_intervention"] is True
    assert "Ground the target" in json.dumps(provider.requests[0])
    assert verify_agent_trace(root)["status"] == "valid", verify_agent_trace(root)


def test_human_assisted_success_preserves_official_true(tmp_path: Path, control_state: Path) -> None:
    spec = _interactive_spec(rounds=1)
    spec["operator_control"]["initial_messages"] = [{"id": "initial", "type": "follow_up", "message": "Check the observed gripper before finishing."}]
    plan = resolve_run_spec(spec)
    root = _run_root(tmp_path, plan)
    result = _execute(root, plan, _SuccessProvider())
    assert result["state"] == "completed", result
    assert result["human_intervention"] is True
    assert result["rounds"][0]["official_success"] is True
    assert result["rounds"][0]["behavioral_sample_valid"] is False
    assert verify_agent_trace(root)["status"] == "valid", verify_agent_trace(root)


def test_abort_does_not_deliver_queued_steer_or_execute_returned_tools(tmp_path: Path, control_state: Path) -> None:
    plan = resolve_run_spec(_interactive_spec(rounds=1))
    root = _run_root(tmp_path, plan)
    provider = _SteeredProvider(root, stop=True)
    result = _execute(root, plan, provider)
    assert result["state"] == "incomplete", result
    assert result["termination"] == "operator_interrupt"
    assert result["human_intervention"] is False
    assert result["decisions"] == 0
    assert len(provider.requests) == 1
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    assert not any(event["type"] in {"operator_message", "tool_call_proposed", "action_proposed"} for event in events)
    assert command_status(root)["acknowledgements"][0]["status"] == "rejected"


def test_control_binding_fault_still_closes_started_policy(tmp_path: Path, control_state: Path) -> None:
    plan = resolve_run_spec(_interactive_spec(rounds=1))
    root = _run_root(tmp_path, plan)
    control = initialize_control(root, plan, root.name)
    _write(control.root / "binding.json", {**control.binding, "plan_sha256": "wrong-plan"})
    provider = _SteeredProvider(root, steer=False)
    result = _execute(root, plan, provider)
    assert result["state"] == "incomplete", result
    assert result["termination"] == "trusted_fault"
    assert result["human_intervention"] is False
    assert result["behavioral_sample_valid"] is False
    assert provider.closed is True
    assert provider.requests == []


def test_operator_trace_refuses_message_inside_an_unfinished_tool_batch() -> None:
    plan = {"experiment": "ora-interactive", "operator_control": {"mode": "file_inbox_v1"}, "plan_sha256": "plan"}
    payload = {"schema": "ora.ora-operator-command.v1", "id": "x", "type": "steer", "message": "Stop and inspect.",
               "run_id": "run", "plan_sha256": "plan", "submitted_unix_ns": 1, "round_index": 0,
               "observation_step": 0, "before_model_call_id": "model-call-000001"}
    events = [{"type": "tool_call_proposed", "tool_call_id": "tool"},
              {"type": "operator_message", "run_id": "run", "observation_step": 0, "visibility": "exact", "payload": payload}]
    assert operator_trace_errors(events, plan) == ["event:1:operator_message_before_tool_completion"]
