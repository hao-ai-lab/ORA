"""Operator stop versus in-flight policy work (RoboDojo D3R-006).

An explicit ``ora stop`` delivers SIGTERM to the Supervisor. Its
handler interrupts the PolicySession from another thread, which closes the
provider under any in-flight request. Every error or reply that surfaces after
that stop is the stop itself: the run ends with operator authority, no retry
or new request is started, no returned tool call runs, and the prefix is not a
valid behavioral sample. An identical failure without a stop stays an
infrastructure fault.
"""

from __future__ import annotations

import copy
import json
import signal
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx
import pytest

from ora.agent.context.native import NativeContextPipeline
from ora.agent.providers.native import (
    AnthropicMessagesProvider, OpenAICompatibleProvider, ProviderClosed,
    translate_anthropic_request,
)
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.policy_factory import build_policy_session
from ora.orchestrator.spec import resolve_run_spec
from test_multi_round_session import live_exp01b_spec


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "synthetic-native-atomic-causal-cache-v1"


def _operator_stop(until: Callable[[], bool]) -> None:
    """Deliver the stop signal and wait for the interrupt thread to land."""

    signal.raise_signal(signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while not until():
        if time.monotonic() > deadline:
            raise AssertionError("operator interrupt did not reach the session")
        time.sleep(0.005)


def _reply(model_call_id: str, name: str = "close_gripper") -> dict[str, Any]:
    now = time.time()
    return {
        "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"{name}-after-stop", "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }],
        },
        "finish_reason": "tool_calls",
        "usage": {
            "input_tokens": 1000, "cached_input_tokens": 0,
            "output_tokens": 20, "reasoning_output_tokens": 0,
            "total_tokens": 1020,
        },
        "raw_events": [{"qualification": True}],
        "model_call_id": model_call_id,
        "started_unix_s": now, "finished_unix_s": now, "attempts": 1,
    }


class _Provider:
    """Scripted provider; ``mode`` selects what one request does."""

    protocol = "anthropic_messages"
    prepare_request = staticmethod(translate_anthropic_request)

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str,
    ) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError(
                "Cannot send a request, as the client has been closed."
            )
        self.requests.append(copy.deepcopy(dict(request)))
        if self.mode == "stop_during_stream":
            _operator_stop(lambda: self.closed)
            raise httpx.ReadError("[Errno 9] Bad file descriptor")
        if self.mode == "reply_after_stop":
            _operator_stop(lambda: self.closed)
            return _reply(model_call_id)
        if self.mode == "transport_failure":
            raise httpx.ReadError("[Errno 9] Bad file descriptor")
        return _reply(model_call_id)

    def close(self) -> None:
        self.closed = True


class _ClosedClientPolicy:
    """A policy runtime whose turn fails with the closed-client error."""

    def __init__(self, delegate: Any, provider: _Provider) -> None:
        self.delegate = delegate
        self.provider = provider
        self.capabilities = delegate.capabilities

    def act(self, turn: Any) -> Any:
        _operator_stop(lambda: self.provider.closed)
        raise RuntimeError(
            "Cannot send a request, as the client has been closed."
        )

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        self.delegate.interrupt(reason)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _plan() -> dict[str, Any]:
    spec = live_exp01b_spec(BASE_CONFIG)
    spec["experiment"] = "operator-stop-cancellation-test"
    spec["output_workspace"] = "outputs/operator-stop-cancellation-test"
    spec["milestone"] = "OPERATOR-STOP-QUALIFICATION"
    spec["runtime"]["terminal_actions"] = 1
    spec["budgets"] = {
        "max_robot_decisions": 2,
        "max_model_calls": 4,
        "max_sim_steps": 2,
        "wall_time_s": 30,
    }
    return resolve_run_spec(spec, root=ROOT)


def _run(
    tmp_path: Path, provider: _Provider, **kwargs: Any,
) -> tuple[dict[str, Any], Path]:
    plan = _plan()
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    previous = signal.getsignal(signal.SIGTERM)
    try:
        result = run_episode(
            run_root=run_root, plan=plan, run_id="operator-stop",
            provider_override=provider, **kwargs,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    return result, run_root


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def _assert_operator_stop(result: Mapping[str, Any], run_root: Path) -> None:
    assert result["state"] == "incomplete"
    assert result["termination"] == "operator_interrupt"
    assert result["termination_authority"] == "operator"
    assert result["behavioral_sample_valid"] is False
    assert result["official_success"] is None
    assert result["decisions"] == 0
    assert "error" not in result
    assert not (run_root / "native" / "runtime-fault.json").exists()
    events = _jsonl(run_root / "events.jsonl")
    kinds = {event.get("type") for event in events}
    assert "runtime_fault" not in kinds
    assert "trusted_fault" not in kinds
    assert "action_submitted" not in kinds
    decisions = [
        event["decision"] for event in events
        if event.get("type") == "termination_authority_decision"
    ]
    assert decisions and decisions[-1]["authority"] == "operator"
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert status["termination"] == "operator_interrupt"
    assert status["termination_authority"] == "operator"


def test_stop_during_provider_stream_is_an_operator_stop_without_retry(
    tmp_path: Path,
) -> None:
    provider = _Provider("stop_during_stream")
    result, run_root = _run(tmp_path, provider)

    _assert_operator_stop(result, run_root)
    assert len(provider.requests) == 1
    agent_events = _jsonl(run_root / "agent-events.jsonl")
    errors = [e for e in agent_events if e.get("type") == "provider_error"]
    assert len(errors) == 1
    assert errors[0]["payload"]["accepted"] is False
    assert sum(e.get("type") == "provider_request" for e in agent_events) == 1


def test_reply_completed_after_stop_runs_no_tool(tmp_path: Path) -> None:
    provider = _Provider("reply_after_stop")
    result, run_root = _run(tmp_path, provider)

    _assert_operator_stop(result, run_root)
    assert len(provider.requests) == 1
    agent_events = _jsonl(run_root / "agent-events.jsonl")
    kinds = [event.get("type") for event in agent_events]
    # The reply stays as evidence; none of its tool calls is dispatched.
    assert kinds.count("provider_response") == 1
    assert not any(str(kind).startswith("tool_") for kind in kinds)
    assert not (run_root / "control" / "submitted-action.json").exists()


def test_stop_during_projection_starts_no_provider_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider("reply")
    original = NativeContextPipeline.project

    def project_then_stop(self: Any, *args: Any, **kwargs: Any) -> Any:
        projected = original(self, *args, **kwargs)
        _operator_stop(lambda: provider.closed)
        return projected

    monkeypatch.setattr(NativeContextPipeline, "project", project_then_stop)
    result, run_root = _run(tmp_path, provider)

    _assert_operator_stop(result, run_root)
    assert provider.requests == []
    agent_events = _jsonl(run_root / "agent-events.jsonl")
    assert not any(e.get("type") == "provider_request" for e in agent_events)


def test_closed_client_error_after_stop_is_not_a_trusted_fault(
    tmp_path: Path,
) -> None:
    provider = _Provider("reply")

    def policy_factory(**kwargs: Any) -> _ClosedClientPolicy:
        return _ClosedClientPolicy(build_policy_session(**kwargs), provider)

    result, run_root = _run(
        tmp_path, provider, policy_session_factory=policy_factory,
    )

    _assert_operator_stop(result, run_root)
    cancelled = [
        event for event in _jsonl(run_root / "events.jsonl")
        if event.get("type") == "policy_turn_cancelled"
    ]
    assert len(cancelled) == 1
    assert cancelled[0]["error_type"] == "RuntimeError"
    assert "client has been closed" in cancelled[0]["error"]
    assert cancelled[0]["discarded_action"] is False


def test_identical_transport_failure_without_stop_stays_infrastructure(
    tmp_path: Path,
) -> None:
    provider = _Provider("transport_failure")
    result, run_root = _run(tmp_path, provider)

    assert result["state"] == "incomplete"
    assert result["termination"] == "trusted_fault"
    assert result["termination_authority"] == "infrastructure"
    assert (run_root / "native" / "runtime-fault.json").is_file()
    assert not any(
        event.get("type") == "policy_turn_cancelled"
        for event in _jsonl(run_root / "events.jsonl")
    )


_REQUEST = {
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 64,
    "stream": True,
}


@pytest.mark.parametrize(
    "provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider],
)
def test_provider_closed_mid_request_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, provider_type: Any,
) -> None:
    monkeypatch.setattr(
        "ora.agent.providers.native.time.sleep", lambda _s: None,
    )

    class Client:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.provider: Any = None

        def stream(self, *_args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs["content"])
            self.provider.close()
            raise httpx.ReadError("[Errno 9] Bad file descriptor")

    client = Client()
    provider = provider_type(
        "https://provider.invalid", "secret", client=client, max_retries=2,
    )
    client.provider = provider
    with pytest.raises(httpx.ReadError) as caught:
        provider.complete(request=_REQUEST, model_call_id="closed-mid-request")
    assert len(client.calls) == 1
    assert caught.value.provider_failure["attempts"] == 1

    with pytest.raises(ProviderClosed):
        provider.complete(request=_REQUEST, model_call_id="after-close")
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider],
)
def test_open_provider_still_retries_the_same_transport_error(
    monkeypatch: pytest.MonkeyPatch, provider_type: Any,
) -> None:
    monkeypatch.setattr(
        "ora.agent.providers.native.time.sleep", lambda _s: None,
    )

    class Client:
        calls: list[Any] = []

        def stream(self, *_args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs["content"])
            raise httpx.ReadError("[Errno 9] Bad file descriptor")

    client = Client()
    client.calls = []
    provider = provider_type(
        "https://provider.invalid", "secret", client=client, max_retries=2,
    )
    with pytest.raises(httpx.ReadError):
        provider.complete(request=_REQUEST, model_call_id="open-retry")
    assert len(client.calls) == 3


def test_proposal_returned_after_stop_is_discarded(tmp_path: Path) -> None:
    from ora.agent import ActionProposal, NoAction
    from ora.orchestrator.operator_stop import act_unless_stopped

    class Planner:
        def act(self, _turn: Any) -> ActionProposal:
            return ActionProposal(action={"action_id": "late"}, model_calls=2)

    events = tmp_path / "events.jsonl"
    stopped = act_unless_stopped(Planner(), None, lambda: True, events)
    assert isinstance(stopped, NoAction)
    assert stopped.reason == "operator_interrupt"
    assert stopped.model_calls == 2
    (record,) = _jsonl(events)
    assert record["type"] == "policy_turn_cancelled"
    assert record["discarded_action"] is True
    assert record["policy_outcome"] == "action"

    running = act_unless_stopped(Planner(), None, lambda: False, events)
    assert isinstance(running, ActionProposal)

    class Failing:
        def act(self, _turn: Any) -> Any:
            raise RuntimeError("transport failed")

    with pytest.raises(RuntimeError, match="transport failed"):
        act_unless_stopped(Failing(), None, lambda: False, events)
    assert len(_jsonl(events)) == 1
