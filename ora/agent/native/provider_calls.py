"""Persist failed transport attempts separately from accepted policy replies."""

from contextlib import contextmanager
from typing import Any, Iterator, Mapping


@contextmanager
def _provider_attempt_evidence(
    session: Any, request_event: Mapping[str, Any], step: int, model_call_id: str,
    *, deadline_monotonic: float | None = None,
) -> Iterator[None]:
    provider = session.provider
    if not hasattr(provider, "attempt_observer"):
        yield
        return
    previous = provider.attempt_observer
    binds_deadline = hasattr(provider, "request_deadline_monotonic")
    previous_deadline = getattr(provider, "request_deadline_monotonic", None)
    if binds_deadline:
        provider.request_deadline_monotonic = deadline_monotonic

    def record(failure: Mapping[str, Any]) -> None:
        # Evidence is persisted before backoff or terminal fault propagation.
        # None of these partial blocks enters the conversation/tool dispatcher.
        artifact = session.artifacts.put_json(
            dict(failure), kind="native_provider_failed_attempt",
        )
        session._append(
            "provider_attempt_failed", causal_parent_id=str(request_event["event_id"]),
            observation_step=step, model_call_id=model_call_id,
            visibility="derived", artifact_ref=artifact,
            payload={key: value for key, value in failure.items()
                     if key not in {"partial_events", "error_chain"}},
        )

    provider.attempt_observer = record
    try:
        yield
    finally:
        provider.attempt_observer = previous
        if binds_deadline:
            provider.request_deadline_monotonic = previous_deadline


def adopt_synthesized_tool_call_id(
    assistant: dict[str, Any], *, index: int, call_id: str,
) -> None:
    """Give one id-less provider tool call a stable id inside history.

    Tool results are paired to their call by id, so an assistant message that
    reached history without one has to adopt the id the dispatcher used.  This
    is the only canonical field the Harness may repair, and it stays sound only
    while no provider continuation is bound: the continuation is
    content-addressed over the canonical assistant message, and both shipped
    adapters assign call ids before they create one.  A provider that returns a
    continuation over an id-less call cannot be replayed at all, so fail closed
    instead of breaking the binding.
    """

    from ora.agent.contracts import PolicySessionError
    from ora.agent.providers.continuation import PROVIDER_CONTINUATION_MESSAGE_KEY

    if PROVIDER_CONTINUATION_MESSAGE_KEY in assistant:
        raise PolicySessionError(
            "provider continuation is bound to a tool call without a stable id"
        )
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or index >= len(calls):
        return
    call = calls[index]
    if isinstance(call, dict):
        call["id"] = call_id
