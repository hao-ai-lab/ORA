"""Transport-attempt diagnostics, without session or artifact ownership."""

from __future__ import annotations

import json
import hashlib
import time
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes

from .stream_failures import STREAM_BREAK_RETRIES


class _StreamInterrupted(PolicySessionError):
    """A complete response was not obtained; no caller has received tools."""

    def __init__(self, events: Sequence[Mapping[str, Any]], *, transient: bool):
        super().__init__("native provider stream failed before message_stop")
        self.partial_events = list(events)
        self.retryable = transient
        self.response_started = bool(self.partial_events)


#: The one message for any provider content that echoed a credential, whether
#: the response completed or broke mid-stream. It is never retried.
CREDENTIAL_ECHO_MESSAGE = (
    "native provider echoed credential material; response was discarded"
)


def echoes_credential(events: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
    """True when raw provider content contains a credential, before redaction."""

    serialized = json.dumps(list(events), ensure_ascii=False)
    return any(key and json.dumps(key, ensure_ascii=False)[1:-1] in serialized for key in keys)


def partial_credential_echo(exc: BaseException, keys: Sequence[str]) -> bool:
    """A broken attempt echoed a credential in the content it streamed.

    The transport worker redacts partial events before they cross to the
    parent, so a redacted copy can no longer show the echo; the worker checks
    the raw events first and carries the verdict as ``credential_echo``.
    """

    if getattr(exc, "credential_echo", False) is True:
        return True
    echoed = echoes_credential(getattr(exc, "partial_events", None) or [], keys)
    if echoed:
        setattr(exc, "credential_echo", True)
    return echoed


def credential_echo_failure(cause: BaseException | None = None) -> PolicySessionError:
    """Fail closed exactly as an echoing complete response does."""

    failure = PolicySessionError(CREDENTIAL_ECHO_MESSAGE)
    failure.retryable = False
    failure.credential_echo = True
    failure.response_started = True
    failure.__cause__ = cause
    return failure


def _failure_evidence(
    exc: Exception, *, keys: Sequence[str], events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chain = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        remote = getattr(current, "remote_error_chain", [])
        if isinstance(remote, list):
            for entry in remote:
                if (len(chain) < 8 and isinstance(entry, Mapping)
                        and isinstance(entry.get("type"), str)
                        and isinstance(entry.get("message"), str)):
                    chain.append({"type": entry["type"], "message": entry["message"]})
        current = current.__cause__ or current.__context__
    original = json.dumps({"error_chain": chain, "partial_events": list(events)}, ensure_ascii=False)
    redacted = original
    # Replace in the serialized form, including JSON property names and escaped
    # credentials. Never serialize request headers or process environment.
    for key in sorted(set(keys), key=len, reverse=True):
        if key:
            redacted = redacted.replace(json.dumps(key, ensure_ascii=False)[1:-1], "[REDACTED]")
    return {**json.loads(redacted), "credentials_redacted": redacted != original}


def record_attempt_failure(provider: Any, exc: Exception, *, body: Mapping[str, Any],
                           model_call_id: str, attempt: int, credential_index: int,
                           next_index: int | None, started: float, started_monotonic: float,
                           retry_delay: float, events: Sequence[Mapping[str, Any]], phase: str) -> None:
    if provider.attempt_observer is None:
        return
    provider.attempt_observer({
        "schema": "ora.provider-attempt-failure.v1",
        "model_call_id": model_call_id, "attempt": attempt,
        "request_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        "credential_index": credential_index, "next_credential_index": next_index,
        "http_status": getattr(exc, "http_status", None), "phase": phase,
        "started_unix_s": started, "finished_unix_s": time.time(),
        "duration_s": max(0.0, time.monotonic() - started_monotonic),
        "retry_scheduled": next_index is not None, "retry_delay_s": retry_delay,
        "partial_event_count": len(events), "tool_execution_started": False,
        **_failure_evidence(exc, keys=getattr(provider, "_credential_ring", (provider.api_key,)), events=events),
        **({"recovery_policy": provider.recovery_policy, "stream_break_retry_limit": STREAM_BREAK_RETRIES}
           if provider.recovery_policy != "legacy_v1" else {}),
        # Transport-phase breaks after content so far in this logical call.
        **({"stream_breaks": exc.stream_breaks} if type(getattr(exc, "stream_breaks", None)) is int else {}),
        # A gateway-reported upstream break: no partial events, yet a break.
        **({"upstream_stream_break": True} if getattr(exc, "upstream_stream_break", False) is True else {}),
        # Worker-redacted partials no longer show the echo they carried.
        **({"credential_echo": True, "credentials_redacted": True}
           if getattr(exc, "credential_echo", False) is True else {}),
    })
