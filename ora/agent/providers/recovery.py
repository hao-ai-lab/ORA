"""One same-credential incident deadline, including active I/O and backoff."""
from __future__ import annotations

import asyncio
import json
import math
import ssl
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes
from .attempts import _StreamInterrupted
from .stream_failures import STREAM_BREAK_RETRIES, http_status_failure, is_stream_break


RECOVERY_POLICY = "same_credential_deadline_v1"
RECOVERY_WINDOW_S = 180.0
_RETRYABLE_TLS_REASONS = frozenset({"SSLV3_ALERT_BAD_RECORD_MAC"})
#: TLS states that report socket flow, never trust.
#:
#: ``SSLWantReadError`` and ``SSLWantWriteError`` mean the operation would
#: block and should be repeated; ``SSLZeroReturnError`` means the peer closed
#: the connection cleanly. None of the three can carry a certificate verdict,
#: so none of them is the trust failure the fail-closed branch below exists to
#: catch. They are skipped rather than made retryable on their own: a benign
#: flow state is not itself a reason to retry, it is only a reason not to
#: disqualify whatever transport error wraps it.
_NON_TRUST_TLS_STATES = (
    ssl.SSLWantReadError, ssl.SSLWantWriteError, ssl.SSLZeroReturnError,
)


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class ProviderRecoveryDeadline(PolicySessionError):
    def __init__(self, events=()):
        super().__init__("native provider recovery deadline exhausted; no reply was accepted")
        self.retryable = False
        self.partial_events = list(events)
        self.response_started = bool(events)


def transient_transport(exc: BaseException) -> bool:
    import httpx

    # HTTP transports can wrap certificate errors as ConnectError. Bad trust
    # configuration must fail closed, while a record-integrity alert before an
    # accepted response is safe to retry inside the existing incident window.
    #
    # Both judgements read the whole cause chain, because the provider raises
    # its faults wrapped. In a live failure the transport error that decides
    # retryability is never the outermost frame: `recovery_worker` classifies
    # the `PolicySessionError` it caught, so testing only that frame answered
    # False for every wrapped `httpx.TransportError` there has ever been.
    retryable_ssl = False
    transport = False
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError) and not isinstance(
            current, _NON_TRUST_TLS_STATES
        ):
            reason = str(getattr(current, "reason", "") or "").upper()
            message = str(current).upper()
            record_integrity_alert = any(
                reason == candidate or f"[SSL: {candidate}]" in message
                for candidate in _RETRYABLE_TLS_REASONS
            )
            if isinstance(current, ssl.SSLEOFError) or record_integrity_alert:
                retryable_ssl = True
            else:
                return False
        if isinstance(current, (httpx.TransportError, EOFError)):
            transport = True
        current = current.__cause__ or current.__context__
    return retryable_ssl or transport


@dataclass
class RecoveryWindow:
    """One logical request's incident: fixed deadline, counted stream breaks."""

    deadline: float
    retries_enabled: bool
    stream_breaks: int = 0

    @classmethod
    def for_provider(cls, provider):
        if provider.recovery_policy == "legacy_v1":
            return None
        if provider.recovery_policy != RECOVERY_POLICY:
            raise PolicySessionError("unsupported native provider recovery policy")
        # A slow provider (Kimi K3 at high effort streams single replies past
        # 180 s) carries its own incident window from the RunPlan.
        window = getattr(provider, "recovery_window_s", None) or RECOVERY_WINDOW_S
        deadline = time.monotonic() + window
        remaining_run = provider.request_deadline_monotonic
        if remaining_run is not None:
            if type(remaining_run) not in (float, int) or not math.isfinite(remaining_run):
                raise PolicySessionError("provider run deadline is invalid")
            deadline = min(deadline, remaining_run)
        return cls(deadline, not provider.preflight_probe and provider.max_retries > 0)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def delay(self, exc: Exception, attempt: int) -> float | None:
        if (not self.retries_enabled or isinstance(exc, ProviderRecoveryDeadline)
                or getattr(exc, "http_status", None) in {401, 402, 403}
                or not (getattr(exc, "retryable", False) or transient_transport(exc))):
            return None
        hint = getattr(exc, "retry_delay_s", None)
        delay = float(hint) if type(hint) in (float, int) and math.isfinite(hint) and hint > 0 else min(2 ** min(attempt - 1, 5), 20.0)
        # Do not start a known-too-early request when Retry-After exceeds the
        # remaining incident budget. Spend the remaining wait then terminate.
        return min(delay, self.remaining()) if self.remaining() > 0 else None

    def transport_delay(self, exc: Exception, attempt: int) -> float | None:
        """``delay`` for a transport-phase failure, bounding stream breaks.

        A break after response content is retried as a fresh request at most
        ``STREAM_BREAK_RETRIES`` times per logical call, inside the same
        deadline and backoff. The running count is stamped on the exception
        so the attempt record states how much of the budget was spent.
        """
        delay = self.delay(exc, attempt)
        if is_stream_break(exc):
            self.stream_breaks += 1
            if self.stream_breaks > STREAM_BREAK_RETRIES:
                delay = None
        setattr(exc, "stream_breaks", self.stream_breaks)
        return delay

    def pause(self, delay: float) -> None:
        time.sleep(min(delay, self.remaining()))
        if self.remaining() <= 0:
            raise ProviderRecoveryDeadline()


def _bounded_request_in_process(provider: Any, body: Mapping[str, Any], window: RecoveryWindow,
                                *, event_sink=None):
    """Use cancellable async I/O beneath the synchronous Provider contract.

    A read timeout alone restarts on each byte and cannot bound a trickling
    stream. wait_for encloses connection, headers, body and client cleanup;
    cancellation closes the attempt before another request may start. No
    detached thread can later publish a partial reply or execute tools.
    """
    import httpx

    events, first = [], None
    anthropic = provider.protocol == "anthropic_messages"
    url = provider._messages_url() if anthropic else provider.base_url + "/chat/completions"
    headers = {"content-type": "application/json"}
    if anthropic:
        headers["anthropic-version"] = provider.anthropic_version
    # Both protocols carry the run's affinity identity; the provider refused it
    # at construction unless its gateway is admitted for it.
    if getattr(provider, "session_affinity_id", None) is not None:
        headers["x-session-id"] = provider.session_affinity_id
    if anthropic and provider.auth_scheme == "x_api_key":
        headers["x-api-key"] = provider.api_key
    else:
        headers["authorization"] = "Bearer " + provider.api_key
    keys = getattr(provider, "_credential_ring", (provider.api_key,))

    async def read():
        nonlocal first
        timeout = min(provider.timeout_seconds, window.remaining())
        if timeout <= 0:
            raise ProviderRecoveryDeadline()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(10.0, timeout))) as client:
            async with client.stream("POST", url, headers=headers, content=canonical_bytes(body)) as stream:
                if stream.status_code >= 400:
                    detail = ""
                    # Authentication rejection is terminal at the headers;
                    # an unbounded error body must not turn it into an outage.
                    if stream.status_code not in {401, 402, 403}:
                        try:
                            raw_detail = await asyncio.wait_for(stream.aread(), min(2.0, window.remaining()))
                            detail = raw_detail.decode("utf-8", errors="replace")[-4000:]
                        except (asyncio.TimeoutError, httpx.TransportError):
                            detail = "error body unavailable"
                    for key in sorted(keys, key=len, reverse=True):
                        detail = detail.replace(key, "[REDACTED]")
                    exc = http_status_failure(stream.status_code, detail, base_url=provider.base_url)
                    hint = retry_after_seconds(stream.headers.get("retry-after"))
                    exc.retry_delay_s = hint if hint is not None else 20.0 if stream.status_code == 429 else None
                    raise exc
                if body.get("stream") is False:
                    value = json.loads(await stream.aread())
                    if not isinstance(value, Mapping):
                        raise PolicySessionError("native provider returned non-object JSON")
                    return [dict(value)], time.monotonic()
                announced, terminal = None, False
                try:
                    async for line in stream.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            if announced is not None or not line[6:].strip():
                                raise PolicySessionError("native provider SSE event lacks its data payload")
                            announced = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            raise PolicySessionError("native provider returned malformed SSE framing")
                        data = line[5:].strip()
                        if data == "[DONE]":
                            terminal = not anthropic
                            break
                        try:
                            parsed = json.loads(data)
                        except ValueError as exc:
                            raise PolicySessionError("native provider returned malformed SSE JSON") from exc
                        if not isinstance(parsed, Mapping) or announced is not None and parsed.get("type") != announced:
                            raise PolicySessionError("native provider SSE event does not match its payload")
                        announced = None
                        first = time.monotonic() if first is None else first
                        events.append(dict(parsed))
                        if event_sink is not None:
                            event_sink(dict(parsed))
                        if anthropic and parsed.get("type") in {"message_stop", "error"}:
                            terminal = True
                            break
                    if announced is not None or not terminal:
                        raise EOFError("native provider stream ended before protocol completion")
                except Exception as exc:
                    if events:
                        raise _StreamInterrupted(events, transient=transient_transport(exc)) from exc
                    raise
        return events, first

    async def run():
        if window.remaining() <= 0:
            raise ProviderRecoveryDeadline()
        try:
            return await asyncio.wait_for(read(), window.remaining())
        except asyncio.TimeoutError as exc:
            raise ProviderRecoveryDeadline(events) from exc
    return asyncio.run(run())


def bounded_request(provider: Any, body: Mapping[str, Any], window: RecoveryWindow):
    """Supervise one transport process, including DNS and client teardown.

    Async cancellation cannot forcibly cancel a platform DNS resolver thread;
    process ownership supplies that last wall-time bound. This worker has no
    tools or environment connection. Parent reaps it before retrying, and
    partial event frames remain failure evidence only.
    """
    if window.remaining() <= 0:
        raise ProviderRecoveryDeadline()
    binding = {"protocol": provider.protocol, "base_url": provider.base_url, "api_key": provider.api_key,
               "timeout_seconds": provider.timeout_seconds, "deadline": window.deadline,
               "session_affinity_id": getattr(provider, "session_affinity_id", None), "body": dict(body)}
    if provider.protocol == "anthropic_messages":
        binding.update(url=provider._messages_url(), auth_scheme=provider.auth_scheme,
            anthropic_version=provider.anthropic_version)
    command = [sys.executable, "-m", "ora.agent.providers.recovery_worker"]
    try:
        outcome = subprocess.run(command, input=canonical_bytes(binding), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=Path(__file__).resolve().parents[3], timeout=window.remaining(), check=False)
        output, timed_out = outcome.stdout, False
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and reaps its exact child before propagating.
        output, timed_out = exc.stdout or b"", True
    frames = []
    for line in output.splitlines():
        try:
            frame = json.loads(line)
        except (ValueError, UnicodeError):
            continue  # A killed process may leave an incomplete final frame.
        if isinstance(frame, dict):
            frames.append(frame)
    partial = [frame["event"] for frame in frames if frame.get("kind") == "event"]
    if timed_out or window.remaining() <= 0:
        raise ProviderRecoveryDeadline(partial)
    final = next((frame for frame in reversed(frames) if frame.get("kind") == "result"), None)
    if outcome.returncode != 0 or final is None:
        exc = PolicySessionError("native provider transport worker failed before completion")
        exc.retryable = True
        exc.partial_events, exc.response_started = partial, bool(partial)
        raise exc
    if final.get("ok") is True:
        return final["events"], final["first_event_at"]
    exc = PolicySessionError(str(final.get("message", "native provider transport failed")))
    for key in ("retryable", "http_status", "retry_delay_s", "response_started", "upstream_stream_break"):
        setattr(exc, key, final.get(key))
    exc.partial_events = final.get("partial_events", partial)
    exc.credential_echo = final.get("credential_echo") is True
    exc.remote_exception_type = final.get("exception_type")
    exc.remote_error_chain = final.get("error_chain", [])
    raise exc
