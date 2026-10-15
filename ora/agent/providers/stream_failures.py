# Copyright contributors to the Robot Harness Runtime project.
"""How a broken provider stream is classified for retry.

``response_started`` is a fact about how far a response got, not a verdict
about whether asking again is sound. Treating the two as the same made a
snapped socket permanently fatal: three runs died inside seventy-five minutes
on transport faults after 53, 51 and 18 model calls, none retried, each
discarding its episode.
"""

from __future__ import annotations

import json
from typing import Mapping
from urllib.parse import urlsplit

from ora.agent.contracts import PolicySessionError


#: Fresh requests granted to one logical model call after transport breaks
#: that arrived with response content, under ``same_credential_deadline_v1``.
#: The fourth such break is terminal even with deadline left: every broken
#: attempt already streamed billable output that is discarded (one recorded
#: RoboDojo attempt streamed 7,301 events for 35 s before its socket
#: dropped). Retries still share the 180 s window and its backoff; this only
#: bounds how many times a partly answered request is asked again. Recorded
#: as ``stream_break_retry_limit`` on every deadline-policy attempt failure.
STREAM_BREAK_RETRIES = 3


#: What a loopback gateway relaying a streamed upstream (CLIProxyAPI in front
#: of the Codex Responses API) puts in ``error.message`` when that upstream
#: stream closed before ``response.completed``. It answers HTTP 408 with
#: ``type: invalid_request_error`` before relaying any event, so neither the
#: status nor the type says that the upstream had begun answering; only this
#: text does. A recorded fold_clothes run died on one as a fatal fault.
GATEWAY_UPSTREAM_STREAM_BREAK = "stream disconnected before completion"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def gateway_upstream_stream_break(status: int, detail: str, base_url: str) -> bool:
    """A loopback gateway's HTTP 408 reporting that its upstream stream broke.

    Asking again is as sound as after a stream that broke in front of us:
    nothing reached the policy and the gateway discarded the partial upstream
    output. Any other 408, a 408 from a remote endpoint, and an error body that
    is truncated or does not carry this message in ``error.message`` stay
    non-retryable.
    """

    if status != 408 or urlsplit(base_url).hostname not in _LOOPBACK_HOSTS:
        return False
    try:
        error = json.loads(detail).get("error")
    except (ValueError, AttributeError, RecursionError):
        return False
    message = error.get("message") if isinstance(error, Mapping) else None
    return isinstance(message, str) and GATEWAY_UPSTREAM_STREAM_BREAK in message


def http_status_failure(status: int, detail: str, *, base_url: str) -> PolicySessionError:
    """One HTTP error answer, classified for a same-request retry.

    HTTP 429 and 5xx are transient. A gateway-reported upstream stream break is
    transient too and is marked ``upstream_stream_break`` so that it spends the
    ``STREAM_BREAK_RETRIES`` budget like a break that arrived with content.
    """

    failure = PolicySessionError(f"native provider HTTP {status}: {detail}".strip())
    failure.http_status = status
    failure.upstream_stream_break = gateway_upstream_stream_break(status, detail, base_url)
    failure.retryable = status == 429 or 500 <= status < 600 or failure.upstream_stream_break
    return failure


def is_stream_break(exc: BaseException) -> bool:
    """A transport-phase failure after response content had arrived.

    Upstream content counts: a gateway that reports its upstream stream broke
    had received output it then discarded, even though none reached us.
    """

    return bool(getattr(exc, "response_started", False) or getattr(exc, "partial_events", None)
                or getattr(exc, "upstream_stream_break", False) is True)


def stream_failure_after_content(cause: BaseException) -> PolicySessionError:
    """Carry a broken stream's verdict past the wrapper that reports it.

    ``transient_transport`` would have admitted the bare fault -- a
    ``RemoteProtocolError`` is an ``httpx.TransportError`` -- but it inspects
    the exception it is handed, and the wrapper is not one. So the wrapper has
    to carry the verdict rather than replace it.

    Asking again is sound because nothing was committed. The partial events are
    scanned for credentials and recorded as transport evidence; they never
    reach the policy, so a second answer replaces nothing. The sibling path in
    ``recovery.py`` already retries a transport worker that died mid-response
    with its partials attached, and this is the same fault one layer down.

    A stream the adapter itself rejected -- malformed framing, an announced
    event without its payload, non-object JSON -- stays fatal. That is not a
    transport fault, and asking the same question again would get the same
    malformed answer.
    """

    from .recovery import transient_transport

    retryable = transient_transport(cause)
    # The text is fixed before the retry decision, so it names the class, not
    # an outcome: the terminal attempt of an exhausted budget raises it too.
    failure = PolicySessionError(
        "native provider stream failed after response content; "
        + ("retryable transport fault" if retryable else "not retried")
    )
    setattr(failure, "retryable", retryable)
    setattr(failure, "response_started", True)
    return failure


__all__ = [
    "GATEWAY_UPSTREAM_STREAM_BREAK", "STREAM_BREAK_RETRIES", "gateway_upstream_stream_break",
    "http_status_failure", "is_stream_break", "stream_failure_after_content",
]
