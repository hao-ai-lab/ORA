"""Deadline and exact retry evidence against local fakes, never live providers."""
from __future__ import annotations

import json
import ssl
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from ora.agent.contracts import PolicySessionError
from ora.agent.providers import native, recovery
from ora.agent.providers.native import AnthropicMessagesProvider, OpenAICompatibleProvider
from ora.agent.providers.stream_failures import STREAM_BREAK_RETRIES


def request(stream=False):
    return {"model": "offline", "max_tokens": 64, "stream": stream,
            "messages": [{"role": "user", "content": "Act from current public evidence."}]}


def success(protocol):
    if protocol == "anthropic":
        return {"type": "message", "id": "accepted", "role": "assistant", "model": "offline",
            "content": [{"type": "text", "text": "accepted"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1}}
    return {"id": "accepted", "choices": [{"message": {"role": "assistant", "content": "accepted"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def provider(protocol, url="https://offline.invalid", **kwargs):
    cls = AnthropicMessagesProvider if protocol == "anthropic" else OpenAICompatibleProvider
    if protocol == "anthropic":
        kwargs["standby_api_keys"] = ("offline-backup-do-not-use",)
        kwargs["session_affinity_id"] = "same-run-session"
    return cls(url, "offline-primary-key", recovery_policy=recovery.RECOVERY_POLICY, **kwargs)


def test_bad_record_mac_is_the_only_retryable_non_eof_tls_error():
    bad_record = ssl.SSLError(
        1,
        "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac",
    )
    wrapped = PolicySessionError("provider transport failed")
    wrapped.__cause__ = bad_record
    assert recovery.transient_transport(wrapped) is True

    assert recovery.transient_transport(ssl.SSLError("generic TLS error")) is False
    assert recovery.transient_transport(
        ssl.SSLCertVerificationError("certificate verification failed")
    ) is False


def test_a_would_block_tls_state_does_not_disqualify_the_transport_error_wrapping_it():
    """A non-fatal TLS flow state must not fail a retryable transport error closed.

    Four of thirteen EXP10A output-ceiling cells died on attempt one of a
    128-second network incident. Each carried the same chain: an
    ``httpx.ConnectTimeout`` -- the exact class the final line of
    ``transient_transport`` retries -- whose ``__context__`` reached a
    ``ssl.SSLWantReadError`` left behind when the deadline cancelled the
    handshake. The cause walk returned False at that frame and never reached
    the ``TransportError`` check, so no retry was scheduled.

    ``SSLWantReadError`` and its siblings say the socket would block or closed
    cleanly. Neither can report a bad certificate, so neither is the trust
    failure the fail-closed branch exists for.
    """

    for benign in (
        ssl.SSLWantReadError("The operation did not complete (read)"),
        ssl.SSLWantWriteError("The operation did not complete (write)"),
        ssl.SSLZeroReturnError("TLS/SSL connection has been closed"),
    ):
        cancelled = TimeoutError()
        cancelled.__context__ = benign
        timeout = httpx.ConnectTimeout("connect timed out")
        timeout.__cause__ = cancelled
        wrapped = PolicySessionError("provider transport failed")
        wrapped.__cause__ = timeout
        assert recovery.transient_transport(wrapped) is True, type(benign).__name__

    # The fail-closed branch still holds when a real trust failure sits in the
    # same position, so this is not a blanket exemption for SSL causes.
    verification = ssl.SSLCertVerificationError("certificate verify failed")
    timeout = httpx.ConnectTimeout("connect timed out")
    timeout.__cause__ = verification
    assert recovery.transient_transport(timeout) is False


def test_a_wrapped_transport_error_is_retryable_at_any_depth_of_the_cause_chain():
    """The frame that decides retryability is never the outermost one live.

    ``recovery_worker`` classifies the exception it caught, and the provider
    raises transport faults wrapped in ``PolicySessionError``. Testing only
    that outermost frame answered False for every wrapped
    ``httpx.TransportError``, so a connect timeout, a DNS failure and a read
    timeout were all treated as permanent.
    """

    for fault in (
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("name resolution failed"),
        httpx.ReadTimeout("read timed out"),
    ):
        wrapped = PolicySessionError("provider transport failed")
        wrapped.__cause__ = fault
        assert recovery.transient_transport(wrapped) is True, type(fault).__name__

    # A trust failure anywhere under the transport error still fails closed,
    # so depth-awareness did not weaken the certificate gate.
    verification = httpx.ConnectError("certificate problem")
    verification.__cause__ = ssl.SSLCertVerificationError("certificate verify failed")
    wrapped = PolicySessionError("provider transport failed")
    wrapped.__cause__ = verification
    assert recovery.transient_transport(wrapped) is False

    # An unrelated wrapped fault stays permanent: this is not "retry anything".
    wrapped = PolicySessionError("provider transport failed")
    wrapped.__cause__ = ValueError("malformed provider payload")
    assert recovery.transient_transport(wrapped) is False


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_recovery_window_counts_requests_and_backoff_without_switching(monkeypatch, protocol):
    clock = [1000.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recovery.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    attempts, failures = [], []
    p = provider(protocol, attempt_observer=failures.append)
    def fail(bound, body, window):
        attempts.append((clock[0], bound.api_key, bound.base_url, json.dumps(body, sort_keys=True)))
        # A non-instant connection failure consumes part of this same incident.
        clock[0] += min(11., window.remaining())
        if window.remaining() == 0:
            raise recovery.ProviderRecoveryDeadline()
        exc = PolicySessionError("injected 503")
        exc.http_status, exc.retryable = 503, True
        raise exc
    monkeypatch.setattr(native, "bounded_request", fail)
    with pytest.raises(recovery.ProviderRecoveryDeadline) as raised:
        p.complete(request=request(), model_call_id="one-logical-call")
    assert clock[0] == 1180.0
    assert len(attempts) > 3
    assert len(failures) == len(attempts)
    assert len({row[3] for row in attempts}) == 1
    assert {row[1] for row in attempts} == {"offline-primary-key"}
    assert {row[2] for row in attempts} == {"https://offline.invalid"}
    assert raised.value.provider_failure["attempts"] == len(attempts)
    assert all(row["tool_execution_started"] is False for row in failures)
    assert all(row["next_credential_index"] in (None, 0) for row in failures)
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("status", [401, 402, 403, 429, 503])
@pytest.mark.parametrize("one_shot", [False, True])
def test_auth_fails_immediately_and_one_shot_never_retries(monkeypatch, protocol, status, one_shot):
    calls, failures = [], []
    p = provider(protocol, max_retries=0 if one_shot else 2, attempt_observer=failures.append)
    def respond(bound, body, window):
        calls.append(bound.api_key)
        if len(calls) == 2:
            return [success(protocol)], 0.
        exc = PolicySessionError("offline rejection")
        exc.http_status, exc.retryable = status, status in {429, 503}
        exc.retry_delay_s = 20. if status == 429 else None
        raise exc
    pauses = []
    monkeypatch.setattr(native, "bounded_request", respond)
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, delay: pauses.append(delay))
    if status < 429 or one_shot:
        with pytest.raises(PolicySessionError):
            p.complete(request=request(), model_call_id="same-call")
        assert len(calls) == 1 and not pauses
    else:
        result = p.complete(request=request(), model_call_id="same-call")
        assert result["message"]["content"] == "accepted" and len(calls) == 2
    assert set(calls) == {"offline-primary-key"}
    if protocol == "anthropic":
        assert p.credential_rotations() == []
    p.close()


def test_run_budget_is_an_absolute_deadline_across_read_only_calls(monkeypatch):
    from ora.agent.native.provider_calls import _provider_attempt_evidence
    clock = [40.]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    p = provider("anthropic")
    session = SimpleNamespace(provider=p)
    with _provider_attempt_evidence(session, {"event_id": "event"}, 0, "call", deadline_monotonic=52.):
        assert recovery.RecoveryWindow.for_provider(p).remaining() == 12.
        clock[0] += 7
        assert recovery.RecoveryWindow.for_provider(p).remaining() == 5.
    assert p.request_deadline_monotonic is None
    p.close()


class Server(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_args):
        pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        self.server.requests.append((self.path, dict(self.headers), body))
        mode = self.server.mode
        if mode == "headers":
            self.server.stopped.wait(10)
            return
        if mode in {"auth", "auth_body_stall"}:
            data = b'{"error":"offline-primary-key"}'
            self.send_response(401)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            if mode == "auth_body_stall":
                self.server.stopped.wait(10)
                return
            self.wfile.write(data)
            return
        if mode == "gateway_408" and len(self.server.requests) <= self.server.gateway_408:
            data = json.dumps(self.server.gateway_408_body).encode()
            self.send_response(408)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if mode == "gateway_408":
            mode = "unary"
        if mode == "recover_partial":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("connection", "close")
            self.end_headers()
            self.close_connection = True
            protocol = self.server.provider_protocol
            first = len(self.server.requests) == 1
            if protocol == "anthropic":
                events = [
                    {"type": "message_start", "message": {"id": "failed" if first else "accepted", "usage": {"input_tokens": 1}}},
                    {"type": "content_block_start", "index": 0, "content_block": {
                        "type": "tool_use", "id": "never-dispatch" if first else "accepted-tool", "name": "move_pose", "input": {}}},
                ]
                if not first:
                    events += [{"type": "content_block_stop", "index": 0},
                               {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
                               {"type": "message_stop"}]
            else:
                events = [{"choices": [{"delta": {"tool_calls": [{"index": 0,
                    "id": "never-dispatch" if first else "accepted-tool", "type": "function",
                    "function": {"name": "move_pose", "arguments": "{" if first else "{}"}}]},
                    "finish_reason": None if first else "tool_calls"}]}]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            if protocol == "openai" and not first:
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if mode == "unary":
            data = json.dumps(success(self.server.provider_protocol)).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        try:
            if mode == "partial":
                events = [
                    {"type": "message_start", "message": {"id": "partial", "usage": {"input_tokens": 1}}},
                    {"type": "content_block_start", "index": 0, "content_block": {
                        "type": "tool_use", "id": "never-dispatch", "name": "move_pose", "input": {}}},
                ]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
            # A byte stream with no new events cannot extend the total budget.
            while not self.server.stopped.wait(.01):
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def server():
    host = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    host.requests, host.stopped = [], threading.Event()
    thread = threading.Thread(target=host.serve_forever, daemon=True)
    thread.start()
    yield host
    host.stopped.set()
    host.shutdown()
    host.server_close()
    thread.join()


@pytest.mark.parametrize("mode", ["headers", "trickle", "partial"])
def test_real_transport_process_stops_and_reaps_at_total_deadline(server, mode):
    server.mode = mode
    failures = []
    p = provider("anthropic", f"http://127.0.0.1:{server.server_port}", attempt_observer=failures.append)
    p.request_deadline_monotonic = time.monotonic() + 1.5
    started = time.monotonic()
    with pytest.raises(recovery.ProviderRecoveryDeadline):
        p.complete(request=request(stream=True), model_call_id="one-call")
    assert time.monotonic() - started < 2.5
    assert len(server.requests) == 1
    assert len(failures) == 1 and failures[0]["retry_scheduled"] is False
    assert failures[0]["partial_event_count"] == (2 if mode == "partial" else 0)
    assert p.api_key == "offline-primary-key" and p.credential_rotations() == []
    # The supervised child was joined before complete returned, not left behind
    # to create a second request or accept late tool content.
    import subprocess
    children = subprocess.run(["ps", "--ppid", str(__import__("os").getpid()), "-o", "args="], capture_output=True, text=True)
    assert "ora.agent.providers.recovery_worker" not in children.stdout
    p.close()


def test_real_worker_keeps_same_auth_and_redacts_failure(server):
    server.mode = "auth"
    failures = []
    p = provider("anthropic", f"http://127.0.0.1:{server.server_port}", attempt_observer=failures.append)
    with pytest.raises(PolicySessionError):
        p.complete(request=request(), model_call_id="one-call")
    assert len(server.requests) == 1
    headers = {k.lower(): v for k, v in server.requests[0][1].items()}
    assert headers["x-api-key"] == "offline-primary-key"
    assert headers["x-session-id"] == "same-run-session"
    assert "offline-primary-key" not in json.dumps(failures)
    assert "offline-backup-do-not-use" not in json.dumps(failures)
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_auth_headers_fail_without_waiting_for_an_error_body(server, protocol):
    server.mode = "auth_body_stall"
    p = provider(protocol, f"http://127.0.0.1:{server.server_port}")
    started = time.monotonic()
    with pytest.raises(PolicySessionError) as raised:
        p.complete(request=request(), model_call_id="auth-failure")
    assert raised.value.http_status == 401
    assert time.monotonic()-started < 3
    assert len(server.requests) == 1
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_partial_stream_then_complete_replays_exact_wire_and_accepts_only_final_tool(server, protocol, monkeypatch):
    server.mode, server.provider_protocol = "recover_partial", protocol
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda self, delay: None)
    failures = []
    p = provider(protocol, f"http://127.0.0.1:{server.server_port}", attempt_observer=failures.append)
    body = request(stream=True)
    body["tools"] = [{"type": "function", "function": {"name": "move_pose", "description": "fixture",
                     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]
    result = p.complete(request=body, model_call_id="one-logical-model-call")
    assert result["attempts"] == 2 and len(server.requests) == 2
    assert server.requests[0][2] == server.requests[1][2]
    assert server.requests[0][0] == server.requests[1][0]
    headers = [{k.lower(): v for k, v in row[1].items()} for row in server.requests]
    auth = "x-api-key" if protocol == "anthropic" else "authorization"
    assert headers[0][auth] == headers[1][auth]
    assert [call["id"] for call in result["message"]["tool_calls"]] == ["accepted-tool"]
    assert "never-dispatch" not in json.dumps(result)
    assert len(failures) == 1 and failures[0]["partial_event_count"] > 0
    assert any(error["type"] == "EOFError" for error in failures[0]["error_chain"])
    assert failures[0]["tool_execution_started"] is False
    if protocol == "anthropic":
        assert p.credential_rotations() == []
        assert headers[0]["x-session-id"] == headers[1]["x-session-id"]
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_unary_worker_timing_is_monotonic_not_a_zero_timestamp(server, protocol):
    server.mode, server.provider_protocol = "unary", protocol
    p = provider(protocol, f"http://127.0.0.1:{server.server_port}")
    result = p.complete(request=request(), model_call_id="unary-call")
    assert 0 <= result["ttft_s"] <= result["finished_unix_s"]-result["started_unix_s"]+.01
    p.close()


@pytest.mark.parametrize("affinity", [None, "same-run-session"])
def test_real_worker_sends_openai_session_affinity_only_when_bound(server, affinity):
    """The supervised transport worker carries X-Session-ID for Chat Completions too.

    Every RoboDojo request goes through this worker (same-credential recovery is
    the RoboDojo default), so a header set only on the in-process path would
    never reach a loopback gateway.
    """
    server.mode, server.provider_protocol = "unary", "openai"
    kwargs = {} if affinity is None else {"session_affinity_id": affinity}
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", **kwargs)
    p.complete(request=request(), model_call_id="unary-call")
    p.complete(request=request(), model_call_id="unary-call-2")
    headers = [{k.lower(): v for k, v in row[1].items()} for row in server.requests]
    assert [row.get("x-session-id") for row in headers] == [affinity, affinity]
    assert all(row["authorization"] == "Bearer offline-primary-key" for row in headers)
    p.close()


#: The exact body the loopback CLIProxyAPI Codex gateway answered with HTTP 408
#: in EXP11A g016 fold_clothes (2026-10-02), killing the run as a fatal fault.
GATEWAY_UPSTREAM_BREAK = {"error": {
    "message": ("stream error: stream disconnected before completion: "
                "stream closed before response.completed"),
    "type": "invalid_request_error",
}}


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("breaks", [1, STREAM_BREAK_RETRIES + 1])
def test_real_worker_retries_a_loopback_gateway_upstream_stream_break_within_the_bound(
        server, monkeypatch, protocol, breaks):
    server.mode, server.provider_protocol = "gateway_408", protocol
    server.gateway_408, server.gateway_408_body = breaks, GATEWAY_UPSTREAM_BREAK
    pauses, failures = [], []
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, delay: pauses.append(delay))
    p = provider(protocol, f"http://127.0.0.1:{server.server_port}", attempt_observer=failures.append)
    if breaks > STREAM_BREAK_RETRIES:
        # Each such break discarded upstream output, so it spends the same budget.
        with pytest.raises(PolicySessionError) as raised:
            p.complete(request=request(), model_call_id="gateway-break")
        assert raised.value.http_status == 408
        assert len(server.requests) == STREAM_BREAK_RETRIES + 1
    else:
        result = p.complete(request=request(), model_call_id="gateway-break")
        assert result["attempts"] == 2 and result["message"]["content"] in ("accepted", [
            {"type": "text", "text": "accepted"}])
        assert len(server.requests) == 2
    assert len({row[2] for row in server.requests}) == 1, "the same request is asked again"
    assert [row["http_status"] for row in failures] == [408] * len(failures)
    assert [row["stream_breaks"] for row in failures] == list(range(1, len(failures) + 1))
    assert all(row["upstream_stream_break"] is True for row in failures)
    assert all(row["partial_event_count"] == 0 for row in failures)
    assert [row["retry_scheduled"] for row in failures] == (
        [True] if breaks == 1 else [True] * STREAM_BREAK_RETRIES + [False])
    assert pauses == [1, 2, 4][:min(breaks, STREAM_BREAK_RETRIES)]
    p.close()


@pytest.mark.parametrize("body", [
    {"error": {"message": "Request Timeout", "type": "invalid_request_error"}},
    {"error": "stream disconnected before completion"},
    {"message": "stream disconnected before completion"},
])
def test_any_other_408_from_the_gateway_stays_fatal(server, body):
    server.mode, server.provider_protocol = "gateway_408", "openai"
    server.gateway_408, server.gateway_408_body = 1, body
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as raised:
        p.complete(request=request(), model_call_id="plain-408")
    assert raised.value.http_status == 408 and len(server.requests) == 1
    assert failures[0]["retry_scheduled"] is False and "upstream_stream_break" not in failures[0]
    p.close()


def test_only_a_loopback_gateway_408_naming_the_upstream_break_is_a_stream_break():
    from ora.agent.providers.stream_failures import (
        gateway_upstream_stream_break, http_status_failure, is_stream_break,
    )
    detail = json.dumps(GATEWAY_UPSTREAM_BREAK)
    for url in ("http://127.0.0.1:8319/v1", "http://localhost:8319", "http://[::1]:8319/v1"):
        assert gateway_upstream_stream_break(408, detail, url) is True
    assert gateway_upstream_stream_break(408, detail, "https://api.openai.com/v1") is False
    assert gateway_upstream_stream_break(400, detail, "http://127.0.0.1:8319/v1") is False
    assert gateway_upstream_stream_break(408, detail[:-1], "http://127.0.0.1:8319/v1") is False
    assert gateway_upstream_stream_break(408, "[]", "http://127.0.0.1:8319/v1") is False
    failure = http_status_failure(408, detail, base_url="http://127.0.0.1:8319/v1")
    assert failure.retryable is True and failure.http_status == 408 and is_stream_break(failure)
    for status, retryable in ((408, False), (429, True), (500, True), (503, True), (400, False)):
        plain = http_status_failure(status, "{}", base_url="http://127.0.0.1:8319/v1")
        assert plain.retryable is retryable and not is_stream_break(plain)
        assert str(plain) == f"native provider HTTP {status}: {{}}"


class _Gateway408Client:
    """In-process Chat Completions client: one gateway 408, then an answer."""

    def __init__(self):
        self.calls = []

    def stream(self, *_args, **kwargs):
        self.calls.append(kwargs["content"])
        first = len(self.calls) == 1

        class Response:
            status_code = 408 if first else 200
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def read(self):
                return json.dumps(GATEWAY_UPSTREAM_BREAK if first else success("openai")).encode()
        return Response()

    def close(self):
        return None


def test_legacy_openai_path_retries_the_same_gateway_upstream_stream_break(monkeypatch):
    monkeypatch.setattr(native.time, "sleep", lambda _delay: None)
    client, failures = _Gateway408Client(), []
    p = OpenAICompatibleProvider("http://127.0.0.1:8319/v1", "offline-primary-key", client=client,
                                 attempt_observer=failures.append)
    result = p.complete(request=request(), model_call_id="legacy-gateway-break")
    assert result["attempts"] == 2 and len(set(client.calls)) == 1 == len(failures)
    assert failures[0]["http_status"] == 408 and failures[0]["retry_scheduled"] is True
    p.close()


def test_openai_session_affinity_is_refused_on_plain_http_off_loopback():
    with pytest.raises(PolicySessionError, match="only for a loopback gateway or an HTTPS endpoint"):
        provider("openai", "http://offline.invalid", session_affinity_id="same-run-session")


def test_retry_after_http_date_is_bounded_by_incident_time(monkeypatch):
    monkeypatch.setattr(recovery.time, "time", lambda: 0)
    assert recovery.retry_after_seconds("Thu, 01 Jan 1970 00:00:30 GMT") == 30
    assert recovery.retry_after_seconds("25") == 25
    assert recovery.retry_after_seconds("NaN") is None
    assert recovery.retry_after_seconds("not a date") is None


def test_openai_parse_failure_records_attempt_without_executing_or_retrying(monkeypatch):
    failures = []
    p = provider("openai", attempt_observer=failures.append)
    monkeypatch.setattr(native, "bounded_request", lambda *_args: ([{"choices": [{"delta": {"content": "partial"}}]}], time.monotonic()))
    with pytest.raises(PolicySessionError) as raised:
        p.complete(request=request(stream=True), model_call_id="malformed")
    assert raised.value.provider_failure["attempts"] == 1
    assert len(failures) == 1 and failures[0]["phase"] == "parse"
    assert failures[0]["retry_scheduled"] is False and failures[0]["tool_execution_started"] is False
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_preflight_is_one_shot_even_when_the_deadline_policy_is_selected(monkeypatch, protocol):
    calls = []
    p = provider(protocol, preflight_probe=True, max_retries=5)
    def fail(*_args):
        calls.append(True)
        exc = PolicySessionError("temporary HTTP 503")
        exc.retryable, exc.http_status = True, 503
        raise exc
    monkeypatch.setattr(native, "bounded_request", fail)
    with pytest.raises(PolicySessionError):
        p.complete(request=request(), model_call_id="preflight")
    assert len(calls) == 1
    p.close()


def test_actual_native_read_tool_continuation_does_not_reset_round_deadline(tmp_path, monkeypatch):
    from test_native_harness import (
        AGENT_RULES, PROMPT, assistant_tool, publish_native_boundary, usage,
    )
    from ora.agent.native.session import NativePlanner
    from ora.agent.providers.native import FakeProvider
    clock = [40.]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    pointer, workspace, tools = publish_native_boundary(tmp_path)
    pointer["remaining_budget"]["wall_time_s"] = 12.
    deadlines = []

    class DeadlineProvider(FakeProvider):
        recovery_policy = recovery.RECOVERY_POLICY
        request_deadline_monotonic = None
        attempt_observer = None
        max_retries = 2
        preflight_probe = False

        def complete(self, **kwargs):
            deadlines.append(recovery.RecoveryWindow.for_provider(self).remaining())
            clock[0] += 7.
            return super().complete(**kwargs)

    p = DeadlineProvider([
        {"message": assistant_tool("read", "manual.read", {"section": "overview"}), "usage": usage(100)},
        {"message": assistant_tool("one-proposal", "robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": .2}), "usage": usage(110)},
    ])
    planner = NativePlanner(tmp_path, "deadline-continuation", tools, p,
                            AGENT_RULES, PROMPT, "fake", "native_minimal_v1", 32768,
                            max_output_tokens=2048)
    try:
        planner.run_observation(task_instruction="Read then propose once.", context=pointer,
                                image_path=workspace / pointer["main_image"])
        assert deadlines == [12., 5.]
        assert p.request_deadline_monotonic is None
    finally:
        planner.close()


# --- Transport breaks after response content ---------------------------------
#
# EXP11A RoboDojo run 1acf0b36b8d9 lost model-call-000086 to three consecutive
# `RemoteProtocolError: incomplete chunked read` breaks after 7301, 2 and 735
# streamed events. A partly answered request is asked again as a fresh request
# at most STREAM_BREAK_RETRIES times, inside the unchanged deadline; every
# broken attempt is recorded and none reaches the policy.

_INCOMPLETE_CHUNKED = ("peer closed connection without sending complete message "
                       "body (incomplete chunked read)")


def tool_request():
    body = request(stream=True)
    body["tools"] = [{"type": "function", "function": {"name": "move_pose", "description": "fixture",
                     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]
    return body


def partial_events(protocol):
    if protocol == "anthropic":
        return [{"type": "message_start", "message": {"id": "discarded", "usage": {"input_tokens": 1}}},
                {"type": "content_block_start", "index": 0, "content_block": {
                    "type": "tool_use", "id": "discarded", "name": "move_pose", "input": {}}}]
    return [{"choices": [{"delta": {"role": "assistant", "reasoning_content": "H"}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "discarded", "type": "function",
             "function": {"name": "move_pose", "arguments": "{"}}]}, "finish_reason": None}]}]


def answered_events(protocol):
    if protocol == "anthropic":
        return [{"type": "message_start", "message": {"id": "accepted", "usage": {"input_tokens": 1}}},
                {"type": "content_block_start", "index": 0, "content_block": {
                    "type": "tool_use", "id": "accepted-tool", "name": "move_pose", "input": {}}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"}]
    return [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "accepted-tool", "type": "function",
             "function": {"name": "move_pose", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}]


def worker_break(protocol, *, retryable=True):
    """What bounded_request raises when its worker saw the stream break."""
    exc = PolicySessionError("native provider stream failed before message_stop")
    exc.retryable, exc.http_status, exc.retry_delay_s = retryable, None, None
    exc.response_started, exc.partial_events = True, partial_events(protocol)
    exc.remote_exception_type = "_StreamInterrupted"
    exc.remote_error_chain = [{"type": "RemoteProtocolError", "message": _INCOMPLETE_CHUNKED}]
    return exc


def scripted_transport(monkeypatch, protocol, script):
    """Replace the supervised child with a script of break/http503/answer."""
    bodies, pauses = [], []
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, delay: pauses.append(delay))

    def respond(_bound, body, _window):
        bodies.append(json.dumps(body, sort_keys=True))
        step = script[min(len(bodies), len(script)) - 1]
        if step == "break":
            raise worker_break(protocol)
        if step == "malformed":
            raise worker_break(protocol, retryable=False)
        if step == "http503":
            exc = PolicySessionError("injected 503")
            exc.http_status, exc.retryable = 503, True
            raise exc
        return answered_events(protocol), time.monotonic()
    monkeypatch.setattr(native, "bounded_request", respond)
    return bodies, pauses


def test_the_stream_break_retry_bound_is_three():
    assert STREAM_BREAK_RETRIES == 3


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("breaks", [1, STREAM_BREAK_RETRIES])
def test_a_stream_broken_after_content_is_asked_again_fresh_and_every_attempt_recorded(
        monkeypatch, protocol, breaks):
    failures = []
    bodies, pauses = scripted_transport(monkeypatch, protocol, ["break"] * breaks + ["answer"])
    p = provider(protocol, attempt_observer=failures.append)
    result = p.complete(request=tool_request(), model_call_id="broken-then-answered")
    assert result["attempts"] == breaks + 1 == len(bodies)
    assert len(set(bodies)) == 1, "every retry is the same fresh request"
    assert [call["id"] for call in result["message"]["tool_calls"]] == ["accepted-tool"]
    assert "discarded" not in json.dumps(result), "partial content never reaches the policy"
    assert pauses == [1, 2, 4][:breaks]
    assert [row["attempt"] for row in failures] == list(range(1, breaks + 1))
    assert [row["stream_breaks"] for row in failures] == list(range(1, breaks + 1))
    for row in failures:
        assert row["retry_scheduled"] is True and row["phase"] == "transport"
        assert row["tool_execution_started"] is False
        assert row["stream_break_retry_limit"] == STREAM_BREAK_RETRIES
        assert row["recovery_policy"] == recovery.RECOVERY_POLICY
        assert row["partial_event_count"] == 2
        assert row["partial_events"][-1] == partial_events(protocol)[-1]
        assert any(error["type"] == "RemoteProtocolError" for error in row["error_chain"])
    if protocol == "anthropic":
        assert p.credential_rotations() == [] and p.api_key == "offline-primary-key"
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_the_fourth_stream_break_is_terminal_with_deadline_left(monkeypatch, protocol):
    failures = []
    bodies, pauses = scripted_transport(monkeypatch, protocol, ["break"])
    p = provider(protocol, attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as raised:
        p.complete(request=tool_request(), model_call_id="exhausted")
    assert len(bodies) == STREAM_BREAK_RETRIES + 1
    assert raised.value.provider_failure["attempts"] == STREAM_BREAK_RETRIES + 1
    assert raised.value.provider_failure["response_started"] is True
    assert pauses == [1, 2, 4]
    assert [row["retry_scheduled"] for row in failures] == [True, True, True, False]
    assert [row["stream_breaks"] for row in failures] == [1, 2, 3, 4]
    assert failures[-1]["next_credential_index"] is None and failures[-1]["retry_delay_s"] == 0
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_failures_before_any_content_do_not_spend_the_stream_break_budget(monkeypatch, protocol):
    failures = []
    script = ["http503", "break", "http503", "break", "break", "answer"]
    bodies, _pauses = scripted_transport(monkeypatch, protocol, script)
    p = provider(protocol, attempt_observer=failures.append)
    result = p.complete(request=tool_request(), model_call_id="mixed")
    assert result["attempts"] == len(bodies) == len(script)
    assert [row["stream_breaks"] for row in failures] == [0, 1, 1, 2, 3]
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_a_stream_the_adapter_rejected_as_malformed_stays_fatal(monkeypatch, protocol):
    failures = []
    bodies, pauses = scripted_transport(monkeypatch, protocol, ["malformed", "answer"])
    p = provider(protocol, attempt_observer=failures.append)
    with pytest.raises(PolicySessionError):
        p.complete(request=tool_request(), model_call_id="malformed")
    assert len(bodies) == 1 and pauses == []
    assert len(failures) == 1 and failures[0]["retry_scheduled"] is False
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("settings", [{"preflight_probe": True}, {"max_retries": 0}])
def test_preflight_and_one_shot_calls_never_retry_a_stream_break(monkeypatch, protocol, settings):
    bodies, pauses = scripted_transport(monkeypatch, protocol, ["break", "answer"])
    p = provider(protocol, **settings)
    with pytest.raises(PolicySessionError):
        p.complete(request=tool_request(), model_call_id="one-shot")
    assert len(bodies) == 1 and pauses == []
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_an_operator_close_during_stream_break_backoff_starts_no_retry(monkeypatch, protocol):
    failures = []
    bodies, _pauses = scripted_transport(monkeypatch, protocol, ["break", "answer"])
    p = provider(protocol, attempt_observer=failures.append)
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, _delay: p.close())
    with pytest.raises(native.ProviderClosed):
        p.complete(request=tool_request(), model_call_id="stopped-in-backoff")
    assert len(bodies) == 1 and len(failures) == 1


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_a_stream_break_caused_by_close_is_never_retried(monkeypatch, protocol):
    failures, bodies, pauses = [], [], []
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, delay: pauses.append(delay))
    p = provider(protocol, attempt_observer=failures.append)

    def closed_mid_stream(_bound, body, _window):
        bodies.append(body)
        p.close()
        raise worker_break(protocol)
    monkeypatch.setattr(native, "bounded_request", closed_mid_stream)
    with pytest.raises(PolicySessionError) as raised:
        p.complete(request=tool_request(), model_call_id="closed-mid-stream")
    assert not isinstance(raised.value, native.ProviderClosed)
    assert len(bodies) == 1 and pauses == []
    assert failures[0]["retry_scheduled"] is False
    with pytest.raises(native.ProviderClosed):
        p.complete(request=tool_request(), model_call_id="after-close")
    assert len(bodies) == 1


class BreakingServer(BaseHTTPRequestHandler):
    """Streams partial SSE then drops the socket mid chunked body."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self.server.requests.append(self.path)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        def chunk(text):
            data = text.encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        protocol = self.server.provider_protocol
        if len(self.server.requests) <= self.server.breaks:
            for event in partial_events(protocol):
                chunk("data: " + json.dumps(event) + "\n\n")
            if self.server.mode == "malformed":
                chunk("this line is not SSE framing\n\n")
            # No terminating zero-length chunk: an incomplete chunked read.
            self.close_connection = True
            return
        for event in answered_events(protocol):
            chunk("data: " + json.dumps(event) + "\n\n")
        if protocol == "openai":
            chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


@pytest.fixture
def breaking_server():
    host = ThreadingHTTPServer(("127.0.0.1", 0), BreakingServer)
    host.requests, host.mode = [], "break"
    thread = threading.Thread(target=host.serve_forever, daemon=True)
    thread.start()
    yield host
    host.shutdown()
    host.server_close()
    thread.join()


@pytest.mark.parametrize("protocol,breaks,mode,answered", [
    ("openai", STREAM_BREAK_RETRIES, "break", True),
    ("anthropic", STREAM_BREAK_RETRIES, "break", True),
    ("openai", STREAM_BREAK_RETRIES + 1, "break", False),
    ("openai", 1, "malformed", False),
])
def test_real_worker_retries_incomplete_chunked_reads_within_the_bound(
        breaking_server, monkeypatch, protocol, breaks, mode, answered):
    breaking_server.provider_protocol, breaking_server.breaks = protocol, breaks
    breaking_server.mode = mode
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, _delay: None)
    failures = []
    p = provider(protocol, f"http://127.0.0.1:{breaking_server.server_port}", attempt_observer=failures.append)
    if answered:
        result = p.complete(request=tool_request(), model_call_id="real-worker")
        assert result["attempts"] == breaks + 1
        assert [call["id"] for call in result["message"]["tool_calls"]] == ["accepted-tool"]
        assert "discarded" not in json.dumps(result)
    else:
        with pytest.raises(PolicySessionError):
            p.complete(request=tool_request(), model_call_id="real-worker")
    expected = breaks + 1 if answered else STREAM_BREAK_RETRIES + 1 if mode == "break" else 1
    assert len(breaking_server.requests) == len(failures) + int(answered) == expected
    assert all(row["partial_event_count"] == 2 for row in failures)
    if mode == "break":
        assert all(any(error["type"] == "RemoteProtocolError" and "incomplete chunked read" in error["message"]
                       for error in row["error_chain"]) for row in failures)
    assert failures[-1]["retry_scheduled"] is answered
    p.close()


# ---------------------------------------------------------------------------
# A credential echoed in a broken stream fails closed, never retried
# ---------------------------------------------------------------------------

def echo_event(protocol):
    if protocol == "anthropic":
        return {"type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "offline-primary-key"}}
    return {"choices": [{"delta": {"content": "key offline-primary-key"}, "finish_reason": None}]}


class EchoBreakingServer(BreakingServer):
    """First response echoes the credential, then drops the socket; later ones answer."""

    def do_POST(self):
        if self.server.requests:
            return super().do_POST()
        self.rfile.read(int(self.headers["content-length"]))
        self.server.requests.append(self.path)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for event in [*partial_events(self.server.provider_protocol), echo_event(self.server.provider_protocol)]:
            data = ("data: " + json.dumps(event) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        self.close_connection = True


@pytest.fixture
def echo_server():
    host = ThreadingHTTPServer(("127.0.0.1", 0), EchoBreakingServer)
    host.requests, host.mode, host.breaks = [], "break", 0
    thread = threading.Thread(target=host.serve_forever, daemon=True)
    thread.start()
    yield host
    host.shutdown()
    host.server_close()
    thread.join()


def assert_echo_failed_closed(caught, failures):
    from ora.agent.providers.attempts import CREDENTIAL_ECHO_MESSAGE
    assert str(caught.value) == CREDENTIAL_ECHO_MESSAGE
    assert caught.value.retryable is False
    assert caught.value.provider_failure["attempts"] == 1
    assert caught.value.provider_failure["response_started"] is True
    assert len(failures) == 1 and failures[0]["retry_scheduled"] is False
    assert failures[0]["credential_echo"] is True and failures[0]["credentials_redacted"] is True
    assert failures[0]["tool_execution_started"] is False
    assert "offline-primary-key" not in json.dumps(failures)
    assert "offline-primary-key" not in str(caught.value)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_real_worker_fails_closed_on_a_credential_echoed_before_the_break(
        echo_server, monkeypatch, protocol):
    """The worker redacts partials before the parent sees them, so the echo
    must be judged on the raw events inside the worker. Before this, the
    redacted copy passed the parent check and the break was retried."""

    echo_server.provider_protocol = protocol
    monkeypatch.setattr(recovery.RecoveryWindow, "pause", lambda _self, _delay: None)
    failures = []
    p = provider(protocol, f"http://127.0.0.1:{echo_server.server_port}", attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as caught:
        p.complete(request=tool_request(), model_call_id="echoed")
    assert len(echo_server.requests) == 1, "an echoed credential is never asked again"
    assert_echo_failed_closed(caught, failures)
    assert failures[0]["partial_event_count"] == 3
    assert "[REDACTED]" in json.dumps(failures[0]["partial_events"])
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_the_worker_echo_verdict_survives_redacted_partials(monkeypatch, protocol):
    bodies, _pauses = scripted_transport(monkeypatch, protocol, ["answer"])

    def echoed_break(_bound, body, _window):
        bodies.append(body)
        exc = worker_break(protocol)
        exc.partial_events = [*exc.partial_events, {"redacted": "[REDACTED]"}]
        exc.credential_echo = True
        raise exc
    monkeypatch.setattr(native, "bounded_request", echoed_break)
    failures = []
    p = provider(protocol, attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as caught:
        p.complete(request=tool_request(), model_call_id="echoed")
    assert len(bodies) == 1
    assert_echo_failed_closed(caught, failures)
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_legacy_retries_fail_closed_on_an_echoed_partial_too(monkeypatch, protocol):
    calls = []

    def broken(_self, _body):
        calls.append(1)
        exc = worker_break(protocol)
        exc.partial_events = [*exc.partial_events, echo_event(protocol)]
        raise exc
    cls = AnthropicMessagesProvider if protocol == "anthropic" else OpenAICompatibleProvider
    monkeypatch.setattr(cls, "_request_once", broken)
    monkeypatch.setattr(native.time, "sleep", lambda _delay: None)
    failures = []
    p = cls("https://offline.invalid", "offline-primary-key", attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as caught:
        p.complete(request=tool_request(), model_call_id="echoed")
    assert len(calls) == 1
    assert_echo_failed_closed(caught, failures)
    p.close()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_a_provider_recovery_window_overrides_the_default_incident_deadline(monkeypatch, protocol):
    """Kimi K3 at high effort streams single replies past 180 s (EXP11A g006)."""
    clock = [100.]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    default = provider(protocol)
    widened = provider(protocol, recovery_window_s=600.0)
    assert recovery.RecoveryWindow.for_provider(default).remaining() == recovery.RECOVERY_WINDOW_S == 180.
    assert recovery.RecoveryWindow.for_provider(widened).remaining() == 600.
    default.close(); widened.close()


def test_build_native_provider_passes_the_recovery_window():
    built = native.build_native_provider("anthropic_messages", base_url="https://offline.invalid", api_key="k",
                                         recovery_policy=recovery.RECOVERY_POLICY, recovery_window_s=600.0)
    assert built.recovery_window_s == 600.0
    built.close()
    with pytest.raises(PolicySessionError, match="recovery_window_s"):
        native.build_native_provider("anthropic_messages", base_url="https://offline.invalid", api_key="k",
                                     recovery_window_s=600.0)
