"""Explicit read-only refusals survive RPC without masking worker faults."""

from __future__ import annotations

import io
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.native.tool_loop import _tool_error
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.rpc import EnvironmentRpcError, LocalRpcEnvironmentSession, RPC_SCHEMA
from ora.environment import worker


def _proxy(reader, writer, connection):
    proxy = object.__new__(LocalRpcEnvironmentSession)
    proxy.process = SimpleNamespace(poll=lambda: None)
    proxy.connection = connection
    proxy.reader = reader
    proxy.writer = writer
    proxy.session_id = "refusal-test"
    proxy.request_sequence = 0
    return proxy


def test_worker_rpc_preserves_expected_refusal_fault_and_following_request(tmp_path, monkeypatch):
    class Session:
        def probe_motion(self, request):
            if request["mode"] == "refusal":
                raise EnvironmentRefusal(
                    "query position is outside the declared workspace",
                    field="queries.q.pose.position_m.z", received=0.6,
                    allowed=[0.7, 1.4],
                )
            if request["mode"] == "fault":
                raise ValueError("unexpected solver failure")
            return {"accepted": True}

        def close(self):
            pass

    monkeypatch.setattr(worker, "_build_session", lambda _config: Session())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": RPC_SCHEMA, "session_id": "refusal-test"}))
    errors = []
    with tempfile.TemporaryDirectory(prefix="rh-refusal-") as socket_dir:
        address = Path(socket_dir) / "s"

        def serve():
            try:
                worker.serve(address, config)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3)
            deadline = time.monotonic() + 3
            while True:
                try:
                    connection.connect(str(address))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise AssertionError(f"fake worker did not become ready: {errors}")
                    time.sleep(0.01)
            with connection.makefile("r", encoding="utf-8") as reader, connection.makefile(
                "w", encoding="utf-8",
            ) as writer:
                proxy = _proxy(reader, writer, connection)
                with pytest.raises(EnvironmentRefusal) as caught:
                    proxy.probe_motion({"mode": "refusal"})
                assert caught.value.field == "queries.q.pose.position_m.z"
                assert caught.value.received == 0.6 and caught.value.allowed == [0.7, 1.4]
                with pytest.raises(EnvironmentRpcError, match="unexpected solver failure") as fault:
                    proxy.probe_motion({"mode": "fault"})
                assert _tool_error(fault.value)["error"]["code"] == "internal_tool_error"
                assert _tool_error(fault.value)["error"]["retryable"] is False
                assert proxy.probe_motion({"mode": "valid"}) == {"accepted": True}
                proxy._request("close", {})
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors


@pytest.mark.parametrize("error", [
    {"kind": "refusal", "message": "bare value error"},
    {"kind": "refusal", "type": "ValueError", "message": "not explicitly typed"},
    "not-an-error-object",
])
def test_malformed_refusal_never_becomes_a_retryable_input_error(error):
    response = {"schema": RPC_SCHEMA, "session_id": "refusal-test", "request_id": 0,
                "ok": False, "error": error}
    proxy = _proxy(io.StringIO(json.dumps(response) + "\n"), io.StringIO(), object())
    with pytest.raises(EnvironmentRpcError, match="envelope is malformed"):
        proxy.probe_motion({})


@pytest.mark.parametrize("error", [
    {"type": "ValueError", "message": "legacy worker fault"},
    {"kind": "fault", "type": "ValueError", "message": "current worker fault"},
])
def test_old_and_new_worker_fault_envelopes_remain_internal(error):
    response = {"schema": RPC_SCHEMA, "session_id": "refusal-test", "request_id": 0,
                "ok": False, "error": error}
    proxy = _proxy(io.StringIO(json.dumps(response) + "\n"), io.StringIO(), object())
    with pytest.raises(EnvironmentRpcError):
        proxy.probe_motion({})


#: The current default surface of each backend whose grasp_plan runs the motion probe.
PROBE_SURFACES = {
    "libero": "libero_placement_direct_overhead_v1",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
}


def _probe_runtime(tmp_path, backend, callback):
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(PROBE_SURFACES[backend])
    runtime.workspace_bounds_m = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.4]}
    runtime.motion_probe_callback = callback
    runtime._current_grip_state = lambda: (
        {"observation_id": "obs-1", "observation_sha256": "a" * 64},
        tmp_path, {}, None, None,
    )
    runtime._known_plan_refs = {}
    runtime._submitted_mutations = []
    return runtime


def _query(position):
    return {"query_id": "q", "arm": "right", "pose": {
        "position_m": position, "orientation_xyzw": [0, 0, 0, 1],
    }}


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_probe_checks_every_query_before_bridge_and_keeps_exact_workspace_boundary(tmp_path, backend):
    calls = []

    def callback(request):
        calls.append(request)
        return {"schema": "ora.motion-probe-result.v1", "results": [
            {"query_id": query["query_id"], "arm": query["arm"],
             "reachable": False, "plan_ref": None, "variants": []}
            for query in request["queries"]
        ]}

    runtime = _probe_runtime(tmp_path, backend, callback)
    with pytest.raises(ToolError) as caught:
        runtime._motion_probe({"queries": [
            {**_query([0, 0, 0.9]), "query_id": "valid"}, _query([0, 0, 0.6]),
        ], "variant_policy": "exact"})
    assert calls == []
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments" and error["retryable"] is True
    assert error["field"] == "queries.q.pose.position_m"
    assert error["constraints"]["allowed"] == runtime.workspace_bounds_m
    assert error["constraints"]["violations"]["z"]["allowed"] == [0.7, 1.4]
    runtime._motion_probe({"queries": [_query([0.6, -0.4, 1.4])], "variant_policy": "exact"})
    assert len(calls) == 1 and calls[0]["queries"][0]["pose"]["position_m"] == [0.6, -0.4, 1.4]


def test_probe_maps_only_declared_environment_refusal_to_tool_error(tmp_path):
    def refuse(_request):
        raise EnvironmentRefusal("known argument refusal", field="queries.q.pose",
                                 received=[0, 0, 0.9], allowed="public workspace")

    runtime = _probe_runtime(tmp_path, "robotwin", refuse)
    arguments = {"queries": [_query([0, 0, 0.9])], "variant_policy": "exact"}
    with pytest.raises(ToolError) as caught:
        runtime._motion_probe(arguments)
    assert caught.value.code == "invalid_arguments" and caught.value.retryable is True
    assert caught.value.constraints == {"allowed": "public workspace"}

    def fault(_request):
        raise ValueError("unexpected IK implementation error")

    runtime.motion_probe_callback = fault
    with pytest.raises(ValueError) as caught:
        runtime._motion_probe(arguments)
    assert _tool_error(caught.value)["error"]["code"] == "internal_tool_error"
    assert _tool_error(caught.value)["error"]["retryable"] is False


def test_robotwin_probe_uses_typed_workspace_refusal_without_changing_action_guard():
    from ora.environment.backends.robotwin.session import (
        RoboTwinLiveError, RoboTwinLiveSession, _validate_workspace_target,
    )

    session = object.__new__(RoboTwinLiveSession)
    session._require_environment = lambda: object()
    session._uses_validated_planner_profile = lambda: True
    session.workspace_bounds_m = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.4]}
    request = {
        "schema": "ora.motion-probe-request.v1",
        "observation_id": "obs-1", "observation_sha256": "a" * 64,
        "queries": [_query([0, 0, 0.6])], "variant_policy": "exact",
    }
    with pytest.raises(EnvironmentRefusal) as caught:
        session.probe_motion(request)
    assert caught.value.field == "queries.q.pose.position_m"
    assert caught.value.allowed == session.workspace_bounds_m
    with pytest.raises(RoboTwinLiveError):
        _validate_workspace_target([0, 0, 0.6], session.workspace_bounds_m)
    session.workspace_bounds_m = {"x": "invalid backend configuration"}
    with pytest.raises(RoboTwinLiveError):
        session.probe_motion(request)


def test_align_plan_survives_every_transport_layer_between_backend_and_policy(tmp_path, monkeypatch):
    """The eight layers a new read-only environment tool must cross, in one run.

    Nothing else checks that these stay in sync. Omitting the worker branch, the
    RPC stub or the facade forwarder does not fail any test: it fails at runtime
    as `internal_tool_error`, with the raw worker message, in a live episode.

    The layers exercised here are the four that are silent when missing --
    backend session method, worker dispatch, RPC stub, and the
    DescribedEnvironmentSession facade the coordinator actually holds.
    """
    from ora.environment.session import DescribedEnvironmentSession

    served = {}

    class Session:
        def align_plan(self, request):
            served["request"] = request
            if request.get("approach_side") == "nowhere":
                raise EnvironmentRefusal(
                    "approach side is not a declared direction",
                    field="approach_side", received="nowhere",
                    allowed=["top", "left", "right", "front", "back"],
                )
            return {
                "schema": "ora.align-plan-result.v1",
                "candidates": [{"reachable": True, "rank": 0}],
                "motion_planner_called": False,
            }

        def close(self):
            pass

    monkeypatch.setattr(worker, "_build_session", lambda _config: Session())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": RPC_SCHEMA, "session_id": "refusal-test"}))
    errors = []
    with tempfile.TemporaryDirectory(prefix="rh-align-") as socket_dir:
        address = Path(socket_dir) / "s"

        def serve():
            try:
                worker.serve(address, config)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3)
            deadline = time.monotonic() + 3
            while True:
                try:
                    connection.connect(str(address))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise AssertionError(f"fake worker did not become ready: {errors}")
                    time.sleep(0.01)
            with connection.makefile("r", encoding="utf-8") as reader, connection.makefile(
                "w", encoding="utf-8",
            ) as writer:
                proxy = _proxy(reader, writer, connection)
                # The coordinator holds the facade, never the proxy, so a
                # forwarder missing there breaks the chain with everything else
                # in place.
                facade = object.__new__(DescribedEnvironmentSession)
                facade.delegate = proxy

                result = facade.align_plan({
                    "schema": "ora.align-plan-request.v1",
                    "target_ref": "target-" + "a" * 24,
                    "approach_side": "top", "arm": "right",
                })
                assert result["schema"] == "ora.align-plan-result.v1"
                assert result["motion_planner_called"] is False
                assert served["request"]["approach_side"] == "top"

                # A correctable input stays correctable across the boundary: it
                # must not arrive as a non-retryable internal fault.
                with pytest.raises(EnvironmentRefusal) as caught:
                    facade.align_plan({
                        "schema": "ora.align-plan-request.v1",
                        "target_ref": "target-" + "a" * 24,
                        "approach_side": "nowhere", "arm": "right",
                    })
                assert caught.value.field == "approach_side"
                assert caught.value.allowed == ["top", "left", "right", "front", "back"]

                # The worker must stay alive: the dispatch branch returns
                # should_close=False, not a mutation flag.
                assert facade.align_plan({
                    "schema": "ora.align-plan-request.v1",
                    "target_ref": "target-" + "b" * 24,
                    "approach_side": "left", "arm": "left",
                })["candidates"]
                proxy._request("close", {})
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors
