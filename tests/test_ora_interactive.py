"""Offline ORA task selection, queued messages, launch guards and transcript delivery."""

from __future__ import annotations

import copy
import io
import json
import threading
import time
from pathlib import Path

import pytest
import yaml

from ora.orchestrator import operator_control, paths
from ora.orchestrator.ora_backend import InteractiveBackend, launch_prepared, serve
from ora.orchestrator.ora_catalog import load_tasks
from ora.orchestrator.spec import LAUNCH_ADMISSION_SCHEMA, RunSpecError


@pytest.fixture
def catalog(tmp_path):
    spec = {
        "schema": "ora.run-spec.v1", "experiment": "prepared-study",
        "output_workspace": "outputs/prepared-study", "config": "task-seed-4", "milestone": "M3",
        "task": "open_laptop", "task_instruction": "Open the laptop.", "seed": 4,
        "policy": {"kind": "native_api", "provider_profile": "explicit-comparison",
                   "memory_policy": "disabled"},
        "runtime": {"kind": "robotwin", "task": "open_laptop", "required_host": "fixture-host"},
        "qualification": {"required": True, "manifest": "fixture-qualification.json"},
        "prerequisite": {"kept": True},
    }
    source = tmp_path / "prepared.yaml"
    source.write_text(yaml.safe_dump(spec))
    return load_tasks([source]), spec


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    root = tmp_path / "ora-interactive-m3-fixture--20261012T000000Z"
    root.mkdir()
    plan = {"experiment": "ora-interactive", "task": "open_laptop", "task_instruction": "Open the laptop.",
            "plan_sha256": "a" * 64, "policy": {"kind": "native_api"},
            "runtime": {"kind": "robotwin"}, "operator_control": {"mode": "file_inbox_v1"}}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "status.json").write_text(json.dumps({"state": "running", "run_id": root.name}))
    (root / "agent-events.jsonl").write_text("")
    operator_control.initialize_control(root, plan, root.name)
    return root, plan


def _backend(catalog, run, *, initial=(), observer=None):
    tasks, _ = catalog
    root, plan = run
    captured = []
    calls = []

    def launch(task, messages, cancelled, report):
        calls.append((task, copy.deepcopy(messages), cancelled))
        if messages:
            plan["operator_control"]["initial_messages"] = copy.deepcopy(messages)
            (root / "plan.json").write_text(json.dumps(plan))
            operator_control.initialize_control(root, plan, root.name)
        return root

    backend = InteractiveBackend(tasks, captured.append, confirm_launch=True, launch=launch, observer=observer)
    backend.start()
    for message in initial:
        backend.request(message)
    backend.request({"id": "start", "type": "prompt", "task": tasks[0].value})
    backend.job.join(timeout=2)
    assert not backend.job.is_alive()
    backend.tick()
    assert backend.state == "running"
    return backend, captured, calls


def test_prepared_task_preserves_policy_task_and_qualification(catalog):
    tasks, source = catalog
    initial = [{"id": "f1", "type": "follow_up", "message": "Check the final position."}]
    spec = tasks[0].interactive_spec("attempt-1", initial)
    for field in ("policy", "runtime", "qualification", "prerequisite", "seed", "task", "task_instruction"):
        assert spec[field] == source[field]
    assert spec["experiment"] == "ora-interactive"
    assert spec["policy"]["kind"] == "native_api"
    assert spec["operator_control"]["mode"] == "file_inbox_v1"
    assert "experiment_identity" not in spec
    assert spec["operator_control"]["initial_messages"] == initial
    assert spec["launch_admission"]["cohort_id"] == "ora-interactive-attempt-1"
    assert tasks[0].spec == source


def test_stale_spec_and_source_bound_launch_gate_are_refused(catalog):
    tasks, source = catalog
    tasks[0].spec["launch_admission"] = {"gate": "original-study-gate.json"}
    with pytest.raises(RuntimeError, match="gate cannot be relabelled"):
        tasks[0].interactive_spec("attempt", [])
    tasks[0].path.write_text(yaml.safe_dump({**source, "seed": 9}))
    with pytest.raises(RuntimeError, match="RunSpec changed"):
        tasks[0].interactive_spec("attempt", [])


def test_context_only_specs_cannot_become_interactive_launches(catalog):
    tasks, _ = catalog
    tasks[0].spec["launch_admission"] = {"schema": LAUNCH_ADMISSION_SCHEMA, "mode": "disabled"}
    with pytest.raises(RuntimeError, match="context-only"):
        tasks[0].interactive_spec("attempt", [])


def test_source_inspection_policy_is_checked_before_namespace_rewrite(catalog, tmp_path, monkeypatch):
    from ora.orchestrator import ora_catalog, spec
    tasks, _ = catalog
    package = tmp_path / "experiments" / "prepared_study"
    package.mkdir(parents=True)
    (package / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": "prepared-study", "configs": {"task-seed-4": {}},
        "launch_admission_policy": {
            "schema": spec.EXPERIMENT_LAUNCH_POLICY_SCHEMA, "default_mode": "inspection_only",
            "validator": "experiment_package_v1", "cells": {"different-cell": {
                "mode": "single_attempt_cell", "cohort_id": "reviewed-cohort", "cell_id": "reviewed-cell",
                "resource_profile": "reviewed-resource", "slot_id": "reviewed-slot",
                "gate": "outputs/prepared-study/gates/reviewed.json",
            }},
        },
    }))
    monkeypatch.setattr(ora_catalog, "PROJECT_ROOT", tmp_path)
    with pytest.raises(RunSpecError, match="source-bound launch admission"):
        tasks[0].interactive_spec("attempt", [])


def test_picker_requires_confirmation_and_rejects_arbitrary_instruction(catalog):
    tasks, _ = catalog
    ops = []
    backend = InteractiveBackend(tasks, ops.append, confirm_launch=False,
                                 launch=lambda *args: pytest.fail("unauthorized launch"))
    backend.request({"id": "p1", "type": "prompt", "task": tasks[0].value})
    assert not ops[-2 if ops[-1]["op"] == "queue" else -1]["success"]
    backend.request({"id": "p2", "type": "prompt", "task": tasks[0].value, "message": "Replace the goal"})
    assert ops[-1]["success"] is False
    assert backend.state == "idle"


def test_followup_is_bound_to_next_plan_and_new_followup_stays_queued(catalog, run):
    followup = {"id": "f1", "type": "follow_up", "message": "Inspect the hinge."}
    backend, ops, calls = _backend(catalog, run, initial=[followup])
    assert calls[0][1] == [followup]
    assert not backend.followups
    backend.request({"id": "f2", "type": "follow_up", "message": "Use the other arm next time."})
    assert [message["id"] for message in backend.followups] == ["f2"]
    control = operator_control.command_status(run[0])
    assert [message["id"] for message in control["pending"]] == ["f1"]
    assert control["pending"][0]["type"] == "follow_up"


def test_steer_is_queued_once_and_visible_only_on_supervisor_delivery(catalog, run):
    backend, ops, _ = _backend(catalog, run)
    root, _ = run
    request = {"id": "s1", "type": "steer", "message": "Measure the lid edge again."}
    backend.request(request)
    backend.request(request)
    assert len(operator_control.command_status(root)["pending"]) == 1
    assert not any(op["op"] == "user" and op.get("text") == request["message"] for op in ops)
    control = operator_control._load_control(root, require_open=True)
    command = control.poll()[0]
    control.acknowledge(command, "applied")
    (root / "agent-events.jsonl").write_text(json.dumps({
        "type": "operator_message", "payload": request,
    }) + "\n")
    backend.tick()
    assert [op["text"] for op in ops if op["op"] == "user"] == ["Open the laptop.", request["message"]]
    assert backend.run_messages["s1"]["status"] == "delivered"


def test_early_steer_survives_supervisor_startup(catalog, run, monkeypatch):
    backend, _, _ = _backend(catalog, run)
    original = operator_control.submit_command
    monkeypatch.setattr(operator_control, "submit_command", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    backend.request({"id": "early", "type": "steer", "message": "Check before lifting."})
    assert backend.run_messages["early"]["status"] == "queued"
    monkeypatch.setattr(operator_control, "submit_command", original)
    backend.tick()
    assert "early" in backend.submitted
    assert operator_control.command_status(run[0])["pending"][0]["id"] == "early"


def test_header_refreshes_from_late_session_metadata(catalog, run):
    backend, ops, _ = _backend(catalog, run)
    (run[0] / "agent-events.jsonl").write_text(json.dumps({
        "type": "session_started", "payload": {
            "model": "provider/deepseek-flash", "reasoning_effort": "low",
            "provider_protocol": "openai_chat_completions",
        },
    }) + "\n")
    backend.tick()
    header = [op for op in ops if op["op"] == "header"][-1]
    assert header["model"] == "deepseek-flash"
    assert header["reasoning_effort"] == "low"
    assert header["provider_protocol"] == "openai_chat_completions"


def test_abort_uses_existing_exact_run_stop(catalog, run, monkeypatch):
    from ora.orchestrator import cli
    backend, ops, _ = _backend(catalog, run)
    stopped = []
    monkeypatch.setattr(cli, "command_stop", lambda run_id, **kwargs: stopped.append(run_id))
    backend.request({"id": "a1", "type": "abort"})
    assert stopped == [run[0].name]
    assert ops[-1]["success"] is True


@pytest.mark.parametrize("abort", [False, True])
def test_abort_survives_spawn_and_detach_race_but_detach_alone_does_not_stop(catalog, run, monkeypatch, abort):
    from ora.orchestrator import cli
    entered, release = threading.Event(), threading.Event()
    stopped = []

    def launch(*args):
        entered.set()  # Spawn has passed the cancellation check.
        assert release.wait(timeout=2)
        return run[0]

    monkeypatch.setattr(cli, "command_stop", lambda run_id, **kwargs: stopped.append(run_id))
    backend = InteractiveBackend(catalog[0], lambda op: None, confirm_launch=True, launch=launch)
    backend.request({"id": "p1", "type": "prompt", "task": "task-1"})
    try:
        assert entered.wait(timeout=2)
        if abort:
            backend.request({"id": "a1", "type": "abort"})
        backend.cancelled.set()  # EOF/detach: the controller no longer calls tick().
    finally:
        release.set()
        backend.job.join(timeout=2)
    assert not backend.job.is_alive()
    assert stopped == ([run[0].name] if abort else [])


def test_finished_run_rejects_late_input_and_preserves_all_evidence(catalog, run):
    backend, ops, _ = _backend(catalog, run)
    root, _ = run
    (root / "status.json").write_text(json.dumps({"state": "completed", "run_id": root.name}))
    (root / "seal.json").write_text("{}")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}
    backend.tick()
    backend.request({"id": "late", "type": "steer", "message": "Do more."})
    assert backend.state == "finished"
    assert ops[-1]["success"] is False
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_end_waits_for_the_supervisor_to_land(catalog, run, monkeypatch):
    """`end` follows the Supervisor's exit: under a cluster job step it belongs to the UI's step."""
    from ora.orchestrator import suite_queue

    backend, ops, _ = _backend(catalog, run)
    root, _ = run
    alive = [True]
    monkeypatch.setattr(suite_queue, "supervisor_alive", lambda path, run_id: alive[0])
    (root / "status.json").write_text(json.dumps({"state": "completed", "run_id": root.name}))
    backend.tick()
    assert backend.state == "running"
    assert not [op for op in ops if op["op"] == "end"]
    assert [op for op in ops if op["op"] == "status"][-1]["text"].startswith("Landing")
    (root / "seal.json").write_text("{}")
    backend.tick()   # sealed, but the lease and scratch are released after the seal
    assert backend.state == "running"
    alive[0] = False
    backend.tick()
    assert backend.state == "finished"
    assert [op for op in ops if op["op"] == "end"][-1]["landing"] == "sealed"


def test_supervisor_exit_without_seal_ends_with_a_warning(catalog, run, monkeypatch):
    from ora.orchestrator import suite_queue

    backend, ops, _ = _backend(catalog, run)
    root, _ = run
    monkeypatch.setattr(suite_queue, "supervisor_alive", lambda path, run_id: False)
    (root / "status.json").write_text(json.dumps({"state": "completed", "run_id": root.name}))
    backend.tick()
    assert backend.state == "finished"
    assert [op for op in ops if op["op"] == "end"][-1]["landing"] == "unsealed"
    assert any(op["op"] == "notice" and "exited before sealing" in op["text"] for op in ops)


def test_failed_launch_keeps_followups(catalog):
    tasks, _ = catalog
    backend = InteractiveBackend(tasks, lambda op: None, confirm_launch=True,
                                 launch=lambda *args: (_ for _ in ()).throw(RuntimeError("preflight blocked")))
    backend.request({"id": "f1", "type": "follow_up", "message": "Inspect first."})
    backend.request({"id": "p1", "type": "prompt", "task": tasks[0].value})
    backend.job.join(timeout=2)
    backend.tick()
    assert backend.state == "idle"
    assert backend.followups[0]["id"] == "f1"


def test_rpc_eof_detaches_without_launch_or_stop(catalog, monkeypatch):
    from ora.orchestrator import cli
    monkeypatch.setattr(cli, "command_stop", lambda *args, **kwargs: pytest.fail("detach must not stop a run"))
    output = io.StringIO()
    assert serve([catalog[0][0].path], confirm_launch=False, input_stream=io.StringIO(""), output_stream=output) == 0
    ops = [json.loads(line) for line in output.getvalue().splitlines()]
    assert {op["op"] for op in ops} >= {"header", "tasks", "queue", "session_state"}


def test_cancelled_launch_never_runs_preflight_or_spawn(catalog, monkeypatch):
    from ora.orchestrator import cli, spec
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(spec, "resolve_run_spec", lambda value: {"policy": {"kind": "native_api"}})
    monkeypatch.setattr(cli, "_preflight", lambda *args, **kwargs: pytest.fail("cancelled preflight"))
    monkeypatch.setattr(cli, "_spawn", lambda *args, **kwargs: pytest.fail("cancelled launch"))
    with pytest.raises(RuntimeError, match="launch cancelled"):
        launch_prepared(catalog[0][0], [], cancelled, lambda message: None)


def test_world_observer_binds_exact_run_and_releases_on_rpc_detach(catalog, run):
    class Observer:
        def __init__(self):
            self.bindings, self.polls, self.closed = [], 0, False
        def bind(self, root): self.bindings.append(root)
        def poll(self): self.polls += 1
        def close(self): self.closed = True

    observer = Observer()
    _backend(catalog, run, observer=observer)
    assert observer.bindings == [None, run[0]]
    assert observer.polls >= 2
    output = io.StringIO()
    assert serve([catalog[0][0].path], confirm_launch=False, input_stream=io.StringIO(""),
                 output_stream=output, observer_factory=lambda **kwargs: observer) == 0
    assert observer.closed


def _until(condition):
    deadline = time.monotonic() + 2
    while not condition():
        assert time.monotonic() < deadline, "world observer did not publish its state"
        time.sleep(0.01)


def test_world_observer_needs_decode_ack_rebinds_and_never_opens_remote_browser(tmp_path):
    from ora.orchestrator.ora_world import WorldObserver

    class Server:
        url = "http://127.0.0.1:5017/live"
        def __init__(self, **kwargs):
            assert kwargs == {"host": "127.0.0.1", "port": 5017}
            self.root, self.connected, self.closed = None, False, False
        def start(self): return self
        def bind(self, root):
            self.root, self.connected = root, False
        def status(self):
            return {"phase": "waiting", "connected": self.connected, "visible": self.connected,
                    "run_id": self.root.name if self.root else None}
        def close(self): self.closed = True

    server = Server(host="127.0.0.1", port=5017)
    output = []
    observer = WorldObserver(output.append, port=5017, server_factory=lambda **kwargs: server,
                             open_browser=lambda url: pytest.fail("SSH cannot open the user's local browser"),
                             environment={"SSH_CONNECTION": "fixture"})
    try:
        _until(lambda: observer._snapshot.get("browser_message"))
        observer.poll()
        assert output[-1]["connected"] is False
        assert "forward" in output[-1]["browser_message"]
        root = tmp_path / "exact-run"
        observer.bind(root)
        _until(lambda: server.root == root)
        server.connected = True  # Equivalent to the server's valid camera decode acknowledgement.
        _until(lambda: observer._snapshot.get("connected"))
        observer.poll()
        assert output[-1]["connected"] is True
        observer.bind(tmp_path / "next-run")
        assert output[-1]["connected"] is False
        _until(lambda: server.root.name == "next-run")
    finally:
        observer.close()
    assert server.closed


def test_world_observer_opens_after_health_and_keeps_failed_browser_nonfatal(tmp_path):
    from ora.orchestrator.ora_world import WorldObserver
    actions = []
    class Server:
        url = "http://127.0.0.1:5019/live"
        def start(self): return self
        def bind(self, root): actions.append(("bind", root))
        def status(self): return {"phase": "waiting", "connected": False, "visible": False}
        def close(self): actions.append(("close",))
    def health(url):
        actions.append(("health", url))
        return True
    def browser(url):
        actions.append(("browser", url))
        raise OSError("no browser configured")
    observer = WorldObserver(lambda op: None, root=tmp_path, server_factory=lambda **kwargs: Server(),
                             health_check=health, open_browser=browser, environment={},
                             public_url="http://127.0.0.1:5020/live")
    try:
        _until(lambda: observer._snapshot.get("browser_message"))
        assert actions[:3] == [("bind", tmp_path), ("health", "http://127.0.0.1:5019/health"),
                               ("browser", "http://127.0.0.1:5020/live")]
        assert observer._snapshot["connected"] is False
        assert observer._snapshot["url"] == "http://127.0.0.1:5020/live"
        assert "Open this URL" in observer._snapshot["browser_message"]
    finally:
        observer.close()
    assert ("close",) in actions


def test_blocked_browser_cannot_block_binding_polling_or_observer_close(tmp_path):
    from ora.orchestrator.ora_world import WorldObserver
    entered, release = threading.Event(), threading.Event()

    class Server:
        url = "http://127.0.0.1:5019/live"
        root, closed = None, False
        def start(self): return self
        def bind(self, root): self.root = root
        def status(self):
            return {"phase": "waiting", "connected": False, "visible": False,
                    "run_id": self.root.name if self.root else None}
        def close(self): self.closed = True

    def browser(url):
        entered.set()
        release.wait(3)
        return False

    server, output = Server(), []
    observer = WorldObserver(output.append, root=tmp_path / "first", server_factory=lambda **kwargs: server,
                             health_check=lambda url: True, open_browser=browser, environment={})
    try:
        assert entered.wait(1)
        _until(lambda: observer._snapshot.get("run_id") == "first")
        observer.poll()
        assert output[-1]["connected"] is False
        observer.bind(tmp_path / "next")
        _until(lambda: server.root == tmp_path / "next" and observer._snapshot.get("run_id") == "next")
        assert server.root == tmp_path / "next"
        observer.close()
        assert server.closed and not observer._thread.is_alive()
        snapshot = dict(observer._snapshot)
        release.set()
        observer._browser_thread.join(timeout=1)
        assert observer._snapshot == snapshot  # A late browser failure cannot overwrite a fresh binding or detach.
    finally:
        release.set()
        observer.close()


def test_public_ora_world_defaults_headless_selection_and_replay_guard(monkeypatch):
    from ora.orchestrator import ora_cli, ora_world
    selections = []
    monkeypatch.setattr(ora_cli.watch_cli, "main", lambda *args, **kwargs: selections.append(kwargs) or 0)
    monkeypatch.delenv("ORA_WORLD_UI", raising=False)
    monkeypatch.setattr(ora_cli.sys.stdout, "isatty", lambda: True)
    assert ora_cli.main(["watch", "fixture-run"]) == 0
    assert selections[-1]["observer_factory"] is not None
    assert ora_cli.main(["watch", "fixture-run", "--emit", "jsonl"]) == 0
    assert selections[-1]["observer_factory"] is None  # A TTY alone cannot opt a JSON consumer into browser side effects.
    monkeypatch.setattr(ora_cli.sys.stdout, "isatty", lambda: False)
    assert ora_cli.main(["watch", "fixture-run", "--emit", "jsonl"]) == 0
    assert selections[-1]["observer_factory"] is None
    assert ora_cli.main(["watch", "fixture-run", "--emit", "jsonl", "--world-port", "0"]) == 0
    assert selections[-1]["observer_factory"] is not None
    assert ora_cli.main(["watch", "fixture-run", "--emit", "jsonl", "--world-port", "5017", "--world-browser", "off"]) == 0
    factory = selections[-1]["observer_factory"]
    assert factory is not None
    monkeypatch.setattr(ora_world, "WorldObserver", lambda *args, **kwargs: pytest.fail("finished replay must not start live view"))
    emitted = []
    assert factory(root=Path("fixture-run"), finished=True, emit=emitted.append) is None
    assert emitted == [{"op": "world_view", "phase": "replay", "connected": False, "visible": False,
                        "url": None, "message": "Live world view is off for a finished-run transcript replay."}]
    assert ora_cli.main(["watch", "fixture-run", "--world-port", "5017", "--world-view", "off"]) == 0
    assert selections[-1]["observer_factory"] is None
    assert ora_cli.main(["watch", "fixture-run", "--world-port", "65536"]) == 2
    for invalid in ("http://127.0.0.1:65536/live", "http://127.0.0.1:5017/", "http://127.0.0.1:5017/live?run=wrong"):
        assert ora_cli.main(["watch", "fixture-run", "--world-public-url", invalid]) == 2
