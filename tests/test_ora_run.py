"""``ora run``: environment binding, services, qualification reuse and spec composition.

No simulator, service, GPU or Provider is started; producers and builders are faked
at their module boundary.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

from ora.orchestrator import ora_run, ora_run_qualification as qualification, ora_run_services as services

PROVIDER = {"DEEPSEEK_LOW_API_KEY": "secret-never-printed"}
GPU_KEYS = {key: f"/srv/{key.lower()}" for keys in ora_run.REQUIRED_KEYS.values() for key in keys}


def test_environment_binds_env_file_host_marker_and_provider_defaults(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_LOW_API_KEY=\n" + "".join(f"{k}={v}\n" for k, v in GPU_KEYS.items())
                        + "ROBOTWIN_ROOT=/from-file\n")
    environ = {**PROVIDER, "ROBOTWIN_ROOT": "/from-shell"}
    ora_run.prepare_environment("robotwin", env_file=env_file, environ=environ)
    assert environ["ORA_EXECUTION_HOST"] == "single-gpu"
    assert environ["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"
    assert environ["DEEPSEEK_LOW_API_KEY"] == "secret-never-printed"  # an empty .env line never blanks it
    assert environ["ROBOTWIN_ROOT"] == "/from-shell"
    assert environ["CUROBO_PYTHON"] == GPU_KEYS["CUROBO_PYTHON"]


def test_environment_refuses_other_hosts_missing_provider_and_missing_install(tmp_path):
    env_file = tmp_path / ".env"
    with pytest.raises(RuntimeError, match="names another execution host"):
        ora_run.prepare_environment("libero", env_file=env_file,
                                    environ={**PROVIDER, **GPU_KEYS, "ORA_EXECUTION_HOST": "ssh23"})
    with pytest.raises(RuntimeError, match="DEEPSEEK_BASE_URL and DEEPSEEK_LOW_API_KEY"):
        ora_run.prepare_environment("libero", env_file=env_file, environ=dict(GPU_KEYS))
    with pytest.raises(RuntimeError, match="base URL must be https://api.deepseek.com"):
        ora_run.prepare_environment("libero", env_file=env_file,
                                    environ={**PROVIDER, **GPU_KEYS, "DEEPSEEK_BASE_URL": "http://elsewhere"})
    with pytest.raises(RuntimeError, match=r"ROBODOJO_ROOT.*ora setup --env robodojo"):
        ora_run.prepare_environment("robodojo", env_file=env_file,
                                    environ={**PROVIDER, **{k: v for k, v in GPU_KEYS.items()
                                                            if k != "ROBODOJO_ROOT"}})


def test_libero_task_names_select_the_base_or_pro_builder():
    assert ora_run.libero_cell("libero_goal/3") == ("libero", "libero_goal", 3)
    assert ora_run.libero_cell("libero_10") == ("libero", "libero_10", 0)
    from ora.environment.benchmarks.libero_pro import CORE_SUITES
    assert ora_run.libero_cell(f"{CORE_SUITES[0]}/1") == ("libero_pro", CORE_SUITES[0], 1)
    with pytest.raises(ValueError, match="unknown LIBERO suite"):
        ora_run.libero_cell("libero_90/0")
    with pytest.raises(ValueError, match="libero_goal/0"):
        ora_run.libero_cell("libero_goal/x")


def _args(**values):
    parsed = ora_run.build_parser().parse_args(["--env", values.pop("env"), "--task", values.pop("task")])
    for key, value in values.items():
        setattr(parsed, key, value)
    return parsed


def test_compose_libero_uses_the_shipped_host_slot_and_a_fresh_cohort(monkeypatch, tmp_path):
    from ora.orchestrator import benchmark_spec

    built = {}

    def build(suite, task_id, **kwargs):
        built.update(suite=suite, task_id=task_id, **kwargs)
        return {"config": "lb-cell"}

    monkeypatch.setattr(benchmark_spec, "build_libero_spec", build)
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: tmp_path / "spec.json")
    path = ora_run.compose(_args(env="libero", task="libero_goal/0"), services=lambda: tmp_path / "report.json",
                           confirm=lambda message: pytest.fail("LIBERO needs no qualification"), stream=io.StringIO())
    assert path == tmp_path / "spec.json"
    assert (built["suite"], built["task_id"], built["seed"]) == ("libero_goal", 0, 0)
    assert (built["host"], built["resource_profile"], built["experiment"]) == ("single-gpu", "libero-gpu0",
                                                                                "ora-interactive")
    assert built["service_report"] == tmp_path / "report.json"
    assert built["cohort_id"].startswith("ora-run-libero-")


def test_compose_robotwin_binds_the_qualified_seed_and_imports_its_manifest(monkeypatch, tmp_path):
    from ora.orchestrator import benchmark_spec

    manifest = tmp_path / "click_bell.json"
    monkeypatch.setattr(qualification, "robotwin_seed", lambda task, **kwargs: (manifest, 100000))
    built, written = {}, {}
    monkeypatch.setattr(benchmark_spec, "build_robotwin_spec", lambda task, **kwargs: built.update(task=task, **kwargs)
                        or {"config": "rt"})
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: written.update(kwargs) or tmp_path / "rt.json")
    ora_run.compose(_args(env="robotwin", task="click_bell"), services=lambda: tmp_path / "r.json", confirm=print,
                    stream=io.StringIO())
    assert (built["seed"], built["seed_manifest"], built["resource_profile"]) == (100000, manifest, "robotwin-gpu0")
    assert written == {"seed_manifest": manifest}


def test_compose_robodojo_binds_the_layered_qualification(monkeypatch, tmp_path):
    from ora.orchestrator import robodojo_spec

    binding = {"card": {"path": "c", "sha256": "1"}, "task": {"path": "t", "sha256": "2"}}
    monkeypatch.setattr(qualification, "robodojo_qualification", lambda task, **kwargs: binding)
    built = {}
    monkeypatch.setattr(robodojo_spec, "build_spec", lambda task, **kwargs: built.update(task=task, **kwargs) or {})
    monkeypatch.setattr(robodojo_spec, "compose", lambda spec: tmp_path / "rd.json")
    ora_run.compose(_args(env="robodojo", task="general_pickup"), services=lambda: tmp_path / "r.json", confirm=print,
                    stream=io.StringIO())
    assert built["infrastructure_qualification"] == binding
    assert (built["gpu"], built["resource_profile"]) == ("0", "robodojo-gpu0")
    with pytest.raises(ValueError, match="layout 0 of seed 0"):
        ora_run.compose(_args(env="robodojo", task="general_pickup", seed=3), services=lambda: tmp_path / "r.json",
                        confirm=print, stream=io.StringIO())


def test_seed_manifest_reuse_takes_the_newest_admitted_seed(monkeypatch, tmp_path):
    from ora.orchestrator import benchmark_spec

    old = tmp_path / "qualification/1/seed-manifests/click_bell.json"
    new = tmp_path / "qualification/2/seed-manifests/click_bell.json"
    for path, seeds in ((old, [100000]), (new, [100002, 100003])):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"accepted_seeds": seeds}))
    import os
    os.utime(old, (1, 1))

    def admitted(path, *, task, seed, task_config):
        # Launch preflight's rule needs the seed's own record in this manifest.
        if (seed not in json.loads(path.read_text())["accepted_seeds"] or seed == 100002
                or task_config != "demo_randomized"):
            raise ValueError("not admitted")
        return b""

    monkeypatch.setattr(benchmark_spec, "admitted_seed_manifest", admitted)
    assert qualification.find_seed_manifest("click_bell", task_config="demo_randomized", seed=None,
                                             workspace=tmp_path) == (new, 100003)
    assert qualification.find_seed_manifest("click_bell", task_config="demo_randomized", seed=100000,
                                             workspace=tmp_path) == (old, 100000)
    assert qualification.find_seed_manifest("click_bell", task_config="demo_clean", seed=None,
                                            workspace=tmp_path) is None


def test_robodojo_reuses_a_card_and_runs_only_the_missing_task_record(monkeypatch, tmp_path):
    card, task_record = tmp_path / "card-qualification.json", tmp_path / "task-qualification.json"
    task_record.write_text("{}")
    monkeypatch.setattr(qualification, "_harness_commit", lambda: "a" * 40)
    monkeypatch.setattr(qualification, "_records", lambda kind: [card] if kind == "card" else [])
    monkeypatch.setattr(qualification, "live_identity", lambda: {"gpu_uuid": "GPU-1", "environment": {}})
    monkeypatch.setattr(qualification, "card_record_admitted", lambda task, record, **kwargs: record == card)
    monkeypatch.setattr(qualification, "task_record_admitted",
                        lambda task, record, **kwargs: {"task": str(record)} if record == task_record else None)
    commands, prompts = [], []

    def produce(command, **kwargs):
        commands.append([str(item) for item in command])
        return f"noise\n{task_record}\n"

    monkeypatch.setattr(qualification, "run_with_progress", produce)
    binding = qualification.robodojo_qualification("general_pickup", requalify=False, confirm=prompts.append,
                                                   stream=io.StringIO())
    assert binding == {"task": str(task_record)}
    assert len(commands) == 1 and commands[0][3:5] == ["task", "general_pickup"]
    assert commands[0][commands[0].index("--card") + 1] == str(card)
    assert "--yes" in commands[0] and "task qualification" in prompts[0]


def test_robodojo_reuses_an_admitted_task_record_without_any_episode(monkeypatch, tmp_path):
    record = tmp_path / "task-qualification.json"
    monkeypatch.setattr(qualification, "_harness_commit", lambda: "a" * 40)
    monkeypatch.setattr(qualification, "_records", lambda kind: [record] if kind == "task" else [])
    monkeypatch.setattr(qualification, "live_identity", lambda: {"gpu_uuid": "GPU-1", "environment": {}})
    monkeypatch.setattr(qualification, "task_record_admitted", lambda task, path, **kwargs: {"reused": True})
    monkeypatch.setattr(qualification, "run_with_progress", lambda *a, **k: pytest.fail("must reuse"))
    assert qualification.robodojo_qualification("general_pickup", requalify=False,
                                                 confirm=lambda m: pytest.fail("nothing to confirm"),
                                                 stream=io.StringIO()) == {"reused": True}


def test_confirmation_requires_yes_off_a_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(RuntimeError, match="--yes"):
        services.confirm("scan", assume_yes=False, stream=io.StringIO())
    services.confirm("scan", assume_yes=True, stream=io.StringIO())
    services.confirm("scan", assume_yes=False, stream=io.StringIO(), reader=lambda prompt: "y")
    with pytest.raises(RuntimeError, match="not confirmed"):
        services.confirm("scan", assume_yes=False, stream=io.StringIO(), reader=lambda prompt: "")


def test_progress_runner_returns_stdout_and_reports_failures_with_the_log(tmp_path):
    stream = io.StringIO()
    out = services.run_with_progress([sys.executable, "-c", "import sys; print('/record.json'); "
                                      "print('noise', file=sys.stderr)"],
                                     label="probe", estimate="instant", log=tmp_path / "probe.log",
                                     stream=stream, poll_s=0.01)
    assert out.strip() == "/record.json" and "probe: done" in stream.getvalue()
    assert "noise" in (tmp_path / "probe.log").read_text()
    with pytest.raises(RuntimeError, match=r"probe failed .*exit 3.*\n.*boom"):
        services.run_with_progress([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); "
                                    "sys.exit(3)"], label="probe", estimate="instant",
                                   log=tmp_path / "fail.log", stream=io.StringIO(), poll_s=0.01)


def _fake_operator(monkeypatch, tmp_path, *, ready_after: int, reports=()):
    from ora.agent.model_services import operator
    from ora.orchestrator import model_services_cli

    fields = {key: 1 for key in operator.FIELDS}
    calls = {"health": 0, "start": 0, "qualify": 0}

    def health():
        calls["health"] += 1
        ready = calls["health"] > ready_after
        return {name: {"ready": ready, "health": dict(fields)} for name in operator.NAMES}

    def lifecycle(command, **kwargs):
        calls[command] += 1
        return {name: {"status": "starting", "log": f"{name}.log"} for name in operator.NAMES}

    def qualify(directory):
        calls["qualify"] += 1
        return directory / "new.json"

    def binding(path, surface, capabilities):
        if path.name not in reports:
            raise ValueError("stale")
        return {"services": {name: {**fields, "qualification": {"created_unix_s": __import__("time").time()}}
                             for name in operator.NAMES}}

    monkeypatch.setattr(operator, "health", health)
    monkeypatch.setattr(operator, "lifecycle", lifecycle)
    monkeypatch.setattr(operator, "qualify", qualify)
    monkeypatch.setattr(operator, "binding", binding)
    monkeypatch.setattr(model_services_cli, "state_directory", lambda explicit: tmp_path)
    (tmp_path / "qualification").mkdir()
    for name in ("stale.json", *reports):
        (tmp_path / "qualification" / name).write_text("{}")
    return calls


def test_services_start_wait_and_qualify_when_nothing_answers(monkeypatch, tmp_path):
    calls = _fake_operator(monkeypatch, tmp_path, ready_after=2)
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    report = services.ensure_services(stream=io.StringIO(), poll_s=0)
    assert report == tmp_path / "qualification/new.json"
    assert (calls["start"], calls["qualify"]) == (1, 1)


def test_running_services_reuse_the_report_naming_their_instances(monkeypatch, tmp_path):
    calls = _fake_operator(monkeypatch, tmp_path, ready_after=0, reports=("current.json",))
    report = services.ensure_services(stream=io.StringIO())
    assert report == tmp_path / "qualification/current.json"
    assert (calls["start"], calls["qualify"]) == (0, 0)


def test_main_prepares_then_forwards_to_interactive_with_explicit_authorization(monkeypatch, tmp_path, capsys):
    from ora.orchestrator import ora_cli

    monkeypatch.setattr(ora_run, "prepare_environment", lambda environment, env_file: {})
    monkeypatch.setattr(ora_run, "require_clean_checkout", lambda: None)
    monkeypatch.setattr(services, "ensure_services", lambda stream: tmp_path / "report.json")
    monkeypatch.setattr(ora_run, "compose", lambda args, services, **kwargs: services() and tmp_path / "spec.json")
    forwarded = []
    monkeypatch.setattr(ora_cli, "main", lambda argv: forwarded.append(argv) or 0)
    assert ora_run.main(["--env", "libero", "--task", "libero_goal/0", "--prepare-only"],
                        stream=io.StringIO()) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "spec.json") and not forwarded
    assert ora_run.main(["--env", "libero", "--task", "libero_goal/0", "--confirm-launch", "--emit", "jsonl"],
                        stream=io.StringIO()) == 0
    argv = forwarded[0]
    assert argv[:3] == ["interactive", "--spec", str(tmp_path / "spec.json")]
    assert "--confirm-launch" in argv and argv[-2:] == ["--emit", "jsonl"] and "--world-port" not in argv
    assert ora_run.main(["--env", "libero", "--task", "libero_goal/0"], stream=io.StringIO()) == 0
    assert "--confirm-launch" not in forwarded[1]


def test_main_reports_a_refusal_without_a_traceback(monkeypatch, capsys):
    def refuse(environment, env_file):
        raise RuntimeError("DEEPSEEK_BASE_URL and DEEPSEEK_LOW_API_KEY unset")

    monkeypatch.setattr(ora_run, "prepare_environment", refuse)
    assert ora_run.main(["--env", "robotwin", "--task", "click_bell"], stream=io.StringIO()) == 2
    assert "ora run: DEEPSEEK_BASE_URL" in capsys.readouterr().err


def test_ora_cli_routes_run(monkeypatch):
    from ora.orchestrator import ora_cli

    seen = []
    monkeypatch.setattr(ora_run, "main", lambda argv: seen.append(argv) or 0)
    assert ora_cli.main(["run", "--env", "robodojo", "--task", "general_pickup"]) == 0
    assert seen == [["--env", "robodojo", "--task", "general_pickup"]]


def test_robodojo_records_must_name_this_gpu_and_interpreter(monkeypatch, tmp_path):
    from ora.orchestrator import robodojo_qualification, robodojo_spec

    record = {"device_binding": {"selected_gpu_uuid": "GPU-1"}, "environment": {"versions": {"torch": "2.7.0+cu128"}}}
    monkeypatch.setattr(robodojo_spec, "layered_qualification", lambda path: {"card": {}, "task": {}})
    monkeypatch.setattr(qualification, "_robodojo_runtime", lambda task: {})
    monkeypatch.setattr(robodojo_qualification, "load_qualification", lambda runtime, **kwargs: record)
    same = {"gpu_uuid": "GPU-1", "environment": {"versions": {"torch": "2.7.0+cu128"}}}
    path = tmp_path / "task-qualification.json"
    assert qualification.task_record_admitted("general_pickup", path, commit="c", identity=same) is not None
    assert qualification.task_record_admitted("general_pickup", path, commit="c",
                                              identity={**same, "gpu_uuid": "GPU-2"}) is None
    assert qualification.task_record_admitted("general_pickup", path, commit="c",
                                              identity={**same, "environment": {"versions": {}}}) is None


def test_robotwin_scan_without_an_accepted_seed_says_how_to_widen_it(monkeypatch, tmp_path):
    from ora.orchestrator import robotwin_evaluation

    plan = {"qualification": {"estimated_wall_minutes": 37.5}, "plan_sha256": "abc"}
    monkeypatch.setattr(robotwin_evaluation, "WORKSPACE", tmp_path)
    monkeypatch.setattr(robotwin_evaluation, "prepare", lambda **kwargs: plan)
    monkeypatch.setattr(robotwin_evaluation, "save", lambda path, value: None)
    monkeypatch.setattr(qualification, "_log", lambda name: tmp_path / f"{name}.log")
    accepted = []

    def scan(command, **kwargs):
        accepted.append(kwargs["accepted_codes"])
        record = tmp_path / "qualification/run-1"
        (record / "seed-manifests").mkdir(parents=True)
        (record / "prepared-plan.json").write_text(json.dumps({"plan_sha256": "abc"}))
        (record / "seed-manifests/click_bell.json").write_text(json.dumps({"accepted_seeds": []}))
        return ""

    monkeypatch.setattr(qualification, "run_with_progress", scan)
    with pytest.raises(RuntimeError, match="larger --candidates"):
        qualification.robotwin_seed("click_bell", task_config="demo_randomized", seed=None, candidates=4,
                                    requalify=True, confirm=lambda message: None,
                                    environ={"ROBOTWIN_ROOT": "/r", "ROBOTWIN_PYTHON": "/p"}, stream=io.StringIO())
    assert accepted == [(0, 2)]


def test_an_old_report_is_not_reused_for_a_new_session(monkeypatch, tmp_path):
    import time as clock
    from ora.agent.model_services import operator

    fields = {key: 1 for key in operator.FIELDS}
    created = {"value": clock.time() - 21 * 3600}
    monkeypatch.setattr(operator, "binding", lambda path, surface, capabilities: {"services": {
        name: {**fields, "qualification": {"created_unix_s": created["value"]}} for name in operator.NAMES}})
    (tmp_path / "report.json").write_text("{}")
    health = {name: {"ready": True, "health": dict(fields)} for name in operator.NAMES}
    assert services.reusable_report(tmp_path, health) is None
    created["value"] = clock.time() - 3600
    assert services.reusable_report(tmp_path, health) == tmp_path / "report.json"
