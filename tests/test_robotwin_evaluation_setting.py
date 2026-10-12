"""Host-only RoboTwin official randomized-evaluation contract checks."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest
import yaml

from ora.environment.benchmarks.robotwin import (
    RANDOMIZATION, TASKS, evaluation_protocol, validate_randomized_config,
)
from ora.environment.backends.robotwin import qualification


def test_roster_and_official_seed_domain():
    assert len(TASKS) == len(set(TASKS)) == 50
    assert set(TASKS) == set(qualification.DEFAULT_TASKS)
    protocol = evaluation_protocol(outer_seed=0)
    assert protocol["start_requested_seed"] == 100000
    assert protocol["accepted_trials_per_task"] == 100
    assert protocol["eval_mode"] is True
    assert protocol["is_test"] is True
    assert protocol["background_split"] == "unseen"
    assert evaluation_protocol(outer_seed=3)["start_requested_seed"] == 400000
    for seed in (-1, True, 0.5):
        with pytest.raises(ValueError):
            evaluation_protocol(outer_seed=seed)
    # Both official scene configurations share the loop and differ in randomization.
    from ora.environment.benchmarks.robotwin import CLEAN_RANDOMIZATION, validate_task_config
    clean = evaluation_protocol(task_config="demo_clean")
    assert clean["task_config"] == "demo_clean"
    assert clean["domain_randomization"] == CLEAN_RANDOMIZATION
    assert clean["start_requested_seed"] == protocol["start_requested_seed"]
    assert not any(CLEAN_RANDOMIZATION[key] for key in ("random_background", "cluttered_table", "random_light"))
    with pytest.raises(ValueError, match="task_config"):
        evaluation_protocol(task_config="demo_easy")
    validate_task_config("demo_clean", {"domain_randomization": dict(CLEAN_RANDOMIZATION), "embodiment": ["aloha-agilex"]})
    with pytest.raises(ValueError, match="demo_clean domain_randomization"):
        validate_task_config("demo_clean", {"domain_randomization": dict(RANDOMIZATION), "embodiment": ["aloha-agilex"]})


def test_evaluation_revision_is_the_vendored_install_the_runtime_loads():
    from ora.environment.backends.robotwin import vendored_install
    from ora.orchestrator import robotwin_evaluation as run
    assert run.REQUIRED_REVISION == vendored_install.INSTALL_REVISION


def test_exact_randomization_not_just_a_config_name():
    config = {"domain_randomization": dict(RANDOMIZATION), "embodiment": ["aloha-agilex"]}
    validate_randomized_config(config)
    for field, value in RANDOMIZATION.items():
        changed = copy.deepcopy(config)
        changed["domain_randomization"][field] = not value if isinstance(value, bool) else value + 1
        with pytest.raises(ValueError, match="randomization"):
            validate_randomized_config(changed)


def test_oracle_defaults_to_evaluation_and_has_explicit_collection_compatibility(tmp_path, monkeypatch):
    config_root = tmp_path / "task_config"
    config_root.mkdir()
    robot = tmp_path / "robot"
    robot.mkdir()
    (robot / "config.yml").write_text("{}\n")
    (config_root / "_embodiment_config.yml").write_text(yaml.safe_dump({"aloha-agilex": {"file_path": str(robot)}}))
    (config_root / "demo_randomized.yml").write_text(yaml.safe_dump({
        "embodiment": ["aloha-agilex"], "domain_randomization": dict(RANDOMIZATION),
    }))
    module = types.ModuleType("envs._GLOBAL_CONFIGS")
    module.CONFIGS_PATH = str(config_root)
    monkeypatch.setitem(sys.modules, "envs._GLOBAL_CONFIGS", module)
    kwargs = dict(robotwin_root=tmp_path, task="rotate_qrcode", task_config="demo_randomized", runtime_output=tmp_path / "output")
    resolved = qualification._load_task_args(**kwargs)
    assert resolved["eval_mode"] is True
    assert resolved["is_test"] is True
    assert resolved["domain_randomization"] == RANDOMIZATION
    assert qualification._load_task_args(**kwargs, evaluation_mode=False)["eval_mode"] is False


@pytest.mark.parametrize("raises", [False, True, "stability", "task_code"])
def test_failed_oracle_preserves_private_terminal_diagnostics(tmp_path, monkeypatch, raises):
    class Task:
        plan_success = True
        def setup_demo(self, **kwargs):
            pass
        def check_success(self):
            return raises == "stability" and getattr(self, "played", False)
        def play_once(self):
            self.played = True
            if raises is True:
                raise OSError("fixture: simulator device unavailable")
            if raises == "task_code":  # the expert's own task code asserts, as envs/utils/action.py does
                exec(compile("assert False, 'target_pose cannot be None'",
                             str(tmp_path / "envs" / "utils" / "action.py"), "exec"), {})
        def get_obs(self):
            return {}
        def close_env(self, **kwargs):
            pass

    monkeypatch.setattr(qualification, "_task_instance", lambda task: Task())
    monkeypatch.setattr(qualification, "_load_task_args", lambda **kwargs: {})
    monkeypatch.setattr(qualification, "selected_environment_assets", lambda *a, **kw: {})
    monkeypatch.setattr(qualification, "verifier_telemetry", lambda task: {"private": "terminal"})
    monkeypatch.setattr(qualification, "_save_rgbd", lambda obs, root: [{"camera": "head", "modality": "rgb"}])
    if raises == "stability":
        def broken_hold(*args, **kwargs):
            raise OSError("fixture: physics hold device unavailable")
        monkeypatch.setattr(qualification, "hold_official_predicate", broken_hold)
    from ora.environment.backends.robotwin import session
    monkeypatch.setattr(session, "_restore_expert_only_evaluator_state", lambda *args: None)
    qualification._run_task(
        robotwin_root=tmp_path, task="place_bread_skillet", task_config="demo_randomized",
        output_root=tmp_path / "evidence", accepted_target=1, start_requested_seed=100000,
        max_requested_seeds=100001, stability_steps=1,
        source={"task_config": "demo_randomized", "evaluation_mode": True},
        process_budget=qualification.ProcessAttemptBudget(1),
    )
    import json
    record = json.loads(next((tmp_path / "evidence").rglob("attempt.json")).read_text())
    if raises == "task_code":  # a rejected seed; the scan moves on
        assert record["classification"] == "oracle_exception" and record["accepted"] is False
        assert "target_pose cannot be None" in record["error"]
        status = json.loads((tmp_path / "evidence/place_bread_skillet/status.json").read_text())
        assert status["next_requested_seed"] == 100001 and status["state"] == "failed"
        return
    if raises:
        assert record["classification"] == "infra_failure"
        status = json.loads((tmp_path / "evidence/place_bread_skillet/status.json").read_text())
        assert status["next_requested_seed"] == 100000
        assert status["state"] == "blocked_infrastructure"
        return
    assert record["classification"] == "oracle_failure"
    assert record["accepted"] is False
    assert record["verifier_telemetry"]["sha256"]
    assert record["artifacts"]


def test_process_rotation_flushes_both_streams_before_exec(tmp_path, monkeypatch):
    events = []
    class Stream:
        def __init__(self, name):
            self.name = name
        def flush(self):
            events.append(self.name)
    def replace_process(*args):
        events.append("exec")
        executed.append(args)
        raise RuntimeError("fixture process replaced")
    executed = []
    monkeypatch.setattr(qualification, "sys", types.SimpleNamespace(
        executable=sys.executable, stdout=Stream("stdout"), stderr=Stream("stderr"),
    ))
    monkeypatch.setattr(qualification, "run", lambda args: qualification.PROCESS_ROTATION_EXIT)
    monkeypatch.setattr(qualification.os, "execv", replace_process)
    with pytest.raises(RuntimeError, match="process replaced"):
        qualification.main(["--robotwin-root", str(tmp_path), "--output-root", str(tmp_path / "output"),
                            "--tasks", "rotate_qrcode"])
    assert events == ["stdout", "stderr", "exec"]
    # Re-entered as the module, never as a bare script: qualification.py
    # imports its siblings relatively, so a script path dies on import.
    assert executed[0][1][1:3] == ["-m", "ora.environment.backends.robotwin.qualification"]


def test_selected_clutter_assets_are_part_of_private_evidence(tmp_path):
    model = tmp_path / "assets/objects/objaverse/bottle/0"
    model.mkdir(parents=True)
    (model / "model.urdf").write_text("<robot/>\n")
    task = types.SimpleNamespace(
        wall_texture=None, table_texture=None,
        record_cluttered_objects=[{"object_type": "bottle", "object_index": "0"}],
        cluttered_item_info={"bottle": {"type": "urdf", "root": "objects/objaverse/bottle"}},
    )
    result = qualification.selected_environment_assets(task, robotwin_root=tmp_path, task="blocks_ranking_rgb")
    assert result["selected_clutter_models"][0]["content"]["files"][0]["sha256"]


def test_prepare_all_50_is_offline_and_estimates_actual_parallel_lanes(tmp_path, monkeypatch):
    from ora.orchestrator import robotwin_evaluation as run
    root = tmp_path / "robotwin"
    (root / "task_config").mkdir(parents=True)
    (root / "task_config/demo_randomized.yml").write_text(yaml.safe_dump({"domain_randomization": dict(RANDOMIZATION), "embodiment": ["aloha-agilex"]}))
    tasks_dir = root / "description/task_instruction"
    tasks_dir.mkdir(parents=True)
    for task in TASKS:
        (tasks_dir / f"{task}.json").write_text("{}")
    textures = root / "assets/background_texture/unseen"
    textures.mkdir(parents=True)
    (textures / "0.png").write_bytes(b"fixture")
    monkeypatch.setattr(run, "git", lambda root, *args: "" if args[0] == "status" else run.REQUIRED_REVISION)
    monkeypatch.setattr(run, "source_dependencies", lambda *args, **kwargs: [])
    monkeypatch.setattr(run, "capacity", lambda: {"cpu_count": 16, "available_memory_gib": 64, "gpus": {"6": {"free_memory_mib": 48000}, "7": {"free_memory_mib": 48000}}})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    plan = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["place_bread_skillet", "rotate_qrcode"], slots=["robotwin-gpu6", "robotwin-gpu7"], candidate_count=3)
    assert plan["required_host"] == "test-host"
    assert all(task["runtime"]["required_host"] == "test-host" for task in plan["all_tasks"])
    assert plan["task_config"] == "demo_randomized" and plan["accepted_per_task"] == 1
    assert all(case["accepted_target"] == 1 for case in plan["qualification"]["cases"])
    assert len(plan["all_tasks"]) == 50
    assert all(task["runtime"]["task_config"] == "demo_randomized" for task in plan["all_tasks"])
    assert all(task["qualification_status"] == "not_bound" for task in plan["all_tasks"])
    assert plan["formal_launch_authorized"] is False
    assert plan["qualification"]["estimated_wall_minutes"] == 28.5
    assert plan["qualification"]["purpose"] == "development_diagnostic"
    assert len(plan["qualification"]["parallel_execution"]["lanes"]) == 2
    assert all(case["repeats"] == 1 for case in plan["qualification"]["cases"])


def test_generic_configs_are_host_safe_and_have_no_experiment_defaults(monkeypatch, capsys):
    import json
    from ora.orchestrator import robotwin_evaluation as run
    monkeypatch.setattr(run, "capacity", lambda: pytest.fail("configs must not inspect GPUs"))
    assert run.main(["configs", "--outer-seed", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["tasks"]) == 50
    assert result["protocol"]["start_requested_seed"] == 300000
    with pytest.raises(SystemExit) as exc:
        run.main(["prepare", "--robotwin-root", "/fixture", "--robotwin-python", sys.executable])
    assert exc.value.code == 2


def _official_robotwin_root(tmp_path):
    from ora.environment.benchmarks.robotwin import CLEAN_RANDOMIZATION
    root = tmp_path / "robotwin"
    (root / "task_config").mkdir(parents=True)
    for name, randomization in (("demo_randomized", RANDOMIZATION), ("demo_clean", CLEAN_RANDOMIZATION)):
        (root / f"task_config/{name}.yml").write_text(yaml.safe_dump({"domain_randomization": dict(randomization), "embodiment": ["aloha-agilex"]}))
    tasks_dir = root / "description/task_instruction"
    tasks_dir.mkdir(parents=True)
    for task in TASKS:
        (tasks_dir / f"{task}.json").write_text("{}")
    textures = root / "assets/background_texture/unseen"
    textures.mkdir(parents=True)
    (textures / "0.png").write_bytes(b"fixture")
    return root


def test_prepare_scans_n_accepted_seeds_per_task_under_one_scene_configuration(tmp_path, monkeypatch):
    """The official protocol keeps the first 100 expert-accepted seeds per task and configuration."""
    from ora.orchestrator import robotwin_evaluation as run
    root = _official_robotwin_root(tmp_path)
    monkeypatch.setattr(run, "git", lambda root, *args: "" if args[0] == "status" else run.REQUIRED_REVISION)
    closures = []
    monkeypatch.setattr(run, "source_dependencies", lambda root, tasks, task_config: closures.append(task_config) or [])
    monkeypatch.setattr(run, "capacity", lambda: {"cpu_count": 16, "available_memory_gib": 64, "gpus": {"6": {"free_memory_mib": 48000}}})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    plan = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"], slots=["robotwin-gpu6"],
                       candidate_count=200, accepted_per_task=100, task_config="demo_clean")
    assert plan["task_config"] == "demo_clean" and plan["accepted_per_task"] == 100
    assert plan["protocol"]["task_config"] == "demo_clean"
    assert plan["official_config_source"]["path"].endswith("task_config/demo_clean.yml")
    assert all(task["runtime"]["task_config"] == "demo_clean" for task in plan["all_tasks"])
    case = plan["qualification"]["cases"][0]
    assert case["accepted_target"] == 100
    assert case["start_requested_seed"] == 100000 and case["exclusive_seed_limit"] == 100200
    clean_key = case["case_key"]
    randomized = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"], slots=["robotwin-gpu6"],
                             candidate_count=200, accepted_per_task=100, task_config="demo_randomized")
    assert randomized["qualification"]["cases"][0]["case_key"] != clean_key
    assert closures == ["demo_clean", "demo_randomized"]
    with pytest.raises(ValueError, match="at least accepted_per_task"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"], slots=["robotwin-gpu6"],
                    candidate_count=50, accepted_per_task=100)
    with pytest.raises(ValueError, match="task_config"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"], slots=["robotwin-gpu6"],
                    candidate_count=1, task_config="demo_easy")


def test_generic_prepare_prints_bound_cost_and_confirmation(tmp_path, monkeypatch, capsys):
    import json
    from ora.orchestrator import robotwin_evaluation as run
    monkeypatch.setattr(run, "WORKSPACE", tmp_path)
    calls = []
    def prepared(**kwargs):
        calls.append(kwargs)
        return {"all_tasks": list(TASKS), "plan_sha256": "fixture",
                "qualification": {"cases": [], "purpose": "targeted_regression",
                                  "estimated_wall_minutes": 10.5}}
    monkeypatch.setattr(run, "prepare", prepared)
    assert run.main(["prepare", "--robotwin-root", "/fixture", "--robotwin-python", sys.executable,
                     "--tasks", "rotate_qrcode", "--slots", "robotwin-gpu0",
                     "--candidate-count", "1", "--accepted-per-task", "1", "--task-config", "demo_clean"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["estimated_wall_minutes"] == 10.5
    assert result["confirmation"] == "QUALIFY-ROBOTWIN:fixture"
    assert calls[0]["slots"] == ["robotwin-gpu0"]
    assert calls[0]["candidate_count"] == 1
    assert calls[0]["accepted_per_task"] == 1 and calls[0]["task_config"] == "demo_clean"
    assert json.loads(Path(result["plan"]).read_text())["plan_sha256"] == "fixture"


@pytest.mark.parametrize("kind", ["tampered", "legacy", "wrong_confirmation"])
def test_generic_qualify_rejects_unbound_plan_before_execution(tmp_path, monkeypatch, kind):
    import json
    from ora.orchestrator import robotwin_evaluation as run
    plan = {"schema": "ora.robotwin-evaluation-preparation.v1"}
    if kind == "legacy":
        plan["schema"] = "ora.robotwin-random50-preparation.v1"
    checksum = run.digest(plan)
    plan["plan_sha256"] = checksum
    if kind == "tampered":
        plan["source_commit"] = "changed"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(run, "capacity", lambda: pytest.fail("must reject before execution"))
    confirm = "wrong" if kind == "wrong_confirmation" else f"QUALIFY-ROBOTWIN:{checksum}"
    with pytest.raises(ValueError, match="confirmation or plan hash"):
        run.qualify(path, confirm)


@pytest.mark.parametrize("timeout", [False, True])
def test_generic_qualify_parallel_bindings_and_lease_cleanup(tmp_path, monkeypatch, timeout):
    import json
    import threading
    from ora.orchestrator import cli
    from ora.orchestrator import robotwin_evaluation as run
    tasks = ["place_bread_skillet", "rotate_qrcode"]
    lanes = [{"tasks": [task], "resource_binding": run.resolve_execution_resource(
        f"robotwin-gpu{index}", runtime_kind="robotwin", required_host="test-host")}
        for index, task in enumerate(tasks)]
    plan = {"schema": "ora.robotwin-evaluation-preparation.v1",
            "source_commit": "fixture", "source_dirty": False,
            "robotwin_root": str(tmp_path), "robotwin_python": sys.executable,
            "robotwin_revision": "fixture",
            "task_config": "demo_clean",
            "qualification": {"dependency_closure": [], "producer_dependency_hash": "deps",
                              "cases": [{"task": task, "start_requested_seed": 100000,
                                         "exclusive_seed_limit": 100003, "accepted_target": 2} for task in tasks],
                              "parallel_execution": {"lanes": lanes}}}
    checksum = run.digest(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**plan, "plan_sha256": checksum}))
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(run, "WORKSPACE", tmp_path / "outputs")
    monkeypatch.setattr(run, "git", lambda root, *args: "" if args[0] == "status" else "fixture")
    monkeypatch.setattr(run, "capacity", lambda: {
        "gpus": {str(index): {"free_memory_mib": 48000} for index in range(2)}})
    claims, released, calls = [], [], []
    def claim(plan, owner, pid):
        claims.append((plan, owner))
        return {"claim_token": owner}
    monkeypatch.setattr(cli, "claim_resource_lease", claim)
    monkeypatch.setattr(cli, "release_resource_lease",
                        lambda owner, claim_token: released.append((owner, claim_token)))
    barrier = threading.Barrier(2, timeout=10)
    def worker(command, **kwargs):
        calls.append((command, kwargs))
        barrier.wait()
        if timeout:  # the seed watchdog gave up on the case
            return 124, [100000]
        output = Path(command[command.index("--output-root") + 1])
        task = command[command.index("--tasks") + 1]
        (output / "summary.json").write_text(json.dumps({"passed": True}))
        (output / "run-manifest.json").write_text(json.dumps({"terminal_stability_physics_steps": 250}))
        ledger = output / task / "seed-ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text("\n".join(json.dumps(record) for record in (
            _ledger_record(100000, accepted=True, hold=(True, 250, None)),
            _ledger_record(100001, accepted=False, classification="oracle_failure"),
            # Upstream collects a seed whose success state does not hold; the worker does too.
            _ledger_record(100002, accepted=True, hold=(False, 9, 9)),
        )) + "\n")
        return 0, []
    from ora.orchestrator import robotwin_seed_lanes
    monkeypatch.setattr(robotwin_seed_lanes, "run_with_seed_watchdog", worker)
    monkeypatch.setattr(robotwin_seed_lanes, "LANE_START_STAGGER_S", 0.01)
    assert run.qualify(path, f"QUALIFY-ROBOTWIN:{checksum}") == (2 if timeout else 0)
    assert len(claims) == len(released) == len(calls) == 2
    assert {owner for _, owner in claims} == {owner for owner, _ in released}
    for command, kwargs in calls:
        assert command[command.index("--scene-mode") + 1] == "evaluation"
        assert command[command.index("--task-config") + 1] == "demo_clean"
        assert command[command.index("--accepted-per-task") + 1] == "2"
        assert command[command.index("--max-requested-seeds") + 1] == "100003"
        assert command[command.index("--process-attempt-limit") + 1] == "1"  # plan predates the field
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"0", "1"}
    report = json.loads(next((tmp_path / "outputs").rglob("qualification-report.json")).read_text())
    assert report["formal_runs_launched"] == 0
    assert len(report["results"]) == 2
    assert {tuple(item["oracle_timeouts"]) for item in report["results"]} == {(100000,) if timeout else ()}
    if timeout:
        assert report["seed_manifests"] == {}
        return
    assert set(report["seed_manifests"]) == set(tasks)
    manifest = json.loads(Path(report["seed_manifests"]["rotate_qrcode"]).read_text())
    assert manifest["schema"] == run.SEED_MANIFEST_SCHEMA
    assert manifest["task"] == "rotate_qrcode" and manifest["task_config"] == "demo_clean"
    assert manifest["accepted_seeds"] == [100000, 100002]
    assert manifest["evaluated_seeds"] == [100000, 100001, 100002]
    assert manifest["rejected"] == [{"requested_seed": 100001, "classification": "oracle_failure", "error": None}]
    assert manifest["complete"] is True and manifest["accepted_target"] == 2
    # Attestations are derived from the ledger, never literals.
    assert manifest["seeds"]["100000"] == {"reset_predicate_false": True, "oracle_success": True,
                                          "post_success_physics_hold": True, "terminal_stability_steps": 250}
    assert manifest["seeds"]["100002"]["post_success_physics_hold"] is False
    assert manifest["seeds"]["100002"]["terminal_stability_steps"] == 9
    assert manifest["oracle_success"] is True and manifest["reset_predicate_false"] is True
    assert manifest["post_success_physics_hold"] is False and manifest["unstable_accepted_seeds"] == [100002]
    assert manifest["required_stability_steps"] == 250 and manifest["terminal_stability_steps"] == 9
    assert manifest["manifest_sha256"] == run.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    assert manifest["robotwin_revision"] == "fixture" and manifest["plan_sha256"] == checksum
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == run.digest(body)


def test_prepare_binds_the_declared_host_and_refuses_an_unregistered_one(tmp_path, monkeypatch):
    from ora.orchestrator import robotwin_evaluation as run
    from ora.orchestrator.resources import ExecutionResourceError

    monkeypatch.delenv("ORA_EXECUTION_HOST", raising=False)
    with pytest.raises(ExecutionResourceError, match="is not set"):
        run.prepare(robotwin_root=tmp_path, python=Path(sys.executable), tasks=["rotate_qrcode"],
                    slots=["robotwin-gpu6"], candidate_count=1)
    monkeypatch.setenv("ORA_EXECUTION_HOST", "no-such-cluster")
    with pytest.raises(ExecutionResourceError, match="unavailable"):
        run.prepare(robotwin_root=tmp_path, python=Path(sys.executable), tasks=["rotate_qrcode"],
                    slots=["robotwin-gpu6"], candidate_count=1)
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    with pytest.raises(ExecutionResourceError, match="does not match"):
        run.prepare(robotwin_root=tmp_path, python=Path(sys.executable), tasks=["rotate_qrcode"],
                    slots=["robotwin-gpu6"], candidate_count=1, required_host="test-arm")


def _ledger_record(seed, *, accepted, classification="accepted", hold=None):
    """A worker ledger record in the shape ``backends.robotwin.qualification`` writes."""
    record = {"requested_seed": seed, "accepted": accepted, "classification": classification, "error": None,
              "reset_official_predicate": False, "stability": {"required_steps": 250}}
    if classification != "reset_predicate_true":
        record["expert"] = {"plan_success": accepted or classification != "oracle_failure",
                            "post_play_official_predicate": accepted, "eval_success_seen": accepted}
    if hold is not None:
        passed, checks, first_false = hold
        record["stability"].update({"passed": passed, "checks": checks, "first_false_step": first_false})
    return record


def test_source_dependencies_bind_the_selected_scene_configuration(tmp_path):
    from ora.orchestrator import robotwin_evaluation as run
    root = tmp_path / "robotwin"
    for relative in ("envs/rotate_qrcode.py", "envs/_base_task.py", "task_config/demo_clean.yml",
                     "task_config/demo_randomized.yml", "task_config/_embodiment_config.yml",
                     "task_config/_camera_config.yml", "task_config/_eval_step_limit.yml",
                     "script/eval_policy.py", "assets/objects/objaverse/list.json", "assets/objects/same.json"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x")
    for folder in ("envs/utils", "envs/robot", "envs/camera"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    paths = {item["path"] for item in run.source_dependencies(root, ["rotate_qrcode"], "demo_clean")}
    assert str(root / "task_config/demo_clean.yml") in paths
    assert str(root / "task_config/demo_randomized.yml") not in paths


@pytest.mark.parametrize("mutation", ["none", "task_config", "revision", "digest", "not_accepted", "reset"])
def test_launch_preflight_admits_official_seeds_on_their_own_evidence(tmp_path, mutation):
    """An official manifest binds configuration and revision; the hold is evidence, not admission."""
    import json
    from ora.orchestrator import preflight
    from ora.orchestrator import robotwin_evaluation as run
    manifest = {"schema": run.SEED_MANIFEST_SCHEMA, "task": "rotate_qrcode", "task_config": "demo_clean",
                "robotwin_revision": "r" * 40, "accepted_seeds": [100002],
                "seeds": {"100002": {"reset_predicate_false": True, "oracle_success": True,
                                     "post_success_physics_hold": False, "terminal_stability_steps": 9}},
                "post_success_physics_hold": False, "terminal_stability_steps": 9}
    plan = {"task": "rotate_qrcode", "seed": 100002, "experiment": "fixture",
            "runtime": {"kind": "robotwin", "task_config": "demo_clean", "required_revision": "r" * 40},
            "qualification": {"required": True, "manifest": {"path": str(tmp_path / "manifest.json")}}}
    if mutation == "task_config":
        plan["runtime"]["task_config"] = "demo_randomized"
    elif mutation == "revision":
        plan["runtime"]["required_revision"] = "s" * 40
    elif mutation == "not_accepted":
        plan["seed"] = 100003
    elif mutation == "reset":
        manifest["seeds"]["100002"]["reset_predicate_false"] = False
    manifest["manifest_sha256"] = run.digest(manifest)
    if mutation == "digest":
        manifest["accepted_seeds"] = [100002, 100003]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = preflight._qualification(plan)
    assert result["passed"] is (mutation == "none"), result
    assert result["detail"]["seed"].get("post_success_physics_hold") is False or mutation == "not_accepted"


def test_an_exception_from_task_code_rejects_the_seed_not_the_scan(tmp_path):
    """Upstream collect_data.py counts a play_once exception as a failed seed."""
    root = tmp_path / "RoboTwin"
    (root / "envs" / "utils").mkdir(parents=True)
    def raised_from(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            exec(compile("assert False, 'target_pose cannot be None'", str(path), "exec"), {})
        except AssertionError as exc:
            return exc
    assert qualification._raised_in_task_code(raised_from(root / "envs" / "utils" / "action.py"), root)
    assert not qualification._raised_in_task_code(raised_from(tmp_path / "curobo" / "solver.py"), root)
    assert not qualification._raised_in_task_code(raised_from(tmp_path / "envs_other" / "x.py"), root)
    assert not qualification._raised_in_task_code(RuntimeError("no traceback"), root)
