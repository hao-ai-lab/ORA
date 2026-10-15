"""The official-suite example composes both benchmarks on the default baseline, offline.

``examples/official_suite`` selects the official LIBERO-Pro core cells and the
RoboTwin cells a sealed seed manifest admits; every RunSpec names no ToolSurface,
prompt bundle or Context, so the default resolver composes the default catalog's
instance for each benchmark. No launch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.official_suite import run, scheduler
from ora.agent.composition.baseline import BASELINE_ID, load_native_baseline
from ora.orchestrator.spec import resolve_run_spec

ROOT = Path(__file__).resolve().parents[2]
SERVO_KEYS = {"servo_profile", "servo_worker_abi", "servo_qualification", "episode_physics_budget"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest(directory: Path, task: str, task_config: str, seeds, complete=True):
    body = {"schema": run.SEED_MANIFEST_SCHEMA, "task": task, "task_config": task_config,
            "robotwin_revision": "cf4872cc90538fd4bca761c8db912d8f65e49f2b", "source_commit": "a" * 40,
            "plan_sha256": "b" * 64, "accepted_seeds": list(seeds), "complete": complete,
            "oracle_success": True, "reset_predicate_false": True, "terminal_stability_steps": 250}
    body["manifest_sha256"] = _digest(body)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{task}.json").write_text(json.dumps(body))


@pytest.fixture
def suite_workspace(tmp_path, monkeypatch):
    from ora.orchestrator import spec as spec_module

    monkeypatch.setattr(run, "workspace", lambda: tmp_path / "outputs" / run.EXPERIMENT)
    monkeypatch.setattr(spec_module, "resolve_workspace_root", lambda root: tmp_path)
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    return tmp_path


def test_libero_selection_is_the_official_core_evaluation():
    cells = run.libero_cells()
    assert len(cells) == 8000
    assert len({cell.config for cell in cells}) == 8000
    first = cells[0]
    assert first.config == "lp-libero_object_swap-t00-s0" and first.arm == "libero_pro"
    assert first.instruction and first.runtime == {"suite": "libero_object_swap", "task_id": 0}
    subset = run.libero_cells(families=["spatial"], perturbations=["position"], init_states=5)
    assert [cell.seed for cell in subset[:5]] == [0, 1, 2, 3, 4] and len(subset) == 50


def test_robotwin_selection_follows_sealed_seed_manifests(tmp_path):
    manifests = tmp_path / "seed-manifests"
    _manifest(manifests, "adjust_bottle", "demo_clean", [100000, 100002, 100005])
    _manifest(manifests, "place_a2b_left", "demo_clean", [100001])
    cells = run.robotwin_cells(task_config="demo_clean", manifests=manifests, tasks=["adjust_bottle"])
    assert [cell.seed for cell in cells] == [100000, 100002, 100005]
    assert cells[0].config == "rt-clean-adjust_bottle-s100000"
    assert cells[0].runtime == {"task_config": "demo_clean"}
    assert cells[0].instruction == run.load_instructions()["adjust_bottle"]["instruction"]
    assert run.robotwin_cells(task_config="demo_clean", manifests=manifests, tasks=["adjust_bottle"],
                              accepted_limit=2)[-1].seed == 100002
    with pytest.raises(RuntimeError, match="names the objects"):
        run.robotwin_cells(task_config="demo_clean", manifests=manifests, tasks=["place_a2b_left"])
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"episodes": {"place_a2b_left:s100001": {"instruction": "Place the red cup left of the blue box."}}}))
    named = run.robotwin_cells(task_config="demo_clean", manifests=manifests, tasks=["place_a2b_left"], episode_names=names)
    assert named[0].instruction == "Place the red cup left of the blue box."
    with pytest.raises(RuntimeError, match="not a demo_randomized seed manifest"):
        run.robotwin_cells(task_config="demo_randomized", manifests=manifests, tasks=["adjust_bottle"])
    tampered = json.loads((manifests / "adjust_bottle.json").read_text())
    tampered["accepted_seeds"].append(100009)
    (manifests / "adjust_bottle.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="digest"):
        run.robotwin_cells(task_config="demo_clean", manifests=manifests, tasks=["adjust_bottle"])


def test_instruction_table_covers_the_roster_and_names_seed_named_tasks():
    from ora.environment.benchmarks.robotwin_instructions import public_instructions

    table = run.load_instructions()
    assert table == public_instructions()
    assert len(table) == 50
    seed_named = sorted(task for task, entry in table.items() if entry.get("seed_named"))
    assert seed_named == ["place_a2b_left", "place_a2b_right"]
    assert all(entry["instruction"] for task, entry in table.items() if task not in seed_named)


def test_libero_cell_composes_the_default_instance_without_servo(suite_workspace):
    cell = run.libero_cells(families=["object"], perturbations=["position"], init_states=1)[0]
    spec = run.build_spec(cell, "libero-gpu0")
    assert "tool_surface" not in spec["policy"] and "baseline" not in spec["policy"]
    assert "policy_context" not in spec and "session" not in spec
    assert not SERVO_KEYS & set(spec["runtime"])
    assert spec["runtime"]["expected_task_language"] == cell.instruction
    assert spec["qualification"] == {"required": False}
    assert spec["launch_admission"]["mode"] == "single_attempt_cell"
    plan = resolve_run_spec(spec, root=ROOT)
    instance = load_native_baseline(ROOT)["instances"]["libero_pro"]
    baseline = plan["policy"]["baseline"]
    assert (baseline["id"], baseline["selection"]) == (BASELINE_ID, "omitted_tool_surface")
    assert plan["policy"]["tool_surface"]["profile"] == instance["tool_surface"]
    assert plan["runtime"]["environment_driver"] == "direct_libero_v1"
    assert plan["runtime"]["resource_binding"]["profile_id"] == "libero-gpu0"
    assert plan["policy"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("task_config", ["demo_clean", "demo_randomized"])
def test_robotwin_cell_composes_the_default_instance_and_delivers_its_contract(suite_workspace, task_config):
    manifests = suite_workspace / "seed-manifests"
    _manifest(manifests, "rotate_qrcode", task_config, [100003])
    cell = run.robotwin_cells(task_config=task_config, manifests=manifests, tasks=["rotate_qrcode"])[0]
    spec = run.build_spec(cell, "robotwin-gpu1")
    assert spec["qualification"]["required"] is True
    assert spec["runtime"]["task"] == spec["task"] == "rotate_qrcode"
    imported = run.workspace() / "qualification" / f"rotate_qrcode-{task_config}-seed-manifest.json"
    assert imported.read_bytes() == (manifests / "rotate_qrcode.json").read_bytes()
    assert "baseline" not in spec["policy"]
    plan = resolve_run_spec(spec, root=ROOT)
    instance = load_native_baseline(ROOT)["instances"]["robotwin"]
    assert plan["policy"]["baseline"]["selection"] == "omitted_tool_surface"
    assert plan["policy"]["tool_surface"]["profile"] == instance["tool_surface"]
    assert plan["policy"]["task_contract"]["delivered"] is True
    assert plan["runtime"]["task_config"] == task_config
    assert plan["runtime"]["resource_binding"]["profile_id"] == "robotwin-gpu1"
    assert plan["policy"]["reasoning_effort"] == "high"


def test_provider_override_and_host_requirement(suite_workspace, monkeypatch):
    cell = run.libero_cells(families=["goal"], perturbations=["task"], init_states=1)[0]
    spec = run.build_spec(cell, "libero-gpu0", provider_profile="deepseek_flash_openai_high_16k_v1")
    assert spec["policy"]["provider_profile"] == "deepseek_flash_openai_high_16k_v1"
    assert spec["policy"]["reasoning_effort"] == "high"
    with pytest.raises(RuntimeError, match="not registered"):
        run.build_spec(cell, "libero-gpu0", provider_profile="no-such-profile")
    monkeypatch.delenv("ORA_EXECUTION_HOST")
    with pytest.raises(RuntimeError, match="execution host"):
        run.build_spec(cell, "libero-gpu0")
    assert run.build_spec(cell, "libero-gpu0", host="test-arm")["runtime"]["required_host"] == "test-arm"


def test_queue_orders_cells_and_resumes_past_sealed_ones(suite_workspace):
    cells = run.libero_cells(families=["object"], perturbations=["object"], init_states=3)
    groups = run.workspace() / "groups" / "earlier"
    groups.mkdir(parents=True)
    (groups / "status.json").write_text(json.dumps({
        "cohort_id": run.definition()["cohort_id"],
        "cells": {cells[0].config: {"state": "sealed_terminal"}, cells[1].config: {"state": "blocked"}}}))
    queued = scheduler.ordered(cells, ["libero-gpu0", "libero-gpu1"], resume=True, cohort_id=run.definition()["cohort_id"])
    assert [cell.config for cell in queued] == [cell.config for cell in cells[1:]]
    assert scheduler.ordered(cells, ["libero-gpu0"], resume=False, cohort_id="x") == cells
    with pytest.raises(RuntimeError, match="distinct"):
        scheduler.ordered(cells, ["libero-gpu0", "libero-gpu0"], resume=False, cohort_id="x")
    with pytest.raises(RuntimeError, match="readiness report"):
        scheduler.run_queue(cells, lanes=["libero-gpu0"], group_id="g", service_report=None)


def test_cli_lists_and_composes(suite_workspace, capsys):
    assert run.main(["cells", "--arm", "libero_pro", "--families", "spatial", "--perturbations", "language",
                     "--init-states", "2"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 20 and listed["configs"][0] == "lp-libero_spatial_lan-t00-s0"
    assert run.main(["compose", "--arm", "libero_pro", "--families", "spatial", "--perturbations", "language",
                     "--init-states", "2", "--config", "lp-libero_spatial_lan-t00-s1", "--lane", "libero-gpu0"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path.name == "lp-libero_spatial_lan-t00-s1.yaml" and path.is_file()
