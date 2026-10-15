"""Host-safe RoboDojo admission, exact binding and isolated construction."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from ora.environment import factory, worker
from ora.environment.benchmarks import robodojo
from ora.environment.definitions import EnvironmentDefinitionError, resolve_runtime_descriptor
from ora.orchestrator import coordinator, preflight, spec
from ora.orchestrator import robodojo_qualification as qualification_module
from ora.orchestrator.robodojo_qualification import (
    WORKER_KERNEL_PATHS, load_infrastructure_qualification,
)
#: Bound at import time, so the autouse stub below never hides the real one.
real_worker_kernel_sha256 = qualification_module.worker_kernel_sha256

#: The qualified RoboDojo worker-kernel content every fixture record claims.
WORKER_KERNEL_DIGEST = "f" * 64


def _identical_admission(measured: str, plan: str) -> dict:
    """What admission reports when the recorded digest is the plan's own."""

    return {
        "mode": "identical_digest", "measured_commit": measured, "plan_commit": plan,
        "recorded_worker_kernel_sha256": WORKER_KERNEL_DIGEST,
        "plan_worker_kernel_sha256": WORKER_KERNEL_DIGEST, "added_paths": [],
    }


@pytest.fixture(autouse=True)
def _qualified_worker_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content binding without a git history behind the fixture commits.

    ``load_infrastructure_qualification`` now admits a record by the content
    of ``WORKER_KERNEL_PATHS`` at the RunPlan's commit rather than by the
    commit id, so every fixture commit resolves to the same qualified digest
    unless a test deliberately moves one of them.
    """

    monkeypatch.setattr(
        qualification_module, "worker_kernel_sha256",
        lambda commit, **_kwargs: WORKER_KERNEL_DIGEST,
    )


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict:
    return {
        "kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
        "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        "task_variant": "arrange_largest_number", "eval_seed": 0, "layout_id": 0,
        "layout_file": "Eval_Layout/RoboDojo/arx_x5/0/arrange_largest_number_0.json",
        "layout_sha256": "a" * 64, "num_envs": 1, "env_config": "arx_x5",
        "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": 1050, "native_actions_per_command": 20,
        "main_camera": "head", "policy_cameras": list(robodojo.PUBLIC_CAMERAS),
        "required_host": "test-host", "require_gpu": True, "physics_device": "cuda:0",
        "renderer_gpu": 2, "cuda_visible_devices": "2", "startup_timeout_s": 600,
        "request_timeout_s": 180, "close_timeout_s": 30,
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        "environment_signal_policy": "lifecycle_only", "expected_task_language": "Pick the item.",
        "infrastructure_qualification": {
            "path": "outputs/robodojo-compat/qualification/fixture.json", "sha256": "b" * 64,
        },
    }


def _qualification(runtime: dict) -> dict:
    return {
        "schema": robodojo.QUALIFICATION_SCHEMA, "passed": True, "harness_commit": "c" * 40,
        "worker_kernel_sha256": WORKER_KERNEL_DIGEST,
        "source_revisions": robodojo.source_revisions(runtime),
        "task": runtime["task_variant"], "eval_seed": runtime["eval_seed"],
        "layout_id": runtime["layout_id"], "layout_file": runtime["layout_file"],
        "layout_sha256": runtime["layout_sha256"], "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"], "configuration": robodojo.configuration_binding(runtime),
        "checks": {name: True for name in robodojo.QUALIFICATION_CHECKS},
        "task_instruction": runtime["expected_task_language"],
        "actual_source_fingerprint": {
            "loaded_files": {"env_cfg/arx_x5.yml": "a" * 64},
            "installed_packages": dict(robodojo.WORKER_PACKAGE_VERSIONS),
            "resolved_environment_config_sha256": "b" * 64,
        },
        "environment": {
            "python_executable": "/fixture/bin/python", "python_binary_sha256": "e" * 64,
            "versions": {**robodojo.WORKER_PACKAGE_VERSIONS, "warp-lang": "1.9.1"},
        },
    }


@pytest.mark.parametrize("field,value", [
    ("required_revision", "d" * 40), ("required_asset_revision", "d" * 40),
    ("task_variant", "pour_liquid_into_cup"), ("layout_id", 1), ("eval_seed", False),
    ("num_envs", True), ("renderer_gpu", False), ("renderer_gpu", 0), ("physics_dt", float("nan")),
    ("startup_timeout_s", float("inf")), ("request_timeout_s", True),
    ("close_timeout_s", 0), ("close_timeout_s", 31), ("unreviewed_extension", True),
    ("layout_sha256", "unpinned"), ("layout_file", "../arrange_largest_number_0.json"),
    ("root_env", "/opt/source"), ("transport", "in_process_direct"),
])
def test_robodojo_runtime_admission_rejects_unqualified_changes(field: str, value: object) -> None:
    runtime = _runtime()
    robodojo.validate_runtime_extension(runtime)
    runtime[field] = value
    with pytest.raises(robodojo.RoboDojoBenchmarkError):
        robodojo.validate_runtime_extension(runtime)


def test_robodojo_layout_selection_is_numeric_source_bound_and_confined(tmp_path: Path) -> None:
    folder = tmp_path / "Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    for number in (10, 2, 30):
        (folder / f"arrange_largest_number_{number}.json").write_text("{}")
    # Another task module's files never enter the ordinal, as in SeedManager.init_eval.
    (folder / "arrange_largest_number_random_0.json").write_text("{}")
    selected = robodojo.selected_layout(tmp_path, task="arrange_largest_number", eval_seed=0, layout_id=0)
    assert selected.name == "arrange_largest_number_2.json"
    # A later ordinal is resolvable; which ordinals a RunPlan may select is the
    # runtime admission's and the reviewed catalog's decision, not this mirror's.
    assert robodojo.selected_layout(tmp_path, task="arrange_largest_number", eval_seed=0,
                                    layout_id=1).name == "arrange_largest_number_10.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="unavailable"):
        robodojo.selected_layout(tmp_path, task="arrange_largest_number", eval_seed=0, layout_id=3)
    for layout_id in (-1, True):
        with pytest.raises(robodojo.RoboDojoBenchmarkError, match="only the reviewed tasks"):
            robodojo.selected_layout(tmp_path, task="arrange_largest_number", eval_seed=0, layout_id=layout_id)
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("{}")
    (folder / "arrange_largest_number_0.json").symlink_to(outside)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="escapes"):
        robodojo.selected_layout(tmp_path, task="arrange_largest_number", eval_seed=0, layout_id=0)


@pytest.mark.parametrize("task,limit", [("hang_mugs", 800), ("pack_objects_into_box", 1300)])
def test_followup_tasks_require_their_own_layout_and_native_limit(task, limit):
    runtime = _runtime()
    runtime.update(task_variant=task, layout_file=f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json",
                   native_action_limit=limit)
    robodojo.validate_runtime_extension(runtime)
    runtime["native_action_limit"] = 1050
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="native_action_limit"):
        robodojo.validate_runtime_extension(runtime)
    runtime["native_action_limit"] = limit
    runtime["layout_file"] = "Eval_Layout/RoboDojo/arx_x5/0/arrange_largest_number_0.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="selected task"):
        robodojo.validate_runtime_extension(runtime)


#: The 42 base RoboDojo simulation tasks at the pinned revision
#: (``scripts/internal/task_inventory.py`` ``DIMENSION_TASKS``), each with the
#: ``self.step_lim`` literal of its ``task/RoboDojo/tasks/<task>.py``. Written
#: out independently of ``NATIVE_ACTION_LIMITS`` so the table cannot vouch for
#: itself; the mounted-source test below re-reads every number upstream.
OFFICIAL_SIM_TASK_LIMITS = {
    "align_blocks": 200, "arrange_largest_number": 1050, "build_tower": 1050,
    "classify_objects": 1100, "classify_objects_by_language": 1100, "cover_blocks": 800,
    "deposit_coin": 300, "fasten_screws": 1900, "fill_egg_holder": 700, "fill_pen_holder": 1100,
    "fold_clothes": 500, "general_pickup": 200, "hang_mugs": 800, "imitate_sorting_sequence": 1600,
    "insert_key": 300, "insert_tubes": 500, "make_kong": 600, "make_toast": 1400,
    "match_and_pick_from_conveyor": 700, "organize_table": 1000, "pack_objects_into_box": 1300,
    "pick_from_conveyor_by_image": 700, "play_stacking_toy": 1200, "play_tic_tac_toe": 1100,
    "play_Xylophone": 500, "plug_in_charger": 400, "pour_balls_into_vase": 600,
    "pour_by_language": 800, "pour_liquid_into_cup": 400, "press_by_number": 700, "push_T": 600,
    "put_bottles_into_dustbin": 700, "solve_equation": 300, "sort_nesting_dolls_by_size": 1050,
    "stack_blocks": 550, "stack_blocks_by_language": 400, "stack_bowls": 800,
    "store_laptop_and_headphones": 800, "store_tools_in_toolbox": 900, "swap_blocks": 700,
    "swap_T": 400, "sweep_blocks": 1000,
}
#: Judged on a Fluid object, which upstream never creates on CUDA physics, so
#: both resolve CPU physics; admitted on 2026-10-04 (0059 pour admission) once
#: each had a reviewed contract and a passing fluid physics scene probe.
FLUID_TASKS = ("pour_by_language", "pour_liquid_into_cup")
#: The fifteen tasks admitted before the 2026-10-02 extension.
FIRST_FIFTEEN = (
    "arrange_largest_number", "build_tower", "classify_objects", "classify_objects_by_language",
    "fold_clothes", "hang_mugs", "imitate_sorting_sequence", "insert_tubes", "make_kong",
    "organize_table", "pack_objects_into_box", "push_T", "put_bottles_into_dustbin",
    "stack_blocks", "stack_bowls",
)
#: The 25 tasks admitted on 2026-10-02.
EXTENDED_TASKS = tuple(sorted(
    set(OFFICIAL_SIM_TASK_LIMITS) - set(FIRST_FIFTEEN) - set(FLUID_TASKS)))
#: Admitted tasks that resolve CPU physics by default (user decision 2026-10-04),
#: written out independently of ``robodojo_physics``.
CPU_PHYSICS_ADMITTED = (
    "fill_egg_holder", "make_toast", "match_and_pick_from_conveyor", "pick_from_conveyor_by_image",
    "press_by_number", "store_laptop_and_headphones", "swap_blocks", *FLUID_TASKS,
)


def _runtime_for(task: str) -> dict:
    from ora.environment.benchmarks.robodojo_asset_revision import asset_revision_for
    runtime = _runtime()
    runtime["required_asset_revision"] = asset_revision_for(task)
    runtime.update(task_variant=task, layout_file=f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json",
                   native_action_limit=OFFICIAL_SIM_TASK_LIMITS[task],
                   physics_device="cpu" if task in CPU_PHYSICS_ADMITTED else "cuda:0")
    return runtime


def test_admitted_tasks_are_every_official_task() -> None:
    """Since the 2026-10-04 pour admission every official task is admitted, the fluid two included."""
    assert len(OFFICIAL_SIM_TASK_LIMITS) == 42 and len(EXTENDED_TASKS) == 25
    assert robodojo.NATIVE_ACTION_LIMITS == OFFICIAL_SIM_TASK_LIMITS
    assert set(FLUID_TASKS) <= set(robodojo.NATIVE_ACTION_LIMITS)


@pytest.mark.parametrize("task", (*EXTENDED_TASKS, *FLUID_TASKS))
def test_extended_tasks_are_admitted_with_their_own_layout_and_upstream_limit(
    tmp_path: Path, task: str,
) -> None:
    from ora.environment.backends.robodojo.bridge import manifest_bindings
    from ora.environment.backends.robodojo.session import RobodojoSession

    limit = OFFICIAL_SIM_TASK_LIMITS[task]
    runtime = _runtime_for(task)
    robodojo.validate_runtime_extension(runtime)
    ceiling = limit * robodojo.PHYSICS_STEPS_PER_NATIVE_ACTION
    assert robodojo.physics_step_ceiling(task) == ceiling
    session = RobodojoSession(config={"task": task, "native_action_limit": limit}, output_root=tmp_path)
    assert (session.native_action_limit, session.max_sim_steps) == (limit, ceiling)
    bindings = manifest_bindings({"action_codec": "motion_action_v3"}, task)
    assert bindings["official_native_action_limit"] == limit
    for wrong in (limit + 10, 1050 if limit != 1050 else 1100):
        runtime["native_action_limit"] = wrong
        with pytest.raises(robodojo.RoboDojoBenchmarkError, match="native_action_limit"):
            robodojo.validate_runtime_extension(runtime)
    runtime["native_action_limit"] = limit
    runtime["layout_file"] = f"Eval_Layout/RoboDojo/arx_x5/0/{task}_random_0.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="selected task"):
        robodojo.validate_runtime_extension(runtime)


@pytest.mark.parametrize("task", [
    "pour_liquid_into_cup_random", "stack_bowls_random", "push_t", "Swap_Blocks", "not_a_task", "",
])
def test_unadmitted_task_names_are_refused_on_every_admission_path(
    tmp_path: Path, task: str,
) -> None:
    from ora.environment.backends.robodojo.bridge import RobodojoError, manifest_bindings
    from ora.environment.backends.robodojo.session import RobodojoSession

    runtime = _runtime()
    runtime.update(task_variant=task, layout_file=f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json",
                   native_action_limit=OFFICIAL_SIM_TASK_LIMITS.get(task, 1050))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="reviewed fixed-layout tasks"):
        robodojo.validate_runtime_extension(runtime)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="reviewed fixed-layout tasks"):
        robodojo.physics_step_ceiling(task)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="only the reviewed tasks"):
        robodojo.selected_layout(tmp_path, task=task, eval_seed=0, layout_id=0)
    with pytest.raises(RobodojoError, match="unsupported RoboDojo task"):
        manifest_bindings({"action_codec": "motion_action_v3"}, task)
    with pytest.raises(RobodojoError):
        RobodojoSession(config={"task": task}, output_root=tmp_path)


@pytest.mark.parametrize("task,other", [
    ("stack_blocks", "stack_blocks_by_language"), ("classify_objects", "classify_objects_by_language"),
])
def test_a_task_never_selects_the_layout_of_a_task_its_name_prefixes(
    tmp_path: Path, task: str, other: str,
) -> None:
    folder = tmp_path / "Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    for name in (f"{other}_0.json", f"{task}_random_0.json", f"{task}_3.json"):
        (folder / name).write_text("{}")
    assert robodojo.selected_layout(tmp_path, task=task, eval_seed=0, layout_id=0).name == f"{task}_3.json"
    runtime = _runtime_for(task)
    runtime["layout_file"] = f"Eval_Layout/RoboDojo/arx_x5/0/{other}_0.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="selected task"):
        robodojo.validate_runtime_extension(runtime)


UPSTREAM_ROOT = Path(os.environ.get("ROBODOJO_ROOT", str(Path.home() / "robodojo-runtime/source/RoboDojo")))
UPSTREAM_ASSETS = Path(os.environ.get("ROBODOJO_ASSET_ROOT", str(Path.home() / "robodojo-runtime/assets/Assets")))


def _pinned_upstream(relative: str) -> str:
    """One file of the pinned RoboDojo revision, whatever the checkout's HEAD."""

    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "show", f"{robodojo.REVISIONS['robodojo']}:{relative}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        pytest.skip(f"pinned RoboDojo revision is not available under {UPSTREAM_ROOT}")
    return result.stdout


@pytest.mark.skipif(not UPSTREAM_ROOT.is_dir(), reason="pinned RoboDojo source is not mounted")
def test_official_task_set_and_limits_are_the_pinned_upstream_ones() -> None:
    import ast

    inventory = ast.parse(_pinned_upstream("scripts/internal/task_inventory.py"))
    groups = [ast.literal_eval(node.value) for node in inventory.body
              if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DIMENSION_TASKS"]
    assert len(groups) == 1
    official = [task for tasks in groups[0].values() for task in tasks]
    assert len(official) == len(set(official)) == 42
    assert set(official) == set(OFFICIAL_SIM_TASK_LIMITS)
    for task in official:
        assignments = [
            node.value for node in ast.walk(ast.parse(_pinned_upstream(f"task/RoboDojo/tasks/{task}.py")))
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "step_lim" for target in node.targets)
        ]
        assert len(assignments) == 1 and isinstance(assignments[0], ast.Constant), task
        assert assignments[0].value == OFFICIAL_SIM_TASK_LIMITS[task], task
    # Why the fluid tasks resolve CPU physics: CUDA physics skips every fluid object.
    scene = ast.unparse(ast.parse(_pinned_upstream("env/scene_manager/scene_manager.py")))
    assert "if self.device.type.startswith('cuda'):\n            if obj_type in ['fluid']:" in scene


@pytest.mark.skipif(not (UPSTREAM_ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
                    reason="pinned RoboDojo Assets absent")
def test_only_the_fluid_task_layouts_need_a_fluid_object_and_they_resolve_cpu_physics() -> None:
    from ora.environment.benchmarks.robodojo_physics import physics_device_for

    for task in OFFICIAL_SIM_TASK_LIMITS:
        layout = json.loads(robodojo.selected_layout(
            UPSTREAM_ASSETS, task=task, eval_seed=0, layout_id=0,
        ).read_text())
        assert bool(layout.get("Fluid")) == (task in FLUID_TASKS), task
        if layout.get("Fluid"):
            assert physics_device_for(task) == "cpu", task


@pytest.mark.parametrize("mutation", [None, "schema", "lineage", "kernel", "legacy_record",
                                      "kernel_shape", "sources", "configuration",
                                      "layout", "checks", "hash", "escape", "missing_fingerprint",
                                      "empty_files", "file_hash", "installed_version", "config_hash",
                                      "empty_instruction", "environment_shape", "python_path",
                                      "python_hash", "environment_version"])
def test_robodojo_infrastructure_qualification_binds_source_without_inventing_an_oracle(
    tmp_path: Path, mutation: str | None, robodojo_activation,
) -> None:
    runtime = _runtime()
    value = _qualification(runtime)
    value["device_binding"] = deepcopy(robodojo_activation.device_binding)
    value["embodiment_activation"] = robodojo_activation(
        (tmp_path / runtime["infrastructure_qualification"]["path"]).parent,
    )
    assert "oracle_success" not in value and "accepted_seeds" not in value
    if mutation == "schema":
        value["schema"] = "seed-oracle.v1"
    elif mutation == "lineage":
        # The commit is no longer compared with the plan's, but it is still
        # required to be a recorded commit id.
        value["harness_commit"] = "measured-somewhere"
    elif mutation == "kernel":
        # The RoboDojo worker kernel moved between measurement and launch.
        value["worker_kernel_sha256"] = "a" * 64
    elif mutation == "legacy_record":
        # A record produced before the content binding existed fails closed.
        value.pop("worker_kernel_sha256")
    elif mutation == "kernel_shape":
        value["worker_kernel_sha256"] = "unbound"
    elif mutation == "sources":
        value["source_revisions"]["isaaclab"] = "d" * 40
    elif mutation == "configuration":
        value["configuration"]["native_actions_per_command"] = 19
    elif mutation == "layout":
        value["layout_sha256"] = "d" * 64
    elif mutation == "checks":
        value["checks"]["official_terminal"] = False
    elif mutation == "missing_fingerprint":
        value.pop("actual_source_fingerprint")
    elif mutation == "empty_files":
        value["actual_source_fingerprint"]["loaded_files"] = {}
    elif mutation == "file_hash":
        value["actual_source_fingerprint"]["loaded_files"]["env_cfg/arx_x5.yml"] = "unbound"
    elif mutation == "installed_version":
        value["actual_source_fingerprint"]["installed_packages"]["isaacsim"] = "6.0.1.0"
    elif mutation == "config_hash":
        value["actual_source_fingerprint"]["resolved_environment_config_sha256"] = "unbound"
    elif mutation == "empty_instruction":
        value["task_instruction"] = "  "
    elif mutation == "environment_shape":
        value["environment"].pop("versions")
    elif mutation == "python_path":
        value["environment"]["python_executable"] = "python"
    elif mutation == "python_hash":
        value["environment"]["python_binary_sha256"] = "unbound"
    elif mutation == "environment_version":
        value["environment"]["versions"]["torch"] = "2.11.0+cu128"
    path = tmp_path / runtime["infrastructure_qualification"]["path"]
    runtime["infrastructure_qualification"]["sha256"] = _write(path, value)
    if mutation == "hash":
        path.write_text(path.read_text() + " ")
    elif mutation == "escape":
        runtime["infrastructure_qualification"]["path"] = "../outside.json"
    if mutation is None:
        # The stored record plus how admission bound its kernel; the record
        # itself is not rewritten.
        assert load_infrastructure_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        ) == {**value, "kernel_admission": _identical_admission("c" * 40, "c" * 40)}
    else:
        with pytest.raises(robodojo.RoboDojoBenchmarkError):
            load_infrastructure_qualification(
                runtime, workspace_root=tmp_path, harness_commit="c" * 40,
            )


def test_infrastructure_record_binds_worker_kernel_content_not_the_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robodojo_activation,
) -> None:
    """A 36-minute live measurement survives unrelated commits, not kernel edits.

    The record keeps ``harness_commit`` as the lineage of where it was
    measured; admission compares the content of ``WORKER_KERNEL_PATHS`` at the
    RunPlan's own commit against the digest the record stored.
    """

    kernels = {
        "c" * 40: WORKER_KERNEL_DIGEST,        # where the probes actually ran
        "d" * 40: WORKER_KERNEL_DIGEST,        # a later, unrelated commit
        "e" * 40: "a" * 64,                    # a commit that moved the kernel
    }
    monkeypatch.setattr(
        qualification_module, "worker_kernel_sha256",
        lambda commit, **_kwargs: kernels[commit],
    )
    runtime = _runtime()
    value = _qualification(runtime)
    path = tmp_path / runtime["infrastructure_qualification"]["path"]
    value["device_binding"] = deepcopy(robodojo_activation.device_binding)
    value["embodiment_activation"] = robodojo_activation(path.parent)
    runtime["infrastructure_qualification"]["sha256"] = _write(path, value)

    assert value["harness_commit"] == "c" * 40
    for commit in ("c" * 40, "d" * 40):
        assert load_infrastructure_qualification(
            runtime, workspace_root=tmp_path, harness_commit=commit,
        ) == {**value, "kernel_admission": _identical_admission("c" * 40, commit)}
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="worker kernel changed"):
        load_infrastructure_qualification(
            runtime, workspace_root=tmp_path, harness_commit="e" * 40,
        )


def test_worker_kernel_digest_moves_only_with_the_kernel(tmp_path: Path) -> None:
    """The digest itself, over a real git history, not a stub."""

    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    for relative in WORKER_KERNEL_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {relative}\n", encoding="utf-8")
    (root / "docs/notes.md").write_text("first\n", encoding="utf-8")
    manifest = root / "ora/orchestrator/robodojo_kernel.py"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f"WORKER_KERNEL_PATHS = {WORKER_KERNEL_PATHS!r}\n", encoding="utf-8")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], text=True, capture_output=True,
            check=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "fixture")
    git("add", "-A")
    git("commit", "-qm", "measured")
    measured = git("rev-parse", "HEAD")
    (root / "docs/notes.md").write_text("second\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "an unrelated documentation commit")
    unrelated = git("rev-parse", "HEAD")
    (root / WORKER_KERNEL_PATHS[0]).write_text("# edited\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "a worker kernel edit")
    edited = git("rev-parse", "HEAD")

    digest = real_worker_kernel_sha256(measured, project_root=root)
    assert real_worker_kernel_sha256(measured, project_root=root) == digest
    assert real_worker_kernel_sha256(unrelated, project_root=root) == digest
    assert real_worker_kernel_sha256(edited, project_root=root) != digest
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="commit id"):
        real_worker_kernel_sha256("HEAD", project_root=root)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="cannot resolve"):
        real_worker_kernel_sha256("b" * 40, project_root=root)


def test_worker_kernel_path_set_covers_the_motion_gate_and_stays_tracked() -> None:
    """Every bound path is tracked source, and the motion set is a subset."""

    from ora.orchestrator.robodojo_qualification import ROBODOJO_MOTION_KERNEL_PATHS

    assert set(ROBODOJO_MOTION_KERNEL_PATHS) <= set(WORKER_KERNEL_PATHS)
    assert len(set(WORKER_KERNEL_PATHS)) == len(WORKER_KERNEL_PATHS)
    root = Path(__file__).resolve().parents[3]
    tracked = set(subprocess.run(
        ["git", "-C", str(root), "ls-files", *WORKER_KERNEL_PATHS],
        text=True, capture_output=True, check=True,
    ).stdout.split())
    assert tracked == set(WORKER_KERNEL_PATHS)


@pytest.mark.parametrize("task,ceiling", [
    ("arrange_largest_number", 10500), ("hang_mugs", 8000),
    ("pack_objects_into_box", 13000),
])
def test_robodojo_admits_exactly_its_own_physics_step_ceiling(
    tmp_path: Path, task: str, ceiling: int,
) -> None:
    """``physics_step_ceiling`` is the bound the session itself enforces."""

    from ora.environment.backends.robodojo.session import (
        RobodojoError, RobodojoSession,
    )

    assert robodojo.physics_step_ceiling(task) == ceiling
    assert ceiling == robodojo.NATIVE_ACTION_LIMITS[task] * (
        robodojo.PHYSICS_STEPS_PER_NATIVE_ACTION
    )
    session = RobodojoSession(
        config={"task": task, "max_sim_steps": ceiling}, output_root=tmp_path,
    )
    assert session.max_sim_steps == ceiling
    # The default is the ceiling, so an omitted budget is admissible too.
    assert RobodojoSession(
        config={"task": task}, output_root=tmp_path,
    ).max_sim_steps == ceiling
    for refused in (ceiling + 10, 10000 if ceiling < 10000 else ceiling + 1000):
        with pytest.raises(RobodojoError, match="max_sim_steps"):
            RobodojoSession(
                config={"task": task, "max_sim_steps": refused},
                output_root=tmp_path,
            )
    with pytest.raises(robodojo.RoboDojoBenchmarkError):
        robodojo.physics_step_ceiling("pour_liquid_into_cup_random")


@pytest.mark.parametrize("mutation", [
    "missing", "extra_field", "card_hash", "descriptor_path", "activation_hash", "review_hash",
    "evidence_hash", "activation_card", "review_join", "runtime_join", "evidence_facts",
    "reset_hash", "reset_join", "source_hash", "report_hash", "worker_count", "symlink_escape",
    "log_hash", "residency_gpu", "residency_pid", "qualified_gpu_drift",
])
def test_robodojo_activation_rechecks_tracked_card_and_immutable_evidence(
    tmp_path: Path, mutation: str, robodojo_activation,
) -> None:
    runtime = _runtime()
    path = tmp_path / runtime["infrastructure_qualification"]["path"]
    value = _qualification(runtime)
    binding = value["embodiment_activation"] = robodojo_activation(path.parent)
    value["device_binding"] = deepcopy(robodojo_activation.device_binding)

    def rewrite(reference, mutate):
        target = path.parent / reference["path"]
        data = json.loads(target.read_text())
        mutate(data)
        target.chmod(0o644)
        reference["sha256"] = _write(target, data)

    if mutation == "missing":
        value.pop("embodiment_activation")
    elif mutation == "extra_field":
        binding["trust_me"] = True
    elif mutation == "card_hash":
        binding["compiled_card"]["sha256"] = "f" * 64
    elif mutation == "descriptor_path":
        binding["descriptor"]["path"] = "../descriptor-v1.json"
    elif mutation in {"activation_hash", "review_hash", "evidence_hash"}:
        name = {"activation_hash": "activation_record", "review_hash": "review_record",
                "evidence_hash": "runtime_evidence"}[mutation]
        binding[name]["sha256"] = "f" * 64
    elif mutation in {"activation_card", "review_join", "runtime_join"}:
        def change_activation(data):
            if mutation == "activation_card":
                data["card_sha256"] = "f" * 64
            elif mutation == "review_join":
                data["review"]["record_sha256"] = "f" * 64
            else:
                data["runtime_verification"]["evidence_sha256"] = "f" * 64
        rewrite(binding["activation_record"], change_activation)
    elif mutation == "symlink_escape":
        target = path.parent / binding["review_record"]["path"]
        outside = tmp_path / "outside-review.json"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
    elif mutation == "log_hash":
        (path.parent / "reset-0/environment/kit-startup-readback.log").write_text("corrupt startup evidence")
    elif mutation == "qualified_gpu_drift":
        value["device_binding"].update(
            selected_gpu_uuid="GPU-abcdef01-9abc-def0-1234-56789abcdef0",
            physics_cuda_uuid="abcdef01-9abc-def0-1234-56789abcdef0", renderer_uuid_prefix="abcdef01",
        )
    else:
        def change_evidence(data):
            if mutation == "evidence_facts":
                data["projected_facts_sha256"] = "f" * 64
            elif mutation == "worker_count":
                data["fresh_worker_evidence"].pop()
            else:
                worker = data["fresh_worker_evidence"][0]
                if mutation == "reset_join":
                    rewrite(worker["reset"], lambda item: item.update(runtime_embodiment={}))
                elif mutation in {"residency_gpu", "residency_pid"}:
                    def change_residency(item):
                        if mutation == "residency_gpu":
                            item["worker_device_residency"]["gpu_uuids"].append("GPU-unqualified")
                        else:
                            item["worker_device_residency"]["worker_pid"] = True
                    rewrite(worker["probe_report"], change_residency)
                else:
                    key = {"reset_hash": "reset", "source_hash": "source_manifest",
                           "report_hash": "probe_report"}[mutation]
                    worker[key]["sha256"] = "f" * 64
        rewrite(binding["runtime_evidence"], change_evidence)
        rewrite(binding["activation_record"], lambda item: item["runtime_verification"].update(
            evidence_sha256=binding["runtime_evidence"]["sha256"],
        ))
    runtime["infrastructure_qualification"]["sha256"] = _write(path, value)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="embodiment activation"):
        load_infrastructure_qualification(runtime, workspace_root=tmp_path, harness_commit="c" * 40)


@pytest.mark.parametrize("field,value", [
    (None, None), ("unknown", True), ("schema", "unqualified"),
    ("renderer_vulkan_index", 0), ("renderer_cuda_index", False),
    ("selected_physical_index", "2"), ("physics_cuda_logical_index", 2),
    ("selected_gpu_uuid", "GPU-fixture"), ("physics_cuda_uuid", "ffffffff-ffff-ffff-ffff-ffffffffffff"),
    ("renderer_uuid_prefix", "1234"), ("renderer_uuid_prefix", "ffffffff"),
    ("renderer_pci_bus_hex", "41"), ("renderer_pci_bus_hex", "no-bus"),
    ("selected_pci_bus_id", "bus31"), ("renderer_identity_resolution", "guessed-index"),
])
def test_robodojo_qualification_rejects_unbound_or_mismatched_gpu_identity(
    tmp_path: Path, field: str | None, value: object, robodojo_activation,
) -> None:
    runtime = _runtime()
    path = tmp_path / runtime["infrastructure_qualification"]["path"]
    qualification = _qualification(runtime)
    if field is not None:
        qualification["device_binding"] = {**robodojo_activation.device_binding, field: value}
    qualification["embodiment_activation"] = robodojo_activation(path.parent)
    runtime["infrastructure_qualification"]["sha256"] = _write(path, qualification)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="device binding"):
        load_infrastructure_qualification(runtime, workspace_root=tmp_path, harness_commit="c" * 40)


def test_robodojo_factory_and_worker_preserve_the_qualified_isolated_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    python = tmp_path / "isolated-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    root, assets, native = tmp_path / "source", tmp_path / "Assets", tmp_path / "run/native"
    monkeypatch.setenv("ROBODOJO_ROOT", str(root))
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(assets))
    monkeypatch.setenv("ROBODOJO_PYTHON", str(python))
    binding = _qualification(runtime)
    session = factory.build_environment_session(
        runtime_config=runtime, output_root=native, task="arrange_largest_number", seed=0,
        max_sim_steps=2000, action_codec="runtime_action_v3", qualified_runtime_binding=binding,
    )
    proxy = session.delegate
    assert proxy.process is None  # Constructing a plan never boots Isaac.
    assert proxy.python_executable == python and proxy.python_executable != python.resolve()
    assert proxy.close_timeout_s == 30
    assert proxy.worker_environment == {"CUDA_VISIBLE_DEVICES": "2"}
    assert proxy.worker_config["output_root"] == str(native.resolve())
    assert proxy.worker_config["robodojo_root"] == str(root.resolve())
    assert proxy.worker_config["asset_root"] == str(assets.resolve())
    assert proxy.worker_config["qualification_binding"] == binding
    assert proxy.worker_config["action_codec"] == "runtime_action_v3"
    assert session.describe()["embodiment_id"] == "arx_x5"
    # Layered fail-closed: a codec the backend does not implement is rejected
    # by its own action-schema set; the implemented sparse v3 codec is
    # admitted only by the source-bound qualification table.
    with pytest.raises(EnvironmentDefinitionError, match="unsupported action"):
        resolve_runtime_descriptor(runtime, action_codec="target_state_v4")
    from ora.environment import definitions as environment_definitions

    # The live row exists only because the device-keyed gate record passed;
    # removing it must close the codec again without touching validation code.
    assert "robodojo" in environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS["motion_action_v3"]
    monkeypatch.setitem(
        environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
        "motion_action_v3", ("robotwin", "libero_pro"),
    )
    with pytest.raises(EnvironmentDefinitionError, match="no source-bound motion qualification for robodojo"):
        resolve_runtime_descriptor(runtime, action_codec="motion_action_v3")
    monkeypatch.setitem(
        environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
        "motion_action_v3", ("robotwin", "libero_pro", "robodojo"),
    )
    sparse = resolve_runtime_descriptor(runtime, action_codec="motion_action_v3").as_dict()
    assert sparse["schema"] == "ora.environment-runtime-descriptor.v2"
    assert sparse["action_modes"] == ["move_pose", "move_free_space", "set_gripper", "recover_motion", "wait"]
    assert sparse["controlled_end_effectors"] == {
        "left": {"controlled_point": "link6", "planner_route": "left"},
        "right": {"controlled_point": "link6", "planner_route": "right"},
    }
    assert sparse["control_groups"] == [
        {"group_id": "arx_x5_dual_arm", "members": ["left", "right"], "synchronous_pose_execution": True},
    ]
    assert sparse["camera_roles"] == {
        "head": "fixed_third_person_workspace", "left_wrist": "left_wrist", "right_wrist": "right_wrist",
    }
    fake_module = ModuleType("ora.environment.backends.robodojo.session")
    captured = {}

    def fake_session(**kwargs):
        captured.update(kwargs)
        return captured

    fake_module.RobodojoSession = fake_session
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    config = {**proxy.worker_config, "shared_root": str(native.parent)}
    assert worker._build_session(config) == captured
    assert captured["output_root"] == native
    assert captured["config"] == config
    with pytest.raises(worker.WorkerProtocolError, match="escapes"):
        worker._build_session({**config, "output_root": str(tmp_path / "elsewhere")})


@pytest.mark.parametrize("mutation", [None, "missing", "duplicate", "failed", "wrong_schema"])
def test_robodojo_coordinator_requires_exactly_one_passing_infrastructure_binding(
    tmp_path: Path, mutation: str | None,
) -> None:
    runtime = _runtime()
    value = _qualification(runtime)
    checks = [{"check": "robodojo_infrastructure_qualification", "passed": True, "detail": value}]
    if mutation == "missing":
        checks.clear()
    elif mutation == "duplicate":
        checks.append(deepcopy(checks[0]))
    elif mutation == "failed":
        checks[0]["passed"] = False
    elif mutation == "wrong_schema":
        value["schema"] = "unqualified"
    _write(tmp_path / "preflight.json", {"checks": checks})
    if mutation is None:
        assert coordinator._qualified_runtime_binding(tmp_path, runtime) == value
    else:
        with pytest.raises(coordinator.SupervisorError, match="infrastructure binding"):
            coordinator._qualified_runtime_binding(tmp_path, runtime)


@pytest.mark.parametrize("mutation", [None, "layout", "source", "package", "environment", "gpu_uuid", "driver",
                                      "low_memory", "dense_slot"])
def test_robodojo_preflight_rechecks_layout_sources_and_worker_readback_without_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None, robodojo_activation,
) -> None:
    runtime = _runtime()
    if mutation == "dense_slot":
        # A dense slot declares its own floor; the GPU check must use it, not 20000.
        runtime["resource_binding"] = {"profile_id": "robodojo-gpu7-d4", "minimum_free_memory_mib": 10240}
    root, assets = tmp_path / "RoboDojo", tmp_path / "Assets"
    root.mkdir()
    layout = assets / runtime["layout_file"]
    runtime["layout_sha256"] = _write(layout, {"fixture": True})
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    for name, value in {"ROBODOJO_ROOT": root, "ROBODOJO_ASSET_ROOT": assets,
                        "ROBODOJO_PYTHON": python, "ORA_EXECUTION_HOST": "test-host"}.items():
        monkeypatch.setenv(name, str(value))
    observed = {"python_executable": str(python), "python_binary_sha256": "e" * 64,
                "versions": {"isaacsim": "5.1.0.0", "torch": "2.7.0+cu128",
                             "numpy": "1.26.0", "warp-lang": "1.9.1"}}
    qualification = _qualification(runtime)
    qualification["device_binding"] = deepcopy(robodojo_activation.device_binding)
    qualification["embodiment_activation"] = robodojo_activation(
        (tmp_path / runtime["infrastructure_qualification"]["path"]).parent,
    )
    qualification["environment"] = deepcopy(observed)
    if mutation == "environment":
        qualification["environment"]["python_executable"] = "/unqualified/python"
    runtime["infrastructure_qualification"]["sha256"] = _write(
        tmp_path / runtime["infrastructure_qualification"]["path"], qualification,
    )
    roots = {root: "robodojo", root / "third_party/IsaacLab": "isaaclab",
             root / "third_party/curobo": "curobo", root / "XPolicyLab": "xpolicylab", assets: "assets"}

    def fake_git(path, *args):
        revision = robodojo.REVISIONS[roots[path]]
        if mutation == "source" and path == root:
            revision = "f" * 40
        return subprocess.CompletedProcess(args, 0, revision if args[0] == "rev-parse" else "", "")

    def runner(command, **kwargs):
        if command[0] == "nvidia-smi":
            gpu_uuid = "GPU-unqualified" if mutation == "gpu_uuid" else robodojo_activation.device_binding["selected_gpu_uuid"]
            driver = "595.71.05" if mutation == "driver" else "580.65.06"
            free = 15000 if mutation in {"low_memory", "dense_slot"} else 40000
            return subprocess.CompletedProcess(command, 0, f"2, {gpu_uuid}, {free}, {driver}\n", "")
        assert command[0] == str(python)
        if command[1].endswith("graphics_compat.py"):
            return subprocess.CompletedProcess(command, 1, "", "driver compatibility is unavailable")
        assert "importlib.metadata" in command[2]
        assert kwargs["timeout"] == qualification_module.GRAPHICS_READBACK_TIMEOUT_S
        versions = deepcopy(observed)
        if mutation == "package":
            versions["versions"]["isaacsim"] = "6.0.1.0"
        return subprocess.CompletedProcess(command, 0, json.dumps(versions), "")

    monkeypatch.setattr(preflight, "_git", fake_git)
    monkeypatch.setattr(preflight, "resolve_workspace_root", lambda _root: tmp_path)
    if mutation == "layout":
        layout.write_text("changed layout")
    checks = preflight._robodojo_runtime({
        "runtime": runtime, "task": "arrange_largest_number", "seed": 0,
        "task_instruction": runtime["expected_task_language"], "source": {"commit": "c" * 40},
    }, runner)
    failures = {check["check"] for check in checks if check["passed"] is not True}
    expected = {None: set(), "layout": {"robodojo_layout"}, "source": {"robodojo_robodojo_source"},
                "package": {"robodojo_python_stack", "robodojo_infrastructure_qualification", "robodojo_gpu", "robodojo_driver"},
                "environment": {"robodojo_infrastructure_qualification", "robodojo_gpu", "robodojo_driver"},
                "gpu_uuid": {"robodojo_gpu"}, "driver": {"robodojo_driver"},
                "low_memory": {"robodojo_gpu"}, "dense_slot": set()}
    assert failures == expected[mutation]
    if mutation in {"package", "environment"}:
        driver_check = next(c for c in checks if c['check'] == 'robodojo_driver')
        assert 'blocked by infrastructure qualification; no drift conclusion' in driver_check['detail']


@pytest.mark.parametrize("mutation", [None, "task", "seed", "language", "codec"])
def test_robodojo_runspec_binds_episode_identity_and_the_admitted_codec(
    tmp_path: Path, mutation: str | None,
) -> None:
    # This isolated source tree declares no reviewed public task contracts.
    from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH, REGISTRY_SCHEMA
    registry = tmp_path / REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema": REGISTRY_SCHEMA, "tasks": {}}))
    for filename in ("agents.md", "prompt.md", "manual.md"):
        (tmp_path / filename).write_text("## Overview\n\nFixture instructions.\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    value = {
        "schema": "ora.run-spec.v1", "experiment": "runtime-v1",
        "config": "fixture", "milestone": "M0", "output_workspace": "outputs/runtime-v1",
        "task": "arrange_largest_number", "task_instruction": "Pick the item.", "seed": 0,
        "evidence_tier": "diagnostic", "runtime": _runtime(),
        "policy": {"kind": "scripted", "model": "scripted", "provider": "trusted",
                   "reasoning_effort": "none", "action_codec": "runtime_action_v3", "actions": [{"x": 1}],
                   "tool_surface": {"profile": "robot_runtime_v1", "component_overrides": {}}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 1, "max_sim_steps": 20, "wall_time_s": 10},
        "policy_context": {"codex_project": ["agents.md"], "episode_prefix": ["prompt.md"],
                           "tool_manual": ["manual.md"]}, "qualification": {"required": False},
    }
    if mutation == "task":
        value["task"] = "swap_blocks"
    elif mutation == "seed":
        value["seed"] = 1
    elif mutation == "language":
        value["task_instruction"] = "A different task."
    elif mutation == "codec":
        value["policy"]["action_codec"] = "target_state_v4"
        value["policy"]["tool_surface"]["profile"] = "robot_runtime_v2"
    if mutation is None:
        resolved = spec.resolve_run_spec(value, root=tmp_path)
        assert resolved["runtime"]["kind"] == "robodojo"
        assert resolved["runtime"]["close_timeout_s"] == 30
    else:
        with pytest.raises(spec.RunSpecError):
            spec.resolve_run_spec(value, root=tmp_path)


def test_robodojo_unified_instance_bindings_are_admitted_and_bounded() -> None:
    """The unified instance adds exactly two runtime fields, both closed.

    ``workspace_profile`` is the source-owned hard action envelope the RunSpec
    resolver expands in place, and ``environment_prompt_profile`` records the
    prompt profile the public context selected. Both are admitted, but only
    the one reviewed RoboDojo envelope is.
    """

    from ora.environment.workspace_profiles import (
        ROBODOJO_ARX_X5_TABLETOP_V1, resolve_workspace_profile,
    )

    runtime = _runtime()
    profile = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)
    runtime["workspace_profile"] = profile
    runtime["workspace_bounds_m"] = profile["bounds_m"]
    runtime["environment_prompt_profile"] = {
        "schema": "ora.environment-prompt-profile.v1",
        "profile_id": "robodojo_arx_x5_unified_bundle_v1",
        "profile_path": (
            "ora/policy/environments/robodojo/context-profile-v1.json"
        ),
        "profile_sha256": "d" * 64, "metadata_sha256": "e" * 64,
    }
    robodojo.validate_runtime_extension(runtime)

    runtime["workspace_profile"] = resolve_workspace_profile(
        "libero_pro_panda_tabletop_v1",
    )
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="workspace profile"):
        robodojo.validate_runtime_extension(runtime)


def test_robodojo_worker_environment_binds_the_leased_slot_or_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slot lease must guard the card the Isaac worker actually renders on.

    RoboDojo selects its device twice -- ``CUDA_VISIBLE_DEVICES`` masks CUDA
    while ``renderer_gpu`` is an unmasked Vulkan ordinal -- so the resolved
    resource binding and the runtime device must agree before the worker
    starts.
    """

    python = tmp_path / "isolated-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    root, assets = tmp_path / "source", tmp_path / "Assets"
    monkeypatch.setenv("ROBODOJO_ROOT", str(root))
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(assets))
    monkeypatch.setenv("ROBODOJO_PYTHON", str(python))
    root.mkdir()
    assets.mkdir()

    from ora.orchestrator.resources import resolve_execution_resource

    binding = resolve_execution_resource(
        "robodojo-gpu3", runtime_kind="robodojo", required_host="test-host",
    )
    assert binding["resource_class"] == "robodojo_sim"
    assert binding["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "3"}
    assert binding["minimum_free_memory_mib"] >= 20480

    runtime = _runtime()
    runtime.update({
        "cuda_visible_devices": "3", "renderer_gpu": 3,
        "resource_binding": binding,
    })
    session = factory.build_robodojo_worker_session(
        runtime=runtime, output_root=tmp_path / "run", task="hang_mugs",
        seed=0, max_sim_steps=2000, action_codec="motion_action_v3",
    )
    assert session.worker_environment == {"CUDA_VISIBLE_DEVICES": "3"}

    # A slot that names another card than the runtime is a lease that guards
    # the wrong device, so the worker must never start.
    mismatched = dict(runtime)
    mismatched["resource_binding"] = resolve_execution_resource(
        "robodojo-gpu3", runtime_kind="robodojo", required_host="test-host",
    )
    mismatched["cuda_visible_devices"] = "5"
    mismatched["renderer_gpu"] = 5
    with pytest.raises(factory.EnvironmentFactoryError, match="different devices"):
        factory.build_robodojo_worker_session(
            runtime=mismatched, output_root=tmp_path / "run2", task="hang_mugs",
            seed=0, max_sim_steps=2000, action_codec="motion_action_v3",
        )


def test_robodojo_sim_slot_admits_a_neighbour_only_by_device_capacity(
    monkeypatch,
) -> None:
    """The card is shared by capacity, never by class, and never blind."""

    from ora.orchestrator import cli
    from ora.orchestrator.resources import (
        ExecutionResourceError, resolve_execution_resource, resource_lease_key,
    )

    robodojo_binding = resolve_execution_resource(
        "robodojo-gpu3", runtime_kind="robodojo", required_host="test-host",
    )
    libero_binding = resolve_execution_resource(
        "libero-gpu3", runtime_kind="libero_pro", required_host="test-host",
    )
    assert resource_lease_key(robodojo_binding) == "robodojo_sim--gpu3"
    assert resource_lease_key(libero_binding) == "libero_pro_sim--gpu3"
    # The RoboDojo slot declares an order-of-magnitude larger floor than the
    # LIBERO slots it may now share a card with, so the two together price at
    # 22528 MiB of declared demand on one device.
    assert robodojo_binding["minimum_free_memory_mib"] == 20480
    assert libero_binding["minimum_free_memory_mib"] == 2048
    held = [{
        "schema": "ora.resource-lease.v1",
        "run_id": "run-libero",
        "resource_class": libero_binding["resource_class"],
        "slot_id": libero_binding["slot_id"],
        "profile_id": libero_binding["profile_id"],
        "runtime_kind": "libero_pro",
        "host_id": "test-host",
        "minimum_free_memory_mib": libero_binding["minimum_free_memory_mib"],
        "worker_environment": dict(libero_binding["worker_environment"]),
    }]

    def _inventory(free_mib: int, total_mib: int):
        return lambda owner_pids=None: {"3": {"free_mib": free_mib, "total_mib": total_mib}}

    # A card that cannot hold both declared floors refuses the claim, and so
    # does a card whose live free memory is below the RoboDojo floor.
    monkeypatch.setattr(cli, "_gpu_memory_inventory", _inventory(21000, 21504))
    refusal = cli._resource_refusal(robodojo_binding, held)
    assert refusal is not None and "rule=device_capacity" in refusal
    monkeypatch.setattr(cli, "_gpu_memory_inventory", _inventory(20479, 46068))
    refusal = cli._resource_refusal(robodojo_binding, held)
    assert refusal is not None and "rule=device_capacity" in refusal

    # Without a live probe the cross-class claim fails closed: the declared
    # floors describe only our own leases and cannot see a foreign process.
    def _refuse(owner_pids=None) -> dict:
        raise ExecutionResourceError("nvidia-smi is unavailable")

    monkeypatch.setattr(cli, "_gpu_memory_inventory", _refuse)
    refusal = cli._resource_refusal(robodojo_binding, held)
    assert refusal is not None and "rule=device_probe_unavailable" in refusal

    # A card with room admits the neighbour: capacity, not class, decides.
    monkeypatch.setattr(cli, "_gpu_memory_inventory", _inventory(40000, 46068))
    assert cli._resource_refusal(robodojo_binding, held) is None

    # The RoboDojo slot itself stays exclusive, and a lease on another card
    # is not a neighbour at all -- no probe is even consulted.
    same_slot = [dict(held[0], run_id="run-robodojo",
                      resource_class=robodojo_binding["resource_class"],
                      slot_id=robodojo_binding["slot_id"])]
    monkeypatch.setattr(cli, "_gpu_memory_inventory", _refuse)
    refusal = cli._resource_refusal(robodojo_binding, same_slot)
    assert refusal is not None and "rule=slot_lease" in refusal
    elsewhere = [{
        "schema": "ora.resource-lease.v1",
        "resource_class": "libero_pro_sim", "slot_id": "gpu4",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "4"},
    }]
    assert cli._resource_refusal(robodojo_binding, elsewhere) is None


def test_robodojo_runtime_binds_the_launchers_registered_host(monkeypatch) -> None:
    """The host is a launch binding: the marker's registered host, refused otherwise."""

    from ora.orchestrator import robodojo_runtime_config as config
    from ora.orchestrator.defaults import _check_runtime_contract
    from ora.orchestrator.resources import resolve_execution_resource

    monkeypatch.delenv("ORA_EXECUTION_HOST", raising=False)
    with pytest.raises(ValueError, match="is not set"):
        config.execution_host()
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-cluster")
    assert config.execution_host() == "test-cluster"
    binding = resolve_execution_resource("robodojo-g3s1", runtime_kind="robodojo", required_host="test-cluster")
    assert binding["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "3"}
    monkeypatch.setenv("ORA_EXECUTION_HOST", "no-such-host")
    with pytest.raises(ValueError, match="unavailable: no-such-host"):
        config.execution_host()

    runtime = _runtime()
    runtime["required_host"] = "test-cluster"
    robodojo.validate_runtime_extension(runtime)
    runtime["required_host"] = "../test-arm"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="required_host"):
        robodojo.validate_runtime_extension(runtime)
    # A RunSpec may bind any registered host; a RunSpec bound to another registered
    # host does not conflict with it.
    _check_runtime_contract({"required_host": "test-cluster"}, {"required_host": "test-host"})
