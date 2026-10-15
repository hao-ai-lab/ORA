"""The maintained RoboTwin and LIBERO-Pro builder: fresh RunSpecs on the default baseline.

New RoboTwin and LIBERO-Pro experiments compose here
(ora/orchestrator/benchmark_spec.py), as RoboDojo experiments do in
robodojo_spec. Offline: no simulator, no Provider and no resident service; the
readiness report and seed manifests are fixtures. These tests also carry the
pitfalls the 2026-10-04 dev-line smoke met by hand: a RoboTwin spec without
``runtime.task`` silently withholds the reviewed contract, an unbound service
prerequisite fails preflight, and the default LIBERO surface takes no Servo
keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_ID, load_native_baseline
from ora.environment.benchmarks.libero_pro import CORE_SUITES, official_core_evaluation
from ora.environment.benchmarks.libero_pro_language import (
    base_bddl_language, base_task_language, core_task_language,
)
from ora.environment.benchmarks.robotwin import TASKS as ROBOTWIN_TASKS
from ora.environment.benchmarks.robotwin_instructions import (
    RoboTwinInstructionError, public_instructions,
)
from ora.orchestrator import benchmark_spec, paths
from ora.orchestrator import robotwin_evaluation as evaluation
from ora.orchestrator import spec as spec_module
from ora.orchestrator.model_visible import model_visible_items
from ora.orchestrator.resources import ExecutionResourceError
from ora.orchestrator.run_spec_inputs import SERVICE_VARIABLES
from ora.orchestrator.spec import RunSpecError, resolve_run_spec

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())
CATALOG = load_native_baseline(ROOT)
EXPERIMENT = "benchmark-spec-fixture"
REVISION = CATALOG["instances"]["robotwin"]["runtime"]["required_revision"]
#: The tool-client endpoints before any test here ran; compose binds them into os.environ.
ENDPOINTS_AT_IMPORT = {variable: os.environ.get(variable) for _, variable in SERVICE_VARIABLES}
#: Runtime keys a Servo-bearing LIBERO surface locks; the default surface has no Servo.
SERVO_KEYS = {"servo_profile", "servo_worker_abi", "servo_qualification", "episode_physics_budget"}


def _provider_owned():
    """Policy fields a Provider profile owns; a builder spec and a fresh default spec differ only here."""
    from ora.orchestrator.run_spec_inputs import PROVIDER_FIELDS
    return {*PROVIDER_FIELDS, "provider", "provider_profile", "provider_session_affinity",
            "provider_qualification_schema", "provider_qualification_version"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A private workspace: compose writes and the resolver reads outputs/<experiment> there."""
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(spec_module, "resolve_workspace_root", lambda root: tmp_path)

    def experiment_root(experiment):
        path = tmp_path / "outputs" / experiment
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(benchmark_spec, "experiment_root", experiment_root)
    # compose binds the tool-client endpoints into os.environ itself, which
    # monkeypatch does not record for an absent variable: restore them here so
    # the fixture endpoints never reach a later test module.
    saved = {variable: os.environ.pop(variable, None) for _, variable in SERVICE_VARIABLES}
    yield tmp_path
    for variable, value in saved.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


def _official_manifest(directory: Path, task="adjust_bottle", task_config="demo_randomized", seeds=(100002,),
                       revision=REVISION, **changes) -> Path:
    """A sealed manifest in the shape robotwin_evaluation qualify writes."""
    records = {str(seed): {"reset_predicate_false": True, "oracle_success": True,
                           "post_success_physics_hold": True, "terminal_stability_steps": 250} for seed in seeds}
    body = {"schema": evaluation.SEED_MANIFEST_SCHEMA, "task": task, "task_config": task_config,
            "robotwin_revision": revision, "source_commit": "a" * 40, "plan_sha256": "b" * 64,
            "accepted_seeds": list(seeds), "complete": True, "seeds": records}
    body.update(changes)
    body["manifest_sha256"] = evaluation.digest(body)
    path = directory / f"{task}-{task_config}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _legacy_manifest(directory: Path, task="adjust_bottle", seed=2, **changes) -> Path:
    """The older per-task shape (seed-qualification-reuse.v1): no scene configuration or revision."""
    body = {"schema": "ora.seed-qualification-reuse.v1", "task": task,
            "accepted_seeds": [seed], "evaluated_seeds": [seed], "oracle_success": True,
            "reset_predicate_false": True, "terminal_stability_steps": 250, "policy_visible": False}
    body.update(changes)
    path = directory / f"{task}-s{seed}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _readiness_report(directory: Path, monkeypatch, *, age_s: float = 0.0) -> Path:
    from ora.agent.model_services import operator as services

    monkeypatch.setenv("B2_SAM_PORT", "18124")
    monkeypatch.setenv("B2_GRASP_PORT", "18135")
    health = {key: "fixture" for key in services.FIELDS}
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time() - age_s, "passed": True,
              "adapter_sha256": services.grounding_grasp_adapter_sha256(),
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    return path


def _robotwin(tmp_path, task="adjust_bottle", seed=100002, task_config="demo_randomized", manifest=None, **kwargs):
    manifest = manifest or _official_manifest(tmp_path / "manifests", task, task_config, (seed,))
    kwargs.setdefault("resource_profile", "robotwin-gpu4")
    spec = benchmark_spec.build_robotwin_spec(task, seed=seed, task_config=task_config, experiment=EXPERIMENT,
                                              seed_manifest=manifest, **kwargs)
    return spec, manifest


def _libero(suite="libero_10_swap", task_id=6, seed=0, **kwargs):
    kwargs.setdefault("resource_profile", "libero-gpu4b")
    return benchmark_spec.build_libero_pro_spec(suite, task_id, seed=seed, experiment=EXPERIMENT, **kwargs)


def _fresh(backend, spec):
    """A fresh default spec with the same cell: only the per-run choices, no surface."""
    from test_native_baseline import native_spec

    fresh = native_spec(backend)
    fresh.update(task=spec["task"], task_instruction=spec["task_instruction"], seed=spec["seed"])
    fresh["runtime"] = dict(spec["runtime"])
    return fresh


# --- the composed RunSpec -------------------------------------------------------------------------

@pytest.mark.parametrize("task_config", ["demo_randomized", "demo_clean"])
def test_a_robotwin_cell_resolves_the_default_instance_and_delivers_its_contract(workspace, monkeypatch,
                                                                                  task_config):
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec, manifest = _robotwin(workspace, task_config=task_config, service_report=report,
                               cohort_id="benchmark-spec-fixture-g001")
    for owned_by_the_baseline in ("tool_surface", "tool_allowlist", "semantic_manifest", "baseline",
                                  "context_profile", "context_composition", "action_codec"):
        assert owned_by_the_baseline not in spec["policy"]
    assert "policy_context" not in spec and "session" not in spec
    tag = "random" if task_config == "demo_randomized" else "clean"
    assert spec["config"] == f"rt-{tag}-adjust_bottle-s100002"
    official = task_config == "demo_randomized"  # seed 100002 is a standard randomized episode
    assert spec["runtime"] == {"kind": "robotwin", "required_host": "test-host", "resource_profile": "robotwin-gpu4",
                               "task": "adjust_bottle", "task_config": task_config,
                               "task_instruction_source": "official_episode_unseen_v1" if official else "task_level"}
    if not official:
        assert spec["task_instruction"] == "Pick up the bottle on the table and hold it head up."
    benchmark_spec.compose(spec, seed_manifest=manifest)
    plan = resolve_run_spec(spec, root=ROOT)
    instance = CATALOG["instances"]["robotwin"]
    baseline = plan["policy"]["baseline"]
    assert (baseline["id"], baseline["instance"], baseline["selection"]) == (
        BASELINE_ID, instance["id"], "omitted_tool_surface")
    assert plan["policy"]["tool_surface"]["profile"] == instance["tool_surface"] == (
        ACTIVATION["instances"]["robotwin"]["tool_surface"]) == "robotwin_toolchain_b2_closed_noservo_v2"
    assert baseline["prompt_bundle"]["path"] == instance["prompt_bundle"]
    contract = plan["policy"]["task_contract"]
    assert contract["delivered"] is True and contract["task"] == "adjust_bottle"
    assert contract["registry"] == ACTIVATION["task_contracts"]["robotwin"]["registry"]
    assert (plan["runtime"]["task"], plan["runtime"]["task_config"]) == ("adjust_bottle", task_config)
    assert plan["runtime"]["required_revision"] == REVISION
    assert plan["runtime"]["resource_binding"]["profile_id"] == "robotwin-gpu4"
    assert plan["qualification"]["required"] is True
    assert plan["qualification"]["manifest"]["path"] == spec["qualification"]["manifest"]
    assert plan["qualification"]["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    prerequisite = plan["prerequisite"]
    assert prerequisite == spec["prerequisite"]
    assert prerequisite["schema"] == "ora.model-service-set-binding.v2"
    assert prerequisite["capability_scope"] == CATALOG["required_service_capabilities"]
    assert {child["tool_surface"] for child in prerequisite["services"].values()} == {instance["tool_surface"]}
    assert plan["policy"]["provider_profile"] == ACTIVATION["default_experiment_provider"]["profile_id"]
    assert plan["budgets"] == {"max_robot_decisions": 80, "max_model_calls": 200, "max_sim_steps": 10000,
                               "wall_time_s": 21600}
    assert plan["launch_admission"]["cell_id"] == spec["config"]
    assert plan["launch_admission"]["retry_allowed"] is False
    # The model sees what a fresh default RoboTwin spec of this cell shows.
    ours = model_visible_items(plan, root=ROOT)
    theirs = model_visible_items(resolve_run_spec(_fresh("robotwin", spec), root=ROOT), root=ROOT)
    for item in ("advertised", "prompt_layers", "embodiment", "observation_projection"):
        assert ours[item] == theirs[item], item
    policy, default = ours["settings"]["policy"], theirs["settings"]["policy"]
    assert {key for key in set(policy) | set(default) if policy.get(key) != default.get(key)} <= _provider_owned()


def test_without_runtime_task_a_robotwin_plan_silently_withholds_the_contract(workspace):
    """The pitfall the builder encodes: the plan still resolves, the contract is just not delivered."""
    spec, manifest = _robotwin(workspace)
    benchmark_spec.compose(spec, seed_manifest=manifest)
    del spec["runtime"]["task"]
    disclosure = resolve_run_spec(spec, root=ROOT)["policy"]["task_contract"]
    assert disclosure["delivered"] is False and disclosure["reason"] == "runtime_task_identity_unavailable"


def test_a_libero_pro_cell_resolves_the_default_instance_without_servo(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec = _libero(service_report=report, cohort_id="benchmark-spec-fixture-g001")
    assert (spec["config"], spec["task"], spec["seed"]) == ("lp-libero_10_swap-t06-s0", "libero_10_swap-t06", 0)
    assert spec["runtime"] == {
        "kind": "libero_pro", "required_host": "test-host", "resource_profile": "libero-gpu4b",
        "suite": "libero_10_swap", "task_id": 6,
        "expected_task_language": "put the white mug on the plate and put the chocolate pudding to the right of the plate"}
    assert spec["task_instruction"] == spec["runtime"]["expected_task_language"]
    assert spec["qualification"] == {"required": False}
    plan = resolve_run_spec(spec, root=ROOT)
    instance = CATALOG["instances"]["libero_pro"]
    assert (plan["policy"]["baseline"]["id"], plan["policy"]["baseline"]["selection"]) == (
        BASELINE_ID, "omitted_tool_surface")
    assert plan["policy"]["tool_surface"]["profile"] == instance["tool_surface"] == (
        "libero_placement_direct_overhead_v1")
    assert plan["runtime"]["environment_driver"] == "direct_libero_v1"
    assert not SERVO_KEYS & set(spec["runtime"]) and not SERVO_KEYS & set(plan["runtime"])
    assert plan["policy"]["task_contract"]["reason"] == "backend_has_no_reviewed_task_contract_registry"
    assert plan["prerequisite"]["services"]["segmentation"]["tool_surface"] == instance["tool_surface"]
    assert plan["budgets"]["max_sim_steps"] == 10000 and plan["policy"]["reasoning_effort"] == "low"
    ours = model_visible_items(plan, root=ROOT)
    theirs = model_visible_items(resolve_run_spec(_fresh("libero_pro", spec), root=ROOT), root=ROOT)
    for item in ("advertised", "prompt_layers", "embodiment", "observation_projection"):
        assert ours[item] == theirs[item], item
    # A Servo key from the official suite's direct_v2 cells is refused: Servo is not in this tree.
    spec["runtime"]["episode_physics_budget"] = 250000
    with pytest.raises(RunSpecError, match="bounded Servo is not part of this tree"):
        resolve_run_spec(spec, root=ROOT)


def test_a_libero_base_cell_runs_on_the_libero_pro_backend_with_the_official_language(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec = benchmark_spec.build_libero_spec("libero_object", 0, seed=3, experiment=EXPERIMENT,
                                            resource_profile="libero-gpu4b", service_report=report,
                                            cohort_id="benchmark-spec-fixture-g001")
    assert (spec["config"], spec["task"], spec["milestone"]) == (
        "lb-libero_object-t00-s3", "libero_object-t00", "LIBERO")
    assert spec["runtime"]["suite"] == "libero_object" and spec["runtime"]["task_id"] == 0
    assert spec["task_instruction"] == spec["runtime"]["expected_task_language"] == (
        "pick up the alphabet soup and place it in the basket")
    assert spec["runtime"]["task_language_divergence_ack"] == {
        "bddl_task_language": "Pick the alphabet soup and place it in the basket",
        "policy_task_language": "pick up the alphabet soup and place it in the basket",
        "review": benchmark_spec.BASE_LANGUAGE_DIVERGENCE_REVIEW}
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["task_language_divergence_ack"] == spec["runtime"]["task_language_divergence_ack"]
    assert plan["policy"]["tool_surface"]["profile"] == CATALOG["instances"]["libero_pro"]["tool_surface"]
    assert plan["runtime"]["suite"] == "libero_object"
    with pytest.raises(ValueError, match="not a LIBERO base suite"):
        benchmark_spec.build_libero_spec("libero_90", 0, seed=0, experiment=EXPERIMENT,
                                         resource_profile="libero-gpu4b")


def test_every_base_cell_bddl_language_is_the_pinned_catalogs():
    from viewer.libero_pro_catalog import TASK_BY_KEY

    for task in TASK_BY_KEY.values():
        if task.base_suite in benchmark_spec.BASE_SUITES:
            assert base_bddl_language(task.base_suite, task.task_id) == task.canonical_instruction


def test_a_base_cell_instructs_with_the_official_file_name_language_and_acknowledges_only_real_divergence(workspace):
    # LIBERO's Task.language, which the original evaluation feeds its policies.
    assert base_task_language("libero_spatial", 4) == (
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate")
    assert base_bddl_language("libero_spatial", 4) == (
        "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate")
    assert base_task_language("libero_10", 4) == base_bddl_language("libero_10", 4) == (
        "put the white mug on the left plate and put the yellow and white mug on the right plate")
    differing = []
    for suite in benchmark_spec.BASE_SUITES:
        for task_id in range(10):
            spec = benchmark_spec.build_libero_spec(suite, task_id, seed=0, experiment=EXPERIMENT,
                                                    resource_profile="libero-gpu4b")
            ack = spec["runtime"].get("task_language_divergence_ack")
            assert spec["runtime"]["expected_task_language"] == base_task_language(suite, task_id)
            if ack is not None:
                differing.append(suite)
                assert ack["bddl_task_language"] == base_bddl_language(suite, task_id)
                assert ack["policy_task_language"] == base_task_language(suite, task_id)
    assert (len(differing), differing.count("libero_10")) == (30, 0)


def test_preflight_admits_a_base_cell_with_its_recorded_divergence_and_refuses_another(workspace):
    from ora.orchestrator.preflight import _libero_probe_matches_plan

    runtime = benchmark_spec.build_libero_spec("libero_spatial", 4, seed=0, experiment=EXPERIMENT,
                                               resource_profile="libero-gpu4b")["runtime"]
    runtime = {**runtime, "public_environment_context": None, "robustness_case": None}
    observed = {"language_matches": True, "init_state_count": 50,
                "bddl_task_language": base_bddl_language("libero_spatial", 4), "bddl_language_matches": False}
    assert _libero_probe_matches_plan(observed, runtime, motion_enabled=False)
    redirected = {**observed, "bddl_task_language": "Pick the akita black bowl on the stove and place it on the plate"}
    assert not _libero_probe_matches_plan(redirected, runtime, motion_enabled=False)


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
def test_rounds_make_one_persistent_session_that_replays_the_cell_seed(workspace, backend):
    if backend == "robotwin":
        spec, _ = _robotwin(workspace, rounds=3)
    else:
        spec = _libero(rounds=3)
    seed = spec["seed"]
    assert spec["session"] == {"cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
                               "round_seeds": [seed, seed, seed]}
    if backend == "robotwin":
        benchmark_spec.compose(spec, seed_manifest=_)
    plan = resolve_run_spec(spec, root=ROOT)
    assert (plan["session"]["rounds"], plan["session"]["round_seeds"]) == (3, [seed] * 3)
    assert "session" not in _libero()
    for rounds in (0, 33, True):
        with pytest.raises(ValueError, match=r"rounds must be an integer in \[1, 32\]"):
            _libero(rounds=rounds)


def _unsolved_manifest(directory, seed=100004, classification="oracle_failure", task="adjust_bottle", **changes):
    """A scan manifest that evaluated ``seed`` and rejected it."""
    body = dict(accepted_seeds=[], evaluated_seeds=[seed], complete=False,
                rejected=[{"classification": classification, "error": None, "requested_seed": seed}])
    body.update(changes)
    path = _official_manifest(directory, task, "demo_randomized", (), **body)
    return path.rename(directory / f"{task}-{seed}-{classification}.json")


def test_an_expert_unsolved_seed_runs_under_a_labelled_exception(workspace, monkeypatch):
    from ora.orchestrator import preflight

    monkeypatch.setattr(preflight, "resolve_workspace_root", lambda root: workspace)
    evidence = [_unsolved_manifest(workspace / "scan-a"), _unsolved_manifest(workspace / "scan-b")]
    spec = benchmark_spec.build_robotwin_spec(
        "adjust_bottle", seed=100004, experiment=EXPERIMENT, resource_profile="robotwin-gpu4",
        expert_unsolved=evidence, expert_unsolved_reason="RPent pair our expert never solved")
    exception = spec["qualification"]["exception"]
    assert spec["qualification"]["required"] is False and len(exception["evidence"]) == 2
    with pytest.raises(ValueError, match="imports its scan evidence"):
        benchmark_spec.compose(spec)
    benchmark_spec.compose(spec, expert_unsolved=evidence)
    plan = resolve_run_spec(spec, root=ROOT)
    bound = plan["qualification"]["exception"]
    assert bound["kind"] == "robotwin_expert_unsolved_seed_v1" and all(len(item["sha256"]) == 64 for item in bound["evidence"])
    check = preflight._qualification(plan)
    assert check["passed"] is True and check["detail"]["exception"] == bound["kind"]
    # Evidence edited after resolve no longer passes.
    (workspace / bound["evidence"][0]["path"]).write_text("{}")
    assert preflight._qualification(plan)["passed"] is False
    # Without the exception the default RoboTwin baseline still refuses an unqualified seed.
    plan["qualification"] = {"required": False}
    assert preflight._qualification(plan)["passed"] is False


@pytest.mark.parametrize("change, match", [
    (dict(accepted_seeds=[100004]), "does not show the expert failed"),
    (dict(rejected=[{"classification": "scene_unstable", "error": "x", "requested_seed": 100004}]),
     "does not show the expert failed"),
    (dict(conflicting_seeds=[100004]), "does not show the expert failed"),
])
def test_only_scan_evidence_of_an_expert_failure_admits_the_exception(workspace, change, match):
    evidence = _unsolved_manifest(workspace / "scan", **change)
    with pytest.raises(ValueError, match=match):
        benchmark_spec.build_robotwin_spec("adjust_bottle", seed=100004, experiment=EXPERIMENT,
                                           resource_profile="robotwin-gpu4", expert_unsolved=[evidence],
                                           expert_unsolved_reason="r")
    with pytest.raises(ValueError, match="exactly one"):
        benchmark_spec.build_robotwin_spec("adjust_bottle", seed=100004, experiment=EXPERIMENT,
                                           resource_profile="robotwin-gpu4")


def test_a_qualification_exception_is_refused_unless_well_formed(workspace):
    spec, manifest = _robotwin(workspace)
    benchmark_spec.compose(spec, seed_manifest=manifest)
    for qualification, match in (
            ({"required": True, "manifest": spec["qualification"]["manifest"],
              "exception": {"kind": "robotwin_expert_unsolved_seed_v1", "reason": "r", "evidence": ["x"]}},
             "requires qualification.required: false"),
            ({"required": False, "exception": {"kind": "other", "reason": "r", "evidence": ["x"]}}, "kind must be"),
            ({"required": False, "exception": {"kind": "robotwin_expert_unsolved_seed_v1", "reason": " ",
                                               "evidence": ["x"]}}, "reason must be"),
            ({"required": False, "exception": {"kind": "robotwin_expert_unsolved_seed_v1", "reason": "r",
                                               "evidence": []}}, "evidence must list")):
        with pytest.raises(RunSpecError, match=match):
            resolve_run_spec({**spec, "qualification": qualification}, root=ROOT)


def test_a_named_provider_profile_replaces_only_provider_owned_fields(workspace):
    low, high = _libero(), _libero(provider_profile="deepseek_flash_openai_high_16k_v1")
    assert (high["policy"]["provider_profile"], high["policy"]["reasoning_effort"]) == (
        "deepseek_flash_openai_high_16k_v1", "high")
    changed = {key for key in set(low["policy"]) | set(high["policy"])
               if low["policy"].get(key) != high["policy"].get(key)}
    assert changed <= _provider_owned() and {key for key in low if low[key] != high[key]} == {"policy"}


# --- maintained cell inputs -----------------------------------------------------------------------

def test_every_core_cell_language_is_the_pinned_catalogs():
    from viewer.libero_pro_catalog import TASK_BY_KEY

    cells = official_core_evaluation(init_states=1)["cells"]
    assert len(cells) == 160 and {cell["suite"] for cell in cells} == set(CORE_SUITES)
    for cell in cells:
        variant = TASK_BY_KEY[(cell["family"], cell["task_id"])].variant(cell["perturbation"])
        assert core_task_language(cell["suite"], cell["task_id"]) == variant.instruction
    assert core_task_language("libero_10_task", 4) == (
        "put the yellow and white mug on the left plate and put the white mug on the right plate")


def test_the_robotwin_instructions_name_seed_named_tasks():
    table = public_instructions()
    assert set(table) == set(ROBOTWIN_TASKS) and len(table) == 50
    assert sorted(task for task, entry in table.items() if entry.get("seed_named")) == [
        "place_a2b_left", "place_a2b_right"]


def test_a_seed_named_task_takes_its_instruction_from_the_episode_names(workspace):
    with pytest.raises(RoboTwinInstructionError, match="place_a2b_left:s100001"):
        _robotwin(workspace, task="place_a2b_left", seed=100001)
    names = {"episodes": {"place_a2b_left:s100001": {"instruction": "Place the red cup left of the blue box."}}}
    spec, _ = _robotwin(workspace, task="place_a2b_left", seed=100001, episode_names=names)
    assert spec["task_instruction"] == "Place the red cup left of the blue box."


@pytest.mark.parametrize("change, match", [
    ({"task_config": "demo_clean"}, "expected_task_config"),
    ({"robotwin_revision": "f" * 40}, "expected_revision"),
    ({"task": "beat_block_hammer"}, "beat_block_hammer"),
    ({"accepted_seeds": [100003]}, '"accepted": false'),
])
def test_an_official_manifest_must_admit_the_cell_as_preflight_would(workspace, change, match):
    manifest = _official_manifest(workspace / "m", **change)
    with pytest.raises(ValueError, match=match):
        _robotwin(workspace, manifest=manifest)
    tampered = _official_manifest(workspace / "t")
    body = json.loads(tampered.read_text())
    body["accepted_seeds"].append(100009)
    tampered.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='"manifest_sha256_valid": false'):
        _robotwin(workspace, manifest=tampered)


def test_an_older_per_task_manifest_admits_on_its_flags_alone(workspace):
    """The smoke's tracked seed sources: no scene configuration or revision, as preflight accepts."""
    legacy = _legacy_manifest(workspace / "legacy")
    for task_config in ("demo_randomized", "demo_clean"):
        spec, _ = _robotwin(workspace, seed=2, task_config=task_config, manifest=legacy)
        assert spec["qualification"]["manifest"].startswith(f"outputs/{EXPERIMENT}/qualification/seed-manifests/")
    with pytest.raises(ValueError, match="terminal_stability_steps"):
        _robotwin(workspace, seed=2, manifest=_legacy_manifest(workspace / "short", terminal_stability_steps=100))
    with pytest.raises(ValueError, match="does not admit"):
        _robotwin(workspace, seed=3, manifest=legacy)


def test_the_cell_must_name_a_registered_host_and_its_slot_for_this_backend(workspace, monkeypatch):
    with pytest.raises(ExecutionResourceError, match="is for libero_pro, not robotwin"):
        _robotwin(workspace, resource_profile="libero-gpu4b")
    with pytest.raises(ExecutionResourceError, match="unavailable: test-host.robotwin-gpu99"):
        _robotwin(workspace, resource_profile="robotwin-gpu99")
    with pytest.raises(ExecutionResourceError, match="host profile is unavailable"):
        _libero(host="no-such-host")
    assert _libero(host="test-arm", resource_profile="libero-gpu0")["runtime"]["required_host"] == "test-arm"
    monkeypatch.delenv("ORA_EXECUTION_HOST")
    with pytest.raises(ValueError, match="execution host is required"):
        _libero()


@pytest.mark.parametrize("suite, task_id, seed, match", [
    ("libero_spatial", 0, 0, "not a LIBERO-Pro core suite"),
    ("libero_10_swap", 10, 0, r"task_id must be in \[0,9\]"),
    ("libero_10_swap", 6, 50, r"initial state must be in \[0, 50\)"),
])
def test_a_libero_pro_cell_is_one_official_core_cell(workspace, suite, task_id, seed, match):
    with pytest.raises(ValueError, match=match):
        _libero(suite, task_id, seed)


def test_a_stale_readiness_report_is_refused_with_the_operators_error(workspace, monkeypatch):
    stale = _readiness_report(workspace / "stale", monkeypatch, age_s=86400 + 60)
    with pytest.raises(ValueError, match="service report is stale, changed or from another adapter"):
        _libero(service_report=stale)
    report = _readiness_report(workspace / "fresh", monkeypatch)
    monkeypatch.setenv("B2_SAM_PORT", "18999")
    with pytest.raises(ValueError, match="configured service endpoint differs"):
        _libero(service_report=report)


# --- writing and the operator entry point ---------------------------------------------------------

def test_compose_writes_the_spec_and_imports_its_seed_manifest_by_content(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec, manifest = _robotwin(workspace, service_report=report)
    path = benchmark_spec.compose(spec, seed_manifest=manifest)
    assert path == workspace / "outputs" / EXPERIMENT / "specs/rt-random-adjust_bottle-s100002.json"
    assert json.loads(path.read_text()) == spec
    imported = workspace / spec["qualification"]["manifest"]
    assert imported.read_bytes() == manifest.read_bytes()
    assert benchmark_spec.compose(spec, seed_manifest=manifest) == path  # idempotent
    with pytest.raises(ValueError, match="imports its seed manifest"):
        benchmark_spec.compose(spec)
    other = _official_manifest(workspace / "other", seeds=(100002, 100004))
    with pytest.raises(ValueError, match="not the seed manifest the RunSpec names"):
        benchmark_spec.compose(spec, seed_manifest=other)
    imported.write_text("{}")
    with pytest.raises(ValueError, match="imported seed manifest changed"):
        benchmark_spec.compose(spec, seed_manifest=manifest)
    # The bound endpoints reach the tool clients; a conflicting one is an error.
    assert benchmark_spec.compose(_libero(service_report=report)).name == "lp-libero_10_swap-t06-s0.json"
    # Set directly: the workspace fixture restores it (monkeypatch would restore compose's value).
    os.environ["ORA_GRASP_URL"] = "http://127.0.0.1:9999"
    with pytest.raises(RuntimeError, match="conflicts with the bound resident service"):
        benchmark_spec.compose(_libero(service_report=report))
    elsewhere = benchmark_spec.compose(_libero(), output=workspace / "elsewhere" / "cell.json")
    assert json.loads(elsewhere.read_text())["config"] == "lp-libero_10_swap-t06-s0"


def test_the_logical_workspace_link_reaches_storage_moved_by_a_selector(tmp_path, monkeypatch):
    storage, workspace_root = tmp_path / "storage", tmp_path / "workspace"
    storage.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", storage)
    target = paths.link_logical_workspace(EXPERIMENT)
    assert target == (storage / EXPERIMENT).resolve()
    logical = workspace_root / "outputs" / EXPERIMENT
    assert logical.is_symlink() and logical.resolve() == target
    assert paths.link_logical_workspace(EXPERIMENT) == target
    logical.unlink()
    logical.mkdir()
    with pytest.raises(RuntimeError, match="conflicts with configured storage"):
        paths.link_logical_workspace(EXPERIMENT)


def test_the_entry_point_lists_prints_and_composes(workspace, monkeypatch, capsys):
    assert benchmark_spec.main(["tasks", "robotwin"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert list(listed) == list(ROBOTWIN_TASKS) and listed["place_a2b_right"]["seed_named"] is True
    assert benchmark_spec.main(["tasks", "libero_pro"]) == 0
    assert json.loads(capsys.readouterr().out)["libero_10_task"][4] == core_task_language("libero_10_task", 4)
    report = _readiness_report(workspace / "reports", monkeypatch)
    manifest = _official_manifest(workspace / "manifests")
    common = ["--experiment", EXPERIMENT, "--resource-profile", "robotwin-gpu4", "--service-report", str(report),
              "--cohort", "benchmark-spec-fixture-g001", "--seed", "100002"]
    assert benchmark_spec.main(["spec", "robotwin", "adjust_bottle", "--qualification", str(manifest), *common]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _robotwin(workspace, manifest=manifest, service_report=report,
                                cohort_id="benchmark-spec-fixture-g001")[0]
    assert benchmark_spec.main(["compose", "robotwin", "adjust_bottle", "--qualification", str(manifest),
                                *common]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert json.loads(path.read_text()) == printed and (workspace / printed["qualification"]["manifest"]).is_file()
    libero = ["--experiment", EXPERIMENT, "--resource-profile", "libero-gpu4b", "--service-report", str(report),
              "--cohort", "benchmark-spec-fixture-g001", "--seed", "0"]
    assert benchmark_spec.main(["compose", "libero_pro", "libero_10_task", "4", *libero]) == 0
    composed = json.loads(Path(capsys.readouterr().out.strip()).read_text())
    assert composed["config"] == "lp-libero_10_task-t04-s0" and composed["launch_admission"]["cohort_id"] == (
        "benchmark-spec-fixture-g001")
    with pytest.raises(SystemExit):  # a RoboTwin cell needs its seed manifest
        benchmark_spec.main(["compose", "robotwin", "adjust_bottle", *common])
    with pytest.raises(SystemExit):  # and every cell a readiness report
        benchmark_spec.main(["spec", "libero_pro", "libero_10_task", "4", *libero[:4], *libero[6:]])


def test_no_bound_endpoint_outlives_this_module():
    """Regression, kept last: compose binds the readiness report's endpoints into os.environ.

    In the 2026-10-05 full suite the fixture endpoints (18124/18135) leaked from
    here into test_grounding_grasp_runtime, whose runtime reads
    ORA_SEGMENTATION_URL/GRASP_URL before its 8114/8115 defaults.
    """
    from ora.agent.model_services import operator as services

    assert {variable: os.environ.get(variable) for _, variable in SERVICE_VARIABLES} == ENDPOINTS_AT_IMPORT
    if not any(name in os.environ for name in ("B2_SAM_PORT", "B2_GRASP_PORT")):
        assert services.endpoints() == {"segmentation": "http://127.0.0.1:8114",
                                        "grasp_proposals": "http://127.0.0.1:8115"}


def test_concurrent_seed_manifest_imports_never_see_a_partial_file(tmp_path):
    # Paired cells of one seed compose together and import the same manifest; a
    # non-atomic write once let a reader refuse a half-written file (G003-03, iab-1).
    import threading
    source = tmp_path / "manifest.json"
    source.write_bytes(b"{" + b'"accepted_seeds": [' + b",".join(str(s).encode() for s in range(100000, 140000)) + b"]}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    experiment = "benchmark-spec-race"
    reference = f"outputs/{experiment}/qualification/seed-manifests/rotate_qrcode-{digest}.json"
    workspace = tmp_path / "ws"
    errors = []

    def one():
        try:
            benchmark_spec._import_seed_manifest(reference, source, workspace, experiment)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=one) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert (workspace / reference.split("/", 2)[2]).read_bytes() == source.read_bytes()


def test_the_official_episode_instruction_table_covers_the_standard_randomized_episodes():
    from ora.environment.benchmarks import robotwin_instructions as ri
    from ora.environment.benchmarks.robotwin import TASKS

    table = json.loads(ri.OFFICIAL_EPISODES_PATH.read_text(encoding="utf-8"))
    assert (table["schema"], table["task_config"], table["instruction_type"]) == (
        ri.OFFICIAL_EPISODES_SCHEMA, "demo_randomized", "unseen")
    tasks = {key.split(":s")[0] for key in table["episodes"]}
    assert tasks == set(TASKS) and len(table["episodes"]) == 5 * len(TASKS)
    for key, entry in table["episodes"].items():
        # handover_block's templates name no object or arm, so its episode values are empty.
        assert entry["instruction"].strip() and (entry["episode_info"] or key.startswith("handover_block:"))
        assert entry["rpent_witness"] in {"consistent", "scene_differs"}
        assert ("rpent_seen_sentence" in entry) == (entry["rpent_witness"] == "scene_differs")


def test_a_standard_randomized_cell_instructs_with_the_official_episode_sentence(workspace):
    from ora.environment.benchmarks.robotwin_instructions import (
        official_episode_instruction, public_instruction,
    )

    official = official_episode_instruction("adjust_bottle", 100002, "demo_randomized")
    assert official and official != public_instruction("adjust_bottle", seed=100002)
    spec, _ = _robotwin(workspace, task="adjust_bottle", seed=100002)
    assert spec["task_instruction"] == official
    assert spec["runtime"]["task_instruction_source"] == "official_episode_unseen_v1"
    clean, _ = _robotwin(workspace, task="adjust_bottle", seed=100002, task_config="demo_clean")
    assert clean["task_instruction"] == public_instruction("adjust_bottle", seed=100002)
    assert clean["runtime"]["task_instruction_source"] == "task_level"
    names = {"episodes": {"adjust_bottle:s100002": {"instruction": "Grab the brown medicine bottle."}}}
    named, _ = _robotwin(workspace, task="adjust_bottle", seed=100002, episode_names=names)
    assert named["task_instruction"] == "Grab the brown medicine bottle."
    assert named["runtime"]["task_instruction_source"] == "episode_name"
