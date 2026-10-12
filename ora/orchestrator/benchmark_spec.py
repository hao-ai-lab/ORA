"""Fresh RoboTwin, LIBERO-Pro and LIBERO RunSpecs from the default baseline plus cell, provider and resource inputs.

New RoboTwin and LIBERO-Pro experiments start here, as new RoboDojo
experiments start from ``robodojo_spec``: the RunSpec this module writes names
no ToolSurface, prompt bundle, Context or semantic manifest, so
``defaults.expand_native_defaults`` composes the default catalog's instance
(``native_full_explicit_motion_direct_v4``: ``robotwin_toolchain_b2_closed_noservo_v2``
or ``libero_placement_direct_overhead_v1``, the ``current_v4`` bundles and
``native_slots_v4`` Context) and the plan records ``baseline.selection:
omitted_tool_surface``. One cell is one episode on one seed or initial state.
What the RunSpec owns comes from here:

- RoboTwin: the task, its public instruction (``robotwin_instructions``), the
  scene configuration, ``runtime.task`` (which binds the reviewed completion
  contract; without it the plan records ``task_contract.delivered: false``),
  and the seed manifest that admits the seed, imported into the namespace;
- LIBERO-Pro: the core suite, task and initial state and the BDDL language of
  that suite's task (``libero_pro_language``) as ``expected_task_language``;
  no Servo keys, because the default LIBERO surface declares no Servo;
- LIBERO: the same for an original base suite (spatial, object, goal, 10) on
  the LIBERO-Pro backend, with the official instruction the original
  evaluation feeds its policies (``Task.language``, derived from the BDDL file
  name) and, where the file's own wording differs, the exact divergence in
  ``runtime.task_language_divergence_ack``;
- optionally ``rounds``: one persistent K-round session that replays the
  cell's seed or initial state in every round (``round_seeds``);
- both: the registered execution host and resource slot, the Provider profile
  (the shared catalog default unless named), the official suite's budgets, the
  resident-service binding of a readiness report and a single-attempt
  admission.

Composing a RunSpec is not host qualification, service startup or authority to
launch: ``scripts/ora plan|preflight|launch`` still resolve and check
the written spec. The H200 official-suite delivery launcher remains on its
delivery tag and composes only there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.libero_pro import BASE_SUITES, CORE_SUITES, OFFICIAL_CORE_INIT_STATES
from ora.environment.benchmarks.libero_pro_language import (
    base_bddl_language, base_task_language, core_task_language,
)
from ora.environment.benchmarks.robotwin import TASK_CONFIGS, TASKS as ROBOTWIN_TASKS
from ora.environment.benchmarks.robotwin_instructions import (
    instruction_source, public_instruction, public_instructions,
)
from ora.agent.composition.baseline import BASELINE_CATALOGS
from ora.orchestrator.paths import PROJECT_ROOT, link_logical_workspace
from ora.orchestrator.resources import HOST_MARKER_ENV, require_registered_host, resolve_execution_resource
from ora.orchestrator.run_spec_inputs import (
    SPEC_SCHEMA, bind_service_environment, default_instance, provider_policy, service_prerequisite,
    single_attempt_admission,
)

ENVIRONMENTS = ("robotwin", "libero_pro", "libero")
#: The official suite's per-episode budgets (the H200 delivery launcher's), for both benchmarks.
DEFAULT_BUDGETS = {"max_robot_decisions": 80, "max_model_calls": 200, "max_sim_steps": 10000,
                   "wall_time_s": 21600}
DEFAULT_MILESTONES = {"robotwin": "ROBOTWIN", "libero_pro": "LIBERO-PRO", "libero": "LIBERO"}
#: The cadence of a persistent K-round session (``rounds``).
SESSION_CADENCE = "persistent_agent_k_rounds_v1"
#: The cell-id tag of each official RoboTwin scene configuration.
TASK_CONFIG_TAGS = {"demo_randomized": "random", "demo_clean": "clean"}
#: Where ``compose`` imports a RoboTwin seed manifest, under ``outputs/<experiment>``.
SEED_MANIFEST_DIRECTORY = "qualification/seed-manifests"
#: Where ``compose`` imports the scan evidence of an expert-unsolved seed.
EXPERT_UNSOLVED_DIRECTORY = "qualification/expert-unsolved"


def execution_host(host: str | None = None) -> str:
    """The registered execution host of the cell: ``host``, else this process's host marker."""
    value = host or os.environ.get(HOST_MARKER_ENV)
    if not value:
        raise ValueError(f"an execution host is required: --host or {HOST_MARKER_ENV}")
    return require_registered_host(value)


def _placement(environment: str, *, host: str | None, resource_profile: str) -> dict[str, Any]:
    """The runtime's host and slot; refused when the slot is not that host's slot for this backend."""
    required_host = execution_host(host)
    resolve_execution_resource(resource_profile, runtime_kind=environment, required_host=required_host)
    return {"kind": environment, "required_host": required_host, "resource_profile": resource_profile}


def _assemble(environment: str, *, experiment: str, config: str, task: str, instruction: str, seed: int,
              runtime: dict[str, Any], qualification: dict[str, Any], provider_profile: str | None,
              provider_catalog: Path | None, budgets: Mapping[str, int] | None, service_report: Path | None,
              cohort_id: str | None, milestone: str | None, evidence_tier: str, root: Path,
              rounds: int | None = None, harness_change: Path | None = None,
              baseline: str | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA, "experiment": experiment, "config": config,
        "output_workspace": f"outputs/{experiment}", "milestone": milestone or DEFAULT_MILESTONES[environment],
        "task": task, "task_instruction": instruction, "seed": seed, "evidence_tier": evidence_tier,
        "policy": provider_policy(provider_profile, catalog=provider_catalog), "runtime": runtime,
        "budgets": {**DEFAULT_BUDGETS, **dict(budgets or {})}, "qualification": qualification,
    }
    if baseline is not None:
        # A registered non-default baseline (policy.baseline); the default resolver composes it whole.
        if baseline not in BASELINE_CATALOGS:
            raise ValueError(f"baseline must be one of {sorted(BASELINE_CATALOGS)}")
        spec["policy"]["baseline"] = baseline
    if rounds is not None:
        spec["session"] = session_rounds(rounds, seed)
    surface = None
    if harness_change is not None:
        # A harness-change record: defaults.expand_native_defaults applies it.
        from ora.orchestrator.harness_change import bind_change, task_key

        spec["policy"]["harness_change"], resolved = bind_change(harness_change, environment, root=root,
                                                                 task=task_key(runtime))
        surface = resolved["profile"]
    if service_report is not None:
        spec["prerequisite"] = service_prerequisite(service_report, environment, root=root, surface=surface,
                                                    baseline=baseline)
    if cohort_id is not None:
        spec["launch_admission"] = single_attempt_admission(cohort_id, config)
    return spec


def session_rounds(rounds: int, seed: int) -> dict[str, Any]:
    """One persistent session of ``rounds`` rounds, each on the cell's own seed or initial state."""
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 32:
        raise ValueError("rounds must be an integer in [1, 32]")
    return {"cadence": SESSION_CADENCE, "rounds": rounds, "round_seeds": [seed] * rounds}


# --- RoboTwin -------------------------------------------------------------------------------------

def robotwin_revision(root: Path = PROJECT_ROOT) -> str:
    """The RoboTwin revision the default instance pins; an official seed manifest must name it."""
    return default_instance("robotwin", root)[1]["runtime"]["required_revision"]


def robotwin_contract_disclosure(task: str, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The reviewed completion contract's disclosure for ``task`` at the default revision."""
    from ora.agent.composition.task_contract import resolve_task_contract_layer

    runtime = {"kind": "robotwin", "task": task, "required_revision": robotwin_revision(root)}
    return resolve_task_contract_layer(runtime, root=root)[1]


def admitted_seed_manifest(path: Path, *, task: str, seed: int, task_config: str,
                           root: Path = PROJECT_ROOT) -> bytes:
    """The manifest's bytes when it admits ``seed`` of ``task`` under ``task_config``; refused otherwise.

    The rule is launch preflight's (``robotwin_evaluation.seed_admission``):
    an official-scale manifest from ``robotwin_evaluation qualify`` must name
    this scene configuration and the default RoboTwin revision and carry the
    seed's own reset/oracle record; an older per-task manifest names neither
    configuration nor revision and is admitted on its task, listed seed and
    reset/oracle/250-step flags.
    """
    from ora.orchestrator.robotwin_evaluation import seed_admission

    data = Path(path).read_bytes()
    manifest = json.loads(data)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{path} is not a RoboTwin seed manifest")
    passed, detail = seed_admission(manifest, task=task, seed=seed, task_config=task_config,
                                    revision=robotwin_revision(root))
    if not passed:
        raise ValueError(f"{path} does not admit {task} seed {seed} under {task_config}: "
                         f"{json.dumps(detail, sort_keys=True)}")
    return data


def unsolved_qualification(evidence: Sequence[Path], *, task: str, seed: int, task_config: str, experiment: str,
                           reason: str | None, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The labelled exception of an expert-unsolved seed; refused unless every manifest shows the failure."""
    from ora.orchestrator.qualification_exception import EXPERT_UNSOLVED_KIND
    from ora.orchestrator.robotwin_evaluation import expert_unsolved as unsolved

    if not reason or not reason.strip():
        raise ValueError("an expert-unsolved cell states its reason")
    references = []
    for path in evidence:
        data = Path(path).read_bytes()
        passed, detail = unsolved(json.loads(data), task=task, seed=seed, task_config=task_config,
                                  revision=robotwin_revision(root))
        if not passed:
            raise ValueError(f"{path} does not show the expert failed {task} seed {seed} under {task_config}: "
                             f"{json.dumps(detail, sort_keys=True)}")
        digest = hashlib.sha256(data).hexdigest()
        references.append(f"outputs/{experiment}/{EXPERT_UNSOLVED_DIRECTORY}/{task}-s{seed}-{digest[:16]}.json")
    return {"required": False, "exception": {"kind": EXPERT_UNSOLVED_KIND, "reason": reason,
                                              "evidence": references}}


def seed_manifest_reference(experiment: str, task: str, data: bytes) -> str:
    """The content-addressed path under ``outputs/<experiment>`` that a RunSpec names for a manifest."""
    digest = hashlib.sha256(data).hexdigest()
    return f"outputs/{experiment}/{SEED_MANIFEST_DIRECTORY}/{task}-{digest[:16]}.json"


def build_robotwin_spec(task: str, *, seed: int, experiment: str, resource_profile: str,
                        seed_manifest: Path | None = None, expert_unsolved: Sequence[Path] | None = None,
                        expert_unsolved_reason: str | None = None,
                        task_config: str = "demo_randomized", episode_names: Mapping[str, Any] | None = None,
                        host: str | None = None, config: str | None = None, provider_profile: str | None = None,
                        provider_catalog: Path | None = None, budgets: Mapping[str, int] | None = None,
                        service_report: Path | None = None, cohort_id: str | None = None,
                        milestone: str | None = None, evidence_tier: str = "behavioral",
                        rounds: int | None = None, harness_change: Path | None = None,
                        baseline: str | None = None,
                        root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One RoboTwin cell (task, seed, scene configuration) on the default baseline.

    ``seed_manifest`` must admit the seed (``admitted_seed_manifest``); the
    spec names its content-addressed import path and ``compose`` imports it.
    Instead, ``expert_unsolved`` names official scan manifests that show the
    expert failed this seed, with ``expert_unsolved_reason``: the cell then
    runs under a labelled ``qualification.exception``
    (``qualification_exception``), for an external case list scored on its
    own denominator. Exactly one of the two is given.
    """
    if task not in ROBOTWIN_TASKS:
        raise ValueError(f"unknown RoboTwin task: {task!r}")
    if task_config not in TASK_CONFIGS:
        raise ValueError("task_config must be demo_clean or demo_randomized")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("a RoboTwin seed must be a non-negative integer")
    default_instance("robotwin", root)
    disclosure = robotwin_contract_disclosure(task, root=root)
    if disclosure.get("delivered") is not True:
        raise ValueError(f"{task} has no deliverable reviewed contract: {disclosure.get('reason')}")
    if (seed_manifest is None) == (not expert_unsolved):
        raise ValueError("give exactly one of seed_manifest and expert_unsolved")
    if expert_unsolved:
        qualification = unsolved_qualification(expert_unsolved, task=task, seed=seed, task_config=task_config,
                                               experiment=experiment, reason=expert_unsolved_reason, root=root)
    else:
        data = admitted_seed_manifest(seed_manifest, task=task, seed=seed, task_config=task_config, root=root)
        qualification = {"required": True, "manifest": seed_manifest_reference(experiment, task, data)}
    runtime = {**_placement("robotwin", host=host, resource_profile=resource_profile),
               "task": task, "task_config": task_config,
               "task_instruction_source": instruction_source(task, seed=seed, episode_names=episode_names,
                                                             task_config=task_config)}
    return _assemble(
        "robotwin", experiment=experiment, config=config or f"rt-{TASK_CONFIG_TAGS[task_config]}-{task}-s{seed}",
        task=task, instruction=public_instruction(task, seed=seed, episode_names=episode_names,
                                                  task_config=task_config), seed=seed,
        runtime=runtime, qualification=qualification,
        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
        service_report=service_report, cohort_id=cohort_id, milestone=milestone,
        evidence_tier=evidence_tier, root=root, rounds=rounds, harness_change=harness_change,
        baseline=baseline)


# --- LIBERO-Pro -----------------------------------------------------------------------------------

def build_libero_pro_spec(suite: str, task_id: int, *, seed: int, experiment: str, resource_profile: str,
                          host: str | None = None, config: str | None = None,
                          provider_profile: str | None = None, provider_catalog: Path | None = None,
                          budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
                          cohort_id: str | None = None, milestone: str | None = None,
                          evidence_tier: str = "behavioral", rounds: int | None = None,
                          harness_change: Path | None = None, baseline: str | None = None,
                          root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One LIBERO-Pro core cell (suite, task, initial state ``seed``) on the default baseline.

    The official core evaluation runs initial states 0-49 of each task
    (``libero_pro.official_core_evaluation``); the Runtime selects state
    ``seed % 50``, so a seed outside that range would repeat one.
    """
    if suite not in CORE_SUITES:
        raise ValueError(f"not a LIBERO-Pro core suite: {suite!r}; choose one of {list(CORE_SUITES)}")
    return _libero_cell(suite, task_id, core_task_language, prefix="lp", benchmark="libero_pro", seed=seed,
                        experiment=experiment, resource_profile=resource_profile, host=host, config=config,
                        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
                        service_report=service_report, cohort_id=cohort_id, milestone=milestone,
                        evidence_tier=evidence_tier, rounds=rounds, harness_change=harness_change,
                        baseline=baseline, root=root)


def build_libero_spec(suite: str, task_id: int, *, seed: int, experiment: str, resource_profile: str,
                      host: str | None = None, config: str | None = None,
                      provider_profile: str | None = None, provider_catalog: Path | None = None,
                      budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
                      cohort_id: str | None = None, milestone: str | None = None,
                      evidence_tier: str = "behavioral", rounds: int | None = None,
                      harness_change: Path | None = None, baseline: str | None = None,
                      root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One original LIBERO cell (base suite, task, initial state ``seed``) on the LIBERO-Pro backend.

    The four source-paired base suites (spatial, object, goal, 10) run on the
    same pinned LIBERO-Pro code and data as the core suites; LIBERO-90 is not
    an official evaluation suite and has no pinned language here. The policy
    receives the official instruction (``base_task_language``); when the BDDL
    file words the same goal differently, the plan records both texts verbatim
    so launch preflight admits exactly that divergence.
    """
    if suite not in BASE_SUITES:
        raise ValueError(f"not a LIBERO base suite: {suite!r}; choose one of {list(BASE_SUITES)}")
    return _libero_cell(suite, task_id, base_task_language, bddl_language=base_bddl_language,
                        prefix="lb", benchmark="libero", seed=seed,
                        experiment=experiment, resource_profile=resource_profile, host=host, config=config,
                        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
                        service_report=service_report, cohort_id=cohort_id, milestone=milestone,
                        evidence_tier=evidence_tier, rounds=rounds, harness_change=harness_change,
                        baseline=baseline, root=root)


#: Why a base cell's official instruction may differ from its BDDL file's wording.
BASE_LANGUAGE_DIVERGENCE_REVIEW = (
    "official base-suite instruction: the original LIBERO evaluation feeds the policy Task.language, which the "
    "benchmark registry derives from the BDDL file name; the file's (:language) rewords the same objects and the "
    "same goal predicate (wording-only class, docs/knowhow/debug-solutions/goal-authoritative-task-language.md); "
    "the 30 differing base rows (5 differ only in capitalisation) audited against their goal atoms 2026-10-07")


def _libero_cell(suite: str, task_id: int, language, *, prefix: str, benchmark: str, seed: int, bddl_language=None,
                 experiment: str, resource_profile: str, host: str | None, config: str | None,
                 provider_profile: str | None, provider_catalog: Path | None, budgets: Mapping[str, int] | None,
                 service_report: Path | None, cohort_id: str | None, milestone: str | None,
                 evidence_tier: str, rounds: int | None, root: Path,
                 harness_change: Path | None = None, baseline: str | None = None) -> dict[str, Any]:
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed < OFFICIAL_CORE_INIT_STATES):
        raise ValueError(f"a LIBERO initial state must be in [0, {OFFICIAL_CORE_INIT_STATES})")
    default_instance("libero_pro", root)
    instruction = language(suite, task_id)
    task = f"{suite}-t{task_id:02d}"
    runtime = {**_placement("libero_pro", host=host, resource_profile=resource_profile),
               "suite": suite, "task_id": task_id, "expected_task_language": instruction}
    bddl_text = bddl_language(suite, task_id) if bddl_language is not None else None
    if bddl_text is not None and bddl_text != instruction:
        runtime["task_language_divergence_ack"] = {
            "bddl_task_language": bddl_text, "policy_task_language": instruction,
            "review": BASE_LANGUAGE_DIVERGENCE_REVIEW}
    return _assemble(
        "libero_pro", experiment=experiment, config=config or f"{prefix}-{task}-s{seed}", task=task,
        instruction=instruction, seed=seed, runtime=runtime, qualification={"required": False},
        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
        service_report=service_report, cohort_id=cohort_id,
        milestone=milestone or DEFAULT_MILESTONES[benchmark], evidence_tier=evidence_tier, root=root,
        rounds=rounds, harness_change=harness_change, baseline=baseline)


# --- writing --------------------------------------------------------------------------------------

def experiment_root(experiment: str) -> Path:
    """The experiment's storage, reachable as ``outputs/<experiment>`` from the shared workspace."""
    return link_logical_workspace(experiment)


def _import_seed_manifest(reference: str, source: Path, workspace: Path, experiment: str) -> None:
    data = Path(source).read_bytes()
    if not Path(reference).stem.endswith("-" + hashlib.sha256(data).hexdigest()[:16]):
        raise ValueError(f"{source} is not the seed manifest the RunSpec names ({reference})")
    target = workspace / Path(reference).relative_to(f"outputs/{experiment}")
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f"an imported seed manifest changed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # Lanes starting one task's seeds compose at once and import the same
        # manifest; a plain write let another lane read a half-written file and
        # refuse it as "changed". Write a private temp file and rename it into place.
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)


def compose(spec: Mapping[str, Any], *, seed_manifest: Path | None = None,
            expert_unsolved: Sequence[Path] | None = None, output: Path | None = None) -> Path:
    """Write ``spec`` (default ``outputs/<experiment>/specs/<config>.json``) and import its seed manifest.

    The manifest goes to the path the spec names under ``outputs/<experiment>``,
    which the resolver reads through the logical workspace link, so ``plan``
    and ``preflight`` find it before any run exists.
    """
    bind_service_environment(spec)
    experiment = spec["experiment"]
    workspace = experiment_root(experiment)
    reference = (spec.get("qualification") or {}).get("manifest")
    if reference is not None:
        if seed_manifest is None:
            raise ValueError("a RoboTwin RunSpec imports its seed manifest: pass seed_manifest")
        _import_seed_manifest(reference, seed_manifest, workspace, experiment)
    exception = (spec.get("qualification") or {}).get("exception") or {}
    if exception:
        if not expert_unsolved or len(expert_unsolved) != len(exception["evidence"]):
            raise ValueError("an expert-unsolved RunSpec imports its scan evidence: pass expert_unsolved")
        for reference, source in zip(exception["evidence"], expert_unsolved):
            _import_seed_manifest(reference, source, workspace, experiment)
    path = Path(output) if output is not None else workspace / "specs" / f"{spec['config']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# --- operator entry point -------------------------------------------------------------------------

def _tasks(environment: str) -> dict[str, Any]:
    if environment == "robotwin":
        return {task: {"instruction": entry.get("instruction"), "seed_named": bool(entry.get("seed_named"))}
                for task, entry in public_instructions().items()}
    if environment == "libero":
        return {suite: [base_task_language(suite, task_id) for task_id in range(10)] for suite in BASE_SUITES}
    return {suite: [core_task_language(suite, task_id) for task_id in range(10)] for suite in CORE_SUITES}


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--seed", type=int, required=True,
                         help="RoboTwin: a seed the manifest admits; LIBERO / LIBERO-Pro: initial state 0-49")
    command.add_argument("--rounds", type=int,
                         help="one persistent K-round session replaying the seed (default: one episode)")
    command.add_argument("--experiment", required=True, help="the new experiment's execution namespace")
    command.add_argument("--resource-profile", required=True, help="a slot the host registers for this backend")
    command.add_argument("--service-report", type=Path, required=True,
                         help="readiness report from `scripts/model-services qualify` (under 24 h old)")
    command.add_argument("--cohort", required=True, help="single-attempt launch admission cohort id")
    command.add_argument("--host", help=f"registered execution host (default: {HOST_MARKER_ENV})")
    command.add_argument("--provider-profile", help="default: the shared catalog's default profile")
    command.add_argument("--provider-catalog", type=Path, help="an experiment-owned Provider catalog")
    command.add_argument("--config", help="cell id (default rt-<random|clean>-<task>-s<seed>, "
                                          "lp-<suite>-tNN-s<seed> or lb-<suite>-tNN-s<seed>)")
    command.add_argument("--milestone", help="default ROBOTWIN, LIBERO-PRO or LIBERO")
    command.add_argument("--output", type=Path, help="compose: write here instead of outputs/<experiment>/specs/")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="list the RoboTwin roster or the LIBERO-Pro / LIBERO languages")
    tasks.add_argument("environment", choices=ENVIRONMENTS)
    for name, text in (("spec", "print the RunSpec"),
                       ("compose", "write the RunSpec under outputs/<experiment>/specs (and import the seed manifest)")):
        command = commands.add_parser(name, help=text)
        backends = command.add_subparsers(dest="environment", required=True)
        robotwin = backends.add_parser("robotwin", help="one RoboTwin task, seed and scene configuration")
        robotwin.add_argument("task", choices=ROBOTWIN_TASKS)
        robotwin.add_argument("--task-config", choices=sorted(TASK_CONFIGS), default="demo_randomized")
        admission = robotwin.add_mutually_exclusive_group(required=True)
        admission.add_argument("--qualification", type=Path,
                              help="seed manifest admitting the seed: <qualify dir>/seed-manifests/<task>.json "
                                   "from robotwin_evaluation qualify, or an older per-task manifest")
        admission.add_argument("--expert-unsolved", type=Path, nargs="+",
                              help="instead of --qualification: official scan manifests showing the expert "
                                   "failed this seed (runs under a labelled qualification exception)")
        robotwin.add_argument("--expert-unsolved-reason", help="why an expert-unsolved seed is run")
        robotwin.add_argument("--episode-names", type=Path,
                              help="public episode names for seed-named tasks "
                                   '({"episodes": {"<task>:s<seed>": {"instruction": ...}}})')
        _add_common(robotwin)
        libero = backends.add_parser("libero_pro", help="one LIBERO-Pro core suite task and initial state")
        libero.add_argument("suite", choices=CORE_SUITES)
        libero.add_argument("task_id", type=int, choices=range(10), metavar="task_id")
        _add_common(libero)
        base = backends.add_parser("libero", help="one original LIBERO base suite task and initial state")
        base.add_argument("suite", choices=BASE_SUITES)
        base.add_argument("task_id", type=int, choices=range(10), metavar="task_id")
        _add_common(base)
    return parser


def _spec_from_arguments(args: argparse.Namespace) -> dict[str, Any]:
    common = dict(seed=args.seed, experiment=args.experiment, resource_profile=args.resource_profile,
                  host=args.host, config=args.config, provider_profile=args.provider_profile,
                  provider_catalog=args.provider_catalog, service_report=args.service_report,
                  cohort_id=args.cohort, milestone=args.milestone, rounds=args.rounds)
    if args.environment == "libero_pro":
        return build_libero_pro_spec(args.suite, args.task_id, **common)
    if args.environment == "libero":
        return build_libero_spec(args.suite, args.task_id, **common)
    names = json.loads(args.episode_names.read_text(encoding="utf-8")) if args.episode_names else None
    return build_robotwin_spec(args.task, task_config=args.task_config, seed_manifest=args.qualification,
                               expert_unsolved=args.expert_unsolved, expert_unsolved_reason=args.expert_unsolved_reason,
                               episode_names=names, **common)


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``python -m ora.orchestrator.benchmark_spec``."""
    args = _parser().parse_args(argv)
    if args.command == "tasks":
        print(json.dumps(_tasks(args.environment), ensure_ascii=False, indent=2))
        return 0
    spec = _spec_from_arguments(args)
    if args.command == "spec":
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return 0
    print(compose(spec, seed_manifest=getattr(args, "qualification", None),
                  expert_unsolved=getattr(args, "expert_unsolved", None), output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
