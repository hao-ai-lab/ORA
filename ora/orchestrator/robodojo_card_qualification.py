"""Bounded RoboDojo infrastructure checks, with no task-solving or expert-success gate.

Moved from the robodojo-compat experiment package on 2026-09-28
(relocation U6). Tests patch ``_episode``, ``_write``, ``OUTPUT_ROOT`` and the
lease helpers on this module, so every name the producer calls is bound here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping
import uuid

from ora.environment.benchmarks.robodojo import (
    CARD_QUALIFICATION_CHECKS, CARD_QUALIFICATION_SCHEMA, QUALIFICATION_CHECKS,
    QUALIFICATION_SCHEMA, TASK_QUALIFICATION_CHECKS, TASK_QUALIFICATION_SCHEMA,
    TASK_UNPROVEN_FACTS, card_configuration_binding, configuration_binding, source_revisions,
    task_configuration_binding,
)
from ora.orchestrator.cli import (
    assign_resource_lease, claim_resource_lease, release_resource_lease,
)
from ora.orchestrator.spec import json_sha256
from ora.orchestrator.resources import (
    ExecutionResourceError, require_registered_host, resolve_execution_resource, resource_lease_key,
)
from ora.orchestrator.robodojo_qualification import (
    inspect_graphics_compatibility, validate_runtime_driver, worker_kernel_sha256,
)

from ora.orchestrator.robodojo_probe_episode import (  # noqa: F401  (re-exported)
    LEFT_X5_BASE_MOUNT_M, MOTION_PREFIX_COMMANDS, PROBE_ARM, UNREACHABLE_PROBE_MARGIN_M,
    _episode, _episode_binding, _measured_checks, _require, _sensor_evidence, _write,
    unreachable_pose_probe_target,
)
from ora.orchestrator.robodojo_runtime_config import (
    OUTPUT_ROOT, PROJECT_ROOT, runtime_configuration,
)


def _registered_host(marker: Any) -> bool:
    try:
        require_registered_host(marker)
    except ExecutionResourceError:
        return False
    return True


def _source_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    _require(not git("status", "--porcelain"), "infrastructure probes require a clean committed Harness source")
    return git("rev-parse", "HEAD")


def _python_stack() -> dict[str, Any]:
    output = subprocess.check_output([
        os.environ["ROBODOJO_PYTHON"], "-c",
        "import json,sys,hashlib; from pathlib import Path; from importlib.metadata import version; "
        "print(json.dumps({'python_executable':sys.executable, "
        "'python_binary_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),"
        "'versions':{k:version(k) for k in ('isaacsim','torch','numpy','warp-lang')}}))",
    ], text=True, timeout=30)
    return json.loads(output.strip().splitlines()[-1])


def _prepare(
    gpu: str, task: str, runtime: Mapping[str, Any] | None, owner_prefix: str,
) -> dict[str, Any]:
    """The clean-source, host, interpreter, GPU and driver gate every probe shares.

    Identical for the combined, card and task probes, so a short task probe is
    admitted on exactly the terms the 35-minute one is: a clean committed
    checkout, the Runtime's host marker, the pinned worker interpreter and package
    set, the requested card actually free, and a driver whose Vulkan
    allocation-limit repair is verified when it needs one.
    """

    _require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", owner_prefix) is not None,
             "the infrastructure probe evidence owner prefix is invalid")
    selected = runtime_configuration(gpu, task) if runtime is None else dict(runtime)
    if runtime is not None:
        _require(str(selected.get("cuda_visible_devices")) == gpu
                 and selected.get("renderer_gpu") == int(gpu),
                 "the supplied Runtime does not select the requested GPU")
        _require(selected.get("task_variant") == task,
                 "the supplied Runtime does not select the requested task")
        # The probes produce the evidence an infrastructure binding names;
        # carrying one in would qualify a Runtime against an older record.
        _require("infrastructure_qualification" not in selected,
                 "an infrastructure probe Runtime may not carry a qualification binding")
    commit = _source_commit()
    # The record is admitted by the *content* of the RoboDojo worker kernel,
    # with ``commit`` kept as lineage, so the digest is measured here once and
    # stored. ``_source_commit`` refuses a dirty tree, so the content at this
    # commit is the content the probes are about to run.
    kernel = worker_kernel_sha256(commit)
    marker = os.environ.get("ORA_EXECUTION_HOST")
    _require(_registered_host(marker) and selected.get("required_host", marker) == marker,
             "the host marker must name the Runtime's RoboDojo host")
    stack = _python_stack()
    memory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,driver_version", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    rows = [line.split(",") for line in memory.splitlines() if line.split(",")[0].strip() == gpu]
    # A slot declares its own floor (dense slots: 10 GiB); unprofiled probes keep 20000.
    floor = int((selected.get("resource_binding") or {}).get("minimum_free_memory_mib", 20000))
    _require(len(rows) == 1 and int(rows[0][1]) >= floor, "selected GPU has insufficient free memory")
    driver_version = rows[0][2].strip() if len(rows[0]) == 3 else ""
    graphics = (inspect_graphics_compatibility(Path(os.environ["ROBODOJO_PYTHON"]))
                if driver_version.startswith("595.") else None)
    validate_runtime_driver(driver_version, graphics_compatibility=graphics)
    owner = f"{owner_prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    root = OUTPUT_ROOT / "qualification" / owner
    root.mkdir(parents=True)
    return {"runtime": selected, "commit": commit, "kernel": kernel, "stack": stack,
            "driver_version": driver_version, "graphics": graphics,
            "owner": owner, "root": root}


def qualify(
    gpu: str, task: str = "arrange_largest_number", *,
    runtime: Mapping[str, Any] | None = None,
    owner_prefix: str = "robodojo-infra",
) -> Path:
    """Qualify RoboDojo infrastructure for one exact Runtime binding.

    ``runtime`` defaults to this package's own configuration, so the compat
    operator surface is unchanged. Any other caller that owns a RoboDojo
    Runtime supplies it here, because ``load_infrastructure_qualification``
    compares the recorded ``configuration`` against ``configuration_binding``
    of the Runtime its RunPlan actually carries, field for field: a
    ``robodojo_x5`` cell resolves another device and the source-owned
    ``robodojo_arx_x5_tabletop_v1`` envelope rather than this package's
    +/-1.5 m bounds, so no record produced from this package's configuration
    can satisfy that cell. Such a caller also owns its ``ROBODOJO_*``
    environment: only the default path reads this package's installation
    defaults, through ``runtime_configuration``.

    ``owner_prefix`` names the evidence directory and the resource-lease
    owner, so a foreign caller's probes stay identifiable inside the shared
    RoboDojo qualification workspace.
    """

    prepared = _prepare(gpu, task, runtime, owner_prefix)
    runtime, commit, kernel = prepared["runtime"], prepared["commit"], prepared["kernel"]
    stack, root, owner = prepared["stack"], prepared["root"], prepared["owner"]
    driver_version, graphics = prepared["driver_version"], prepared["graphics"]
    plan = {"schema": "ora.robodojo-infrastructure-plan.v1", "runtime": runtime,
            "harness_commit": commit, "worker_kernel_sha256": kernel,
            "episodes": 2, "task_solving": False, "environment": stack,
            "nvidia_driver_version": driver_version, "graphics_compatibility": graphics}
    plan["plan_sha256"] = json_sha256(plan)
    _write(root / "probe-plan.json", plan)
    lease = claim_resource_lease(plan, owner, os.getpid())
    record: dict[str, Any] = {
        "schema": QUALIFICATION_SCHEMA, "passed": False, "harness_commit": commit,
        "worker_kernel_sha256": kernel,
        "source_revisions": source_revisions(runtime), "task": task, "eval_seed": 0,
        **{key: runtime[key] for key in ("layout_id", "layout_file", "layout_sha256", "cuda_visible_devices", "sensor_profile")},
        "configuration": configuration_binding(runtime), "environment": stack,
        "nvidia_driver_version": driver_version,
        "graphics_compatibility": graphics,
        "checks": {key: False for key in QUALIFICATION_CHECKS}, "plan_sha256": plan["plan_sha256"],
    }
    destination = root / "qualification.json"
    try:
        assign_resource_lease(lease, os.getpid())
        episodes = []
        for index in range(2):
            print(f"Starting infrastructure episode {index + 1}/2 on GPU {gpu}; evidence {root}", flush=True)
            episodes.append(_episode(runtime, root / f"reset-{index}"))
        binding = _episode_binding(episodes, runtime)
        record["device_binding"] = binding["device_binding"]
        record["task_instruction"] = binding["task_instruction"]
        record["actual_source_fingerprint"] = binding["actual_source_fingerprint"]
        record["episode_reports"] = [f"reset-{index}/probe-report.json" for index in range(2)]
        # Measured per name, not a blanket flag. ``repeated_reset`` is the
        # combined record's name for what two *fresh workers* agree on, which
        # the binding comparison above is; the other four are what each
        # episode's own assertions actually completed.
        record["checks"] = {
            **_measured_checks(episodes, [
                name for name in QUALIFICATION_CHECKS if name != "repeated_reset"
            ]),
            "repeated_reset": True,
        }
        record["scene_registration"] = episodes[0]["scene_registration"]
        _require(all(item["scene_registration"] == record["scene_registration"] for item in episodes),
                 "fresh workers registered different scene objects for one layout")
        record["native_timeout_exercised"] = any(item["native_timeout_exercised"] for item in episodes)
        _require(record["native_timeout_exercised"], "neither fixed hold probe exercised the task's official native-action timeout")
        from ora.orchestrator.robodojo_embodiment import create_activation
        record["embodiment_activation"] = create_activation(root, episodes)
        _require(all(record["checks"][name] for name in QUALIFICATION_CHECKS),
                 "an infrastructure check did not complete its own assertions")
        record["passed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _write(destination, record)
        finally:
            release_resource_lease(owner, owner_pid=os.getpid(), claim_token=lease["claim_token"])
    return destination


def _card_episode_runtimes(
    runtime: Mapping[str, Any], profiles: tuple[str, str] | None,
) -> list[dict[str, Any]]:
    if profiles is None:
        return [deepcopy(dict(runtime)) for _ in range(2)]
    _require(len(profiles) == 2, "parallel card qualification requires exactly two resource profiles")
    runtimes = []
    slots = set()
    for profile in profiles:
        binding = resolve_execution_resource(
            profile, runtime_kind=str(runtime["kind"]),
            required_host=runtime.get("required_host"),
        )
        _require(binding["resource_class"] != "exclusive-host",
                 "parallel card episodes require qualified simulator slots")
        slot = resource_lease_key(binding)
        _require(slot not in slots, "parallel card episodes require distinct resource slots")
        slots.add(slot)
        _require(binding["worker_environment"].get("CUDA_VISIBLE_DEVICES")
                 == str(runtime["cuda_visible_devices"])
                 and int(runtime["renderer_gpu"]) == int(runtime["cuda_visible_devices"]),
                 "parallel card episodes must select the same CUDA and renderer GPU")
        selected = deepcopy(dict(runtime))
        selected.update(resource_profile=profile, resource_binding=binding)
        _require(card_configuration_binding(selected) == card_configuration_binding(runtime),
                 "parallel resource selection changed the measured card configuration")
        runtimes.append(selected)
    return runtimes


def _run_card_episodes(
    plan: Mapping[str, Any], root: Path, owner: str,
    runtimes: list[dict[str, Any]], *, parallel: bool,
) -> list[dict[str, Any]]:
    """Each concurrent worker has its own source-resolved lease and RPC session.

    Reserve the complete pair before boot: admission must account for both
    memory reservations. Threads only coordinate isolated RPC subprocesses;
    they never import the simulator or mutate the process environment.
    """
    leases = []
    try:
        for index in range(2 if parallel else 1):
            episode_owner = f"{owner}-reset-{index}" if parallel else owner
            episode_plan = dict(plan)
            if parallel:
                episode_plan.update(runtime=runtimes[index], episodes=1, episode_index=index)
                episode_plan.pop("plan_sha256", None)
                episode_plan["plan_sha256"] = json_sha256(episode_plan)
                _write(root / f"reset-{index}" / "probe-plan.json", episode_plan)
            lease = claim_resource_lease(episode_plan, episode_owner, os.getpid())
            leases.append((episode_owner, lease))
            assign_resource_lease(lease, os.getpid())

        def run(index: int) -> dict[str, Any]:
            print(f"Starting card episode {index + 1}/2 on GPU "
                  f"{runtimes[index]['cuda_visible_devices']}; evidence {root / f'reset-{index}'}",
                  flush=True)
            return _episode(runtimes[index], root / f"reset-{index}", scope="card")

        if parallel:
            # Executor shutdown joins both workers even if one failed, keeping
            # the other episode's lease until its session has actually closed.
            with ThreadPoolExecutor(max_workers=2) as executor:
                return list(executor.map(run, range(2)))
        return [run(index) for index in range(2)]
    finally:
        for episode_owner, lease in reversed(leases):
            release_resource_lease(episode_owner, owner_pid=os.getpid(),
                                   claim_token=lease["claim_token"])


def qualify_card(
    gpu: str, task: str = "arrange_largest_number", *,
    runtime: Mapping[str, Any] | None = None,
    owner_prefix: str = "robodojo-card",
    parallel_resource_profiles: tuple[str, str] | None = None,
) -> Path:
    """Measure the per-card, per-installation half, once, on the long probe.

    This is the same two full-length episodes ``qualify`` runs -- 35 minutes
    for an 800-action task, 56 for a 1300-action one -- and it is deliberately
    unshortened: driving one task to its own native-action limit is the only
    way to witness the official terminal transition, the counter
    reconciliation, the post-terminal refusal and the immutable post-terminal
    verdict, and those are worker facts that hold for every task on this card.

    ``task`` selects only which task's limit the boundary is witnessed on. The
    record names it in ``measured_task`` and is admissible for a different
    task, because nothing else in it is scene-derived. Pick the cheapest
    registered task for a card that has one.

    ``parallel_resource_profiles`` selects two distinct qualified slots on
    this same physical GPU. Both full episodes run concurrently, each under
    its own exact resource binding and lease. Omit it when only one slot is
    available; that path retains the serial two-episode probe.
    """

    prepared = _prepare(gpu, task, runtime, owner_prefix)
    runtime, commit, kernel = prepared["runtime"], prepared["commit"], prepared["kernel"]
    stack, root, owner = prepared["stack"], prepared["root"], prepared["owner"]
    driver_version, graphics = prepared["driver_version"], prepared["graphics"]
    episode_runtimes = _card_episode_runtimes(runtime, parallel_resource_profiles)
    plan = {"schema": "ora.robodojo-card-plan.v1", "runtime": runtime,
            "harness_commit": commit, "worker_kernel_sha256": kernel,
            "episodes": 2, "scope": "card", "task_solving": False, "environment": stack,
            "nvidia_driver_version": driver_version, "graphics_compatibility": graphics}
    if parallel_resource_profiles is not None:
        plan["parallel_execution"] = {
            "workers": 2,
            "episode_resource_bindings": [item["resource_binding"] for item in episode_runtimes],
        }
    plan["plan_sha256"] = json_sha256(plan)
    _write(root / "probe-plan.json", plan)
    record: dict[str, Any] = {
        "schema": CARD_QUALIFICATION_SCHEMA, "passed": False, "harness_commit": commit,
        "worker_kernel_sha256": kernel, "source_revisions": source_revisions(runtime),
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
        "card_configuration": card_configuration_binding(runtime),
        "environment": stack, "nvidia_driver_version": driver_version,
        "graphics_compatibility": graphics,
        "checks": {key: False for key in CARD_QUALIFICATION_CHECKS},
        "plan_sha256": plan["plan_sha256"],
        "claim": (
            "One card, one worker environment, one source content. Proves the "
            "GPU identity, sensor/RGB-D calibration, motion and control "
            "contract, official terminal lifecycle at the measured task's own "
            "native-action limit, bounded close and embodiment activation. "
            "Proves nothing about any other task's layout or scene."
        ),
    }
    destination = root / "card-qualification.json"
    try:
        episodes = _run_card_episodes(
            plan, root, owner, episode_runtimes,
            parallel=parallel_resource_profiles is not None,
        )
        binding = _episode_binding(episodes, runtime)
        record["device_binding"] = binding["device_binding"]
        record["actual_source_fingerprint"] = binding["actual_source_fingerprint"]
        record["runtime_embodiment"] = binding["runtime_embodiment"]
        record["episode_reports"] = [f"reset-{index}/probe-report.json" for index in range(2)]
        record["checks"] = {
            **_measured_checks(episodes, [
                name for name in CARD_QUALIFICATION_CHECKS if name != "worker_determinism"
            ]),
            "worker_determinism": True,
        }
        exercised = any(item["native_timeout_exercised"] for item in episodes)
        _require(exercised, "neither fixed hold probe exercised the task's official native-action timeout")
        record["measured_task"] = {
            "task": task, "layout_file": runtime["layout_file"],
            "layout_sha256": runtime["layout_sha256"],
            "native_action_limit": int(runtime["native_action_limit"]),
            "native_timeout_exercised": True,
            "task_instruction": binding["task_instruction"],
        }
        from ora.orchestrator.robodojo_embodiment import create_activation
        record["embodiment_activation"] = create_activation(root, episodes)
        _require(all(record["checks"][name] for name in CARD_QUALIFICATION_CHECKS),
                 "a card check did not complete its own assertions")
        record["passed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write(destination, record)
    return destination


def _card_reference(card_qualification: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read the card record a task probe will rest on, and its exact pair."""

    from ora.orchestrator.paths import workspace_record_locator

    path = Path(card_qualification).resolve()
    data = path.read_bytes()
    card = json.loads(data)
    _require(isinstance(card, dict) and card.get("schema") == CARD_QUALIFICATION_SCHEMA
             and card.get("passed") is True,
             f"{path} is not a passing RoboDojo card qualification record")
    # The reference the task record stores must be the same workspace-relative
    # pair the Runtime binding will name, so a card record kept anywhere else
    # is refused here rather than producing an unpairable task record. The card
    # step writes under the outputs root, which may lie outside the workspace.
    try:
        locator = workspace_record_locator(card_qualification, label="the card qualification")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    reference = {"path": locator, "sha256": hashlib.sha256(data).hexdigest()}
    return card, reference


def qualify_task(
    gpu: str, task: str, *, card_qualification: Path,
    runtime: Mapping[str, Any] | None = None,
    owner_prefix: str = "robodojo-task",
) -> Path:
    """Measure the per-task, per-scene half against one existing card record.

    Two fresh workers, each stopping after the eleven substantive commands and
    then resetting its own live plant once more. What that buys, and nothing
    else: this layout's identity and content hash, every layout instance
    resolving to a scene object, the official evaluator's registered check
    lists, this task's own registered native-action limit read back at reset,
    the upstream public instruction, RGB-D calibration inside this scene, the
    same motion and control assertions the long probe's first eleven commands
    make, and one identity-preserving repeated reset -- which the long probe
    never establishes for any task, because its two episodes are two fresh
    worker processes and ``bridge._keep_layout_identities`` only runs on a
    second reset of a live plant.

    The card record is read first and its join fields are re-measured here, so
    a probe run against a different card, a different worker interpreter or
    drifted worker-side source content produces a record that cannot pair.

    It explicitly does not establish the official terminal boundary at this
    task's limit; ``TASK_UNPROVEN_FACTS`` says so inside the record.
    """

    card, card_reference = _card_reference(card_qualification)
    prepared = _prepare(gpu, task, runtime, owner_prefix)
    runtime, commit, kernel = prepared["runtime"], prepared["commit"], prepared["kernel"]
    stack, root, owner = prepared["stack"], prepared["root"], prepared["owner"]
    driver_version, graphics = prepared["driver_version"], prepared["graphics"]
    # Fail before booting anything: a card record for another device, another
    # worker environment or another source content can never pair with the
    # record this probe is about to write.
    for field, observed in (("worker_kernel_sha256", kernel),
                            ("source_revisions", source_revisions(runtime)),
                            ("cuda_visible_devices", runtime["cuda_visible_devices"]),
                            ("sensor_profile", runtime["sensor_profile"]),
                            ("environment", stack),
                            ("card_configuration", card_configuration_binding(runtime))):
        _require(card.get(field) == observed,
                 f"the card qualification does not describe this card/installation: {field}")
    plan = {"schema": "ora.robodojo-task-plan.v1", "runtime": runtime,
            "harness_commit": commit, "worker_kernel_sha256": kernel,
            "episodes": 2, "scope": "task", "task_solving": False, "environment": stack,
            "motion_prefix_commands": MOTION_PREFIX_COMMANDS,
            "card_qualification": card_reference,
            "nvidia_driver_version": driver_version, "graphics_compatibility": graphics}
    plan["plan_sha256"] = json_sha256(plan)
    _write(root / "probe-plan.json", plan)
    lease = claim_resource_lease(plan, owner, os.getpid())
    record: dict[str, Any] = {
        "schema": TASK_QUALIFICATION_SCHEMA, "passed": False, "harness_commit": commit,
        "worker_kernel_sha256": kernel, "source_revisions": source_revisions(runtime),
        "task": task, "eval_seed": runtime["eval_seed"], "layout_id": runtime["layout_id"],
        "layout_file": runtime["layout_file"], "layout_sha256": runtime["layout_sha256"],
        "task_configuration": task_configuration_binding(runtime),
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
        "nvidia_driver_version": driver_version, "graphics_compatibility": graphics,
        "card_qualification": card_reference,
        "checks": {key: False for key in TASK_QUALIFICATION_CHECKS},
        "not_established": TASK_UNPROVEN_FACTS,
        "plan_sha256": plan["plan_sha256"],
        "claim": (
            "One task and one scene, on the card its card_qualification "
            "names. Proves this layout's identity and hash, its scene-object "
            "and official-evaluator registration, its registered "
            "native-action limit, its public instruction, RGB-D calibration "
            "in this scene, the eleven-command motion prefix and one "
            "identity-preserving repeated reset. Does not establish the "
            "official terminal boundary at this task's native-action limit."
        ),
    }
    destination = root / "task-qualification.json"
    try:
        assign_resource_lease(lease, os.getpid())
        episodes = []
        for index in range(2):
            print(f"Starting task episode {index + 1}/2 on GPU {gpu}; evidence {root}", flush=True)
            episodes.append(_episode(runtime, root / f"reset-{index}", scope="task"))
        binding = _episode_binding(episodes, runtime)
        record["task_instruction"] = binding["task_instruction"]
        record["episode_reports"] = [f"reset-{index}/probe-report.json" for index in range(2)]
        record["checks"] = _measured_checks(episodes, TASK_QUALIFICATION_CHECKS)
        registration = episodes[0]["scene_registration"]
        repeated = episodes[0]["repeated_reset_scene_registration"]
        for item in episodes:
            _require(item["scene_registration"] == registration
                     and item["repeated_reset_scene_registration"] == repeated,
                     "fresh workers registered different scene objects for one layout")
        record["scene_registration"] = registration
        record["repeated_reset_scene_registration"] = repeated
        # Measured here, in this probe's own workers, and compared with the
        # card record at admission: a copied header would join nothing.
        record["card_join"] = {
            "worker_kernel_sha256": kernel,
            "source_revisions": source_revisions(runtime),
            "cuda_visible_devices": runtime["cuda_visible_devices"],
            "sensor_profile": runtime["sensor_profile"],
            "device_binding": binding["device_binding"],
            "environment": stack,
            "actual_source_fingerprint": binding["actual_source_fingerprint"],
            "card_configuration": card_configuration_binding(runtime),
            "runtime_embodiment": binding["runtime_embodiment"],
        }
        _require(all(record["checks"][name] for name in TASK_QUALIFICATION_CHECKS),
                 "a task check did not complete its own assertions")
        record["passed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _write(destination, record)
        finally:
            release_resource_lease(owner, owner_pid=os.getpid(), claim_token=lease["claim_token"])
    return destination


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``python -m ora.orchestrator.robodojo_card_qualification``."""
    import argparse

    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = sorted(NATIVE_ACTION_LIMITS)
    single = commands.add_parser(
        "qualify", help="run source-bound infrastructure probes under the shared lease",
    )
    single.add_argument("--task", choices=tasks, default="arrange_largest_number")
    single.add_argument("--yes", action="store_true", help="confirm two bounded infrastructure episodes")
    card = commands.add_parser(
        "qualify-card",
        help="measure the per-card/installation half once; reusable by every task on this card",
    )
    card.add_argument("--task", choices=tasks, default="hang_mugs",
                      help="which task's native-action limit witnesses the official terminal transition")
    card.add_argument("--yes", action="store_true", help="confirm two full-length infrastructure episodes")
    scene = commands.add_parser("qualify-task", help="measure the per-task/scene half against a card record")
    scene.add_argument("--task", choices=tasks, required=True)
    scene.add_argument("--card", type=Path, required=True,
                       help="path to the card-qualification.json this task record rests on")
    scene.add_argument("--yes", action="store_true", help="confirm two short scene episodes")
    for command in (single, card, scene):
        command.add_argument("--gpu", required=True, choices=list("01234567"))
    args = parser.parse_args(argv)
    if not args.yes:
        parser.error(f"{args.command} requires --yes to confirm its bounded infrastructure episodes")
    if args.command == "qualify":
        print(qualify(args.gpu, args.task))
    elif args.command == "qualify-card":
        print(qualify_card(args.gpu, args.task))
    else:
        print(qualify_task(args.gpu, args.task, card_qualification=args.card))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
