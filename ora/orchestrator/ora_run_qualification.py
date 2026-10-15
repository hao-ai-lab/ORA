"""The one-time qualification ``ora run`` needs before a GPU task, reused when still valid.

- RoboTwin: a seed manifest that admits a seed of the task under the scene
  configuration (``benchmark_spec.admitted_seed_manifest``, launch
  preflight's rule); otherwise a scan of that task alone
  (``robotwin_evaluation prepare`` + ``qualify``) that stops at the first
  accepted seed.
- RoboDojo: a card and task record pair that this checkout's RoboDojo
  admission accepts for the task's one-round runtime
  (``robodojo_qualification.load_qualification``); otherwise the missing
  layers (``robodojo_lane_qualification card`` then ``task``).

Reuse is decided offline with the same validators preflight runs; launch
preflight still checks the live GPU, graphics and source bindings again.
Producers run as subprocesses of the exact CLI an operator would type, with a
progress line and their logs under ``outputs/ora-interactive/logs``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .paths import PROJECT_ROOT, WORKSPACE_ROOT, link_logical_workspace
from .ora_run_services import run_with_progress

ROBOTWIN_SLOT = "robotwin-gpu0"
ROBODOJO_SLOT = "robodojo-gpu0"
ROBODOJO_GPU = "0"
#: Measured on the 2026-10-14 verification machine (RTX PRO 6000 Blackwell).
ROBODOJO_CARD_ESTIMATE = "about 9 min on an RTX PRO 6000; 35-56 min on slower cards"
ROBODOJO_TASK_ESTIMATE = "about 4 min"

Confirm = Callable[[str], None]


def _log(name: str) -> Path:
    return link_logical_workspace("ora-interactive") / "logs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{name}.log"


def _newest(paths) -> list[Path]:
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


# --- RoboTwin -------------------------------------------------------------------------------------

def admitted_seed(path: Path, *, task: str, task_config: str, seed: int | None) -> int | None:
    """``seed``, or the first accepted seed, when the manifest at ``path`` admits it; else None."""
    from .benchmark_spec import admitted_seed_manifest

    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for candidate in [seed] if seed is not None else list(manifest.get("accepted_seeds") or []):
        try:
            admitted_seed_manifest(path, task=task, seed=candidate, task_config=task_config)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return candidate
    return None


def find_seed_manifest(task: str, *, task_config: str, seed: int | None,
                       workspace: Path | None = None) -> tuple[Path, int] | None:
    """The newest manifest admitting ``seed`` (or its first accepted seed) for ``task``."""
    from .robotwin_evaluation import WORKSPACE

    for path in _newest((workspace or WORKSPACE).glob(f"qualification/*/seed-manifests/{task}.json")):
        admitted = admitted_seed(path, task=task, task_config=task_config, seed=seed)
        if admitted is not None:
            return path, admitted
    return None


def _scan_manifest(workspace: Path, checksum: str, task: str) -> Path:
    for prepared in _newest(workspace.glob("qualification/*/prepared-plan.json")):
        try:
            if json.loads(prepared.read_text()).get("plan_sha256") == checksum:
                return prepared.parent / "seed-manifests" / f"{task}.json"
        except (OSError, ValueError):
            continue
    raise RuntimeError("the seed scan finished without its qualification record")


def robotwin_seed(task: str, *, task_config: str, seed: int | None, candidates: int, requalify: bool,
                  confirm: Confirm, environ: Mapping[str, str], stream=None) -> tuple[Path, int]:
    """A seed manifest and admitted seed for ``task``, scanning once when none is reusable."""
    from .robotwin_evaluation import WORKSPACE, prepare, save

    output = stream or sys.stderr
    if not requalify:
        found = find_seed_manifest(task, task_config=task_config, seed=seed)
        if found is not None:
            print(f"RoboTwin {task}: reusing seed {found[1]} from {found[0]}", file=output, flush=True)
            return found
    plan = prepare(robotwin_root=Path(environ["ROBOTWIN_ROOT"]), python=Path(environ["ROBOTWIN_PYTHON"]),
                   tasks=[task], slots=[ROBOTWIN_SLOT], candidate_count=candidates, accepted_per_task=1,
                   task_config=task_config)
    path = WORKSPACE / "preparations" / f"{time.time_ns()}" / "plan.json"
    save(path, plan)
    minutes = plan["qualification"]["estimated_wall_minutes"]
    confirm(f"RoboTwin {task}: one-time expert seed scan on this machine (up to {candidates} seeds from the "
            f"official start seed, stops at the first the expert solves). Upper-bound estimate {minutes:.1f} min; "
            "click_bell took 2.5 min on an RTX PRO 6000.")
    checksum = plan["plan_sha256"]
    run_with_progress([sys.executable, "-m", "ora.orchestrator.robotwin_evaluation", "qualify",
                       "--plan", path, "--confirm", f"QUALIFY-ROBOTWIN:{checksum}"],
                      label=f"RoboTwin {task} seed scan", estimate=f"estimate <= {minutes:.0f} min",
                      log=_log(f"robotwin-{task}-seed-scan"), stream=output,
                      # 2: the scan finished but did not pass (no accepted seed); explained below.
                      accepted_codes=(0, 2))
    manifest = _scan_manifest(WORKSPACE, checksum, task)
    admitted = admitted_seed(manifest, task=task, task_config=task_config, seed=seed)
    if admitted is None:
        accepted = json.loads(manifest.read_text()).get("accepted_seeds") if manifest.is_file() else []
        if seed is not None:
            raise RuntimeError(f"seed {seed} is not admitted (accepted: {accepted}); omit --seed")
        raise RuntimeError(f"the scan accepted no seed of {task} among {candidates} candidates; "
                           "rerun with a larger --candidates")
    return manifest, admitted


# --- RoboDojo -------------------------------------------------------------------------------------

def _harness_commit() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _robodojo_runtime(task: str) -> dict[str, Any]:
    """The one-round runtime a composed RunSpec resolves to, as preflight compares it."""
    from .robodojo_spec import qualification_runtime

    return qualification_runtime(task, gpu=ROBODOJO_GPU, resource_profile=ROBODOJO_SLOT, rounds=1)


def live_identity() -> dict[str, Any]:
    """What launch preflight compares a RoboDojo record with: GPU 0's UUID and the interpreter readback."""
    from .robodojo_card_qualification import _python_stack

    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                   text=True, timeout=60).splitlines()
    uuids = [row.split(", ")[1].strip() for row in rows if row.split(", ")[0].strip() == ROBODOJO_GPU]
    return {"gpu_uuid": uuids[0] if len(uuids) == 1 else None, "environment": _python_stack()}


def _matches_live(record: Mapping[str, Any], identity: Mapping[str, Any] | None) -> bool:
    if identity is None:
        return True
    device = record.get("device_binding") or {}
    return (device.get("selected_gpu_uuid") == identity["gpu_uuid"]
            and record.get("environment") == identity["environment"])


def task_record_admitted(task: str, record: Path, *, commit: str,
                         identity: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    """The ``{card, task}`` binding of ``record`` when this checkout admits it for ``task``.

    Besides the offline admission, the record must name this GPU and this
    interpreter readback, which launch preflight compares live; a new card or a
    reinstalled RoboDojo environment therefore requalifies here, not at launch.
    """
    from .robodojo_qualification import load_qualification
    from .robodojo_spec import layered_qualification

    try:
        binding = layered_qualification(record)
        runtime = {**_robodojo_runtime(task), "infrastructure_qualification": binding}
        loaded = load_qualification(runtime, workspace_root=WORKSPACE_ROOT, harness_commit=commit)
    except Exception:  # every refusal means "not reusable"; the producer decides next
        return None
    return binding if _matches_live(loaded, identity) else None


def card_record_admitted(task: str, record: Path, *, commit: str,
                         identity: Mapping[str, Any] | None = None) -> bool:
    from .robodojo_lane_qualification import card_reference
    from .robodojo_qualification import load_card_qualification

    try:
        loaded = load_card_qualification(_robodojo_runtime(task), workspace_root=WORKSPACE_ROOT,
                                         harness_commit=commit, reference=card_reference(record))
    except Exception:
        return False
    return _matches_live(loaded, identity)


def _records(kind: str) -> list[Path]:
    from .robodojo_runtime_config import OUTPUT_ROOT

    return _newest((OUTPUT_ROOT / "qualification").glob(f"robodojo-lane-{kind}*/{kind}-qualification.json"))


def _produced(stdout: str, label: str) -> Path:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    path = Path(lines[-1]) if lines else None
    if path is None or not path.is_file():
        raise RuntimeError(f"{label} printed no record path")
    return path


def robodojo_qualification(task: str, *, requalify: bool, confirm: Confirm, stream=None) -> Mapping[str, Any]:
    """The card/task qualification binding for a one-round ``task`` cell on GPU 0."""
    output = stream or sys.stderr
    commit = _harness_commit()
    identity = None if requalify else live_identity()
    if not requalify:
        for record in _records("task"):
            binding = task_record_admitted(task, record, commit=commit, identity=identity)
            if binding is not None:
                print(f"RoboDojo {task}: reusing task qualification {record}", file=output, flush=True)
                return binding
    card = None if requalify else next(
        (record for record in _records("card")
         if card_record_admitted(task, record, commit=commit, identity=identity)), None)
    module = [sys.executable, "-m", "ora.orchestrator.robodojo_lane_qualification"]
    lane = [task, "--gpu", ROBODOJO_GPU]
    if card is None:
        confirm(f"RoboDojo: one-time card qualification for this GPU, driver and installation (two full-length "
                f"{task} episodes; {ROBODOJO_CARD_ESTIMATE}), then the {task} task qualification "
                f"({ROBODOJO_TASK_ESTIMATE}).")
        stdout = run_with_progress([*module, "card", *lane, "--resource-profile", ROBODOJO_SLOT, "--yes"],
                                   label="RoboDojo card qualification", estimate=ROBODOJO_CARD_ESTIMATE,
                                   log=_log("robodojo-card"), stream=output)
        card = _produced(stdout, "card qualification")
    else:
        print(f"RoboDojo: reusing card qualification {card}", file=output, flush=True)
        confirm(f"RoboDojo {task}: one-time task qualification on that card ({ROBODOJO_TASK_ESTIMATE}).")
    stdout = run_with_progress([*module, "task", *lane, "--resource-profile", ROBODOJO_SLOT, "--card", card,
                                "--yes"], label=f"RoboDojo {task} task qualification",
                               estimate=ROBODOJO_TASK_ESTIMATE, log=_log(f"robodojo-{task}-task"), stream=output)
    record = _produced(stdout, "task qualification")
    binding = task_record_admitted(task, record, commit=commit)
    if binding is None:
        raise RuntimeError(f"the new task qualification {record} is not admitted for {task}; "
                           "run ./scripts/ora preflight on a composed spec for the reason")
    return binding


__all__ = ["admitted_seed", "card_record_admitted", "find_seed_manifest", "live_identity", "robodojo_qualification", "robotwin_seed",
           "task_record_admitted"]
