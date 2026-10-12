"""Operator CLI for plan, preflight, launch, watch, stop and sealing."""

from __future__ import annotations

from ora.orchestrator.resource_admission import (
    _visible_gpu_ids, _memory_floor_mib, _lease_refusal,
    _resource_refusal as _resource_admission_refusal,
)

from .resource_inventory import _gpu_memory_inventory, lease_owner_pids
from .lease_claim import claim_resource_lease as _claim_resource_lease, lease_off_node

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .evaluation import EvaluationError, evaluate, export, seal
from .ownership_audit import write_ownership_audit
from .context_audit import write_context_audit
from . import paths
from .preflight import run_preflight
from .provider_gateway import gateway_preflight
from . import host_storage
from .metrics import write_runtime_metrics
from .spec import (
    PROJECT_ROOT, RunSpecError, load_run_spec, resolve_run_spec,
    require_launch_admission_for_spawn, validate_run_plan,
)
from .resources import ExecutionResourceError, resolve_execution_resource
from ora.orchestrator.coordinator import run_episode
from .context_manifest import build_context_manifest, context_placeholder
from ora.agent.composition import materialize_policy
from ora.agent.providers import load_builtin_provider_profiles
from ora.agent.providers.credentials import launch_environment
from ora.agent.evidence.policy_state import (
    PolicyStateError,
    load_policy_state,
    verify_policy_state,
)
from ora.agent.evidence.trajectory import (
    TrajectoryError, export_sft, verify_agent_trace,
)
from ora.agent.evidence import request_archive


LEASE_PATH = paths.STATE_ROOT / "active-run.json"
LEASE_ROOT = paths.STATE_ROOT / "leases"
LEASE_LOCK_PATH = LEASE_ROOT / ".registry.lock"
LEASE_SCHEMA = "ora.runtime-resource-lease.v1"

# Stable operator surface. Internal child-process commands are intentionally
# excluded; tests and the CLI reference bind this exact ordered set.
PUBLIC_COMMANDS = (
    "run",
    "doctor",
    "validate",
    "plan",
    "context",
    "preflight",
    "launch",
    "retry",
    "status",
    "workspace",
    "providers",
    "baselines",
    "experiments",
    "lock",
    "revalidate",
    "watch",
    "stop",
    "metadata",
    "state",
    "metrics",
    "ownership",
    "evaluate",
    "seal",
    "export",
    "trace-validate",
    "export-sft",
)
INTERNAL_COMMANDS = ("_supervise",)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _operator_status(root: Path) -> dict[str, Any]:
    """The run's persisted status."""

    return _load(root / "status.json")


def _run_root(run_id: str) -> Path:
    return paths.resolve_run_root(run_id)


def _runtime_revision(runtime: Mapping[str, Any]) -> str | None:
    if runtime["kind"] == "robodojo":
        from ora.environment.benchmarks.robodojo import source_revisions

        revision = "/".join(value[:12] for value in source_revisions(runtime).values())
        return revision + f" layout={runtime['layout_id']} sha256={runtime['layout_sha256'][:12]}"
    if runtime["kind"] == "libero_pro":
        # The direct driver resolves no RLinf checkout, so it has no RLinf pin.
        driver = runtime.get("environment_driver", "rlinf_v1")
        first = runtime["required_rlinf_revision"][:12] if driver == "rlinf_v1" else driver
        return "/".join((
            first,
            runtime["required_libero_pro_revision"][:12],
            runtime["required_data_revision"][:12],
        ))
    return runtime.get("required_revision")


def _display_plan(plan: Mapping[str, Any]) -> None:
    policy = plan["policy"]
    runtime = plan["runtime"]
    runtime_revision = _runtime_revision(runtime)
    rows = [
        ("plan", plan["plan_id"]), ("experiment/config", f"{plan['experiment']} / {plan['config']}"),
        ("milestone/task/seed", f"{plan['milestone']} / {plan['task']} / {plan['seed']}"),
        ("experiment identity", str(plan.get("experiment_identity") or "legacy/unmapped")),
        ("evidence", plan["evidence_tier"]),
        ("model", f"{policy['model']} / {policy['reasoning_effort']} / {policy['provider']}"),
        ("policy/runtime", f"{policy['kind']} / {runtime['kind']}"),
        ("baseline", str(policy.get("baseline", {}).get("id", "custom/legacy"))),
        ("tool surface", str(policy.get("tool_surface", {}).get("profile", "n/a"))),
        ("context retention", str(policy.get("context_retention", "n/a"))),
        ("tools", ", ".join(policy.get("tool_allowlist", ["robot.submit_action"]))),
        (
            "auto image",
            str(
                policy.get("supports_images", False)
                and policy.get("initial_observation") == "automatic"
            ),
        ),
        ("context projection", str(policy.get("context_projection", policy.get("context_profile", "n/a")))),
        ("agent session", str(policy.get("agent_session_backend", "n/a"))),
        ("rollover policy", str(policy.get("context_rollover_policy", "n/a"))),
        ("source", f"{plan['source']['commit']} dirty={plan['source']['dirty']}"),
        ("model visible", str(plan.get("policy_identity", {}).get("model_visible_sha256", "not recorded"))),
        ("evidence workspace", str(paths.workspace_from_plan(plan))),
        (
            "policy composition",
            plan.get("policy_composition", {}).get(
                "composition_sha256", plan["policy_context"]["composition_sha256"]
            ),
        ),
        ("runtime revision", runtime_revision or "synthetic"),
        (
            "execution resource",
            str(
                runtime.get("resource_binding", {}).get(
                    "profile_id", "implicit-exclusive-host"
                )
            ),
        ),
        ("budgets", json.dumps(plan["budgets"], sort_keys=True)),
        ("plan sha256", plan["plan_sha256"]),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def _preflight(plan: Mapping[str, Any], *, persist: bool = True) -> dict[str, Any]:
    result = gateway_preflight(plan, run_preflight)  # loopback gateway client first (U13)
    if persist:
        _atomic(paths.workspace_from_plan(plan) / "preflight" / f"{plan['plan_sha256']}.json", result)
    for check in result["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
    print(f"preflight: {'PASS' if result['passed'] else 'BLOCKED'}")
    return result


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _legacy_lease() -> dict[str, Any] | None:
    if not LEASE_PATH.is_file():
        return None
    try:
        value = _load(LEASE_PATH)
        if not _pid_alive(value["pid"]):
            raise ValueError("stale legacy lease")
        return {
            **value,
            "schema": "ora.legacy-active-run-lease.v1",
            "resource_class": "exclusive_host",
            "slot_id": "legacy",
            "lease_path": str(LEASE_PATH),
        }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        LEASE_PATH.unlink(missing_ok=True)
        return None


def _plan_resource_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("RunPlan runtime is unavailable for resource leasing")
    binding = runtime.get("resource_binding")
    if isinstance(binding, Mapping):
        return dict(binding)
    return resolve_execution_resource(
        None,
        runtime_kind=str(runtime.get("kind", "synthetic")),
        required_host=(
            str(runtime["required_host"])
            if isinstance(runtime.get("required_host"), str)
            else None
        ),
    )


def _read_resource_lease(path: Path) -> dict[str, Any] | None:
    try:
        value = _load(path)
        if value.get("schema") != LEASE_SCHEMA:
            raise ValueError("invalid resource lease schema")
        if lease_off_node(value):  # kept unverified: its PIDs live on another node
            return {**value, "lease_path": str(path)}
        owner_pid = value.get("supervisor_pid") or value.get("launcher_pid")
        if not _pid_alive(owner_pid):
            raise ValueError("stale resource lease")
        return {**value, "lease_path": str(path)}
    except (OSError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def _active_resource_leases_unlocked() -> list[dict[str, Any]]:
    if not LEASE_ROOT.is_dir():
        return []
    active = []
    for path in sorted(LEASE_ROOT.glob("*.json")):
        lease = _read_resource_lease(path)
        if lease is not None:
            active.append(lease)
    return active


def _active_leases() -> list[dict[str, Any]]:
    active = []
    legacy = _legacy_lease()
    if legacy is not None:
        active.append(legacy)
    active.extend(_active_resource_leases_unlocked())
    return active


def _resource_refusal(binding, active, *, inventory=None, probe_error=None):
    # Retained operator queues preview the same capacity rule as atomic claims.
    error = probe_error
    if inventory is None and error is None and any(
            _visible_gpu_ids(binding) & _visible_gpu_ids(lease) for lease in active):
        try:
            inventory = _gpu_memory_inventory(lease_owner_pids(LEASE_ROOT))
        except ExecutionResourceError as exc:
            error = str(exc)
    return _resource_admission_refusal(binding, active, inventory=inventory, probe_error=error)


# Retained source-bound operator queues use the previous helper name.
_resource_conflict = _resource_refusal


def _create_lease(
    plan: Mapping[str, Any], run_id: str, launcher_pid: int,
) -> dict[str, Any]:
    paths.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return _claim_resource_lease(_plan_resource_binding(plan), run_id=run_id,
        plan_sha256=plan["plan_sha256"], launcher_pid=launcher_pid,
        lease_root=LEASE_ROOT, lock_path=LEASE_LOCK_PATH, schema=LEASE_SCHEMA,
        active_leases=_active_leases, memory_inventory=_gpu_memory_inventory)


def _assign_lease(lease: Mapping[str, Any], supervisor_pid: int) -> dict[str, Any]:
    path = Path(str(lease["lease_path"]))
    with LEASE_LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _load(path)
        if (
            current.get("run_id") != lease.get("run_id")
            or current.get("claim_token") != lease.get("claim_token")
        ):
            raise RuntimeError("resource lease changed before supervisor assignment")
        assigned = {**current, "supervisor_pid": int(supervisor_pid)}
        _atomic(path, assigned)
        return {**assigned, "lease_path": str(path)}


def _lease_for_run(run_id: str) -> dict[str, Any] | None:
    return next(
        (lease for lease in _active_leases() if lease.get("run_id") == run_id),
        None,
    )


def _release_lease(
    run_id: str, *, supervisor_pid: int | None = None,
    claim_token: str | None = None,
) -> None:
    paths.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    with LEASE_LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lease = _lease_for_run(run_id)
        if lease is None:
            return
        if lease.get("schema") == "ora.legacy-active-run-lease.v1":
            if (
                supervisor_pid is not None
                and int(lease.get("pid", -1)) == supervisor_pid
            ):
                LEASE_PATH.unlink(missing_ok=True)
            return
        if supervisor_pid is not None and int(
            lease.get("supervisor_pid", -1)
        ) != supervisor_pid:
            return
        if claim_token is not None and lease.get("claim_token") != claim_token:
            return
        Path(str(lease["lease_path"])).unlink(missing_ok=True)


def claim_resource_lease(
    plan: Mapping[str, Any], owner_id: str, launcher_pid: int,
) -> dict[str, Any]:
    """Atomically claim a source-resolved Runtime or resident-service resource."""

    return _create_lease(plan, owner_id, launcher_pid)


def assign_resource_lease(
    lease: Mapping[str, Any], owner_pid: int,
) -> dict[str, Any]:
    """Transfer a claimed resource lease to its long-lived owner process."""

    return _assign_lease(lease, owner_pid)


def release_resource_lease(
    owner_id: str, *, owner_pid: int | None = None,
    claim_token: str | None = None,
) -> None:
    """Release exactly one matching resource lease without broad cleanup."""

    _release_lease(
        owner_id, supervisor_pid=owner_pid, claim_token=claim_token,
    )


def resource_lease_for_owner(owner_id: str) -> dict[str, Any] | None:
    """Return the active exact-owner lease, including its internal claim token."""

    return _lease_for_run(owner_id)


def _public_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    """Return operator-visible ownership without exposing the claim token."""
    return {
        key: value for key, value in lease.items()
        if key not in {"claim_token", "lease_path"}
    }


def _require_launchable(plan: Mapping[str, Any], *, retry: bool) -> None:
    require_launch_admission_for_spawn(plan)
    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping):
        return
    mode = admission.get("mode")
    if mode == "disabled":
        raise RuntimeError("this RunPlan is context-only and cannot be launched")
    if mode == "single_attempt_cell" and retry:
        raise RuntimeError("this RunPlan forbids retries after its cohort claim")


def _claim_launch_admission(
    plan: Mapping[str, Any], run_id: str | None = None,
) -> dict[str, Any] | None:
    """Atomically consume an optional plan-bound single-attempt cell."""

    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping):
        return None
    if admission.get("mode") != "single_attempt_cell":
        return None
    root = (
        paths.workspace_from_plan(plan) / "launch-admissions"
        / str(admission["cohort_id"])
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{admission['cell_id']}.json"
    claim = {
        "schema": "ora.run-launch-claim.v1",
        "run_id": run_id,
        "experiment": plan["experiment"],
        "config": plan["config"],
        "cohort_id": admission["cohort_id"],
        "cell_id": admission["cell_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source"]["commit"],
        "created_unix_s": time.time(),
    }
    encoded = (
        json.dumps(claim, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{admission['cell_id']}.", suffix=".tmp", dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            # The immutable prior claim identifies the consumed attempt. A
            # corrupt record still blocks admission, but must not hide why.
            try:
                previous = _load(path)
                if not isinstance(previous, Mapping):
                    raise ValueError("claim is not an object")
                owner = (
                    f"run_id={previous.get('run_id') or 'unrecorded'}, "
                    f"plan_id={previous.get('plan_id', 'unknown')}, "
                    f"created_unix_s={previous.get('created_unix_s', 'unknown')}"
                )
            except (OSError, ValueError):
                owner = "existing claim metadata unavailable"
            raise RuntimeError(
                "single-attempt cohort cell was already claimed: "
                f"{admission['cohort_id']}/{admission['cell_id']}; {owner}; "
                "inspect that attempt; a new attempt requires separate authorization"
            ) from exc
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": claim["schema"],
        "path": str(path),
        "claim_sha256": hashlib.sha256(encoded).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
    }


def _snapshot_launch_gate(plan: Mapping[str, Any], root: Path) -> None:
    """Copy the byte-verified admission gate into the run directory.

    The gate file under ``outputs/<experiment>/gates/`` is overwritten in
    place by every re-qualification, so the run directory keeps the exact
    launch-time bytes as its own qualification lineage.
    """

    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping) or admission.get("mode") == "disabled":
        return
    gate = admission.get("gate")
    if not isinstance(gate, Mapping) or not gate.get("path"):
        return
    workspace = paths.resolve_workspace_root(PROJECT_ROOT)
    payload = (workspace / str(gate["path"])).read_bytes()
    if (
        len(payload) != gate.get("bytes")
        or hashlib.sha256(payload).hexdigest() != gate.get("sha256")
    ):
        raise RuntimeError("launch gate changed while staging its run snapshot")
    target = root / "qualification"
    target.mkdir(parents=True, exist_ok=True)
    (target / "launch-gate.json").write_bytes(payload)
    _atomic(target / "launch-gate-snapshot.json", {
        "schema": "ora.launch-gate-snapshot.v1",
        "gate": {
            "path": gate["path"], "sha256": gate["sha256"],
            "bytes": gate["bytes"],
        },
    })


def _seal_at_landing(root: Path) -> None:
    """Seal the landed run in the same operator action, fail-open to manual.

    A refusal is recorded as evidence instead of failing the landing; no
    output may be written after a successful seal because the supervisor log
    is part of the sealed inventory.
    """

    if (root / "seal.json").exists():
        return
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        seal(root, mode="landing")
    except Exception as exc:
        _atomic(root / "seal-refusal.json", {
            "schema": "ora.seal-refusal.v1",
            "run_id": root.name, "stage": "landing", "reason": str(exc),
        })
        print(f"landing seal refused: {exc}")
        return
    request_archive.compact_after_seal(root)  # lossless, silent, fail-open (agent/evidence/request_archive.py)


def _spawn(plan: Mapping[str, Any], preflight: Mapping[str, Any], *, yes: bool, retry_of: str | None = None) -> str:
    validate_run_plan(plan)
    _require_launchable(plan, retry=retry_of is not None)
    if (
        preflight.get("schema") != "ora.preflight.v1"
        or preflight.get("passed") is not True
        or preflight.get("plan_id") != plan["plan_id"]
        or preflight.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise RuntimeError("launch preflight is not bound to this passing RunPlan")
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if (
        head.returncode or dirty.returncode
        or head.stdout.strip() != plan["source"]["commit"]
        or bool(dirty.stdout.strip())
    ):
        raise RuntimeError("source changed after preflight; refusing to stage run")
    credentialed = launch_environment(plan, preflight, os.environ)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"--retry-{int(time.time() * 1000) % 1_000_000:06d}" if retry_of else ""
    run_id = f"{plan['plan_id']}--{stamp}{suffix}"
    root = paths.new_run_root(plan, run_id)
    if paths.existing_run_location(plan, run_id) is not None:
        raise RuntimeError(f"run-id already exists: {run_id}")
    # Per-run TMPDIR off the system partition and own dead-PID carb segments
    # swept, both after confirmation and before the lease.
    staged = host_storage.stage_launch(plan, run_id)
    child_environment = {**credentialed, **host_storage.scratch_environment(staged)}
    lease: dict[str, Any] | None = None
    try:
        lease = _create_lease(plan, run_id, os.getpid())
        launch_admission = _claim_launch_admission(plan, run_id)
        root.mkdir(parents=True)
        _atomic(root / "plan.json", plan)
        _atomic(root / "preflight.json", preflight)
        _atomic(root / "launch.json", {
            "run_id": run_id, "non_interactive_confirmation": bool(yes),
            "launcher_pid": os.getpid(), "created_unix_s": time.time(),
            "retry_of": retry_of,
            "workspace": paths.workspace_topology(),
            "execution_resource": _plan_resource_binding(plan),
            "launch_admission": launch_admission,
            **staged,
        })
        _snapshot_launch_gate(plan, root)
        paths.register_run(run_id, plan, root)
        log = (root / "supervisor.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "ora.orchestrator.cli",
                    "_supervise", run_id, str(lease["claim_token"]),
                ],
                cwd=PROJECT_ROOT, env=child_environment, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
        finally:
            log.close()
        lease = _assign_lease(lease, process.pid)
    except Exception:
        if lease is not None:
            _release_lease(run_id, claim_token=str(lease["claim_token"]))
        # The staged scratch goes; unlinked dead-PID segments stay unlinked.
        host_storage.discard_staged(staged)
        raise
    _atomic(root / "process.json", {"run_id": run_id, "pid": process.pid, "process_group": process.pid})
    host_storage.bind_scratch_owner(staged, process.pid)
    print(f"launched {run_id} pid={process.pid}")
    print(f"watch: ./scripts/ora watch {run_id}")
    return run_id


def command_launch(args: argparse.Namespace) -> int:
    spec = load_run_spec(args.run_spec)
    plan = resolve_run_spec(spec)
    _display_plan(plan)
    print(f"command              ./scripts/ora launch {args.run_spec}")
    preflight = _preflight(plan)
    if not preflight["passed"]:
        return 2
    if not args.yes:
        answer = input(f"Type confirmation code {plan['confirmation_code']} to launch: ").strip()
        if answer != plan["confirmation_code"]:
            print("launch cancelled")
            return 3
    _spawn(plan, preflight, yes=bool(args.yes))
    return 0


def command_retry(args: argparse.Namespace) -> int:
    previous = _run_root(args.run_id)
    if (previous / "seal.json").exists():
        raise RuntimeError("sealed attempts cannot be retried")
    result = _load(previous / "result.json")
    # A refused proposal reached the environment although it spent no decision.
    dispatched = int(result.get("decisions", -1)) + int(result.get("refused_proposals", 0))
    if result.get("state") != "incomplete" or dispatched != 0:
        raise RuntimeError("retry is allowed only before any robot action was published")
    plan = _load(previous / "plan.json")
    validate_run_plan(plan)
    _require_launchable(plan, retry=True)
    _display_plan(plan)
    print(f"retry of             {args.run_id}")
    preflight = _preflight(plan)
    if not preflight["passed"]:
        return 2
    if not args.yes:
        answer = input(f"Type confirmation code {plan['confirmation_code']} to retry: ").strip()
        if answer != plan["confirmation_code"]:
            print("retry cancelled")
            return 3
    _spawn(plan, preflight, yes=bool(args.yes), retry_of=args.run_id)
    return 0


def command_supervise(run_id: str, claim_token: str) -> int:
    root = _run_root(run_id)
    plan = _load(root / "plan.json")
    formal_plan = (
        plan.get("schema") == "ora.run-plan.v1"
        and isinstance(plan.get("plan_sha256"), str)
        and len(plan["plan_sha256"]) == 64
    )
    if formal_plan:
        validate_run_plan(plan)
        preflight = _load(root / "preflight.json")
        if (
            preflight.get("passed") is not True
            or preflight.get("plan_sha256") != plan["plan_sha256"]
            or preflight.get("plan_id") != plan["plan_id"]
        ):
            raise RuntimeError("supervisor preflight is not bound to its RunPlan")
        head = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if (
            head.returncode or dirty.returncode
            or head.stdout.strip() != plan["source"]["commit"]
            or bool(dirty.stdout.strip())
        ):
            raise RuntimeError("supervisor source differs from the admitted clean RunPlan")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        pending = _lease_for_run(run_id)
        if (
            pending is not None
            and pending.get("claim_token") == claim_token
            and pending.get("supervisor_pid") == os.getpid()  # null until the launcher records it
        ):
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("supervisor did not receive its exact resource lease")
    try:
        result = run_episode(run_root=root, plan=plan, run_id=run_id)
        write_runtime_metrics(root)
        write_context_audit(root)
        _seal_at_landing(root)
        return 0 if result["state"] == "completed" else 1
    finally:
        _release_lease(
            run_id, supervisor_pid=os.getpid(), claim_token=claim_token,
        )
        host_storage.release_run_scratch(root)


def command_status(run_id: str | None) -> int:
    if run_id:
        root = _run_root(run_id)
        print(json.dumps(
            _operator_status(root), ensure_ascii=False, indent=2, sort_keys=True,
        ))
        return 0
    active = [_public_lease(item) for item in _active_leases()]
    print(f"active: {json.dumps(active, ensure_ascii=False, sort_keys=True) if active else 'none'}")
    for root in paths.iter_run_roots():
        status = _load(root / "status.json") if (root / "status.json").is_file() else {"state": "staged"}
        plan = _load(root / "plan.json") if (root / "plan.json").is_file() else {}
        print(f"{root.name}\t{plan.get('experiment', '—')}\t{status.get('state')}\t{status.get('official_success', '—')}")
    return 0


def command_context(plan: Mapping[str, Any]) -> int:
    policy = plan["policy"]
    if policy["kind"] != "native_api":
        raise RuntimeError("context preview is meaningful only for a model policy")
    placeholder = context_placeholder(plan)
    with tempfile.TemporaryDirectory(prefix="ora-context-") as temporary:
        compiled = materialize_policy(
            plan["policy_context"], project_root=PROJECT_ROOT,
            destination=Path(temporary),
        )
        manifest = build_context_manifest(
            plan, compiled_policy=compiled, context=placeholder
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_watch(run_id: str) -> int:
    root = _run_root(run_id)
    events = root / "events.jsonl"
    offset = 0
    try:
        while True:
            if events.is_file():
                with events.open("r", encoding="utf-8") as stream:
                    stream.seek(offset)
                    for line in stream:
                        event = json.loads(line)
                        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
                    offset = stream.tell()
            status = _load(root / "status.json") if (root / "status.json").is_file() else {}
            if status.get("state") in {"completed", "incomplete", "invalid", "blocked"}:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 130


def command_stop(run_id: str) -> int:
    root = _run_root(run_id)
    process = _load(root / "process.json")
    lease = _lease_for_run(run_id)
    if not lease or lease.get("run_id") != run_id or process.get("run_id") != run_id:
        raise RuntimeError("the exact run-id is not the active lease")
    if node := lease_off_node(lease):
        raise RuntimeError(f"{run_id} is supervised on {node}; run stop on that node")
    pid = int(process["pid"])
    probe = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    if probe.returncode or run_id not in probe.stdout:
        raise RuntimeError("refusing to signal a PID that is not this supervisor")
    os.killpg(int(process["process_group"]), signal.SIGTERM)
    print(f"graceful stop requested for {run_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ora", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quickstart = sub.add_parser("run", help="run a bounded Synthetic episode with your model API")
    quickstart.add_argument("--env", choices=("synthetic",), default="synthetic")
    doctor = sub.add_parser("doctor", help="check setup without displaying credentials or starting simulators")
    doctor.add_argument("--env", choices=("synthetic", "robotwin", "libero_pro", "robodojo"), default="synthetic")
    doctor.add_argument("--json", action="store_true")
    read_only = {
        "validate": "validate and bind one RunSpec without writing a run",
        "plan": "print the resolved, content-addressed RunPlan",
        "context": "preview exact project-owned model context and tools",
        "preflight": "run launch gates without starting an episode",
    }
    for name, help_text in read_only.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_spec", type=Path)
    launch = sub.add_parser("launch", help="preflight and launch one confirmed run")
    launch.add_argument("run_spec", type=Path)
    launch.add_argument("--yes", action="store_true")
    retry = sub.add_parser("retry", help="retry an unsealed action-free incomplete run")
    retry.add_argument("run_id")
    retry.add_argument("--yes", action="store_true")
    status = sub.add_parser("status", help="show one run or the complete local run registry")
    status.add_argument("run_id", nargs="?")
    sub.add_parser(
        "workspace",
        help="show the shared evidence/state roots and legacy discovery roots",
    )
    providers = sub.add_parser(
        "providers", help="list reusable source-tracked provider registrations",
    )
    providers.add_argument("--json", action="store_true")
    baselines = sub.add_parser(
        "baselines", help="inspect the host-independent Native robot default composition",
    )
    baselines.add_argument("--json", action="store_true")
    baselines.add_argument("--explain", action="store_true",
                           help="show how the default resolves and what each tool does")
    baselines.add_argument("--environment", metavar="KIND",
                           help="with --explain, show only this environment instance")
    experiments = sub.add_parser("experiments", help="inspect experiment families and generation lineage")
    experiments.add_argument("family", nargs="?")
    experiments.add_argument("--json", action="store_true")
    lock = sub.add_parser("lock", help="write policy.lock, the sha256 of every tracked policy asset")
    lock.add_argument("--check", action="store_true", help="exit 1 and list drift instead of writing")
    lock.add_argument("--strip-pins", action="store_true",
                      help="rewrite composable assets to path-only references, then write policy.lock")
    revalidate = sub.add_parser("revalidate", help="review completed runs into separate reports without changing originals")
    revalidate.add_argument("run_dirs", nargs="*", type=Path)
    revalidate.add_argument("--run-list", type=Path, help="JSON list of absolute run directories")
    revalidate.add_argument("--output-dir", required=True, type=Path)
    revalidate.add_argument("--workers", type=int, default=2)
    run_commands = {
        "watch": "stream one run's causal events until termination",
        "stop": "request SIGTERM for the exact active run lease",
        "metadata": "print one run's context and environment manifest",
        "state": "verify and print the latest or one saved public policy state",
        "metrics": "rebuild and print deterministic Runtime metrics",
        "ownership": "rebuild and print the behavioral-ownership audit",
        "evaluate": "deterministically evaluate one completed trace",
        "seal": "make an evaluated run inventory immutable",
        "export": "archive one sealed run for transfer",
        "trace-validate": "validate one canonical agent trace and artifact graph",
        "export-sft": "export native model calls as OpenAI tool-call JSONL",
    }
    for name, help_text in run_commands.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id")
    sub.choices["state"].add_argument("--step", type=int)
    sub.choices["seal"].add_argument(
        "--backfill", action="store_true",
        help="seal an old unsealed run with today's evaluator, pinning its "
             "recorded source identity and marking the seal as a backfill",
    )
    export_parser = sub.choices["export"]
    export_parser.add_argument("--destination", type=Path)
    sub.choices["export-sft"].add_argument("--destination", type=Path)
    internal = sub.add_parser(
        "_supervise",
        help="internal child-process entry; never invoke manually",
    )
    internal.add_argument("run_id")
    internal.add_argument("claim_token")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            from .quickstart import command_run
            return command_run(args)
        if args.command == "doctor":
            from .doctor import command_doctor
            return command_doctor(args)
        if args.command in {"validate", "plan", "context", "preflight"}:
            plan = resolve_run_spec(load_run_spec(args.run_spec))
            if args.command == "validate":
                print(json.dumps({"valid": True, "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"]}, indent=2))
            elif args.command == "plan":
                _display_plan(plan)
                print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            elif args.command == "context":
                return command_context(plan)
            else:
                return 0 if _preflight(plan)["passed"] else 2
            return 0
        if args.command == "launch":
            return command_launch(args)
        if args.command == "retry":
            return command_retry(args)
        if args.command == "_supervise":
            return command_supervise(args.run_id, args.claim_token)
        if args.command == "status":
            return command_status(args.run_id)
        if args.command == "workspace":
            print(json.dumps(paths.workspace_topology(), indent=2, sort_keys=True))
            return 0
        if args.command == "experiments":
            from ora.orchestrator.experiment_identity import command_experiments

            return command_experiments(args)
        if args.command == "lock":
            from .policy_lock import command_lock

            return command_lock(args, paths.PROJECT_ROOT)
        if args.command == "baselines":
            from .baseline_explain import command_baselines

            return command_baselines(args, paths.PROJECT_ROOT)
        if args.command == "providers":
            profiles = load_builtin_provider_profiles()
            if args.json:
                print(json.dumps(
                    profiles, ensure_ascii=False, indent=2, sort_keys=True,
                ))
            else:
                for profile_id, profile in profiles.items():
                    print(
                        f"{profile_id}\tprotocol={profile['protocol']}"
                        f"\tmodel={profile['model']}"
                        f"\tadmission={profile.get('admission', 'unspecified')}"
                        f"\tusage={profile.get('usage_scope', 'unspecified')}"
                        f"\timages={str(profile['supports_images']).lower()}"
                    )
            return 0
        if args.command == "watch":
            return command_watch(args.run_id)
        if args.command == "stop":
            return command_stop(args.run_id)
        if args.command == "revalidate":
            from .revalidation import revalidate_runs
            roots = list(args.run_dirs)
            if args.run_list:
                values = json.loads(args.run_list.read_text(encoding="utf-8"))
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise EvaluationError("--run-list requires a JSON list of run directory paths")
                roots.extend(Path(value) for value in values)
            print(json.dumps(revalidate_runs(roots, args.output_dir, workers=args.workers),
                             ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        root = _run_root(args.run_id)
        if args.command == "metadata":
            print(json.dumps(
                _load(root / "context-manifest.json"),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "state":
            verification = verify_policy_state(root)
            if verification["status"] != "valid":
                raise RuntimeError(
                    "invalid policy-state evidence: " + ", ".join(verification["errors"])
                )
            print(json.dumps(
                load_policy_state(root, args.step),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "metrics":
            write_context_audit(root)
            print(json.dumps(
                write_runtime_metrics(root),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "ownership":
            print(json.dumps(
                write_ownership_audit(root),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "evaluate":
            print(json.dumps(evaluate(root), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "seal":
            seal_mode = "backfill" if getattr(args, "backfill", False) else "operator"
            print(json.dumps(
                seal(root, mode=seal_mode),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "export":
            destination = args.destination or (paths.workspace_for_run(root) / "exports" / f"{args.run_id}.tar.gz")
            print(export(root, destination))
        elif args.command == "trace-validate":
            verification = verify_agent_trace(root)
            print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if verification["status"] == "valid" else 2
        elif args.command == "export-sft":
            destination = args.destination or (
                paths.workspace_for_run(root) / "exports" / f"{args.run_id}-sft"
            )
            print(export_sft(root, destination))
        return 0
    except (
        RunSpecError, EvaluationError, PolicyStateError, RuntimeError, ValueError,
        OSError, json.JSONDecodeError, TrajectoryError,
    ) as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
