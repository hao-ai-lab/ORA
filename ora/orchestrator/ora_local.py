"""Local ORA entry: pinned CPU MuJoCo inputs, read-only checks and prepared tasks.

This module never controls a simulator or calls a Provider. Task selection goes
through ``ora interactive`` and its ordinary preflight and Supervisor boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ora.agent.native.tui import cli as watch_cli

from . import paths

# Keep entry-point parsing usable before ``ora setup`` installs dependencies.
# Tests bind these public names to the host-safe registry after installation.
LOCAL_SUITES = ("libero_object", "libero_goal", "libero_spatial", "libero_10")
HOST_MARKER_ENV = "ORA_EXECUTION_HOST"

DEFAULT_LOCAL_ENTRY = {
    "command": "local", "required_host": "local-cpu",
    "provider_profile": "deepseek_flash_openai_low_16k_v1",
    "suite": "libero_goal", "task_id": 0, "seed": 0,
    "confirm_launch_required": True, "simulator_cadence": "action_driven",
    "credential_alias": "DEEPSEEK_API_KEY",
}
#: A read-only ``git status`` of a fresh clone of a large asset tree cannot
#: persist its index refresh, so it re-reads files; under concurrent downloads
#: it took longer than 15 s on the 2026-10-14 single-GPU verification.
GIT_STATUS_TIMEOUT_S = 180
SOURCE_ENV = {"source_root": "LIBERO_PRO_ROOT", "data_root": "LIBERO_PRO_DATA_ROOT",
              "python": "LIBERO_PRO_PYTHON"}


def add_local_options(parser) -> None:
    parser.add_argument("--root", type=Path, help="local setup root (default: ORA_LOCAL_ROOT or the user data directory)")
    parser.add_argument("--suite", choices=LOCAL_SUITES, default=DEFAULT_LOCAL_ENTRY["suite"])
    parser.add_argument("--task-id", type=int, default=DEFAULT_LOCAL_ENTRY["task_id"],
                        help="source-bound task in the selected suite, 0 through 9 (default: 0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_LOCAL_ENTRY["seed"],
                        help="official initial state, 0 through 49 (default: 0)")
    parser.add_argument("--confirm-launch", action="store_true",
                        help="authorize the manually selected task after its normal preflight passes")
    parser.add_argument("--emit", choices=("jsonl",), help="JSON-lines interactive backend instead of the Pi UI")
    parser.add_argument("--poll", type=float, default=0.2)


def local_environment(root: Path | None = None, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return process bindings, refusing partial external simulator overrides.

    Credentials never enter the installation manifest or RunSpec. The common
    DeepSeek credential name is an alias for the unchanged default Low profile
    only when its canonical credential variable is absent.
    """
    from . import ora_local_setup as setup

    values = dict(os.environ if environ is None else environ)
    explicit = [name for name in SOURCE_ENV.values() if values.get(name)]
    if explicit and len(explicit) != len(SOURCE_ENV):
        raise RuntimeError("set all of LIBERO_PRO_ROOT, LIBERO_PRO_DATA_ROOT and LIBERO_PRO_PYTHON together, "
                           "or remove them to use ora setup")
    result: dict[str, str] = {}
    if not explicit:
        installed = setup.local_paths(root)
        result.update({env: str(installed[key]) for key, env in SOURCE_ENV.items()})
    if values.get("MUJOCO_GL"):
        graphics = setup.graphics_environment(values["MUJOCO_GL"])
    elif explicit:
        graphics = setup.graphics_environment()
    else:
        graphics = setup.environment(root)
    for name, value in graphics.items():
        if values.get(name) and values[name] != value:
            raise RuntimeError(f"{name} conflicts with the selected local graphics backend; "
                               "remove the stale graphics override or select a matching MUJOCO_GL")
        result.setdefault(name, value)
    marker = values.get(HOST_MARKER_ENV)
    expected = DEFAULT_LOCAL_ENTRY["required_host"]
    if marker and marker != expected:
        raise RuntimeError(f"local execution requires {HOST_MARKER_ENV}={expected}; "
                           "the existing host marker is for another execution treatment")
    result[HOST_MARKER_ENV] = expected
    result.update(provider_environment(values))
    return result


def provider_environment(values: Mapping[str, str], profile: str | None = None) -> dict[str, str]:
    """Default Provider bindings: the profile endpoint and the common key alias.

    Preflight's ``provider_marker`` needs the base URL as well as the key; an
    unset URL defaults to the default profile's own endpoint, so it can never
    point anywhere else. A set URL is kept and checked as configured.
    """
    from .run_spec_inputs import provider_policy

    selected = profile or DEFAULT_LOCAL_ENTRY["provider_profile"]
    policy = provider_policy(selected)
    result: dict[str, str] = {}
    if not values.get(policy["base_url_env"]):
        result[policy["base_url_env"]] = policy["endpoint"]
    if (selected == DEFAULT_LOCAL_ENTRY["provider_profile"]
            and not values.get("DEEPSEEK_LOW_API_KEY") and values.get("DEEPSEEK_API_KEY")):
        result["DEEPSEEK_LOW_API_KEY"] = values["DEEPSEEK_API_KEY"]
    return result


def provider_check(values: Mapping[str, str], profile: str | None = None) -> dict[str, Any]:
    """Launch preflight's offline ``provider_marker`` configuration check on ``values``."""
    from .preflight import provider_marker_configuration
    from .run_spec_inputs import provider_policy

    selected = profile or DEFAULT_LOCAL_ENTRY["provider_profile"]
    policy = provider_policy(selected)
    passed, detail = provider_marker_configuration(policy, values)
    return {"name": "provider_marker_configuration", "passed": passed,
            "detail": {**detail, "profile": selected,
                       "live_probe": "launch preflight"}}


def _git_check(root: Path, expected: str, name: str) -> dict[str, Any]:
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10, check=False,
                                  env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        clean = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=GIT_STATUS_TIMEOUT_S, check=False,
                               env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        passed = revision.returncode == clean.returncode == 0 and revision.stdout.strip() == expected and not clean.stdout.strip()
        detail = {"path": str(root), "expected_revision": expected,
                  "observed_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                  "clean": clean.returncode == 0 and not clean.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        passed, detail = False, {"path": str(root), "expected_revision": expected, "error": "cannot inspect source"}
    return {"name": name, "passed": passed, "detail": detail}


def doctor(root: Path | None = None, *, environ: Mapping[str, str] | None = None,
           require_frontend: bool = True, project_root: Path | None = None) -> dict[str, Any]:
    """Check inputs without network traffic, simulator imports or evidence writes.

    Readiness here is local configuration readiness. Launch still repeats the
    normal source, controller, graphics, Provider and resource admission checks.
    """
    from . import ora_local_setup as setup
    from .resources import require_registered_host

    values = dict(os.environ if environ is None else environ)
    checks: list[dict[str, Any]] = []
    try:
        bindings = local_environment(root, environ=values)
    except (OSError, RuntimeError, ValueError) as exc:
        bindings = {}
        checks.append({"name": "local_bindings", "passed": False, "detail": str(exc)})
    effective = {**values, **provider_environment(values), **bindings}
    explicit = all(values.get(name) for name in SOURCE_ENV.values())
    if not explicit:
        installation = setup.doctor(root)
        checks.extend(installation["checks"])
    else:
        checks.extend((_git_check(Path(values["LIBERO_PRO_ROOT"]).expanduser(), setup.CODE_REVISION, "libero_source"),
                       _git_check(Path(values["LIBERO_PRO_DATA_ROOT"]).expanduser(), setup.DATA_REVISION, "libero_data")))
        python = Path(values["LIBERO_PRO_PYTHON"]).expanduser()
        checks.append({"name": "environment_python", "passed": python.is_file() and os.access(python, os.X_OK),
                       "detail": {"path": str(python)}})
    try:
        require_registered_host(DEFAULT_LOCAL_ENTRY["required_host"])
        registered = True
    except ValueError:
        registered = False
    checks.append({"name": "local_host_profile", "passed": registered,
                   "detail": {"host": DEFAULT_LOCAL_ENTRY["required_host"],
                              "qualification": "identity only; launch preflight remains required"}})
    credential = "DEEPSEEK_LOW_API_KEY" if values.get("DEEPSEEK_LOW_API_KEY") else (
        "DEEPSEEK_API_KEY" if values.get("DEEPSEEK_API_KEY") else None)
    checks.append({"name": "provider_credential", "passed": credential is not None,
                   "detail": {"profile": DEFAULT_LOCAL_ENTRY["provider_profile"], "credential_env": credential,
                              "hint": "export DEEPSEEK_LOW_API_KEY before starting ORA"}})
    checks.append(provider_check(effective))
    if require_frontend:
        checks.append({"name": "terminal_ui", "passed": watch_cli._pi_front_end() is not None,
                       "detail": "Node and npm ci --prefix ora-tui"})
    source_root = project_root or paths.PROJECT_ROOT
    try:
        source = subprocess.run(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=GIT_STATUS_TIMEOUT_S, check=False,
                                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        source_ready = source.returncode == 0 and not source.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        source_ready = False
    checks.append({"name": "harness_source", "passed": source_ready,
                   "detail": "a clean committed checkout is required before live launch"})
    # Deliberately report only input paths and credential variable names.
    input_paths = {key: effective.get(env) for key, env in SOURCE_ENV.items()}
    return {"schema": "ora.ora-local-doctor.v1", "ready": all(check["passed"] for check in checks),
            "checks": checks, "paths": input_paths, "live_preflight_required": True,
            "default": dict(DEFAULT_LOCAL_ENTRY)}


def print_doctor(report: Mapping[str, Any], *, stream=None) -> None:
    import sys

    output = stream or sys.stdout
    print("ORA local: ready for task selection" if report["ready"] else "ORA local: setup needs attention", file=output)
    for check in report["checks"]:
        if check["passed"]:
            continue
        print(f"  {check['name']}: {json.dumps(check['detail'], ensure_ascii=False)}", file=output)
    if not report["ready"]:
        print("Run ./scripts/ora setup for the pinned CPU simulator and terminal UI. "
              "Set DEEPSEEK_LOW_API_KEY, then run ./scripts/ora local --confirm-launch.", file=output)


def prepare_task(*, suite: str, task_id: int, seed: int, root: Path | None = None) -> Path:
    """Compose and validate a source-bound local task, then write immutable inputs.

    No launch admission is consumed here. ``PreparedTask.interactive_spec``
    binds a new single-attempt cohort only after manual task selection.
    """
    from ora.agent.composition.ora_local import compose_local_spec
    from ora.environment.benchmarks.libero_pro_language import base_bddl_language, base_task_language
    from .benchmark_spec import BASE_LANGUAGE_DIVERGENCE_REVIEW
    from .run_spec_inputs import provider_policy
    from . import ora_local_setup as setup
    from .spec import resolve_run_spec

    task = {"suite": suite, "task_id": task_id, "task_language": base_task_language(suite, task_id),
            "bddl_task_language": base_bddl_language(suite, task_id)}
    if task["bddl_task_language"] != task["task_language"]:
        task["task_language_divergence_ack"] = {
            "bddl_task_language": task["bddl_task_language"], "policy_task_language": task["task_language"],
            "review": BASE_LANGUAGE_DIVERGENCE_REVIEW,
        }
    source = {"required_libero_pro_revision": setup.CODE_REVISION,
              "required_data_revision": setup.DATA_REVISION,
              "python_env": "LIBERO_PRO_PYTHON", "libero_pro_root_env": "LIBERO_PRO_ROOT",
              "data_root_env": "LIBERO_PRO_DATA_ROOT"}
    spec = compose_local_spec(task=task, source=source,
                              provider=provider_policy(DEFAULT_LOCAL_ENTRY["provider_profile"]), seed=seed,
                              resources={"required_host": DEFAULT_LOCAL_ENTRY["required_host"]})
    resolve_run_spec(spec)
    encoded = (json.dumps(spec, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    workspace = paths.link_logical_workspace("ora-interactive")
    target = workspace / "specs" / f"local-{suite}-t{task_id:02d}-s{seed}-{digest[:16]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"prepared local RunSpec changed: {target}")
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=".ora-local-spec-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-content concurrent writer is harmless; the immutable filename
        # binds the bytes, and there is no execution claim at this stage.
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def interactive_arguments(args, *, environment: MutableMapping[str, str] | None = None,
                          world_explicit: bool = False) -> list[str]:
    """Bind local defaults and return the existing interactive command arguments."""
    import sys

    selected = os.environ if environment is None else environment
    report = doctor(args.root, environ=selected, require_frontend=args.emit is None)
    if not report["ready"]:
        print_doctor(report, stream=sys.stderr)
        raise RuntimeError("local configuration is incomplete; run ora doctor for details")
    selected.update(local_environment(args.root, environ=selected))
    path = prepare_task(suite=args.suite, task_id=args.task_id, seed=args.seed, root=args.root)
    forwarded = ["interactive", "--spec", str(path), "--poll", str(args.poll),
                 "--world-view", args.world_view, "--world-browser", args.world_browser]
    # A synthetically forwarded default port must not turn a quiet JSON
    # consumer into an implicit browser/server launch. Port zero is an opt-in
    # only when the operator actually supplied the world option.
    if world_explicit:
        forwarded += ["--world-port", str(args.world_port)]
    if args.world_public_url:
        forwarded += ["--world-public-url", args.world_public_url]
    if args.confirm_launch:
        forwarded.append("--confirm-launch")
    if args.emit:
        forwarded += ["--emit", args.emit]
    return forwarded


__all__ = ["DEFAULT_LOCAL_ENTRY", "add_local_options", "doctor", "interactive_arguments", "local_environment",
           "prepare_task", "print_doctor", "provider_check", "provider_environment"]
