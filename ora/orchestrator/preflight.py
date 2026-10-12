"""Fail-closed host, provider, simulator and qualification admission."""

from __future__ import annotations

from ora.agent.composition.baseline import BASELINE_CATALOGS

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping


from .provider_probe_diagnostics import ProviderProbeDiagnostics
from .provider_image_answer import image_answer_word as _image_answer_word, names_red as _names_red
from .provider_probe_stages import (
    _NATIVE_PROVIDER_PROBE_SCHEMA,
    ProbeRun,
    probe_failure_detail,
    probe_request_options,
    response_level_miss,
    provider_marker_verdict,
    run_probe_stages,
)
from . import host_storage
from .environment_preflight import developing_backend_checks
from .checks import check as _check, execution_host_check
# The LIBERO helpers stay importable from here for existing callers.
from .libero_preflight import (
    _libero_controller_contract_check,
    _libero_probe_matches_plan,
    _offscreen_graphics_runtime,
    libero_pro_runtime,
)
from .spec import PROJECT_ROOT
from .spec import file_sha256
from .paths import resolve_workspace_root
from .resources import (
    ExecutionResourceError,
    RESOURCE_BINDING_SCHEMA,
    resolve_execution_resource,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


SERVICE_SET_BINDING_SCHEMA = "ora.model-service-set-binding.v2"

# Binding schema -> (client class, endpoint override variable, default endpoint).
# A schema absent from this table is one this check cannot verify; it is
# reported as unverified rather than silently treated as satisfied.
_RESIDENT_SERVICE_CLIENTS: dict[str, tuple[str, str, str]] = {
    "ora.segmentation-service-binding.v1": (
        "Sam3ApiSegmenter", "ORA_SEGMENTATION_URL",
        "http://127.0.0.1:8114",
    ),
    "ora.grasp-proposal-service-binding.v1": (
        "ContactGraspNetApiProposer", "ORA_GRASP_URL",
        "http://127.0.0.1:8115",
    ),
}

_SERVICE_HEALTH_FIELDS = (
    "instance_id", "pid", "adapter_sha256", "model", "warmup_complete",
)


def _bound_tool_surface(plan: Mapping[str, Any]) -> Any:
    policy = plan.get("policy")
    surface = policy.get("tool_surface") if isinstance(policy, Mapping) else None
    return surface.get("profile") if isinstance(surface, Mapping) else None


def _verify_resident_service(
    binding: Mapping[str, Any], profile: Any,
) -> dict[str, Any]:
    """Compare one bound service against the live service, or raise."""

    schema = binding.get("schema")
    entry = _RESIDENT_SERVICE_CLIENTS.get(str(schema))
    if entry is None:
        raise ValueError(f"unknown resident service binding schema: {schema!r}")
    class_name, endpoint_variable, default_endpoint = entry
    if binding.get("tool_surface") != profile:
        raise ValueError(
            f"{schema} prerequisite is bound to a different tool surface"
        )
    from ora.agent.model_services.binding import service_client_arguments
    from ora.agent.tools import grasp as grasp_tools

    client = getattr(grasp_tools, class_name)(
        **service_client_arguments(binding, endpoint_variable, default_endpoint),
        timeout_s=5.0,
    )
    try:
        health = client.require_ready()
    finally:
        client.close()
    return {
        "schema": schema,
        "endpoint": binding.get("endpoint"),
        **{field: health.get(field) for field in _SERVICE_HEALTH_FIELDS},
    }


def _resident_service_prerequisite(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a source-bound resident service or service set before launch.

    Both shapes are checked: the single v1 segmentation binding, and the v2
    set binding an experiment emits when it holds several services at once.
    Only an absent prerequisite is "not required" -- a set binding used to fall
    through the single-schema comparison and pass vacuously while SAM3 and
    Contact-GraspNet went unverified.
    """

    prerequisite = plan.get("prerequisite")
    if not isinstance(prerequisite, Mapping):
        return _check("resident_service_prerequisite", True, "not required")
    schema = str(prerequisite.get("schema", ""))
    profile = _bound_tool_surface(plan)
    try:
        if schema == SERVICE_SET_BINDING_SCHEMA:
            services = prerequisite.get("services")
            if not isinstance(services, Mapping) or not services:
                raise ValueError("service-set binding declares no services")
            if plan.get("policy", {}).get("baseline", {}).get("id") in BASELINE_CATALOGS:
                schemas = {
                    child.get("schema") for child in services.values()
                    if isinstance(child, Mapping)
                }
                if schemas != set(_RESIDENT_SERVICE_CLIENTS):
                    raise ValueError(
                        "Native full baseline requires both segmentation and grasp services"
                    )
            scope = prerequisite.get("capability_scope")
            if not isinstance(scope, list) or not scope:
                raise ValueError("service-set binding declares no capability scope")
            observed = {
                name: _verify_resident_service(child, profile)
                for name, child in sorted(services.items())
                if isinstance(child, Mapping)
            }
            if len(observed) != len(services):
                raise ValueError("service-set binding holds a malformed service")
            return _check("resident_service_prerequisite", True, {
                "schema": schema,
                "capability_scope": list(scope),
                "services": observed,
            })
        if schema in _RESIDENT_SERVICE_CLIENTS:
            return _check(
                "resident_service_prerequisite", True,
                _verify_resident_service(prerequisite, profile),
            )
    except Exception as exc:
        return _check("resident_service_prerequisite", False, str(exc))
    # A prerequisite this check does not own. Say so instead of implying
    # a resident service was checked and found healthy.
    return _check("resident_service_prerequisite", True, {
        "verified": False,
        "reason": "no resident-service verifier for this binding schema",
        "schema": schema or "prerequisite carries no schema",
    })


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True,
        timeout=60, check=False,
    )


def _native_provider_candidate(
    plan: Mapping[str, Any], api_key: str, credential_pool: list[str],
) -> dict[str, Any]:
    """Run the complete qualification against exactly one host credential.

    The stages, their requests and the verdict live in
    ``provider_probe_stages.py``; a stage that raises refuses the candidate
    with that stage's name and the probes recorded so far.
    """

    policy = plan["policy"]
    request_options = probe_request_options(policy)
    marker = f"ROBOT-HARNESS-{secrets.token_hex(8)}"
    run = ProbeRun(ProviderProbeDiagnostics(), credential_pool)
    try:
        outcome = run_probe_stages(
            run, policy, api_key=api_key, marker=marker,
            request_options=request_options,
        )
    except Exception as exc:
        return _check("provider_marker", False, probe_failure_detail(run, exc))
    passed, detail = provider_marker_verdict(policy, run, marker, outcome)
    return _check("provider_marker", passed, detail)



def _provider_marker(plan: Mapping[str, Any], runner: Runner) -> dict[str, Any]:
    policy = plan["policy"]
    if policy["kind"] == "native_api":
        if not policy.get("require_live_marker", True):
            return _check("provider_marker", True, "not required for this policy")
        base_url = os.environ.get(policy["base_url_env"])
        api_key = os.environ.get(policy["api_key_env"])
        if not base_url or not api_key:
            return _check("provider_marker", False, {
                "configured": False,
                "required_environment": [policy["base_url_env"], policy["api_key_env"]],
            })
        if base_url.rstrip("/") != policy["endpoint"]:
            return _check("provider_marker", False, {
                "configured": True,
                "endpoint_matches_plan": False,
                "expected_endpoint": policy["endpoint"],
            })
        from ora.agent.providers.credentials import (
            CREDENTIAL_HTTP_STATUSES, SELECTION_SCHEMA, bind_credential_handoff,
            credential_ring, redact_credentials,
        )

        keys, _ = credential_ring(
            api_key, os.environ.get(f"{policy['api_key_env']}S", "").split(","),
            selected_index=0,
        )
        _, start = credential_ring(
            keys[0], keys[1:],
            (plan.get("runtime") or {}).get("resource_binding", {}).get("slot_id"),
        )
        attempts = []
        selected = None
        for offset in range(len(keys)):
            index = (start + offset) % len(keys)
            for response_retry in (False, True):
                check = _native_provider_candidate(plan, keys[index], keys)
                detail = redact_credentials(check["detail"], keys)
                check["detail"] = detail
                status = detail.get("http_status")
                attempts.append({
                    "credential_index": index, "passed": check["passed"],
                    "stage": detail.get("stage", "qualification"),
                    "http_status": status,
                    "error_type": detail.get("error_type"),
                    "response_retry": response_retry,
                })
                if check["passed"] or not response_level_miss(detail):
                    break
            if check["passed"]:
                selected = index
                break
            if status not in CREDENTIAL_HTTP_STATUSES:
                break
        check["detail"]["credential_selection"] = {
            "schema": SELECTION_SCHEMA, "pool_size": len(keys),
            "start_index": start, "selected_index": selected, "attempts": attempts,
        }
        if selected is not None:
            return bind_credential_handoff(check, plan, keys, selected)
        return check
    return _check("provider_marker", True, "not required for this policy")


def _qualification(plan: Mapping[str, Any]) -> dict[str, Any]:
    qualification = plan["qualification"]
    if not qualification["required"] and qualification.get("exception") is not None:
        from .qualification_exception import check_exception
        passed, detail = check_exception(plan, resolve_workspace_root(PROJECT_ROOT))
        return _check("seed_qualification", passed, detail)
    if not qualification["required"]:
        if (
            plan.get("policy", {}).get("baseline", {}).get("id") in BASELINE_CATALOGS
            and plan.get("runtime", {}).get("kind") == "robotwin"
        ):
            return _check(
                "seed_qualification", False,
                "Native full RoboTwin baseline requires an explicit qualified seed manifest",
            )
        return _check("seed_qualification", True, "not required")
    relative = Path(qualification["manifest"]["path"])
    path = (
        resolve_workspace_root(PROJECT_ROOT) / relative
        if relative.parts[:2] == ("outputs", plan["experiment"])
        else PROJECT_ROOT / relative
    )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _check("seed_qualification", False, str(exc))
    from .robotwin_evaluation import seed_admission
    runtime = plan.get("runtime", {})
    passed, detail = seed_admission(
        manifest, task=plan["task"], seed=plan["seed"], task_config=runtime.get("task_config"),
        revision=runtime.get("required_revision"),
    )
    return _check("seed_qualification", passed, detail)


def _robodojo_runtime(plan: Mapping[str, Any], runner: Runner) -> list[dict[str, Any]]:
    from .robodojo_preflight import _robodojo_runtime as check_runtime
    return check_runtime(plan, runner, project_root=PROJECT_ROOT, check=_check, git=_git,
                         workspace_resolver=resolve_workspace_root)


def _robotwin_control_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    from .controller_preflight import robotwin_control_binding
    return robotwin_control_binding(plan, project_root=PROJECT_ROOT, check=_check)


def _runtime(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime = plan["runtime"]
    resource_check = _execution_resource(plan)
    if runtime["kind"] == "synthetic":
        return [
            resource_check,
            _check("runtime", True, "synthetic policy-visible boundary"),
        ]
    if runtime["kind"] == "libero_pro":
        return [resource_check, *_libero_pro_runtime(plan)]
    if runtime["kind"] == "robodojo":
        return [resource_check, *developing_backend_checks(plan, robodojo_check=_robodojo_runtime, runner=subprocess.run)]
    checks = [resource_check, _robotwin_control_binding(plan)]
    checks.append(execution_host_check(runtime, os.environ.get("ORA_EXECUTION_HOST")))
    root_env = runtime["root_env"]
    raw_root = os.environ.get(root_env)
    root = Path(raw_root).expanduser().resolve() if raw_root else None
    checks.append(_check("robotwin_root", bool(root and (root / "envs").is_dir()), {
        "env": root_env, "configured": bool(raw_root),
    }))
    if root and (root / ".git").exists():
        revision = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
        checks.append(_check("robotwin_revision", revision.returncode == 0 and revision.stdout.strip() == runtime["required_revision"], {
            "expected": runtime["required_revision"], "observed": revision.stdout.strip(),
        }))
        checks.append(_check("robotwin_clean", dirty.returncode == 0 and not dirty.stdout.strip(), dirty.stdout.strip() or "clean"))
    python_env = runtime["python_env"]
    raw_python = os.environ.get(python_env)
    python = (
        Path(os.path.abspath(os.path.expanduser(raw_python))) if raw_python else None
    )
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {
        "env": python_env,
        "configured": bool(raw_python),
        "executable": str(python) if python else None,
    }))
    if executable and root:
        probe_environment = os.environ.copy()
        python_paths = [str(PROJECT_ROOT), str(root), str(root / "envs/curobo/src")]
        if probe_environment.get("PYTHONPATH"):
            python_paths.append(probe_environment["PYTHONPATH"])
        probe_environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        probe_script = """
import importlib, json, platform, sys
names = ['ora.environment.worker', 'numpy', 'PIL', 'yaml', 'sapien', 'torch', 'curobo', 'envs.' + sys.argv[1]]
modules = {name: importlib.import_module(name) for name in names}
print(json.dumps({
    'executable': sys.executable,
    'python': platform.python_version(),
    'modules': {name: getattr(module, '__file__', None) for name, module in modules.items()},
}, sort_keys=True))
"""
        try:
            probe = subprocess.run(
                [str(python), "-c", probe_script, plan["task"]],
                cwd=root,
                env=probe_environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            observed = None
            if probe.returncode == 0:
                try:
                    observed = json.loads(probe.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    pass
            checks.append(_check(
                "environment_runtime_imports",
                probe.returncode == 0 and isinstance(observed, dict),
                observed or {
                    "returncode": probe.returncode,
                    "stderr_tail": probe.stderr.strip()[-2000:],
                    "stdout_tail": probe.stdout.strip()[-1000:],
                },
            ))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(_check("environment_runtime_imports", False, str(exc)))
    if runtime.get("require_gpu", True):
        nvidia = shutil.which("nvidia-smi")
        result = subprocess.run([nvidia, "-L"], capture_output=True, text=True, timeout=30, check=False) if nvidia else None
        checks.append(_check("gpu", bool(result and result.returncode == 0 and result.stdout.strip()), result.stdout.strip() if result else "nvidia-smi unavailable"))
    return checks


def _execution_resource(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a source-tracked resource profile and its current GPU capacity."""

    runtime = plan["runtime"]
    binding = runtime.get("resource_binding")
    if binding is None:
        return _check(
            "execution_resource", True,
            "implicit exclusive-host Runtime lease",
        )
    detail: dict[str, Any] = {"binding": binding}
    try:
        if not isinstance(binding, Mapping) or binding.get("schema") != (
            RESOURCE_BINDING_SCHEMA
        ):
            raise ExecutionResourceError("resource binding schema is invalid")
        expected = resolve_execution_resource(
            binding.get("profile_id"),
            runtime_kind=str(runtime["kind"]),
            required_host=(
                str(runtime["required_host"])
                if isinstance(runtime.get("required_host"), str)
                else None
            ),
        )
        if dict(binding) != expected:
            raise ExecutionResourceError("resource profile drifted after planning")
        worker_environment = binding.get("worker_environment", {})
        cuda = worker_environment.get("CUDA_VISIBLE_DEVICES")
        if cuda is None:
            return _check("execution_resource", True, detail)
        # The first listed GPU is the lane's compute device (its CUDA ordinal 0);
        # a Mesa LIBERO lane also lists 0 so robosuite accepts EGL device 0.
        devices = cuda.split(",") if isinstance(cuda, str) else []
        if not devices or not all(item.isdigit() for item in devices):
            raise ExecutionResourceError(
                "CUDA_VISIBLE_DEVICES must list numeric GPUs"
            )
        cuda = devices[0]
        nvidia = shutil.which("nvidia-smi")
        if nvidia is None:
            raise ExecutionResourceError("nvidia-smi is unavailable")
        query = subprocess.run(
            [
                nvidia, "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if query.returncode:
            raise ExecutionResourceError(
                query.stderr.strip() or "cannot query GPU resources"
            )
        inventory: dict[str, int] = {}
        for line in query.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
                inventory[fields[0]] = int(fields[1])
        if cuda not in inventory:
            raise ExecutionResourceError(f"GPU {cuda} is unavailable")
        # One binding's own floor against live free memory. Co-residency is
        # admitted by the lease claim (cli._resource_refusal), which adds the
        # declared-capacity sum over every active lease on this device; this
        # check stays a per-binding probe and deliberately has no lease view.
        minimum = int(binding["minimum_free_memory_mib"])
        detail.update({
            "gpu_index": int(cuda),
            "free_memory_mib": inventory[cuda],
            "minimum_free_memory_mib": minimum,
        })
        if inventory[cuda] < minimum:
            raise ExecutionResourceError(
                f"GPU {cuda} has {inventory[cuda]} MiB free; {minimum} required"
            )
        return _check("execution_resource", True, detail)
    except (ExecutionResourceError, KeyError, TypeError, ValueError) as exc:
        detail["error"] = str(exc)
        return _check("execution_resource", False, detail)


def _libero_pro_runtime(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the selected LIBERO driver without starting MuJoCo."""
    return libero_pro_runtime(plan, git=_git, project_root=PROJECT_ROOT)


def run_preflight(
    plan: Mapping[str, Any], *, runner: Runner = subprocess.run,
) -> dict[str, Any]:
    revision = _git(PROJECT_ROOT, "rev-parse", "HEAD")
    dirty = _git(PROJECT_ROOT, "status", "--porcelain", "--untracked-files=normal")
    current_source = {
        "expected": plan["source"]["commit"],
        "observed": revision.stdout.strip(),
        "dirty": bool(dirty.stdout.strip()),
    }
    checks = [
        _check(
            "source_identity",
            revision.returncode == 0
            and dirty.returncode == 0
            and current_source["observed"] == current_source["expected"]
            and current_source["dirty"] is False,
            current_source,
        ),
        *host_storage.storage_checks(plan, check=_check),
        _qualification(plan),
        *_runtime(plan),
        _resident_service_prerequisite(plan),
        _provider_marker(plan, runner),
    ]
    with tempfile.TemporaryDirectory(prefix="ora-preflight-") as directory:
        root = Path(directory)
        probe = root / "scratch/probe.txt"
        probe.parent.mkdir(parents=True)
        probe.write_text("ok", encoding="utf-8")
        read_only = root / "inputs/read-only.txt"
        read_only.parent.mkdir(parents=True)
        read_only.write_text("visible", encoding="utf-8")
        read_only.chmod(0o444)
        checks.append(_check(
            "runtime_filesystem_contract",
            probe.read_text() == "ok" and read_only.read_text() == "visible",
            {
                "scratch_writable": True,
                "inputs_readable": True,
                "os_read_namespace_isolation_claimed": False,
            },
        ))
    return {
        "schema": "ora.preflight.v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "created_unix_s": time.time(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
