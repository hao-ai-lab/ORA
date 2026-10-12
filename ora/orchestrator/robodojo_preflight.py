"""Host-safe RoboDojo source, interpreter and qualification preflight."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping
from .paths import resolve_workspace_root
from .spec import file_sha256
Runner = Callable[..., subprocess.CompletedProcess[str]]

UNPROFILED_MINIMUM_FREE_MEMORY_MIB = 20000
#: RDJ-10: nvidia-smi answered in 6-13 s on a loaded simulator host; 15 s refused healthy starts.
NVIDIA_SMI_TIMEOUT_S = 60


def _robodojo_runtime(plan: Mapping[str, Any], runner: Runner, *, project_root: Path, check: Any, git: Any, workspace_resolver: Any) -> list[dict[str, Any]]:
    PROJECT_ROOT, _check, _git = project_root, check, git
    resolve_workspace_root = workspace_resolver
    from ora.environment.benchmarks.robodojo import (
        RoboDojoBenchmarkError, REVISION_FIELDS, selected_layout,
    )

    from .robodojo_qualification import (
        GRAPHICS_READBACK_TIMEOUT_S, LAYERED_SCHEMA, inspect_graphics_compatibility,
        load_qualification,
        validate_runtime_driver,
    )

    runtime = plan["runtime"]
    checks = [_check(
        "execution_host",
        bool(runtime.get("required_host"))
        and os.environ.get("ORA_EXECUTION_HOST") == runtime["required_host"],
        {"expected": runtime.get("required_host"), "observed": os.environ.get("ORA_EXECUTION_HOST")},
    )]
    if runtime.get("physics_device") == "cpu":
        # The RunSpec rule, re-checked on the exact plan: a CPU-physics session
        # never restarts a round in place (robodojo_physics_attestation).
        from .robodojo_physics_attestation import cpu_round_restart_refusal
        rounds = int((plan.get("session") or {}).get("rounds", 1))
        refusal = cpu_round_restart_refusal(runtime, rounds=rounds)
        checks.append(_check("robodojo_cpu_round_restart", refusal is None, refusal or {
            "physics_device": "cpu", "rounds": rounds,
            "layout_schedule": runtime.get("layout_schedule") is not None}))
    roots: dict[str, Path] = {}
    for field in ("root_env", "asset_root_env", "python_env"):
        raw = os.environ.get(runtime[field])
        path = Path(os.path.abspath(os.path.expanduser(raw))) if raw else None
        ready = bool(path and (path.is_file() if field == "python_env" else path.is_dir()))
        checks.append(_check(f"robodojo_{field}", ready, {"environment_variable": runtime[field], "path": str(path) if path else None}))
        if ready and path:
            roots[field] = path
    if len(roots) != 3:
        return checks
    root, assets, python = (roots[key] for key in ("root_env", "asset_root_env", "python_env"))
    from ora.agent.composition.robodojo_task_contract import verify_contract_dependencies
    try:
        checked_contract = verify_contract_dependencies(
            plan.get("policy", {}).get("task_contract") or {},
            project_root=PROJECT_ROOT, source_root=root, asset_root=assets,
        )
        checks.append(_check("robodojo_task_contract_dependencies", True, checked_contract))
    except (OSError, ValueError, RuntimeError) as exc:
        checks.append(_check("robodojo_task_contract_dependencies", False, str(exc)))
    source_roots = {
        "robodojo": root, "isaaclab": root / "third_party/IsaacLab",
        "curobo": root / "third_party/curobo", "xpolicylab": root / "XPolicyLab",
        "assets": assets,
    }
    for name, source_root in source_roots.items():
        revision = _git(source_root, "rev-parse", "HEAD")
        dirty = _git(source_root, "status", "--porcelain", "--untracked-files=no")
        checks.append(_check(f"robodojo_{name}_source", (
            revision.returncode == 0 and revision.stdout.strip() == runtime[REVISION_FIELDS[name]]
            and dirty.returncode == 0 and not dirty.stdout.strip()
        ), {"expected_revision": runtime[REVISION_FIELDS[name]], "observed_revision": revision.stdout.strip(), "clean": dirty.returncode == 0 and not dirty.stdout.strip()}))
    try:
        layout = selected_layout(assets, task=plan["task"], eval_seed=plan["seed"], layout_id=runtime["layout_id"])
        checks.append(_check("robodojo_layout", (
            layout.relative_to(assets.resolve()).as_posix() == runtime["layout_file"]
            and file_sha256(layout) == runtime["layout_sha256"]
        ), {"path": str(layout), "sha256": file_sha256(layout)}))
    except (OSError, ValueError) as exc:
        checks.append(_check("robodojo_layout", False, str(exc)))
    schedule = runtime.get("layout_schedule")
    if schedule is not None:
        # Every scheduled round's file must still be its planned ordinal and bytes.
        try:
            observed = []
            for entry in schedule["rounds"]:
                path = selected_layout(assets, task=plan["task"], eval_seed=plan["seed"], layout_id=entry["layout_id"])
                observed.append({"layout_id": entry["layout_id"], "path": path.relative_to(assets.resolve()).as_posix(),
                                 "sha256": file_sha256(path)})
            checks.append(_check("robodojo_layout_schedule", all(
                item["path"] == entry["layout_file"] and item["sha256"] == entry["layout_sha256"]
                for item, entry in zip(observed, schedule["rounds"], strict=True)), {"layouts": observed}))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks.append(_check("robodojo_layout_schedule", False, str(exc)))
    observed: dict[str, Any] | None = None
    try:
        process = runner(
            [str(python), "-c", (
                "import json,sys,hashlib; from pathlib import Path; from importlib.metadata import version; "
                "print(json.dumps({'python_executable':sys.executable, "
                "'python_binary_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),"
                "'versions':{k:version(k) for k in ('isaacsim','torch','numpy','warp-lang')}}))"
            )], cwd=root, env=os.environ.copy(), text=True, capture_output=True,
            timeout=GRAPHICS_READBACK_TIMEOUT_S, check=False,
        )
        observed = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else None
        versions = observed.get("versions", {}) if isinstance(observed, dict) else {}
        checks.append(_check("robodojo_python_stack", (
            versions.get("isaacsim") == "5.1.0.0" and versions.get("torch") == "2.7.0+cu128"
            and versions.get("numpy") == "1.26.0"
        ), observed or {"returncode": process.returncode}))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        checks.append(_check("robodojo_python_stack", False, str(exc)))
    qualified_device: Mapping[str, Any] | None = None
    qualified_graphics: Mapping[str, Any] | None = None
    try:
        # One entry point for both contracts: the combined single record, or
        # the card/task layered pair, which fails closed when either layer is
        # missing, drifted or paired with the wrong other layer.
        qualification = load_qualification(
            runtime, workspace_root=resolve_workspace_root(PROJECT_ROOT),
            harness_commit=plan["source"]["commit"],
        )
        if qualification.get("environment") != observed:
            raise RoboDojoBenchmarkError("qualified worker interpreter/package readback drift")
        if qualification.get("task_instruction") != plan["task_instruction"]:
            raise RoboDojoBenchmarkError("official task language drift")
        qualified_device = qualification["device_binding"]
        qualified_graphics = qualification.get("graphics_compatibility")
        checks.append(_check("robodojo_infrastructure_qualification", True, qualification))
        if qualification.get("schema") == LAYERED_SCHEMA:
            # Report the layering itself, and name in the operator's own
            # report the fact the task layer does not establish and the
            # evidence that does. It is deliberately a passing check with an
            # explicit disclosure, not a silent inheritance.
            checks.append(_check("robodojo_qualification_layers", True, {
                "layers": qualification["layers"],
                "card_record": runtime["infrastructure_qualification"]["card"],
                "task_record": runtime["infrastructure_qualification"]["task"],
                "card_measured_task": qualification["card_measured_task"],
                "not_established_by_the_task_layer": qualification["not_established"],
            }))
    except (OSError, ValueError, KeyError) as exc:
        checks.append(_check("robodojo_infrastructure_qualification", False, str(exc)))
    try:
        query = ["nvidia-smi", "--query-gpu=index,uuid,memory.free,driver_version", "--format=csv,noheader,nounits"]
        try:
            gpu = runner(query, text=True, capture_output=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            # RDJ-10: a loaded node answered in 6-13 s; retry once before refusing.
            gpu = runner(query, text=True, capture_output=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
        rows = [row.strip().split(", ") for row in gpu.stdout.splitlines()]
        selected = [row for row in rows if row[0] == runtime["cuda_visible_devices"]]
        # A profiled slot declares its own floor (dense slots: 10 GiB); an
        # unprofiled plan keeps the host-exclusive default.
        floor = int((runtime.get("resource_binding") or {}).get(
            "minimum_free_memory_mib", UNPROFILED_MINIMUM_FREE_MEMORY_MIB))
        checks.append(_check("robodojo_gpu", (
            gpu.returncode == 0 and len(selected) == 1 and int(selected[0][2]) >= floor
            and qualified_device is not None and selected[0][1] == qualified_device.get("selected_gpu_uuid")
        ), {"selected_device": selected, "minimum_free_memory_mib": floor,
            "qualification_comparison": "available" if qualified_device is not None else "blocked_by_infrastructure_qualification"}))
        try:
            driver = selected[0][3] if gpu.returncode == 0 and len(selected) == 1 else ""
            graphics = inspect_graphics_compatibility(python, runner=runner) if driver.startswith("595.") else None
            validate_runtime_driver(driver, graphics_compatibility=graphics)
            if qualified_device is None:
                raise RoboDojoBenchmarkError("driver qualification comparison blocked by infrastructure qualification; no drift conclusion")
            if graphics != qualified_graphics:
                raise RoboDojoBenchmarkError("qualified Vulkan compatibility fingerprint drift")
            checks.append(_check("robodojo_driver", True, {
                "version": driver, "graphics_compatibility": graphics, "live_qualification_still_required": True}))
        except (ValueError, IndexError) as exc:
            checks.append(_check("robodojo_driver", False, str(exc)))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        checks.append(_check("robodojo_gpu", False, str(exc)))
    return checks
