"""LIBERO driver preflight: roots, probe and controller contract without MuJoCo.

Split from ``preflight.py``, which forwards to :func:`libero_pro_runtime` and
supplies the two dependencies tests patch on that module (``_git`` and
``PROJECT_ROOT``) at call time.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment import graphics_probe
from ora.environment.controllers.libero_osc import (
    libero_osc_action_contract,
)

from .checks import check as _check, execution_host_check
from .spec import file_sha256

def _libero_controller_contract_check(
    observed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the exact controller mapping reported by the probe."""

    contract = observed.get("controller_action_contract") if observed else None
    binding = observed.get("controller_binding") if observed else None
    try:
        if not isinstance(contract, Mapping) or not isinstance(binding, Mapping):
            raise ValueError("controller contract or binding is absent")
        recomputed = libero_osc_action_contract({
            "control_delta": contract.get("control_delta"),
            "input_min": contract.get("input_min"),
            "input_max": contract.get("input_max"),
            "output_min": contract.get("output_min"),
            "output_max": contract.get("output_max"),
        })
        if dict(contract) != recomputed:
            raise ValueError("controller contract hash or derived fields drifted")
        output_min = [float(value) for value in contract["output_min"]]
        output_max = [float(value) for value in contract["output_max"]]
        servo = contract.get("servo_clip")
        if not isinstance(servo, Mapping):
            raise ValueError("controller servo clip is absent")
        physical_capacity = [
            min(abs(output_min[index]), abs(output_max[index]))
            for index in range(6)
        ]
        if float(servo.get("translation_m", float("inf"))) > min(physical_capacity[:3]):
            raise ValueError("translation servo clip exceeds controller output range")
        if float(servo.get("rotation_rad", float("inf"))) > min(physical_capacity[3:]):
            raise ValueError("rotation servo clip exceeds controller output range")
        for field in (
            "robosuite_module", "robosuite_module_sha256",
            "controller_config_file", "controller_config_sha256",
            "panda_gripper_module", "panda_gripper_module_sha256",
            "panda_gripper_file", "panda_gripper_sha256",
        ):
            if not isinstance(binding.get(field), str) or not binding[field]:
                raise ValueError(f"controller binding lacks {field}")
        robosuite_module = Path(binding["robosuite_module"]).resolve()
        controller_config = Path(binding["controller_config_file"]).resolve()
        panda_gripper_module = Path(binding["panda_gripper_module"]).resolve()
        panda_gripper = Path(binding["panda_gripper_file"]).resolve()
        robosuite_root = robosuite_module.parent
        for label, path, digest_field in (
            ("robosuite module", robosuite_module, "robosuite_module_sha256"),
            ("controller config", controller_config, "controller_config_sha256"),
            (
                "Panda gripper module", panda_gripper_module,
                "panda_gripper_module_sha256",
            ),
            ("Panda gripper", panda_gripper, "panda_gripper_sha256"),
        ):
            if not path.is_file():
                raise ValueError(f"{label} binding is unavailable")
            if file_sha256(path) != binding[digest_field]:
                raise ValueError(f"{label} binding hash drifted")
        if not controller_config.is_relative_to(robosuite_root):
            raise ValueError("controller config escapes the imported robosuite package")
        if not panda_gripper_module.is_relative_to(robosuite_root):
            raise ValueError("Panda gripper module escapes the imported robosuite package")
        if not panda_gripper.is_relative_to(robosuite_root):
            raise ValueError("Panda gripper escapes the imported robosuite package")
        gripper_probe = observed.get("gripper_action_probe") if observed else None
        if not isinstance(gripper_probe, Mapping):
            raise ValueError("Panda gripper action probe is absent")
        close = [float(value) for value in gripper_probe.get("close", [])]
        hold = [float(value) for value in gripper_probe.get("zero_hold", [])]
        reopened = [float(value) for value in gripper_probe.get("reopen", [])]
        if (
            len(close) != 2
            or len(hold) != 2
            or len(reopened) != 2
            or any(abs(close[index] - hold[index]) > 1e-9 for index in range(2))
            or not close[0] < 0 < close[1]
            or any(abs(value) > 1e-9 for value in reopened)
        ):
            raise ValueError("Panda gripper sign-integrator behavior drifted")
        try:
            bound_controller_config = json.loads(
                controller_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("bound controller config is not readable JSON") from exc
        if not isinstance(bound_controller_config, Mapping):
            raise ValueError("bound controller config is not an object")
        file_contract = libero_osc_action_contract(bound_controller_config)
        if file_contract != dict(contract):
            raise ValueError(
                "reported controller contract does not match the bound config file"
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _check("libero_controller_action_contract", False, str(exc))
    return _check("libero_controller_action_contract", True, {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": contract["contract_sha256"],
        "input_min": contract["input_min"],
        "input_max": contract["input_max"],
        "output_min": contract["output_min"],
        "output_max": contract["output_max"],
        "servo_clip": contract["servo_clip"],
        "gripper_action_probe": dict(gripper_probe),
        "binding": dict(binding),
    })

def _libero_probe_matches_plan(
    observed: Any, runtime: Mapping[str, Any], *, motion_enabled: bool,
) -> bool:
    """Validate the metadata/init/context identity returned by the worker probe."""

    acknowledged = runtime.get("task_language_divergence_ack")
    bddl_language = (
        observed.get("bddl_task_language")
        if isinstance(observed, Mapping) else None
    )
    # The registry language comes from the BDDL file name; the evaluated goal
    # lives in the file. When a perturbation suite rewrites the goal but keeps
    # the base file name, the policy is instructed to do one task and scored
    # on another. Refuse that unless the plan records the exact divergence.
    language_binding_valid = (
        bddl_language is None
        or observed.get("bddl_language_matches") is True
        or (
            isinstance(acknowledged, Mapping)
            and acknowledged.get("bddl_task_language") == bddl_language
            and acknowledged.get("policy_task_language")
            == str(runtime["expected_task_language"]).strip()
        )
    )
    return bool(
        isinstance(observed, Mapping)
        and (
            observed.get("language_matches") is True
            or observed.get("bddl_language_matches") is True
            and bddl_language == str(runtime["expected_task_language"]).strip()
        )
        and language_binding_valid
        and isinstance(observed.get("init_state_count"), int)
        and observed["init_state_count"] > 0
        and (
            runtime.get("robustness_case") is None
            or isinstance(observed.get("robustness"), Mapping)
            and observed["robustness"].get("case")
            == runtime["robustness_case"]
        )
        and (
            runtime.get("public_environment_context") is None
            or observed.get("public_environment_context")
            == runtime["public_environment_context"]
            or motion_enabled
            and isinstance(observed.get("public_environment_context"), Mapping)
            and {
                **dict(observed["public_environment_context"]),
                "controller": "CUROBO_JOINT_POSITION+OSC_POSE",
            } == runtime["public_environment_context"]
        )
    )

def _offscreen_graphics_runtime(
    python: Path, environment: Mapping[str, str],
) -> dict[str, Any]:
    """Name a broken offscreen GL binding before the LIBERO import probe runs.

    MuJoCo and PyOpenGL resolve EGL from forwarded process environment that no
    other check covers, and the LIBERO import chain swallows the resulting
    loader error behind a generic ``ImportError``.  Enumerating devices in the simulator interpreter
    keeps that failure attributable to the graphics stack.
    """

    script = Path(graphics_probe.__file__).resolve()
    try:
        probe = subprocess.run(
            [str(python), str(script)], env=dict(environment),
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _check("offscreen_graphics_runtime", False, str(exc))
    observed: Any = None
    if probe.returncode == 0:
        try:
            observed = json.loads(probe.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            observed = None
    if (
        not isinstance(observed, Mapping)
        or observed.get("schema") != graphics_probe.GRAPHICS_PROBE_SCHEMA
    ):
        return _check("offscreen_graphics_runtime", False, {
            "returncode": probe.returncode,
            "stderr_tail": probe.stderr.strip()[-2000:],
            "stdout_tail": probe.stdout.strip()[-1000:],
        })
    return _check(
        "offscreen_graphics_runtime", bool(observed.get("usable")), observed,
    )

def libero_pro_runtime(
    plan: Mapping[str, Any], *, git: Callable[..., Any], project_root: Path,
) -> list[dict[str, Any]]:
    """Validate the selected LIBERO driver without starting MuJoCo."""
    # preflight passes these at call time, so a test that patches
    # preflight._git or preflight.PROJECT_ROOT still reaches this check.
    _git, PROJECT_ROOT = git, project_root
    runtime = plan["runtime"]
    # Sparse motion v3 on LIBERO runs the hybrid CuRobo/OSC bindings, so it
    # takes the motion preflight (planner probe included).
    motion_enabled = (
        plan.get("policy", {}).get("action_codec") == "motion_action_v3"
        and plan.get("runtime", {}).get("kind") == "libero_pro"
    )
    checks: list[dict[str, Any]] = []
    workspace_profile = runtime.get("workspace_profile")
    checks.append(_check(
        "libero_action_workspace_profile",
        isinstance(workspace_profile, Mapping)
        and workspace_profile.get("schema")
        == "ora.action-workspace-profile.v1"
        and workspace_profile.get("semantics")
        == "hard_action_envelope_not_reachability"
        and isinstance(workspace_profile.get("profile_sha256"), str)
        and workspace_profile.get("bounds_m") == runtime.get("workspace_bounds_m"),
        workspace_profile or "source-owned workspace profile is required",
    ))
    checks.append(execution_host_check(runtime, os.environ.get("ORA_EXECUTION_HOST")))
    roots: dict[str, Path | None] = {}
    for name, env_field, revision_field in (
        ("libero_pro", "libero_pro_root_env", "required_libero_pro_revision"),
        ("libero_pro_data", "data_root_env", "required_data_revision"),
    ):
        env_name = runtime[env_field]
        raw = os.environ.get(env_name)
        root = Path(raw).expanduser().resolve() if raw else None
        roots[name] = root
        valid = bool(root and (root / ".git").exists())
        checks.append(_check(f"{name}_root", valid, {
            "env": env_name, "configured": bool(raw), "root": str(root) if root else None,
        }))
        if valid and root is not None:
            revision = _git(root, "rev-parse", "HEAD")
            dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
            checks.append(_check(
                f"{name}_revision",
                revision.returncode == 0
                and revision.stdout.strip() == runtime[revision_field],
                {"expected": runtime[revision_field], "observed": revision.stdout.strip()},
            ))
            checks.append(_check(
                f"{name}_clean", dirty.returncode == 0 and not dirty.stdout.strip(),
                dirty.stdout.strip() or "clean",
            ))
    python_env = runtime["python_env"]
    raw_python = os.environ.get(python_env)
    python = Path(os.path.abspath(os.path.expanduser(raw_python))) if raw_python else None
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {
        "env": python_env, "configured": bool(raw_python),
        "executable": str(python) if python else None,
    }))
    libero_pro_root = roots.get("libero_pro")
    data_root = roots.get("libero_pro_data")
    # Every probe runs under the lane's own worker environment, as its worker
    # will: a Mesa lane's EGL vendor and a lane's CUDA device are per lane, and
    # the queue process's inherited CUDA_VISIBLE_DEVICES may name a dead GPU.
    binding = runtime.get("resource_binding")
    lane_environment = {
        str(name): str(value)
        for name, value in (
            binding.get("worker_environment", {}) if isinstance(binding, Mapping) else {}
        ).items()
    }
    graphics = None
    if executable and python is not None:
        graphics = _offscreen_graphics_runtime(python, {**os.environ, **lane_environment})
        checks.append(graphics)
    if executable and libero_pro_root and data_root and (
        graphics is not None and graphics["passed"]
    ):
        probe_environment = {**os.environ, **lane_environment}
        probe_environment["LIBERO_TYPE"] = "standard"
        python_paths = [str(PROJECT_ROOT)]
        if probe_environment.get("PYTHONPATH"):
            python_paths.append(probe_environment["PYTHONPATH"])
        probe_environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        try:
            with tempfile.TemporaryDirectory(
                prefix="ora-libero-probe-",
            ) as runtime_config_root:
                command = [
                    str(python), "-m",
                    "ora.environment.backends.libero_pro.probe",
                    "--environment-driver", "direct_libero_v1",
                    "--libero-pro-root", str(libero_pro_root),
                    "--data-root", str(data_root),
                    "--runtime-config-root", runtime_config_root,
                    "--suite", str(runtime["suite"]),
                    "--task-id", str(runtime["task_id"]),
                    "--expected-task-language",
                    str(runtime["expected_task_language"]),
                    "--include-init-states",
                    "--include-controller",
                ]
                if runtime.get("robustness_case") is not None:
                    command.extend((
                        "--robustness-case", str(runtime["robustness_case"]),
                    ))
                probe = subprocess.run(
                    command, cwd=libero_pro_root, env=probe_environment,
                    capture_output=True, text=True, timeout=180, check=False,
                )
            observed = None
            if probe.returncode == 0:
                try:
                    observed = json.loads(probe.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    pass
            checks.append(_check(
                "libero_pro_runtime_imports",
                probe.returncode == 0
                and _libero_probe_matches_plan(
                    observed, runtime, motion_enabled=motion_enabled,
                ),
                observed or {
                    "returncode": probe.returncode,
                    "stderr_tail": probe.stderr.strip()[-2000:],
                    "stdout_tail": probe.stdout.strip()[-1000:],
                },
            ))
            checks.append(_libero_controller_contract_check(
                observed if isinstance(observed, Mapping) else None,
            ))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(_check("libero_pro_runtime_imports", False, str(exc)))
            checks.append(_check(
                "libero_controller_action_contract", False, str(exc),
            ))
    elif graphics is not None and not graphics["passed"]:
        # Running the import probe under a broken graphics stack only produces
        # a generic ImportError, which reads as a broken LIBERO install.
        skipped = (
            "not evaluated: offscreen_graphics_runtime failed, so a LIBERO "
            "import failure here would misattribute the graphics fault"
        )
        checks.append(_check("libero_pro_runtime_imports", False, skipped))
        checks.append(_check("libero_controller_action_contract", False, skipped))
    if motion_enabled:
        planner = runtime.get("motion_planner")
        try:
            if not isinstance(planner, Mapping):
                raise RuntimeError("runtime.motion_planner is required")
            python_name = str(planner["python_env"])
            root_name = str(planner["root_env"])
            python_raw = os.environ.get(python_name)
            root_raw = os.environ.get(root_name)
            if not python_raw or not root_raw:
                raise RuntimeError("CuRobo Python/root environment is missing")
            from ora.environment.controllers.curobo_motion import (
                CuroboPlannerClient,
            )

            with tempfile.TemporaryDirectory(
                prefix="ora-curobo-preflight-",
            ) as directory:
                client = CuroboPlannerClient(
                    python_executable=Path(os.path.abspath(os.path.expanduser(python_raw))),
                    curobo_root=Path(root_raw),
                    required_revision=str(planner["required_revision"]),
                    output_root=Path(directory),
                    device=str(planner.get("device", "cuda:0")),
                    request_timeout_s=float(planner.get("request_timeout_s", 180.0)),
                    environment_overrides=lane_environment,
                )
                try:
                    planner_binding = client.start()
                finally:
                    # Readiness can fail after the isolated worker is spawned.
                    # The preflight owns that process even when no binding is returned.
                    client.close()
            checks.append(_check("curobo_motion_planner", True, planner_binding))
        except Exception as exc:
            checks.append(_check("curobo_motion_planner", False, str(exc)))
    if runtime.get("require_gpu", True):
        nvidia = shutil.which("nvidia-smi")
        result = subprocess.run(
            [nvidia, "-L"], capture_output=True, text=True, timeout=30, check=False,
        ) if nvidia else None
        checks.append(_check(
            "gpu", bool(result and result.returncode == 0 and result.stdout.strip()),
            result.stdout.strip() if result else "nvidia-smi unavailable",
        ))
    return checks
