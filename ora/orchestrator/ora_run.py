"""``ora run``: one GPU task on the shipped ``single-gpu`` host, from installed inputs to ORA.

The command does what ``docs/getting-started/single-gpu.md`` §9.6 does by
hand, in order, and stops at the first failure with the reason:

1. load the checkout's ``.env`` (environment values win; empty values are
   ignored, so a template line cannot blank a key), bind the ``single-gpu``
   host marker, default ``DEEPSEEK_BASE_URL`` to the profile endpoint and run
   preflight's offline ``provider_marker`` configuration check;
2. reuse or run the task's one-time qualification (``ora_run_qualification``);
3. make the SAM3 / Contact-GraspNet services ready and bind a readiness
   report for the running instances (``ora_run_services``), after step 2 so a
   long qualification cannot age the report out;
4. compose the RunSpec under ``outputs/ora-interactive/specs/`` with a fresh
   single-attempt cohort, then open ``ora interactive`` on it.

Launch authority is unchanged: ``ora interactive`` repeats the full preflight
and launches only with ``--confirm-launch`` and the operator's selection.
Credentials are read from the environment only and never written anywhere.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

# Not ``paths.PROJECT_ROOT``: importing ``paths`` resolves the outputs root, and
# ``.env`` (which may set ORA_OUTPUTS_ROOT) is loaded first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST = "single-gpu"
HOST_MARKER_ENV = "ORA_EXECUTION_HOST"
EXPERIMENT = "ora-interactive"
SLOTS = {"libero": "libero-gpu0", "robotwin": "robotwin-gpu0", "robodojo": "robodojo-gpu0"}
SERVICE_KEYS = ("B2_SAM_PYTHON", "B2_SAM_ROOT", "B2_SAM_CHECKPOINT", "B2_SAM_GPU", "B2_GRASP_PYTHON",
                "B2_GRASP_ROOT", "B2_GRASP_CHECKPOINT", "B2_GRASP_CONFIG", "B2_GRASP_GPU")
REQUIRED_KEYS = {
    "libero": ("LIBERO_PRO_PYTHON", "LIBERO_PRO_ROOT", "LIBERO_PRO_DATA_ROOT", "CUROBO_ROOT", "CUROBO_PYTHON",
               *SERVICE_KEYS),
    "robotwin": ("ROBOTWIN_ROOT", "ROBOTWIN_PYTHON", "CUROBO_ROOT", "CUROBO_PYTHON", *SERVICE_KEYS),
    # Unset ROBODOJO_* silently fall back to ~/robodojo-runtime in the runtime configuration.
    "robodojo": ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT", "ROBODOJO_PYTHON", *SERVICE_KEYS),
}
EXAMPLES = {"libero": "libero_goal/0", "robotwin": "click_bell", "robodojo": "general_pickup"}


def build_parser() -> argparse.ArgumentParser:
    from .ora_world import add_world_options

    parser = argparse.ArgumentParser(
        prog="ora run", description="Prepare one GPU task on the single-gpu host and open ORA on it.",
        epilog="Examples: ora run --env libero --task libero_goal/0 --confirm-launch; "
               "ora run --env robotwin --task click_bell --confirm-launch; "
               "ora run --env robodojo --task general_pickup --confirm-launch")
    parser.add_argument("--env", required=True, choices=tuple(SLOTS), dest="environment")
    parser.add_argument("--task", required=True,
                        help="LIBERO: <suite>/<task id> (base or LIBERO-Pro core suite); RoboTwin and RoboDojo: "
                             "the task name")
    parser.add_argument("--seed", type=int, help="LIBERO initial state 0-49 (default 0); RoboTwin: an admitted "
                        "seed (default: the first seed the qualification accepted)")
    parser.add_argument("--task-config", default="demo_randomized", choices=("demo_randomized", "demo_clean"),
                        help="RoboTwin scene configuration")
    parser.add_argument("--candidates", type=int, default=4, help="RoboTwin seeds a new scan may try (default 4)")
    parser.add_argument("--requalify", action="store_true", help="run the qualification even if a record is reusable")
    parser.add_argument("--yes", action="store_true", help="confirm qualification simulator episodes without asking")
    parser.add_argument("--confirm-launch", action="store_true",
                        help="authorize launching the selected task after its normal preflight passes")
    parser.add_argument("--prepare-only", action="store_true",
                        help="stop after composing the RunSpec and print its path")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--provider-profile", help="Provider profile from the shared catalog (default: deepseek_flash_openai_low_16k_v1); e.g. muse_spark_open_openrouter_high_16k_v1")
    parser.add_argument("--emit", choices=("jsonl",), help="JSON-lines interactive backend instead of the Pi UI")
    parser.add_argument("--poll", type=float, default=0.2)
    add_world_options(parser)
    return parser


def prepare_environment(environment: str, *, env_file: Path,
                        environ: MutableMapping[str, str] | None = None,
                        provider_profile: str | None = None) -> dict[str, Any]:
    """Bind ``.env``, the host marker and Provider defaults into ``environ``; refuse missing inputs."""
    from . import ora_env_file

    values = os.environ if environ is None else environ
    loaded = ora_env_file.load_into(values, env_file)
    from .ora_local import provider_check, provider_environment  # imports paths: after .env

    marker = values.get(HOST_MARKER_ENV)
    if marker and marker != HOST:
        raise RuntimeError(f"ora run uses the shipped {HOST} host; {HOST_MARKER_ENV}={marker} names another "
                           "execution host (unset it or use that host's own workflow)")
    values[HOST_MARKER_ENV] = HOST
    if values.get("ORA_GPU_PREFIX"):
        # libGLU / the Vulkan loader ``ora setup`` unpacked without sudo; workers inherit it.
        from .ora_gpu_recipes import add_library_path
        add_library_path(values, Path(values["ORA_GPU_PREFIX"]))
    values.update(provider_environment(values, provider_profile))
    check = provider_check(values, provider_profile)
    if not check["passed"]:
        detail = check["detail"]
        raise RuntimeError("Provider configuration is incomplete: set "
                           + " and ".join(detail.get("required_environment", []))
                           + (f" (base URL must be {detail['expected_endpoint']})" if "expected_endpoint" in detail
                              else "") + "; ora run never stores credentials")
    missing = [key for key in REQUIRED_KEYS[environment] if not values.get(key)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} unset; run ./scripts/ora setup --env {environment} "
                           f"(it writes these keys to {env_file})")
    return {"env_file_keys": loaded}


def require_clean_checkout(root: Path = PROJECT_ROOT) -> None:
    """Qualification and launch both refuse a dirty tree; say so before minutes of work."""
    try:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=180, check=False,
                                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect the Harness checkout: {exc}") from exc
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("the Harness checkout has local changes or untracked files; qualification and launch "
                           "require a clean committed tree (commit or move them, then rerun)")


def libero_cell(task: str) -> tuple[str, str, int]:
    """``(builder, suite, task_id)`` for ``<suite>/<id>`` or a bare suite (task 0)."""
    from ora.environment.benchmarks.libero_pro import BASE_SUITES, CORE_SUITES

    suite, _, number = task.partition("/")
    try:
        task_id = int(number) if number else 0
    except ValueError as exc:
        raise ValueError(f"LIBERO task must look like libero_goal/0, not {task!r}") from exc
    if suite in BASE_SUITES:
        return "libero", suite, task_id
    if suite in CORE_SUITES:
        return "libero_pro", suite, task_id
    raise ValueError(f"unknown LIBERO suite {suite!r}; base suites {list(BASE_SUITES)}, "
                     f"LIBERO-Pro core suites {list(CORE_SUITES)}")


def cohort(environment: str) -> str:
    """A fresh single-attempt cohort per ``ora run``, so rerunning a cell is a new admission."""
    return f"ora-run-{environment}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def compose(args: argparse.Namespace, *, services: Callable[[], Path], confirm, stream) -> Path:
    """Run the needed qualification, then bind the service report and write the RunSpec.

    The report is chosen after the qualification, which can take most of an
    hour, so a reused report cannot age out between selection and composition.
    """
    from . import ora_run_qualification as qualification

    environment = args.environment
    common = {"experiment": EXPERIMENT, "cohort_id": cohort(environment),
              "provider_profile": getattr(args, "provider_profile", None)}
    if environment == "libero":
        from .benchmark_spec import build_libero_pro_spec, build_libero_spec, compose as write

        builder, suite, task_id = libero_cell(args.task)
        build = build_libero_spec if builder == "libero" else build_libero_pro_spec
        spec = build(suite, task_id, seed=0 if args.seed is None else args.seed,
                     resource_profile=SLOTS[environment], host=HOST, service_report=services(), **common)
        return write(spec)
    if environment == "robotwin":
        from .benchmark_spec import build_robotwin_spec, compose as write

        manifest, seed = qualification.robotwin_seed(
            args.task, task_config=args.task_config, seed=args.seed, candidates=args.candidates,
            requalify=args.requalify, confirm=confirm, environ=os.environ, stream=stream)
        spec = build_robotwin_spec(args.task, seed=seed, seed_manifest=manifest, task_config=args.task_config,
                                   resource_profile=SLOTS[environment], host=HOST, service_report=services(),
                                   **common)
        return write(spec, seed_manifest=manifest)
    from .robodojo_spec import build_spec, compose as write

    if args.seed not in (None, 0):
        raise ValueError("a RoboDojo cell runs official layout 0 of seed 0; --seed does not apply")
    binding = qualification.robodojo_qualification(args.task, requalify=args.requalify, confirm=confirm,
                                                   stream=stream)
    spec = build_spec(args.task, gpu=qualification.ROBODOJO_GPU, resource_profile=SLOTS[environment],
                      infrastructure_qualification=binding, service_report=services(), **common)
    return write(spec)


def interactive_arguments(args: argparse.Namespace, spec: Path, *, world_explicit: bool) -> list[str]:
    forwarded = ["interactive", "--spec", str(spec), "--poll", str(args.poll),
                 "--world-view", args.world_view, "--world-browser", args.world_browser]
    if world_explicit:
        forwarded += ["--world-port", str(args.world_port)]
    if args.world_public_url:
        forwarded += ["--world-public-url", args.world_public_url]
    if args.confirm_launch:
        forwarded.append("--confirm-launch")
    if args.emit:
        forwarded += ["--emit", args.emit]
    return forwarded


def main(argv: list[str] | None = None, *, stream=None) -> int:
    output = stream or sys.stderr
    values = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(values)
    world_explicit = any(value == "--world-port" or value.startswith("--world-port=") for value in values)
    try:
        prepare_environment(args.environment, env_file=args.env_file, provider_profile=args.provider_profile)
        from .ora_run_services import check_service_ports, confirm as ask, ensure_services

        require_clean_checkout()
        check_service_ports()  # before a qualification that can take most of an hour
        print(f"ora run: {args.environment} {args.task} on the {HOST} host", file=output, flush=True)
        spec = compose(args, services=lambda: ensure_services(stream=output),
                       confirm=lambda message: ask(message, assume_yes=args.yes, stream=output), stream=output)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"ora run: {exc}", file=sys.stderr)
        return 2
    print(f"RunSpec {spec}", file=output, flush=True)
    if args.prepare_only:
        print(spec)
        return 0
    from .ora_cli import main as ora_main

    return ora_main(interactive_arguments(args, spec, world_explicit=world_explicit))


__all__ = ["EXAMPLES", "HOST", "REQUIRED_KEYS", "build_parser", "compose", "libero_cell", "main",
           "prepare_environment", "require_clean_checkout"]
