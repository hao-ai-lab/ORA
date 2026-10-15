"""``ora setup --env``: the verified single-GPU installation as one idempotent command.

Each environment expands to ordered components and steps (``ora_gpu_recipes``)
that run exactly the command sequence of
``docs/getting-started/single-gpu.md`` §9. A step is skipped when its
real state already holds (a pinned clean checkout) or when its marker under
``<prefix>/manifests/ora-setup/`` records the same recipe (the pinned
packages, the CUDA architecture, the requirement-file digest). Setup writes
only the ``.env`` keys it owns and never a credential; it starts no simulator,
service or Provider call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = ("libero", "robotwin", "robodojo", "perception")
MARKER_SCHEMA = "ora.ora-setup-step.v1"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"


class GpuSetupError(RuntimeError):
    """A setup step could not complete; the message names the step and its log."""


@dataclass(frozen=True)
class Step:
    """One idempotent setup action.

    ``recipe`` (a mapping, or a callable evaluated just before the check) is
    what the step installs; with ``marker`` the step is done only when its
    marker records the same recipe. ``done`` adds a real-state check and is
    the whole check when ``marker`` is false. A step with neither always runs.
    """

    component: str
    name: str
    title: str
    run: Callable[["Context"], None]
    recipe: Mapping[str, Any] | Callable[["Context"], Mapping[str, Any]] = field(default_factory=dict)
    done: Callable[["Context"], bool] | None = None
    marker: bool = True

    @property
    def key(self) -> str:
        return f"{self.component}.{self.name}"


@dataclass
class Context:
    """Paths, toolchain and the command runner shared by every step."""

    prefix: Path
    project_root: Path = PROJECT_ROOT
    python: str = sys.executable
    cuda_home: Path | None = None
    cuda_arch: str | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    runner: Callable[..., None] | None = None
    current_step: str = "setup"

    def path(self, relative: str) -> Path:
        return self.prefix / relative

    @property
    def log_root(self) -> Path:
        return self.prefix / "logs" / "ora-setup"

    def run(self, argv: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None,
            timeout: float = 7200, log_name: str | None = None) -> None:
        """Run one command with its output appended to the step log; raise with the log tail."""
        argv = [str(item) for item in argv]
        if self.runner is not None:
            self.runner(argv, cwd=cwd, env=env, timeout=timeout)
            return
        merged = {**self.environ, "GIT_TERMINAL_PROMPT": "0", **dict(env or {})}
        self.log_root.mkdir(parents=True, exist_ok=True)
        log = self.log_root / f"{log_name or self.current_step}.log"
        with log.open("ab") as stream:
            stream.write(f"\n$ {' '.join(argv)}\n".encode())
            stream.flush()
            try:
                # Own process group: a timeout also stops git-lfs, pip or compiler children.
                process = subprocess.Popen(argv, cwd=cwd, env=merged, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except OSError as exc:
                raise GpuSetupError(f"{Path(argv[0]).name} failed: {exc} (log: {log})") from exc
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise GpuSetupError(f"{Path(argv[0]).name} timed out after {timeout:.0f} s (log: {log})") from exc
        if code != 0:
            tail = log.read_text(errors="replace").splitlines()[-25:]
            raise GpuSetupError(f"{Path(argv[0]).name} exited {code} (log: {log})\n" + "\n".join(tail))

    def probe(self, argv: Sequence[str], *, timeout: float = 60) -> str | None:
        """A read-only command's standard output, or None when it fails or times out."""
        try:
            result = subprocess.run([str(item) for item in argv], capture_output=True, text=True,
                                    timeout=timeout, check=False,
                                    env={**self.environ, "GIT_OPTIONAL_LOCKS": "0"})
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def output(self, argv: Sequence[str], *, timeout: float = 60) -> str:
        """``probe`` with failure as an empty string, for identity reads."""
        return self.probe(argv, timeout=timeout) or ""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _marker_path(context: Context, step: Step) -> Path:
    return context.path(f"manifests/ora-setup/{step.key}.json")


def _recipe(context: Context, step: Step) -> Mapping[str, Any]:
    return step.recipe(context) if callable(step.recipe) else step.recipe


def step_done(context: Context, step: Step) -> bool:
    if step.marker:
        try:
            saved = json.loads(_marker_path(context, step).read_text())
        except (OSError, ValueError):
            return False
        if saved.get("recipe_sha256") != digest(_recipe(context, step)):
            return False
    elif step.done is None:
        return False
    return step.done(context) if step.done is not None else True


def _write_marker(context: Context, step: Step) -> None:
    path = _marker_path(context, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    recipe = _recipe(context, step)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"schema": MARKER_SCHEMA, "step": step.key, "recipe": recipe,
                                     "recipe_sha256": digest(recipe), "completed_unix_s": time.time()},
                                    indent=2, default=str) + "\n")
    temporary.replace(path)


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def execute(context: Context, steps: Sequence[Step], *, dry_run: bool = False,
            stream=None) -> list[dict[str, Any]]:
    """Run each pending step in order; a failure stops setup with the step's log."""
    output = stream or sys.stdout
    report = []
    for step in steps:
        if step_done(context, step):
            print(f"[ora setup] {step.component}: {step.title}: already done", file=output, flush=True)
            report.append({"step": step.key, "status": "done"})
            continue
        if dry_run:
            print(f"[ora setup] {step.component}: {step.title}: pending", file=output, flush=True)
            report.append({"step": step.key, "status": "pending"})
            continue
        print(f"[ora setup] {step.component}: {step.title} ...", file=output, flush=True)
        context.current_step = step.key
        started = time.monotonic()
        step.run(context)
        if step.marker:
            _write_marker(context, step)
        elapsed = time.monotonic() - started
        print(f"[ora setup] {step.component}: {step.title}: done ({_duration(elapsed)})", file=output, flush=True)
        report.append({"step": step.key, "status": "ran", "seconds": round(elapsed, 1)})
    return report


def default_prefix(environ: Mapping[str, str], env_values: Mapping[str, str]) -> Path:
    explicit = environ.get("ORA_GPU_PREFIX") or env_values.get("ORA_GPU_PREFIX")
    if explicit:
        return Path(explicit).expanduser().absolute()
    base = Path(environ.get("XDG_DATA_HOME") or (Path.home() / ".local/share"))
    return base.expanduser().absolute() / "ora" / "gpu"


def _check_prefix(prefix: Path, project_root: Path) -> None:
    resolved = prefix.resolve()
    if resolved == project_root.resolve() or resolved.is_relative_to(project_root.resolve()):
        raise GpuSetupError(f"--prefix {prefix} is inside the Harness checkout; launches require a clean "
                            "tree, so choose a directory outside it")


def _free_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024 ** 3


def main(argv: list[str] | None = None, *, stream=None) -> int:
    from . import ora_env_file, ora_gpu_recipes as recipes

    output = stream or sys.stdout
    parser = argparse.ArgumentParser(
        prog="ora setup --env",
        description="Install the pinned single-GPU simulator, CuRobo and perception stacks (CUDA 12.8 "
                    "everywhere) and write their .env keys. Idempotent; no simulator, service or Provider starts.")
    parser.add_argument("--env", action="append", choices=ENVIRONMENTS, required=True, dest="environments",
                        help="libero, robotwin and robodojo include the perception services (repeatable)")
    parser.add_argument("--prefix", type=Path, help="persistent install directory (default: ORA_GPU_PREFIX "
                        "from the environment or .env, else ~/.local/share/ora/gpu)")
    parser.add_argument("--cuda-home", type=Path, help="CUDA 12.8 toolkit for the CuRobo build "
                        "(default: CUDA_HOME, /usr/local/cuda-12.8, /usr/local/cuda)")
    parser.add_argument("--cuda-arch", help="TORCH_CUDA_ARCH_LIST base for CuRobo, e.g. 12.0 "
                        "(default: nvidia-smi --query-gpu=compute_cap)")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env",
                        help="the .env to update (default: the checkout's .env)")
    parser.add_argument("--dry-run", action="store_true", help="list each step as done or pending; change nothing")
    parser.add_argument("--json", action="store_true", help="print the step report as JSON")
    args = parser.parse_args(argv)
    env_values = ora_env_file.read(args.env_file)
    prefix = (args.prefix.expanduser().absolute() if args.prefix
              else default_prefix(os.environ, env_values))
    try:
        _check_prefix(prefix, PROJECT_ROOT)
        context = Context(prefix=prefix, cuda_home=args.cuda_home, cuda_arch=args.cuda_arch)
        recipes.add_library_path(context.environ, prefix)
        components = recipes.components_for(args.environments)
        need = recipes.disk_gib(components)
        free = _free_gib(prefix)
        print(f"[ora setup] {', '.join(components)} into {prefix} (needs up to ~{need} GB; {free:.0f} GB free)",
              file=output, flush=True)
        if not args.dry_run:
            recipes.check_toolchain(context, components)
            if free < need:
                print(f"[ora setup] warning: {free:.0f} GB free is less than the ~{need} GB these "
                      "components can use; completed steps are kept if a download fails", file=output, flush=True)
        steps = recipes.steps_for(context, components)
        report = execute(context, steps, dry_run=args.dry_run, stream=output)
        keys = recipes.env_keys(context, components)
        if args.dry_run:
            changed = {}
        else:
            changed = ora_env_file.update(args.env_file, {**keys, "ORA_GPU_PREFIX": str(prefix)},
                                          defaults={"DEEPSEEK_BASE_URL": DEEPSEEK_ENDPOINT})
    except (GpuSetupError, OSError, ValueError) as exc:
        print(f"[ora setup] incomplete: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"prefix": str(prefix), "components": components, "steps": report,
                          "env_file": str(args.env_file), "env_keys_written": sorted(changed)}, indent=2),
              file=output)
    elif not args.dry_run:
        print(f"[ora setup] {args.env_file}: {len(changed)} key(s) updated ({', '.join(sorted(changed)) or 'none'})",
              file=output)
        runnable = [name for name in args.environments if name != "perception"]
        if runnable:
            print("Next: export DEEPSEEK_LOW_API_KEY=<your key>, then "
                  f"./scripts/ora run --env {runnable[0]} --task <task> --confirm-launch", file=output)
    return 0


__all__ = ["Context", "ENVIRONMENTS", "GpuSetupError", "Step", "default_prefix", "execute", "main", "step_done"]
