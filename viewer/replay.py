"""Host-side orchestration for simulator-derived controller replay frames."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_SCHEMA = "ora.viewer-controller-replay-job.v2"
RESULT_SCHEMA = "ora.viewer-controller-replay.v2"
VIEWER_REPLAY_TIMEOUT_S = 2 * 60 * 60
REPLAY_MODES = frozenset({"review", "forensic"})
ROBOTWIN_REVIEW_MAX_FRAMES_PER_ACTION = 4
ROBOTWIN_REVIEW_HOLD_STEPS_PER_SAMPLE = 8
ROBOTWIN_FORENSIC_CAPTURE_STRIDE = 3


class ViewerReplayError(RuntimeError):
    """A run cannot be rendered through the controller-replay path."""


@contextmanager
def _exclusive_replay_lock(output: Path):
    """Serialize builders that target the same derived replay identity."""

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ViewerReplayError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed_dotenv_values(names: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    candidates = [PROJECT_ROOT / ".env"]
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        check=False, capture_output=True, text=True,
    )
    if common.returncode == 0:
        common_path = Path(common.stdout.strip())
        if not common_path.is_absolute():
            common_path = (PROJECT_ROOT / common_path).resolve()
        candidates.append(common_path.parent / ".env")
    for path in reversed(list(dict.fromkeys(candidates))):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name not in names:
                continue
            try:
                parts = shlex.split(value.strip(), comments=True, posix=True)
            except ValueError:
                continue
            if len(parts) == 1:
                result[name] = parts[0]
    return result


def _sampling_profile(replay_mode: str) -> dict[str, Any]:
    if replay_mode == "review":
        return {
            "mode": "action_keyframes_v1",
            "max_frames_per_action": ROBOTWIN_REVIEW_MAX_FRAMES_PER_ACTION,
            "hold_steps_per_sample": ROBOTWIN_REVIEW_HOLD_STEPS_PER_SAMPLE,
        }
    if replay_mode == "forensic":
        return {
            "mode": "physics_stride_v1",
            "capture_stride": ROBOTWIN_FORENSIC_CAPTURE_STRIDE,
        }
    raise ViewerReplayError(f"unsupported Viewer replay mode: {replay_mode}")


def _replay_implementation_hashes() -> dict[str, str]:
    """Bind reconstruction to the code that constructs and advances its world."""
    paths = {
        "worker": "ora/environment/replay_worker.py",
        "controller_replay": "ora/environment/controller_replay.py",
        "libero_session": "ora/environment/backends/libero_pro/session.py",
        "robotwin_session": "ora/environment/backends/robotwin/session.py",
    }
    return {name: _hash(PROJECT_ROOT / relative) for name, relative in paths.items()}


def replay_identity(root: Path, *, replay_mode: str = "review") -> tuple[str, Path]:
    manifest = root / "native/controller-replay/manifest.json"
    if not manifest.is_file():
        raise ViewerReplayError("run has no controller replay manifest")
    # A v2 header hashes a committed journal prefix. Validate that prefix and
    # its artifacts even when an already-encoded video would bypass the worker.
    from ora.environment.controller_replay import (
        MANIFEST_SCHEMA, load_controller_replay_manifest,
    )
    if _load(manifest).get("schema") == MANIFEST_SCHEMA:
        load_controller_replay_manifest(manifest)
    identity = hashlib.sha256(
        json.dumps(
            {
                "run_id": root.name,
                "manifest_sha256": _hash(manifest),
                "reader_sha256": _hash(PROJECT_ROOT / "ora/environment/controller_replay.py"),
                "implementation_sha256": _replay_implementation_hashes(),
                "replay_profile": _sampling_profile(replay_mode),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    workspace = root.parent.parent if root.parent.name == "runs" else root.parent
    return identity, workspace / "viewer-derived" / root.name / f"replay-{identity[:16]}"


def ensure_controller_replay(
    root: Path, *, replay_mode: str = "review",
) -> tuple[Path, dict[str, Any]]:
    """Compile or reuse frames outside the canonical run evidence."""

    root = root.resolve()
    identity, output = replay_identity(root, replay_mode=replay_mode)
    with _exclusive_replay_lock(output):
        cached = output / "manifest.json"
        if cached.is_file():
            result = _load(cached)
            if (
                result.get("schema") == RESULT_SCHEMA
                and result.get("identity") == identity
            ):
                return output, result
        return _compile_controller_replay(root, replay_mode, identity, output)


def _compile_controller_replay(
    root: Path, replay_mode: str, identity: str, output: Path,
) -> tuple[Path, dict[str, Any]]:
    cached = output / "manifest.json"
    plan = _load(root / "plan.json")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ViewerReplayError("run plan omits Runtime configuration")
    python_name = runtime.get("python_env")
    if not isinstance(python_name, str) or not python_name:
        raise ViewerReplayError("run plan omits replay Python binding")
    required_names = {python_name}
    for field in (
        "root_env", "libero_pro_root_env", "data_root_env",
    ):
        value = runtime.get(field)
        if isinstance(value, str) and value:
            required_names.add(value)
    environment = os.environ.copy()
    for name, value in _allowed_dotenv_values(required_names).items():
        environment.setdefault(name, value)
    python = environment.get(python_name)
    if not python:
        raise ViewerReplayError(
            f"Viewer replay requires environment variable {python_name}"
        )
    binding = runtime.get("resource_binding")
    if isinstance(binding, Mapping):
        worker_environment = binding.get("worker_environment")
        if isinstance(worker_environment, Mapping):
            for name, value in worker_environment.items():
                if isinstance(name, str) and isinstance(value, str):
                    environment[name] = value
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not current_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    job = output.parent / f".{output.name}.job.json"
    job.write_text(
        json.dumps(
            {
                "schema": JOB_SCHEMA,
                "identity": identity,
                "run_root": str(root),
                "output_root": str(output),
                "project_root": str(PROJECT_ROOT),
                "sampling": _sampling_profile(replay_mode),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [str(python), "-m", "ora.environment.replay_worker", "--job", str(job)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=VIEWER_REPLAY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ViewerReplayError(
            f"controller replay timed out after {VIEWER_REPLAY_TIMEOUT_S} s"
        ) from exc
    finally:
        job.unlink(missing_ok=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ViewerReplayError(detail[-4000:] or "controller replay worker failed")
    result = _load(cached)
    if result.get("schema") != RESULT_SCHEMA:
        raise ViewerReplayError("controller replay worker omitted its result manifest")
    return output, result


__all__ = [
    "REPLAY_MODES",
    "ViewerReplayError",
    "ensure_controller_replay",
    "replay_identity",
]
