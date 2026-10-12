"""Host-side orchestration for simulator-derived controller replay frames."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_SCHEMA = "ora.viewer-controller-replay-job.v2"
RESULT_SCHEMA = "ora.viewer-controller-replay.v2"
HISTORICAL_JOB_SCHEMA = (
    "ora.viewer-historical-libero-reenactment-job.v1"
)
HISTORICAL_RESULT_SCHEMA = (
    "ora.viewer-historical-libero-reenactment.v1"
)
VIEWER_REPLAY_TIMEOUT_S = 2 * 60 * 60
REPLAY_MODES = frozenset({"review", "forensic"})
ROBOTWIN_REVIEW_MAX_FRAMES_PER_ACTION = 4
ROBOTWIN_REVIEW_HOLD_STEPS_PER_SAMPLE = 8
ROBOTWIN_FORENSIC_CAPTURE_STRIDE = 3
HISTORICAL_REVIEW_MAX_FRAMES_PER_ACTION = 4
HISTORICAL_VALIDATION_THRESHOLDS = {
    "translation_m": 0.02,
    "orientation_rad": 0.10,
    "gripper_normalized": 0.08,
}
VIEWER_GPU_ENV = "ORA_VIEWER_GPU_ID"


class ViewerReplayError(RuntimeError):
    """A run cannot be rendered through the controller-replay path."""


class ViewerReplayValidationError(ViewerReplayError):
    """A historical replay ran but diverged from its sealed receipt."""


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


def _git_common_project_root() -> Path:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        return PROJECT_ROOT
    common = Path(completed.stdout.strip())
    if not common.is_absolute():
        common = (PROJECT_ROOT / common).resolve()
    return common.parent


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _exact_dependency_root(
    source_root: Path, *, revision: str, label: str,
) -> Path:
    """Resolve an immutable exact-revision local clone without moving source HEAD."""

    source_root = source_root.expanduser().resolve()
    if _git_revision(source_root) == revision:
        return source_root
    available = subprocess.run(
        ["git", "-C", str(source_root), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False, capture_output=True, text=True,
    )
    if available.returncode:
        raise ViewerReplayError(
            f"historical {label} revision is unavailable locally: {revision}"
        )
    workspace = Path(
        os.environ.get("ORA_WORKSPACE_ROOT", str(_git_common_project_root()))
    ).expanduser().resolve()
    cache = workspace / "outputs/viewer-derived/historical-dependencies"
    target = cache / f"{label}-{revision[:16]}"
    if target.exists():
        if _git_revision(target) != revision:
            raise ViewerReplayError(
                f"historical dependency cache identity drifted: {target}"
            )
        dirty = subprocess.run(
            ["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"],
            check=False, capture_output=True, text=True,
        )
        if dirty.returncode or dirty.stdout.strip():
            raise ViewerReplayError(f"historical dependency cache is modified: {target}")
        return target
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise ViewerReplayError(f"historical dependency staging path exists: {temporary}")
    try:
        cloned = subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(source_root), str(temporary)],
            check=False, capture_output=True, text=True,
        )
        if cloned.returncode:
            raise ViewerReplayError(cloned.stderr.strip() or "dependency clone failed")
        checked_out = subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--detach", revision],
            check=False, capture_output=True, text=True,
        )
        if checked_out.returncode:
            raise ViewerReplayError(
                checked_out.stderr.strip() or "dependency checkout failed"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
    return target


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _historical_libero_qualification(root: Path) -> dict[str, Any] | None:
    """Recover the exact reset-time controller qualification from evidence."""

    for event in _jsonl(root / "events.jsonl"):
        if event.get("type") != "reset":
            continue
        record = event.get("record")
        binding = (
            record.get("qualified_controller_binding")
            if isinstance(record, Mapping) else None
        )
        return dict(binding) if isinstance(binding, Mapping) else None
    return None


def _apply_viewer_gpu_override(environment: dict[str, str]) -> dict[str, str] | None:
    """Bind derived rendering to one explicit physical GPU when requested."""

    raw = os.environ.get(VIEWER_GPU_ENV)
    if raw is None:
        return None
    value = raw.strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise ViewerReplayError(f"{VIEWER_GPU_ENV} must be one physical GPU index")
    environment["CUDA_VISIBLE_DEVICES"] = value
    environment["MUJOCO_EGL_DEVICE_ID"] = value
    return {
        "schema": "ora.viewer-render-resource.v1",
        "physical_gpu_id": int(value),
        "cuda_visible_devices": value,
        "mujoco_egl_device_id": int(value),
    }


def _historical_libero_actions(root: Path) -> tuple[list[dict[str, Any]], int]:
    """Pair exact submitted actions and receipts while retaining round order."""

    entries: list[dict[str, Any]] = []
    pending: dict[str, tuple[dict[str, Any], int]] = {}
    round_index = -1
    round_action_sequences: dict[int, int] = {}
    for event in _jsonl(root / "events.jsonl"):
        event_type = event.get("type")
        if event_type == "reset":
            round_index += 1
        elif event_type == "action_submitted":
            action = event.get("action")
            action_id = action.get("action_id") if isinstance(action, Mapping) else None
            if not isinstance(action_id, str) or not isinstance(action, Mapping):
                raise ViewerReplayError("historical action submission is malformed")
            if action.get("schema") not in {
                "ora-action.v3",
                "ora-action.v4",
            }:
                raise ViewerReplayError("historical action codec is unsupported")
            pending[action_id] = (dict(action), max(round_index, 0))
        elif event_type == "motion_receipt":
            receipt = event.get("record")
            action_id = receipt.get("action_id") if isinstance(receipt, Mapping) else None
            if not isinstance(action_id, str) or action_id not in pending:
                raise ViewerReplayError("historical action/receipt order is incomplete")
            action, action_round = pending.pop(action_id)
            local_sequence = round_action_sequences.get(action_round, 0)
            round_action_sequences[action_round] = local_sequence + 1
            entries.append({
                "action_sequence": len(entries),
                "round_index": action_round,
                "round_action_sequence": local_sequence,
                "action": action,
                "recorded_receipt": dict(receipt),
            })
    if pending:
        raise ViewerReplayError("historical run has unpaired submitted actions")
    round_count = max(round_index + 1, 1)
    for entry in entries:
        entry["round_action_count"] = round_action_sequences.get(
            int(entry["round_index"]), 0,
        )
        entry["round_count"] = round_count
    return entries, round_count


def historical_libero_replay_identity(root: Path) -> tuple[str, Path] | None:
    """Return an identity only for legacy LIBERO runs lacking controller trace."""

    root = root.resolve()
    if (root / "native/controller-replay/manifest.json").is_file():
        return None
    if not (root / "seal.json").is_file():
        return None
    plan_path = root / "plan.json"
    events_path = root / "events.jsonl"
    if not plan_path.is_file() or not events_path.is_file():
        return None
    plan = _load(plan_path)
    runtime = plan.get("runtime")
    source = plan.get("source")
    if not isinstance(runtime, Mapping) or runtime.get("kind") != "libero_pro":
        return None
    commit = source.get("commit") if isinstance(source, Mapping) else None
    if not isinstance(commit, str) or len(commit) != 40:
        return None
    worker = PROJECT_ROOT / "ora/environment/historical_libero_replay_worker.py"
    actions, round_count = _historical_libero_actions(root)
    identity = hashlib.sha256(json.dumps({
        "run_id": root.name,
        "events_sha256": _hash(events_path),
        "plan_sha256": _hash(plan_path),
        "historical_harness_commit": commit,
        "worker_sha256": _hash(worker),
        "sampling": {
            "mode": "action_keyframes_v1",
            "max_frames_per_action": HISTORICAL_REVIEW_MAX_FRAMES_PER_ACTION,
        },
        "validation_thresholds": HISTORICAL_VALIDATION_THRESHOLDS,
        "action_count": len(actions),
        "round_count": round_count,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    workspace = root.parent.parent if root.parent.name == "runs" else root.parent
    output = (
        workspace / "viewer-derived" / root.name
        / f"historical-reenactment-{identity[:16]}"
    )
    return identity, output


def _extract_historical_source(commit: str, target: Path) -> None:
    archive = target.parent / f"{commit}.tar"
    completed = subprocess.run(
        [
            "git", "-C", str(PROJECT_ROOT), "archive", "--format=tar",
            f"--output={archive}", commit,
        ],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise ViewerReplayError(
            completed.stderr.strip() or f"historical commit is unavailable: {commit}"
        )
    try:
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                member_path = (target / member.name).resolve()
                if not member_path.is_relative_to(target.resolve()):
                    raise ViewerReplayError("historical source archive escaped its root")
            stream.extractall(target)
    finally:
        archive.unlink(missing_ok=True)


def ensure_historical_libero_replay(root: Path) -> tuple[Path, dict[str, Any]]:
    """Compile validated exact-source semantic re-enactment for one legacy run."""

    root = root.resolve()
    identity_binding = historical_libero_replay_identity(root)
    if identity_binding is None:
        raise ViewerReplayError("run is not eligible for historical LIBERO re-enactment")
    identity, output = identity_binding
    with _exclusive_replay_lock(output):
        cached = output / "manifest.json"
        if cached.is_file():
            result = _load(cached)
            if (
                result.get("schema") == HISTORICAL_RESULT_SCHEMA
                and result.get("identity") == identity
            ):
                return output, result
        return _compile_historical_libero_replay(root, identity, output)


def _compile_historical_libero_replay(
    root: Path, identity: str, output: Path,
) -> tuple[Path, dict[str, Any]]:
    cached = output / "manifest.json"
    plan = _load(root / "plan.json")
    runtime = plan.get("runtime")
    source = plan.get("source")
    if not isinstance(runtime, Mapping) or not isinstance(source, Mapping):
        raise ViewerReplayError("historical run plan is malformed")
    commit = source.get("commit")
    python_name = runtime.get("python_env")
    if not isinstance(commit, str) or not isinstance(python_name, str):
        raise ViewerReplayError("historical source/Python binding is unavailable")
    environment = os.environ.copy()
    required_names = {python_name}
    field_names = {
        **({"rlinf_root": "rlinf_root_env"}
           if runtime.get("environment_driver", "rlinf_v1") == "rlinf_v1" else {}),
        "libero_pro_root": "libero_pro_root_env",
        "data_root": "data_root_env",
    }
    for field in field_names.values():
        name = runtime.get(field)
        if isinstance(name, str):
            required_names.add(name)
    for name, value in _allowed_dotenv_values(required_names).items():
        environment.setdefault(name, value)
    python = environment.get(python_name)
    if not python:
        raise ViewerReplayError(
            f"historical Viewer replay requires environment variable {python_name}"
        )
    runtime_paths: dict[str, str] = {}
    revisions = {
        "rlinf_root": runtime.get("required_rlinf_revision"),
        "libero_pro_root": runtime.get("required_libero_pro_revision"),
        "data_root": runtime.get("required_data_revision"),
    }
    for label, field in field_names.items():
        name = runtime.get(field)
        value = environment.get(str(name)) if isinstance(name, str) else None
        if not value:
            raise ViewerReplayError(f"historical Viewer replay requires {name}")
        revision = revisions[label]
        if not isinstance(revision, str):
            raise ViewerReplayError(f"historical run omits {label} revision")
        runtime_paths[label] = str(_exact_dependency_root(
            Path(value), revision=revision, label=label.replace("_root", ""),
        ))
    binding = runtime.get("resource_binding")
    if isinstance(binding, Mapping):
        worker_environment = binding.get("worker_environment")
        if isinstance(worker_environment, Mapping):
            for name, value in worker_environment.items():
                if isinstance(name, str) and isinstance(value, str):
                    environment[name] = value
    viewer_resource = _apply_viewer_gpu_override(environment)
    actions, round_count = _historical_libero_actions(root)
    sampling = {
        "mode": "action_keyframes_v1",
        "max_frames_per_action": HISTORICAL_REVIEW_MAX_FRAMES_PER_ACTION,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    worker = PROJECT_ROOT / "ora/environment/historical_libero_replay_worker.py"
    with tempfile.TemporaryDirectory(
        prefix=f"historical-source-{commit[:8]}-", dir=output.parent,
    ) as source_name:
        source_root = Path(source_name)
        _extract_historical_source(commit, source_root)
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root) if not current_pythonpath
            else str(source_root) + os.pathsep + current_pythonpath
        )
        job = output.parent / f".{output.name}.job.json"
        job.write_text(json.dumps({
            "schema": HISTORICAL_JOB_SCHEMA,
            "identity": identity,
            "run_root": str(root),
            "output_root": str(output),
            "historical_harness_commit": commit,
            "source_events": {
                "path": "events.jsonl", "sha256": _hash(root / "events.jsonl"),
            },
            "plan": plan,
            "runtime_paths": runtime_paths,
            "qualified_controller_binding": _historical_libero_qualification(root),
            "viewer_resource": viewer_resource,
            "actions": actions,
            "round_count": round_count,
            "sampling": sampling,
            "validation_thresholds": HISTORICAL_VALIDATION_THRESHOLDS,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(python), str(worker), "--job", str(job)],
                cwd=source_root, env=environment, check=False,
                capture_output=True, text=True, timeout=VIEWER_REPLAY_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise ViewerReplayError(
                f"historical LIBERO re-enactment timed out after "
                f"{VIEWER_REPLAY_TIMEOUT_S} s"
            ) from exc
        finally:
            job.unlink(missing_ok=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        final_line = lines[-1] if lines else detail
        if "semantic re-enactment diverged at action" in final_line:
            raise ViewerReplayValidationError(final_line[-2000:])
        raise ViewerReplayError(
            detail[-4000:] or "historical LIBERO re-enactment worker failed"
        )
    result = _load(cached)
    if (
        result.get("schema") != HISTORICAL_RESULT_SCHEMA
        or result.get("identity") != identity
    ):
        raise ViewerReplayError("historical replay worker omitted its result manifest")
    return output, result


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
        "root_env", "rlinf_root_env", "libero_pro_root_env", "data_root_env",
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
    "ViewerReplayValidationError",
    "ensure_controller_replay",
    "ensure_historical_libero_replay",
    "historical_libero_replay_identity",
    "replay_identity",
]
