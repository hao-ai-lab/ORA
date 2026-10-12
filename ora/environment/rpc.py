"""Local process transport for the transport-neutral environment session."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.contracts.refusal import EnvironmentRefusal

from .base import validate_environment_status


RPC_SCHEMA = "ora.environment-rpc.v1"
SESSION_RECORD_SCHEMA = "ora.environment-process-session.v1"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
# Linux sockaddr_un.sun_path is 108 bytes including the terminating NUL.
MAX_UNIX_SOCKET_PATH_BYTES = 107
#: A Kit worker exits with this before starting Kit when its PID collides with
#: another user's carb segment (sysexits EX_CANTCREAT; RoboTwin qualification
#: already uses 75). The launcher relaunches it under a new PID.
PID_SEGMENT_COLLISION_EXIT = 73
MAX_PID_COLLISION_RELAUNCHES = 3
#: Worker backends that start Kit in-process; only these are checked and
#: relaunched, and only their session record carries pid_collision_relaunches.
KIT_BACKENDS = frozenset({"robodojo"})
#: Written next to the socket by a colliding worker: {"worker_pid", "segment"}.
PID_COLLISION_FILE = "pid-collision.json"
WORKER_ENVIRONMENT_ALLOWLIST = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID",
    "PYOPENGL_PLATFORM",
    # libglvnd reads its vendor JSON from __EGL_VENDOR_LIBRARY_DIRS. Hosts with
    # a system-installed libEGL never need it, which is why it was missing; on a
    # host whose graphics runtime lives in user space (for example aarch64
    # nodes that unpack theirs under the home directory) dropping it leaves the worker unable to find any
    # vendor at all. EGL then enumerates zero devices and MuJoCo rejects any
    # MUJOCO_EGL_DEVICE_ID with "must be an integer between 0 and -1", which
    # reads like a device-index bug rather than a missing loader path.
    "__EGL_VENDOR_LIBRARY_DIRS",
    # A Mesa llvmpipe LIBERO lane names its vendor file and llvmpipe thread
    # budget in its resource worker_environment (orchestrator/resources.py).
    "__EGL_VENDOR_LIBRARY_FILENAMES", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS",
    "DISPLAY", "XDG_RUNTIME_DIR", "TMPDIR", "TMP", "TEMP",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "LIBERO_CONFIG_PATH", "LIBERO_TYPE", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
}

#: Isaac resolves scene MDL materials from a public S3 bucket. On hosts that
#: reach the internet only through a local proxy, dropping these left an
#: uncached material fetch hanging until the 600 s start limit. A proxy URL can
#: carry credentials or name a private host, so only a loopback proxy without
#: user information is passed through; anything else is dropped.
WORKER_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
WORKER_NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")


def _loopback_proxy(value: str) -> bool:
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return (parts.scheme in {"http", "https", "socks5", "socks5h"}
            and parts.username is None and parts.password is None
            and parts.hostname in {"127.0.0.1", "localhost", "::1"})


class EnvironmentRpcError(RuntimeError):
    """The isolated environment worker failed or violated the RPC contract."""


def _worker_environment(
    project_root: Path,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal simulator environment without policy credentials."""

    environment = {
        name: value for name, value in os.environ.items()
        if name in WORKER_ENVIRONMENT_ALLOWLIST
    }
    proxies = {name: os.environ[name] for name in WORKER_PROXY_VARIABLES
               if _loopback_proxy(os.environ.get(name, ""))}
    environment.update(proxies)
    if proxies:
        environment.update({name: os.environ[name] for name in WORKER_NO_PROXY_VARIABLES
                            if os.environ.get(name)})
    environment["PYTHONPATH"] = str(project_root.resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    for name, value in (overrides or {}).items():
        if name not in WORKER_ENVIRONMENT_ALLOWLIST:
            raise EnvironmentRpcError(
                f"environment worker override is not allowlisted: {name}"
            )
        if not isinstance(value, str) or not value or "\x00" in value:
            raise EnvironmentRpcError(
                f"environment worker override is invalid: {name}"
            )
        environment[name] = value
    return environment


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalRpcEnvironmentSession:
    """EnvironmentSession proxy backed by one isolated local worker process."""

    def __init__(
        self,
        *,
        python_executable: Path,
        project_root: Path,
        output_root: Path,
        worker_config: Mapping[str, Any],
        socket_root: Path | None = None,
        startup_timeout_s: float = 120.0,
        request_timeout_s: float = 1800.0,
        close_timeout_s: float | None = None,
        redacted_environment_names: Sequence[str] = (),
        worker_environment: Mapping[str, str] | None = None,
    ) -> None:
        # Keep the configured venv/conda entry path. Resolving its symlink would
        # discard Python's environment discovery and silently use the base env.
        self.python_executable = Path(
            os.path.abspath(os.path.expanduser(str(python_executable)))
        )
        self.project_root = project_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.session_id = f"env-{uuid.uuid4().hex}"
        self.requested_socket_root = (
            socket_root.expanduser().resolve()
            if socket_root is not None
            else self.output_root / "transport"
        )
        requested_socket = (
            self.requested_socket_root / self.session_id[4:20] / "worker.sock"
        )
        self.socket_root_fallback = (
            len(os.fsencode(str(requested_socket))) > MAX_UNIX_SOCKET_PATH_BYTES
        )
        self.socket_root = (
            Path(tempfile.gettempdir()).resolve()
            if self.socket_root_fallback else self.requested_socket_root
        )
        selected_socket = self.socket_root / self.session_id[4:20] / "worker.sock"
        if len(os.fsencode(str(selected_socket))) > MAX_UNIX_SOCKET_PATH_BYTES:
            raise EnvironmentRpcError(
                "no safe filesystem path is short enough for the environment socket"
            )
        self.worker_config = dict(worker_config)
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        if close_timeout_s is not None and (
            isinstance(close_timeout_s, bool)
            or not math.isfinite(float(close_timeout_s))
            or float(close_timeout_s) <= 0
        ):
            raise EnvironmentRpcError("environment close timeout must be positive and finite")
        self.close_timeout_s = (
            float(close_timeout_s) if close_timeout_s is not None else None
        )
        self.redacted_environment_names = tuple(redacted_environment_names)
        self.worker_environment = dict(worker_environment or {})
        self.process: subprocess.Popen[str] | None = None
        self.pid_collision_relaunches: list[dict[str, Any]] = []
        self.connection: socket.socket | None = None
        self.reader: Any = None
        self.writer: Any = None
        self.request_sequence = 0
        self.started = False
        self._closed = False
        self.created_unix_s = time.time()

    @property
    def transport_root(self) -> Path:
        return self.socket_root / self.session_id[4:16]

    @property
    def socket_path(self) -> Path:
        return self.transport_root / "s.sock"

    @property
    def config_path(self) -> Path:
        return self.transport_root / "c.json"

    @property
    def record_path(self) -> Path:
        return self.output_root / "environment-session.json"

    def _process_record(self, *, state: str, **fields: Any) -> dict[str, Any]:
        executable = self.python_executable
        return {
            "schema": SESSION_RECORD_SCHEMA,
            "session_id": self.session_id,
            "created_unix_s": self.created_unix_s,
            "state": state,
            "transport": "local_rpc",
            "protocol": RPC_SCHEMA,
            "worker_pid": self.process.pid if self.process is not None else None,
            "python_executable": str(executable),
            "python_binary": str(executable.resolve()) if executable.exists() else None,
            "python_sha256": _file_sha256(executable.resolve()) if executable.is_file() else None,
            "socket": str(self.socket_path),
            "config": str(self.config_path),
            "socket_root_requested": str(self.requested_socket_root),
            "socket_root_selected": str(self.socket_root),
            "socket_root_fallback": self.socket_root_fallback,
            "worker_environment": dict(sorted(self.worker_environment.items())),
            **({"pid_collision_relaunches": list(self.pid_collision_relaunches)}
               if self.worker_config.get("backend") in KIT_BACKENDS else {}),
            "updated_unix_s": time.time(),
            **fields,
        }

    def _persist_record(self, *, state: str, **fields: Any) -> None:
        _atomic_json(self.record_path, self._process_record(state=state, **fields))

    def _spawn_worker(self, log_path: Path, config_path: Path, environment: Mapping[str, str]) -> None:
        log = log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [
                    str(self.python_executable),
                    "-m",
                    "ora.environment.worker",
                    "--socket",
                    str(self.socket_path),
                    "--config",
                    str(config_path),
                ],
                cwd=self.project_root,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            log.close()
        self._persist_record(state="starting")

    def _relaunch_after_pid_collision(
        self, log_path: Path, config_path: Path, environment: Mapping[str, str],
    ) -> bool:
        """Start a fresh worker (new PID) after a reported carb PID collision."""

        if (self.process is None or self.process.returncode != PID_SEGMENT_COLLISION_EXIT
                or self.worker_config.get("backend") not in KIT_BACKENDS
                or len(self.pid_collision_relaunches) >= MAX_PID_COLLISION_RELAUNCHES):
            return False
        report = self.transport_root / PID_COLLISION_FILE
        try:
            segment = json.loads(report.read_text(encoding="utf-8")).get("segment")
        except (OSError, ValueError, AttributeError):
            segment = None
        report.unlink(missing_ok=True)
        self.pid_collision_relaunches.append({
            "worker_pid": self.process.pid, "returncode": self.process.returncode,
            "segment": segment if isinstance(segment, str) else None, "unix_s": time.time(),
        })
        self.socket_path.unlink(missing_ok=True)
        self._spawn_worker(log_path, config_path, environment)
        return True

    def _launch(self) -> None:
        if self._closed:
            raise EnvironmentRpcError("environment worker session is closed")
        if self.process is not None:
            raise EnvironmentRpcError("environment worker was already launched")
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise EnvironmentRpcError(
                f"environment Python is not executable: {self.python_executable}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        transport_root = self.transport_root
        transport_root.mkdir(parents=True, exist_ok=True)
        transport_root.chmod(0o700)
        if len(os.fsencode(self.socket_path)) >= 108:
            raise EnvironmentRpcError(
                "environment worker socket path exceeds the AF_UNIX limit"
            )
        self.socket_path.unlink(missing_ok=True)
        config_path = self.config_path
        config = {
            **self.worker_config,
            "schema": RPC_SCHEMA,
            "session_id": self.session_id,
            "shared_root": str(self.output_root.parent.resolve()),
        }
        _atomic_json(config_path, config)
        config_path.chmod(0o600)
        log_path = self.output_root / "environment-worker.log"
        environment = _worker_environment(
            self.project_root, self.worker_environment,
        )
        for name in self.redacted_environment_names:
            environment.pop(name, None)
        deadline = time.monotonic() + self.startup_timeout_s
        self._spawn_worker(log_path, config_path, environment)
        last_error = "worker socket not ready"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                if self._relaunch_after_pid_collision(log_path, config_path, environment):
                    continue
                raise EnvironmentRpcError(
                    f"environment worker exited during startup with code {self.process.returncode}; "
                    f"see {log_path}"
                )
            if self.socket_path.exists():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.settimeout(min(1.0, max(0.1, deadline - time.monotonic())))
                try:
                    connection.connect(str(self.socket_path))
                except OSError as exc:
                    last_error = str(exc)
                    connection.close()
                else:
                    connection.settimeout(self.request_timeout_s)
                    self.connection = connection
                    self.reader = connection.makefile("r", encoding="utf-8", newline="\n")
                    self.writer = connection.makefile("w", encoding="utf-8", newline="\n")
                    self._request("ping", {})
                    self._persist_record(state="ready")
                    return
            time.sleep(0.05)
        raise EnvironmentRpcError(f"environment worker startup timed out: {last_error}")

    def _request(self, method: str, params: Mapping[str, Any]) -> Any:
        if self.process is None or self.connection is None or self.reader is None or self.writer is None:
            raise EnvironmentRpcError("environment worker is not connected")
        returncode = self.process.poll()
        if returncode is not None:
            raise EnvironmentRpcError(f"environment worker exited with code {returncode}")
        request_id = self.request_sequence
        self.request_sequence += 1
        request = {
            "schema": RPC_SCHEMA,
            "session_id": self.session_id,
            "request_id": request_id,
            "method": method,
            "params": dict(params),
        }
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode()) > MAX_MESSAGE_BYTES:
            raise EnvironmentRpcError("environment RPC request exceeds size limit")
        try:
            self.writer.write(encoded + "\n")
            self.writer.flush()
            line = self.reader.readline(MAX_MESSAGE_BYTES + 1)
        except (OSError, TimeoutError) as exc:
            raise EnvironmentRpcError(f"environment RPC {method} failed: {exc}") from exc
        if not line:
            returncode = self.process.poll()
            detail = "closed the RPC connection" if returncode is None else f"exited with code {returncode}"
            raise EnvironmentRpcError(f"environment worker {detail} during {method}")
        if len(line.encode()) > MAX_MESSAGE_BYTES:
            raise EnvironmentRpcError("environment RPC response exceeds size limit")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnvironmentRpcError("environment worker returned invalid JSON") from exc
        if (
            not isinstance(response, Mapping)
            or response.get("schema") != RPC_SCHEMA
            or response.get("session_id") != self.session_id
            or response.get("request_id") != request_id
        ):
            raise EnvironmentRpcError("environment RPC response identity mismatch")
        if response.get("ok") is not True:
            error = response.get("error", {})
            if not isinstance(error, Mapping):
                raise EnvironmentRpcError("environment RPC error envelope is malformed")
            if error.get("kind") == "refusal":
                try:
                    refusal = EnvironmentRefusal.from_dict(error)
                except (TypeError, ValueError) as exc:
                    raise EnvironmentRpcError("environment RPC refusal envelope is malformed") from exc
                raise refusal
            raise EnvironmentRpcError(
                f"environment {method} failed: {error.get('type', 'Error')}: "
                f"{error.get('message', 'unknown worker failure')}"
            )
        return response.get("result")

    def start(self) -> Mapping[str, Any]:
        self._launch()
        result = self._request("start", {})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment start result must be an object")
        self.started = True
        self._persist_record(state="running", reset_schema=result.get("schema"))
        return result

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> Mapping[str, Any]:
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index <= 0:
            raise EnvironmentRpcError("round_index must be a positive integer")
        if (
            seed is not None
            and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            )
        ):
            raise EnvironmentRpcError("round seed must be a non-negative integer")
        params: dict[str, Any] = {"round_index": round_index}
        if seed is not None:
            params["seed"] = seed
        result = self._request("restart_round", params)
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment round reset result must be an object")
        self._persist_record(
            state="running", reset_schema=result.get("schema"),
            round_index=round_index, seed=result.get("seed"),
        )
        return result

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        result = self._request("observe", {"destination": str(destination.resolve())})
        if not isinstance(result, dict) or not isinstance(result.get("observation"), dict):
            raise EnvironmentRpcError("environment observe result is invalid")
        images = result.get("images")
        if not isinstance(images, list) or not all(isinstance(path, str) for path in images):
            raise EnvironmentRpcError("environment image references are invalid")
        return result["observation"], [Path(path) for path in images]

    def begin_action_sequence(self, action: Mapping[str, Any]) -> None:
        self._request("begin_action_sequence", {"action": dict(action)})

    def end_action_sequence(self, action_id: str) -> None:
        self._request("end_action_sequence", {"action_id": action_id})

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        result = self._request("execute", {"action": dict(action)})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment execution receipt must be an object")
        return result

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        result = self._request("align_plan", {"request": dict(request)})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment alignment result must be an object")
        return result

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        result = self._request("probe_motion", {"request": dict(request)})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment motion probe result must be an object")
        return result

    def measure_jaw_state(self) -> dict[str, Any]:
        """Private jaw evidence for the scripted aperture probe only."""

        result = self._request("measure_jaw_state", {})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment jaw state must be an object")
        return result

    def status(self) -> dict[str, Any]:
        result = self._request("status", {})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment status must be an object")
        return validate_environment_status(result)

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def abort(self, reason: str) -> None:
        self._request("abort", {"reason": reason})

    def persist_official_verdict(self) -> dict[str, Any]:
        result = self._request("persist_official_verdict", {})
        if not isinstance(result, dict):
            raise EnvironmentRpcError("environment verdict must be an object")
        return result

    def finalize_operator_video(self) -> dict[str, Any] | None:
        result = self._request("finalize_operator_video", {})
        if result is not None and not isinstance(result, dict):
            raise EnvironmentRpcError("environment video record must be null or an object")
        return result

    def close(self) -> None:
        if self._closed:
            return
        process = self.process
        close_error = None
        try:
            if process is not None and process.poll() is None and self.connection is not None:
                try:
                    # Isaac cleanup can hang independently of healthy action
                    # requests. Preserve legacy deadlines unless selected.
                    if self.close_timeout_s is not None:
                        self.connection.settimeout(self.close_timeout_s)
                    self._request("close", {})
                except (EnvironmentRpcError, OSError) as exc:
                    close_error = str(exc)
        finally:
            for stream in (self.reader, self.writer):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.reader = None
            self.writer = None
            if self.connection is not None:
                try:
                    self.connection.close()
                except OSError:
                    pass
                self.connection = None
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
            if process is not None:
                self._persist_record(
                    state="closed",
                    worker_returncode=process.poll(),
                    started=self.started,
                    **({
                        "close_timeout_s": self.close_timeout_s,
                        "close_error": close_error,
                    } if self.close_timeout_s is not None else {}),
                )
            self.socket_path.unlink(missing_ok=True)
            self.config_path.unlink(missing_ok=True)
            try:
                self.transport_root.rmdir()
            except OSError:
                pass
            self._closed = True
