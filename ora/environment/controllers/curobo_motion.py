"""Host-safe lifecycle and wire contract for an isolated CuRobo planner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.planning_diagnostics import validate_planning_diagnostics


CUROBO_PLANNER_SCHEMA = "ora.curobo-planner-binding.v1"
CUROBO_REQUEST_SCHEMA = "ora.curobo-motion-request.v1"
CUROBO_RESPONSE_SCHEMA = "ora.curobo-motion-response.v1"
CUROBO_REQUIRED_VERSION = "v0.7.8"
_PANDA_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]


class CuroboPlannerError(RuntimeError):
    """The isolated planner was absent, drifted, timed out, or rejected a request."""


def validate_curobo_request(value: Any) -> dict[str, Any]:
    """Validate the deliberately small robot-only planning request."""

    if not isinstance(value, Mapping):
        raise CuroboPlannerError("CuRobo request must be an object")
    common = {"joint_names", "joint_positions"}
    goal_fields = {"target_pose_base_xyzw", "goal_joint_positions"}
    if set(value) - common - goal_fields or not common.issubset(value):
        raise CuroboPlannerError("CuRobo request fields drifted")
    present_goals = goal_fields.intersection(value)
    if len(present_goals) != 1:
        raise CuroboPlannerError("CuRobo request requires exactly one goal")
    names = value["joint_names"]
    joints = value["joint_positions"]
    if (
        names != _PANDA_JOINT_NAMES
        or not isinstance(joints, list) or len(joints) != 7
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in joints
        )
    ):
        raise CuroboPlannerError("CuRobo request has invalid current joints")
    goal_name = next(iter(present_goals))
    expected_length = 7
    goal = value[goal_name]
    if (
        not isinstance(goal, list) or len(goal) != expected_length
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in goal
        )
    ):
        raise CuroboPlannerError("CuRobo request has an invalid goal")
    if goal_name == "target_pose_base_xyzw" and not math.isclose(
        sum(float(item) ** 2 for item in goal[3:]), 1.0, abs_tol=1e-5,
    ):
        raise CuroboPlannerError("CuRobo target quaternion must be normalized")
    return json.loads(json.dumps(value))


def _curobo_failure_reason(value: Mapping[str, Any]) -> str:
    """Preserve implementation errors rather than relabeling them unreachable."""
    if value.get("status") == "planner_error":
        return "planner_error"
    return str(value.get("planner_status") or value.get("status") or "planner_error")


def _curobo_valid_query(value: Mapping[str, Any]) -> bool | None:
    """Project recorded validity; old incomplete responses remain unknown."""
    if isinstance(value.get("valid_query"), bool):
        return value["valid_query"]
    if value.get("status") == "planner_error":
        return None
    if "INVALID_START_STATE_" in str(value.get("planner_status")):
        return False
    return True if value.get("status") == "planned" else None


def validate_curobo_response(value: Any) -> dict[str, Any]:
    """Validate the minimum planner result before it reaches execution."""

    if not isinstance(value, Mapping) or value.get("schema") != CUROBO_RESPONSE_SCHEMA:
        raise CuroboPlannerError("CuRobo worker response schema drift")
    if value.get("world_model") != "none":
        raise CuroboPlannerError("CuRobo response must declare world_model=none")
    allowed = {
        "request_id", "schema", "status", "reachable", "world_model",
        "planner_status", "planning_time_s", "start_ee_pose_base_xyzw",
        "requested_target_pose_base_xyzw", "requested_goal_joint_positions",
        "trajectory", "error", "valid_query", "request", "start_state",
        "planning_diagnostics", "planning_diagnostic_evidence",
    }
    if set(value) - allowed:
        raise CuroboPlannerError("CuRobo response fields drifted")
    if "planning_diagnostics" in value:
        try:
            validate_planning_diagnostics(value["planning_diagnostics"])
        except (ValueError, TypeError) as exc:
            raise CuroboPlannerError("CuRobo planning diagnostics are malformed") from exc
    if "planning_diagnostic_evidence" in value:
        evidence = value["planning_diagnostic_evidence"]
        try:
            valid_evidence = isinstance(evidence, Mapping) and len(json.dumps(evidence, allow_nan=False)) <= 1_000_000
        except (ValueError, TypeError, RecursionError):
            valid_evidence = False
        if not valid_evidence:
            raise CuroboPlannerError("CuRobo planning diagnostic evidence is invalid or oversized")
    reachable = value.get("reachable")
    if not isinstance(reachable, bool):
        raise CuroboPlannerError("CuRobo response has invalid reachability")
    status = value.get("status")
    if status not in {"planned", "not_reachable", "planner_error"} or (
        reachable != (status == "planned")
    ):
        raise CuroboPlannerError("CuRobo response status and reachability disagree")
    error = value.get("error")
    if status == "planner_error":
        if not isinstance(error, Mapping) or set(error) != {"type", "message"} or any(
            not isinstance(error[key], str) or not error[key] for key in error
        ):
            raise CuroboPlannerError("CuRobo planner_error omitted its error identity")
    elif error is not None:
        raise CuroboPlannerError("ordinary CuRobo result contains an implementation error")
    planner_status = value.get("planner_status")
    success_statuses = {"success", "SUCCESS", "MotionGenStatus.SUCCESS"}
    if status == "planned" and planner_status is not None and (
        not isinstance(planner_status, str) or planner_status not in success_statuses
    ):
        raise CuroboPlannerError("CuRobo planned result has a failure status")
    if status == "not_reachable" and (
        not isinstance(planner_status, str) or not planner_status
        or planner_status in success_statuses | {"planner_error"}
    ):
        raise CuroboPlannerError("CuRobo refusal omitted a valid planner status")
    if "valid_query" in value:
        valid_query = value["valid_query"]
        if (valid_query is not None and not isinstance(valid_query, bool)) or (
            status == "planned" and valid_query is not True
        ) or (
            status == "not_reachable" and not isinstance(valid_query, bool)
        ) or (
            "INVALID_START_STATE_" in str(planner_status) and valid_query is not False
        ):
            raise CuroboPlannerError("CuRobo response has inconsistent query validity")
    if "request" in value:
        if value["request"] is None:
            if status != "planner_error":
                raise CuroboPlannerError("CuRobo result omitted its request evidence")
        else:
            validate_curobo_request(value["request"])
    if "start_state" in value and value["start_state"] is not None:
        start = value["start_state"]
        if not isinstance(start, Mapping) or start.get("schema") != (
            "ora.curobo-start-state.v1"
        ) or start.get("status") not in {"checking", "valid", "invalid"}:
            raise CuroboPlannerError("CuRobo start-state evidence is malformed")
        expected_valid = {"checking": None, "valid": True, "invalid": False}[start["status"]]
        if start.get("valid") is not expected_valid or (
            status != "planner_error" and start["status"] == "checking"
        ) or (
            status == "planned" and start["valid"] is not True
        ) or (
            status != "planner_error" and start["valid"] is False
            and value.get("valid_query") is not False
        ) or (
            "INVALID_START_STATE_" in str(planner_status) and start["valid"] is not False
        ):
            raise CuroboPlannerError("CuRobo result and start-state evidence disagree")
    if "planning_time_s" in value and (
        isinstance(value["planning_time_s"], bool)
        or not isinstance(value["planning_time_s"], (int, float))
        or not math.isfinite(value["planning_time_s"]) or value["planning_time_s"] < 0
    ):
        raise CuroboPlannerError("CuRobo planning time is malformed")
    trajectory = value.get("trajectory")
    if reachable:
        if not isinstance(trajectory, Mapping):
            raise CuroboPlannerError("reachable CuRobo response omitted trajectory")
        positions = trajectory.get("positions")
        names = trajectory.get("joint_names")
        dt_s = trajectory.get("dt_s")
        if (
            not isinstance(positions, list) or not positions
            or any(
                not isinstance(row, list) or len(row) != 7
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in row
                )
                for row in positions
            )
            or names != _PANDA_JOINT_NAMES
            or isinstance(dt_s, bool) or not isinstance(dt_s, (int, float))
            or not math.isfinite(float(dt_s)) or not 0 < float(dt_s) <= 1
        ):
            raise CuroboPlannerError("CuRobo trajectory is malformed")
        velocities = trajectory.get("velocities")
        if velocities is not None and (
            not isinstance(velocities, list) or len(velocities) != len(positions)
            or any(
                not isinstance(row, list) or len(row) != 7
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) for item in row)
                for row in velocities
            )
        ):
            raise CuroboPlannerError("CuRobo trajectory velocities are malformed")
    elif trajectory is not None:
        raise CuroboPlannerError("unreachable CuRobo response contains trajectory")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CuroboPlannerError("CuRobo response contains invalid JSON evidence") from exc


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, text=True,
        timeout=30, check=False,
    )
    if completed.returncode:
        raise CuroboPlannerError(
            completed.stderr.strip() or completed.stdout.strip() or "git probe failed"
        )
    return completed.stdout.strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CuroboPlannerClient:
    python_executable: Path
    curobo_root: Path
    required_revision: str
    output_root: Path
    device: str = "cuda:0"
    request_timeout_s: float = 180.0
    # Preflight starts the planner outside the lane's worker; it passes the
    # lane's worker environment so the probe selects the lane's GPU.
    environment_overrides: Mapping[str, str] | None = None
    _process: subprocess.Popen[str] | None = field(init=False, default=None, repr=False)
    _sequence: int = field(init=False, default=0)
    binding: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # absolute(), not resolve(): a venv's bin/python is a SYMLINK to the base
        # interpreter, and Python decides it is in a venv from the pyvenv.cfg next
        # to the path it was invoked as. Resolving the symlink lands on the shared
        # base interpreter, which has no pyvenv.cfg, so the venv's site-packages
        # silently disappear and the worker dies on `import numpy` — surfacing only
        # as "CuRobo worker closed without a response". conda envs hide this because
        # their bin/python is a real file.
        self.python_executable = self.python_executable.expanduser().absolute()
        self.curobo_root = self.curobo_root.expanduser().resolve()
        self.output_root = self.output_root.expanduser().resolve()
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise CuroboPlannerError("CuRobo Python is unavailable or not executable")
        revision = _git(self.curobo_root, "rev-parse", "HEAD")
        describe = _git(self.curobo_root, "describe", "--tags", "--exact-match")
        dirty = _git(self.curobo_root, "status", "--porcelain", "--untracked-files=no")
        if revision != self.required_revision or describe != CUROBO_REQUIRED_VERSION or dirty:
            raise CuroboPlannerError(
                "CuRobo source must be clean, exact v0.7.8, and match required_revision"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.binding = {
            "schema": CUROBO_PLANNER_SCHEMA,
            "backend": "curobo_motion_gen",
            "version": CUROBO_REQUIRED_VERSION,
            "revision": revision,
            "python": str(self.python_executable),
            "python_sha256": _sha(self.python_executable),
            "root": str(self.curobo_root),
            "device": self.device,
            "world_model": "none",
            "endpoint_solver": "curobo_motion_gen",
            "joint_limits": True,
            "self_collision": True,
        }

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            return dict(self.binding)
        environment = os.environ.copy()
        environment.update(self.environment_overrides or {})
        project_root = Path(__file__).resolve().parents[3]
        source_root = self.curobo_root / "src"
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(project_root), str(source_root), environment.get("PYTHONPATH"),
        )))
        self._process = subprocess.Popen(
            [
                str(self.python_executable), "-m",
                "ora.environment.backends.libero_pro.curobo_worker",
                "--device", self.device,
            ],
            cwd=project_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(self.output_root / "worker.stderr.log").open("a", encoding="utf-8"),
            text=True,
            bufsize=1,
        )
        ready = self._exchange({"schema": CUROBO_REQUEST_SCHEMA, "command": "ready"})
        if ready.get("status") != "ready" or ready.get("version") != CUROBO_REQUIRED_VERSION:
            self.close()
            raise CuroboPlannerError("CuRobo worker readiness identity drift")
        self.binding["worker"] = {
            "pid": self._process.pid,
            "ready": ready,
        }
        return dict(self.binding)

    def _exchange(self, request: Mapping[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise CuroboPlannerError("CuRobo worker is not running")
        request_id = f"curobo-{self._sequence:06d}-{uuid.uuid4().hex[:8]}"
        self._sequence += 1
        payload = {**dict(request), "request_id": request_id}
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + self.request_timeout_s
            received = bytearray()
            while b"\n" not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CuroboPlannerError("CuRobo worker request timed out")
                ready, _, _ = select.select([process.stdout], [], [], remaining)
                if not ready:
                    raise CuroboPlannerError("CuRobo worker request timed out")
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    raise CuroboPlannerError("CuRobo worker closed without a response")
                received.extend(block)
                if len(received) > 32 * 1024 * 1024:
                    raise CuroboPlannerError("CuRobo worker response exceeds its bound")
            line, trailing = bytes(received).split(b"\n", 1)
            if trailing.strip():
                raise CuroboPlannerError("CuRobo worker returned unsolicited response data")
            try:
                response = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise CuroboPlannerError("CuRobo worker returned invalid JSON") from exc
            if not isinstance(response, dict) or response.get("request_id") != request_id:
                raise CuroboPlannerError("CuRobo worker response identity drift")
            return response
        except (CuroboPlannerError, OSError, ValueError) as exc:
            # A timed-out or malformed stream can contain a late response. Never
            # reuse it for another request, even when the process is still alive.
            self._invalidate_worker()
            if isinstance(exc, CuroboPlannerError):
                raise
            raise CuroboPlannerError("CuRobo worker transport failed") from exc

    def _invalidate_worker(self) -> None:
        """Retire only this client's process and pipes after transport failure."""

        process, self._process = self._process, None
        self.binding.pop("worker", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_curobo_request(request)
        self.start()
        response = self._exchange({
            "schema": CUROBO_REQUEST_SCHEMA,
            "command": "plan",
            **validated,
        })
        try:
            response = validate_curobo_response(response)
            if "request" in response and response["request"] != validated:
                raise CuroboPlannerError("CuRobo response request evidence drift")
        except CuroboPlannerError:
            self._invalidate_worker()
            raise
        return response

    def solve_ik(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Read-only IK for exact poses (``curobo_ik.py``); plans and executes nothing."""
        from .curobo_ik import validate_curobo_ik_request, validate_curobo_ik_response

        validated = validate_curobo_ik_request(request)
        self.start()
        response = self._exchange({"schema": CUROBO_REQUEST_SCHEMA, "command": "ik", **validated})
        try:
            return validate_curobo_ik_response(
                {key: value for key, value in response.items() if key != "request_id"}, validated)
        except CuroboPlannerError:
            self._invalidate_worker()
            raise

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({
                    "schema": CUROBO_REQUEST_SCHEMA,
                    "command": "close",
                    "request_id": "close",
                }) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


__all__ = [
    "CUROBO_PLANNER_SCHEMA", "CUROBO_REQUIRED_VERSION", "CUROBO_REQUEST_SCHEMA",
    "CUROBO_RESPONSE_SCHEMA", "CuroboPlannerClient", "CuroboPlannerError",
    "validate_curobo_request", "validate_curobo_response",
]
