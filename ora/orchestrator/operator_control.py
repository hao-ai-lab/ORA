"""Run-bound ORA operator commands, outside immutable run evidence.

The UI writes atomic, immutable inbox files. Only the Supervisor consumes them
at a Native model-decision boundary and records their delivery in run evidence.
This module never discovers, reads or changes a Runtime lease, signals a process
or writes to the run directory.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import paths


MODE = "file_inbox_v1"
MAX_COMMAND_BYTES = 65536
_TERMINAL = {"completed", "incomplete", "invalid", "blocked"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class OperatorControlError(RuntimeError):
    """An operator command does not belong to a writable interactive session."""


def intervention_reasons(plan: Mapping[str, Any], result: Mapping[str, Any], events: list[Mapping[str, Any]]) -> list[str]:
    """Reconcile persistent-context contamination with immutable delivery events."""

    deliveries = [event.get("payload", {}) for event in events if event.get("type") == "operator_message"]
    if "operator_control" not in plan and not deliveries:
        return []
    if "operator_control" not in plan or any(not isinstance(item, Mapping) for item in deliveries):
        return ["operator_intervention_metadata_mismatch"]
    count = len(deliveries)
    if (
        result.get("human_intervention") is not bool(count)
        or type(result.get("operator_message_count")) is not int
        or result.get("operator_message_count") != count
        or (count and result.get("behavioral_sample_valid") is not False)
    ):
        return ["operator_intervention_metadata_mismatch"]
    rounds = result.get("rounds", [])
    for record in rounds if isinstance(rounds, list) else []:
        if not isinstance(record, Mapping) or type(record.get("round_index")) is not int:
            continue  # The ordinary round-contract checker owns this error.
        cumulative = sum(type(item.get("round_index")) is int and item["round_index"] <= record["round_index"] for item in deliveries)
        if (
            record.get("human_intervention") is not bool(cumulative)
            or type(record.get("operator_message_count")) is not int
            or record.get("operator_message_count") != cumulative
            or (cumulative and record.get("behavioral_sample_valid") is not False)
        ):
            return ["operator_intervention_metadata_mismatch"]
    return []


def _read_json(path: Path, *, max_bytes: int = MAX_COMMAND_BYTES) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise OperatorControlError(f"not a bounded regular control file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OperatorControlError(f"invalid control JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise OperatorControlError(f"control JSON must be an object: {path.name}")
    return value


def _create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Publish a complete file once; never replace an existing command/ack."""

    data = (json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / f".{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise OperatorControlError("command/run id must be a bounded stable identifier")
    return value


def normalize_operator_control(value: Any, *, experiment: str, policy_kind: str) -> dict[str, Any]:
    """The only interactive configuration; initial follow-ups are plan-bound."""

    if (
        not isinstance(value, Mapping)
        or set(value) - {"mode", "initial_messages"}
        or value.get("mode") != MODE
        or experiment != "ora-interactive"
        or policy_kind != "native_api"
    ):
        raise OperatorControlError("operator_control requires mode=file_inbox_v1, experiment=ora-interactive and policy.kind=native_api")
    initial = value.get("initial_messages", [])
    if not isinstance(initial, list) or len(initial) > 64:
        raise OperatorControlError("operator_control.initial_messages must be a bounded list")
    messages = []
    seen = set()
    for raw in initial:
        if not isinstance(raw, Mapping) or raw.get("type") != "follow_up":
            raise OperatorControlError("initial operator messages must have type=follow_up")
        message = _command(raw, allow_follow_up=True)
        if message["id"] in seen:
            raise OperatorControlError("initial operator message ids must be unique")
        seen.add(message["id"])
        messages.append(message)
    return {"mode": MODE, **({"initial_messages": messages} if messages else {})}


def _enabled(plan: Mapping[str, Any]) -> None:
    normalized = normalize_operator_control(
        plan.get("operator_control"), experiment=plan.get("experiment", ""),
        policy_kind=plan.get("policy", {}).get("kind", ""),
    )
    if normalized != plan["operator_control"]:
        raise OperatorControlError("persisted operator control is not normalized")


def _binding(run_root: Path, plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    _enabled(plan)
    if not isinstance(plan.get("plan_sha256"), str) or not plan["plan_sha256"]:
        raise OperatorControlError("operator control requires an exact plan hash")
    return {
        "schema": "ora.ora-operator-binding.v1",
        "run_id": _identifier(run_id),
        "run_root": str(run_root.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "mode": MODE,
        "supervisor_node": socket.gethostname(),
    }


def _root(run_id: str) -> Path:
    base = paths.STATE_ROOT.resolve() / "ora-interactive"
    result = base / _identifier(run_id)
    if base.is_symlink() or result.is_symlink():
        raise OperatorControlError("operator control directory cannot be a symbolic link")
    return result


def _require_open(run_root: Path, root: Path) -> None:
    if (run_root / "seal.json").exists() or (root / "closed.json").exists():
        raise OperatorControlError("interactive run is closed or sealed")
    status_path = run_root / "status.json"
    if status_path.exists() and _read_json(status_path).get("state") in _TERMINAL:
        raise OperatorControlError("interactive run is terminal")


def _load_control(run_root: Path, *, require_open: bool) -> FileOperatorControl:
    run_root = run_root.resolve()
    plan = _read_json(run_root / "plan.json", max_bytes=16 * 1024 * 1024)
    _enabled(plan)
    # Native run directories have the exact run id as their basename. Tests
    # and embedded callers can also use status.json's explicit identity.
    status_path = run_root / "status.json"
    run_id = (_read_json(status_path).get("run_id") if status_path.exists() else None) or run_root.name
    root = _root(run_id)
    binding = _binding(run_root, plan, run_id)
    if _read_json(root / "binding.json") != binding:
        raise OperatorControlError("operator control does not match this exact run and plan")
    if require_open:
        _require_open(run_root, root)
    return FileOperatorControl(run_root, root, binding, tuple(plan["operator_control"].get("initial_messages", ())))


def initialize_control(run_root: Path, plan: Mapping[str, Any], run_id: str) -> FileOperatorControl:
    """Initialize only the external, exact-run inbox; called by its Supervisor."""

    run_root = run_root.resolve()
    binding = _binding(run_root, plan, run_id)
    root = _root(run_id)
    if root.is_relative_to(run_root):
        raise OperatorControlError("operator inbox must remain outside sealed run evidence")
    _require_open(run_root, root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("inbox", "acknowledgements"):
        directory = root / name
        if directory.is_symlink():
            raise OperatorControlError("operator inbox directory cannot be a symbolic link")
        directory.mkdir(exist_ok=True)
    if not _create_json(root / "binding.json", binding) and _read_json(root / "binding.json") != binding:
        raise OperatorControlError("operator inbox is bound to another run or plan")
    control = FileOperatorControl(run_root, root, binding, tuple(plan["operator_control"].get("initial_messages", ())))
    for message in control.initial_messages:
        _queue(control, message)
    return control


def _command(value: Mapping[str, Any], *, allow_follow_up: bool = False) -> dict[str, str]:
    types = {"steer", "follow_up"} if allow_follow_up else {"steer"}
    if set(value) != {"id", "type", "message"} or value.get("type") not in types:
        raise OperatorControlError("operator inbox accepts id/type=steer/message only; follow_up requires a new plan")
    message = value.get("message")
    if not isinstance(message, str) or not message.strip():
        raise OperatorControlError("steer message must be non-empty text")
    result = {"id": _identifier(value.get("id")), "type": value["type"], "message": message}
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_COMMAND_BYTES - 2048:
        raise OperatorControlError("steer message exceeds control size limit")
    return result


def submit_command(run_root: Path, command: Mapping[str, Any]) -> dict[str, Any]:
    """Queue one steer without writing run evidence; repeated ids are idempotent."""

    control = _load_control(run_root, require_open=True)
    normalized = _command(command)
    record = _queue(control, normalized)
    try:
        _require_open(control.run_root, control.root)
    except OperatorControlError:
        # The Supervisor can finish between the open check and atomic publish.
        # Never leave a command looking deliverable after that race. A delivery
        # already acknowledged before close remains an accepted command.
        ack = control.root / "acknowledgements" / f"{record['id']}.json"
        if ack.exists() and _read_json(ack).get("status") == "applied":
            return record
        control.acknowledge(record, "rejected", reason="session_closed_before_delivery")
        raise
    return record


def _queue(control: FileOperatorControl, normalized: Mapping[str, str]) -> dict[str, Any]:
    path = control.root / "inbox" / f"{normalized['id']}.json"
    record = {
        "schema": "ora.ora-operator-command.v1",
        **normalized,
        "run_id": control.binding["run_id"],
        "plan_sha256": control.binding["plan_sha256"],
        "submitted_unix_ns": time.time_ns(),
    }
    if not _create_json(path, record):
        existing = _read_json(path)
        if any(existing.get(key) != value for key, value in record.items() if key != "submitted_unix_ns"):
            raise OperatorControlError("command id already exists with different contents")
        record = existing
    return record


def command_status(run_root: Path) -> dict[str, Any]:
    """Read the bound queue and acknowledgements without changing any file."""

    control = _load_control(run_root, require_open=False)
    acknowledged = [_read_json(path) for path in sorted((control.root / "acknowledgements").glob("*.json"))]
    completed = {item["id"] for item in acknowledged}
    pending = [item for item in control.commands() if item["id"] not in completed]
    return {
        "binding": control.binding,
        "closed": (control.root / "closed.json").exists(),
        "pending": pending,
        "acknowledgements": acknowledged,
    }


@dataclass(frozen=True)
class FileOperatorControl:
    run_root: Path
    root: Path
    binding: Mapping[str, Any]
    initial_messages: tuple[Mapping[str, str], ...] = ()

    def commands(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted((self.root / "inbox").glob("*.json")):
            record = _read_json(path)
            command = _command({key: record.get(key) for key in ("id", "type", "message")}, allow_follow_up=True)
            if (
                set(record) != {"schema", "id", "type", "message", "run_id", "plan_sha256", "submitted_unix_ns"}
                or record["schema"] != "ora.ora-operator-command.v1"
                or record["id"] != path.stem
                or record["run_id"] != self.binding["run_id"]
                or record["plan_sha256"] != self.binding["plan_sha256"]
                or type(record["submitted_unix_ns"]) is not int
            ):
                raise OperatorControlError("operator command identity does not match the run")
            if command["type"] == "follow_up" and command not in self.initial_messages:
                raise OperatorControlError("follow-up command is not bound in the RunPlan")
            records.append(record)
        return sorted(records, key=lambda item: (item["submitted_unix_ns"], item["id"]))

    def poll(self) -> list[dict[str, Any]]:
        _require_open(self.run_root, self.root)
        return [item for item in self.commands() if not (self.root / "acknowledgements" / f"{item['id']}.json").exists()]

    def acknowledge(self, command: Mapping[str, Any], status: str, **details: Any) -> dict[str, Any]:
        if status not in {"applied", "rejected"}:
            raise OperatorControlError("unknown operator acknowledgement status")
        record = {
            "schema": "ora.ora-operator-acknowledgement.v1",
            "id": _identifier(command["id"]),
            "run_id": self.binding["run_id"],
            "plan_sha256": self.binding["plan_sha256"],
            "status": status,
            "acknowledged_unix_ns": time.time_ns(),
            **details,
        }
        path = self.root / "acknowledgements" / f"{record['id']}.json"
        if not _create_json(path, record):
            return _read_json(path)
        return record

    def close(self) -> None:
        _create_json(self.root / "closed.json", {**self.binding, "closed_unix_ns": time.time_ns()})
        for command in self.commands():
            self.acknowledge(command, "rejected", reason="session_closed_before_delivery")
