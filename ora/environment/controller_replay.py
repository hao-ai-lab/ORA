"""Compact simulator-controller traces for Viewer-side derived replay.

The live Runtime records the numeric commands that crossed the simulator
controller boundary.  It never records rendered replay frames here.  A Viewer
worker may later reset the exact simulator source and seed, re-apply these
commands, and render a derived multi-camera video outside the sealed run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


LEGACY_MANIFEST_SCHEMA = "ora.controller-replay-manifest.v1"
MANIFEST_SCHEMA = "ora.controller-replay-manifest.v2"
JOURNAL_SCHEMA = "ora.controller-replay-events.v1"
ENTRY_SCHEMA = "ora.controller-replay-entry.v1"


class ControllerReplayError(RuntimeError):
    """A controller replay trace is malformed or cannot be persisted."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ControllerReplayError(
        f"controller replay metadata is not canonical JSON: {type(value).__name__}"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _numeric_arrays(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    arrays: dict[str, Any] = {}
    descriptions: dict[str, Any] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ControllerReplayError("controller replay array names must be non-empty")
        if value is None:
            descriptions[name] = None
            continue
        array = np.asarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise ControllerReplayError(
                f"controller replay array {name!r} must be numeric"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ControllerReplayError(
                f"controller replay array {name!r} contains non-finite values"
            )
        arrays[name] = array
        descriptions[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    return arrays, descriptions


@dataclass
class ControllerReplayRecorder:
    """Append one compact replay entry for every submitted world mutation."""

    output_root: Path
    backend: str
    environment_identity: Mapping[str, Any]
    camera_panels: list[str]
    round_index: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    _journal_digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)
    _journal_bytes: int = field(default=0, init=False, repr=False)
    _journal_events: int = field(default=0, init=False, repr=False)
    _failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_root = self.output_root.resolve()
        if self.backend not in {"robotwin", "libero_pro"}:
            raise ControllerReplayError(f"unsupported replay backend: {self.backend}")
        if not self.camera_panels or not all(
            isinstance(item, str) and item for item in self.camera_panels
        ):
            raise ControllerReplayError("controller replay cameras are invalid")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.entries or self.manifest_path.exists() or self.journal_path.exists():
            raise ControllerReplayError("refusing to overwrite an existing controller replay")
        self.journal_path.touch(exist_ok=False)
        self._write_manifest()

    @property
    def root(self) -> Path:
        return self.output_root / "controller-replay"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def journal_path(self) -> Path:
        return self.root / "events.jsonl"

    def _append_event(self, event: Mapping[str, Any]) -> None:
        """Publish one event and then its bounded, hash-bound commit pointer.

        A crash before the atomic pointer replacement leaves the old prefix
        readable. Never continue writing after a failed commit: the simulator
        and the durable evidence may no longer agree about that operation.
        """
        if self._failed:
            raise ControllerReplayError("controller replay writer has a failed commit")
        payload = (json.dumps(_jsonable({"seq": self._journal_events, **event}),
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        try:
            if self.journal_path.stat().st_size != self._journal_bytes:
                raise ControllerReplayError("controller replay journal size changed")
            with self.journal_path.open("ab") as stream:
                stream.write(payload)
            self._journal_digest.update(payload)
            self._journal_bytes += len(payload)
            self._journal_events += 1
            self._write_manifest()
        except Exception:
            self._failed = True
            raise

    def _write_manifest(self) -> None:
        if self._failed:
            raise ControllerReplayError("controller replay writer has a failed commit")
        try:
            _atomic_json(
                self.manifest_path,
                {
                    "schema": MANIFEST_SCHEMA,
                    "backend": self.backend,
                    "environment_identity": dict(self.environment_identity),
                    "camera_panels": list(self.camera_panels),
                    "round_index": self.round_index,
                    "journal": {
                        "schema": JOURNAL_SCHEMA,
                        "path": "events.jsonl",
                        "bytes": self._journal_bytes,
                        "events": self._journal_events,
                        "sha256": self._journal_digest.hexdigest(),
                    },
                    "default_video_source": "viewer_derived_controller_replay",
                    "runtime_media_capture": False,
                    "fidelity": {
                        "input": "recorded_controller_targets",
                        "expected": "same_source_seed_near_deterministic",
                        "not_claimed": [
                            "byte_exact_physics_reproduction",
                            "original_visual_evidence",
                            "official_verdict_reproduction",
                        ],
                    },
                    "authority": "control_only_not_policy_visible",
                },
            )
        except Exception:
            self._failed = True
            raise

    def begin_round(self, round_index: int) -> None:
        if self._failed:
            raise ControllerReplayError("controller replay writer has a failed commit")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ControllerReplayError("round index must be a non-negative integer")
        self.round_index = round_index
        self._write_manifest()

    def record(
        self,
        *,
        action_sequence: int,
        action: Mapping[str, Any],
        controller: str,
        arrays: Mapping[str, Any],
        physics_steps: int,
        replayable: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if self._failed:
            raise ControllerReplayError("controller replay writer has a failed commit")
        if type(action_sequence) is not int or action_sequence != len(self.entries):
            raise ControllerReplayError(
                "controller replay action sequence must be append-only and contiguous"
            )
        action_id = action.get("action_id")
        action_name = action.get("action")
        if not isinstance(action_id, str) or not action_id:
            raise ControllerReplayError("controller replay action id is missing")
        if not isinstance(action_name, str) or not action_name:
            raise ControllerReplayError("controller replay action name is missing")
        if not isinstance(controller, str) or not controller:
            raise ControllerReplayError("controller replay controller is missing")
        if (
            isinstance(physics_steps, bool)
            or not isinstance(physics_steps, int)
            or physics_steps < 0
        ):
            raise ControllerReplayError("controller replay physics steps are invalid")
        numeric, descriptions = _numeric_arrays(arrays)
        artifact = None
        if numeric:
            path = self.root / f"action-{action_sequence:04d}.npz"
            if path.exists():
                raise ControllerReplayError(f"refusing to overwrite replay trace: {path}")
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **numeric)
            os.replace(temporary, path)
            artifact = {
                "path": path.relative_to(self.output_root).as_posix(),
                "sha256": _file_hash(path),
                "bytes": path.stat().st_size,
            }
        entry = {
            "schema": ENTRY_SCHEMA,
            "action_sequence": action_sequence,
            "round_index": self.round_index,
            "action_id": action_id,
            "action": action_name,
            "controller": controller,
            "physics_steps": physics_steps,
            "replayable": bool(replayable and artifact is not None),
            "arrays": descriptions,
            "artifact": artifact,
            "semantic_action": _jsonable(action),
            "metadata": _jsonable(dict(metadata or {})),
        }
        self._append_event({"type": "record", "entry": entry})
        self.entries.append(entry)
        return dict(entry)

    def update(
        self, action_sequence: int, *, execution_status: str,
        settle_steps: int = 0, execution_fault: str | None = None,
    ) -> None:
        if self._failed:
            raise ControllerReplayError("controller replay writer has a failed commit")
        if type(action_sequence) is not int or not 0 <= action_sequence < len(self.entries):
            raise ControllerReplayError("controller replay entry does not exist")
        if not isinstance(execution_status, str) or not execution_status:
            raise ControllerReplayError("controller replay execution status is missing")
        if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps < 0:
            raise ControllerReplayError("controller replay settle steps are invalid")
        if execution_fault is not None and not isinstance(execution_fault, str):
            raise ControllerReplayError("controller replay execution fault is invalid")
        execution = {
            "status": execution_status,
            "settle_steps": settle_steps,
            "fault": execution_fault,
        }
        self._append_event({"type": "update", "action_sequence": action_sequence,
                            "execution": execution})
        self.entries[action_sequence]["execution"] = execution


def _read_journal(path: Path, reference: Any) -> list[dict[str, Any]]:
    if (not isinstance(reference, dict) or reference.get("schema") != JOURNAL_SCHEMA
            or reference.get("path") != "events.jsonl"):
        raise ControllerReplayError("invalid controller replay journal reference")
    length, count = reference.get("bytes"), reference.get("events")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (length, count)):
        raise ControllerReplayError("invalid controller replay journal bounds")
    journal = path.parent / "events.jsonl"
    if journal.is_symlink() or not journal.is_file() or journal.stat().st_size < length:
        raise ControllerReplayError("controller replay journal is unavailable or truncated")
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    remaining = length
    with journal.open("rb") as stream:
        for sequence in range(count):
            line = stream.readline(remaining + 1)
            if not line.endswith(b"\n") or len(line) > remaining:
                raise ControllerReplayError("controller replay journal commit is incomplete")
            remaining -= len(line)
            digest.update(line)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ControllerReplayError("invalid controller replay journal JSON") from exc
            if (not isinstance(event, dict) or type(event.get("seq")) is not int
                    or event["seq"] != sequence):
                raise ControllerReplayError("controller replay journal event order differs")
            if event.get("type") == "record":
                entry = event.get("entry")
                if (not isinstance(entry, dict) or type(entry.get("action_sequence")) is not int
                        or entry["action_sequence"] != len(entries)):
                    raise ControllerReplayError("controller replay entries are not contiguous")
                entries.append(entry)
            elif event.get("type") == "update":
                index, execution = event.get("action_sequence"), event.get("execution")
                if (type(index) is not int or not 0 <= index < len(entries)
                        or not isinstance(execution, dict)
                        or set(execution) != {"status", "settle_steps", "fault"}
                        or not isinstance(execution["status"], str) or not execution["status"]
                        or type(execution["settle_steps"]) is not int or execution["settle_steps"] < 0
                        or (execution["fault"] is not None and not isinstance(execution["fault"], str))):
                    raise ControllerReplayError("invalid controller replay execution update")
                entries[index]["execution"] = execution
            else:
                raise ControllerReplayError("unknown controller replay journal event")
    if remaining or digest.hexdigest() != reference.get("sha256"):
        raise ControllerReplayError("controller replay journal hash/count drifted")
    return entries


def load_controller_replay_manifest(path: Path) -> dict[str, Any]:
    """Load v1 inline entries or reconstruct the committed v2 journal prefix.

    Uncommitted trailing journal bytes are ignored; all committed bytes, numeric
    artifacts and attempt journals are verified before any replay.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in {MANIFEST_SCHEMA, LEGACY_MANIFEST_SCHEMA}:
        raise ControllerReplayError("invalid controller replay manifest")
    if value["schema"] == MANIFEST_SCHEMA:
        if "entries" in value:
            raise ControllerReplayError("v2 controller replay cannot contain inline entries")
        value["entries"] = _read_journal(path, value.get("journal"))
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ControllerReplayError("controller replay entries are missing")
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, Mapping)
            or entry.get("schema") != ENTRY_SCHEMA
            or entry.get("action_sequence") != index
        ):
            raise ControllerReplayError("controller replay entries are not contiguous")
        attempt = entry.get("metadata", {}).get("attempt_journal")
        if attempt is not None:
            root = path.parent.parent.resolve()
            if not isinstance(attempt, Mapping) or not isinstance(attempt.get("path"), str):
                raise ControllerReplayError("controller attempt journal is malformed")
            target = (root / attempt["path"]).resolve()
            if not target.is_relative_to(root) or not target.is_file() or _file_hash(target) != attempt.get("sha256"):
                raise ControllerReplayError("controller attempt journal is unavailable or changed")
        artifact = entry.get("artifact")
        if artifact is None:
            continue
        relative = artifact.get("path") if isinstance(artifact, Mapping) else None
        if not isinstance(relative, str):
            raise ControllerReplayError("controller replay artifact is malformed")
        target = (path.parent.parent / relative).resolve()
        root = path.parent.parent.resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ControllerReplayError("controller replay artifact is unavailable")
        if _file_hash(target) != artifact.get("sha256"):
            raise ControllerReplayError("controller replay artifact hash drifted")
    return value


def load_controller_replay_arrays(
    manifest_path: Path, entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one hash-verified numeric controller artifact.

    This is intentionally simulator-neutral.  Viewer workers and trusted
    source-bound qualification may interpret the recorded channel structure,
    but neither is allowed to bypass the manifest's path and digest binding.
    """

    import numpy as np

    artifact = entry.get("artifact")
    if artifact is None:
        return {}
    relative = artifact.get("path") if isinstance(artifact, Mapping) else None
    if not isinstance(relative, str):
        raise ControllerReplayError("controller replay artifact is malformed")
    root = manifest_path.parent.parent.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ControllerReplayError("controller replay artifact is unavailable")
    if _file_hash(target) != artifact.get("sha256"):
        raise ControllerReplayError("controller replay artifact hash drifted")
    with np.load(target, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def materialize_controller_sequence(
    entry: Mapping[str, Any], arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a recorded dense-controller channel map without executing it."""

    metadata = entry.get("metadata")
    structure = (
        metadata.get("control_sequence") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(structure, Mapping):
        raise ControllerReplayError("controller replay control structure is unavailable")
    result: dict[str, Any] = {}
    for channel, value in structure.items():
        if value is None:
            result[str(channel)] = None
            continue
        if not isinstance(value, Mapping):
            raise ControllerReplayError("controller replay channel is malformed")
        compiled: dict[str, Any] = {}
        for field, item in value.items():
            if isinstance(item, Mapping) and isinstance(item.get("array"), str):
                key = str(item["array"])
                if key not in arrays:
                    raise ControllerReplayError(
                        f"controller replay array is missing: {key}"
                    )
                compiled[str(field)] = arrays[key]
            else:
                compiled[str(field)] = item
        result[str(channel)] = compiled
    return result


__all__ = [
    "ControllerReplayError",
    "ControllerReplayRecorder",
    "ENTRY_SCHEMA",
    "MANIFEST_SCHEMA",
    "LEGACY_MANIFEST_SCHEMA",
    "load_controller_replay_arrays",
    "load_controller_replay_manifest",
    "materialize_controller_sequence",
]
