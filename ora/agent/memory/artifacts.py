"""Session-scoped, policy-readable references for recoverable Context payloads."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from ora.agent.composition.tool_surface import validate_schema_instance
from ora.agent.tools.robot import ToolError
from .index_persistence import BatchedIndexPersistence, atomic_json
from .artifact_openers import ARTIFACT_READ_NAMES, artifact_read_name, recoverable_artifact_open, public_artifact_ref
from .artifact_reader import (
    RefHashError, RefScopeError, RefStorageError, RefSyntaxError, RefUnknownError, bounded_pointer_value,
    check_observe_ref_arguments, check_read_arguments, json_pointer, policy_resolve, read_range,
)


LOSSLESS_CONTEXT_RETENTION = "action_step_lossless_v4"
CONTEXT_ARTIFACT_INDEX_SCHEMA = "ora.context-artifact-index.v1"
CONTEXT_ARTIFACT_REF_SCHEMA = "ora.context-artifact-ref.v1"
RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA = (
    "ora.recoverable-artifact-envelope.v1"
)
RECOVERABLE_ARTIFACT_POINTER_SCHEMA = (
    "ora.recoverable-artifact-pointer.v1"
)
RECOVERABLE_ARTIFACT_TAG = "recoverable_artifact"
ARTIFACT_READ_LIMIT_BYTES = 8192
_REF_PATTERN = re.compile(
    r"artifact://(?P<scope>[A-Za-z0-9][A-Za-z0-9._-]{0,159})/"
    r"(?P<sha256>[0-9a-f]{64})"
)



def recoverable_artifact_envelope(
    reference: Mapping[str, Any], *, source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode one policy-visible, path-free recovery instruction."""

    ref = reference.get("ref")
    sha256 = reference.get("sha256")
    media_type = reference.get("media_type")
    byte_count = reference.get("bytes")
    policy_scope = reference.get("policy_scope")
    kind = reference.get("kind")
    if (
        not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(policy_scope, str)
        or not isinstance(kind, str)
        or ref != f"artifact://{policy_scope}/{sha256}"
    ):
        raise ValueError("recoverable Artifact reference is invalid")
    envelope = {
        "schema": RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
        "artifact": {
            "ref": ref,
            "sha256": sha256,
            "kind": kind,
            "media_type": media_type,
            "bytes": byte_count,
            "policy_scope": policy_scope,
            "round": reference.get("round"),
            "action_step": reference.get("action_step"),
        },
        "source": copy.deepcopy(dict(source or {})),
        "open": recoverable_artifact_open(reference),
    }
    validate_recoverable_artifact_envelope(envelope)
    return envelope


def validate_recoverable_artifact_envelope(
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on drift between the reference and its advertised decoder."""

    if set(envelope) != {"schema", "artifact", "source", "open"}:
        raise ValueError("recoverable Artifact envelope fields are invalid")
    artifact = envelope.get("artifact")
    source = envelope.get("source")
    opener = envelope.get("open")
    if (
        envelope.get("schema") != RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA
        or not isinstance(artifact, Mapping)
        or set(artifact) != {
            "ref", "sha256", "kind", "media_type", "bytes", "policy_scope",
            "round", "action_step",
        }
        or not isinstance(source, Mapping)
        or not isinstance(opener, Mapping)
        or set(opener) != {"tool", "arguments"}
        or not isinstance(opener.get("arguments"), Mapping)
        or set(opener["arguments"]) != {"ref"}
    ):
        raise ValueError("recoverable Artifact envelope is malformed")
    ref = artifact.get("ref")
    sha256 = artifact.get("sha256")
    media_type = artifact.get("media_type")
    byte_count = artifact.get("bytes")
    scope = artifact.get("policy_scope")
    if (
        not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(artifact.get("kind"), str)
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(scope, str)
        or ref != f"artifact://{scope}/{sha256}"
        or not isinstance(opener.get("tool"), str)
        or opener.get("tool") not in (
            {"observe"} if media_type.startswith("image/") else ARTIFACT_READ_NAMES
        )
        or opener["arguments"].get("ref") != ref
    ):
        raise ValueError("recoverable Artifact envelope identity drift")
    for field in ("round", "action_step"):
        value = artifact.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"recoverable Artifact {field} is invalid")
    return copy.deepcopy(dict(envelope))


def recoverable_artifact_pointer(
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode only non-derivable recovery fields for one Artifact."""

    ref = reference.get("ref")
    sha256 = reference.get("sha256")
    policy_scope = reference.get("policy_scope")
    media_type = reference.get("media_type")
    byte_count = reference.get("bytes")
    match = _REF_PATTERN.fullmatch(ref) if isinstance(ref, str) else None
    if (
        not isinstance(ref, str)
        or match is None
        or sha256 != match.group("sha256")
        or policy_scope != match.group("scope")
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ValueError("recoverable Artifact pointer source is invalid")
    pointer = {
        "v": 1,
        "ref": ref,
        "mime": media_type,
        "bytes": byte_count,
        "at": [reference.get("round"), reference.get("action_step")],
        "open": recoverable_artifact_open(reference)["tool"],
    }
    return validate_recoverable_artifact_pointer(pointer)


def validate_recoverable_artifact_pointer(
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on one compact, directly callable Artifact pointer."""

    if set(pointer) != {"v", "ref", "mime", "bytes", "at", "open"}:
        raise ValueError("recoverable Artifact pointer fields are invalid")
    ref = pointer.get("ref")
    media_type = pointer.get("mime")
    byte_count = pointer.get("bytes")
    location = pointer.get("at")
    opener = pointer.get("open")
    if (
        pointer.get("v") != 1
        or not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(location, list)
        or len(location) != 2
        or not isinstance(opener, str)
        or opener not in (
            {"observe"} if media_type.startswith("image/") else ARTIFACT_READ_NAMES
        )
    ):
        raise ValueError("recoverable Artifact pointer is malformed")
    for field, value in zip(("round", "action_step"), location, strict=True):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"recoverable Artifact pointer {field} is invalid")
    return copy.deepcopy(dict(pointer))


def _pointer_from_marker_value(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") == RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA:
        envelope = validate_recoverable_artifact_envelope(value)
        return recoverable_artifact_pointer({
            **envelope["artifact"], "open": envelope["open"],
        })
    return validate_recoverable_artifact_pointer(value)


def encode_recoverable_artifact_marker(value: Mapping[str, Any]) -> str:
    validated = _pointer_from_marker_value(value)
    return (
        f"<{RECOVERABLE_ARTIFACT_TAG}>"
        + json.dumps(
            validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        + f"</{RECOVERABLE_ARTIFACT_TAG}>"
    )


def decode_recoverable_artifact_marker(value: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("recoverable Artifact marker must be text")
    opening = f"<{RECOVERABLE_ARTIFACT_TAG}>"
    closing = f"</{RECOVERABLE_ARTIFACT_TAG}>"
    if not value.startswith(opening) or not value.endswith(closing):
        raise ValueError("recoverable Artifact marker framing is invalid")
    body = value[len(opening):-len(closing)]
    try:
        pointer = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("recoverable Artifact marker JSON is invalid") from exc
    if not isinstance(pointer, Mapping):
        raise ValueError("recoverable Artifact marker must contain an object")
    return _pointer_from_marker_value(pointer)


ARTIFACT_READ_TOOL: dict[str, Any] = {
    "name": "artifact.read",
    "description": (
        "Retrieve exact text or JSON evidence from this Session's artifact store. "
        "Use when: exact historical text, JSON, a state dump, receipt, or diagnostic "
        "was replaced by an artifact:// reference. Effect: read-only and does not "
        "change the robot or Memory. Returns: one bounded byte range, or one JSON "
        "Pointer value, together with the verified content identity and continuation "
        "offset. Does not: expose host paths, cross Session scope, bypass artifact "
        "hash checks, or return images; reopen image references with observe(ref=...). "
        "On error: correct the reference/range or re-ground from current evidence; "
        "missing, tampered, wrong-media, and cross-scope references fail closed."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "minLength": 1,
                    "description": "Exact Session-scoped artifact:// reference supplied by a recoverable-artifact pointer."},
            "offset": {"type": ["integer", "null"], "minimum": 0,
                       "description": "Starting UTF-8 byte offset for a text range; omit or pass null to start at zero."},
            "limit": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": ARTIFACT_READ_LIMIT_BYTES,
                "description": "Maximum bytes to read in one text range; omit or pass null for the bounded default.",
            },
            "json_pointer": {"type": ["string", "null"],
                             "description": "JSON Pointer selecting one value from a JSON artifact; do not combine with a byte range. Omit or pass null for byte-range reading."},
        },
        "required": ["ref"],
        "additionalProperties": False,
    },
}

# Keep the mutually exclusive modes explicit on the model-visible schema.
# Null/default range values retain the existing decoder's accepted semantics.
_ARTIFACT_READ_PROPERTIES = ARTIFACT_READ_TOOL["inputSchema"]["properties"]
ARTIFACT_READ_TOOL["inputSchema"]["oneOf"] = [
    {
        "type": "object",
        "properties": {
            **copy.deepcopy(_ARTIFACT_READ_PROPERTIES),
            "json_pointer": {
                "type": "null", "description": "Omit or pass null for byte-range reading.",
            },
        },
        "required": ["ref"],
        "additionalProperties": False,
    },
    {
        "type": "object",
        "properties": {
            "ref": copy.deepcopy(_ARTIFACT_READ_PROPERTIES["ref"]),
            "json_pointer": {
                "type": "string", "description": "Existing JSON Pointer selecting one JSON value.",
            },
            "offset": {
                "enum": [None, 0], "description": "Omit in Pointer mode; null or the default zero is accepted.",
            },
            "limit": {
                "enum": [None, 4096], "description": "Omit in Pointer mode; null or the default 4096 is accepted.",
            },
        },
        "required": ["ref", "json_pointer"],
        "additionalProperties": False,
    },
]


ARTIFACT_READ_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "ok"},
        "ref": {
            "type": "string",
            "pattern": (
                r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/"
                r"[0-9a-f]{64}$"
            ),
        },
        "sha256": {"type": "string"},
        "media_type": {"type": "string"},
        "bytes": {"type": "integer"},
        "offset": {"type": "integer"},
        "next_offset": {"type": "integer"},
        "eof": {"type": "boolean"},
        "content": {"type": ["string", "null"]},
        "json_value": {},
    },
    "required": [
        "status", "ref", "sha256", "media_type", "bytes", "offset",
        "next_offset", "eof", "content", "json_value",
    ],
    "additionalProperties": False,
}


HISTORICAL_OBSERVE_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "actual_historical_artifact"},
        "ref": {"type": "string"},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "media_type": {"type": "string", "pattern": r"^image/"},
        "bytes": {"type": "integer", "minimum": 0},
        "round": {"type": ["integer", "null"], "minimum": 0},
        "action_step": {"type": ["integer", "null"], "minimum": 0},
        "views": {
            "type": "array",
            "items": {"type": "string", "const": "historical_artifact"},
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": [
        "status", "ref", "sha256", "media_type", "bytes", "round",
        "action_step", "views",
    ],
    "additionalProperties": False,
}


CURRENT_OBSERVE_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "actual"},
        "views": {"type": "array", "items": {"type": "string"}},
        "view_dimensions": {"type": "object"},
    },
    "required": ["status", "views"],
    "additionalProperties": False,
}



def recoverable_artifact_definitions(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add the orthogonal Context reader without changing a frozen ToolSurface."""

    values = [copy.deepcopy(dict(item)) for item in definitions]
    observe = next((item for item in values if item.get("name") == "observe"), None)
    if not isinstance(observe, dict):
        raise ValueError("lossless Context requires an observe tool")
    schema = observe.get("inputSchema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping):
        raise ValueError("observe input schema is unavailable")
    current_schema = copy.deepcopy(dict(schema))
    ref_schema = {
        "type": "string",
        "description": (
            "Exact Session-scoped artifact:// reference from a "
            "<recoverable_artifact> pointer."
        ),
        "pattern": (
            r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/"
            r"[0-9a-f]{64}$"
        ),
    }
    historical_schema = {
        "type": "object",
        "properties": {"ref": copy.deepcopy(ref_schema)},
        "required": ["ref"],
        "additionalProperties": False,
    }
    root_properties = copy.deepcopy(dict(properties))
    root_properties["ref"] = copy.deepcopy(ref_schema)
    observe["inputSchema"] = {
        # Provider tool protocols require a top-level object declaration. The
        # branches remain authoritative for the two mutually exclusive modes.
        "type": "object",
        "properties": root_properties,
        "oneOf": [current_schema, historical_schema],
        "additionalProperties": False,
    }
    observe["description"] = (
        str(observe.get("description", "")).rstrip()
        + " A <recoverable_artifact> pointer gives the exact historical mode: "
        "pass its ref by itself to the named open tool to reopen that image. "
        "The Runtime verifies Session scope, media type and SHA-256 before returning it."
    )
    reader = copy.deepcopy(ARTIFACT_READ_TOOL)
    reader["name"] = artifact_read_name(definitions)
    values.append(reader)
    return values


def recoverable_artifact_result_schemas(
    result_schemas: Mapping[str, Mapping[str, Any]],
    *, reader_name: str | None = None,
) -> dict[str, Any]:
    values = copy.deepcopy(dict(result_schemas))
    observe = values.get("observe")
    if observe is None:
        observe = {"oneOf": [copy.deepcopy(CURRENT_OBSERVE_RESULT_SCHEMA)]}
        values["observe"] = observe
    if not isinstance(observe, dict):
        raise ValueError("observe result schema must be an object")
    branches = observe.get("oneOf")
    if not isinstance(branches, list):
        metadata = {
            key: copy.deepcopy(observe[key])
            for key in ("$schema", "$defs") if key in observe
        }
        current = {
            key: copy.deepcopy(value) for key, value in observe.items()
            if key not in metadata
        }
        observe = {**metadata, "oneOf": [current]}
        values["observe"] = observe
        branches = observe["oneOf"]
    branches.append(copy.deepcopy(HISTORICAL_OBSERVE_RESULT_SCHEMA))
    # Keyed the way this surface spells its tools; the schemas map is keyed by
    # advertised name and must agree with the definition appended beside it.
    values[reader_name or artifact_read_name([{"name": key} for key in result_schemas])] = (
        copy.deepcopy(ARTIFACT_READ_RESULT_SCHEMA)
    )
    return values


_atomic_json = atomic_json  # historical private name


@dataclass
class ContextArtifactCatalog(BatchedIndexPersistence):
    """Bind storage refs to stable logical refs within one PolicySession."""

    run_root: Path
    policy_session_id: str
    artifacts: ArtifactStore
    reader_name: str = "artifact.read"

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        if self.reader_name not in ARTIFACT_READ_NAMES:
            raise ValueError("unsupported Context Artifact reader name")
        if _REF_PATTERN.fullmatch(
            f"artifact://{self.policy_session_id}/" + "0" * 64
        ) is None:
            raise ValueError("policy_session_id cannot form an artifact scope")
        self.index_path = self.run_root / "native/context-artifacts/index.json"
        self._entries: dict[str, dict[str, Any]] = {}
        self._persist()

    def _write_index(self) -> None:
        # Serialization never mutates the entries and sort_keys fixes their
        # order, so no defensive copy is needed.
        atomic_json(self.index_path, {
            "schema": CONTEXT_ARTIFACT_INDEX_SCHEMA,
            "policy_session_id": self.policy_session_id,
            "artifacts": self._entries,
        }, durability=getattr(self.artifacts, "durability", None))

    def register_storage_ref(
        self,
        storage_ref: Mapping[str, Any],
        *,
        kind: str,
        round_index: int | None,
        action_step: int | None,
    ) -> dict[str, Any]:
        for label, value in (
            ("round_index", round_index), ("action_step", action_step),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"artifact {label} is invalid")
        digest = storage_ref.get("sha256")
        relative = storage_ref.get("path")
        media_type = storage_ref.get("media_type")
        byte_count = storage_ref.get("bytes")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(relative, str)
            or not isinstance(media_type, str)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("storage artifact reference is invalid")
        path = (self.run_root / relative).resolve()
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(self.run_root)
            or not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            or path.stat().st_size != byte_count
        ):
            raise ValueError("storage artifact failed path/hash validation")
        logical = f"artifact://{self.policy_session_id}/{digest}"
        existing = self._entries.get(logical)
        entry = {
            "schema": CONTEXT_ARTIFACT_REF_SCHEMA,
            "ref": logical,
            "sha256": digest,
            "kind": kind,
            "media_type": media_type,
            "bytes": byte_count,
            "policy_scope": self.policy_session_id,
            "sources": [{"round": round_index, "action_step": action_step}],
            "storage_ref": copy.deepcopy(dict(storage_ref)),
        }
        if existing is not None:
            immutable = {
                key: existing.get(key) for key in (
                    "schema", "ref", "sha256", "kind", "media_type", "bytes",
                    "policy_scope", "storage_ref",
                )
            }
            candidate = {key: entry.get(key) for key in immutable}
            if canonical_bytes(immutable) != canonical_bytes(candidate):
                raise ValueError("logical artifact identity drift")
            source = {"round": round_index, "action_step": action_step}
            sources = existing.setdefault("sources", [])
            if source not in sources:
                sources.append(source)
                sources.sort(key=lambda item: (
                    -1 if item.get("round") is None else int(item["round"]),
                    -1 if item.get("action_step") is None else int(item["action_step"]),
                ))
                self._persist()
            return self.public_ref(
                existing, round_index=round_index, action_step=action_step,
            )
        self._entries[logical] = entry
        self._persist()
        return self.public_ref(
            entry, round_index=round_index, action_step=action_step,
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        suffix: str,
        media_type: str,
        round_index: int | None,
        action_step: int | None,
    ) -> dict[str, Any]:
        storage = self.artifacts.put_bytes(
            data, kind=kind, suffix=suffix, media_type=media_type,
        )
        return self.register_storage_ref(
            storage, kind=kind, round_index=round_index, action_step=action_step,
        )

    def public_ref(
        self, entry: Mapping[str, Any], *, round_index: int | None = None,
        action_step: int | None = None,
    ) -> dict[str, Any]:
        return public_artifact_ref(entry, reader_name=self.reader_name,
                                   round_index=round_index, action_step=action_step)

    def resolve(self, logical_ref: str) -> tuple[dict[str, Any], bytes]:
        match = _REF_PATTERN.fullmatch(logical_ref)
        if match is None:
            raise RefSyntaxError("artifact reference syntax is invalid")
        if match.group("scope") != self.policy_session_id:
            raise RefScopeError("artifact reference is outside this PolicySession")
        entry = self._entries.get(logical_ref)
        if entry is None:
            raise RefUnknownError("artifact reference is unavailable")
        storage = entry.get("storage_ref")
        relative = storage.get("path") if isinstance(storage, Mapping) else None
        path = (self.run_root / str(relative)).resolve() if isinstance(relative, str) else None
        if (
            path is None
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(self.run_root)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RefStorageError("artifact storage is unavailable")
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != entry.get("sha256")
            or len(data) != entry.get("bytes")
            or match.group("sha256") != entry.get("sha256")
        ):
            raise RefHashError("artifact content hash does not match its reference")
        return copy.deepcopy(entry), data

    def public_refs_for_round(self, round_index: int | None) -> list[dict[str, Any]]:
        """List policy-readable refs first observed in one physical Episode."""

        if round_index is not None and (
            isinstance(round_index, bool) or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise ValueError("Artifact round index is invalid")
        values: list[dict[str, Any]] = []
        for entry in self._entries.values():
            sources = entry.get("sources")
            if not isinstance(sources, list):
                continue
            matching = [
                source for source in sources
                if isinstance(source, Mapping) and source.get("round") == round_index
            ]
            if not matching:
                continue
            action_steps = [
                source.get("action_step") for source in matching
                if isinstance(source.get("action_step"), int)
                and not isinstance(source.get("action_step"), bool)
            ]
            reference = self.public_ref(
                entry,
                round_index=round_index,
                action_step=min(action_steps) if action_steps else None,
            )
            reference["open"] = recoverable_artifact_open(reference)
            values.append(reference)
        return sorted(values, key=lambda item: (
            -1 if item.get("action_step") is None else int(item["action_step"]),
            str(item.get("ref")),
        ))


_json_pointer = json_pointer  # moved to artifact_reader.py


@dataclass
class RecoverableArtifactToolRuntime:
    base: Any
    run_root: Path
    event_path: Path

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.artifact_read_tool_name = artifact_read_name(self.base.definitions())
        self._surface = _RecoverableArtifactSurface(
            self.base.surface, self.artifact_read_tool_name,
        )
        self.catalog: ContextArtifactCatalog | None = None
        self.event_sink: Any | None = None

    @property
    def surface(self) -> Any:
        return self._surface

    @property
    def control_root(self) -> Path:
        return self.base.control_root

    @property
    def controlled_end_effector(self) -> str | None:
        return getattr(self.base, "controlled_end_effector", None)

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self.base.allowed_tools) + (self.artifact_read_tool_name,)

    def bind_context_artifacts(
        self, catalog: ContextArtifactCatalog, *, event_sink: Any | None = None,
    ) -> None:
        if self.catalog is not None and self.catalog is not catalog:
            raise RuntimeError("recoverable Artifact catalog was already bound")
        catalog.reader_name = self.artifact_read_tool_name
        self.catalog = catalog
        if event_sink is not None:
            self.event_sink = event_sink

    def definitions(self) -> list[dict[str, Any]]:
        return recoverable_artifact_definitions(self.base.definitions())

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            return "read_only"
        return str(self.base.classify_call_effect(name, arguments))

    def _require_catalog(self) -> ContextArtifactCatalog:
        if self.catalog is None:
            raise RuntimeError("recoverable Artifact catalog is not bound")
        return self.catalog

    def _record(
        self, *, name: str, arguments: Mapping[str, Any], status: str,
        call_context: Mapping[str, Any], error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        if not callable(self.event_sink):
            raise RuntimeError("recoverable Artifact Trace sink is not bound")
        self.event_sink(
            "context_artifact_retrieval",
            visibility="derived",
            observation_step=call_context.get("observation_step"),
            model_call_id=call_context.get("model_call_id"),
            tool_call_id=call_context.get("tool_call_id"),
            payload={
                "tool": name,
                "ref": arguments.get("ref"),
                "status": status,
                "error": error,
                "duration_ms": duration_ms,
            },
        )

    def _observe_ref(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        check_observe_ref_arguments(arguments)
        catalog = self._require_catalog()
        entry, data = policy_resolve(catalog, str(arguments["ref"]))
        media_type = str(entry["media_type"])
        if not media_type.startswith("image/"):
            raise ToolError(
                f"observe(ref) requires an image artifact; use {self.artifact_read_tool_name} for text/JSON",
                code="invalid_arguments", field="ref", received=arguments["ref"],
                constraints={"media_type": media_type, "open": {
                    "tool": self.artifact_read_tool_name,
                    "arguments": {"ref": arguments["ref"]},
                }},
            )
        sources = entry.get("sources")
        source = (
            sources[0]
            if isinstance(sources, list) and sources
            and isinstance(sources[0], Mapping) else {}
        )
        result = {
            "status": "actual_historical_artifact",
            "ref": entry["ref"],
            "sha256": entry["sha256"],
            "media_type": media_type,
            "bytes": entry["bytes"],
            "round": source.get("round"),
            "action_step": source.get("action_step"),
            "views": ["historical_artifact"],
            "images_base64": [base64.b64encode(data).decode("ascii")],
        }
        validate_schema_instance(
            {key: value for key, value in result.items() if key != "images_base64"},
            HISTORICAL_OBSERVE_RESULT_SCHEMA,
        )
        return result

    def _read_ref(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        check_read_arguments(arguments, ARTIFACT_READ_TOOL["inputSchema"], ARTIFACT_READ_LIMIT_BYTES)
        catalog = self._require_catalog()
        entry, data = policy_resolve(catalog, str(arguments["ref"]))
        media_type = str(entry["media_type"])
        if media_type.startswith("image/"):
            raise ToolError(
                "ref names an image artifact; this reader returns text or JSON only",
                code="invalid_arguments", field="ref", received=arguments["ref"],
                constraints={"media_type": media_type},
            )
        pointer = arguments.get("json_pointer")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 4096)
        if pointer is not None:
            if media_type != "application/json":
                raise ToolError(
                    "json_pointer must be a string selecting a JSON artifact",
                    code="invalid_arguments", field="json_pointer", received=pointer,
                    constraints={"allowed": {
                        "type": "string", "media_type": "application/json",
                    }},
                )
            value = json.loads(data.decode("utf-8"))
            try:
                selected = _json_pointer(value, pointer)
            except (ValueError, KeyError) as error:
                root_keys = sorted(str(key) for key in value) if isinstance(value, Mapping) else []
                raise ToolError(
                    f"{error.args[0]}; select a path from this artifact's root",
                    code="invalid_arguments", field="json_pointer", received=pointer,
                    constraints={
                        "allowed": "an existing JSON Pointer; empty string selects the root",
                        "top_level_keys": root_keys,
                        "root_type": (
                            "object" if isinstance(value, Mapping) else
                            "array" if isinstance(value, list) else "scalar"
                        ),
                        **({"root_array_length": len(value)} if isinstance(value, list) else {}),
                    },
                ) from error
            return {
                "status": "ok", "ref": entry["ref"], "sha256": entry["sha256"],
                "media_type": media_type, "bytes": entry["bytes"], "offset": 0,
                "next_offset": len(data), "eof": True, "content": None,
                "json_value": bounded_pointer_value(selected, pointer, ARTIFACT_READ_LIMIT_BYTES),
            }
        offset = 0 if offset is None else offset
        text, next_offset = read_range(data, offset, 4096 if limit is None else limit)
        return {
            "status": "ok", "ref": entry["ref"], "sha256": entry["sha256"],
            "media_type": media_type, "bytes": entry["bytes"], "offset": offset,
            "next_offset": next_offset, "eof": next_offset >= len(data),
            "content": text, "json_value": None,
        }

    def invoke_with_context(
        self, name: str, arguments: Mapping[str, Any], *, call_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            started_ns = time.monotonic_ns()
            try:
                result = (
                    self._read_ref(arguments) if name in ARTIFACT_READ_NAMES
                    else self._observe_ref(arguments)
                )
                if name in ARTIFACT_READ_NAMES:
                    validate_schema_instance(result, ARTIFACT_READ_RESULT_SCHEMA)
            except Exception as exc:
                self._record(
                    name=name, arguments=arguments, status="rejected",
                    call_context=call_context, error=str(exc),
                    duration_ms=round(
                        (time.monotonic_ns() - started_ns) / 1_000_000, 3,
                    ),
                )
                raise
            self._record(
                name=name, arguments=arguments, status="ok",
                call_context=call_context,
                duration_ms=round(
                    (time.monotonic_ns() - started_ns) / 1_000_000, 3,
                ),
            )
            return result
        invoke = getattr(self.base, "invoke_with_context", None)
        return (
            invoke(name, arguments, call_context=call_context)
            if callable(invoke) else self.base.invoke(name, arguments)
        )

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            return self.invoke_with_context(name, arguments, call_context={})
        return self.base.invoke(name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass(frozen=True)
class _RecoverableArtifactSurface:
    base: Any
    reader_name: str

    @property
    def result_schemas(self) -> dict[str, Any]:
        return recoverable_artifact_result_schemas(
            self.base.result_schemas, reader_name=self.reader_name,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def build_recoverable_artifact_runtime(
    plan: Mapping[str, Any], *, run_root: Path, event_path: Path, base_runtime: Any,
) -> RecoverableArtifactToolRuntime | Any:
    if plan.get("policy", {}).get("context_retention") != LOSSLESS_CONTEXT_RETENTION:
        return base_runtime
    return RecoverableArtifactToolRuntime(
        base=base_runtime, run_root=run_root, event_path=event_path,
    )


__all__ = [
    "ARTIFACT_READ_NAMES", "ARTIFACT_READ_RESULT_SCHEMA", "ARTIFACT_READ_TOOL",
    "CONTEXT_ARTIFACT_INDEX_SCHEMA", "CONTEXT_ARTIFACT_REF_SCHEMA",
    "CURRENT_OBSERVE_RESULT_SCHEMA", "HISTORICAL_OBSERVE_RESULT_SCHEMA",
    "RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA", "RECOVERABLE_ARTIFACT_POINTER_SCHEMA",
    "RECOVERABLE_ARTIFACT_TAG",
    "ContextArtifactCatalog", "LOSSLESS_CONTEXT_RETENTION",
    "RecoverableArtifactToolRuntime", "build_recoverable_artifact_runtime",
    "decode_recoverable_artifact_marker", "encode_recoverable_artifact_marker",
    "recoverable_artifact_envelope", "recoverable_artifact_open", "validate_recoverable_artifact_envelope",
    "recoverable_artifact_pointer", "validate_recoverable_artifact_pointer",
    "recoverable_artifact_definitions", "recoverable_artifact_result_schemas",
]
