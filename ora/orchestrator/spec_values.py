"""RunSpec value validation and deterministic content identities.

``_resource`` and ``_resource_list`` bind project files by path, SHA-256 and
size; ignored workspace files are bound by ``spec._generated_resource``.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

class RunSpecError(ValueError):
    """A RunSpec is incomplete, ambiguous or unsafe."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunSpecError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunSpecError(f"{label} must be a positive integer")
    return value


def _stable_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise RunSpecError(f"{label} must be a stable id")
    return value


def _resource(root: Path, value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise RunSpecError(f"{label} must be a project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RunSpecError(f"{label} must remain inside the project")
    path = (root / relative).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise RunSpecError(f"{label} is unavailable: {value}")
    return {"path": relative.as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _resource_list(root: Path, value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RunSpecError(f"{label} must be a non-empty ordered list")
    bound = [_resource(root, item, f"{label}[{index}]") for index, item in enumerate(value)]
    paths = [item["path"] for item in bound]
    if len(paths) != len(set(paths)):
        raise RunSpecError(f"{label} must not contain duplicate paths")
    return bound
