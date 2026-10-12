"""Safe evidence-file reads and shared evaluation error identity."""
from __future__ import annotations
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

class EvaluationError(RuntimeError):
    """A run cannot be evaluated, sealed or exported."""


def _hash(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise EvaluationError(f"cannot safely open artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationError(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size,
        ):
            raise EvaluationError(f"artifact changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"expected object: {path}")
    return value


def _receipt_execution_fault(path: Path) -> bool:
    """Whether one receipt records a backend-declared execution fault."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, Mapping):
        return False
    return (
        record.get("execution_status") == "execution_fault"
        or isinstance(record.get("execution_fault"), Mapping)
    )
