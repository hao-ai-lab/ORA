"""Resolve the two canonical public-state receipt forms without executing tools."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .policy_state import _sanitize_embedded_receipt
from ora.contracts.action import (
    ActionValidationError, MOTION_ACTION_V3_SCHEMA, validate_runtime_action,
)


def public_worker_arguments_match(action: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    """Accept exact arguments or canonical v3 motion normalization noise.

    Repeated validation may change a unit quaternion's final floating-point
    bit. Use the shared bounded vector comparison after action validation;
    non-vector fields retain their exact identity.
    """
    if receipt.get("arguments") == action.get("arguments"):
        return True
    if action.get("schema") != MOTION_ACTION_V3_SCHEMA or action.get("action") not in {"move_pose", "move_free_space"}:
        return False
    try:
        return public_motion_arguments_match(
            validate_runtime_action(action)["arguments"], receipt.get("arguments"),
        )
    except (ActionValidationError, TypeError, KeyError):
        return False


def public_motion_arguments_match(expected: Any, actual: Any) -> bool:
    """Allow only IEEE normalization noise in canonical public TCP vectors.

    Validation can renormalize a quaternion by its final floating-point bit.
    Positions and orientations in explicit arm targets tolerate at most 1e-12;
    all keys, other values and types retain exact identity. This does not join
    action IDs, observation IDs or source hashes: callers must compare those.
    """
    def compare(first: Any, second: Any, path: tuple[str, ...]) -> bool:
        if (len(path) == 3 and path[0] == "targets" and path[1] in {"left", "right"}
                and path[2] in {"position_m", "orientation_xyzw"}):
            size = 3 if path[2] == "position_m" else 4
            return (isinstance(first, list) and isinstance(second, list)
                    and len(first) == len(second) == size
                    and all(type(a) in (int, float) and type(b) in (int, float)
                            and math.isfinite(a) and math.isfinite(b)
                            and abs(a - b) <= 1e-12
                            for a, b in zip(first, second, strict=True)))
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            return set(first) == set(second) and all(
                compare(first[key], second[key], (*path, key)) for key in first)
        if isinstance(first, list) and isinstance(second, list):
            return len(first) == len(second) and all(
                compare(a, b, (*path, str(index)))
                for index, (a, b) in enumerate(zip(first, second, strict=True)))
        if type(first) is not type(second):
            return False
        if isinstance(first, float) and (not math.isfinite(first) or not math.isfinite(second)):
            return False
        return first == second

    return isinstance(expected, Mapping) and isinstance(actual, Mapping) and compare(expected, actual, ())


def public_actions_match(expected: Any, actual: Any) -> bool:
    """Join every action identity field exactly, then its canonical arguments."""
    return (isinstance(expected, Mapping) and isinstance(actual, Mapping)
            and set(expected) == set(actual) and "arguments" in expected
            and all(type(value) is type(actual[key]) and value == actual[key]
                    for key, value in expected.items() if key != "arguments")
            and public_motion_arguments_match(expected["arguments"], actual["arguments"]))


def canonical_public_receipt(
    state: Mapping[str, Any], run_root: Path | None,
) -> dict[str, Any] | None:
    """Return a direct receipt or a verified, already-sanitized inline copy.

    Coordinator snapshots embed a direct execution receipt. Environment-only
    observations may instead carry ``{path, sha256, receipt}``, relative to
    the run or its native evidence root. A wrapper is never trusted solely
    because its inline body looks plausible. Missing, ambiguous or changed
    evidence returns None, so callers cannot advance an action-stage or grant
    a permit using it. No private artifact content is added to the result.
    """

    value = state.get("previous_execution_receipt")
    if not isinstance(value, Mapping):
        return None
    if not {"path", "sha256", "receipt"}.intersection(value):
        return copy.deepcopy(dict(value))
    inline = value.get("receipt")
    relative, expected = value.get("path"), value.get("sha256")
    if (
        set(value) != {"path", "sha256", "receipt"}
        or not isinstance(inline, Mapping) or not isinstance(relative, str)
        or not relative or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
    ):
        return None
    candidates: list[Path] = []
    if run_root is None:
        return None
    try:
        root = run_root.resolve()
        for base in (root, root / "native"):
            candidate = base / relative
            if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root):
                continue
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file() and resolved not in candidates:
                candidates.append(resolved)
    except (OSError, RuntimeError, ValueError):
        return None
    if len(candidates) != 1:
        return None
    descriptor = None
    try:
        descriptor = os.open(candidates[0], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            return None
        if hashlib.sha256(data).hexdigest() != expected:
            return None
        original = json.loads(data)
    except (OSError, ValueError, UnicodeError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    matches = isinstance(original, Mapping) and _sanitize_embedded_receipt(original) == inline
    if not matches:
        return None
    return copy.deepcopy(dict(inline))


__all__ = ["canonical_public_receipt", "public_actions_match", "public_motion_arguments_match",
           "public_worker_arguments_match"]
