"""Physical motion receipts and the run-scoped JSON artifacts they reference.

Learning eligibility audits every receipt that drove the robot. Composite and
skill-sequence envelopes nest the receipts that actually
reached a controller; ``_physical_motion_receipts`` flattens them so each
driven receipt is judged under its own contract. ``_referenced_json_artifact``
resolves one hash-bound evidence reference inside the run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping


def _referenced_json_artifact(
    run_root: Path | None, reference: Any,
) -> dict[str, Any] | None:
    """Resolve one run-scoped JSON reference and verify its exact bytes/hash."""

    if run_root is None or not isinstance(reference, Mapping):
        return None
    relative_raw = reference.get("path")
    expected_hash = reference.get("sha256")
    if (
        not isinstance(relative_raw, str) or not relative_raw
        or not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
    ):
        return None
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    root = run_root.resolve()
    candidates: list[Path] = []
    for base in (root, root / "native"):
        raw_path = base / relative
        cursor = base
        unsafe = False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                unsafe = True
                break
        if unsafe:
            continue
        path = raw_path.resolve()
        if path.is_relative_to(root) and path.is_file() and path not in candidates:
            candidates.append(path)
    if len(candidates) != 1:
        return None
    path = candidates[0]
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size,
        ):
            return None
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            return None
        expected_bytes = reference.get("bytes")
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or len(payload) != expected_bytes
        ):
            return None
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return value if isinstance(value, dict) else None


def _physical_motion_receipts(
    events: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return physical receipts nested under composite and skill-sequence actions."""

    receipts: list[Mapping[str, Any]] = []

    def flatten(record: Mapping[str, Any]) -> None:
        schema = record.get("schema")
        if schema == "ora.composite-motion-receipt.v1":
            phases = record.get("phase_receipts")
            if not isinstance(phases, list):
                receipts.append({})
                return
            for phase in phases:
                if not isinstance(phase, Mapping):
                    receipts.append({})
                    continue
                phase_receipt = phase.get("receipt")
                if isinstance(phase_receipt, Mapping):
                    flatten(phase_receipt)
                elif phase.get("status") not in {"skipped", "not_requested"}:
                    receipts.append({})
            return
        if schema == "ora.manipulation-skill-sequence-receipt.v1":
            # A closed manipulation skill is an envelope over the same driven
            # receipts an atomic action would have produced, one level deeper
            # than a composite. Each dispatched stage carries its own receipt
            # and its own controller contract, so the evidence is audited per
            # stage rather than per skill. A measurement stage -- the absorbed
            # verify, or a policy-visible stop -- drives nothing and has no
            # action_id, so it owes no controller evidence; a stage that was
            # dispatched and lost its receipt must still fail.
            stages = record.get("stage_receipts")
            if not isinstance(stages, list):
                receipts.append({})
                return
            for stage in stages:
                if not isinstance(stage, Mapping):
                    receipts.append({})
                    continue
                stage_receipt = stage.get("receipt")
                if isinstance(stage_receipt, Mapping):
                    flatten(stage_receipt)
                elif stage.get("action_id") is not None:
                    receipts.append({})
            return
        receipts.append(record)

    for event in events:
        if event.get("type") != "motion_receipt":
            continue
        record = event.get("record")
        if isinstance(record, Mapping) and isinstance(record.get("receipt"), Mapping):
            record = record["receipt"]
        if not isinstance(record, Mapping):
            continue
        flatten(record)
    return receipts
