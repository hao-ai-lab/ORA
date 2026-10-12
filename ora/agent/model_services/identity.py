"""Content identity for the shared grounding/grasp service adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path


def grounding_grasp_adapter_sha256() -> str:
    """Bind the entry point, shared implementation and identity algorithm."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(b"ora.grounding-grasp-adapter.v1\0")
    for name in ("identity.py", "resident.py", "grounding_grasp.py"):
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = ["grounding_grasp_adapter_sha256"]
