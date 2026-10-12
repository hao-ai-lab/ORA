"""Stable value hashes for numeric input/output evidence."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping

def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
