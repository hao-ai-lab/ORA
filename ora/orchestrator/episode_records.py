"""The three file writes of a running episode.

`_atomic_json` replaces one JSON evidence file, `_append` adds one fsynced line
to an append-only journal such as `events.jsonl`, and `_status` rewrites the
operator-facing `status.json`. `coordinator` re-exports all three.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _status(path: Path, *, run_id: str, state: str, **fields: Any) -> None:
    _atomic_json(path, {
        "schema": "ora.run-status.v1",
        "run_id": run_id, "state": state, "updated_unix_s": time.time(), **fields,
    })
