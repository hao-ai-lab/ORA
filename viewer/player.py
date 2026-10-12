"""Packaged, dependency-free presentation player embedded in the Notebook."""

from __future__ import annotations

import json
from pathlib import Path


def render_process_player(run_id: str) -> str:
    template = (Path(__file__).parent / "templates/process_player.html").read_text(encoding="utf-8")
    payload = json.dumps({"run_id": run_id}, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("<!--PROCESS_PLAYER_DATA-->", payload)
