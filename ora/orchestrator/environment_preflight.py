"""Delegate source-bound developing-backend admission without simulator imports."""
from __future__ import annotations

import subprocess
from typing import Any, Callable, Mapping


def developing_backend_checks(
    plan: Mapping[str, Any], *,
    robodojo_check: Callable[..., list[dict[str, Any]]],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[dict[str, Any]]:
    """Keep each backend's own evidence checker and the supplied host runner."""

    if plan["runtime"]["kind"] == "robodojo":
        return robodojo_check(plan, runner)
    raise ValueError("developing backend preflight requires robodojo")
