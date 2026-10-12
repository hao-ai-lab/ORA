"""The single preflight check record shape, shared by every backend's preflight."""
from __future__ import annotations

from typing import Any


def check(label: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": label, "passed": bool(passed), "detail": detail}


def execution_host_check(runtime: Any, marker: str | None) -> dict[str, Any]:
    """A live run names a registered host; an undeclared one is refused, not assumed."""
    from .resources import ExecutionResourceError, require_registered_host

    required_host = runtime.get("required_host")
    detail = {"expected": required_host or "runtime.required_host is not declared", "observed": marker}
    passed = bool(required_host) and marker == required_host
    if passed:
        try:
            require_registered_host(required_host)
        except ExecutionResourceError as exc:
            passed, detail["profile"] = False, str(exc)
    return check("execution_host", passed, detail)
