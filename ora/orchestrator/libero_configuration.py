"""Host-safe validation of selected LIBERO driver configuration."""
from __future__ import annotations
from typing import Any
from .spec_values import RunSpecError, _positive_int


def validate_libero_configuration(runtime: dict[str, Any]) -> None:
    if runtime["transport"] != "local_rpc":
        raise RunSpecError("libero_pro runtime.transport must be local_rpc")
    driver = runtime.get("environment_driver", "rlinf_v1")
    if driver not in {"rlinf_v1", "direct_libero_v1"}:
        raise RunSpecError("runtime.environment_driver is unsupported")
    legacy_fields = ("rlinf_root_env", "required_rlinf_revision")
    if driver == "direct_libero_v1" and any(key in runtime for key in legacy_fields):
        raise RunSpecError("direct LIBERO runtime must not bind RLinf")
    for field in (
        "python_env", "libero_pro_root_env", "data_root_env",
        "required_libero_pro_revision", "required_data_revision", "suite",
        "expected_task_language", *(legacy_fields if driver == "rlinf_v1" else ()),
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RunSpecError(f"runtime.{field} must be non-empty")
    task_id = runtime.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise RunSpecError("runtime.task_id must be a non-negative integer")
    runtime.setdefault("main_camera", "agentview")
    if runtime["main_camera"] != "agentview":
        raise RunSpecError("libero_pro runtime.main_camera must be agentview")
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 1800)
    runtime.setdefault("servo_max_steps", 80)
    for field in ("startup_timeout_s", "request_timeout_s", "servo_max_steps"):
        runtime[field] = _positive_int(runtime[field], f"runtime.{field}")
