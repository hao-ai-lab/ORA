"""Host-only controller binding checks shared by admission callers."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping


def robotwin_control_binding(
    plan: Mapping[str, Any], *, project_root: Path, check: Any,
) -> dict[str, Any]:
    from ora.environment.factory import (
        EnvironmentFactoryError, validate_robotwin_control_binding,
    )

    runtime = plan["runtime"]
    try:
        binding = validate_robotwin_control_binding(
            runtime, action_codec=plan.get("policy", {}).get("action_codec"),
            project_root=project_root,
        )
    except (EnvironmentFactoryError, ValueError, OSError, KeyError, TypeError) as exc:
        return check("robotwin_control_binding", False, str(exc))
    return check("robotwin_control_binding", True, {
        "required": binding is not None,
        "profile": runtime.get("control_profile"),
        "robotwin_revision": runtime.get("required_revision"),
    })
