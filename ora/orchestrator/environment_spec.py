"""Environment-specific RunSpec context defaults and exact profile bindings.

Source composition stays outside the generic RunSpec validator; the resolver
still owns global policy validation, resource binding and RunPlan identity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.environment_context import (
    EnvironmentContextError, resolve_environment_prompt_profile,
)
from .spec_values import RunSpecError


def resolve_typed_environment_prompt_profile(
    runtime: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Validate and resolve the exact public metadata/profile pairing."""

    public_environment_context = runtime.get("public_environment_context")
    environment_prompt_profile = runtime.get("environment_prompt_profile")
    if not isinstance(public_environment_context, Mapping):
        raise RunSpecError(
            "typed native slots require runtime.public_environment_context"
        )
    if not isinstance(environment_prompt_profile, Mapping):
        raise RunSpecError(
            "typed native slots require runtime.environment_prompt_profile"
        )
    try:
        resolved_environment_profile = resolve_environment_prompt_profile(
            public_environment_context,
            project_root=root,
            profile_id=(
                environment_prompt_profile.get("profile_id")
                if isinstance(environment_prompt_profile.get("profile_id"), str)
                else None
            ),
        )
    except EnvironmentContextError as exc:
        raise RunSpecError(str(exc)) from exc
    expected_environment_binding = {
        key: resolved_environment_profile[key] for key in (
            "schema", "profile_id", "profile_path", "profile_sha256",
            "metadata_sha256",
        )
    }
    if dict(environment_prompt_profile) != expected_environment_binding:
        raise RunSpecError(
            "runtime.environment_prompt_profile does not match public metadata"
        )
    return resolved_environment_profile
