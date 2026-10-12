"""Resolve eager environment/embodiment prompt sources from public metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .locked_assets import locked_asset


PUBLIC_ENVIRONMENT_CONTEXT_SCHEMA = "ora.public-environment-context.v1"
ENVIRONMENT_PROMPT_PROFILE_SCHEMA = "ora.environment-prompt-profile.v1"


class EnvironmentContextError(RuntimeError):
    """Public environment metadata has no qualified prompt profile."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_environment_prompt_profile(
    metadata: Mapping[str, Any], *, project_root: Path,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Select one source-bound prompt profile without experiment-name logic."""

    value = dict(metadata)
    if value.get("schema") != PUBLIC_ENVIRONMENT_CONTEXT_SCHEMA:
        raise EnvironmentContextError("public environment context schema is invalid")
    required = {
        "environment_id", "embodiment_id", "robot_models", "controller",
        "public_cameras",
    }
    if not required.issubset(value):
        raise EnvironmentContextError("public environment context is incomplete")
    if not all(isinstance(value.get(field), str) and value[field] for field in (
        "environment_id", "embodiment_id", "controller",
    )):
        raise EnvironmentContextError("public environment context identity is invalid")
    if not isinstance(value["robot_models"], list) or not all(
        isinstance(item, str) and item for item in value["robot_models"]
    ):
        raise EnvironmentContextError("public robot_models are invalid")
    if not isinstance(value["public_cameras"], list) or not all(
        isinstance(item, str) and item for item in value["public_cameras"]
    ):
        raise EnvironmentContextError("public cameras are invalid")

    profiles = []
    root = project_root.resolve()
    profile_root = root / "ora/policy/environments"
    for path in sorted(profile_root.glob("*/context-profile-*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentContextError(f"invalid environment prompt profile: {path}") from exc
        if not isinstance(profile, dict) or profile.get("schema") != (
            ENVIRONMENT_PROMPT_PROFILE_SCHEMA
        ):
            raise EnvironmentContextError(f"invalid environment prompt profile: {path}")
        match = profile.get("match")
        if not isinstance(match, Mapping):
            raise EnvironmentContextError(f"environment prompt profile lacks match: {path}")
        if (
            (profile_id is None or profile.get("profile_id") == profile_id)
            and
            match.get("environment_id") == value["environment_id"]
            and match.get("embodiment_id") == value["embodiment_id"]
            and match.get("robot_models") == value["robot_models"]
            and match.get("controller") == value["controller"]
            and match.get("public_cameras") == value["public_cameras"]
        ):
            profiles.append((path, profile))
    if profile_id is None and len(profiles) > 1:
        explicitly_default = [
            item for item in profiles if item[1].get("default") is True
        ]
        ordinary_profiles = [
            item for item in profiles if item[1].get("default") is not False
        ]
        profiles = explicitly_default or (
            ordinary_profiles if len(ordinary_profiles) == 1 else [
                item for item in profiles
                if item[0].name == "context-profile-v1.json"
            ]
        )
    if len(profiles) != 1:
        raise EnvironmentContextError(
            "public environment context must match exactly one prompt profile"
        )
    profile_path, profile = profiles[0]
    sources = profile.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "environment_policy", "embodiment_policy",
    }:
        raise EnvironmentContextError("environment prompt profile sources are invalid")
    resolved_sources: dict[str, str] = {}
    for kind, raw in sources.items():
        if not isinstance(raw, str) or not raw:
            raise EnvironmentContextError(f"environment prompt source {kind} is invalid")
        source = (root / raw).resolve()
        if not source.is_file() or not source.is_relative_to(root):
            raise EnvironmentContextError(f"environment prompt source is unavailable: {raw}")
        resolved_sources[kind] = Path(raw).as_posix()
    compiled = profile.get("compiled_embodiment")
    if compiled is not None:
        from .embodiment import (
            EmbodimentDescriptorError, audit_embodiment_sources,
            load_embodiment_descriptor, render_embodiment_card,
        )
        from .embodiment_geometry import bind_public_geometry_descriptor, load_public_geometry_binding
        if (not isinstance(compiled, Mapping)
                or set(compiled) != {"descriptor", "review", "card_sha256", "runtime_verification"}
                or compiled["runtime_verification"] != "loaded_robot_geometry_before_policy_v1"):
            raise EnvironmentContextError("compiled embodiment requires exact sources and a before-policy Runtime gate")
        try:
            # Descriptor and wording review are named by path; the resolved
            # binding records the sha256 of the bytes read here.
            descriptor_record = bind_public_geometry_descriptor(compiled["descriptor"], project_root=root)
            load_public_geometry_binding(descriptor_record, project_root=root)
            descriptor = load_embodiment_descriptor(root / descriptor_record["path"])
            audit_embodiment_sources(descriptor, project_root=root)
            generated = render_embodiment_card(descriptor)
            card = (root / resolved_sources["embodiment_policy"]).read_text()
            if card != generated or hashlib.sha256(card.encode()).hexdigest() != compiled["card_sha256"]:
                raise EnvironmentContextError("embodiment manual differs from its deterministic compiled card")
            # The wording review attests the card text, the one human pin: the
            # card must be the descriptor's deterministic projection and hash
            # to card_sha256. Descriptor bytes are identified by policy.lock and
            # recorded in the binding below, so an edit that leaves the card
            # unchanged (provenance, say) needs no new review.
            review_asset = locked_asset(root, compiled["review"], label="compiled embodiment wording review",
                                        error=EnvironmentContextError)
            review = json.loads(review_asset.data)
            if (not isinstance(review, Mapping)
                    or set(review) != {"schema", "decision", "reviewer", "card_sha256", "scope"}
                    or review["schema"] != "ora.embodiment-wording-review.v1"
                    or review["decision"] != "accepted"
                    or review["reviewer"] != "codex_source_review"
                    or review["scope"] != "public_robot_geometry_wording_not_runtime_or_behavior_qualification"
                    or review["card_sha256"] != compiled["card_sha256"]):
                raise EnvironmentContextError("compiled embodiment wording review does not accept these exact sources")
            compiled = {**compiled, "descriptor": descriptor_record, "review": review_asset.record()}
        except (EmbodimentDescriptorError, OSError, ValueError, TypeError, KeyError) as exc:
            raise EnvironmentContextError(f"invalid compiled embodiment binding: {exc}") from exc
    metadata_sha256 = hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "schema": ENVIRONMENT_PROMPT_PROFILE_SCHEMA,
        "profile_id": profile["profile_id"],
        "profile_path": profile_path.relative_to(root).as_posix(),
        "profile_sha256": _sha256(profile_path),
        "metadata_sha256": metadata_sha256,
        **resolved_sources,
        **({"compiled_embodiment": dict(compiled)} if compiled is not None else {}),
    }
