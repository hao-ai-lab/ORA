"""Modular prompt-bundle authoring, ownership, and delivery contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.agent.composition.prompt_bundle import (
    EMBODIMENT_SECTION_HEADINGS,
    PROMPT_BUNDLE_AUDIT_SCHEMA,
    PROMPT_BUNDLE_SCHEMA,
    PromptBundleError,
    audit_prompt_bundle,
    load_prompt_bundle,
    prompt_bundle_template,
    validate_prompt_bundle,
)
from ora.agent.composition.baseline import load_native_baseline


ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = ROOT / "ora/policy/prompt_bundles"


def _pinned_source_refusal(descriptor: dict) -> str:
    """The refusal of a descriptor whose first repository source still pins its bytes (stage 2 unit L3)."""
    index = next(i for i, source in enumerate(descriptor["provenance"]["sources"]) if "local_path" in source)
    return (f"descriptor.provenance.sources[{index}] names a repository source by path only; "
            "['revision', 'sha256'] are not accepted (the RunPlan source commit identifies repository bytes)")


def _current_bundle(backend: str, kind: str = "native_api") -> Path:
    instance = load_native_baseline(ROOT)["instances"][backend]
    return ROOT / instance["prompt_bundle"].replace("native_api", kind)


LIBERO = _current_bundle("libero_pro")
ROBOTWIN = _current_bundle("robotwin")


def _audit(path: Path) -> dict:
    return audit_prompt_bundle(load_prompt_bundle(path), project_root=ROOT)


def test_prompt_bundle_schema_and_draft_template_are_closed() -> None:
    schema = json.loads((BUNDLE_ROOT / "schema-v1.json").read_text(encoding="utf-8"))
    assert schema["$id"] == PROMPT_BUNDLE_SCHEMA
    draft = prompt_bundle_template("example_robot_bundle_v1")
    assert draft["schema"] == PROMPT_BUNDLE_SCHEMA
    assert list(draft["layers"]) == [
        "planner",
        "core",
        "environment",
        "embodiment",
        "episode_task",
        "tool_manual",
        "tool_surface",
        "dynamic_runtime",
    ]
    with pytest.raises(PromptBundleError, match="unresolved"):
        validate_prompt_bundle(draft)
    unknown = copy.deepcopy(draft)
    unknown.pop("bundle_id")
    unknown["name"] = "wrong"
    with pytest.raises(PromptBundleError, match="keys differ"):
        validate_prompt_bundle(unknown)


def test_standard_panda_and_aloha_bundles_pass_static_preflight() -> None:
    libero = _audit(LIBERO)
    robotwin = _audit(ROBOTWIN)
    assert libero["schema"] == PROMPT_BUNDLE_AUDIT_SCHEMA
    assert libero["passed"] is True
    assert robotwin["passed"] is True
    assert all(check["passed"] for check in libero["checks"])
    assert all(check["passed"] for check in robotwin["checks"])
    assert libero["embodiment_descriptor"]["descriptor_id"] == (
        "franka_panda_public_geometry_direct_v1"
    )
    assert robotwin["embodiment_descriptor"]["descriptor_id"] == "aloha_agilex_public_geometry_v5"
    assert robotwin["embodiment_control_profile"]["schema"] == (
        "embodiment-control-profile.v10"
    )
    assert robotwin["environment_profile"]["embodiment_policy"].endswith(
        "aloha-agilex/manual-public-geometry-v3.md"
    )


def test_bundle_preflight_checks_nine_embodiment_nodes_and_camera_inventory() -> None:
    result = _audit(ROBOTWIN)
    check = {item["check"]: item for item in result["checks"]}
    assert check["embodiment_nine_node_format"]["detail"]["required"] == list(
        EMBODIMENT_SECTION_HEADINGS
    )
    assert check["embodiment_camera_inventory"]["detail"]["advertised"] == [
        "head_camera",
        "front_camera",
        "left_camera",
        "right_camera",
    ]


def test_bundle_preflight_rejects_profile_source_and_layer_ownership_drift() -> None:
    bundle = load_prompt_bundle(ROBOTWIN)
    drifted = copy.deepcopy(bundle)
    drifted["layers"]["environment"]["source"] = (
        "ora/policy/environments/libero_pro/boundary-v2.md"
    )
    result = audit_prompt_bundle(drifted, project_root=ROOT)
    checks = {item["check"]: item for item in result["checks"]}
    assert result["passed"] is False
    assert checks["environment_profile_exact_sources"]["passed"] is False
    assert checks["environment_identity_present"]["passed"] is False

    leaked = copy.deepcopy(bundle)
    leaked["layers"]["planner"]["sources"] = [
        "ora/policy/embodiments/aloha-agilex/manual-public-geometry-v3.md"
    ]
    result = audit_prompt_bundle(leaked, project_root=ROOT)
    checks = {item["check"]: item for item in result["checks"]}
    assert checks["planner_single_owner_vocabulary"]["passed"] is False
    assert "ALOHA" in checks["planner_single_owner_vocabulary"]["detail"][
        "forbidden_terms_present"
    ]


def test_native_tool_schema_is_eager_while_manual_is_manifest_only() -> None:
    result = _audit(ROBOTWIN)
    by_layer = {item["layer"]: item for item in result["delivery_inventory"]}
    assert by_layer["planner"]["provider_role"] == "system"
    assert by_layer["planner"]["provider_visible"] is True
    assert by_layer["tool_surface"] == {
        "layer": "tool_surface",
        "source_path": None,
        "source_sha256": None,
        "delivery": "provider_tools_field",
        "provider_role": "tools",
        "provider_visible": True,
        "provider_visibility_status": "exact",
        "delivery_timing": "every_model_call",
    }
    assert by_layer["tool_manual"]["provider_visible"] is False
    assert by_layer["tool_manual"]["delivery_timing"] == "never"

    false_claim = load_prompt_bundle(ROBOTWIN)
    false_claim["layers"]["tool_manual"].update({
        "delivery": "on_demand",
        "reader_tool": "read_manual",
    })
    result = audit_prompt_bundle(false_claim, project_root=ROOT)
    checks = {item["check"]: item for item in result["checks"]}
    assert checks["tool_manual_delivery_truth"]["passed"] is False
