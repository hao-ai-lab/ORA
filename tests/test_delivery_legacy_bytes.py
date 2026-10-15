"""Surfaces that do not declare the 0059 delivery capabilities keep exact bytes.

D1 (``last_action_summary_v1``), D2 (``policy_angles_degrees_v1``) and D3a
(``within_payload_deduplication_v1``) activate on the next surface generation.
Every current and frozen surface composes, projects and publishes the bytes it
did before them. Of the direct_v3 default surfaces only RoboTwin's is still
registered; it stands for the surfaces that do not declare them.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from delivery_lessons_fixtures import DEFAULT_PROFILES, NEW_CAPABILITIES, robotwin_receipt, skill_receipt
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.native.receipt_projection import (
    _RECEIPT_EFFECT_PROJECTORS, ReceiptProjectionContext, resolve_receipt_effect_projector,
)

#: The direct_v3 default surfaces that are still registered.
LEGACY_PROFILES = {backend: profile for backend, profile in DEFAULT_PROFILES.items()
                   if profile in PROFILE_CATALOG}

RECEIPTS = {
    "robotwin": (robotwin_receipt(), robotwin_receipt(status="reached", stage=None, code=None,
                                                      executed=True, pose_error=(0.002, 0.007)),
                 skill_receipt()),
}


@pytest.mark.parametrize("backend", sorted(LEGACY_PROFILES))
def test_the_direct_v3_default_surfaces_declare_none_of_the_new_capabilities(backend):
    surface = resolve_tool_surface(LEGACY_PROFILES[backend])
    assert not set(NEW_CAPABILITIES) & set(surface.capabilities)


def test_the_degree_binder_never_runs_for_a_surface_that_does_not_declare_it(monkeypatch):
    from ora.agent.composition import policy_units_schema

    def refuse(*args, **kwargs):
        raise AssertionError("bound on a surface that did not declare it")

    import json

    declaring = {profile for profile, path in PROFILE_CATALOG.items()
                 if policy_units_schema.CAPABILITY in json.loads(path.read_text(encoding="utf-8"))["capabilities"]}
    # Exactly the three defaults declare it.
    assert declaring == {"robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                         "robodojo_placement_closed_skills_v2"}
    monkeypatch.setattr(policy_units_schema, "bind_policy_angles_degrees", refuse)
    for profile in sorted(set(PROFILE_CATALOG) - declaring):
        surface = resolve_tool_surface(profile)
        assert policy_units_schema.CAPABILITY not in surface.capabilities
        assert not any(item["slot"].startswith(f"derived:{policy_units_schema.CAPABILITY}")
                       for item in surface.components)


@pytest.mark.parametrize("backend", sorted(LEGACY_PROFILES))
def test_effects_on_the_default_surfaces_are_the_legacy_bytes(backend):
    surface = resolve_tool_surface(LEGACY_PROFILES[backend])
    for receipt in RECEIPTS[backend]:
        context = ReceiptProjectionContext(last_action_id=receipt["action_id"],
                                           last_action_public_name=None,
                                           surface_supports=surface.supports)
        legacy = _RECEIPT_EFFECT_PROJECTORS[receipt["schema"]](receipt, context)
        delivered = resolve_receipt_effect_projector(receipt["schema"])(receipt, context)
        assert json.dumps(delivered, ensure_ascii=False, sort_keys=True) == json.dumps(
            legacy, ensure_ascii=False, sort_keys=True)


@pytest.mark.parametrize("backend", sorted(LEGACY_PROFILES))
def test_published_context_is_the_legacy_bytes(tmp_path, backend):
    """Passing the surface's own capabilities to the projector changes nothing."""
    from test_current_visual_delivery import published

    context, projector, store, _surface, source = published(tmp_path, backend)
    observation = json.loads((source / "observation.json").read_text())
    # The direct_v3 default surface, which declares none of the delivery capabilities.
    frozen = resolve_tool_surface(LEGACY_PROFILES[backend])
    bound = dataclasses.replace(projector, workspace=tmp_path / "bound-workspace",
                                control_root=tmp_path / "bound-control",
                                capabilities=frozen.capabilities)
    again = bound.publish(observation=observation, observation_root=source,
                          task_instruction="Inspect", remaining_budget={},
                          policy_state=store.steps[-1])
    assert json.dumps(again["public_robot_facts"], sort_keys=True) == json.dumps(
        context["public_robot_facts"], sort_keys=True)


@pytest.mark.parametrize("backend", sorted(LEGACY_PROFILES))
def test_tool_results_pass_both_gateways_unchanged(backend):
    from ora.agent.native.gateway import NativeToolGateway

    surface = resolve_tool_surface(LEGACY_PROFILES[backend])
    result = {"candidates": [{"orientation_error_rad": 0.5}], "status": "ok"}

    class Runtime:
        def __init__(self):
            self.surface = surface

    delivered = NativeToolGateway(Runtime()).record_proposed_result(
        "align_plan", {"payload": {"arguments": {}}}, result)
    assert delivered == {"candidates": [{"orientation_error_rad": 0.5}], "status": "ok"}
