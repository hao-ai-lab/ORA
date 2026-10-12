"""Fresh source-bound bundle templates of the Native policy runtime.

The templates are the bundles the default baseline names for each backend and
consumer. Since 2026-09-29 the RoboTwin default is the B2 closed-skill surface
without Servo, and since 2026-10-04 (direct_v4, integration 0059 unit A0) no
default declares Servo on any backend.
"""
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.baseline import load_native_baseline
from ora.agent.composition.prompt_bundle import audit_prompt_bundle, load_prompt_bundle
from ora.orchestrator.defaults import expand_native_defaults

ROOT = Path(__file__).resolve().parents[1]
POLICY_KINDS = ("native_api",)
TEMPLATES = [
    (backend, kind, ROOT / instance["prompt_bundle"].replace("native_api", kind))
    for backend, instance in load_native_baseline(ROOT)["instances"].items()
    for kind in POLICY_KINDS
]


@pytest.mark.parametrize("backend,kind,path", TEMPLATES,
                         ids=[f"{backend}-{kind}" for backend, kind, _ in TEMPLATES])
def test_fresh_servo_templates_bind_actual_help_and_prompt_ownership(backend, kind, path):
    bundle = load_prompt_bundle(path)
    audit = audit_prompt_bundle(bundle, project_root=ROOT)
    assert audit["passed"], [row for row in audit["checks"] if not row["passed"]]
    spec = {"policy": {"kind": kind, "supports_images": True}, "runtime": {"kind": backend}}
    expanded = expand_native_defaults(spec, root=ROOT)
    assert bundle["runtime"]["policy_kind"] == kind
    assert expanded["policy"]["tool_surface"]["profile"] == bundle["layers"]["tool_surface"]["profile"]
    assert expanded["runtime"]["environment_prompt_profile"]["profile_id"] == bundle["bindings"]["environment_prompt_profile"]["profile_id"]
    assert bundle["layers"]["embodiment"]["source"] in expanded["policy_context"]["project_instructions"]
    surface = resolve_tool_surface(bundle["layers"]["tool_surface"]["profile"])
    # No v2 default declares Servo (user decision 2026-10-04).
    assert not surface.supports("bounded_servo_v1")
    assert "servo" not in surface.mutation_tools
    if backend != "libero_pro":
        # The dual-arm defaults are closed-skill surfaces; place_execute is
        # retired on the v2 generation with place_plan.
        assert surface.supports("closed_manipulation_skills_v1")
        assert "grasp_execute" in surface.mutation_tools and "place_execute" not in surface.mutation_tools


def test_template_inventory_does_not_drop_a_backend():
    backends = {(backend, kind) for backend in ("libero_pro", "robotwin") for kind in POLICY_KINDS}
    # direct_v4 adds the RoboDojo instance (integration 0059 unit A0).
    assert {(backend, kind) for backend, kind, _ in TEMPLATES} == backends | {("robodojo", "native_api")}
    assert len({path for _, _, path in TEMPLATES}) == 3
