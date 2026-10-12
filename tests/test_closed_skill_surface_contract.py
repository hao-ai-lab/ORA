"""The closed-skill surfaces this tree composes advertise only what they serve.

Help may not name a tool the surface does not declare, driver messages follow
the surface's spelling, and the tools retired from the live closed surfaces
stay undeclared in the tool list and in everything delivered with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.tools.runtime import ManipulationToolRuntime


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Publication: the closed arm, and what makes it a control rather than a copy
# --------------------------------------------------------------------------

#: The closed-skill surfaces this tree composes (their atomic sibling arms left
#: the tree with the closed_skills_v1 generation).
CLOSED = ("robotwin_toolchain_b2_closed_noservo_v2", "robodojo_placement_closed_skills_v2")
#: Retired from the live closed arms by undeclaration (user decision
#: 2026-09-29; RoboTwin revision 20, LIBERO revision 17): never called
#: (place_execute, place_resume) or returning no candidate (place_plan) across
#: every recorded run. Their definitions stay in the shared files, and their
#: handlers stay in the runtime.
RETIRED_PLACEMENT = ("place_plan", "place_execute", "place_resume")
#: Read only by the code paths of the retired placement tools and the absorbed
#: retention verifier (placement_runtime, placement_depth, placement_refusal,
#: retention.FULL_REFERENCE_CAPABILITY, and the held-object offset only
#: place_plan sets), so the closed arms stop declaring them.
RETIRED_CAPABILITIES = {
    "flat_placement_arguments_v1", "public_rgbd_placement_support_v1",
    "public_rgbd_dual_arm_placement_support_v1", "public_rgbd_placement_depth_edges_v1",
    "placement_refusal_measurements_v1", "public_rgbd_grasp_retention_verification_v4",
    "held_object_offset_v1",
}


@pytest.mark.parametrize("profile", CLOSED)
def test_no_driver_message_sends_a_policy_to_a_name_its_surface_does_not_use(profile):
    """These modules became policy-reachable when the closed arm declared them.

    An error that directs a policy to another tool is the shape that misleads
    when the name is not the one the policy holds. `canonicalise_prose` rewrites
    help text and schemas at composition; a ToolError built at call time is not
    composed, so the driver resolves the spelling through the host's surface.
    """
    import inspect
    import re

    from ora.agent.composition import resolve_tool_surface
    from ora.agent.tools import skill_driver, skill_ledger, skill_receipt

    surface = resolve_tool_surface(profile)
    advertised = {item["name"] for item in surface.definitions}
    dotted = re.compile(
        r"(?<![\w.])((?:grasp|place|vision|motion|align|payload)\.[a-z_]+)(?![\w.])"
    )
    for module in (skill_driver, skill_ledger, skill_receipt):
        for line in inspect.getsource(module).splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "``", "*")) or not (
                '"' in stripped or "'" in stripped
            ):
                continue
            for literal, _ in re.findall(r'"([^"\\]{8,400})"|\'([^\'\\]{8,400})\'', stripped):
                for name in dotted.findall(literal):
                    snake = name.replace(".", "_")
                    assert snake not in advertised, (
                        f"{module.__name__} shows {name!r} while {profile} "
                        f"advertises {snake!r}"
                    )


def test_the_advertised_spelling_follows_the_surface(monkeypatch):
    """Both arms share this driver, and they do not share one spelling policy."""
    from types import SimpleNamespace

    from ora.agent.composition import resolve_tool_surface
    from ora.agent.tools.skill_candidates import _advertised

    class _Host:
        def __init__(self, surface):
            self.surface = surface

    canonical = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")
    assert _advertised(_Host(canonical), "grasp_plan") == "grasp_plan"
    assert _advertised(_Host(canonical), "grasp.plan") == "grasp_plan"

    # No surface this tree composes keeps the dotted spelling; a host showing it
    # gets it back.
    dotted = SimpleNamespace(definitions=[{"name": "grasp.plan"}, {"name": "place.plan"}])
    assert _advertised(_Host(dotted), "grasp_plan") == "grasp.plan"
    assert _advertised(_Host(dotted), "place_plan") == "place.plan"
    # A surface that declares neither gets the literal back, naming what was meant.
    assert _advertised(_Host(None), "grasp_plan") == "grasp_plan"
    assert _advertised(_Host(canonical), "place_plan") == "place_plan"


B2_SURFACES = (
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v1",
    "robotwin_toolchain_b2_closed_noservo_v2",
    "robotwin_toolchain_b2_closed_composed_v1",
    "robodojo_placement_closed_skills_v2",
    "robodojo_placement_closed_composed_v1",
)


@pytest.mark.parametrize("profile", B2_SURFACES)
def test_no_help_names_a_tool_the_surface_does_not_declare(profile):
    """Help may not send the policy after a call that is not on the surface.

    The closed arm's own `place_plan` help carried a five-step placement
    sequence naming `place_verify` and `open_gripper`. Neither is declared
    there -- absorbing `place_verify` is what the closed arm is for, and
    `open_gripper` retired into `set_gripper` a generation earlier. The
    policy read it, wrote "place_verify isn't in my tool list", worked out
    the mapping itself and fell back to raw motion: `place_execute` was
    called zero times across three officially successful rounds.

    A retired name in help is not cosmetic. Retirement is undeclaration, so
    the name stays syntactically plausible and the only thing telling the
    policy it is gone is its absence from a list of twenty.
    """

    from ora.agent.composition.tool_surface import resolve_tool_surface

    surface = resolve_tool_surface(profile)
    declared = {definition["name"] for definition in surface.definitions}
    # Every name this generation retired, in both spellings: the canonical
    # layer rewrites dotted names, so help written against the old surface
    # reaches the policy already snake_cased and looking current.
    retired = {
        "place_verify", "place.verify",
        "grasp_verify_retention", "grasp.verify_retention",
        "open_gripper", "close_gripper",
        "motion_probe", "motion.probe",
        "vision_offset", "vision.offset",
    } - declared
    offenders = {}
    for definition in surface.definitions:
        help_text = definition.get("description") or ""
        named = sorted(name for name in retired if name in help_text)
        if named:
            offenders[definition["name"]] = named
    assert not offenders, (
        f"{profile}: help names undeclared tools {offenders}. A surface that "
        "advertises a call it does not answer spends the policy's turns on "
        "discovering that."
    )


# --------------------------------------------------------------------------
# Retirement from the live closed surfaces (user decision 2026-09-29)
# --------------------------------------------------------------------------

#: Live closed-skill surface -> the revision that retired the tools. Every later
#: revision must keep them undeclared; the test pins the floor, not the current
#: revision (robodojo revision 8 adds only the D2 motion stall help).
LIVE_CLOSED = {
    # Integration 0059 unit A0 (user decision 2026-10-04, "v2 新面也退役"): the
    # v2 default generations retire the same tools from their first revision.
    "robodojo_placement_closed_skills_v2": 1,
    "robotwin_toolchain_b2_closed_noservo_v2": 1,
}
#: The LIBERO v2 default is the direct surface (no closed skills); it retires
#: place_verify with the placement family, since the placement support it
#: requires is retired with place_plan. Since integration 0084 the default is
#: its overhead successor; direct_v2 stays selectable but no bundle binds it.
LIVE_DIRECT = {"libero_placement_direct_overhead_v1": 1}
RETIRED_FROM_LIVE = (*RETIRED_PLACEMENT, "grasp_verify_retention")
#: Both spellings: frozen help written against dotted names reaches a canonical
#: surface already snake_cased, so either form would be a live reference.
RETIRED_SPELLINGS = (*RETIRED_FROM_LIVE, "place.plan", "place.execute", "place.resume",
                     "grasp.verify_retention")
#: Task-contract registries delivered as the episode-task layer on these backends.
TASK_CONTRACT_BINDINGS = (
    "ora/policy/environments/robotwin/task-contracts/registry-v4.json",
    "ora/policy/environments/robodojo/task-contracts/registry-v8.json",
)


def _bundles_delivering(profile: str) -> list[Path]:
    """Every prompt bundle a surface is delivered with: named by the bundle or bound by a semantic manifest."""
    import yaml

    bundles = set()
    for path in sorted((ROOT / "ora/policy/prompt_bundles").rglob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        layers = value.get("layers") if isinstance(value, dict) else None
        surface = layers.get("tool_surface") if isinstance(layers, dict) else None
        if isinstance(surface, dict) and surface.get("profile") == profile:
            bundles.add(path)
    for path in sorted((ROOT / "ora/policy/capabilities").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for cell, instance in (value.get("instances") or {}).items():
            if isinstance(instance, dict) and instance.get("surface") == profile:
                bundles.add(ROOT / value["prompt_bundles"][cell])
    return sorted(bundles)


def _delivered_prompt_files(profile: str) -> set[Path]:
    from ora.agent.composition.prompt_bundle import audit_prompt_bundle, load_prompt_bundle

    files = set()
    for bundle in _bundles_delivering(profile):
        report = audit_prompt_bundle(load_prompt_bundle(bundle), project_root=ROOT)
        assert report["passed"], bundle
        # Every layer source, including the manifest-only tool manual.
        files |= {ROOT / item["source_path"] for item in report["delivery_inventory"] if item["source_path"]}
        profile_path = (report.get("environment_profile") or {}).get("profile_path")
        if profile_path:
            files.add(ROOT / profile_path)
    for registry in TASK_CONTRACT_BINDINGS:
        value = json.loads((ROOT / registry).read_text(encoding="utf-8"))
        for entry in value.get("tasks", value).values():
            if isinstance(entry, dict) and (entry.get("layer") or entry.get("path")):
                files.add(ROOT / (entry.get("layer") or entry["path"]))
    return files


def _named(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(name for name in RETIRED_SPELLINGS if name in lowered)


@pytest.mark.parametrize("profile", sorted({**LIVE_CLOSED, **LIVE_DIRECT}))
def test_the_live_closed_surfaces_retire_placement_and_retention_by_undeclaration(profile):
    """Neither the tool list nor anything delivered with it names a retired tool.

    Retirement is undeclaration: the manifest excludes the tool, the loader
    skips every shared layer entry for it, and the handler stays for the
    frozen profiles and sealed runs that still reference it.
    """
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.tool_surface import PROFILE_CATALOG

    surface = resolve_tool_surface(profile)
    # Retired at LIVE_CLOSED[profile]; the checks below hold every later revision to it.
    assert surface.revision >= {**LIVE_CLOSED, **LIVE_DIRECT}[profile]
    names = [item["name"] for item in surface.definitions]
    manifest = json.loads(PROFILE_CATALOG[profile].read_text(encoding="utf-8"))
    retired = (*RETIRED_FROM_LIVE, *(("place_verify",) if profile in LIVE_DIRECT else ()))
    for name in retired:
        assert name not in names and name not in surface.result_schemas, name
        assert name not in {*surface.mutation_tools, *surface.lifecycle_tools, *surface.observation_tools}
        assert name not in manifest["tool_order"] and name in manifest["exclude_tools"], name
    assert RETIRED_CAPABILITIES.isdisjoint(surface.capabilities)
    assert ("grasp_execute" in names) is (profile in LIVE_CLOSED)
    assert {"move_pose", "set_gripper"} <= set(names)
    # Help, input schemas and descriptions of every declared tool.
    offenders = {item["name"]: _named(item["description"] + json.dumps(item["inputSchema"]))
                 for item in surface.definitions}
    assert not {name: found for name, found in offenders.items() if found}, profile
    # Prompt layers, tool manuals, environment profiles and task contracts.
    delivered = _delivered_prompt_files(profile)
    assert _bundles_delivering(profile), profile  # not vacuous: a bundle binds it
    found = {str(path.relative_to(ROOT)): _named(path.read_text(encoding="utf-8")) for path in delivered}
    assert not {path: names for path, names in found.items() if names}, profile
    # No semantic manifest instantiates a retired tool on a cell bound to it.
    for path in sorted((ROOT / "ora/policy/capabilities").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        cells = {cell for cell, instance in (value.get("instances") or {}).items()
                 if isinstance(instance, dict) and instance.get("surface") == profile}
        for family in (value.get("families") or {}).values():
            for cell in cells:
                entry = family["instances"][cell]
                if entry.get("instantiated"):
                    tools = entry.get("tools", family["semantic_tools"])
                    assert set(RETIRED_FROM_LIVE).isdisjoint(tools), (path.name, cell, tools)
    # Undeclared here, not deleted: robotwin_toolchain_b2_closed_noservo_v1 still advertises them.
    for handler in ("_place_plan", "_grasp_verify_retention", "_closed_skill"):
        assert callable(getattr(ManipulationToolRuntime, handler)), handler
