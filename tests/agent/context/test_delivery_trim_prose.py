"""Fixed prose leaves a result only where a description states it, word for word.

Every entry of ``delivery_trim_prose.ENTRIES`` is the exact text its producer
writes, and the composed description of its tool on each listed backend's v2
surface states it once (in the degree form that surface delivers). A text that
differs by a character stays in the result.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.context import delivery_trim as trim
from ora.agent.context.delivery_trim_prose import ENTRIES, PROSE_KEYS
from ora.agent.context.policy_units import degrees_text

ROOT = Path(__file__).resolve().parents[3]
V2_SURFACES = {
    "robodojo": "robodojo_placement_closed_skills_v2",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}


def _constant(reference: str) -> str:
    module, _, name = reference.partition(":")
    value = importlib.import_module(module)
    for step in name.split("."):
        value = value[step] if isinstance(value, dict) else getattr(value, step)
    return value


def _inline_literals(path: str) -> set[str]:
    tree = ast.parse((ROOT / "ora" / path).read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


#: Where each text comes from, as the producer writes it.
PRODUCERS = {
    "OBSERVED_CLEARANCE": lambda: _constant("ora.agent.tools.observed_clearance:CLAIM_BOUNDARY"),
    "ROBODOJO_ONE_ARM_IK_SCOPE": lambda: _constant(
        "ora.environment.backends.robodojo.ik:INTER_ARM_PATH_EVIDENCE_SCOPE"),
    "ROBODOJO_BOTH_ARMS_IK_SCOPE": lambda: _constant(
        "ora.environment.backends.robodojo.ik:PATH_EVIDENCE_SCOPE"),
    "ROBODOJO_REACH": lambda: _constant("ora.agent.tools.move_pose_reach:REACH_CLAIM"),
    "ROBODOJO_NEAREST_ORIENTATION": lambda: _constant(
        "ora.agent.composition.reach_orientation_schema:CLAIM"),
    "ROBOTWIN_REACH": lambda: _robotwin_reach(),
    "ROBOTWIN_REACH_UNSOLVED": lambda: _unsolved(_robotwin_reach()),
    "LIBERO_REACH": lambda: _constant("ora.environment.backends.libero_pro.reach_check:CLAIM_BOUNDARY"),
    "LIBERO_REACH_UNSOLVED": lambda: _unsolved(
        _constant("ora.environment.backends.libero_pro.reach_check:CLAIM_BOUNDARY")),
    "NEAREST_ORIENTATION": lambda: _constant("ora.agent.composition.reach_reference_schema:NEAREST_CLAIM"),
    "GRIPPER_STALL": lambda: _constant(
        "ora.environment.controllers.gripper_stall:GRIPPER_STALL_CLAIM_BOUNDARY"),
    "GRASP_PLAN_IK_REFERENCE": lambda: _constant("ora.agent.tools.reach_check:GRASP_ANNOTATION_CLAIM"),
    "GRASP_PLAN_JAW_SYMMETRY": lambda: _constant(
        "ora.agent.composition.grasp_jaw_symmetry_schema:CLAIM_BOUNDARY"),
    "GRASP_EXECUTE": lambda: _constant("ora.agent.tools.skill_driver:CLAIM_BOUNDARY"),
    "GRASP_JAW_HOLD": lambda: _constant("ora.agent.tools.jaw_hold:CLAIM_BOUNDARY"),
}
#: Texts written inline: the literal must occur in that producer's source.
INLINE = {
    "LIBERO_PLANNING_NOTE": "environment/backends/libero_pro/planning_diagnostics.py",
    "SEGMENTATION": "agent/tools/segmentation_batch.py",
    "GRASP_EXECUTE_PRECOMMIT": "agent/tools/grasp_feasibility.py",
    "GRASP_VERIFICATION": "agent/tools/localization.py",
    "GRASP_LIFT_PROBE": "agent/tools/skill_retention.py",
    "WAIT": "contracts/wait.py",
    "ROBOTWIN_ALIGN": "environment/controllers/reachability.py",
    "ROBODOJO_ALIGN": "environment/backends/robodojo/reachability.py",
}


def _robotwin_reach() -> str:
    module = importlib.import_module("ora.environment.backends.robotwin.reach_inter_arm")
    return module.CLAIM_BOUNDARIES[module.ENDPOINT]


def _unsolved(claim: str) -> str:
    """``reach_reference.unsolved``: the sampling claim, a space, the backend's claim."""
    return f"{_constant('ora.agent.tools.reach_reference:SAMPLING_CLAIM')} {claim}"


@pytest.fixture(scope="module")
def surfaces():
    return {backend: resolve_tool_surface(profile) for backend, profile in V2_SURFACES.items()}


def test_every_entry_has_one_producer_and_a_known_key():
    names = [entry.name for entry in ENTRIES]
    assert len(names) == len(set(names))
    assert set(PRODUCERS) | set(INLINE) == set(names)
    assert not set(PRODUCERS) & set(INLINE)
    for entry in ENTRIES:
        assert entry.key in PROSE_KEYS, entry.name
        assert set(entry.backends) <= set(V2_SURFACES), entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_each_text_is_what_its_producer_writes(entry):
    if entry.name in PRODUCERS:
        assert PRODUCERS[entry.name]() == entry.text
    else:
        assert entry.text in _inline_literals(INLINE[entry.name]), entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_each_text_is_stated_in_its_tool_description_on_every_listed_backend(entry, surfaces):
    for backend in entry.backends:
        surface = surfaces[backend]
        # The fixed-prose rule is in v1 and in v2, which the default surfaces declare.
        assert trim.RULE_FIXED_PROSE in trim.declared_rules(surface.capabilities), backend
        description = next(item["description"] for item in surface.definitions if item["name"] == entry.tool)
        assert entry.text in description or degrees_text(entry.text) in description, (backend, entry.name)


def test_only_an_exact_canonical_text_leaves():
    observed = next(entry for entry in ENTRIES if entry.name == "OBSERVED_CLEARANCE")
    result = {"compiled_pose": {"targets": {"right": {"observed_clearance": {
        "claim_boundary": observed.text, "clearance_m": -0.0126}}}},
        "other": {"claim_boundary": observed.text + " "}, "note": "free text the tool wrote now"}
    delivered = trim.drop_canonical_prose(result)
    assert delivered["compiled_pose"]["targets"]["right"]["observed_clearance"] == {"clearance_m": -0.0126}
    assert delivered["other"] == {"claim_boundary": observed.text + " "}
    assert delivered["note"] == "free text the tool wrote now"
    # The same text under a key no rule names stays.
    assert trim.drop_canonical_prose({"summary": observed.text}) == {"summary": observed.text}


def test_the_degree_form_a_surface_delivers_also_matches():
    libero = next(entry for entry in ENTRIES if entry.name == "LIBERO_REACH")
    assert "rad" in libero.text and degrees_text(libero.text) != libero.text
    assert trim.drop_canonical_prose({"claim_boundary": degrees_text(libero.text)}) == {}
