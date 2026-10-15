"""D2: every angle a policy sees is in degrees (``policy_angles_degrees_v1``).

Receipts reported radians (an acceptance residual of 2.09 rad against a 0.1
rad tolerance). One explicit table maps each radian key to its ``_deg`` key;
canonical evidence stays in radians and only the delivered copy is converted.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from pathlib import Path

import pytest

from tests.support.delivery_lessons_fixtures import (
    DEFAULT_PROFILES, DEGREES, composed_with, libero_receipt, project, robodojo_receipt,
    with_capabilities,
)
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import ToolSurfaceError, resolve_tool_surface
from ora.agent.context.policy_units import (
    CAPABILITY, RADIAN_INPUT_KEYS, RADIAN_TO_DEGREE_KEYS, degrees_text, policy_degrees,
)

ROOT = Path(__file__).resolve().parents[3]
RAD_KEY = re.compile(r"(?:[a-z0-9_]+\.)*([a-z][a-z0-9_]*_rad)")
#: The direct_v3 default surfaces that are still registered; none declares the degree rule.
LEGACY_PROFILES = {backend: profile for backend, profile in DEFAULT_PROFILES.items()
                   if profile in PROFILE_CATALOG}
#: The RoboDojo default surface, which declares the degree rule.
DOJO_DEGREES = "robodojo_placement_closed_skills_v2"


def _source_radian_keys() -> set[str]:
    """Every radian key a producer can write: Python string literals and JSON keys."""
    found: set[str] = set()
    for path in (ROOT / "ora").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = RAD_KEY.fullmatch(node.value)
                if match:
                    found.add(match.group(1))

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9_]*_rad", key):
                    found.add(key)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in (ROOT / "ora").rglob("*.json"):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(key)
            _keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _keys(child, found)
    return found


def test_capability_name_is_the_one_the_fixtures_declare():
    assert CAPABILITY == DEGREES


def test_one_table_names_every_radian_key_a_producer_can_write():
    """A radian key missing from the table would reach the policy unconverted."""
    missing = _source_radian_keys() - set(RADIAN_TO_DEGREE_KEYS) - set(RADIAN_INPUT_KEYS)
    assert not missing, f"add these to RADIAN_TO_DEGREE_KEYS: {sorted(missing)}"
    assert not set(RADIAN_TO_DEGREE_KEYS) & set(RADIAN_INPUT_KEYS)
    for key, renamed in RADIAN_TO_DEGREE_KEYS.items():
        stem = key[:-len("_rad")] if key.endswith("_rad") else key
        assert renamed == f"{stem}_deg"
    # The two unlabeled joint vectors the inventory names are in the same table.
    assert RADIAN_TO_DEGREE_KEYS["joint_target"] == "joint_target_deg"
    assert RADIAN_TO_DEGREE_KEYS["stored_joints"] == "stored_joints_deg"


def test_radian_inputs_stay_radian_where_a_policy_still_sends_them():
    """An input a policy types in radians keeps its name in what it reads back."""
    inputs: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                inputs.update(name for name in properties if name.endswith("_rad"))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for profile in PROFILE_CATALOG:
        for definition in resolve_tool_surface(profile).definitions:
            walk(definition["inputSchema"])
    # rotation_vector_rad is the input of the bounded Servo tool, which frozen
    # definition files still carry but every surface now withholds;
    # rotation_rotvec_rad is the servo-request input of the motion contracts,
    # which no registered surface exposes.
    assert inputs | {"rotation_vector_rad", "rotation_rotvec_rad"} == set(RADIAN_INPUT_KEYS)


def test_values_convert_and_keys_rename_without_touching_the_source():
    source = {
        "pose_error": {"orientation_rad": math.pi / 2, "translation_m": 0.01},
        "within_tolerance": {"orientation_rad": True, "position_m": False},
        "canonical_interval_rad": [-math.pi, math.pi],
        "unwind_target": {"joint_target": [math.pi, 0.0], "stored_joints": [0.0, math.pi / 4],
                          "correction_rad": [0.0, 2 * math.pi]},
        "action_abstraction": "joint_target", "joint_target": "unknown",
        "rotation_vector_rad": [0.1, 0.0, 0.0], "orientation_tolerance_rad": None,
        "image_base64": "QUJD 12 rad",
    }
    original = copy.deepcopy(source)
    delivered = policy_degrees(source)
    assert source == original
    assert delivered["pose_error"] == {"orientation_deg": 90.0, "translation_m": 0.01}
    assert delivered["within_tolerance"] == {"orientation_deg": True, "position_m": False}
    assert delivered["canonical_interval_deg"] == [-180.0, 180.0]
    assert delivered["unwind_target"] == {
        "joint_target_deg": [180.0, 0.0], "stored_joints_deg": [0.0, 45.0],
        "correction_deg": [0.0, 360.0]}
    # A string that names a mode, and a non-numeric value, are not joint vectors.
    assert delivered["action_abstraction"] == "joint_target"
    assert delivered["joint_target"] == "unknown"
    # Inputs a policy still types in radians keep their unit and their name.
    assert delivered["rotation_vector_rad"] == [0.1, 0.0, 0.0]
    assert delivered["orientation_tolerance_deg"] is None
    assert delivered["image_base64"] == "QUJD 12 rad"


@pytest.mark.parametrize(("text", "expected"), [
    ("current residual 3.3 mm / 0.0064 rad against 5.0 mm / 0.0500 rad",
     "current residual 3.3 mm / 0.367 deg against 5.0 mm / 2.86 deg"),
    ("IK branch is 1.234 rad away", "IK branch is 70.7 deg away"),
    ("limit +/-3.14 rad", "limit +/-179.91 deg"),
    ("a raw joint move larger than pi rad", "a raw joint move larger than 180 deg"),
    ("orientation_error_rad in radians", "orientation_error_deg in degrees"),
    ("Read joint_distance_start_rad and joint_distance_end_rad.",
     "Read joint_distance_start_deg and joint_distance_end_deg."),
    ('contract: {"orientation_tolerance_rad":0.1,"position_tolerance_m":0.01}',
     'contract: {"orientation_tolerance_deg":5.729578,"position_tolerance_m":0.01}'),
    ("joint_delta_rad=1.5, tolerance_rad: 0.1", "joint_delta_deg=85.943669, tolerance_deg: 5.729578"),
    ("math.radians(degrees) directly", "math.radians(degrees) directly"),
    ("Signed DEGREES, not radians.", "Signed DEGREES, not radians."),
    ("Finite axis-angle rotation_vector_rad in radians", "Finite axis-angle rotation_vector_rad in radians"),
    ("abc/12rad/+=", "abc/12rad/+="),
])
def test_free_text_states_the_same_numbers_in_degrees(text, expected):
    assert degrees_text(text) == expected


def test_a_robodojo_effect_reaches_the_policy_without_one_radian_key():
    legacy = project(robodojo_receipt(), ())
    delivered = project(robodojo_receipt(), (DEGREES, "current_public_robot_facts_v1"))
    radian = {key for key in _keys(delivered) if key.endswith("_rad")}
    assert radian <= set(RADIAN_INPUT_KEYS), radian
    check = delivered["detail"]["acceptance_checks"][0]
    assert check["tolerance"] == {"orientation_deg": 5.729578, "position_m": 0.01}
    assert check["arms"]["left"]["residual"]["orientation_deg"] == round(math.degrees(0.0282), 6)
    arm = delivered["detail"]["arms"]["left"]
    assert arm["pose_error"]["orientation_deg"] == round(math.degrees(0.0282), 6)
    assert arm["segments"][0]["joint_target_deg"][1] == round(math.degrees(1.7734), 6)
    assert delivered["detail"]["controller_feedback"]["orientation_tolerance_deg"] == 5.729578
    # Only the unit changes: everything that is not an angle is the legacy value.
    assert delivered["detail"]["arms"]["left"]["pose_error"]["translation_m"] == 0.0087
    assert legacy["detail"]["arms"]["left"]["pose_error"]["orientation_rad"] == 0.0282


def test_free_text_inside_an_effect_follows_the_unit():
    delivered = project(libero_receipt(), (DEGREES, "current_public_robot_facts_v1"))
    contract = delivered["detail"]["arms"]["right"]["endpoint_contract"]
    assert contract["summary"].endswith("current residual 3.3 mm / 0.367 deg against 5.0 mm / 2.86 deg.")
    assert contract["judged_residual"] == {"position_m": 0.0033, "orientation_deg": 0.366693}


def test_facts_and_tool_results_deliver_degrees_only_when_declared():
    from ora.agent.context.policy_delivery import deliver_public_facts, deliver_tool_result

    facts = {"observation_id": "o-1", "controller_feedback": {"orientation_tolerance_rad": 0.1}}
    assert deliver_public_facts(facts, capabilities=()) is facts
    assert deliver_public_facts(facts, capabilities=(DEGREES,))["controller_feedback"] == {
        "orientation_tolerance_deg": 5.729578}
    surface = with_capabilities(resolve_tool_surface(LEGACY_PROFILES["robotwin"]))
    assert DEGREES not in surface.capabilities
    result = {"candidates": [{"orientation_error_rad": 0.5, "position_error_m": 0.01}],
              "error": None, "images_base64": ["QUJD 1 rad"]}
    assert deliver_tool_result(surface, "align_plan", result) is result
    degree_surface = with_capabilities(surface, DEGREES)
    delivered = deliver_tool_result(degree_surface, "align_plan", result)
    assert delivered["candidates"] == [{"orientation_error_deg": 28.64789, "position_error_m": 0.01}]
    assert delivered["images_base64"] == ["QUJD 1 rad"]


def test_policy_authored_results_are_never_rewritten():
    """compute output and Memory notes are the policy's own words and names."""
    from ora.agent.context.policy_delivery import deliver_tool_result

    surface = with_capabilities(resolve_tool_surface(LEGACY_PROFILES["robotwin"]), DEGREES)
    computed = {"value": {"rotation_rad": 0.5}, "stdout": "yaw 0.5 rad"}
    assert deliver_tool_result(surface, "compute", computed) is computed
    note = {"content": "turned 0.5 rad last round"}
    assert deliver_tool_result(surface, "memory.write", note) is note
    assert deliver_tool_result(surface, "artifact_read", note) is note


def test_the_tool_gateway_delivers_the_degree_result():
    """Native results pass the delivery rule, after the canonical record."""
    from ora.agent.native.gateway import NativeToolGateway

    surface = with_capabilities(resolve_tool_surface(LEGACY_PROFILES["robotwin"]), DEGREES)
    canonical = {"status": "ok", "candidates": [{"orientation_error_rad": 0.5}]}

    class Runtime:
        def __init__(self):
            self.surface = surface

        def definitions(self):
            return []

        def invoke(self, name, arguments):
            return copy.deepcopy(canonical)

    native = NativeToolGateway(Runtime()).record_proposed_result(
        "align_plan", {"payload": {"arguments": {}}}, copy.deepcopy(canonical))
    assert native["candidates"] == [{"orientation_error_deg": 28.64789}]
    assert canonical["candidates"] == [{"orientation_error_rad": 0.5}]


@pytest.mark.parametrize("backend", sorted(LEGACY_PROFILES))
def test_help_text_on_a_degree_surface_names_no_radian_quantity(monkeypatch, backend):
    legacy = resolve_tool_surface(LEGACY_PROFILES[backend])
    surface = composed_with(monkeypatch, LEGACY_PROFILES[backend], DEGREES)
    assert surface.composition_sha256 != legacy.composition_sha256
    assert {item["slot"] for item in surface.components} - {item["slot"] for item in legacy.components} == {
        "derived:policy_angles_degrees_v1:policy_units_schema",
        "derived:policy_angles_degrees_v1:policy_units"}
    pattern = re.compile(r"\d rad\b|\bpi rad\b|\b([a-z][a-z0-9_]*_rad)\b")

    def descriptions(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "description" and isinstance(child, str):
                    yield child
                else:
                    yield from descriptions(child)
        elif isinstance(value, list):
            for child in value:
                yield from descriptions(child)

    for definition in surface.definitions:
        for text in descriptions(definition):
            for match in pattern.finditer(text):
                assert match.group(1) in RADIAN_INPUT_KEYS, (definition["name"], match.group(0))
    # Only descriptions change: names, schemas and result contracts do not.
    for old, new in zip(legacy.definitions, surface.definitions):
        assert old["name"] == new["name"]
    assert surface.result_schemas == legacy.result_schemas


def test_generated_controller_rule_is_restated_in_degrees():
    surface = resolve_tool_surface(DOJO_DEGREES)
    assert DEGREES in surface.capabilities
    text = next(item["description"] for item in surface.definitions if item["name"] == "set_gripper")
    assert "orientation error <= 5.73 deg" in text
    assert '"orientation_tolerance_deg":5.729578' in text
    align = next(item["description"] for item in surface.definitions if item["name"] == "align_plan")
    assert "orientation_error_deg in degrees" in align
    assert "larger than 180 deg" in align


def test_degree_binder_refuses_a_radian_top_level_result_property():
    """Native projects only declared top-level result keys; a renamed one would vanish."""
    from ora.agent.composition.policy_units_schema import bind_policy_angles_degrees

    results = {"probe": {"type": "object", "properties": {"orientation_error_rad": {"type": "number"}}}}
    with pytest.raises(ToolSurfaceError, match="orientation_error_rad"):
        bind_policy_angles_degrees([], results, [DEGREES])
