"""An optional argument can be sent as null, and null means omitted.

EXP11A g016 (EXP11B g003): GPT-6 Astra filled all five optional
``geometry_query`` properties on every one of 160 calls, inventing
``region`` [0,0,1,1] and ``world_points_m`` [[0,0,0]] for the unused ones,
and the handler refused all 153 that reached it. The one optional property it
left null was ``step`` -- the only one already typed ``["integer", "null"]``.
``optional_argument_null_as_omitted_v1`` gives every top-level optional
argument that legal "unused" value, on every protocol and both executors,
without special-casing any placeholder.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.null_optional_arguments import (
    CAPABILITY, NULL_SENTENCE, accepts_null, bind_null_optional_arguments,
    omit_null_optional_arguments, optional_argument_names,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.native.gateway import NativeToolGateway, _openai_parameters, openai_tool_definitions
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import translate_anthropic_request
from ora.agent.tools.tool_errors import ToolError
from test_runtime_v1 import publish_synthetic

#: The default surfaces and the RoboTwin v1 surface, which declare the rule.
DECLARING = (
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v1",
    "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_skills_v2",
)
DOJO = "robodojo_placement_closed_skills_v2"
OPERATIONS = ("points", "region", "world_points_m")


def _definition(surface, name):
    return next(item for item in surface.definitions if item["name"] == name)


def _properties(schema):
    branches = [schema, *schema.get("oneOf", []), *schema.get("anyOf", [])]
    merged = {}
    for branch in branches:
        merged.update(branch.get("properties", {}))
    return merged


@pytest.mark.parametrize("profile", DECLARING)
def test_every_optional_argument_accepts_null_and_no_required_one_does(profile):
    surface = resolve_tool_surface(profile)
    assert CAPABILITY in surface.capabilities
    for definition in surface.definitions:
        schema = definition["inputSchema"]
        optional = optional_argument_names(schema)
        for name, prop in _properties(schema).items():
            assert accepts_null(prop) is (name in optional), (definition["name"], name)


def test_geometry_query_delivers_null_for_its_unused_operations():
    surface = resolve_tool_surface(DOJO)
    properties = _definition(surface, "geometry_query")["inputSchema"]["properties"]
    for name in (*OPERATIONS, "window_px"):
        assert properties[name]["type"][-1] == "null", name
        assert properties[name]["description"].endswith(NULL_SENTENCE), name
    # Required and already-nullable arguments keep their bytes.
    assert properties["camera"]["type"] == "string"
    assert properties["step"]["type"] == ["integer", "null"]
    assert not properties["step"]["description"].endswith(NULL_SENTENCE)
    assert properties["region"]["minItems"] == properties["region"]["maxItems"] == 4


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_every_protocol_delivers_the_same_nullable_schema(protocol):
    surface = resolve_tool_surface(DOJO)
    definition = _definition(surface, "geometry_query")
    expected = definition["inputSchema"]
    tools = openai_tool_definitions(["geometry_query"], definitions=[definition])
    delivered = tools[0]["function"]["parameters"]
    if protocol == "anthropic":
        provider = SimpleNamespace(prepare_request=translate_anthropic_request)
        request = NativeRequestAssembler(
            model="offline", reasoning_effort="provider_default", max_output_tokens=64,
            stream=False, provider=provider,
        ).prepare({"tools": tools, "messages": [{"role": "user", "content": "x"}]})
        delivered = request.wire["tools"][0]["input_schema"]
    assert delivered == expected
    assert delivered["properties"]["region"]["type"] == ["array", "null"]


def test_a_surface_that_does_not_declare_the_rule_keeps_its_bytes():
    surface = resolve_tool_surface("robot_runtime_v1")
    assert CAPABILITY not in surface.capabilities
    schema = _definition(surface, "geometry.query")["inputSchema"]
    assert schema["properties"]["points"]["type"] == "array"


class _Recorder:
    def __init__(self, surface, handler=None):
        self.surface = surface
        self.calls = []
        self._handler = handler

    def definitions(self):
        return [dict(item) for item in self.surface.definitions]

    def invoke(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return self._handler(arguments) if self._handler else {"status": "ok"}

    def classify_call_effect(self, name, arguments):
        self.calls.append(("classify", dict(arguments)))
        return "read_only"


G016_UNUSED_AS_NULL = {
    "camera": "head", "points": [[2, 1]], "region": None, "step": None,
    "window_px": None, "world_points_m": None,
}


def test_the_native_gateway_hands_the_handler_an_omitted_argument():
    runtime = _Recorder(resolve_tool_surface(DOJO))
    gateway = NativeToolGateway(runtime)
    gateway.classify_effect("geometry_query", G016_UNUSED_AS_NULL)
    gateway.execute("geometry_query", G016_UNUSED_AS_NULL)
    assert runtime.calls == [
        ("classify", {"camera": "head", "points": [[2, 1]]}),
        ("geometry_query", {"camera": "head", "points": [[2, 1]]}),
    ]
    # A required argument is never dropped: its null still reaches the handler.
    gateway.execute("geometry_query", {"camera": None, "points": [[2, 1]]})
    assert runtime.calls[-1] == ("geometry_query", {"camera": None, "points": [[2, 1]]})


def _geometry_gateway(tmp_path, profile):
    _o, _p, _w, _c, tools, _s = publish_synthetic(tmp_path)
    runtime = _Recorder(
        resolve_tool_surface(profile),
        handler=lambda arguments: tools._geometry({**arguments, "camera": "head_camera"}),
    )
    return NativeToolGateway(runtime)


def test_exactly_one_of_counts_only_the_operations_actually_sent(tmp_path):
    gateway = _geometry_gateway(tmp_path, DOJO)
    result = gateway.execute("geometry_query", G016_UNUSED_AS_NULL)
    assert [entry["valid"] for entry in result["points"]] == [True]


def test_placeholder_values_are_still_values(tmp_path):
    gateway = _geometry_gateway(tmp_path, DOJO)
    with pytest.raises(ToolError) as caught:
        gateway.execute("geometry_query", {
            "camera": "head", "points": [[2, 1]], "region": [0, 0, 1, 1],
            "world_points_m": [[0, 0, 0]], "window_px": 2, "step": None,
        })
    assert caught.value.constraints == {"exactly_one_of": list(OPERATIONS), "received": list(OPERATIONS)}


def test_a_non_declaring_surface_still_refuses_an_explicit_null(tmp_path):
    gateway = _geometry_gateway(tmp_path, "robot_runtime_v1")
    with pytest.raises(ToolError) as caught:
        gateway.execute("geometry_query", G016_UNUSED_AS_NULL)
    assert caught.value.constraints["received"] == list(OPERATIONS)


def test_omission_returns_the_same_object_when_nothing_is_null():
    surface = resolve_tool_surface(DOJO)
    arguments = {"camera": "head", "points": [[2, 1]]}
    assert omit_null_optional_arguments(surface, "geometry_query", arguments) is arguments
    assert omit_null_optional_arguments(None, "geometry_query", G016_UNUSED_AS_NULL) is G016_UNUSED_AS_NULL


def test_a_root_union_gains_the_same_nullable_property_in_every_branch():
    definitions = [{"name": "submit", "description": "d", "inputSchema": {
        "oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["action", "pose"],
             "properties": {"action": {"type": "string", "enum": ["move"]}, "pose": {"type": "array"},
                            "note": {"type": "string", "description": "Free text."}}},
            {"type": "object", "additionalProperties": False, "required": ["action", "gripper"],
             "properties": {"action": {"type": "string", "enum": ["grip"]}, "gripper": {"type": "number"},
                            "note": {"type": "string", "description": "Free text."}}},
        ],
    }}]
    bind_null_optional_arguments(definitions)
    branches = definitions[0]["inputSchema"]["oneOf"]
    assert [branch["properties"]["note"]["type"] for branch in branches] == [["string", "null"]] * 2
    assert branches[0]["properties"]["pose"]["type"] == "array"
    assert branches[1]["properties"]["gripper"]["type"] == "number"
    flattened = _openai_parameters(definitions[0]["inputSchema"])
    assert flattened["properties"]["note"]["type"] == ["string", "null"]


def test_an_enum_gains_null_and_an_untyped_optional_argument_is_refused():
    definitions = [{"name": "t", "inputSchema": {"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["a", "b"], "description": "Mode."},
    }}}]
    bind_null_optional_arguments(definitions)
    mode = definitions[0]["inputSchema"]["properties"]["mode"]
    assert mode["type"] == ["string", "null"] and mode["enum"] == ["a", "b", None]
    untyped = [{"name": "t", "inputSchema": {"type": "object", "properties": {
        "choice": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
    }}}]
    with pytest.raises(ToolSurfaceError, match="t.choice"):
        bind_null_optional_arguments(copy.deepcopy(untyped))


# --- every depth, the final tool list, root unions, per-arm maps ------------

def _walk(schema, path, defs):
    """Yield (path, property schema, optional?) for every object member at every depth."""
    from ora.agent.composition.null_optional_arguments import _object_branches, _resolve

    schema = _resolve(schema, defs)
    if not isinstance(schema, dict):
        return
    optional = optional_argument_names(schema, defs)
    for branch in _object_branches(schema, defs):
        for name, child in branch["properties"].items():
            yield f"{path}.{name}", _resolve(child, defs), name in optional
            yield from _walk(child, f"{path}.{name}", defs)
    if isinstance(schema.get("items"), dict):
        yield from _walk(schema["items"], f"{path}[]", defs)
    for keyword in ("oneOf", "anyOf"):
        for branch in schema.get(keyword) or ():
            for name, child in (_resolve(branch, defs).get("properties") or {}).items():
                yield from _walk(child, f"{path}.{name}", defs)


def _delivered(profile):
    from ora.agent.composition.null_optional_arguments import deliver_tool_definitions
    from ora.agent.memory.artifacts import recoverable_artifact_definitions

    surface = resolve_tool_surface(profile)
    return deliver_tool_definitions(recoverable_artifact_definitions(surface.definitions), surface.capabilities)


@pytest.mark.parametrize("profile", DECLARING)
def test_the_delivered_list_is_flat_and_every_optional_member_accepts_null(profile):
    delivered = _delivered(profile)
    assert {"observe", "artifact_read"} <= {item["name"] for item in delivered}
    for item in delivered:
        schema = item["inputSchema"]
        assert "oneOf" not in schema and "anyOf" not in schema, item["name"]
        defs = schema.get("$defs", {})
        for path, prop, optional in _walk(schema, item["name"], defs):
            if optional:
                assert accepts_null(prop), path


def test_nested_optional_members_are_nullable_on_the_composed_surface():
    surface = resolve_tool_surface(DOJO)
    targets = _definition(surface, "set_gripper")["inputSchema"]["properties"]["targets"]
    assert targets["properties"]["left"]["enum"] == ["open", "close", None]
    assert targets["properties"]["left"]["description"].endswith(NULL_SENTENCE)
    arm = _definition(surface, "move_pose")["inputSchema"]["properties"]["targets"]["properties"]["left"]
    assert "null" in arm["type"]
    for companion in ("position_m", "translation_m", "orientation_xyzw", "approach_world"):
        assert accepts_null(arm["properties"][companion]), companion
    for required in arm.get("required", []):
        assert not accepts_null(arm["properties"][required]), required
    binding = _definition(surface, "compute")["inputSchema"]["properties"]["bindings"]["items"]
    assert accepts_null(binding["properties"]["from"])


def test_observe_ref_and_artifact_read_are_covered_once_delivered():
    delivered = {item["name"]: item["inputSchema"] for item in _delivered(DOJO)}
    observe = delivered["observe"]
    assert observe["required"] == [] and observe["properties"]["ref"]["type"] == ["string", "null"]
    reader = delivered["artifact_read"]
    assert reader["required"] == ["ref"]
    assert reader["properties"]["json_pointer"]["type"] == ["string", "null"]


def test_a_typeless_enum_with_null_is_already_nullable_and_one_without_gains_it():
    assert accepts_null({"enum": [None, 4096]})
    definitions = [{"name": "t", "inputSchema": {"type": "object", "properties": {
        "limit": {"enum": [None, 4096]}, "mode": {"enum": ["a", "b"], "description": "Mode."}}}}]
    bind_null_optional_arguments(definitions)
    properties = definitions[0]["inputSchema"]["properties"]
    assert properties["limit"] == {"enum": [None, 4096]}
    assert properties["mode"]["enum"] == ["a", "b", None] and "type" not in properties["mode"]


def test_a_root_union_that_cannot_be_flattened_fails_delivery():
    from ora.agent.composition.null_optional_arguments import deliver_tool_definitions

    definitions = [{"name": "t", "inputSchema": {"oneOf": [
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    ]}}]
    with pytest.raises(ToolSurfaceError, match="branch definitions of x differ"):
        deliver_tool_definitions(definitions, [CAPABILITY])
    assert deliver_tool_definitions(definitions, []) == definitions


def test_nulls_are_dropped_at_every_depth_and_an_all_null_map_is_the_empty_map():
    surface = resolve_tool_surface(DOJO)
    gateway = NativeToolGateway(_Recorder(surface))
    assert gateway.arguments("set_gripper", {"targets": {"left": "close", "right": None}, "target_ref": None}) == {
        "targets": {"left": "close"}}
    assert gateway.arguments("set_gripper", {"targets": {"left": None, "right": None}}) == {"targets": {}}
    nested = {"targets": {"left": {"interaction": None, "position_mode": "delta", "translation_m": [0, 0, 0.01],
                                    "position_frame": "world", "orientation_mode": "hold_current",
                                    "position_m": None, "orientation_xyzw": None}, "right": None}}
    assert gateway.arguments("move_pose", nested) == {"targets": {"left": {
        "position_mode": "delta", "translation_m": [0, 0, 0.01], "position_frame": "world",
        "orientation_mode": "hold_current"}}}
    compute = {"expressions": "y = x", "bindings": [{"name": "x", "value": 1, "from": None}], "outputs": None}
    assert gateway.arguments("compute", compute) == {"expressions": "y = x", "bindings": [{"name": "x", "value": 1}]}


@pytest.mark.parametrize(("tool", "commanded"), [
    ("set_gripper", "close"),
    ("rotate_wrist", 20.0),
    ("rotate_pitch", -10.0),
    ("move_to", {"position_m": [0.0, 0.0, 0.95757]}),
])
def test_a_filled_null_arm_is_never_commanded(tmp_path, tool, commanded):
    from test_observed_clearance_runtime import _consume, _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    gateway = NativeToolGateway(runtime)
    submitted = []
    for targets in ({"left": commanded}, {"left": commanded, "right": None}):
        gateway.execute(tool, {"targets": targets})
        action = json.loads((runtime.control_root / "submitted-action.json").read_text())
        submitted.append(action["arguments"])
        _consume(runtime)
    assert submitted[0] == submitted[1]
    assert set(submitted[1]["targets"]) == {"left"}
