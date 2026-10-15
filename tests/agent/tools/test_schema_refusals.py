"""Generic input-schema refusals name the field, the violated rule and what arrived.

EXP11A g016: ``mark_point`` with a flattened ``points_uv`` came back as
"$input.points_uv[0] has the wrong input type" with ``field``, ``received``
and ``constraints`` empty, and the census found the validator silently skipping
``uniqueItems``, ``minProperties``, ``maxLength``, ``pattern`` and ``anyOf`` on
the delivered surface. Every input keyword the model is shown is now checked,
and every mismatch is one structured ``invalid_arguments``.
"""

from __future__ import annotations

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import (
    SchemaViolation, ToolSurfaceError, lint_input_schema, validate_schema_instance,
)
from ora.agent.tools.tool_errors import ToolError, schema_refusal

DOJO = "robodojo_placement_closed_skills_v2"


def _input(name):
    return next(item for item in resolve_tool_surface(DOJO).definitions if item["name"] == name)["inputSchema"]


def _refusal(name, arguments) -> dict:
    with pytest.raises(SchemaViolation) as caught:
        validate_schema_instance(arguments, _input(name), root_path="$input")
    return schema_refusal(caught.value).as_dict()


def test_a_flattened_pixel_list_names_the_field_the_type_and_what_arrived():
    error = _refusal("mark_point", {"view": "head", "point_ids": ["a"], "points_uv": [316, 191]})
    assert error["code"] == "invalid_arguments" and error["retry_requires_changed_arguments"] is True
    assert error["field"] == "points_uv[0]"
    assert error["received"] == {"type": "integer"}
    assert error["constraints"] == {"violated": "type", "type": ["array"]}
    assert "expected array" in error["message"]


@pytest.mark.parametrize(("name", "arguments", "violated", "field"), [
    ("set_gripper", {"targets": {}}, "minProperties", "targets"),
    ("vision_ground", {"camera": "head", "text": "x" * 121}, "maxLength", "text"),
    ("grasp_plan", {"arm": "left", "target_ref": "not-a-ref"}, "pattern", "target_ref"),
    ("mark_point", {"view": "head", "point_ids": ["a", "a"], "points_uv": [[1, 1], [2, 2]]}, "uniqueItems", "point_ids"),
    ("set_gripper", {"targets": {"left": "grip"}}, "enum", "targets.left"),
    ("geometry_query", {"camera": "head", "points": [[1, 1]], "window_px": 99}, "maximum", "window_px"),
    ("geometry_query", {"camera": "head", "region": [0, 0, 1]}, "minItems", "region"),
    ("observe", {"views": ["head"], "bogus": 1}, "additionalProperties", None),
])
def test_every_input_keyword_is_checked_and_reported(name, arguments, violated, field):
    error = _refusal(name, arguments)
    assert error["constraints"]["violated"] == violated
    assert error["field"] == field
    assert "received" in error and error["received"]["type"]


def test_received_describes_the_shape_never_the_values():
    error = _refusal("observe", {"views": ["head"], "bogus": "secret-value"})
    assert error["received"] == {"type": "object", "names": ["bogus", "views"]}
    assert error["constraints"]["unknown"] == ["bogus"]
    assert "secret-value" not in repr(error)


def test_an_enum_and_a_bound_are_stated_in_the_message():
    assert "['open', 'close', None]" in _refusal("set_gripper", {"targets": {"left": "grip"}})["message"]
    assert "maximum 15" in _refusal("geometry_query", {"camera": "head", "points": [[1, 1]], "window_px": 99})["message"]


def test_results_keep_the_keyword_set_they_were_checked_against():
    schema = {"type": "object", "properties": {"ref": {"type": "string", "pattern": "^target-"}}}
    validate_schema_instance({"ref": "other"}, schema)  # results: pattern stays unchecked
    with pytest.raises(SchemaViolation):
        validate_schema_instance({"ref": "other"}, schema, root_path="$input")


def test_any_of_must_match_at_least_one_branch_on_inputs():
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    validate_schema_instance(3, schema, root_path="$input")
    with pytest.raises(SchemaViolation, match="at least one input branch"):
        validate_schema_instance(1.5, schema, root_path="$input")


def test_composition_refuses_an_input_keyword_the_runtime_does_not_check():
    lint_input_schema({"type": "object", "properties": {"x": {"type": "number", "maximum": 2,
                                                              "description": "d", "default": 1}}}, source="t")
    with pytest.raises(ToolSurfaceError, match="multipleOf"):
        lint_input_schema({"type": "object", "properties": {"x": {"type": "number", "multipleOf": 2}}}, source="t")


def test_the_live_runtime_refuses_through_the_structured_envelope(tmp_path):
    from tests.agent.motion.test_observed_clearance_runtime import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    with pytest.raises(ToolError) as caught:
        runtime.invoke("set_gripper", {"targets": {}})
    error = caught.value.as_dict()
    assert error["field"] == "targets" and error["constraints"]["violated"] == "minProperties"
