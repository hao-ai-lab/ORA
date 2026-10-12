"""compute publishes and refuses from one allow-list, and four evaluator defects are gone.

EXP11A g016: GPT-6 Astra wrote ``p[2] = ...`` (7x), ``p[2] += ...`` (1x) and
``np.diag`` (1x) against a description that named none of them, and the
refusals named an AST class. ``compute_grammar_from_evaluator_v1`` appends a
section generated from ``agent/tools/compute_grammar.py`` -- the same tables
``validate_source`` enforces. The evaluator's capabilities are unchanged:
subscript assignment stays refused (it silently truncates a float into an
integer array), and the refusal now says how to build the changed value.

Census C6 also fixed here: a 0-d array (``np.where`` of scalars) was reported as
"more than two dimensions"; ``x ** -2`` was refused as a non-literal exponent;
binding refusals named ``expressions`` instead of ``bindings[i]``; the
64-element value bound and the 120-character ``from`` bound were unenforced.
"""

from __future__ import annotations

import ast

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import compute_arguments
from ora.agent.tools.compute import RestrictedComputeRuntime, validate_source
from ora.agent.tools.compute_grammar import (
    _ALLOWED_NODES, _NAMED_REJECTIONS, _NODE_LABELS, RESERVED_NAMES, grammar, grammar_text,
)
from ora.agent.tools.robot import ToolError

DECLARING = (
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v1",
    "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_skills_v2",
)


@pytest.fixture(scope="module")
def runtime():
    instance = RestrictedComputeRuntime()
    try:
        yield instance
    finally:
        instance.close()


def _refusal(source: str) -> dict:
    with pytest.raises(ToolError) as caught:
        validate_source(source)
    return caught.value.as_dict()


def test_every_admitted_node_has_a_published_label():
    assert set(_ALLOWED_NODES) <= set(_NODE_LABELS)


@pytest.mark.parametrize("profile", DECLARING)
def test_the_published_section_is_the_generated_text_once_at_the_end(profile):
    compute = next(item for item in resolve_tool_surface(profile).definitions if item["name"] == "compute")
    assert compute["description"].endswith(grammar_text())
    assert compute["description"].count("Evaluator grammar (generated") == 1


def test_the_text_states_every_refusal_and_every_reserved_name():
    text = grammar_text()
    for label in {*_NAMED_REJECTIONS.values(), "subscript or attribute assignment (x[i] = v)"}:
        assert label in text, label
    for name in RESERVED_NAMES:
        assert name in text, name


def test_subscript_and_augmented_assignment_are_refused_by_name_with_a_recovery():
    subscript = _refusal("p = np.array([1.0, 2.0, 3.0])\np[2] = 0.5")
    assert "subscript and attribute assignment are not available" in subscript["message"]
    assert subscript["constraints"]["allowed_statements"] == grammar()["statements"]
    assert "np.array([p[0], p[1], z])" in subscript["constraints"]["recovery"]
    augmented = _refusal("x = 1.0\nx += 2.0")
    assert augmented["received"] == "augmented assignment (x += v)"
    assert augmented["constraints"]["recovery"] == "Assign the new value to a name: `x = x + v`."


def test_an_unreachable_attribute_lists_what_is_reachable():
    error = _refusal("m = np.diag([1.0, 2.0])")
    assert error["received"] == "diag" and "linalg" in error["constraints"]["reachable"]


def test_a_signed_literal_exponent_is_a_literal(runtime):
    validate_source("y = 2.0 ** -2")
    assert runtime.evaluate("y = 2.0 ** -2", {})["outputs"] == [{"name": "y", "value": 0.25, "kind": "scalar"}]
    assert _refusal("y = 2.0 ** -9")["constraints"]["maximum_absolute_exponent"] == 8
    assert _refusal("e = 2\ny = 2.0 ** e")["message"].startswith("the exponent of ** must be a numeric literal")


def test_a_zero_dimensional_array_is_a_scalar(runtime):
    result = runtime.evaluate("s = np.where(1.0 > 0.0, 2.0, 3.0)\nflag = np.array(True)", {})
    assert {entry["name"]: (entry["value"], entry["kind"]) for entry in result["outputs"]} == {
        "s": (2.0, "scalar"), "flag": (True, "boolean")}


@pytest.mark.parametrize(("binding", "field", "constraint"), [
    ({"name": "np", "value": 1}, "bindings[0].name", "reserved"),
    ({"name": "_x", "value": 1}, "bindings[0].name", None),
    ({"name": "a", "value": list(range(65))}, "bindings[0].value", "maximum_elements"),
    ({"name": "a", "value": 1, "from": "x" * 121}, "bindings[0].from", "maximum_characters"),
])
def test_binding_refusals_name_their_entry(binding, field, constraint):
    with pytest.raises(ToolError) as caught:
        compute_arguments.read_bindings([binding])
    error = caught.value.as_dict()
    assert error["field"] == field
    if constraint:
        assert constraint in error["constraints"]


def test_a_named_rejection_is_still_an_ast_class_the_allow_list_omits():
    assert not set(_NAMED_REJECTIONS) & set(_ALLOWED_NODES)
    assert ast.AugAssign in _NAMED_REJECTIONS
