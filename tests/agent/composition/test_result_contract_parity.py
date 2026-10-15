"""The two maintained backends publish the same result contracts, or say why.

The candidate ledger and the canonical localization were both recorded as "half
done, three tools three shapes". Measured, they were neither half-designed nor
inconsistent between tools: they were complete on LIBERO and degraded to
``{"type": "object"}`` on RoboTwin, for the same runtime producers. A contract
that exists on one backend and is an empty object on the other is not a design
gap, and looking for the missing design is how it stayed unfixed.

An unconstrained object node is the failure mode the authoring guide names for
results: a budget nobody bounded. It also reads to a policy as "anything may
appear here", which is the opposite of what the tool actually returns.
"""

from __future__ import annotations

import pytest

from ora.agent.composition import resolve_tool_surface


ARMS = ("libero", "robotwin")
LIVE = {"libero": "libero_placement_direct_overhead_v1", "robotwin": "robotwin_toolchain_b2_closed_noservo_v2"}

#: Shared on both backends, so not an asymmetry. `compiled_pose` is assembled
#: differently by each motion path and `view_dimensions` is keyed by camera
#: name, so bounding either means inventing a contract rather than publishing
#: one that already exists. On the defaults only recover_pose still leaves
#: `compiled_pose` open; the pose tools publish a per-target contract. Listed
#: so the count below cannot drift upward silently, and so the next person
#: does not rediscover them as new.
KNOWN_UNCONSTRAINED = {
    ("recover_pose", "/properties/compiled_pose"),
    ("observe", "/properties/view_dimensions"),
}


def _unconstrained(profile: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for tool, schema in resolve_tool_surface(profile).result_schemas.items():
        def walk(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                if (
                    node.get("type") == "object"
                    and "properties" not in node
                    and "additionalProperties" not in node
                    and "$ref" not in node
                    # A oneOf node is constrained only when every branch is closed.
                    and not all(branch.get("additionalProperties") is False or "$ref" in branch
                                for branch in node.get("oneOf") or [{}])
                ):
                    found.add((tool, path))
                for key, value in node.items():
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
        walk(schema)
    return found


@pytest.mark.parametrize("backend", ARMS)
def test_no_result_carries_an_unconstrained_object_beyond_the_known_two(backend: str):
    assert _unconstrained(LIVE[backend]) == KNOWN_UNCONSTRAINED


def test_both_backends_carry_exactly_the_same_unconstrained_set():
    """The asymmetry is the finding; parity is the fix."""
    assert _unconstrained(LIVE["libero"]) == _unconstrained(LIVE["robotwin"])


@pytest.mark.parametrize("tool,defs", [
    ("grasp_plan", ("backend", "diagnostics", "overlay",
                    "rejected_backend_candidate", "rejected_compiler_candidate")),
    ("vision_ground", ("backend", "diagnostics", "localization", "mask_summary",
                       "overlay")),
    ("vision_segment", ("diagnostics", "localization", "mask_summary")),
])
def test_the_same_named_contracts_exist_on_both_backends(tool: str, defs: tuple):
    """One name, one contract, both backends -- or the name means two things."""
    for backend in ARMS:
        schema = resolve_tool_surface(LIVE[backend]).result_schemas[tool]
        for name in defs:
            body = schema.get("$defs", {}).get(name)
            assert isinstance(body, dict), f"{backend}/{tool} lacks $defs.{name}"
            assert body.get("additionalProperties") is False, (
                f"{backend}/{tool} $defs.{name} is not closed"
            )


@pytest.mark.parametrize("backend", ARMS)
def test_the_grasp_candidate_ledger_is_required_not_merely_allowed(backend: str):
    """A rejection ledger a result may omit is a ledger a policy cannot rely on.

    `grasp_plan` is where a policy learns why the candidates it did not get were
    not offered: `rejected` carries the backend's per-candidate reasons and
    `compiler_rejected` the compiler's, beside the counts at every stage.
    """
    schema = resolve_tool_surface(LIVE[backend]).result_schemas["grasp_plan"]
    diagnostics = schema["$defs"]["diagnostics"]
    for field in ("rejected", "compiler_rejected", "compiler_returned_candidate_count"):
        assert field in diagnostics["required"], field
    for name in ("rejected", "compiler_rejected"):
        entry = diagnostics["properties"][name]["items"]["$ref"].rsplit("/", 1)[-1]
        required = schema["$defs"][entry]["required"]
        assert "candidate_id" in required, name
        # The two entries disagree on the plural: the backend ledger names
        # `reasons` and the compiler ledger `reason`. Asserted as it is rather
        # than as it should be -- renaming a published field is a surface
        # generation, and this test exists to stop the contracts drifting, not
        # to hide that they already have.
        assert {"reason", "reasons"} & set(required), name


@pytest.mark.parametrize("backend", ARMS)
def test_a_grounding_candidate_carries_the_same_localization_as_a_segment_one(
    backend: str,
):
    """Both are `grasp._mask_depth_localization` output, so both quote one contract."""
    surface = resolve_tool_surface(LIVE[backend])
    ground = surface.result_schemas["vision_ground"]
    segment = surface.result_schemas["vision_segment"]
    assert ground["$defs"]["localization"] == segment["$defs"]["localization"]
    assert ground["$defs"]["mask_summary"] == segment["$defs"]["mask_summary"]
    for schema in (ground, segment):
        assert schema["$defs"]["candidate"]["properties"]["localization"] == {
            "$ref": "#/$defs/localization"
        }
