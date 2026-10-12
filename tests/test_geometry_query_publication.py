"""`geometry_query` on the two live default surfaces: published, and answerable.

The capability was never missing. `RobotToolRuntime._geometry` has served
world->pixel projection, pixel->world back-projection and region depth all
along, and five of the 89 registered profiles -- three directories, because
`robot_runtime_v2/` registers three of them -- declared it as `geometry.query`.
What was missing was the publication: `ManipulationToolRuntime` inherits
the method but overrides `invoke` without delegating, so every surface built on
it -- which is most of them, including both defaults -- could not dispatch the
name however it was spelled. Across 177 sealed runs, 51 show the policy
deriving `u = cx + fx * x / z` by hand against intrinsics it read off
calibration, and those are exactly the runs whose surface omits the tool.

Two things are checked here that nothing else joins.

The **shape a policy is handed**: the published input schema drops the
`oneOf` the three frozen surfaces carry. `agent/native/gateway.py` can only
flatten a root union whose branches carry `properties`; `geometry.query`'s
branches carry `required` alone, so `_openai_parameters` returns it verbatim
and the union reaches the provider. Run H6-013 is what that costs -- a missing
field became "must match exactly one branch" with `constraints={}` -- and
`placement_arguments` already established the repair: named scalars, and a
structured error that names the field where the schema cannot. `_geometry`
already raises exactly that error for none-or-two operations, so the union
enforced nothing the handler does not.

The **result contract against the handler**: the released `geometry.query`
result schema declares `region_result` as a bare object and `points` as a bare
array, which cannot catch field drift. The published schema closes all three
branches, so it has to be exercised against real handler output rather than
asserted -- a too-strict result schema fails on a live call, not here.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG, ToolSurfaceError, resolve_tool_surface,
    validate_schema_instance,
)
from ora.agent.context.projection import causal_provider_value
from ora.agent.native.gateway import openai_tool_definitions
from ora.agent.native.session import _declared_result_properties
from ora.agent.tools.robot import ToolError
from test_runtime_v1 import publish_synthetic

DEFAULTS = ("libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2")

def _maintained_generation() -> set[str]:
    """Every profile composed from the `explicit_motion_v3` components.

    Derived, not listed. The closed manipulation arm hash-reuses these exact
    files, so it inherits whatever the generation publishes without appearing
    in any literal -- and a name list is edited by whoever adds a sibling,
    which is the edit that gets forgotten.
    """
    import json as _json

    from ora.agent.composition.tool_surface import PROFILE_CATALOG as _CATALOG

    found: set[str] = set()
    for profile, path in _CATALOG.items():
        if path.parent.name == "explicit_motion_v3":
            found.add(profile)
            continue
        manifest = _json.loads(path.read_text(encoding="utf-8"))
        refs = [manifest.get("base_definitions") or "", manifest.get("additional_definitions") or ""]
        refs += list((manifest.get("components") or {}).values())
        if any("explicit_motion_v3/" in str(ref) for ref in refs):
            found.add(profile)
    return found

CAMERAS = {
    "libero_placement_direct_overhead_v1": ["agentview", "wrist"],
    "robotwin_toolchain_b2_closed_noservo_v2": [
        "head_camera", "front_camera", "left_camera", "right_camera",
    ],
}


def _definition(profile: str):
    surface = resolve_tool_surface(profile)
    return surface, next(
        item for item in surface.definitions if item["name"] == "geometry_query"
    )


@pytest.mark.parametrize("profile", DEFAULTS)
def test_both_defaults_publish_the_snake_case_name_and_bind_their_own_cameras(profile):
    surface, definition = _definition(profile)
    assert surface.supports("rgbd")
    assert definition["inputSchema"]["properties"]["camera"]["enum"] == CAMERAS[profile]
    # Effect classification needs no code: absent from both lists is read_only.
    assert "geometry_query" not in surface.mutation_tools
    assert "geometry_query" not in surface.lifecycle_tools
    assert "geometry_query" not in surface.observation_tools


def test_every_frozen_surface_keeps_the_spelling_its_runs_were_shown():
    """A frozen surface is evidence of what a past run saw; do not re-spell it.

    Counted over the catalog rather than quoted: of the five dotted profiles
    the guide counted, only the robot_runtime ones are still registered.
    """

    frozen = {"robot_runtime_v1", "robot_runtime_v2"} & set(PROFILE_CATALOG)
    old, new = set(), set()
    for profile in PROFILE_CATALOG:
        names = {item["name"] for item in resolve_tool_surface(profile).definitions}
        if "geometry.query" in names:
            old.add(profile)
        if "geometry_query" in names:
            new.add(profile)
    assert old == frozen
    # The maintained generation, derived: its closed sibling inherits the
    # canonical spelling by hash-reusing the same components.
    assert new == _maintained_generation()
    assert not old & new, "no profile may advertise both spellings of one tool"


@pytest.mark.parametrize("profile", DEFAULTS)
def test_the_published_schema_carries_no_union_to_the_provider(profile):
    """H6-013's shape, at the root, where the projection cannot unwrap it."""

    definition = _definition(profile)[1]
    encoded = json.dumps(definition["inputSchema"])
    for forbidden in ("oneOf", "anyOf", "allOf", "$ref", "$defs"):
        assert forbidden not in encoded, f"geometry_query still delivers {forbidden}"
    delivered = openai_tool_definitions(
        ["geometry_query"], definitions=[definition],
    )[0]["function"]["parameters"]
    assert delivered == definition["inputSchema"]
    for name in ("points", "region", "world_points_m"):
        assert name in delivered["properties"], name
        assert "Exactly one of" in delivered["properties"][name]["description"]
    assert delivered["required"] == ["camera"]


@pytest.mark.parametrize("profile", DEFAULTS)
def test_a_missing_or_doubled_operation_is_an_error_that_names_all_three(profile, tmp_path):
    """What the flattened schema gives up, the handler already said better."""

    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    definition = _definition(profile)[1]
    schema = definition["inputSchema"]
    # The schema admits both; the handler is the one that refuses them. The
    # schema half is checked with a camera this profile publishes, the handler
    # half with the synthetic camera that actually has artifacts behind it.
    published = CAMERAS[profile][0]
    for operations in ({}, {"points": [[2, 1]], "region": [0, 0, 2, 2]}):
        validate_schema_instance(
            {"camera": published, **operations}, schema, root_path="$input")
        with pytest.raises(ToolError) as error:
            tools._geometry({"camera": "head_camera", **operations})
        reported = error.value.as_dict()
        assert reported["code"] == "invalid_arguments"
        assert reported["retryable"] is True
        for name in ("points", "region", "world_points_m"):
            assert name in reported["message"], name
    # And a camera outside the published list never reaches the handler.
    with pytest.raises(ToolSurfaceError, match="enum"):
        validate_schema_instance(
            {"camera": "not_a_camera", "region": [0, 0, 2, 2]}, schema,
            root_path="$input",
        )


@pytest.mark.parametrize("profile", DEFAULTS)
def test_every_result_branch_validates_against_what_the_handler_returns(profile, tmp_path):
    """The closed result schema, exercised rather than asserted.

    `ManipulationToolRuntime.invoke` validates the visible result against
    the surface's schema, so a branch this schema gets wrong fails on a live
    call. The synthetic camera is 4x3 with a constant depth plane, which is
    enough to drive all three operations and both per-entry variants.
    """

    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    surface = _definition(profile)[0]
    result_schema = surface.result_schemas["geometry_query"]

    back_projected = tools._geometry(
        {"camera": "head_camera", "points": [[2, 1], [0, 0]], "window_px": 0})
    validate_schema_instance(back_projected, result_schema)
    assert [entry["pixel_uv"] for entry in back_projected["points"]] == [[2, 1], [0, 0]]

    region = tools._geometry({"camera": "head_camera", "region": [0, 0, 4, 3]})
    validate_schema_instance(region, result_schema)
    assert region["region_result"]["region"] == [0, 0, 4, 3]
    assert 0.0 <= region["region_result"]["valid_fraction"] <= 1.0

    # Forward projection is the operation with measured demand behind it, and
    # the only one no other tool on these surfaces can answer in any spelling.
    forward = tools._geometry({
        "camera": "head_camera",
        "world_points_m": [back_projected["points"][0]["world_xyz_m"],
                           [0.0, 0.0, -50.0]],
    })
    validate_schema_instance(forward, result_schema)
    assert len(forward["calibration_sha256"]) == 64
    seen, behind = forward["world_points"]
    assert seen["in_front"] is True and "pixel_uv" in seen
    assert seen["pixel_uv_rounded"] == [2, 1], seen
    # A point behind the camera returns the short variant: no pixel at all,
    # which is why the branch's required set is the intersection of the two.
    assert behind["in_front"] is False and "pixel_uv" not in behind


@pytest.mark.parametrize("profile", DEFAULTS)
def test_no_returned_field_is_dropped_on_the_way_to_the_provider(profile, tmp_path):
    """Validated by the runtime is not the same as delivered to the model.

    `native/session.py` keeps only the result schema's declared *top-level*
    properties and then runs `causal_provider_value` over them, which removes
    `_CAUSAL_FORBIDDEN_KEYS` at any depth and strips `backend` / `diagnostics`
    unless declared. Both filters are silent. A tool whose numbers are correct,
    validated and then quietly removed looks identical from the handler's side
    to one that works, so the projection is exercised rather than reasoned
    about: `calibration_sha256` and `source_state_sha256` survive because the
    forbidden set matches whole keys and holds bare `sha256`, not a substring.
    """

    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    surface = _definition(profile)[0]
    schema = surface.result_schemas["geometry_query"]
    declared = frozenset(_declared_result_properties(schema))

    for arguments in ({"camera": "head_camera", "points": [[2, 1]], "window_px": 0},
                      {"camera": "head_camera", "region": [0, 0, 4, 3]},
                      {"camera": "head_camera", "world_points_m": [[2.0, 1.0, 1.0]]}):
        result = tools._geometry(arguments)
        kept = {key: result[key] for key in declared if key in result}
        assert kept == result, "a returned field is not a declared top-level property"
        assert causal_provider_value(kept, declared=declared) == result, (
            "the causal projection removed part of the published result"
        )


def test_a_drifted_result_field_is_caught_rather_than_passed_through(tmp_path):
    """Why the released bare `{"type": "object"}` region result was not reused."""

    surface = _definition(DEFAULTS[0])[0]
    schema = surface.result_schemas["geometry_query"]
    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    region = tools._geometry({"camera": "head_camera", "region": [0, 0, 4, 3]})
    drifted = {**region, "region_result": {**region["region_result"], "mean": 1.0}}
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(drifted, schema)
