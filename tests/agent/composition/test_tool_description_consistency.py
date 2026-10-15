"""Tool help on the current surfaces says what the handlers do (EXP11B, 2026-10-03).

The EXP11B g003 contract audit read the delivered tool descriptions against
their handlers and found sentences that contradict them, plus one rule copied
into four tools. Each fix is a per-surface description or schema layer
generated from that surface's own composed text with exact sentence edits; the
shared files the old sentences live in are bound by frozen surfaces too, so
they keep their bytes (resolution would refuse a changed hash).

- `grasp_plan` on a surface declaring `grasp_plan_ik_reachability_annotation_v1`
  checks its whole pool with read-only IK (`reach_check.select_grasp_candidates`),
  so it cannot also say reachability is delegated and no probe runs.
- `geometry_query` reports missing depth inside a successful result
  (`RobotToolRuntime._geometry`); it never fails the call for it.
- `mark_point` fails one batch entry with `inconsistent_mark`
  (`mark_point_batch.execute_batch`); it does not refuse the call.
- A default is stated as a value: with `optional_argument_null_as_omitted_v1`
  null also means omitted, so "omit for three" was a two-step instruction, and
  `vision_ground` told the policy to "raise it from one" while its default is 3.
- Under `policy_failure_exit_withheld_v1` no help may offer a failure exit.
- The compute routing rule has one owner, `compute`'s own Use when; the motion
  tools point to it.

The current set is derived from `tests/data/default-expectations.json` and the EXP11A /
EXP11B experiment binding rather than listed, so a new default is checked
without editing this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy
import pytest

from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.tools import alignment
from tests.orchestrator.test_runtime_v1 import publish_synthetic

ROOT = Path(__file__).resolve().parents[3]
IK_ANNOTATION = "grasp_plan_ik_reachability_annotation_v1"
WITHHELD_EXIT = "policy_failure_exit_withheld_v1"


def current_surfaces() -> tuple[str, ...]:
    """The defaults and the defaults they replaced (still bound by live experiments).

    A replaced default that is no longer registered cannot be bound and is not listed.
    """

    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
    found = {instance["tool_surface"] for instance in activation["instances"].values()}
    parity = json.loads((ROOT / "ci/backend-parity.json").read_text(encoding="utf-8"))
    found |= {record["profile_id"] for record in parity["previous_defaults"].values()
              if record["profile_id"] in PROFILE_CATALOG}
    return tuple(sorted(found))


CURRENT = current_surfaces()


def _definitions(profile: str) -> dict[str, dict]:
    return {item["name"]: item for item in resolve_tool_surface(profile).definitions}


def test_the_current_set_is_the_three_defaults_and_the_ones_they_replaced():
    # Integration 0059 unit A0: the v2 generations are the defaults (RoboDojo's
    # included); the stale B2 v4 Dojo entry left with developing_surfaces.robodojo.
    # Integration 0084: LIBERO's default is direct_v2's overhead successor. The
    # replaced LIBERO and RoboDojo defaults are no longer registered.
    assert CURRENT == (
        "libero_placement_direct_overhead_v1",
        "robodojo_placement_closed_skills_v2",
        "robotwin_toolchain_b2_closed_noservo_v1",
        "robotwin_toolchain_b2_closed_noservo_v2",
    )


@pytest.mark.parametrize("profile", CURRENT)
def test_grasp_plan_says_reachability_is_delegated_only_where_it_runs_no_ik(profile):
    surface = resolve_tool_surface(profile)
    grasp_plan = _definitions(profile)["grasp_plan"]
    help_text = grasp_plan["description"]
    cap = grasp_plan["inputSchema"]["properties"]["max_candidates"]["description"]
    if surface.supports(IK_ANNOTATION):
        assert "Reachability is delegated" not in help_text
        assert "does not run a motion probe" not in help_text
        assert help_text.count("IK selection") == 2  # the pointer and the paragraph it names
        assert "IK-ordered pool" in cap
    else:
        assert "Reachability is delegated to move_pose" in help_text
        assert "IK selection" not in help_text and "IK-ordered" not in cap


def _zero_depth(monkeypatch):
    real = numpy.load

    def load(path, *args, **kwargs):
        value = real(path, *args, **kwargs)
        return numpy.zeros_like(value) if "depth" in str(path) else value

    monkeypatch.setattr(numpy, "load", load)


@pytest.mark.parametrize("profile", CURRENT)
def test_geometry_query_help_matches_missing_depth_in_a_successful_result(profile, tmp_path, monkeypatch):
    help_text = _definitions(profile)["geometry_query"]["description"]
    assert "fails explicitly" not in help_text
    assert "Missing or invalid depth is reported, never inferred, and does not fail the call" in help_text
    for named in ("valid false", "valid_count 0", "scene_depth_m"):
        assert named in help_text, named

    # What the sentence describes, from the handler every surface dispatches to.
    tools = publish_synthetic(tmp_path)[4]
    visible = tools._geometry({"camera": "head_camera", "points": [[2, 1]], "window_px": 0})
    world = visible["points"][0]["world_xyz_m"]
    _zero_depth(monkeypatch)
    points = tools._geometry({"camera": "head_camera", "points": [[2, 1]], "window_px": 0})
    assert points["points"] == [{"pixel_uv": [2, 1], "valid": False}]
    region = tools._geometry({"camera": "head_camera", "region": [0, 0, 4, 3]})["region_result"]
    assert region["valid_count"] == 0 and "depth_m" not in region
    seen = tools._geometry({"camera": "head_camera", "world_points_m": [world]})["world_points"][0]
    assert seen["in_frame"] is True
    assert seen["scene_depth_m"] is None and seen["depth_residual_m"] is None


@pytest.mark.parametrize("profile", CURRENT)
def test_a_disagreeing_mark_is_a_failed_entry_in_every_help_that_mentions_it(profile):
    surface = resolve_tool_surface(profile)
    definitions = _definitions(profile)
    assert surface.supports("mark_point_batch_v1")
    mark = definitions["mark_point"]["description"]
    assert "refused as inconsistent_mark" not in mark
    assert "a disagreement beyond 25 mm fails only that entry, with error_code inconsistent_mark" in mark
    geometry = definitions["geometry_query"]["description"]
    assert "refuses a disagreement" not in geometry
    assert "(a disagreement beyond 25 mm fails that entry)" in geometry
    entry = surface.result_schemas["mark_point"]["properties"]["marks"]["items"]["properties"]
    assert "inconsistent_mark" in entry["error_code"]["enum"]


#: Optional arguments whose default the help must state as a value.
DEFAULTED = (
    ("vision_segment", "max_candidates"), ("vision_ground", "max_candidates"),
    ("grasp_plan", "max_candidates"), ("align_plan", "max_candidates"),
    ("align_plan", "standoff_m"), ("geometry_query", "window_px"),
)
OMISSION = re.compile(r"\b[Oo]mit(ted)? (for|means)\b")


@pytest.mark.parametrize("profile", CURRENT)
def test_every_default_is_stated_as_its_value(profile):
    assert resolve_tool_surface(profile).supports("optional_argument_null_as_omitted_v1")
    definitions = _definitions(profile)
    checked = 0
    for tool, name in DEFAULTED:
        if tool not in definitions:
            continue
        node = definitions[tool]["inputSchema"]["properties"][name]
        text = node["description"]
        assert not OMISSION.search(text), f"{tool}.{name}: {text}"
        default = node["default"]
        stated = f"{default:.2f} m" if isinstance(default, float) else str(default)
        assert re.search(rf"\b[Dd]efault {re.escape(stated)}", text), f"{tool}.{name}: {text}"
        assert text.endswith("Null means omitted.")
        checked += 1
    assert checked == (6 if "align_plan" in definitions else 4)
    ground = definitions["vision_ground"]["inputSchema"]["properties"]["max_candidates"]["description"]
    assert "raise it from one" not in ground
    if "align_plan" in definitions:
        properties = definitions["align_plan"]["inputSchema"]["properties"]
        assert properties["max_candidates"]["default"] == alignment.DEFAULT_CANDIDATES
        assert properties["standoff_m"]["default"] == alignment.DEFAULT_STANDOFF_M


@pytest.mark.parametrize("profile", CURRENT)
def test_no_help_offers_an_exit_the_surface_withholds(profile):
    if not resolve_tool_surface(profile).supports(WITHHELD_EXIT):
        pytest.skip("this surface keeps the failure exit")
    for name, definition in _definitions(profile).items():
        assert "finish with failure" not in definition["description"], name
    recover = _definitions(profile)["recover_pose"]["description"]
    assert "finish_episode names the accepted outcomes" in recover


MOVED_RULE = "before using a newly calculated target component or angle, obtain it from compute"
POINTER = "Numerical inputs follow compute's Use when rule; send every value in this tool's declared units and frame."


@pytest.mark.parametrize("profile", CURRENT)
def test_the_compute_rule_has_one_owner(profile):
    definitions = _definitions(profile)
    assert all(MOVED_RULE not in item["description"] for item in definitions.values())
    assert "call compute" in definitions["compute"]["description"].split("Use when:", 1)[1]
    pointers = sorted(name for name, item in definitions.items() if POINTER in item["description"])
    assert pointers == ["move_pose", "move_to", "rotate_pitch", "rotate_wrist"]


@pytest.mark.parametrize("profile", CURRENT)
def test_the_immediate_return_comes_before_the_later_receipt(profile):
    for name, definition in _definitions(profile).items():
        assert "Returns: the later receipt" not in definition["description"], name
