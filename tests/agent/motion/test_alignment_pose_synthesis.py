"""Approach synthesis must obey the embodiment's declared gripper axes.

The first version wrote the approach direction into column 0 of the rotation,
which silently assumes local +X is the approach axis. That is true of
aloha-agilex and false of franka-panda, whose descriptor declares approach +Z
and closing +X. On the wrong embodiment the synthesised pose points the
jaw-separation line at the object and rolls every candidate about the closing
axis -- a well-formed answer to a question nobody asked, and one no error
message would ever report.
"""
from __future__ import annotations

import numpy as np
import pytest

from ora.agent.tools.alignment import (
    _APPROACH_DIRECTIONS, frame_local_axes,
    synthesise_approach_poses,
)
from ora.contracts.pose import quaternion_matrix

CONVENTIONS = {"aloha-agilex": "aloha_grip_center_tcp", "franka-panda": "panda_grip_site"}


def declared_axes(embodiment: str) -> dict[str, list[float]]:
    """Through the reader that owns axis values, never a second copy of them."""

    return frame_local_axes(CONVENTIONS[embodiment])


def test_the_two_maintained_embodiments_really_do_declare_different_axes():
    """The premise of this whole module. If it ever stops holding, say so loudly."""

    aloha, franka = declared_axes("aloha-agilex"), declared_axes("franka-panda")
    assert aloha != franka
    assert franka["approach"] != aloha["approach"]


@pytest.mark.parametrize("embodiment", ["aloha-agilex", "franka-panda"])
@pytest.mark.parametrize("side", sorted(_APPROACH_DIRECTIONS))
def test_the_declared_approach_axis_points_at_the_target(embodiment, side):
    """Whichever local axis the descriptor calls the approach, it faces the object."""

    axes = declared_axes(embodiment)
    centre = [0.4, -0.1, 0.9]
    poses = synthesise_approach_poses(
        centre, approach_side=side, standoff_m=0.12, candidate_count=3,
        local_axes=axes,
    )
    expected = np.asarray(_APPROACH_DIRECTIONS[side], dtype=float)
    for pose in poses:
        rotation = np.asarray(quaternion_matrix(pose[3:]), dtype=float)[:3, :3]
        # R maps the declared local approach axis to the world direction that
        # travels from the standoff position toward the target.
        assert np.allclose(rotation @ np.asarray(axes["approach"]), expected, atol=1e-9)
        # And the position stands off along that same direction, so the tool
        # travels toward the centroid rather than away from it.
        offset = np.asarray(centre) - np.asarray(pose[:3])
        assert np.allclose(offset / np.linalg.norm(offset), expected, atol=1e-9)
        assert np.isclose(np.linalg.norm(offset), 0.12)


@pytest.mark.parametrize("embodiment", ["aloha-agilex", "franka-panda"])
def test_the_closing_axis_stays_perpendicular_and_the_roll_actually_varies(embodiment):
    """Candidates differ only in how the fingers straddle the object."""

    axes = declared_axes(embodiment)
    poses = synthesise_approach_poses(
        [0.4, -0.1, 0.9], approach_side="top", standoff_m=0.1, candidate_count=4,
        local_axes=axes,
    )
    expected = np.asarray(_APPROACH_DIRECTIONS["top"], dtype=float)
    jaw_lines = []
    for pose in poses:
        rotation = np.asarray(quaternion_matrix(pose[3:]), dtype=float)[:3, :3]
        jaw = rotation @ np.asarray(axes["closing"])
        # A jaw line with a component along the approach direction would mean
        # the fingers close partly into the object rather than across it.
        assert abs(float(jaw @ expected)) < 1e-9
        assert np.isclose(np.linalg.norm(jaw), 1.0)
        jaw_lines.append(jaw)
    # Four distinct rolls, not four copies of one orientation.
    for index in range(1, len(jaw_lines)):
        assert not np.allclose(jaw_lines[index], jaw_lines[0], atol=1e-6)


@pytest.mark.parametrize("profile,embodiment", [
    ("robotwin_toolchain_b2_closed_noservo_v2", "aloha-agilex"),
    ("libero_placement_direct_overhead_v1", "franka-panda"),
])
def test_the_convention_reaching_synthesis_is_the_one_the_surface_declares(
    profile, embodiment,
):
    """There is no default, so this cannot pass by matching one embodiment.

    The module previously defaulted to the ALOHA frame, which was correct only
    while align_plan was advertised on RoboTwin alone -- and was therefore the
    thing standing between the tool and every other embodiment. A wrong
    convention yields well-formed candidates rolled about the closing axis,
    which no error message reports, so the binding is asserted rather than
    assumed.
    """
    import inspect

    from ora.agent.composition import resolve_tool_surface
    from ora.agent.tools import alignment

    surface = resolve_tool_surface(profile)
    declared = (surface.view_config or {})["controlled_frame_convention"]
    assert declared == CONVENTIONS[embodiment]
    assert alignment.frame_local_axes(declared) == declared_axes(embodiment)
    # No caller may omit it: an optional convention is the defect itself.
    for name in ("synthesise_approach_poses", "build_request", "plan"):
        signature = inspect.signature(getattr(alignment, name))
        parameter = signature.parameters.get("local_axes") or signature.parameters["frame_convention"]
        assert parameter.default is inspect.Parameter.empty, name
    assert not hasattr(alignment, "DEFAULT_FRAME_CONVENTION")


def test_the_two_maintained_embodiments_do_not_share_an_approach_axis():
    """The premise of the parameter: a shared default cannot serve both."""

    assert declared_axes("aloha-agilex") != declared_axes("franka-panda")
    assert declared_axes("aloha-agilex")["approach"] == [1.0, 0.0, 0.0]
    assert declared_axes("franka-panda")["approach"] == [0.0, 0.0, 1.0]
