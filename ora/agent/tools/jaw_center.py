"""Where the jaw centre sits relative to a surface's controlled point.

The RoboDojo features that hold or measure from the jaw centre read
``reference_parameters.contact_center_distance_m`` from the public geometry
descriptor (``contact_point_projection.jaw_contact_center_offsets``): on the
ARX X5 the controlled point is the link6 wrist flange and the reviewed pad
contact centre lies 0.1501 m along its approach axis. The ALOHA and Panda
descriptors publish no such field, and need none: their controlled point is
already the grip centre --

* ALOHA ``aloha_grip_center_tcp``: the descriptor's ``left_tcp``/``right_tcp``
  (``description: tool_center``, 0.12 m from the flange), the
  ``parallel_jaw_grip_center_tcp`` the control profile's arrival check
  measures; the finger collision bounds straddle it along the approach axis
  (-0.050 to +0.036 m) and ``centerline_gap_at_tcp_m`` is taken at its depth.
* Panda ``panda_grip_site``: RoboSuite's ``grip_site``
  (``description: grip_center``), 0.097 m from the hand; the pad bounds
  straddle it along the approach axis (-0.0116 to +0.0044 m).

Adding the descriptor formula's ``offset_m`` there would count the flange
offset twice. So on a grip-centre frame the jaw centre *is* the controlled
point (zero offset), checked against the descriptor's own controlled-point
description rather than assumed from the convention name alone; every other
frame keeps the descriptor contact centre. Nothing here edits a descriptor.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
#: Grip-centre conventions and the controlled-point description their
#: descriptor must state for the zero offset to apply.
GRIP_CENTRE_CONVENTIONS = {
    "aloha_grip_center_tcp": "tool_center",
    "panda_grip_site": "grip_center",
}
#: Frames whose jaw centre comes from the descriptor's reviewed contact centre.
CONTACT_CENTRE_CONVENTIONS = ("arx_x5_link6",)
SUPPORTED_CONVENTIONS = (*GRIP_CENTRE_CONVENTIONS, *CONTACT_CENTRE_CONVENTIONS)


class JawCenterError(ValueError):
    """The surface binds no descriptor this module can read a jaw centre from."""


def load_descriptor(view_config: Mapping[str, Any], *, project_root: Path = _ROOT) -> dict[str, Any]:
    """The hash-bound public geometry descriptor the surface selects."""
    from ora.agent.composition.embodiment import load_embodiment_descriptor

    binding = view_config.get("public_geometry_descriptor")
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise JawCenterError("the jaw centre requires a public geometry descriptor")
    path = (Path(project_root) / binding["path"]).resolve()
    if (not path.is_relative_to(Path(project_root).resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")):
        raise JawCenterError("public geometry descriptor is missing or drifted")
    return load_embodiment_descriptor(path)


def grip_centre_offsets(descriptor: Mapping[str, Any], convention: str) -> dict[str, list[float]]:
    """Zero per arm, after checking every controlled point is the grip centre the convention names."""
    expected = GRIP_CENTRE_CONVENTIONS[convention]
    offsets: dict[str, list[float]] = {}
    for point in descriptor["control"]["controlled_points"]:
        if point.get("description") != expected:
            raise JawCenterError(
                f"{convention} controlled point {point.get('id')!r} is not a {expected}")
        manipulator = str(point["manipulator_id"])
        if not manipulator.endswith("_arm"):
            raise JawCenterError("descriptor manipulator does not name a public arm")
        offsets[manipulator[: -len("_arm")]] = [0.0, 0.0, 0.0]
    return offsets


def jaw_center_offsets(view_config: Mapping[str, Any], *, project_root: Path = _ROOT) -> dict[str, list[float]]:
    """Per-arm jaw-centre offset in the controlled point's local frame."""
    convention = str(view_config.get("controlled_frame_convention"))
    if convention in GRIP_CENTRE_CONVENTIONS:
        return grip_centre_offsets(load_descriptor(view_config, project_root=project_root), convention)
    if convention in CONTACT_CENTRE_CONVENTIONS:
        from ora.agent.context.contact_point_projection import jaw_contact_center_offsets

        return jaw_contact_center_offsets(view_config, project_root=project_root)
    raise JawCenterError(f"no jaw-centre rule for controlled frame {convention!r}")


def approach_and_offset(surface: Any, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """The controlled point's unit approach axis and jaw-centre offset, in its own frame."""
    from .alignment import frame_local_axes

    view = surface.view_config or {}
    approach = np.asarray(frame_local_axes(str(view.get("controlled_frame_convention")))["approach"], float)
    offset = np.asarray(jaw_center_offsets(view)[arm], float)
    return approach / np.linalg.norm(approach), offset


__all__ = [
    "CONTACT_CENTRE_CONVENTIONS", "GRIP_CENTRE_CONVENTIONS", "JawCenterError", "SUPPORTED_CONVENTIONS",
    "approach_and_offset", "grip_centre_offsets", "jaw_center_offsets", "load_descriptor",
]
