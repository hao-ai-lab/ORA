"""V2 bounded robot-only geometry and nine-node deterministic card projection.

Source identity and verification records remain in the descriptor/manifest;
the card contains only typed public facts. No simulator imports belong here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .embodiment import EmbodimentDescriptorError, _exact_keys, _object


# The current source-bound descriptor of each controlled-frame convention, and
# every descriptor a registered surface may bind for it (the current one first).
# Source-only successors (controller feedback; the 2026-10-01 RoboTwin session
# split, which moved the public adapter into its own modules; the LIBERO session
# split, which moved the Panda camera, pose, calibration and motion code into
# its own modules) preserve all robot geometry, axes and the compiled card.
# ARX v5 changes only the card projection (full_finger_reference_v3:
# the two-finger encoder sentence). The LIBERO direct descriptor is the one the
# default LIBERO surface binds.
PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS = {
    "arx_x5_link6": ("arx-x5/descriptor-public-geometry-v5.json",),
    "panda_grip_site": ("franka-panda/descriptor-public-geometry-v8-controller-feedback-v1-session-split-v1.json",
                        "franka-panda/descriptor-public-geometry-direct-v1-controller-feedback-v1-session-split-v1.json"),
    "aloha_grip_center_tcp": ("aloha-agilex/descriptor-public-geometry-v5-session-split-v1.json",),
}
PUBLIC_GEOMETRY_DESCRIPTORS = {frame: paths[0] for frame, paths in PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS.items()}


def bind_public_geometry_descriptor(reference: Any, *, project_root: Path) -> dict[str, str]:
    """Resolve an asset's path-only descriptor reference to its ``{path, sha256}`` record.

    ToolSurface manifests and context profiles name the descriptor by its
    source-relative path. The record carries the sha256 of the bytes read now;
    the RunPlan and the isolated worker verify that record with
    :func:`load_public_geometry_binding`.
    """
    from .locked_assets import locked_asset

    if not isinstance(reference, str):
        raise EmbodimentDescriptorError(
            "geometry descriptor reference must be a source-relative path; "
            "inline sha256 pins are not accepted (policy.lock carries asset integrity)")
    asset = locked_asset(project_root, reference, label="geometry descriptor", error=EmbodimentDescriptorError)
    return asset.record()


def load_public_geometry_binding(value: Any, *, project_root: Path) -> dict[str, Any]:
    """Bind the exact descriptor and public robot boxes for the isolated worker.

    ``value`` is the resolved ``{path, sha256}`` record a RunPlan carries, not
    an asset reference: the worker re-reads the bytes the plan resolved.
    """
    import hashlib
    from .embodiment import load_embodiment_descriptor
    from ora.contracts.robot_geometry import GEOMETRY_BINDING_SCHEMA

    value = _object(value, "public_geometry_descriptor")
    _exact_keys(value, required={"path", "sha256"}, optional=set(), path="public_geometry_descriptor")
    if not isinstance(value["path"], str):
        raise EmbodimentDescriptorError("geometry descriptor requires a source-relative path")
    path = (project_root / value["path"]).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        raise EmbodimentDescriptorError("geometry descriptor escapes or is missing from the source checkout")
    if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise EmbodimentDescriptorError("geometry descriptor source hash drift")
    descriptor = load_embodiment_descriptor(path)
    geometry = {gripper["end_effector_id"]: gripper["contact_geometry"] for gripper in descriptor["geometry"]["grippers"]}
    return {"schema": GEOMETRY_BINDING_SCHEMA, "descriptor_sha256": value["sha256"], "geometry": geometry}


def public_local_axes(convention: str) -> dict[str, list[float]]:
    """Resolve the same versioned descriptor used by the compiled manual.

    Do not create a second hand-maintained registry of axis values. The tool
    manifest names these files by path, and this function validates their type.
    """
    from .embodiment import load_embodiment_descriptor
    path = PUBLIC_GEOMETRY_DESCRIPTORS.get(convention)
    if path is None:
        raise EmbodimentDescriptorError("unknown public controlled-frame convention")
    root = Path(__file__).resolve().parents[2] / "policy" / "embodiments"
    descriptor = load_embodiment_descriptor(root / path)
    points = descriptor["control"]["controlled_points"]
    axes = validate_local_axes(points[0]["local_axes"])
    if any(validate_local_axes(point["local_axes"]) != axes for point in points[1:]):
        raise EmbodimentDescriptorError("this convention requires the same axis roles for all arms")
    return axes


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in value
    ):
        raise EmbodimentDescriptorError(f"{label} requires {size} finite numbers")
    return [float(x) for x in value]


def validate_local_axes(value: Any) -> dict[str, list[float]]:
    axes = _object(value, "local_axes")
    _exact_keys(axes, required={"approach", "closing"}, optional=set(), path="local_axes")
    result = {key: _vector(axes[key], 3, key) for key in axes}
    if any(abs(sum(x*x for x in axis)-1) > 1e-9 for axis in result.values()):
        raise EmbodimentDescriptorError("local axes must be unit vectors")
    if abs(sum(x*y for x, y in zip(result["approach"], result["closing"]))) > 1e-9:
        raise EmbodimentDescriptorError("approach and closing axes must be orthogonal")
    return result


def validate_contact_geometry(value: Any, *, controlled_points: set[str]) -> None:
    geometry = _object(value, "contact_geometry")
    _exact_keys(geometry, required={"controlled_point", "unit", "reference_finger_qpos_m", "fingers", "claim"},
                optional={"centerline_gap_at_tcp_m"}, path="contact_geometry")
    if geometry["controlled_point"] not in controlled_points or geometry["unit"] != "metre":
        raise EmbodimentDescriptorError("contact geometry must bind a declared controlled-point frame in metres")
    if geometry["claim"] != "robot_geometry_not_contact_or_grasp_guarantee":
        raise EmbodimentDescriptorError("contact geometry has an invalid evidence claim")
    _vector(geometry["reference_finger_qpos_m"], 2, "reference_finger_qpos_m")
    fingers = geometry["fingers"]
    if not isinstance(fingers, list) or len(fingers) != 2:
        raise EmbodimentDescriptorError("contact geometry requires exactly two public fingers")
    for finger in fingers:
        finger = _object(finger, "finger")
        _exact_keys(finger, required={"id", "collision_bounds_m", "opening_direction_local"},
                    optional={"pad_bounds_m"}, path="finger")
        if finger["id"] not in {"finger_1", "finger_2"}:
            raise EmbodimentDescriptorError("invalid public finger identity")
        direction = _vector(finger["opening_direction_local"], 3, "opening_direction_local")
        if abs(sum(x*x for x in direction)-1) > 1e-9:
            raise EmbodimentDescriptorError("finger opening direction must be unit length")
        for name in ("collision_bounds_m", "pad_bounds_m"):
            if name not in finger:
                continue
            bounds = _object(finger[name], name)
            _exact_keys(bounds, required={"minimum", "maximum"}, optional=set(), path=name)
            lower, upper = (_vector(bounds[key], 3, name) for key in ("minimum", "maximum"))
            if any(a >= b for a, b in zip(lower, upper)):
                raise EmbodimentDescriptorError("finger bounds must have positive extent")
    if {finger["id"] for finger in fingers} != {"finger_1", "finger_2"}:
        raise EmbodimentDescriptorError("duplicate public finger identity")
    if "centerline_gap_at_tcp_m" in geometry:
        gap = _vector([geometry["centerline_gap_at_tcp_m"]], 1, "centerline_gap_at_tcp_m")[0]
        if gap < 0:
            raise EmbodimentDescriptorError("centerline gap must be nonnegative")


def _group_identical(items, render, *, grouped: bool):
    """Group items whose rendered sentence differs only by their own name.

    ``grouped`` is off unless the descriptor opts in. Compiled cards are
    byte-compared against this renderer, so changing what an existing
    descriptor renders would invalidate every frozen card that pins its bytes.

    ``render`` takes the item and a plural flag, because a grouped subject needs
    its own agreement: one name takes "is the controlled reference", two take
    "are the controlled references". Grouping is decided on the singular text,
    which is what actually differs only by the name.
    """

    if not grouped:
        return [f"`{item['id']}` " + render(item, False) for item in items]
    collected: list[tuple[list[str], str, Any]] = []
    for item in items:
        body = render(item, False)
        for names, existing, _ in collected:
            if existing == body:
                names.append(item["id"])
                break
        else:
            collected.append(([item["id"]], body, item))
    return [" and ".join(f"`{name}`" for name in names) + " "
            + (render(item, len(names) > 1) if len(names) > 1 else body)
            for names, body, item in collected]


_AXIS_NAMES = {(1, 0, 0): "X", (0, 1, 0): "Y", (0, 0, 1): "Z"}
#: ``full_finger_reference_v3`` is v2 plus the two-finger encoder sentence
#: (ARX descriptor v5); both name one controlled point.
_ONE_NAME_PROJECTIONS = frozenset({"full_finger_reference_v2", "full_finger_reference_v3"})


def _third_axis_name(axes: Mapping[str, Any]) -> str:
    """The local axis perpendicular to approach and jaw closing, named as ``X``/``Y``/``Z``.

    ``full_finger_reference_v2`` names the axis a nominal pad interval runs
    along instead of calling it "lateral", which a reader takes to be the
    jaw-closing line. Only an axis-aligned frame has such a name.
    """
    a, c = axes["approach"], axes["closing"]
    third = (a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0])
    key = tuple(int(round(abs(value))) for value in third)
    if key not in _AXIS_NAMES or abs(sum(abs(value) for value in third) - 1) > 1e-9:
        raise EmbodimentDescriptorError("full_finger_reference_v2 requires axis-aligned local axes")
    return _AXIS_NAMES[key]


def _render_gripper(gripper: Mapping[str, Any], lines: list[str], *,
                    geometry: Mapping[str, Any], control: Mapping[str, Any],
                    name_frame: bool = False) -> None:
    """Render one gripper: the head clause first, then its body lines."""

    contact = gripper["contact_geometry"]
    maximum = gripper["maximum_aperture_m"]
    lines.append(f"nominal maximum jaw separation {maximum:g} m." if maximum is not None
                 else "no single aperture is asserted across the tapered fingers.")
    # Compact axis intervals are the actionable static detail; the complete
    # typed boxes remain in the descriptor for deterministic geometry tools.
    point = next(p for p in control["controlled_points"] if p["id"] == contact["controlled_point"])
    axis = point["local_axes"]["approach"]

    def interval(bounds: Mapping[str, Any]) -> list[float]:
        terms = [(axis[i]*bounds["minimum"][i], axis[i]*bounds["maximum"][i]) for i in range(3)]
        return [round(sum(min(t) for t in terms)*1000, 4), round(sum(max(t) for t in terms)*1000, 4)]

    finger = contact["fingers"][0]
    lines.append(f"At reference finger qpos {contact['reference_finger_qpos_m']} m, finger collision extent along local approach from the controlled point is {interval(finger['collision_bounds_m'])} mm.")
    if "pad_bounds_m" in finger:
        lines.append(f"The separate pad occupies {interval(finger['pad_bounds_m'])} mm on that axis; pad and whole-finger extrema differ.")
    if "centerline_gap_at_tcp_m" in contact:
        lines.append(f"At that reference opening, the mesh centerline gap at TCP depth is {contact['centerline_gap_at_tcp_m']*1000:.4f} mm; the gap varies with depth and is not a general minimum grasp width.")
    if geometry.get("card_projection") in {"full_finger_reference_v1", *_ONE_NAME_PROJECTIONS}:
        for item in contact["fingers"]:
            bounds = item["collision_bounds_m"]
            xyz = {name: [round(bounds[key][i]*1000, 4) for key in ("minimum", "maximum")]
                   for i, name in enumerate(("X", "Y", "Z"))}
            lines.append(f"`{item['id']}` collision bounds in that controlled-point frame are {json.dumps(xyz, separators=(',', ':'))} mm; opening direction is {item['opening_direction_local']}. These are bounds at the stated reference opening, not current contact surfaces.")
    parameters = gripper.get("reference_parameters")
    if parameters is not None:
        pad_x, pad_z = (parameters[key] for key in ("pad_approach_interval_m", "pad_lateral_interval_m"))
        # The collision bounds rendered just above are in the controlled-point
        # frame while this block is a length from the flange, so an unlabelled
        # interval reads as a contradiction of the line before it. Only cards
        # that opt in say so: compiled cards are byte-compared against this
        # renderer, and rewording an existing one invalidates its frozen bytes.
        if geometry.get("card_projection") in _ONE_NAME_PROJECTIONS:
            lines.append(f"Nominal inner pad region in the controlled-point frame: {pad_x[0]*1000:g}\u2013{pad_x[1]*1000:g} mm along the approach axis and {pad_z[0]*1000:g}\u2013{pad_z[1]*1000:g} mm along local {_third_axis_name(point['local_axes'])}. The jaw contact centre is +{parameters['contact_center_distance_m']:.4f} m along the approach axis from `{point['id']}`. Both are nominal source geometry, not a calibrated physical contact point.")
        elif name_frame:
            lines.append(f"Nominal inner pad region, measured from the manipulator flange rather than the controlled point: approach {pad_x[0]*1000:g}-{pad_x[1]*1000:g} mm, lateral {pad_z[0]*1000:g}-{pad_z[1]*1000:g} mm; reference contact center {parameters['contact_center_distance_m']*1000:g} mm along approach. This source geometry is not a calibrated physical contact point.")
        else:
            lines.append(f"Nominal inner pad region: local approach {pad_x[0]*1000:g}\u2013{pad_x[1]*1000:g} mm, local lateral {pad_z[0]*1000:g}\u2013{pad_z[1]*1000:g} mm; reference contact center {parameters['contact_center_distance_m']*1000:g} mm along approach. This source geometry is not a calibrated physical contact point.")
        if geometry.get("card_projection") == "full_finger_reference_v3":
            # The backend publishes both fingers' mean, and the
            # pair can shift to one side under load.
            lines.append(f"Nominal full-open pad separation is {parameters['nominal_maximum_aperture_m']*1000:g} mm. Physical pad aperture remains uncalibrated. The public encoder reading is the mean of both finger joints on the normalized command scale (empty-close reference about {parameters['empty_close_encoder_reference']:.3f}, full-open 1); it is not measured pad separation, and a pair shifted to one side is no evidence of holding or release. A close directive is not a request for a measured zero, and reaching the encoder floor proves neither contact nor retention.")
            return
        lines.append(f"Nominal full-open pad separation is {parameters['nominal_maximum_aperture_m']*1000:g} mm. Physical pad aperture remains uncalibrated. The public encoder reading is normalized joint position, with empty-close reference about {parameters['empty_close_encoder_reference']:.3f} and full-open reference 1; it is not measured pad separation. A close directive is not a request for a measured zero, and reaching the encoder floor proves neither contact nor retention.")


def render_geometry_card(descriptor: Mapping[str, Any], *, token_limit: int) -> str:
    """Use the same nine-node template as manually authored embodiment cards."""
    control = descriptor["control"]
    geometry = descriptor["geometry"]
    topology = descriptor["topology"]
    arms = ", ".join(f"{arm['side']} {arm['joint_dof']}-DoF arm" for arm in topology["manipulators"])
    # Display names and source names are deliberately not rendered.
    # Off unless the descriptor asks for it. Compiled cards are byte-compared
    # against this renderer, so grouping an existing descriptor's identical
    # sides would invalidate every frozen card that pins its bytes.
    group_symmetric = geometry.get("card_projection") == "grouped_symmetric_sides_v1"
    identity = {"franka_panda": "Franka Panda", "aloha_agilex": "ALOHA-AgileX", "arx_x5": "ARX X5"}.get(descriptor["embodiment_id"], "Robot")
    lines = [f"# {identity} compiled embodiment v2", "", "## Identity and topology", "",
             f"{descriptor['identity']['base_mobility'].capitalize()}-base {identity}: {arms}; parallel-jaw grippers.", "", "## Controlled points", ""]
    # A symmetric dual arm states the same controlled-point sentence twice with
    # only the name changed, so identical ones are named together.
    one_name = geometry.get("card_projection") in _ONE_NAME_PROJECTIONS
    if one_name and any(value != 0 for point in control["controlled_points"] for value in point.get("offset_m", [1])):
        raise EmbodimentDescriptorError("full_finger_reference_v2 requires zero controlled-point offsets")
    lines += [f"`{point['id']}` is the {point['manipulator_id'].replace('_', ' ')}'s controlled point, its "
              f"{point['description'].replace('_', ' ')}, not a fingertip; commands and measured poses already "
              "refer to it, so add no offset." for point in control["controlled_points"]] if one_name else []
    lines += [] if one_name else _group_identical(control["controlled_points"], grouped=group_symmetric, render=lambda point, many: (
        ("are the controlled references, not fingertips. Their local offset is "
         if many else "is the controlled reference, not a fingertip. Its local offset is ")
        + f"{json.dumps(point.get('offset_m'), separators=(',', ':'))} m from the declared "
        + f"{point.get('offset_from', 'root').replace('_', ' ')}; commands already target "
        + ("them" if many else "it") + ". Do not add this offset again."))
    lines += ["", "## Frames and units", "", "Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.", "", "## End-effector axes", ""]
    lines += _group_identical(control["controlled_points"], grouped=group_symmetric, render=lambda point, many: (
        f"local approach {point['local_axes']['approach']}; jaw-closing line "
        f"{point['local_axes']['closing']}. "
        + (f"Local {_third_axis_name(point['local_axes'])} is perpendicular to both. " if one_name else "")
        + "Current world directions are supplied with measured state."))
    lines += ["", "## Motion capabilities", "", "The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.", "", "## Gripper semantics", ""]
    # Grippers whose rendered description is identical are described once. The
    # left and right blocks of a symmetric dual arm differ only by the name of
    # their controlled point, so comparing the rendered text rather than the raw
    # objects is what actually finds them. Repeating every line per side buys
    # the card nothing and spends the budget the aperture facts now need.
    grouped: list[tuple[list[str], list[str], str]] = []
    for gripper in (geometry["grippers"] if group_symmetric else []):
        rendered: list[str] = []
        _render_gripper(gripper, rendered, geometry=geometry, control=control,
                        name_frame=group_symmetric)
        head, body = rendered[0], rendered[1:]
        for names, existing, existing_head in grouped:
            if existing == body and existing_head == head:
                names.append(gripper["end_effector_id"])
                break
        else:
            grouped.append(([gripper["end_effector_id"]], body, head))
    if not group_symmetric:
        for gripper in geometry["grippers"]:
            rendered: list[str] = []
            _render_gripper(gripper, rendered, geometry=geometry, control=control,
                        name_frame=group_symmetric)
            lines.append(f"`{gripper['end_effector_id']}`: " + rendered[0])
            lines += rendered[1:]
    for names, body, head in grouped:
        lines.append(" and ".join(f"`{name}`" for name in names) + ": " + head)
        lines += body
    lines += ["", "## Sensors and cameras", ""]
    for camera in descriptor["sensing"]["cameras"]:
        moving = camera["mounting"] in {"manipulator", "end_effector"}
        lines.append(f"`{camera['public_name']}`: {'arm-mounted moving' if moving else 'fixed'} RGB-D; use calibration from the same observation.")
    lines += ["", "## Preservation and evidence limits", "", "Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.", "", "## Dynamic facts and provenance", "", "Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only."]
    text = "\n".join(lines) + "\n"
    estimate = (len(text.encode())+3)//4
    if estimate > token_limit:
        raise EmbodimentDescriptorError(f"compiled card estimate {estimate} exceeds {token_limit} tokens")
    return text


def validate_reference_parameters(value: Any) -> None:
    """Typed nominal body facts; excluded from loaded collision-box verification."""
    value = _object(value, "reference_parameters")
    _exact_keys(value, required={"pad_approach_interval_m", "pad_lateral_interval_m",
        "contact_center_distance_m", "nominal_maximum_aperture_m",
        "empty_close_encoder_reference", "claim"}, optional=set(), path="reference_parameters")
    for key in ("pad_approach_interval_m", "pad_lateral_interval_m"):
        lower, upper = _vector(value[key], 2, key)
        if lower >= upper:
            raise EmbodimentDescriptorError("pad interval must have positive extent")
    for key in ("contact_center_distance_m", "nominal_maximum_aperture_m"):
        if _vector([value[key]], 1, key)[0] <= 0:
            raise EmbodimentDescriptorError("reference body length must be positive")
    if not 0 <= _vector([value["empty_close_encoder_reference"]], 1, "encoder")[0] <= 1:
        raise EmbodimentDescriptorError("normalized encoder reference must be in [0,1]")
    if value["claim"] != "nominal_source_geometry_not_physical_aperture_calibration":
        raise EmbodimentDescriptorError("reference parameters must state their nominal scope")
