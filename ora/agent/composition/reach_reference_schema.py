"""The move-target reach pre-check that says which IK answered it (integration 0059, M1/M3).

``move_pose_reach_along_request_v1`` and ``reach_nearest_orientation_v1`` were
written for RoboDojo, whose reach check replays the route ``move_pose`` takes
(``matches_move_pose_admission: true``); their schema and claim say so. On
RoboTwin the same internal reach check is one damped-least-squares IK descent
per pose with joint limits only (``robotwin_single_seed_damped_least_squares_v1``),
while ``move_pose`` admits through the vendored 128-seed collision-checked
CuRobo IK. Porting the RoboDojo wording would deliver a false claim, so these
generations carry the backend's own answer instead:

``move_pose_reach_along_request_v2``
    ``compiled_pose.targets.<arm>.reach_along_request`` of ``move_pose`` names
    the reach check's ``ik_model``, ``matches_move_pose_admission``, the
    measured disagreement with ``move_pose``'s own admission when the two are
    not the same IK (:data:`MEASURED_DISAGREEMENT`), and the backend's claim
    boundary -- on a solved target too, where v1 said only ``true``.
``pose_primitive_reach_along_request_v2``
    the same field on ``move_to``, ``rotate_wrist`` and ``rotate_pitch``.
``reach_nearest_orientation_v2``
    the unsolved form also carries ``nearest_reachable_orientation`` (the v1
    shape), swept by the same reach check and holding the jaw centre the
    surface's controlled frame defines (``agent/tools/jaw_center.py``: the
    controlled point itself on a grip-centre frame).

Each is its own capability so every surface declaring a v1 name keeps its
bytes. Runtime half: ``agent/tools/reach_reference.py`` through
``move_pose_reach``. The jaw-symmetry turn (``move_pose_jaw_symmetry_headroom_v1``)
stays an ARX X5 capability and is not part of this generation.

M6 changes which check answers, not the field: a RoboTwin surface declaring
``robotwin_reach_check_inter_arm_screen_v1`` (:data:`INTER_ARM_SCREEN_CAPABILITY`)
has the backend screen every solved configuration against the other arm held
at its measured joints (``environment/backends/robotwin/reach_inter_arm.py``;
the RunSpec derives the session flag). The screened check names its own
``ik_model``, whose disagreement with ``move_pose``'s admission is not measured, so
:data:`MEASURED_DISAGREEMENT` lists it with ``None``. Its help restates the M1
sentence for the screened model in place of M1's component
(``closed_skills_ports_v1/robotwin-move-pose-reach-reference-inter-arm-endpoint-help-v1.json``);
:func:`bind_inter_arm_screen` refuses a surface whose move_pose help still names
the unscreened model, and gives the pooled grasp_plan annotation the screened
reason set.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.contracts.reach_check import (
    ROBOTWIN_INTER_ARM_IK_MODELS, ROBOTWIN_INTER_ARM_REASONS, ROBOTWIN_REASONS, SHARED_REASONS,
)

from .grasp_ik_reachability_schema import CAPABILITY as GRASP_IK_CAPABILITY
from .move_pose_reach_schema import REACH_CAPABILITY as MOVE_POSE_V1, REACH_PROPERTY
from .pose_primitive_reach_schema import CAPABILITY as PRIMITIVE_V1, PRIMITIVES
from .reach_orientation_schema import CAPABILITY as NEAREST_V1, NEAREST_PROPERTY

MOVE_POSE_CAPABILITY = "move_pose_reach_along_request_v2"
PRIMITIVE_CAPABILITY = "pose_primitive_reach_along_request_v2"
NEAREST_CAPABILITY = "reach_nearest_orientation_v2"
#: RoboTwin's reach-check other-arm endpoint screen (M6).
INTER_ARM_SCREEN_CAPABILITY = "robotwin_reach_check_inter_arm_screen_v1"
#: The one screen mode a surface can select.
INTER_ARM_SCREEN_MODE = "endpoint"
_REF = "ora/agent/composition/reach_reference_schema.py"


def inter_arm_screen_mode(capabilities: Sequence[str]) -> str | None:
    """``"endpoint"`` or ``None`` from a surface's capabilities."""
    return INTER_ARM_SCREEN_MODE if INTER_ARM_SCREEN_CAPABILITY in capabilities else None

#: Measured disagreement of a reach check whose IK is not ``move_pose``'s
#: admission, keyed by the ``ik_model`` the backend names. Fractions:
#: ``refused_given_move_pose_admits`` is the share of poses the admission IK
#: admits that this check refuses; ``reachable_given_move_pose_refuses`` the
#: share of poses it refuses that this check calls reachable.
#:
#: RoboTwin: the measurement of the same solver (``read_only_ik_reachability_v1``,
#: the same ``arrival_check`` tolerances) behind ``align_plan``, recorded in its
#: tool description (``explicit_motion_v3/robotwin-additional-definitions.json``)
#: and ``docs/reference/public-interfaces.md`` "Internal read-only reach check":
#: 7,860 poses ``align_plan`` synthesised from 500 measured target-and-arm states
#: across 114 runs against the backend's 128-seed collision-checked CuRobo IK;
#: about 5% false refusals (run-clustered 95% interval 5.0-9.4% before the
#: acceptance bound was corrected) and about 4% false admissions (2.9-5.0%),
#: concentrated on back, front and left approaches. None of that IK's 4,364
#: refusals was a joint limit: the false admissions are collisions this solver
#: does not model. A backend whose reach check *is* the admission IK, or one
#: never measured, gets ``null``.
#:
#: The screened RoboTwin model (M6) is listed as not measured: the screen
#: removes some of those false admissions and may add refusals, so neither
#: figure above describes it.
MEASURED_DISAGREEMENT: dict[str, dict[str, float] | None] = {
    "robotwin_single_seed_damped_least_squares_v1": {
        "refused_given_move_pose_admits": 0.05,
        "reachable_given_move_pose_refuses": 0.04,
    },
    ROBOTWIN_INTER_ARM_IK_MODELS[INTER_ARM_SCREEN_MODE]: None,
}

NEAREST_CLAIM = (
    "Read-only IK at each candidate endpoint by the same reach check as reach_along_request (its ik_model, "
    "matches_move_pose_admission and measured_disagreement apply), from the measured joints: the requested "
    "orientation turned about axes perpendicular to its approach axis, holding the jaw centre; no path, "
    "collision, payload or arrival claim, and an orientation outside the sweep may solve."
)

_FRACTION = {"type": "number", "minimum": 0, "maximum": 1}
DISAGREEMENT_PROPERTY = {
    "type": ["object", "null"], "additionalProperties": False,
    "required": ["refused_given_move_pose_admits", "reachable_given_move_pose_refuses"],
    "description": "Measured fractions against the IK move_pose admits through, when this check is not that IK: "
                   "of poses it admits, the share refused here; of poses it refuses, the share called reachable "
                   "here. null when this check is move_pose's admission or was not measured.",
    "properties": {"refused_given_move_pose_admits": dict(_FRACTION),
                   "reachable_given_move_pose_refuses": dict(_FRACTION)},
}
BASIS_PROPERTIES = {
    "ik_model": {"type": ["string", "null"]},
    "matches_move_pose_admission": {"type": ["boolean", "null"]},
    "measured_disagreement": DISAGREEMENT_PROPERTY,
}


def _v1_variant(solved: bool) -> dict[str, Any]:
    return copy.deepcopy(next(item for item in REACH_PROPERTY["oneOf"]
                              if item.get("properties", {}).get("target_reachable") == {"const": solved}))


def _with_basis(variant: dict[str, Any]) -> dict[str, Any]:
    variant["properties"].update(copy.deepcopy(BASIS_PROPERTIES))
    variant["properties"].setdefault("claim_boundary", {"type": "string", "minLength": 1})
    variant["required"] = [*variant["required"],
                           *[key for key in (*BASIS_PROPERTIES, "claim_boundary") if key not in variant["required"]]]
    return variant


REACH_REFERENCE_PROPERTY = {
    "type": "object",
    "description": "Read-only reach pre-check of the submitted target by this backend's reach check (ik_model), "
                   "from the measured joints. matches_move_pose_admission says whether that check is the IK "
                   "move_pose admits through; when it is not, the check is approximate and measured_disagreement "
                   "gives its measured error. When the target does not solve, the straight interpolation from "
                   "the measured pose is sampled.",
    "oneOf": [
        _with_basis(_v1_variant(True)),
        _with_basis(_v1_variant(False)),
        copy.deepcopy(next(item for item in REACH_PROPERTY["oneOf"]
                           if item.get("properties", {}).get("status") == {"const": "unavailable"})),
    ],
}


def _error(message: str) -> Exception:
    from .tool_surface import ToolSurfaceError

    return ToolSurfaceError(message)


def _descriptor(capability: str) -> dict[str, str]:
    return {"slot": f"derived:{capability}", "ref": _REF,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _target_properties(results: Mapping[str, Any], name: str, capability: str) -> dict[str, Any]:
    result = results.get(name)
    compiled = result.get("properties", {}).get("compiled_pose") if isinstance(result, dict) else None
    if not isinstance(compiled, dict) or compiled.get("type") != "object":
        raise _error(f"{capability} requires the {name} compiled_pose result")
    targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
    return targets.setdefault("additionalProperties", {"type": "object"}).setdefault("properties", {})


def _place(properties: dict[str, Any], name: str, capability: str) -> None:
    if "reach_along_request" in properties:
        raise _error(f"{capability} would overwrite {name} reach_along_request")
    properties["reach_along_request"] = copy.deepcopy(REACH_REFERENCE_PROPERTY)


def bind_move_pose_reach_reference(results: dict[str, Any], capabilities: Sequence[str]) -> dict[str, str]:
    """Add the v2 reach_along_request to move_pose's compiled_pose targets; identify the derivation."""
    if MOVE_POSE_V1 in capabilities:
        raise _error(f"{MOVE_POSE_CAPABILITY} and {MOVE_POSE_V1} are alternatives; declare one")
    if "enhanced_move_pose_v1" not in capabilities:
        raise _error(f"{MOVE_POSE_CAPABILITY} requires enhanced_move_pose_v1")
    _place(_target_properties(results, "move_pose", MOVE_POSE_CAPABILITY), "move_pose", MOVE_POSE_CAPABILITY)
    return _descriptor(MOVE_POSE_CAPABILITY)


def bind_pose_primitive_reach_reference(results: dict[str, Any], capabilities: Sequence[str]) -> dict[str, str]:
    """Add the v2 reach_along_request to every declared primitive's compiled_pose targets."""
    if PRIMITIVE_V1 in capabilities:
        raise _error(f"{PRIMITIVE_CAPABILITY} and {PRIMITIVE_V1} are alternatives; declare one")
    for required in ("semantic_pose_primitives_v1", "motion_intent_router_v1"):
        if required not in capabilities:
            raise _error(f"{PRIMITIVE_CAPABILITY} requires {required}")
    names = [name for name in PRIMITIVES if results.get(name) is not None]
    if not names:
        raise _error(f"{PRIMITIVE_CAPABILITY} requires at least one of {', '.join(PRIMITIVES)}")
    for name in names:
        _place(_target_properties(results, name, PRIMITIVE_CAPABILITY), name, PRIMITIVE_CAPABILITY)
    return _descriptor(PRIMITIVE_CAPABILITY)


def bind_reach_nearest_orientation_reference(
    results: dict[str, Any], capabilities: Sequence[str], view_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Add nearest_reachable_orientation to every declared unsolved v2 reach_along_request."""
    from ..tools.jaw_center import SUPPORTED_CONVENTIONS

    if NEAREST_V1 in capabilities:
        raise _error(f"{NEAREST_CAPABILITY} and {NEAREST_V1} are alternatives; declare one")
    if MOVE_POSE_CAPABILITY not in capabilities:
        raise _error(f"{NEAREST_CAPABILITY} requires {MOVE_POSE_CAPABILITY}")
    view = view_config or {}
    if not isinstance(view.get("public_geometry_descriptor"), Mapping):
        raise _error(f"{NEAREST_CAPABILITY} requires the public geometry descriptor for the jaw centre")
    if view.get("controlled_frame_convention") not in SUPPORTED_CONVENTIONS:
        raise _error(f"{NEAREST_CAPABILITY} requires a controlled frame with a known jaw centre")
    names = ["move_pose", *([name for name in PRIMITIVES if name in results]
                            if PRIMITIVE_CAPABILITY in capabilities else [])]
    for name in names:
        reach = _target_properties(results, name, NEAREST_CAPABILITY).get("reach_along_request")
        variant = next((item for item in (reach or {}).get("oneOf", [])
                        if item.get("properties", {}).get("target_reachable") == {"const": False}), None)
        if variant is None or "ik_model" not in variant["properties"]:
            raise _error(f"{NEAREST_CAPABILITY} requires the {name} v2 reach_along_request result")
        if "nearest_reachable_orientation" in variant["properties"]:
            raise _error(f"{NEAREST_CAPABILITY} would overwrite {name} nearest_reachable_orientation")
        variant["properties"]["nearest_reachable_orientation"] = copy.deepcopy(NEAREST_PROPERTY)
        variant["required"] = [*variant["required"], "nearest_reachable_orientation"]
    return _descriptor(NEAREST_CAPABILITY)


#: The ik_model the unscreened RoboTwin help sentence names; a screened surface
#: must not deliver it.
UNSCREENED_ROBOTWIN_MODEL = "robotwin_single_seed_damped_least_squares_v1"
#: The help component that restates M1's move_pose sentence for the screened model.
INTER_ARM_HELP = {
    INTER_ARM_SCREEN_MODE:
        "closed_skills_ports_v1/robotwin-move-pose-reach-reference-inter-arm-endpoint-help-v1.json",
}


def _move_pose_help(definitions: Sequence[Mapping[str, Any]]) -> str:
    entries = [item for item in definitions if item.get("name") == "move_pose"]
    return str(entries[0].get("description", "")) if len(entries) == 1 else ""


def _screen_grasp_reasons(results: dict[str, Any], mode: str) -> None:
    """Swap the RoboTwin stage reason enum of the pooled grasp_plan annotation for the screened set."""
    released = [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
    try:
        stages = results["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]["properties"]
        reasons = [stages[stage]["properties"]["reason"] for stage in ("pregrasp", "approach")]
    except (KeyError, TypeError) as error:
        raise _error(f"{INTER_ARM_SCREEN_CAPABILITY} requires the bound grasp_plan ik_reachability stages") \
            from error
    for reason in reasons:
        if reason.get("enum") != released:
            raise _error(f"{INTER_ARM_SCREEN_CAPABILITY} requires the unscreened RoboTwin stage reason set")
        reason["enum"] = [*SHARED_REASONS, *ROBOTWIN_INTER_ARM_REASONS[mode], None]


def bind_inter_arm_screen(
    definitions: Sequence[Mapping[str, Any]], results: dict[str, Any], capabilities: Sequence[str],
) -> dict[str, str]:
    """Bind RoboTwin's reach-check other-arm screen (M6); identify the derivation.

    The screen is backend-side (the RunSpec derives the session flags from these
    capabilities); here it widens the pooled grasp_plan annotation's stage reason
    enum to the screened set, and refuses a surface whose move_pose help still
    names the unscreened model. Runs after the grasp annotation binder.
    """
    mode = inter_arm_screen_mode(capabilities)
    if mode is None:
        raise _error(f"{INTER_ARM_SCREEN_CAPABILITY} is not declared")
    if "robotwin_rgbd_v1" not in capabilities:
        raise _error(f"{INTER_ARM_SCREEN_CAPABILITY} screens RoboTwin's reach check and requires robotwin_rgbd_v1")
    consumers = [name for name in (MOVE_POSE_CAPABILITY, GRASP_IK_CAPABILITY) if name in capabilities]
    if not consumers:
        raise _error(f"{INTER_ARM_SCREEN_CAPABILITY} requires a reach-check consumer: "
                     f"{MOVE_POSE_CAPABILITY} or {GRASP_IK_CAPABILITY}")
    if MOVE_POSE_CAPABILITY in capabilities:
        text = _move_pose_help(definitions)
        if f"ik_model {ROBOTWIN_INTER_ARM_IK_MODELS[mode]}" not in text or UNSCREENED_ROBOTWIN_MODEL in text:
            raise _error(f"{INTER_ARM_SCREEN_CAPABILITY}: move_pose help must describe the screened check; "
                         f"use {INTER_ARM_HELP[mode]} in place of M1's unscreened component")
    if GRASP_IK_CAPABILITY in capabilities:
        _screen_grasp_reasons(results, mode)
    return _descriptor(INTER_ARM_SCREEN_CAPABILITY)


__all__ = [
    "BASIS_PROPERTIES", "DISAGREEMENT_PROPERTY", "INTER_ARM_SCREEN_CAPABILITY",
    "MEASURED_DISAGREEMENT", "MOVE_POSE_CAPABILITY", "NEAREST_CAPABILITY", "NEAREST_CLAIM", "PRIMITIVE_CAPABILITY",
    "REACH_REFERENCE_PROPERTY", "bind_inter_arm_screen", "bind_move_pose_reach_reference",
    "bind_pose_primitive_reach_reference", "bind_reach_nearest_orientation_reference", "inter_arm_screen_mode",
]
