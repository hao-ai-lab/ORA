"""Leave out of the policy's copy what carries no information (``observation_delivery_trim_v2``).

Decided 2026-10-05 from the six DeepSeek smoke runs at ``cf4ccf11``: a fresh
observation carried 6.6-10.1 KB of JSON, and a measurable part of it was
stated elsewhere in the same request. Every rule below names one field at one
place, the value under which it is left out, and where the information still
is. No rule detects constants, strips nulls in general or compares against
data seen earlier in the Session.

The rules apply to the delivered copy only, as the last step of each delivery
site (:mod:`.policy_delivery`, :mod:`ora.agent.native.effect_delivery`
and the Native session's observation envelope). The canonical observation,
receipt, ``tool_result`` event, sealed evidence and evaluation keep every byte.

Active rules (``CAPABILITY`` declared by the surface):

* **Per-camera identity defaults**: ``native_pixel_to_image_offset_xy`` when
  it is ``[0, 0]`` and ``image_view`` when it names the entry's own key. The
  ``observe`` description states both defaults.
* **Frame labels**: ``orientation_order`` and ``controlled_frame`` of an
  end effector (per arm, or once under ``conventions``). The embodiment card's
  "Controlled points" and "Frames and units" sections name the controlled
  point every pose refers to and the quaternion order ``[x, y, z, w]``.
* **Gripper arrival tolerance**: ``end_effectors.<arm>.gripper.feedback.arrival_check.tolerance_normalized``
  when it equals ``controller_feedback.gripper_tolerance_normalized`` of the
  same facts.
* **Untargeted arms**: an arm with ``targeted: false`` in ``detail.arms``
  loses its null fields and an ``inactive_arm_drift`` at or below 0.1 mm and
  0.01 deg; the block goes when only ``targeted`` would remain. ``target_arms``
  still names every targeted arm and the arm's state is in the facts.
* **Fixed prose** (:data:`CANONICAL_PROSE`): a ``claim_boundary``,
  ``evidence_scope`` or ``note`` string that equals, character for character,
  a canonical text the producing tool's description states once.
* **Static-facts absent listing** and **envelope defaults**: applied by the
  Native round static-facts slot and :mod:`.delivery_trim_native`.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

from .delivery_trim_prose import PROSE_KEYS, canonical_texts
from .policy_units import degrees_text

CAPABILITY = "observation_delivery_trim_v2"

RULE_IDENTITY_DEFAULTS = "camera_identity_defaults"
RULE_FRAME_LABELS = "frame_labels"
RULE_ARRIVAL_TOLERANCE = "gripper_arrival_tolerance"
RULE_UNTARGETED_ARMS = "untargeted_arms"
RULE_FIXED_PROSE = "fixed_prose"
RULE_ABSENT_LISTING = "static_facts_absent_listing"
RULE_ENVELOPE_DEFAULTS = "envelope_defaults"
RULES = frozenset({
    RULE_IDENTITY_DEFAULTS, RULE_FRAME_LABELS, RULE_ARRIVAL_TOLERANCE, RULE_UNTARGETED_ARMS,
    RULE_FIXED_PROSE, RULE_ABSENT_LISTING, RULE_ENVELOPE_DEFAULTS,
})
NO_RULES: frozenset[str] = frozenset()


def declared_rules(capabilities: Any) -> frozenset[str]:
    """The trim rules a surface's capabilities select."""
    return RULES if CAPABILITY in set(capabilities or ()) else NO_RULES


# -- Public robot facts ----------------------------------------------------------

#: The per-camera mapping offset that means "native pixels are image pixels".
IDENTITY_OFFSET = (0, 0)
#: End-effector labels the embodiment card states.
FRAME_LABEL_KEYS = ("orientation_order", "controlled_frame")


def _camera_entry(name: str, entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    result: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "native_pixel_to_image_offset_xy" and isinstance(value, list) \
                and tuple(value) == IDENTITY_OFFSET:
            continue
        if key == "image_view" and value == name:
            continue
        result[key] = value
    return result


def _without(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _arrival_tolerance(arm: Any, tolerance: Any) -> Any:
    """Drop the per-arm arrival tolerance that restates the controller's."""
    gripper = arm.get("gripper") if isinstance(arm, Mapping) else None
    feedback = gripper.get("feedback") if isinstance(gripper, Mapping) else None
    check = feedback.get("arrival_check") if isinstance(feedback, Mapping) else None
    if not isinstance(check, Mapping) or "tolerance_normalized" not in check \
            or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or check["tolerance_normalized"] != tolerance:
        return arm
    check = _without(check, ("tolerance_normalized",))
    return {**arm, "gripper": {**gripper, "feedback": {**feedback, "arrival_check": check}}}


def _end_effector(arm: Any, tolerance: Any) -> Any:
    if not isinstance(arm, Mapping):
        return arm
    return _arrival_tolerance(_without(arm, FRAME_LABEL_KEYS), tolerance)


def trim_public_facts(facts: Any) -> Any:
    """The current public robot facts without the rules' fields."""
    if not isinstance(facts, Mapping):
        return facts
    result = dict(facts)
    cameras = facts.get("cameras")
    if isinstance(cameras, Mapping):
        result["cameras"] = {name: _camera_entry(name, entry) for name, entry in cameras.items()}
    controller = facts.get("controller_feedback")
    tolerance = (controller.get("gripper_tolerance_normalized")
                 if isinstance(controller, Mapping) else None)
    effectors = facts.get("end_effectors")
    if isinstance(effectors, Mapping):
        result["end_effectors"] = {arm: _end_effector(value, tolerance)
                                   for arm, value in effectors.items()}
    conventions = facts.get("conventions")
    shared = conventions.get("end_effectors.<arm>") if isinstance(conventions, Mapping) else None
    if isinstance(shared, Mapping):
        kept = _without(shared, FRAME_LABEL_KEYS)
        result["conventions"] = ({**conventions, "end_effectors.<arm>": kept} if kept
                                 else _without(conventions, ("end_effectors.<arm>",)))
        if not result["conventions"]:
            del result["conventions"]
    return result


# -- Action effects --------------------------------------------------------------

#: An untargeted arm's drift at or below these bounds is measurement noise.
DRIFT_NOISE_TRANSLATION_M = 1e-4
DRIFT_NOISE_ORIENTATION_DEG = 0.01


def _drift_is_noise(drift: Any) -> bool:
    if not isinstance(drift, Mapping):
        return False
    translation = drift.get("translation_m")
    if "orientation_deg" in drift:
        orientation = drift.get("orientation_deg")
    elif "orientation_rad" in drift and isinstance(drift.get("orientation_rad"), (int, float)):
        orientation = math.degrees(drift["orientation_rad"])
    else:
        return False
    return all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in (translation, orientation)) \
        and abs(translation) <= DRIFT_NOISE_TRANSLATION_M \
        and abs(orientation) <= DRIFT_NOISE_ORIENTATION_DEG


def _untargeted_arm(state: Any) -> Any:
    if not isinstance(state, Mapping) or state.get("targeted") is not False:
        return state
    kept = {key: value for key, value in state.items()
            if value is not None and not (key == "inactive_arm_drift" and _drift_is_noise(value))}
    return None if set(kept) == {"targeted"} else kept


def _effect_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in detail.items():
        if key == "arms" and isinstance(value, Mapping):
            arms = {arm: _untargeted_arm(state) for arm, state in value.items()}
            value = {arm: state for arm, state in arms.items() if state is not None}
        result[key] = value
    return result


def trim_action_effect(effect: Any) -> Any:
    """One delivered action effect without the rules' fields."""
    if not isinstance(effect, Mapping):
        return effect
    result = dict(effect)
    if isinstance(effect.get("detail"), Mapping):
        result["detail"] = _effect_detail(effect["detail"])
    return drop_canonical_prose(result)


# -- Fixed prose -----------------------------------------------------------------

#: Canonical texts by key (:mod:`.delivery_trim_prose`), each stated once in the
#: description layer of the tool whose result or effect carries it, in the form
#: written and in the degree form ``policy_angles_degrees_v1`` delivers.
CANONICAL_PROSE: Mapping[str, frozenset[str]] = MappingProxyType({
    key: frozenset({*texts, *(degrees_text(text) for text in texts)})
    for key, texts in canonical_texts().items()})


def _is_canonical(key: Any, value: Any) -> bool:
    return key in PROSE_KEYS and isinstance(value, str) \
        and value in CANONICAL_PROSE.get(key, frozenset())


def drop_canonical_prose(value: Any) -> Any:
    """A copy without any ``claim_boundary``/``evidence_scope``/``note`` equal to a canonical text."""
    if isinstance(value, Mapping):
        return {key: drop_canonical_prose(child) for key, child in value.items()
                if not _is_canonical(key, child)}
    if isinstance(value, list):
        return [drop_canonical_prose(item) for item in value]
    return value


# -- Tool results ------------------------------------------------------------------

def trim_tool_result(result: Any) -> Any:
    """One robot-tool result (or refusal) without canonical prose."""
    if not isinstance(result, Mapping):
        return result
    delivered = dict(result)
    if isinstance(result.get("public_robot_facts"), Mapping):
        delivered["public_robot_facts"] = trim_public_facts(result["public_robot_facts"])
    return drop_canonical_prose(delivered)


__all__ = [
    "CANONICAL_PROSE", "CAPABILITY", "DRIFT_NOISE_ORIENTATION_DEG", "DRIFT_NOISE_TRANSLATION_M",
    "NO_RULES", "RULES", "RULE_ABSENT_LISTING", "RULE_ENVELOPE_DEFAULTS", "declared_rules",
    "drop_canonical_prose", "trim_action_effect", "trim_public_facts", "trim_tool_result",
]
