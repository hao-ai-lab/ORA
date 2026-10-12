"""``check.hold``: one hold verdict fused from the evidence a grasp already measured.

Whether the jaws are holding something is spread over several schemas: the
closure-extent evidence (jaw separation beside the target's visible extent),
the lift probe's visible-surface motion, RoboDojo's jaw-stall hold evidence,
and, last, a runtime-model judgement from the shared question bank. This
module fuses their **verdicts**, never their geometry, by a versioned rule
table: any contradicting source makes ``contradicted``; otherwise any
supporting source makes ``supported``; otherwise ``unknown``. Every source is
kept in the output with what it said and how the rule read it, and a model
judgement is labelled as one.

A verdict is evidence, not a guarantee: ``supported`` is not attachment,
stable transport or task success, and ``contradicted`` is not proof the
object is elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = "ora.hold-verdict.v1"
RULES = "hold-fusion-v1"
CLAIM_BOUNDARY = (
    "a fusion of measured hold evidence: supported is not attachment, stable transport or task success, "
    "and contradicted is not proof of where the object is"
)

#: (source, value) -> how the rule reads it. Values not listed are neutral.
RULE_TABLE: dict[str, dict[str, str]] = {
    "closure_extent": {
        "strong_negative": "contradicts",
        "weak_negative": "neutral",
        "not_contradictory": "neutral",
        "unavailable": "neutral",
    },
    "lift_surface_motion": {
        "rigid_motion_supported": "supports",
        "stationary_surface_supported": "contradicts",
        "unknown": "neutral",
    },
    "jaw_stall": {
        "likely": "supports",
        "unlikely": "contradicts",
        "unknown": "neutral",
    },
    "model_judgment": {
        "held": "supports",
        "empty": "contradicts",
        "unclear": "neutral",
        "unavailable": "neutral",
    },
}


def _source(name: str, schema: str | None, value: Any, reason: Any) -> dict[str, Any]:
    reading = RULE_TABLE[name].get(value, "neutral") if isinstance(value, str) else "neutral"
    return {"source": name, "schema": schema, "value": value, "reason": reason, "reading": reading}


def sources_from_verification(verification: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The hold sources a grasp verify stage's verification block carries."""

    if not isinstance(verification, Mapping):
        return []
    found = []
    if "strength" in verification:
        found.append(_source("closure_extent", verification.get("schema"), verification.get("strength"),
                             verification.get("reason")))
    probe = verification.get("lift_probe")
    if isinstance(probe, Mapping) and isinstance(probe.get("surface_motion"), Mapping):
        motion = probe["surface_motion"]
        found.append(_source("lift_surface_motion", None, motion.get("status"), motion.get("reason")))
    jaw = verification.get("jaw_hold")
    if isinstance(jaw, Mapping):
        found.append(_source("jaw_stall", jaw.get("schema"), jaw.get("held"), jaw.get("reason")))
    return found


def judgment_source(answer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A question-bank answer as the last, labelled source."""

    if not isinstance(answer, Mapping):
        return None
    value = answer.get("answer") if answer.get("status") == "answered" else "unavailable"
    source = _source("model_judgment", answer.get("question"), value, answer.get("reason"))
    source["model_judgment"] = True
    return source


def fuse(sources: list[Mapping[str, Any]]) -> dict[str, Any]:
    readings = [item["reading"] for item in sources]
    if "contradicts" in readings:
        hold = "contradicted"
    elif "supports" in readings:
        hold = "supported"
    else:
        hold = "unknown"
    return {
        "schema": SCHEMA, "hold": hold, "rules": RULES, "sources": [dict(item) for item in sources],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def hold_verdict(verification: Mapping[str, Any] | None, judgment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``check.hold``: fuse a verify block's sources, and a model judgement if given."""

    sources = sources_from_verification(verification)
    extra = judgment_source(judgment)
    if extra is not None:
        sources.append(extra)
    return fuse(sources)
