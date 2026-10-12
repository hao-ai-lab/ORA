# Copyright contributors to the Robot Harness Runtime project.
"""What the policy named when it chose a target, resolved from the artifacts.

A grasp receipt that reports only jaw separation and visible extents cannot be
read for the question that actually goes wrong: whether the thing in the jaws
is the thing that was meant. The first closed-arm run re-grounded after every
grasp to ask it, and still spent a stretch of an episode treating one object as
another -- while the target it had grasped first was one it had selected by
pixel, never by phrase, and nothing in the receipt said so.

What this resolves is the commitment, never a verdict. A phrase is what the
policy said; whether it is what it got is a judgment made from the fresh frames
the next turn already delivers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

#: A target selected by pointing carries no phrase. Saying that is the point:
#: a selection made without language is itself the thing worth knowing, and
#: borrowing a neighbouring query's words would invent a commitment.
NOT_GROUNDED = "not_grounded_from_a_phrase"
GROUNDED = "grounded_from_a_phrase"


def unnamed_target() -> dict[str, Any]:
    """The record for a target no grounding query claims."""

    return {
        "phrase": None, "score": None, "rank": None,
        "camera": None, "observation_id": None, "selection": NOT_GROUNDED,
    }


def resolve_target_semantics(artifact_root: Any, target_ref: str) -> dict[str, Any]:
    """Find the grounding query that produced *target_ref*, if a phrase did."""

    root = Path(artifact_root) / "grounding-queries"
    for manifest_path in sorted(root.glob("*/manifest.json")) if root.is_dir() else ():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        identity = manifest.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        for candidate in manifest.get("candidates", ()):
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("target_ref")) != target_ref:
                continue
            score = candidate.get("backend_score")
            return {
                "phrase": str(identity.get("text") or "") or None,
                "score": float(score) if isinstance(score, (int, float)) else None,
                "rank": candidate.get("rank"),
                "camera": str(identity.get("camera") or "") or None,
                "observation_id": str(identity.get("observation_id") or "") or None,
                "selection": GROUNDED,
            }
    return unnamed_target()


__all__ = ["GROUNDED", "NOT_GROUNDED", "resolve_target_semantics", "unnamed_target"]
