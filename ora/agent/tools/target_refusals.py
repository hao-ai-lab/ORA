# Copyright contributors to the Robot Harness Runtime project.
#
# The refusals a target-bound tool raises, with the value each one refused.
#
# Three tools refuse a target that belongs to an earlier observation. Two of
# them carried the ref and one of those also named the way back; the third --
# the one `align_plan` reaches, and the most-called of the three -- passed only
# `field="target_ref"`. Thirteen refusals in one cohort told the policy that
# something was stale without saying what, and the same method threw away the
# quality numbers it had already measured. Both refusals live here so the pair
# is written once and reads the same from every caller.
#
# EXP11B census B3/D1-D3: a ref that cannot be loaded was one code for three
# different failures -- a malformed value, a well-formed ref no target carries,
# and a target store whose index or manifest drifted. The first two are the
# policy's argument; the third is the Harness's own store and must not blame
# the argument. `unavailable_target` keeps them apart, and every refusal of a
# target value states that only a different value can succeed.
"""Target-refusal errors that report the value and the measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifact_errors import ArtifactIntegrityError, TargetRefMalformed, TargetRefUnknown
from .robot import ToolError

#: The published shape of a segmented-target reference (``grasp.TARGET_REF_PATTERN``).
TARGET_REF_PATTERN = "^target-[0-9a-f]{24}$"
#: Refusals list at most this many current target refs.
MAX_LISTED_TARGET_REFS = 32


class GeometryQualityRefusal(Exception):
    """Carries the measurement that failed, so the refusal can report it."""

    def __init__(self, geometry: Mapping[str, Any]) -> None:
        super().__init__("target geometry quality is insufficient")
        self.geometry = geometry


def stale_target(
    *, target_ref: str, identity: Any, current_observation_id: str,
    field: str = "target_ref",
) -> ToolError:
    """The ref, the observation it was bound to, and the one in force now."""

    bound = (
        identity.get("observation_id") if isinstance(identity, Mapping) else None
    )
    return ToolError(
        f"{field} is stale: it was segmented on observation {bound}, and the "
        f"current observation is {current_observation_id}; a target_ref is "
        "valid only on the observation it was segmented on",
        code="stale_observation", field=field,
        received=target_ref, retryable=True,
        retry_requires_changed_arguments=True,
        constraints={
            "bound_observation_id": bound,
            "current_observation_id": current_observation_id,
        },
    )


def unmeasurable_geometry(
    *, target_ref: str, geometry: Mapping[str, Any],
) -> ToolError:
    """Which quality gates failed, what they measured, and what each needed.

    All three are already in the geometry result. Raising the sentence alone
    left the policy to guess whether to re-segment, move the camera, or give up
    on the target.
    """

    return ToolError(
        "current target mask/depth quality is insufficient to measure its "
        "geometry from this view",
        code="target_geometry_unavailable", field="target_ref",
        received=target_ref, retryable=True,
        retry_requires_changed_arguments=True,
        constraints={
            "failed_checks": geometry.get("quality_failures"),
            "measured": geometry.get("source_quality"),
            "minimums": geometry.get("quality_minimums"),
        },
    )


def current_target_refs(artifact_root: Path, observation_id: str) -> list[str]:
    """The indexed target refs segmented on one observation, sorted."""

    index = Path(artifact_root) / "segmented-target-index"
    refs = []
    for path in sorted(index.glob("target-*.json")) if index.is_dir() else ():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, Mapping) and record.get("observation_id") == observation_id:
            refs.append(str(record.get("target_ref")))
    return refs


def unavailable_target(
    error: ValueError, *, target_ref: Any, field: str = "target_ref",
    artifact_root: Path | None = None, observation_id: str | None = None,
) -> ToolError:
    """The envelope for a ref ``load_segmented_target`` could not load.

    A malformed or unknown ref is the policy's argument (``target_ref_unavailable``,
    an argument-refusal code) and names the pattern and the refs segmented on
    the current observation. A store fault keeps its own infrastructure code
    and is never retryable.
    """

    if isinstance(error, ArtifactIntegrityError):
        return ToolError(
            f"the segmented-target store failed for {field}: {error}",
            code=error.code, retryable=False, field=field, received=target_ref,
        )
    if isinstance(error, (TargetRefMalformed, TargetRefUnknown)):
        constraints: dict[str, Any] = {"pattern": TARGET_REF_PATTERN}
        if artifact_root is not None and observation_id is not None:
            refs = current_target_refs(artifact_root, observation_id)
            constraints.update(
                current_observation_id=observation_id,
                current_target_refs=refs[:MAX_LISTED_TARGET_REFS],
                current_target_ref_count=len(refs),
            )
        return ToolError(
            f"{field}: {error}", code="target_ref_unavailable", field=field,
            received=target_ref, constraints=constraints,
        )
    return ToolError(
        f"the segmented-target store failed for {field}: {error}",
        code="artifact_integrity_fault", retryable=False, field=field,
        received=target_ref,
    )


__all__ = [
    "GeometryQualityRefusal", "MAX_LISTED_TARGET_REFS", "TARGET_REF_PATTERN",
    "current_target_refs", "stale_target", "unavailable_target",
    "unmeasurable_geometry",
]
