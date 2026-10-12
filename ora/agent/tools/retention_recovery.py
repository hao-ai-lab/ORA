"""Source-joined multi-view retention; view quality is evidence, not a veto.

The Agent associates regions across cameras. Each temporal comparison still
uses one camera and one closed-gripper interval. A conflicting negative is
never hidden by selecting the best positive view.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from . import retention
from .robot import ToolError


CAPABILITY = "public_rgbd_grasp_retention_recovery_v5"
SCHEMA = "ora.grasp-retention-verification.v5"


def enabled(surface: Any) -> bool:
    return surface is not None and surface.supports(CAPABILITY)


def evidence_class(result: Mapping[str, Any]) -> str:
    if result["status"] == "consistent":
        return "supports_holding"
    if result["status"] == "inconsistent":
        return "contradicts_holding"
    reason = result["reason"]
    if not result["baseline"]["closed_interval_verified"]:
        return "invalid_source_interval"
    if reason == "predicted_target_motion_is_below_noise_separation":
        return "insufficient_motion"
    if result["comparison_eligible"] is False or reason.startswith("changed_view"):
        return "visibility_insufficient"
    return "motion_ambiguous"


def compare_views(
    *, arm: str, before_ref: str, current_ref: str,
    additional_pairs: Any,
    load_target: Callable[[str], tuple[dict[str, Any], dict[str, Any]]],
    load_step: Callable[[int], Mapping[str, Any]],
    current_state: Mapping[str, Any],
) -> dict[str, Any]:
    if (not isinstance(additional_pairs, list) or len(additional_pairs) > 2
            or any(not isinstance(pair, dict)
                   or set(pair) != {"before_target_ref", "current_target_ref"}
                   or any(not isinstance(ref, str) or not ref for ref in pair.values())
                   for pair in additional_pairs)):
        raise ToolError("additional_pairs must contain at most two before/current target pairs",
                        code="invalid_arguments", field="additional_pairs")
    pairs = [{"before_target_ref": before_ref, "current_target_ref": current_ref}, *additional_pairs]
    if len({tuple(pair[key] for key in ("before_target_ref", "current_target_ref"))
            for pair in pairs}) != len(pairs):
        raise ToolError("retention pairs must be unique", code="invalid_arguments")
    comparisons = []
    baseline_hash = None
    for pair in pairs:
        old_geometry, old_state = load_target(pair["before_target_ref"])
        new_geometry, new_state = load_target(pair["current_target_ref"])
        if baseline_hash is None:
            baseline_hash = old_state["state_sha256"]
        if (old_state["state_sha256"] != baseline_hash
                or new_state["state_sha256"] != current_state["state_sha256"]):
            raise ToolError("all camera pairs must join the same baseline and current public states",
                            code="stale_observation")
        result = retention.verify_retention(
            arm=arm, before_ref=pair["before_target_ref"], current_ref=pair["current_target_ref"],
            before_geometry=old_geometry, current_geometry=new_geometry,
            before_state=load_step(old_state["step"]), current_state=load_step(new_state["step"]),
            load_step=load_step, full_reference=True, allow_border_evidence=True,
        )
        result["quality_flags"] = [f"{label}_image_border" for label, geometry in (
            ("before", old_geometry), ("current", new_geometry))
            if geometry.get("source_quality", {}).get("touches_image_boundary") is True]
        result["evidence_class"] = evidence_class(result)
        comparisons.append(result)
    positives = [result for result in comparisons if result["status"] == "consistent"]
    negatives = [result for result in comparisons if result["status"] == "inconsistent"]
    # Preserve the witness's exact references and geometry; the separate
    # anchor joins the gate's existing baseline without relabelling a camera.
    witness = (negatives or positives or comparisons)[0]
    result = copy.deepcopy(witness)
    result["schema"] = SCHEMA
    result["anchor_before_target_ref"] = before_ref
    result["view_association"] = "agent_selected_same_object_not_authenticated"
    result["comparisons"] = comparisons
    if positives and negatives:
        result.update(status="unknown", reason="conflicting_camera_motion_evidence",
                      evidence_class="conflicting_views")
    result["recovery"] = {
        "baseline_step": comparisons[0]["baseline"]["before_state_step"],
        "current_step": current_state["step"],
        "next_step": (
            "use_current_permit_then_verify_again" if result["status"] == "consistent" else
            "inspect_or_open_before_regrasp" if result["status"] == "inconsistent" else
            "inspect_same_camera_baseline_and_current_or_use_remaining_local_probe"
        ),
        "prior_verified_baseline_step": None,
    }
    return result
