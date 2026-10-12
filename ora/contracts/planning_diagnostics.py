"""Bounded planning-failure feedback shared across isolated backends.

Diagnostics describe solver evidence, not task truth or execution authority.
Full seed vectors and implementation traces stay in canonical artifacts.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

SCHEMA = "ora.planning-diagnostics.v1"

#: Probe request that asks a backend to keep its bounded IK candidate evidence
#: on each variant's planner record. Only a surface declaring
#: ``motion_probe_ik_candidate_feedback_v1`` sends it; every other surface keeps
#: sending the v1 request, so a frozen probe result schema never meets the
#: extra planner fields.
PROBE_FEEDBACK_REQUEST_SCHEMA = "ora.motion-probe-request.v2"
PROBE_IK_CANDIDATE_FEEDBACK = "ik_candidate_counts_v1"
#: The measured IK filter names a planner record may count, in the RoboTwin
#: spelling the probe summary already carries, followed by the filters only a
#: bounded SDK solve applies (endpoint FK residual, raw joint jump and the
#: RoboDojo pitch-joint winding refusal).
IK_CONSTRAINT_KEYS = (
    "non_finite_solution", "solver_unsuccessful", "joint_limit",
    "fk_residual_exceeded", "joint_jump", "pitch_joint_wrap",
)
PHASES = frozenset({"ik", "ik_candidate_validation", "trajectory_planning", "joint_space", "start_state", "unknown"})
REQUIRED = frozenset({"phase", "outcome", "candidate_counts", "rejection_counts", "details_available", "recovery"})
OPTIONAL = frozenset({"schema", "note", "unknown_reasons", "evidence_scope"})


def validate_planning_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not REQUIRED <= value.keys() or set(value) - REQUIRED - OPTIONAL:
        raise ValueError("planning diagnostics fields drifted")
    if value["phase"] not in PHASES or type(value["details_available"]) is not bool:
        raise ValueError("planning diagnostics phase or availability is invalid")
    if "schema" in value and value["schema"] != SCHEMA:
        raise ValueError("planning diagnostics schema drifted")
    for key in ("outcome", "note", "evidence_scope"):
        if key in value and (not isinstance(value[key], str) or not value[key] or len(value[key]) > 1200):
            raise ValueError("planning diagnostics text is invalid")
    for key in ("candidate_counts", "rejection_counts"):
        counts = value[key]
        if (not isinstance(counts, Mapping) or len(counts) > 32
                or any(not isinstance(k, str) or not k or len(k) > 100
                       or v is not None and (type(v) is not int or v < 0) for k, v in counts.items())):
            raise ValueError("planning diagnostics counts are invalid")
    for key in ("recovery", "unknown_reasons"):
        if key in value and (not isinstance(value[key], list) or len(value[key]) > 4
                or any(not isinstance(x, str) or not x or len(x) > 1200 for x in value[key])):
            raise ValueError("planning diagnostics explanations are invalid")
    return copy.deepcopy(dict(value))


def public_planning_diagnostics(value: Any) -> dict[str, Any]:
    """Omit unavailable/invalid historical diagnostics without inventing zeros."""
    if not isinstance(value, Mapping):
        return {}
    try:
        result = validate_planning_diagnostics({k: v for k, v in value.items() if k in REQUIRED | OPTIONAL})
    except (ValueError, TypeError):
        return {}
    result.pop("schema", None)
    return result
