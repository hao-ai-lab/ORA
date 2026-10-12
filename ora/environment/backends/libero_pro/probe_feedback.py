"""LIBERO's answer to the v2 motion probe: CuRobo IK candidate counts per variant.

A mixin of ``session.LiberoProSession`` placed before ``LiberoMotionProbeMixin``
in the bases; methods only, no dataclass fields. A surface declaring
``motion_probe_ik_candidate_feedback_v1`` sends ``ora.motion-probe-request.v2``
(``contracts/planning_diagnostics.py``) and RoboDojo answers it with its SDK
IK candidates. On LIBERO every probe variant is a read-only CuRobo
``plan_single``, and its IK stage is the MotionGen IK solver the reach check
(``reach_check.py``) solves with; the worker's IK observer
(``planning_diagnostics.install_ik_observer``) already keeps that stage's seed
population in each plan response. So the counts here are the evidence the
variant was admitted or refused on, not a second solve.

``motion_probe.py`` is a pinned source of the LIBERO embodiment descriptors and
is not edited: a v2 request runs the released v1 probe unchanged while the
planner's responses are recorded in call order (one plan per variant), and each
variant's planner record then gains exactly the three v2 fields RoboDojo adds --
``constraint_rejection_counts`` over every ``IK_CONSTRAINT_KEYS`` filter,
``planning_diagnostics`` with ``candidate_counts`` ``{returned, valid_endpoint}``,
and ``ik_validation_artifact``. A v1 request, or any other schema, reaches the
released probe untouched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS, PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
    public_planning_diagnostics,
)

from .common import LiberoProError, _atomic_json
from .planning_diagnostics import TALLY_ORDER, ik_candidate_tally

PROBE_REQUEST_SCHEMA = "ora.motion-probe-request.v1"
_FIELDS = frozenset({"schema", "observation_id", "observation_sha256", "queries", "variant_policy"})
IK_EVIDENCE_SCHEMA = "ora.libero-probe-ik.v1"
TALLY_NOTE = (
    "LIBERO CuRobo IK seed population of the plan's last IK attempt: returned counts every optimised seed, "
    "valid_endpoint those feasible and within the solver's thresholds; each other seed is counted once, under "
    "joint_limit, else solver_unsuccessful (constraint check failed, including self-collision), else "
    "fk_residual_exceeded (not converged). joint_jump and pitch_joint_wrap are not LIBERO filters. No world model."
)


class _RecordingPlanner:
    """Forward to the real planner client and keep each plan response, in order."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.responses: list[Mapping[str, Any]] = []

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self.inner.plan(request)
        self.responses.append(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _diagnostics(planned: Mapping[str, Any], tally: Mapping[str, Any] | None) -> dict[str, Any]:
    """The plan's public diagnostics with the counts in the shared probe vocabulary."""
    base = public_planning_diagnostics(planned.get("planning_diagnostics"))
    if not base:
        return {}
    counts = {} if tally is None else {"returned": tally["returned"], "valid_endpoint": tally["valid_endpoint"]}
    rejections = {} if tally is None else {key: tally["rejections"][key] for key in TALLY_ORDER}
    return {**base, "candidate_counts": counts, "rejection_counts": rejections,
            "details_available": tally is not None, "note": TALLY_NOTE}


class LiberoProbeFeedbackMixin:
    """Answer the v2 probe request; leave every other request to the released probe."""

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping) or request.get("schema") != PROBE_FEEDBACK_REQUEST_SCHEMA:
            return super().probe_motion(request)  # type: ignore[misc]
        if (set(request) != _FIELDS | {"planner_feedback"}
                or request.get("planner_feedback") != PROBE_IK_CANDIDATE_FEEDBACK):
            raise LiberoProError("motion probe request fields drifted")
        if self.motion_planner is None:
            raise LiberoProError("CuRobo planner is not active")
        released = {**{key: request[key] for key in _FIELDS}, "schema": PROBE_REQUEST_SCHEMA}
        first_evidence = getattr(self, "curobo_probe_sequence", 0)
        recorder = _RecordingPlanner(self.motion_planner)
        self.motion_planner = recorder
        try:
            result = super().probe_motion(released)  # type: ignore[misc]
        finally:
            self.motion_planner = recorder.inner
        rows = [(query["query_id"], variant) for query in result["results"] for variant in query["variants"]]
        if len(rows) != len(recorder.responses):
            raise LiberoProError("motion probe variants and planner calls disagree")
        for index, ((query_id, variant), planned) in enumerate(zip(rows, recorder.responses)):
            variant["planner"] = {**variant["planner"], **self._ik_feedback(
                planned, request=request, query_id=query_id, variant_id=variant["variant_id"],
                planner_evidence=f"curobo/probes/request-{first_evidence + index:06d}.json")}
        return result

    def _ik_feedback(self, planned: Mapping[str, Any], *, request: Mapping[str, Any], query_id: str,
                     variant_id: str, planner_evidence: str) -> dict[str, Any]:
        """The three v2 planner fields of one variant; its IK evidence written beside the probe."""
        evidence = planned.get("planning_diagnostic_evidence")
        attempts = evidence.get("attempts") if isinstance(evidence, Mapping) else None
        attempt = attempts[-1] if isinstance(attempts, list) and attempts else None
        tally = ik_candidate_tally(attempt)
        evidence_path = self.output_root / planner_evidence
        record = {
            "schema": IK_EVIDENCE_SCHEMA, "observation_id": request["observation_id"],
            "observation_sha256": request["observation_sha256"],
            "world_mutation_sequence": self.world_mutation_sequence, "round_index": self.round_index,
            "query_id": query_id, "variant_id": variant_id,
            "planner_status": planned.get("planner_status") or planned.get("status"),
            "planner_evidence": {"path": planner_evidence, "sha256": (
                hashlib.sha256(evidence_path.read_bytes()).hexdigest() if evidence_path.is_file() else None)},
            "ik_attempt": attempt, "tally": tally, "tally_rule": TALLY_NOTE,
        }
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        relative = f"motion-probes/ik-diagnostics/{digest}.json"
        _atomic_json(self.output_root / relative, record)
        diagnostics = _diagnostics(planned, tally)
        return {
            "constraint_rejection_counts": {key: 0 if tally is None else int(tally["rejections"][key])
                                            for key in IK_CONSTRAINT_KEYS},
            **({"planning_diagnostics": diagnostics} if diagnostics else {}),
            "ik_validation_artifact": {"path": relative, "schema": IK_EVIDENCE_SCHEMA, "sha256": hashlib.sha256(
                (self.output_root / relative).read_bytes()).hexdigest()},
        }


__all__ = ["IK_EVIDENCE_SCHEMA", "LiberoProbeFeedbackMixin", "TALLY_NOTE"]
