"""How a closed skill's pose stage reports a sparse RoboDojo motion stop.

A Dojo stage move that stalls (``environment/backends/robodojo/motion_stall.py``)
carries per-arm ``motion_progress`` in its driven receipt, which the policy
never sees: the skill result projects only each stage's status, reason and
measurements. The measured stop therefore joins the stage measurements, and
the stage reason and endpoint residual name the public failure code rather
than the backend's private one (``native_action_budget_exhausted`` is
``controller_timeout`` in every other policy-visible receipt).
"""
from __future__ import annotations

from typing import Any, Mapping

from .gripper_stage import DOJO_SPARSE_RECEIPT_SCHEMA

__all__ = ["dojo_motion_stage_evidence"]


def _public_code(execution: Mapping[str, Any]) -> str | None:
    from ora.agent.evidence.motion_feedback import stall_evidenced_code
    from ora.agent.native.receipt_projection import public_motion_failure_code

    code = public_motion_failure_code(execution.get("failure_stage"), execution.get("failure_code"))
    return stall_evidenced_code(execution, code)


def dojo_motion_stage_evidence(
    execution: Mapping[str, Any], residual: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The stage residual with a public reason, and the extra stage measurements.

    Other receipts pass through unchanged with no extra measurement.
    """
    if execution.get("schema") != DOJO_SPARSE_RECEIPT_SCHEMA:
        return dict(residual), []
    public = dict(residual)
    if execution.get("failure_code") is not None and residual.get("reason") == execution.get("failure_code"):
        public["reason"] = _public_code(execution)
    progress = {arm: dict(effect["motion_progress"]) for arm, effect in (execution.get("arms") or {}).items()
                if isinstance(effect, Mapping) and isinstance(effect.get("motion_progress"), Mapping)
                and effect["motion_progress"].get("outcome") != "arrived"}
    return public, ([{"quantity": "motion_progress", "value": progress}] if progress else [])
