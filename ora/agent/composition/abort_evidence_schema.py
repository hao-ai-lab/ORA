"""The infrastructure-failure class an evidence-gated abort counts.

Under ``policy_abort_infrastructure_round_v2`` (``finish_outcomes_schema``)
``finish_episode(outcome=abort)`` is admitted only when this round's Harness
record holds an infrastructure-class failure. The record is the run's own event
log, never anything the policy asserts: the tool errors the tool runtime wrote
and the environment receipts the Supervisor wrote since the round's reset. The
reader is ``agent/tools/abort_evidence.py``.

The infrastructure class is a closed list taken from the Harness's own error
taxonomy. Tool errors whose code says the Harness, a model service, the
read-only bridge or a sensor failed (``TOOL_ERROR_CODES``); receipts whose
backend declared an execution fault or an uncertain write, whose planner stage
failed in the implementation rather than the solve (``receipt_projection``
publishes it as ``planner_error``: "never tell the policy to change its target
to repair our software"), or whose measured state was invalid; and a skill
sequence that stopped on a trusted execution or stage-driver fault. Behavioural
outcomes -- ``planner_failed``/``ik_failed``, plan rejections, ``controller_stalled``,
``gripper_stalled_before_target``, ``not_reached``, budget exhaustion -- and
refusals of invalid or stale input are not infrastructure, and a false
``check_task`` is a task state.
"""
from __future__ import annotations

#: Tool-error codes that say the Harness, not the policy's input, failed: the
#: one table every envelope's ``infrastructure`` flag also reads (EXP11B E1/A1).
from ..error_classes import INFRASTRUCTURE_CODES as TOOL_ERROR_CODES
#: Raw planner-stage codes of an implementation failure (``receipt_projection``).
PLANNER_FAULT_CODES = frozenset({"planner_error", "planner_internal_error", "planner_timeout"})
#: Raw execution-stage codes that are a sensor or controller fault, not a motion outcome.
EXECUTION_FAULT_CODES = frozenset({"invalid_measured_state"})
#: Skill-sequence stop reasons the Supervisor records for a trusted fault.
SKILL_FAULT_STOP_REASONS = frozenset({"trusted_execution_fault", "stage_driver_fault"})

ABORT_REQUIRES = "an infrastructure failure recorded by the Harness in this round"


def counted_classes() -> dict[str, list[str]]:
    """The classes the gate counts, read from its own tables."""
    return {
        "tool_error_codes": sorted(TOOL_ERROR_CODES),
        "receipt_classes": ["execution_fault", "execution_uncertain", *sorted(EXECUTION_FAULT_CODES)],
        "planner_fault_codes": sorted(PLANNER_FAULT_CODES),
        "skill_stop_reasons": sorted(SKILL_FAULT_STOP_REASONS),
    }


def infrastructure_classes_text() -> str:
    """The published class list: one generated section, from the gate's own tables (census A1)."""
    value = counted_classes()
    return (
        "Infrastructure failures the abort gate counts (generated from its own table; every tool error "
        "reports infrastructure true or false by the same table): tool errors with code "
        f"{', '.join(value['tool_error_codes'])}; environment receipts with {', '.join(value['receipt_classes'])} "
        f"or a planner stage failing with {', '.join(value['planner_fault_codes'])}; skill sequences stopped "
        f"by {', '.join(value['skill_stop_reasons'])}."
    )


__all__ = [
    "ABORT_REQUIRES", "EXECUTION_FAULT_CODES", "PLANNER_FAULT_CODES", "SKILL_FAULT_STOP_REASONS",
    "TOOL_ERROR_CODES", "counted_classes", "infrastructure_classes_text",
]
