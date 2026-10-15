"""Which executed proposals spend the per-round robot-decision allowance.

The policy is told that the allowance counts accepted world mutations. Until
this module, the Coordinator charged every dispatched proposal, so a planner or
route refusal that executed nothing and left the world unchanged still cost one
decision: one recorded RoboDojo batch had 113 ``planner_failed`` and
five ``plan_rejected`` receipts, each with ``state_changed: false`` and zero
native actions, and each one was charged.

A refusal is recognised only from an explicit zero-write certificate on the
environment's own receipt: ``state_changed`` is ``False`` *and* at least one
native execution count is present and equal to zero, with no count nonzero.
``state_changed: false`` alone is not enough -- RoboTwin omits its zero
certificate exactly when execution was attempted or faulted, and missing
tracking is not evidence of zero execution. Anything without such a
certificate (composite phases that executed, committed skill sequences,
Synthetic receipts) keeps spending one decision.

Refusals remain fully recorded: they still close the observation, publish a
receipt and a fresh observation, spend model calls, and are counted separately
as ``refused_proposals`` on the round result, run result and status. The
model-call and wall-time allowances stay the hard stop for a policy that keeps
proposing refused actions.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Native execution counts a receipt may carry as its zero-write certificate.
ZERO_WRITE_COUNT_FIELDS = ("native_actions", "steps_used", "physics_steps")

COMPOSITE_RECEIPT_SCHEMA = "ora.composite-motion-receipt.v1"


def _record(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = receipt.get("receipt")
    return nested if isinstance(nested, Mapping) else receipt


def _certified_zero_write(record: Mapping[str, Any]) -> bool:
    if record.get("state_changed") is not False:
        return False
    counts = [record[field] for field in ZERO_WRITE_COUNT_FIELDS if field in record]
    return bool(counts) and all(
        type(count) is int and count == 0 for count in counts
    )


def is_zero_write_refusal(receipt: Any) -> bool:
    """True only for a receipt that certifies nothing executed or changed.

    A staged target-state composite qualifies when every phase that reached the
    environment is itself a certified zero-write refusal and at least one did;
    a phase skipped before dispatch (for example the gripper after a refused
    pose) carries no receipt and executed nothing.
    """

    if not isinstance(receipt, Mapping):
        return False
    record = _record(receipt)
    if record.get("schema") == COMPOSITE_RECEIPT_SCHEMA:
        phases = record.get("phase_receipts")
        if not isinstance(phases, list):
            return False
        dispatched = [
            phase.get("receipt") for phase in phases
            if isinstance(phase, Mapping) and phase.get("receipt") is not None
        ]
        return bool(dispatched) and all(
            is_zero_write_refusal(item) for item in dispatched
        )
    return _certified_zero_write(record)


__all__ = [
    "COMPOSITE_RECEIPT_SCHEMA", "ZERO_WRITE_COUNT_FIELDS", "is_zero_write_refusal",
]
