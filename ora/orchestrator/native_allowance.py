"""The native-action allowance a policy is spending, reported beside the others.

A backend with an official native-action limit ends the episode when that
limit is spent, and a Harness physics-step cap below it ends the episode
sooner. Until this module the policy saw neither: every turn reported robot
decisions, model calls and wall time, while RoboDojo's binding limit (550 to
1600 native actions per task, 1000 under an earlier physics cap for a 1600-action
task) and the actions already used appeared nowhere, only the per-command
reservation and each receipt's own count.

The limit is read from the RunPlan: ``runtime.native_action_limit`` (the
official, task-registered limit) and ``budgets.max_sim_steps`` divided by the
backend's physics steps per native action (the Harness cap). The effective
limit is the smaller. Usage is summed from the executed receipts' own
``native_actions`` counts -- the same worker counters the RoboDojo evaluator
reconciles with the official verdict -- descending through staged composite
and committed skill-sequence wrappers. A receipt that carries no count makes
the remainder unknown (``None``) rather than guessed. A backend without an
official native-action limit reports nothing, so its remaining-budget shape is
unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.benchmarks import robodojo as _robodojo

#: Physics steps one native action advances, per backend that declares an
#: official native-action limit. A kind absent here has no Harness cap to fold.
PHYSICS_STEPS_PER_NATIVE_ACTION = {
    "robodojo": _robodojo.PHYSICS_STEPS_PER_NATIVE_ACTION,
}

#: Wrappers whose children carry the worker counts.
_CHILD_RECORD_KEYS = ("phase_receipts", "stage_receipts")
_WRAPPER_SCHEMAS = {
    "ora.composite-motion-receipt.v1",
    "ora.manipulation-skill-sequence-receipt.v1",
}


def effective_native_action_limit(plan: Mapping[str, Any]) -> int | None:
    """min(official limit, Harness physics cap), or None without a limit."""

    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    official = runtime.get("native_action_limit")
    if type(official) is not int or official <= 0:
        return None
    per_action = PHYSICS_STEPS_PER_NATIVE_ACTION.get(str(runtime.get("kind")))
    budgets = plan.get("budgets") if isinstance(plan.get("budgets"), Mapping) else {}
    cap = budgets.get("max_sim_steps")
    if per_action and type(cap) is int and cap > 0:
        return min(official, cap // per_action)
    return official


def receipt_native_actions(receipt: Any) -> int | None:
    """Native actions one executed receipt consumed, or None when uncounted."""

    if not isinstance(receipt, Mapping):
        return None
    nested = receipt.get("receipt")
    record = nested if isinstance(nested, Mapping) else receipt
    count = record.get("native_actions")
    if type(count) is int and count >= 0:
        return count
    if record.get("schema") not in _WRAPPER_SCHEMAS:
        return None
    total = 0
    for key in _CHILD_RECORD_KEYS:
        for item in record.get(key) or ():
            child = item.get("receipt") if isinstance(item, Mapping) else None
            if child is None:
                continue  # skipped or tool-driven stage: nothing dispatched
            counted = receipt_native_actions(child)
            if counted is None:
                return None
            total += counted
    return total


class NativeActionAllowance:
    """Per-round native-action usage against the effective limit."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        self.limit = effective_native_action_limit(plan)
        self.used: int | None = 0

    def reset(self) -> None:
        self.used = 0

    def record(self, receipt: Any) -> None:
        if self.limit is None or self.used is None:
            return
        counted = receipt_native_actions(receipt)
        self.used = None if counted is None else self.used + counted

    def fields(self) -> dict[str, Any]:
        """Remaining-budget entries; empty for a backend without a limit."""

        if self.limit is None:
            return {}
        return {
            "native_actions_limit": self.limit,
            "native_actions_remaining": (
                None if self.used is None else max(0, self.limit - self.used)
            ),
        }


__all__ = [
    "NativeActionAllowance", "PHYSICS_STEPS_PER_NATIVE_ACTION",
    "effective_native_action_limit", "receipt_native_actions",
]
