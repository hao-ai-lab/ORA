"""The pre-act round-start isolation check of a persistent multi-round session.

A persistent session deliberately carries one thing across a round boundary:
the AgentSession's own context.  Everything else -- the world, the official
evaluator, the Control-owned policy-visible files and the tool runtime's
in-memory references -- belongs to exactly one round, and a value that
survives is not a harmless leftover.  Two of them are actively destructive:
``policy-task-check.json`` is read verbatim by ``check_task``, so a copy left
behind answers the new round with the *previous* round's official outcome, and
a carried execution receipt or ``plan_ref`` describes a world the reset has
already thrown away.

This module is the single place the Supervisor asks "is the new round
actually isolated?", and it asks *before* the round's first observation is
published, so a failed restore stops the run instead of reaching the policy.
It measures rather than asserts: the tool runtime reports what it still holds
and the environment's reset record carries the backend's own measured proof.
A failure here is an infrastructure fault -- the round keeps a null official
verdict -- never a task outcome.
"""

from __future__ import annotations

from typing import Any, Mapping


ROUND_START_ISOLATION_SCHEMA = "ora.round-start-isolation.v1"
SESSION_ISOLATION_SCHEMA = "ora.session-round-isolation.v1"


class RoundIsolationError(RuntimeError):
    """A new round would start on a world or channel that is not restored."""


def _tool_report(tools: Any) -> dict[str, Any]:
    """What the tool runtime still holds, or an explicit absence of evidence."""

    reporter = getattr(tools, "round_start_isolation", None)
    if not callable(reporter):
        return {
            "schema": ROUND_START_ISOLATION_SCHEMA,
            "passed": True,
            "audited_control_files": [],
            "audited_runtime_registries": [],
            "residual_control_files": [],
            "residual_runtime_registries": [],
            "failures": [],
            "available": False,
        }
    report = dict(reporter())
    report["available"] = True
    return report


def _environment_proof(reset: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the backend's own restore proof when it publishes one.

    A backend that proves its restore -- RoboDojo does -- carries a
    ``round_isolation`` block in its round-reset record.  When the block is
    present it must pass; when it is absent the report says so in
    ``environment_proof: null`` rather than silently claiming a guarantee the
    backend never made.  The backend-specific evidence join is what refuses a
    *missing* block, because only that join knows which backend owes one.
    """

    proof = reset.get("round_isolation")
    if proof is None:
        return None, []
    if not isinstance(proof, Mapping):
        return None, ["environment round-isolation proof is not a record"]
    failures = [
        f"environment: {reason}"
        for reason in proof.get("failures", [])
        if isinstance(reason, str)
    ]
    if proof.get("passed") is not True and not failures:
        failures.append("environment round-isolation proof did not pass")
    checks = proof.get("checks")
    if isinstance(checks, Mapping):
        failures.extend(
            f"environment check {name} did not hold"
            for name, passed in sorted(checks.items())
            if passed is not True
        )
    return dict(proof), sorted(set(failures))


def round_start_isolation(
    tools: Any, reset: Mapping[str, Any], *, round_index: int,
) -> dict[str, Any]:
    """Report whether ``round_index`` may start, without acting on it."""

    failures: list[str] = []
    report = _tool_report(tools)
    failures.extend(str(reason) for reason in report.get("failures", []))
    proof, proof_failures = _environment_proof(reset)
    failures.extend(proof_failures)
    recorded_round = reset.get("round_index")
    if recorded_round is not None and recorded_round != round_index:
        failures.append(
            f"environment reset record names round {recorded_round!r}, "
            f"not {round_index}"
        )
    return {
        "schema": SESSION_ISOLATION_SCHEMA,
        "passed": not failures,
        "round_index": round_index,
        "tool_runtime": report,
        "environment_proof": proof,
        "carried_across_rounds": ["agent_session_context"],
        "failures": sorted(set(failures)),
    }


def require_round_isolation(
    tools: Any, reset: Mapping[str, Any], *, round_index: int,
) -> dict[str, Any]:
    """Return the report, or refuse the round before the policy can act."""

    report = round_start_isolation(tools, reset, round_index=round_index)
    if not report["passed"]:
        raise RoundIsolationError(
            f"round {round_index} would start on a world that is not fully "
            "restored: " + "; ".join(report["failures"])
        )
    return report


__all__ = [
    "ROUND_START_ISOLATION_SCHEMA",
    "SESSION_ISOLATION_SCHEMA",
    "RoundIsolationError",
    "require_round_isolation",
    "round_start_isolation",
]
