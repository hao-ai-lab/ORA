"""Closed grasp and place skills: one policy decision, several driven stages.

What the redesign changes
------------------------

The old shape was plan, then execute, then call a separate read-only tool to ask
whether it worked. The policy made a decision at every joint, including joints
where there was nothing to decide. These two skills collapse that::

    grasp_plan  -> look at candidates -> grasp_execute(candidate_id, rationale)
    place_plan  -> look at candidates -> place_execute(candidate_id, rationale,
                                                       stop_before_commit?)

``grasp_execute`` runs approach, close and verify with **no mid-way stop**. The
hard part of picking is the observation *before* it: which candidate, on what
evidence. Once the approach is committed there is nothing useful for a policy to
decide halfway down, and offering it a stop there only invites it to abandon a
good grasp on a partial view.

``place_execute`` runs transport, descend, an **optional** stop, release, a
measured settle, and verify. The stop lives here and only here, because
placing's hard moment is the instant of release: it is the last point at which a
close view can still change the outcome, and the only point in either skill
where more information is genuinely actionable.

Why a ledger rather than a loop
-------------------------------

A tool call submits exactly one action. The Supervisor executes it and the
receipt and fresh observation arrive on the following turn, so "one call" cannot
mean "three world mutations inside one handler". It means the *policy* is asked
once. This ledger holds the committed sequence across those turns so the Harness
can advance it without returning a decision point, in the same shape as the
holding gate's pending/reconcile ledger.

Nothing here moves a robot, and nothing here decides task success.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .robot import ToolError
from .skill_receipt import MAX_STAGES, validate_rationale


SCHEMA = "ora.manipulation-skill-ledger.v1"

#: Approach, close, verify. The verify stage absorbs what the separate
#: retention tool used to answer: displacement consistency plus the lift check.
GRASP_STAGES = ("approach", "close", "verify")
PREPARED_GRASP_STAGES = ("open", "pregrasp", "approach", "close", "lift", "verify")

#: Transport, descend, [stop], release, settle, verify. ``settle`` is measured
#: rather than assumed: the release is not the end of the motion.
PLACE_STAGES = ("transport", "descend", "stop", "release", "settle", "verify")

#: The one stage at which a closed skill hands the policy a view and waits.
POLICY_VISIBLE_STOP = "stop"

SKILLS = {"grasp_execute": GRASP_STAGES, "place_execute": PLACE_STAGES}

#: Stages that verify rather than mutate. They consume evidence and cannot fail
#: the skill by failing to change the world.
READ_ONLY_STAGES = frozenset({"stop", "verify"})

PHASES = frozenset({"running", "awaiting_receipt", "stopped", "completed", "aborted", "terminated"})


def stages_for(
    skill: str, *, stop_before_commit: bool = False, prepared_grasp: bool = False,
) -> tuple[str, ...]:
    """The committed stage sequence for one skill call.

    ``stop_before_commit`` is accepted only by ``place_execute``. Asking for it
    on a grasp is refused rather than ignored: silently dropping a requested
    stop would leave the policy believing it would be consulted.
    """

    if skill not in SKILLS:
        raise ToolError(f"unknown skill: {skill}", code="invalid_arguments", field="skill")
    if skill == "grasp_execute":
        if stop_before_commit:
            raise ToolError(
                "grasp_execute has no stop point: approach, close and verify run "
                "as one committed sequence. The decision a stop would serve is "
                "the candidate choice, which happens before this call.",
                code="invalid_arguments", field="stop_before_commit",
            )
        return PREPARED_GRASP_STAGES if prepared_grasp else GRASP_STAGES
    if stop_before_commit:
        return PLACE_STAGES
    return tuple(name for name in PLACE_STAGES if name != POLICY_VISIBLE_STOP)


def begin(
    *, skill: str, candidate_id: str, rationale: str,
    observation_id: str, observation_sha256: str,
    stop_before_commit: bool = False,
    prepared_grasp: bool = False,
) -> dict[str, Any]:
    """Commit one skill call and its stage sequence."""

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ToolError(
            "candidate_id must name a candidate from the matching plan call",
            code="invalid_arguments", field="candidate_id",
        )
    sequence = stages_for(
        skill, stop_before_commit=stop_before_commit, prepared_grasp=prepared_grasp,
    )
    if len(sequence) > MAX_STAGES:
        raise ToolError("skill sequence exceeds the receipt's stage budget",
                        code="internal_tool_error", retryable=False)
    return {
        "schema": SCHEMA,
        "skill": skill,
        "candidate_id": candidate_id,
        "rationale": validate_rationale(rationale),
        "stop_before_commit": bool(stop_before_commit),
        "sequence": list(sequence),
        "completed": [],
        "phase": "running",
        "observation_id": observation_id,
        "observation_sha256": observation_sha256,
    }


def validate_state(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or value.get("schema") != SCHEMA
            or value.get("skill") not in SKILLS
            or value.get("phase") not in PHASES
            or not isinstance(value.get("sequence"), list)
            or not isinstance(value.get("completed"), list)
            or len(value["completed"]) > len(value["sequence"])
            or [item["stage"] for item in value["completed"]]
            != value["sequence"][:len(value["completed"])]):
        raise ToolError("manipulation skill ledger is malformed",
                        code="skill_ledger_state", retryable=False)
    return copy.deepcopy(dict(value))


def pending_stage(state: Mapping[str, Any]) -> str | None:
    """The stage that is due next, or ``None`` when the sequence is finished."""

    state = validate_state(state)
    if state["phase"] in {"completed", "aborted", "terminated"}:
        return None
    done = len(state["completed"])
    if done >= len(state["sequence"]):
        return None
    return state["sequence"][done]


def yields_to_policy(state: Mapping[str, Any]) -> bool:
    """Whether the due stage hands the policy a view and waits.

    True only at ``place_execute``'s requested stop. Every other stage advances
    without consulting the policy, which is what makes these skills closed.
    """

    return pending_stage(state) == POLICY_VISIBLE_STOP


def record(
    state: Mapping[str, Any], *, stage: str, status: str, reason: str,
    action_id: str | None = None,
    measurements: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], str]:
    """Record one stage's outcome and advance, or fail the skill closed.

    A mutating stage that did not execute ends the skill. Continuing past an
    unconfirmed approach would descend from an unknown pose, and continuing past
    an unconfirmed release would verify a placement that never happened — both
    of which would produce a confident receipt describing something that did not
    occur.
    """

    updated = validate_state(state)
    due = pending_stage(updated)
    if due is None:
        raise ToolError("the skill sequence is already finished",
                        code="skill_ledger_state", retryable=False)
    if stage != due:
        raise ToolError(
            f"stage {stage} is out of order; {due} is due",
            code="skill_ledger_state", retryable=False,
        )
    updated["completed"].append({
        "stage": stage, "status": status, "reason": reason,
        "action_id": action_id, "measurements": [dict(item) for item in measurements],
    })
    if status == "terminated":
        updated["phase"] = "terminated"
        return updated, "environment_terminal"
    if status != "executed" and stage not in READ_ONLY_STAGES:
        updated["phase"] = "aborted"
        return updated, f"{stage}_unconfirmed_skill_aborted"
    if stage == POLICY_VISIBLE_STOP and status == "stopped":
        updated["phase"] = "stopped"
        return updated, "stopped_for_close_view_before_release"
    if pending_stage(updated) is None:
        updated["phase"] = "completed"
        return updated, "sequence_completed"
    updated["phase"] = "running"
    return updated, f"{stage}_recorded"


def resume(state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Continue a sequence the policy stopped, without re-deciding the rest.

    The stop returns a view, not the candidate choice. Resuming commits the
    remaining stages exactly as they stood; a policy that wants a different
    candidate must abandon this call and plan again.
    """

    updated = validate_state(state)
    if updated["phase"] != "stopped":
        raise ToolError("only a stopped skill can be resumed",
                        code="skill_ledger_state", retryable=False)
    updated["phase"] = "running"
    return updated, "resumed_after_close_view"


def abandon(state: Mapping[str, Any], reason: str) -> tuple[dict[str, Any], str]:
    """End a skill without executing its remaining stages."""

    updated = validate_state(state)
    if updated["phase"] in {"completed", "aborted", "terminated"}:
        raise ToolError("the skill has already ended",
                        code="skill_ledger_state", retryable=False)
    updated["phase"] = "aborted"
    return updated, f"abandoned:{reason}"


def terminal_status(state: Mapping[str, Any]) -> str:
    """Project the ledger phase onto the receipt's terminal status."""

    phase = validate_state(state)["phase"]
    if phase == "completed":
        return "completed"
    if phase == "stopped":
        return "stopped"
    if phase in {"aborted", "terminated"}:
        return phase
    raise ToolError("the skill has not reached a terminal status",
                    code="skill_ledger_state", retryable=False)
