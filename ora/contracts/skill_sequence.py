"""One committed manipulation sequence, submitted as a single policy decision.

Why this is an action schema rather than a loop in a tool handler
----------------------------------------------------------------

A tool call writes one ``submitted-action.json`` and returns. The Supervisor
executes it, and the receipt plus the fresh observation reach the policy on the
*next* turn. So "approach, close and verify in one call" cannot mean three
mutations inside one handler.

The Harness already has the shape it does mean. ``target_state_phases``
expands one v4 action into a pose phase and a gripper phase and the trusted
executor runs both, skipping the gripper when the pose did not arrive. It
counts as **one** robot decision and returns **one** receipt.

This schema is the second member of that family and the first whose per-stage
decisions are geometric. The policy is asked once; the
trusted executor drives the committed stages; the ledger in
``ora/agent/tools/skill_ledger.py`` is the state it drives.

What the action may and may not carry
-------------------------------------

The action carries the *commitment* — which skill, which candidate, why, and
the exact stage sequence that was committed — and no geometry. Stage poses are
produced by the tool runtime, which owns the candidate artifacts and their
observation binding; putting a pose here would create a second place where the
same number lives and a second thing to keep in agreement.

``resume_from`` exists for exactly one case: ``place_execute`` stopped at its
requested pre-release view and the policy resumed it. Resuming re-submits the
same commitment with the already-completed prefix skipped. It cannot change the
skill, the candidate or the sequence, because the stop returned a view, not the
candidate choice back.
"""

from __future__ import annotations

from typing import Any, Mapping


SKILL_SEQUENCE_ACTION_SCHEMA = "ora-skill-sequence-action.v1"

#: The declared skills and their full stage vocabularies. This module owns the
#: transport-neutral *shape*; ``skill_ledger`` owns which stages a given call
#: commits to and what a stage outcome means.
SKILL_SEQUENCE_STAGES = {
    "grasp_execute": ("open", "pregrasp", "approach", "close", "lift", "verify"),
    "place_execute": (
        "transport", "descend", "stop", "release", "settle", "verify",
    ),
}

#: The most stages any committed sequence may declare. It matches
#: ``skill_receipt.MAX_STAGES`` because one receipt reports one sequence.
MAX_SEQUENCE_STAGES = 6

#: The rationale bound is the receipt's, restated at the trust boundary rather
#: than assumed: a transport-neutral contract cannot import the tool package.
MAX_RATIONALE_CHARS = 200


def validate_skill_sequence_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one committed skill sequence at the Agent/Supervisor boundary.

    Raises ``ValueError``; ``validate_runtime_action`` rewrites it into the
    shared ``ActionValidationError`` the rest of the action contract uses.
    """

    for field in ("observation_id", "action_id", "bound_observation_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be non-empty")
    if payload.get("action") != "execute_skill_sequence":
        raise ValueError("action must be execute_skill_sequence")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be an object")
    expected = {
        "skill", "candidate_id", "rationale", "stop_before_commit",
        "sequence", "resume_from",
    }
    if set(arguments) != expected:
        raise ValueError(
            "skill sequence arguments must contain exactly "
            + ", ".join(sorted(expected))
        )
    skill = arguments["skill"]
    if skill not in SKILL_SEQUENCE_STAGES:
        raise ValueError("skill must be grasp_execute or place_execute")
    for field in ("candidate_id", "rationale"):
        value = arguments[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if len(arguments["rationale"].strip()) > MAX_RATIONALE_CHARS:
        raise ValueError(
            f"rationale must be at most {MAX_RATIONALE_CHARS} characters"
        )
    stop_before_commit = arguments["stop_before_commit"]
    if not isinstance(stop_before_commit, bool):
        raise ValueError("stop_before_commit must be a boolean")
    if stop_before_commit and skill != "place_execute":
        raise ValueError("only place_execute declares a pre-release stop")
    sequence = arguments["sequence"]
    declared = SKILL_SEQUENCE_STAGES[skill]
    if (
        not isinstance(sequence, list) or not sequence
        or len(sequence) > MAX_SEQUENCE_STAGES
        or any(stage not in declared for stage in sequence)
        # A committed sequence is a subsequence of the skill's declared stages
        # in declared order. Reordering it would let a submitted action release
        # before it descended.
        or [stage for stage in declared if stage in set(sequence)] != sequence
    ):
        raise ValueError(
            f"sequence must be an ordered subset of {list(declared)}"
        )
    if ("stop" in sequence) != stop_before_commit:
        raise ValueError(
            "the stop stage is present exactly when stop_before_commit is set"
        )
    resume_from = arguments["resume_from"]
    if (
        isinstance(resume_from, bool) or not isinstance(resume_from, int)
        or not 0 <= resume_from < len(sequence)
    ):
        raise ValueError("resume_from must index a stage of the sequence")
    if resume_from and sequence[resume_from - 1] != "stop":
        raise ValueError(
            "a sequence resumes only after its requested pre-release stop"
        )
    return {
        "schema": SKILL_SEQUENCE_ACTION_SCHEMA,
        "observation_id": payload["observation_id"],
        "action_id": payload["action_id"],
        "action": "execute_skill_sequence",
        "arguments": {
            "skill": skill,
            "candidate_id": arguments["candidate_id"],
            "rationale": arguments["rationale"].strip(),
            "stop_before_commit": stop_before_commit,
            "sequence": list(sequence),
            "resume_from": int(resume_from),
        },
        "bound_observation_sha256": payload["bound_observation_sha256"],
    }
