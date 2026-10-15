"""The v2 finish contract: success or give_up with a reason; abort on recorded infrastructure.

Two capabilities,
declared only by the v2 surface generations:

- ``policy_finish_outcomes_v2`` (F1): ``finish_episode`` accepts exactly
  ``success`` and ``give_up``. Every outcome needs a reason that is not empty,
  a token or a placeholder; a success reason cites observable evidence and
  names its source; ``give_up`` names a ``category``. The released ``failure``
  outcome and its ``attempts`` are retired. A give_up ends the round as a valid
  behavioural failure sample and the round's official verdict is kept.
- ``policy_abort_infrastructure_round_v2`` (F2, requires F1): ``abort`` is
  added as the third outcome and is admitted only when this round's Harness
  record holds an infrastructure failure -- the class table of
  ``abort_evidence_schema`` -- on RoboTwin and LIBERO as on RoboDojo. An
  admitted abort ends only the current round and marks its sample invalid
  (``orchestrator/policy_finish.py``).

They replaced the released RoboDojo pair (a withheld failure exit and an
evidence-gated abort) on the v2 generation.

The text below is the policy-visible contract; the numbers in it are generated
from the constants the runtime half (``agent/tools/finish_outcomes.py``)
enforces. A help sentence elsewhere that still names the retired failure exit
is rewritten here (``RETIRED_EXIT_CLAUSES``) or refused, so no v2 surface
offers an exit the runtime would refuse.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

from .abort_evidence_schema import infrastructure_classes_text

CAPABILITY = "policy_finish_outcomes_v2"
ABORT_CAPABILITY = "policy_abort_infrastructure_round_v2"
REQUIRES = "faithful_check_task"
OUTCOMES = ("success", "give_up")
ABORT = "abort"
#: give_up categories, in published order. Infrastructure failures are not a
#: category: they are abort.
CATEGORIES = {
    "unreachable": "a pose the task needs cannot be reached by the arm",
    "object_lost": "a task object left the workspace or is no longer visible in any camera",
    "scene_damaged": "the scene changed so that the task can no longer be completed",
    "repeated_failure": "the same step failed repeatedly from the current state",
    "other": "none of the above; the reason names the obstacle",
}
MIN_REASON_CHARACTERS = 24
MIN_REASON_WORDS = 4
#: Sources a success reason may name, as published in a refusal. Matching is
#: loose: by these stems in the reason with camelCase split, case folded and
#: underscores and hyphens read as spaces ("check_task", "checkTask", "task
#: check" all name the checker), and by their Chinese equivalents.
EVIDENCE_TERMS = ("check_task", "observation", "camera", "image", "depth", "measured", "visible",
                  "verified", "receipt", "sensor")
EVIDENCE_STEMS = (
    "check task", "task check", "checker", "observ", "camera", "image", "depth", "measur", "visib",
    "seen", "shows", "verif", "receipt", "sensor",
    "检查", "相机", "摄像", "图像", "画面", "观测", "观察", "深度", "测量", "可见", "看到", "显示",
)
REASON_REFUSAL_CODE = "finish_reason_insufficient"
CATEGORY_REFUSAL_CODE = "finish_category_required"
ABORT_REFUSAL_CODE = "abort_without_infrastructure_evidence"
ABORT_ADMISSION = "infrastructure_evidence"
#: Released help clauses that name the retired failure exit, and their v2 text.
RETIRED_EXIT_CLAUSES = {
    "stop and finish with failure or abort if both planned recovery routes are exhausted or unsafe.":
        "if both planned recovery routes are exhausted or unsafe, finish_episode names the accepted outcomes.",
    "otherwise finish with failure honestly.":
        "otherwise finish_episode names the accepted outcomes.",
}
RETIRED_EXIT_MARKERS = ("finish with failure", "exhausted attempts")


def _categories_text() -> str:
    return "; ".join(f"{name} ({meaning})" for name, meaning in CATEGORIES.items())


def _refused_text(*, abort: bool) -> str:
    """The ex-ante refusals of finish_episode, as rules (published-limits N2).

    The codes are the ones the runtime raises: ``agent/tools/runtime/lifecycle.py``
    (pending fresh observation), ``finish_outcomes.py`` (arguments, reason,
    category, abort) and ``finish_task_check.py`` (latched and unconfirmed
    success). This binder writes the whole description after every layer, so
    the sentence lives here rather than in a published-limits layer.
    """

    outcomes = "success, give_up and abort" if abort else "success and give_up"
    abort_rule = (
        "; abort while this round's Harness record holds no infrastructure failure "
        f"({ABORT_REFUSAL_CODE})" if abort else ""
    )
    return (
        "Refused: an action still waiting for its fresh observation (action_pending_fresh_observation); "
        f"an outcome other than {outcomes}, or an unknown field (invalid_arguments); a reason that breaks "
        f"the Reason rule below ({REASON_REFUSAL_CODE}, naming reason); give_up without a category "
        f"({CATEGORY_REFUSAL_CODE}) or a category with another outcome (invalid_arguments); success without "
        "check_task true at the current boundary (task_success_not_confirmed); any outcome but success "
        f"after check_task confirmed success (task_success_latched){abort_rule}."
    )


def description(*, abort: bool) -> str:
    """The finish_episode help for F1 alone or F1 with F2."""

    outcomes = "success, give_up or abort" if abort else "success or give_up"
    use_abort = (
        ", or this round's Harness record holds an infrastructure failure (abort): a tool, sensor, bridge "
        "or model-service error the Harness classifies as infrastructure, or a controller execution fault"
        if abort else ""
    )
    effect_abort = (
        " An admitted abort ends only this round, records no official verdict for it and marks the round "
        "invalid as a behavioural sample. Planner and IK refusals, stalls, unreached targets, refused or "
        "stale inputs and check_task false are not infrastructure failures."
        if abort else ""
    )
    does_not = [
        "move the physical world", "create official task success",
        "accept success without same-boundary check_task=true",
        "accept an empty, placeholder or token reason",
        *(("accept abort without a recorded infrastructure failure",) if abort else ()),
    ]
    reason_abort = "; an abort reason names the recorded infrastructure failure" if abort else ""
    text = (
        f"End this round's policy lifecycle with {outcomes}; every outcome needs a reason. Use when: "
        "check_task has returned true at the same valid state boundary (success), or the policy concludes "
        f"that this round's task cannot be completed (give_up, with a category){use_abort}. A round "
        "otherwise ends at official success, an official ending condition or when an allowance in "
        "remaining_budget is spent. Effect: lifecycle mutation; persists the outcome, reason and category, "
        "ends this round and prevents further robot proposals in it. The round's official verdict is still "
        f"recorded, and give_up is recorded as a behavioural failure sample.{effect_abort} Returns: the "
        "accepted outcome after lifecycle finalization is admitted; it is not a new official task verdict.\n"
        f"Does not: {', '.join(does_not[:-1])}, or {does_not[-1]}. On error: {_refused_text(abort=abort)} "
        "Resolve pending action and fresh-observation boundaries first, call check_task before retrying "
        "success, and correct the field the refusal names.\n"
        f"Reason rule: at least {MIN_REASON_CHARACTERS} characters and {MIN_REASON_WORDS} words (a CJK "
        "character counts as two characters and half a word), not a placeholder; a success reason cites the observable evidence and names its source (check_task, an "
        "observation, a camera image, depth, or a visible or measured value); a give_up reason states what "
        f"was observed and attempted{reason_abort}. give_up categories: {_categories_text()}."
    )
    return f"{text}\n{infrastructure_classes_text()}" if abort else text


def input_schema(*, abort: bool) -> dict[str, Any]:
    """The finish_episode input schema for F1 alone or F1 with F2."""

    outcomes = [*OUTCOMES, *((ABORT,) if abort else ())]
    return {
        "type": "object", "additionalProperties": False, "required": ["outcome", "reason"],
        "properties": {
            "outcome": {
                "type": "string", "enum": outcomes,
                "description": (
                    "success requires same-boundary check_task=true; give_up requires category"
                    + ("; abort requires an infrastructure failure the Harness recorded in this round."
                       if abort else ".")
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    f"Required for every outcome: at least {MIN_REASON_CHARACTERS} characters and "
                    f"{MIN_REASON_WORDS} words, not a placeholder. success: the observable evidence and its "
                    "source; give_up: what was observed and attempted"
                    + ("; abort: the recorded infrastructure failure." if abort else ".")
                ),
            },
            "category": {
                "type": "string", "enum": list(CATEGORIES),
                "description": "Required with give_up and refused with any other outcome; the categories "
                               "are defined in this tool's description.",
            },
        },
    }


def result_schema(*, abort: bool) -> dict[str, Any]:
    outcomes = [*OUTCOMES, *((ABORT,) if abort else ())]
    return {
        "type": "object", "additionalProperties": False, "required": ["success", "outcome"],
        "properties": {
            "success": {"type": "boolean", "const": True},
            "outcome": {"type": "string", "enum": outcomes},
        },
    }


def _rewrite(definitions: list[dict[str, Any]], *, abort: bool) -> None:
    from .tool_surface import ToolSurfaceError

    entries = [item for item in definitions if item.get("name") == "finish_episode"]
    if len(entries) != 1:
        raise ToolSurfaceError(f"{CAPABILITY} requires one declared finish_episode")
    entries[0]["description"] = description(abort=abort)
    entries[0]["inputSchema"] = input_schema(abort=abort)


def _descriptor(capability: str) -> dict[str, str]:
    path = Path(__file__)
    return {
        "slot": f"derived:{capability}",
        "ref": "ora/agent/composition/finish_outcomes_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def bind_finish_outcomes(definitions: list[dict[str, Any]], results: dict[str, Any],
                         capabilities: Sequence[str]) -> dict[str, str]:
    """Declare success and give_up with a reason; retire the failure exit everywhere."""
    from .tool_surface import ToolSurfaceError

    if REQUIRES not in capabilities:
        raise ToolSurfaceError(f"{CAPABILITY} requires {REQUIRES}")
    _rewrite(definitions, abort=False)
    results["finish_episode"] = result_schema(abort=False)
    for item in definitions:
        text = str(item.get("description", ""))
        for old, new in RETIRED_EXIT_CLAUSES.items():
            text = text.replace(old, new)
        item["description"] = text
        if item.get("name") != "finish_episode" and any(
                marker in text for marker in RETIRED_EXIT_MARKERS):
            raise ToolSurfaceError(
                f"{CAPABILITY}: {item.get('name')} still names the retired failure exit")
    return _descriptor(CAPABILITY)


def bind_abort_round(definitions: list[dict[str, Any]], results: dict[str, Any],
                     capabilities: Sequence[str]) -> dict[str, str]:
    """Add the evidence-gated, round-scoped abort to the v2 contract."""
    from .tool_surface import ToolSurfaceError

    if CAPABILITY not in capabilities:
        raise ToolSurfaceError(f"{ABORT_CAPABILITY} requires {CAPABILITY}")
    _rewrite(definitions, abort=True)
    results["finish_episode"] = result_schema(abort=True)
    return _descriptor(ABORT_CAPABILITY)


def accepted_outcomes(capabilities: Sequence[str]) -> tuple[str, ...]:
    return (*OUTCOMES, *((ABORT,) if ABORT_CAPABILITY in capabilities else ()))


__all__ = [
    "ABORT", "ABORT_ADMISSION", "ABORT_CAPABILITY", "ABORT_REFUSAL_CODE", "CAPABILITY", "CATEGORIES",
    "CATEGORY_REFUSAL_CODE", "EVIDENCE_STEMS", "EVIDENCE_TERMS", "MIN_REASON_CHARACTERS",
    "MIN_REASON_WORDS", "OUTCOMES", "REASON_REFUSAL_CODE", "REQUIRES", "RETIRED_EXIT_CLAUSES",
    "RETIRED_EXIT_MARKERS", "accepted_outcomes", "bind_abort_round", "bind_finish_outcomes",
    "description", "input_schema", "result_schema",
]
