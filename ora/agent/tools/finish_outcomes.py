"""Runtime half of the v2 finish contract (``composition/finish_outcomes_schema.py``).

``admit`` runs before any lifecycle state is written (called by the
dispatcher for every ``finish_episode``) and refuses, in order:
an argument or outcome the contract does not declare; a give_up without a
category from the enum, or a category on another outcome; an empty, token or
placeholder reason, or a success reason that names no observable evidence
source; and an abort the round's Harness record does not support. Reason and
category refusals are retryable and name their field. The abort refusal is
retryable too, without the changed-arguments flag: the record is re-read on
every call, so the same abort is admitted once the Harness records an
infrastructure failure in the round (EXP11B census E2). It lists the classes
the gate counts (``counted``) and the signals it read. ``record_fields`` adds
the contract, the abort evidence and the give_up candidate ledger to the
persisted ``policy-finish.json``; the orchestrator reads that record to carry
the outcome onto the round (``orchestrator/policy_finish.py``).
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from ora.agent.composition.abort_evidence_schema import ABORT_REQUIRES, counted_classes
from ora.agent.composition.finish_outcomes_schema import (
    ABORT, ABORT_ADMISSION, ABORT_CAPABILITY, ABORT_REFUSAL_CODE, CAPABILITY, CATEGORIES,
    CATEGORY_REFUSAL_CODE, EVIDENCE_STEMS, EVIDENCE_TERMS, MIN_REASON_CHARACTERS, MIN_REASON_WORDS,
    REASON_REFUSAL_CODE,
)
from ora.agent.composition.finish_outcomes_schema import accepted_outcomes as _accepted

from . import abort_evidence, finish_candidates
from .robot import ToolError

ARGUMENTS = ("outcome", "reason", "category")
#: Words a reason may not consist of alone: placeholders, the outcome and
#: category names, and filler.
TOKEN_WORDS = frozenset({
    "n", "a", "an", "na", "none", "null", "nil", "nothing", "todo", "tbd", "reason", "reasons",
    "placeholder", "test", "testing", "ok", "okay", "yes", "done", "finish", "finished", "complete",
    "completed", "success", "successful", "succeeded", "failure", "failed", "fail", "give", "up", "gave",
    "giving", "abort", "aborted", "aborting", "stop", "stopped", "unknown", "no", "not", "applicable",
    "see", "above", "same", "as", "the", "task", "is", "was", "episode", "round", "outcome", "x", "xx",
    "xxx", "category", "end", "ended", "here", "this", "it", "its",
    "give_up", *CATEGORIES,
})
_WORD = re.compile(r"\w+")
#: CJK ideographs, kana and Hangul: one character carries about a syllable or a
#: morpheme, so it counts as two characters and half a word.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def declared(surface: Any) -> bool:
    return surface.supports(CAPABILITY)


def accepted_outcomes(surface: Any) -> tuple[str, ...] | None:
    """The outcomes a v2 surface accepts, or ``None`` on a released surface."""

    return _accepted(surface.capabilities) if declared(surface) else None


def reason_problem(outcome: str, reason: Any) -> str | None:
    """Why a reason is not admissible, or ``None``."""

    if not isinstance(reason, str):
        return "missing" if reason is None else "not_text"
    text = reason.strip()
    if not text:
        return "missing"
    cjk = _CJK.findall(text)
    words = _WORD.findall(_CJK.sub(" ", text.lower()))
    if len(text) + len(cjk) < MIN_REASON_CHARACTERS:
        return "too_short"
    if len(words) + (len(cjk) + 1) // 2 < MIN_REASON_WORDS or len(set(words) | set(cjk)) < 3:
        return "too_few_words"
    if not cjk and all(word in TOKEN_WORDS for word in words):
        return "placeholder"
    if outcome == "success" and not names_evidence(text):
        return "no_evidence_source"
    return None


def names_evidence(text: str) -> bool:
    """Whether a reason names an observable evidence source (loose stem match)."""

    spaced = re.sub(r"[\s_\-]+", " ", _CAMEL.sub(" ", text).lower())
    return any(stem in spaced for stem in EVIDENCE_STEMS)


def _refuse_reason(outcome: str, reason: Any, problem: str) -> ToolError:
    constraints: dict[str, Any] = {
        "problem": problem, "outcome": outcome,
        "minimum_characters": MIN_REASON_CHARACTERS, "minimum_words": MIN_REASON_WORDS,
    }
    rule = (f"every finish_episode outcome needs a reason of at least {MIN_REASON_CHARACTERS} characters "
            f"and {MIN_REASON_WORDS} words that is not a placeholder")
    if outcome == "success":
        constraints["evidence_terms"] = list(EVIDENCE_TERMS)
        rule += "; a success reason names its observable evidence source"
    received = reason[:200] if isinstance(reason, str) else reason
    return ToolError(f"finish_episode reason refused ({problem}): {rule}", code=REASON_REFUSAL_CODE,
                     retryable=True, field="reason", received=received, constraints=constraints)


def _check_arguments(arguments: Any, allowed: tuple[str, ...]) -> str:
    """Every malformed argument is the policy's refusal, never an internal error.

    An exception here would be logged as ``internal_tool_error``, which the
    abort gate counts as infrastructure evidence: the policy could then
    manufacture its own admissible abort. Every check is type-safe.
    """
    if not isinstance(arguments, Mapping):
        raise ToolError("finish_episode arguments must be an object", code="invalid_arguments",
                        retryable=True, constraints={"accepted": list(ARGUMENTS)})
    unknown = sorted(str(name) for name in set(arguments) - set(ARGUMENTS))
    if unknown:
        raise ToolError(f"finish_episode does not accept {', '.join(unknown)}", code="invalid_arguments",
                        retryable=True, field=unknown[0], constraints={"accepted": list(ARGUMENTS)})
    outcome = arguments.get("outcome")
    if not isinstance(outcome, str) or outcome not in allowed:
        raise ToolError(f"finish_episode accepts outcome {', '.join(allowed)}", code="invalid_arguments",
                        retryable=True, field="outcome", received=outcome, constraints={"allowed": list(allowed)})
    return str(outcome)


def _check_category(outcome: str, category: Any) -> None:
    if outcome == "give_up" and not (isinstance(category, str) and category in CATEGORIES):
        raise ToolError("give_up requires a category", code=CATEGORY_REFUSAL_CODE, retryable=True,
                        field="category", received=category, constraints={"allowed": list(CATEGORIES)})
    if outcome != "give_up" and category is not None:
        raise ToolError("category belongs to give_up only", code="invalid_arguments", retryable=True,
                        field="category", received=category, constraints={"outcome": outcome})


def admit(host: Any, arguments: Mapping[str, Any]) -> None:
    """Refuse a finish the v2 contract does not admit; return otherwise."""

    allowed = accepted_outcomes(host.surface)
    if allowed is None:
        return
    outcome = _check_arguments(arguments, allowed)
    _check_category(outcome, arguments.get("category"))
    problem = reason_problem(outcome, arguments.get("reason"))
    if problem is not None:
        raise _refuse_reason(outcome, arguments.get("reason"), problem)
    if outcome == ABORT:
        _admit_abort(host, allowed)


def _admit_abort(host: Any, allowed: tuple[str, ...]) -> None:
    """Refuse an abort this round's Harness record does not support.

    Retryable, with no changed-arguments claim: the record is re-read on every
    call and the unchanged abort is admitted once it holds a counted signal.
    """

    signals = abort_evidence.recorded_signals(host.event_path)
    if signals:
        return
    raise ToolError(
        "abort is admitted only after the Harness records an infrastructure failure in this round, and none "
        "is recorded so far; the record is re-read on every call, so the same abort is admitted once one is "
        "recorded in this round",
        code=ABORT_REFUSAL_CODE, retryable=True, field="outcome", received=ABORT,
        constraints={"allowed": list(allowed), "abort_requires": ABORT_REQUIRES,
                     "counted": counted_classes(), "recorded_infrastructure_signals": signals,
                     "remaining_budget": host._runtime_remaining_budget()},
    )


def record_fields(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """What the persisted finish record adds on a v2 surface."""

    if not declared(host.surface):
        return {}
    outcome = arguments.get("outcome")
    fields: dict[str, Any] = {"finish_contract": CAPABILITY, "category": arguments.get("category")}
    if outcome == ABORT and host.surface.supports(ABORT_CAPABILITY):
        fields["abort_admission"] = ABORT_ADMISSION
        fields["abort_infrastructure_evidence"] = abort_evidence.recorded_signals(host.event_path)
    if outcome == "give_up":
        fields["untried_candidates"] = finish_candidates.for_host(host, str(arguments.get("reason", "")))
    return fields


__all__ = ["ARGUMENTS", "TOKEN_WORDS", "accepted_outcomes", "admit", "declared", "names_evidence",
           "reason_problem", "record_fields"]
