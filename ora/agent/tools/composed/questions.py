"""The shared question bank: the only questions a composed tool may ask a model.

A question lives at ``ora/policy/composed_tools/questions/<id>/<version>.yaml``
and is referenced as ``<id>@<version>``. It fixes the prompt template (which
may name only its declared image roles and value slots), the answer structure
(a closed enum plus a bounded reason), what the answer cannot establish, and
its evaluation: a labelled set from sealed runs and the accuracy measured on
it. A question is **admitted** -- usable from ``sdk.ask`` -- only once that
accuracy reaches its registered threshold. Questions are edited offline and
are immutable once versioned, like the tools that use them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "ora.composed-question.v1"
QUESTIONS_ROOT = Path(__file__).resolve().parents[3] / "policy" / "composed_tools" / "questions"
_REF = re.compile(r"^([a-z][a-z0-9_]*)@(v[0-9]+)$")
_SLOT = re.compile(r"\{([a-z][a-z0-9_]*)\}")
MAX_REASON_CHARS = 200


class QuestionError(ValueError):
    """A question file, reference or answer is invalid."""


@dataclass(frozen=True)
class Question:
    id: str
    version: str
    prompt: str
    images: Mapping[str, int]
    values: Mapping[str, Mapping[str, Any]]
    answers: tuple[str, ...]
    reason_max_chars: int
    claim_boundary: str
    eval_set: str | None
    accuracy: float | None
    threshold: float

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def admitted(self) -> bool:
        return self.eval_set is not None and self.accuracy is not None and self.accuracy >= self.threshold

    def render(self, values: Mapping[str, Any], images: list[str]) -> str:
        """The prompt with its slots filled; images are named by role."""

        unknown = set(values) - set(self.values)
        if unknown:
            raise QuestionError(f"{self.ref}: undeclared value slots {sorted(unknown)}")
        for name, spec in self.values.items():
            if name not in values:
                raise QuestionError(f"{self.ref}: value {name!r} is required")
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise QuestionError(f"{self.ref}: value {name!r} must be a number")
            if not spec["minimum"] <= value <= spec["maximum"]:
                raise QuestionError(f"{self.ref}: value {name!r} is outside [{spec['minimum']}, {spec['maximum']}]")
        counts: dict[str, int] = {}
        for role in images:
            if role not in self.images:
                raise QuestionError(f"{self.ref}: image role {role!r} is not declared")
            counts[role] = counts.get(role, 0) + 1
            if counts[role] > self.images[role]:
                raise QuestionError(f"{self.ref}: more than {self.images[role]} {role!r} images")
        filled = {**{name: f"{float(value):.4g}" for name, value in values.items()},
                  **{role: f"[image: {role}]" for role in self.images}}
        return _SLOT.sub(lambda match: filled[match.group(1)], self.prompt)

    def answer(self, raw: Any) -> dict[str, Any]:
        """A model reply checked against the answer structure; anything else is unavailable."""

        if (isinstance(raw, Mapping) and raw.get("answer") in self.answers
                and isinstance(raw.get("reason"), str) and len(raw["reason"]) <= self.reason_max_chars):
            return {"status": "answered", "question": self.ref, "answer": raw["answer"], "reason": raw["reason"]}
        return {"status": "unavailable", "question": self.ref, "answer": None, "reason": "answer_outside_structure"}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise QuestionError(message)


def parse_question(document: Any, *, path: str) -> Question:
    _expect(isinstance(document, Mapping) and document.get("schema") == SCHEMA, f"{path}: schema must be {SCHEMA}")
    expected = {"schema", "id", "version", "prompt", "images", "values", "answer", "claim_boundary", "eval"}
    _expect(set(document) == expected, f"{path}: fields must be exactly {sorted(expected)}")
    ref = f"{document['id']}@{document['version']}"
    _expect(bool(_REF.match(ref)), f"{path}: id and version must form <id>@v<n>")
    prompt = document["prompt"]
    _expect(isinstance(prompt, str) and prompt.strip() != "", f"{path}: prompt must be text")
    images = document["images"]
    _expect(isinstance(images, Mapping) and all(isinstance(v, int) and v >= 1 for v in images.values()),
            f"{path}: images maps roles to a positive maximum count")
    values = document["values"]
    _expect(isinstance(values, Mapping), f"{path}: values maps slots to {{unit, minimum, maximum}}")
    for name, spec in values.items():
        _expect(isinstance(spec, Mapping) and set(spec) == {"unit", "minimum", "maximum"}
                and spec["minimum"] < spec["maximum"], f"{path}: value {name!r} needs unit, minimum < maximum")
    slots = set(_SLOT.findall(prompt))
    _expect(slots <= set(images) | set(values), f"{path}: prompt names undeclared slots {sorted(slots - set(images) - set(values))}")
    answer = document["answer"]
    _expect(isinstance(answer, Mapping) and answer.get("type") == "enum" and isinstance(answer.get("values"), list)
            and len(answer["values"]) >= 2 and isinstance(answer.get("reason_max_chars"), int)
            and 1 <= answer["reason_max_chars"] <= MAX_REASON_CHARS,
            f"{path}: answer is an enum of at least two values with reason_max_chars in [1, {MAX_REASON_CHARS}]")
    claim = document["claim_boundary"]
    _expect(isinstance(claim, str) and claim.strip() != "", f"{path}: claim_boundary must be text")
    evaluation = document["eval"]
    _expect(isinstance(evaluation, Mapping) and set(evaluation) == {"set", "accuracy", "threshold"},
            f"{path}: eval is {{set, accuracy, threshold}}")
    _expect(isinstance(evaluation["threshold"], (int, float)) and 0 < evaluation["threshold"] <= 1,
            f"{path}: eval.threshold must be in (0, 1]")
    _expect(evaluation["accuracy"] is None or (evaluation["set"] is not None and 0 <= evaluation["accuracy"] <= 1),
            f"{path}: an accuracy needs its evaluation set")
    return Question(
        id=document["id"], version=document["version"], prompt=prompt, images=dict(images),
        values={name: dict(spec) for name, spec in values.items()}, answers=tuple(answer["values"]),
        reason_max_chars=answer["reason_max_chars"], claim_boundary=claim, eval_set=evaluation["set"],
        accuracy=evaluation["accuracy"], threshold=float(evaluation["threshold"]),
    )


def load_question(ref: str, root: Path | None = None) -> Question:
    match = _REF.match(ref) if isinstance(ref, str) else None
    if match is None:
        raise QuestionError(f"question reference {ref!r} must be <id>@v<n>")
    path = Path(root or QUESTIONS_ROOT) / match.group(1) / f"{match.group(2)}.yaml"
    if not path.is_file():
        raise QuestionError(f"question {ref} does not exist")
    question = parse_question(yaml.safe_load(path.read_text(encoding="utf-8")), path=str(path))
    if question.ref != ref:
        raise QuestionError(f"{path} declares {question.ref}, not {ref}")
    return question


def load_questions(root: Path | None = None) -> dict[str, Question]:
    base = Path(root or QUESTIONS_ROOT)
    questions = [load_question(f"{path.parent.name}@{path.stem}", base) for path in sorted(base.glob("*/v*.yaml"))]
    return {item.ref: item for item in questions}
