"""The declaration header of a composed tool.

A ``tool.star`` file binds ``TOOL = composed_tool(...)`` at top level. The
constructors below are *pure* host functions (see :mod:`.starlark_runtime`):
they run while the module loads, check their own arguments and return tagged
plain data. :func:`parse_declaration` then validates the whole header into a
:class:`Declaration`, which is the only input the compiler and executor read.

Everything model-visible about a composed tool is generated from this header:
the purpose sentence, ``Use when`` and ``Does not`` text come from it, while
``Effect``, ``Returns`` and ``On error`` are derived from its stages, budgets
and refusal codes. Nothing in the body can add text a model reads.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ora.agent.tools.skill_receipt import MAX_RATIONALE_CHARS, MAX_STAGES
from ora.contracts.action import FORBIDDEN_KEY_FRAGMENTS

ARM_PROFILES: dict[str, tuple[str, ...]] = {
    "single_arm": ("right",),
    "dual_arm": ("left", "right"),
}
#: The primitive a declared stage drives; ``check`` is a read-only stage and a
#: ``tool`` stage runs one pinned composed tool (``stage(..., tool="<name>/c<n>")``)
#: as a step of this one, its actions dispatched as that stage's attempts.
STAGE_PRIMITIVES = ("gripper", "move", "tool", "check")

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^c[0-9]+$")
_TAG = "kind"
MAX_SUMMARY_CHARS = 160
MAX_USE_WHEN_CHARS = 600
MAX_CLAUSE_CHARS = 200
MAX_DOES_NOT = 6
MAX_REFUSALS = 8
MAX_PRIMITIVE_ACTIONS = 24
MAX_QUERIES = 32
MAX_WALL_S = 600
MAX_BODY_MEASUREMENTS_PER_STAGE = 8
#: How deep ``tool`` stages may nest: a tool may run a tool that runs a tool, no deeper.
MAX_NESTING_DEPTH = 2
_TOOL_REF = re.compile(r"^([a-z][a-z0-9_]*)/(c[0-9]+)$")
MAX_CHOICES = 12


class DeclarationError(ValueError):
    """A composed tool header is malformed; the message names the field."""


# ---- constructors (pure host functions, callable at load) ----------------------------------------


def _tagged(kind: str, **fields: Any) -> dict[str, Any]:
    return {_TAG: kind, **fields}


def composed_tool(
    *,
    name: str,
    version: str,
    family: str,
    summary: str,
    use_when: str,
    does_not: list,
    inputs: dict,
    stages: list,
    budgets: dict,
    arm_profiles: list,
    parameters: dict | None = None,
    refusals: list | None = None,
    refers_to: list | None = None,
    requires: list | None = None,
    claim_boundary: str | None = None,
    evidence: list | None = None,
) -> dict[str, Any]:
    return _tagged(
        "composed_tool", name=name, version=version, family=family, summary=summary,
        use_when=use_when, does_not=does_not, inputs=inputs, stages=stages, budgets=budgets,
        arm_profiles=arm_profiles, parameters=parameters or {}, refusals=refusals or [],
        refers_to=refers_to or [], requires=requires or [], claim_boundary=claim_boundary,
        evidence=evidence or [],
    )


def stage(name: str, primitive: str, summary: str, tool: str | None = None) -> dict[str, Any]:
    return _tagged("stage", name=name, primitive=primitive, summary=summary, tool=tool)


def arm_input(description: str | None = None) -> dict[str, Any]:
    return _tagged("arm_input", description=description)


def rationale_input() -> dict[str, Any]:
    return _tagged("rationale_input")


def annotation_input(description: str, max_chars: int = 200) -> dict[str, Any]:
    return _tagged("annotation_input", description=description, max_chars=max_chars)


def candidate_input(kind: str, description: str) -> dict[str, Any]:
    return _tagged("candidate_input", candidate_kind=kind, description=description)


def text_input(description: str, max_chars: int = 200, required: bool = True) -> dict[str, Any]:
    """Free text the body reads, for example an object description it grounds."""
    return _tagged("text_input", description=description, max_chars=max_chars, required=required)


def ref_input(description: str, required: bool = True) -> dict[str, Any]:
    """A target_ref the policy measured on the current observation (vision_ground, vision_segment, mark_point)."""
    return _tagged("ref_input", description=description, required=required)


def number_input(description: str, minimum: float, maximum: float, unit: str | None = None,
                 required: bool = True) -> dict[str, Any]:
    return _tagged("number_input", description=description, minimum=minimum, maximum=maximum, unit=unit,
                   required=required)


def choice_input(description: str, options: list, required: bool = True) -> dict[str, Any]:
    return _tagged("choice_input", description=description, options=options, required=required)


def tunable(id: str, expose: bool = False) -> dict[str, Any]:  # noqa: A002 - matches the header field
    return _tagged("tunable", id=id, expose=expose)


def budgets(
    primitive_actions: int, queries: int = 0, wall_s: int = 120, measurements_per_stage: int = 4,
) -> dict[str, Any]:
    return _tagged(
        "budgets", primitive_actions=primitive_actions, queries=queries, wall_s=wall_s,
        measurements_per_stage=measurements_per_stage,
    )


def refusal(code: str, when: str, retryable: bool = True) -> dict[str, Any]:
    return _tagged("refusal", code=code, when=when, retryable=retryable)


DECLARATION_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "composed_tool": composed_tool,
    "stage": stage,
    "arm_input": arm_input,
    "rationale_input": rationale_input,
    "annotation_input": annotation_input,
    "candidate_input": candidate_input,
    "text_input": text_input,
    "ref_input": ref_input,
    "number_input": number_input,
    "choice_input": choice_input,
    "tunable": tunable,
    "budgets": budgets,
    "refusal": refusal,
}


# ---- the validated header ------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    primitive: str
    summary: str
    tool: str | None = None  # the pinned "<name>/c<n>" a tool stage runs

    @property
    def read_only(self) -> bool:
        return self.primitive == "check"


#: Candidate kinds a ``candidate_input`` may name: the planner whose published
#: candidates the id refers to.
CANDIDATE_KINDS = ("grasp",)
MAX_CLAIM_CHARS = 400
MAX_EVIDENCE = 4


#: Inputs whose value the body reads (besides arm, candidate and rationale).
VALUE_INPUTS = ("text_input", "ref_input", "number_input", "choice_input")


@dataclass(frozen=True)
class Input:
    name: str
    kind: str  # arm_input | rationale_input | annotation_input | candidate_input | one of VALUE_INPUTS
    description: str | None = None
    max_chars: int | None = None
    candidate_kind: str | None = None
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Parameter:
    name: str
    tunable: str
    expose: bool


@dataclass(frozen=True)
class Refusal:
    code: str
    when: str
    retryable: bool


@dataclass(frozen=True)
class Budgets:
    primitive_actions: int
    queries: int
    wall_s: int
    measurements_per_stage: int


@dataclass(frozen=True)
class Declaration:
    name: str
    version: str
    family: str
    summary: str
    use_when: str
    does_not: tuple[str, ...]
    inputs: tuple[Input, ...]
    stages: tuple[Stage, ...]
    budgets: Budgets
    arm_profiles: tuple[str, ...]
    parameters: tuple[Parameter, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    refers_to: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    claim_boundary: str | None = None
    evidence: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def stage_primitives(self) -> dict[str, str]:
        return {item.name: item.primitive for item in self.stages}

    @property
    def action_stages(self) -> tuple[Stage, ...]:
        return tuple(item for item in self.stages if not item.read_only)

    def input(self, kind: str) -> Input | None:
        return next((item for item in self.inputs if item.kind == kind), None)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DeclarationError(message)


def _text(value: Any, where: str, limit: int) -> str:
    _expect(isinstance(value, str) and value.strip() != "", f"{where} must be non-empty text")
    text = " ".join(value.split())
    _expect(len(text) <= limit, f"{where} is longer than {limit} characters")
    return text


def _identifier(value: Any, where: str) -> str:
    _expect(isinstance(value, str) and bool(_NAME.match(value)), f"{where} must be lowercase snake_case")
    return value


def _tagged_item(value: Any, kind: str, where: str) -> Mapping[str, Any]:
    _expect(isinstance(value, Mapping) and value.get(_TAG) == kind, f"{where} must be built with {kind}()")
    return value


def _privileged(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS)


def parse_declaration(raw: Any) -> Declaration:
    """Validate the ``TOOL`` value of a body; raise :class:`DeclarationError`."""

    tool = _tagged_item(raw, "composed_tool", "TOOL")
    name = _identifier(tool["name"], "name")
    _expect(isinstance(tool["version"], str) and bool(_VERSION.match(tool["version"])), "version must look like c1")
    family = _identifier(tool["family"], "family")
    summary = _text(tool["summary"], "summary", MAX_SUMMARY_CHARS)
    _expect(summary.endswith(".") and summary.count(". ") == 0, "summary must be one sentence ending in a period")
    use_when = _text(tool["use_when"], "use_when", MAX_USE_WHEN_CHARS)
    _expect(not use_when.lower().startswith("use when"), "use_when must not repeat its own label")

    does_not_raw = tool["does_not"]
    _expect(isinstance(does_not_raw, list) and 1 <= len(does_not_raw) <= MAX_DOES_NOT,
            f"does_not must list 1..{MAX_DOES_NOT} non-guarantees")
    does_not = tuple(_text(item, f"does_not[{index}]", MAX_CLAUSE_CHARS).rstrip(".")
                     for index, item in enumerate(does_not_raw))

    inputs_raw = tool["inputs"]
    _expect(isinstance(inputs_raw, Mapping), "inputs must be a dict of name -> input constructor")
    inputs = []
    for key, value in inputs_raw.items():
        input_name = _identifier(key, f"inputs key {key!r}")
        _expect(not _privileged(input_name), f"input {input_name!r} names privileged state")
        _expect(isinstance(value, Mapping) and value.get(_TAG) in {
            "arm_input", "rationale_input", "annotation_input", "candidate_input", *VALUE_INPUTS,
        }, f"inputs[{input_name!r}] must be built by an input constructor")
        kind = value[_TAG]
        if kind in VALUE_INPUTS:
            _expect(input_name not in ("arm", "rationale", "candidate_id"), f"inputs[{input_name!r}] is a reserved name")
            inputs.append(_value_input(input_name, kind, value))
        elif kind == "arm_input":
            _expect(input_name == "arm", "the arm input must be named arm")
            description = value.get("description")
            if description is not None:
                description = _text(description, "inputs['arm'].description", MAX_CLAUSE_CHARS)
            inputs.append(Input(input_name, kind, description))
        elif kind == "rationale_input":
            _expect(input_name == "rationale", "the rationale input must be named rationale")
            inputs.append(Input(input_name, kind, max_chars=MAX_RATIONALE_CHARS))
        elif kind == "candidate_input":
            _expect(input_name == "candidate_id", "the candidate input must be named candidate_id")
            _expect(value.get("candidate_kind") in CANDIDATE_KINDS,
                    f"inputs['candidate_id'] kind must be one of {list(CANDIDATE_KINDS)}")
            inputs.append(Input(input_name, kind, _text(value.get("description"), "inputs['candidate_id'].description",
                                                         MAX_USE_WHEN_CHARS), candidate_kind=value["candidate_kind"]))
        else:
            max_chars = value.get("max_chars")
            _expect(isinstance(max_chars, int) and not isinstance(max_chars, bool) and 1 <= max_chars <= 200,
                    f"inputs[{input_name!r}].max_chars must be 1..200")
            inputs.append(Input(input_name, kind, _text(value.get("description"), f"inputs[{input_name!r}].description",
                                                         MAX_CLAUSE_CHARS), max_chars))
    kinds = [item.kind for item in inputs]
    _expect(kinds.count("rationale_input") == 1,
            "a composed tool is one committed robot decision and takes exactly one rationale input")

    stages_raw = tool["stages"]
    _expect(isinstance(stages_raw, list) and 1 <= len(stages_raw) <= MAX_STAGES, f"stages must list 1..{MAX_STAGES} stages")
    stages = []
    for index, value in enumerate(stages_raw):
        item = _tagged_item(value, "stage", f"stages[{index}]")
        primitive = item["primitive"]
        _expect(primitive in STAGE_PRIMITIVES, f"stages[{index}].primitive must be one of {list(STAGE_PRIMITIVES)}")
        ref = item.get("tool")
        _expect((primitive == "tool") == (ref is not None),
                f"stages[{index}]: a tool stage, and only a tool stage, names tool = \"<name>/c<n>\"")
        _expect(ref is None or (isinstance(ref, str) and bool(_TOOL_REF.match(ref))),
                f"stages[{index}].tool must look like <name>/c<n>")
        stages.append(Stage(_identifier(item["name"], f"stages[{index}].name"), primitive,
                            _text(item["summary"], f"stages[{index}].summary", MAX_CLAUSE_CHARS).rstrip("."), ref))
    names = [item.name for item in stages]
    _expect(len(set(names)) == len(names), "stage names must be unique")
    _expect(any(not item.read_only for item in stages), "a composed tool needs at least one action stage")

    raw_budgets = _tagged_item(tool["budgets"], "budgets", "budgets")
    budget = Budgets(
        primitive_actions=_bounded_int(raw_budgets["primitive_actions"], "budgets.primitive_actions", 1, MAX_PRIMITIVE_ACTIONS),
        queries=_bounded_int(raw_budgets["queries"], "budgets.queries", 0, MAX_QUERIES),
        wall_s=_bounded_int(raw_budgets["wall_s"], "budgets.wall_s", 1, MAX_WALL_S),
        measurements_per_stage=_bounded_int(raw_budgets["measurements_per_stage"], "budgets.measurements_per_stage",
                                            0, MAX_BODY_MEASUREMENTS_PER_STAGE),
    )
    _expect(budget.primitive_actions >= len([item for item in stages if not item.read_only]),
            "budgets.primitive_actions is smaller than the number of action stages")

    profiles_raw = tool["arm_profiles"]
    _expect(isinstance(profiles_raw, list) and profiles_raw and len(set(profiles_raw)) == len(profiles_raw)
            and all(item in ARM_PROFILES for item in profiles_raw),
            f"arm_profiles must list distinct members of {sorted(ARM_PROFILES)}")
    _expect("dual_arm" not in profiles_raw or "arm_input" in kinds or "candidate_input" in kinds,
            "a dual_arm tool needs arm_input() or candidate_input(): the policy or its candidate chooses the arm")

    parameters = []
    params_raw = tool["parameters"]
    _expect(isinstance(params_raw, Mapping), "parameters must be a dict of name -> tunable()")
    for key, value in params_raw.items():
        param_name = _identifier(key, f"parameters key {key!r}")
        _expect(not _privileged(param_name), f"parameter {param_name!r} names privileged state")
        _expect(param_name not in inputs_raw, f"parameter {param_name!r} shadows an input")
        item = _tagged_item(value, "tunable", f"parameters[{param_name!r}]")
        _expect(isinstance(item["id"], str) and item["id"] != "", f"parameters[{param_name!r}].id must name a tunable")
        _expect(isinstance(item["expose"], bool), f"parameters[{param_name!r}].expose must be a bool")
        parameters.append(Parameter(param_name, item["id"], item["expose"]))

    refusals_raw = tool["refusals"]
    _expect(isinstance(refusals_raw, list) and len(refusals_raw) <= MAX_REFUSALS, f"refusals must list at most {MAX_REFUSALS}")
    refusals = []
    for index, value in enumerate(refusals_raw):
        item = _tagged_item(value, "refusal", f"refusals[{index}]")
        _expect(isinstance(item["retryable"], bool), f"refusals[{index}].retryable must be a bool")
        refusals.append(Refusal(_identifier(item["code"], f"refusals[{index}].code"),
                                _text(item["when"], f"refusals[{index}].when", MAX_CLAUSE_CHARS).rstrip("."),
                                item["retryable"]))
    codes = [item.code for item in refusals]
    _expect(len(set(codes)) == len(codes), "refusal codes must be unique")

    claim = tool["claim_boundary"]
    if claim is not None:
        claim = _text(claim, "claim_boundary", MAX_CLAIM_CHARS)
    evidence_raw = _list(tool["evidence"], "evidence")
    _expect(len(evidence_raw) <= MAX_EVIDENCE, f"evidence names at most {MAX_EVIDENCE} items")
    evidence = tuple(_identifier(item, "evidence item") for item in evidence_raw)
    _expect(len(set(evidence)) == len(evidence), "evidence names must be unique")
    _expect(not set(evidence) & {"success", "status", "tool", "version", "arm", "committed_sequence", "cost",
                                 "claim_boundary", "intent", "candidate_id"},
            "evidence names must not shadow result fields")

    return Declaration(
        name=name, version=tool["version"], family=family, summary=summary, use_when=use_when,
        does_not=does_not, inputs=tuple(inputs), stages=tuple(stages), budgets=budget,
        arm_profiles=tuple(profiles_raw), parameters=tuple(parameters), refusals=tuple(refusals),
        refers_to=tuple(_identifier(item, "refers_to item") for item in _list(tool["refers_to"], "refers_to")),
        requires=tuple(_identifier(item, "requires item") for item in _list(tool["requires"], "requires")),
        claim_boundary=claim, evidence=evidence,
        raw=dict(tool),
    )


def _value_input(name: str, kind: str, value: Mapping[str, Any]) -> Input:
    """A text, ref, number or choice input: a value the body reads and the policy supplies."""

    where = f"inputs[{name!r}]"
    description = _text(value.get("description"), f"{where}.description", MAX_CLAUSE_CHARS)
    required = value.get("required")
    _expect(isinstance(required, bool), f"{where}.required must be a bool")
    if kind == "text_input":
        max_chars = value.get("max_chars")
        _expect(isinstance(max_chars, int) and not isinstance(max_chars, bool) and 1 <= max_chars <= 400,
                f"{where}.max_chars must be 1..400")
        return Input(name, kind, description, max_chars=max_chars, required=required)
    if kind == "ref_input":
        return Input(name, kind, description, max_chars=200, required=required)
    if kind == "number_input":
        low, high = value.get("minimum"), value.get("maximum")
        _expect(all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (low, high))
                and low < high, f"{where} needs numeric minimum < maximum")
        unit = value.get("unit")
        _expect(unit is None or (isinstance(unit, str) and 0 < len(unit) <= 16), f"{where}.unit must be short text")
        return Input(name, kind, description, required=required, minimum=float(low), maximum=float(high), unit=unit)
    options = value.get("options")
    _expect(isinstance(options, list) and 2 <= len(options) <= MAX_CHOICES and len(set(options)) == len(options)
            and all(isinstance(item, str) and _NAME.match(item) for item in options),
            f"{where}.options must list 2..{MAX_CHOICES} distinct snake_case words")
    return Input(name, kind, description, required=required, options=tuple(options))


def _bounded_int(value: Any, where: str, low: int, high: int) -> int:
    _expect(isinstance(value, int) and not isinstance(value, bool) and low <= value <= high,
            f"{where} must be an integer in [{low}, {high}]")
    return value


def _list(value: Any, where: str) -> Sequence[Any]:
    _expect(isinstance(value, list), f"{where} must be a list")
    return value
