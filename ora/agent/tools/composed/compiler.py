"""Compile a composed tool's declaration into its model-visible contract.

Per arm profile the compiler emits one document holding the tool definition
(``policy_tool_help_v1`` description and closed input schema), the public
result schema of the immediate submission, and the stage, budget and refusal
tables the executor enforces. Help text is generated, not written: stage
order and counts, the dispatch budget, the returned fields and the refusal
codes all come from the same header the executor runs, so the description
cannot drift from the behavior.

The compiled document is committed next to ``tool.star`` and its
``compiled_sha256`` is the identity a ToolSurface or a sealed run binds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ora.agent.composition.schema_validation import lint_input_schema
from ora.agent.composition.tool_surface import canonical_sha256
from ora.agent.tools.skill_receipt import MAX_RATIONALE_CHARS

from .declaration import ARM_PROFILES, Declaration, DeclarationError
from .tunables import TunableTable

COMPILED_SCHEMA = "ora.composed-tool-compiled.v1"
HELP_LABELS = ("Use when:", "Effect:", "Returns:", "Does not:", "On error:")
CLAIM_BOUNDARY = (
    "submitted is not executed: the staged receipt on the next turn reports what each stage "
    "did, and no stage outcome certifies task success"
)
#: Refusals every composed tool inherits from the committed-sequence lifecycle,
#: in the order its help lists them.
LIFECYCLE_REFUSALS: tuple[tuple[str, str], ...] = (
    ("skill_sequence_active", "another committed sequence is still running"),
    ("skill_sequence_paused", "a placement is waiting at its pre-release stop"),
    ("episode_terminal", "the episode has reached its terminal boundary"),
)
_ARM_DESCRIPTION = (
    "The arm that runs the whole sequence. The other arm is not commanded and holds its current targets."
)
_RATIONALE_DESCRIPTION = (
    f"One short sentence saying why you chose this action now, at most {MAX_RATIONALE_CHARS} characters. "
    "It is echoed into the receipt and is the only record of the reason: the trajectory keeps the call "
    "and its result, never the reasoning behind them."
)


class CompileError(ValueError):
    """The declaration cannot be compiled against the tunable table."""


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _input_schema(declaration: Declaration, arms: Sequence[str], tunables: TunableTable) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in declaration.inputs:
        if item.kind == "arm_input":
            if len(arms) > 1:
                properties["arm"] = {
                    "type": "string", "enum": list(arms),
                    "description": item.description or _ARM_DESCRIPTION,
                }
                required.append("arm")
        elif item.kind == "candidate_input":
            properties["candidate_id"] = {
                "type": "string", "minLength": 1, "description": item.description,
            }
            required.append("candidate_id")
        elif item.kind == "rationale_input":
            properties["rationale"] = {
                "type": "string", "minLength": 1, "maxLength": MAX_RATIONALE_CHARS,
                "description": _RATIONALE_DESCRIPTION,
            }
            required.append("rationale")
        elif item.kind == "annotation_input":
            properties[item.name] = {
                "type": ["string", "null"], "minLength": 1, "maxLength": item.max_chars,
                "description": f"{item.description} Optional; echoed as an agent annotation, never resolved or measured.",
            }
        else:
            properties[item.name] = _value_property(item)
            if item.required:
                required.append(item.name)
    for parameter in declaration.parameters:
        if not parameter.expose:
            continue
        entry = tunables[parameter.tunable]
        unit = f" Unit: {entry.unit}." if entry.unit else ""
        frame = f" Frame: {entry.frame}." if entry.frame else ""
        text = f"{entry.description}{unit}{frame} Omit or send null for the embodiment default."
        if entry.type == "enum":
            properties[parameter.name] = {"type": ["string", "null"], "enum": [*entry.enum, None], "description": text}
        else:
            assert entry.range is not None
            properties[parameter.name] = {
                "type": ["number", "null"], "minimum": entry.range[0], "maximum": entry.range[1], "description": text,
            }
    schema = {"type": "object", "additionalProperties": False, "properties": properties, "required": required}
    lint_input_schema(schema, source=f"composed tool {declaration.name}")
    return schema


def _value_property(item: Any) -> dict[str, Any]:
    """The schema of a text, ref, number or choice input; optional ones also accept null."""

    nullable = [] if item.required else ["null"]
    optional = "" if item.required else " Optional."
    if item.kind == "text_input":
        return {"type": ["string", *nullable], "minLength": 1, "maxLength": item.max_chars,
                "description": item.description + optional}
    if item.kind == "ref_input":
        return {"type": ["string", *nullable], "minLength": 1, "maxLength": item.max_chars,
                "description": item.description + " A target_ref from a perception tool on the current observation."
                + optional}
    if item.kind == "number_input":
        unit = f" Unit: {item.unit}." if item.unit else ""
        return {"type": ["number", *nullable], "minimum": item.minimum, "maximum": item.maximum,
                "description": item.description + unit + optional}
    return {"type": ["string", *nullable], "enum": [*item.options, *([None] if nullable else [])],
            "description": item.description + optional}


def _effect(declaration: Declaration, arms: Sequence[str]) -> str:
    stages = ", then ".join(f"{item.name} ({item.summary})" for item in declaration.stages)
    if declaration.input("candidate_input") and not declaration.input("arm_input"):
        arm = "on the arm the candidate was compiled for" + ("; the other arm is not commanded" if len(arms) > 1 else "")
    elif len(arms) > 1:
        arm = "on the arm you name; the other arm is not commanded"
    else:
        arm = f"on the {arms[0]} arm"
    budget = declaration.budgets.primitive_actions
    parts = [
        f"runs one committed sequence through the Supervisor and consumes one robot decision: {stages}, {arm}.",
        f"It dispatches at most {_plural(budget, 'primitive action')}; the Supervisor admits every one like the "
        "matching atomic command.",
    ]
    if budget > len(declaration.action_stages):
        parts.append("A stage may repeat its action within that budget; the receipt reports its attempt count.")
    if len(declaration.stages) > 1:
        parts.append("When a stage does not arrive the sequence stops there and the remaining stages are skipped.")
    if any(item.read_only for item in declaration.stages):
        parts.append("Check stages only read public state.")
    if any(item.primitive == "tool" for item in declaration.stages):
        parts.append("A tool stage runs the composed tool it names as one step of this sequence; its actions count "
                     "against this budget.")
    if not any(item.primitive in ("move", "tool") for item in declaration.stages):
        parts.append("The end-effector pose target is held unchanged.")
    return "Effect: " + " ".join(parts)


def _returns(declaration: Declaration) -> str:
    annotations = [item.name for item in declaration.inputs if item.kind == "annotation_input"]
    echo = ""
    if annotations:
        names = " and ".join(annotations)
        echo = f", with {names} echoed under intent labelled agent_annotation_only"
    evidence = ""
    if declaration.evidence:
        evidence = f"; {' and '.join(declaration.evidence)} when the call produced it"
    return (
        f"Returns: a submitted acknowledgement now{echo}{evidence}; on the next turn a staged receipt with one "
        "entry per stage that ran (status, reason, measurements), the stages not reached, and a fresh observation."
    )


def _on_error(declaration: Declaration) -> str:
    clauses = [f"{item.when} ({item.code})" for item in declaration.refusals]
    clauses.append("an argument outside this schema (invalid_arguments, naming the field)")
    clauses.extend(f"{when} ({code})" for code, when in LIFECYCLE_REFUSALS)
    return (
        "On error: Refused: " + "; ".join(clauses) + ". A refusal submits nothing: correct the named field, "
        "or act on current evidence, before calling again."
    )


def _description(declaration: Declaration, arms: Sequence[str]) -> str:
    return "\n".join([
        declaration.summary,
        f"Use when: {declaration.use_when}",
        _effect(declaration, arms),
        _returns(declaration),
        "Does not: " + "; ".join(declaration.does_not) + ".",
        _on_error(declaration),
    ])


def _result_schema(declaration: Declaration, arms: Sequence[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "success": {"type": "boolean"},
        "status": {"type": "string", "enum": ["submitted"]},
        "tool": {"type": "string", "const": declaration.name},
        "version": {"type": "string", "const": declaration.version},
        "arm": {"type": "string", "enum": list(arms)},
        "committed_sequence": {
            "type": "array", "items": {"type": "string", "enum": [item.name for item in declaration.stages]},
            "minItems": len(declaration.stages), "maxItems": len(declaration.stages),
        },
        "cost": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "robot_decisions": {"type": "integer", "minimum": 1, "maximum": 1},
                "max_primitive_actions": {
                    "type": "integer", "minimum": 1, "maximum": declaration.budgets.primitive_actions,
                },
            },
            "required": ["robot_decisions", "max_primitive_actions"],
        },
        "claim_boundary": {"type": "string"},
    }
    required = ["success", "status", "tool", "version", "arm", "committed_sequence", "cost", "claim_boundary"]
    if declaration.input("candidate_input"):
        properties["candidate_id"] = {"type": "string"}
        required.append("candidate_id")
    for name in declaration.evidence:
        properties[name] = {"type": "object"}
    annotations = [item.name for item in declaration.inputs if item.kind == "annotation_input"]
    if annotations:
        properties["intent"] = {
            "type": "object", "additionalProperties": False,
            "properties": {
                **{name: {"type": ["string", "null"]} for name in annotations},
                "role": {"type": "string", "const": "agent_annotation_only"},
            },
            "required": [*annotations, "role"],
        }
        required.append("intent")
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": required}


def _check_parameters(declaration: Declaration, tunables: TunableTable) -> None:
    for parameter in declaration.parameters:
        if parameter.tunable not in tunables:
            raise CompileError(f"parameter {parameter.name!r} names unknown tunable {parameter.tunable!r}")
        entry = tunables[parameter.tunable]
        if parameter.expose and not entry.qualified:
            raise CompileError(
                f"parameter {parameter.name!r} exposes {entry.id}, which has no qualification evidence and bound; "
                "only its default is admissible"
            )


def compile_tool(
    declaration: Declaration, *, source_sha256: str, tunables: TunableTable, arm_profile: str,
) -> dict[str, Any]:
    """The compiled document for one arm profile; raises CompileError."""

    if arm_profile not in declaration.arm_profiles:
        raise CompileError(f"{declaration.name} does not declare arm profile {arm_profile!r}")
    _check_parameters(declaration, tunables)
    arms = ARM_PROFILES[arm_profile]
    try:
        input_schema = _input_schema(declaration, arms, tunables)
    except DeclarationError:
        raise
    except Exception as exc:  # schema lint failures carry their own message
        raise CompileError(str(exc)) from None
    document = {
        "schema": COMPILED_SCHEMA,
        "tool": declaration.name,
        "version": declaration.version,
        "arm_profile": arm_profile,
        "arms": list(arms),
        "source_sha256": source_sha256,
        "tunables_sha256": tunables.sha256,
        "definition": {
            "name": declaration.name,
            "description": _description(declaration, arms),
            "inputSchema": input_schema,
        },
        "result_schema": _result_schema(declaration, arms),
        "claim_boundary": declaration.claim_boundary or CLAIM_BOUNDARY,
        "evidence": list(declaration.evidence),
        "stages": [{"name": item.name, "primitive": item.primitive, **({"tool": item.tool} if item.tool else {})}
                   for item in declaration.stages],
        "budgets": {
            "primitive_actions": declaration.budgets.primitive_actions,
            "queries": declaration.budgets.queries,
            "wall_s": declaration.budgets.wall_s,
            "measurements_per_stage": declaration.budgets.measurements_per_stage,
        },
        "refusals": [
            {"code": item.code, "retryable": item.retryable} for item in declaration.refusals
        ],
        "parameters": [
            {"name": item.name, "tunable": item.tunable, "expose": item.expose} for item in declaration.parameters
        ],
    }
    document["compiled_sha256"] = "sha256:" + canonical_sha256(document)
    return document


def help_sections(description: str) -> list[str]:
    """Split a compiled description into its six lines, labels checked in order."""

    lines = description.split("\n")
    if len(lines) != 6:
        raise CompileError(f"a policy_tool_help_v1 description has 6 lines, got {len(lines)}")
    for label, line in zip(HELP_LABELS, lines[1:]):
        if not line.startswith(label + " "):
            raise CompileError(f"help line {line[:24]!r} does not start with {label!r}")
    return lines


def compiled_identity(document: Mapping[str, Any]) -> str:
    """Recompute a compiled document's hash; equal to its ``compiled_sha256`` when untouched."""

    body = {key: value for key, value in document.items() if key != "compiled_sha256"}
    return "sha256:" + canonical_sha256(body)
