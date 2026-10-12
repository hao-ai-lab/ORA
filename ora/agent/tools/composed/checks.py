"""Layer-1 static checks: what a composed tool must pass before it is admitted.

Loading already enforces the language boundary (Starlark dialect, only SDK
symbols, no ``load``, no host query or action at top level) and the header
schema. These checks add what loading cannot see:

* every arm profile compiles against the tunable table, and the committed
  ``compiled/`` document equals a fresh compile (no stale or hand-edited help);
* the help has its six labelled lines in order and passes the ToolSurface
  ``policy_tool_help_v1`` contract (closed schema, described properties,
  result schema present);
* declared free text names no tool except the tool itself and ``refers_to``;
* every ``stage = "..."`` and ``sdk.measure("...", ...)`` literal in the body
  names a declared stage, and every action stage is driven somewhere;
* the worst-case receipt fits the skill-receipt budget without eliding a
  measurement;
* every ``sdk.read`` names a read tool literally, and every ``tool`` stage
  resolves (next to the tool, then in the library) to a tool declaring the
  same arm profiles, with no cycle and no nesting past the cap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ora.agent.composition.tool_surface import validate_tool_help_contract
from ora.agent.tools import skill_receipt

from . import sdk
from .compiler import CLAIM_BOUNDARY, CompileError, compile_tool, help_sections
from .library import ComposedTool
from .tunables import TunableTable

_STAGE_LITERAL = re.compile(r"\bstage\s*=\s*\"([^\"]*)\"")
_MEASURE_LITERAL = re.compile(r"\bsdk\.measure\(\s*\"([^\"]*)\"")
_ASK_LITERAL = re.compile(r"\bsdk\.ask\(\s*([^,)]*)")
_READ_LITERAL = re.compile(r"\bsdk\.read\(\s*([^,)]*)")
_WORD = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?")
TOOL_SURFACE_ROOT = Path(__file__).resolve().parents[3] / "policy" / "tool_surfaces"


@dataclass(frozen=True)
class Finding:
    tool: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.tool}: {self.check}: {self.message}"


def known_tool_names(root: Path | None = None) -> frozenset[str]:
    """Every tool name any registered ToolSurface definition catalog declares.

    Definition catalogs may spell a tool by its dotted source name; a surface
    that canonicalizes names advertises the underscored one. Without an
    explicit ``root``, the names every registered profile advertises after
    resolution are included too, so a canonical name counts even when no
    catalog file spells it.
    """

    names: set[str] = set()
    if root is None:
        from ora.agent.composition.tool_surface import PROFILE_CATALOG, resolve_tool_surface

        for profile_id in PROFILE_CATALOG:
            names.update(item["name"] for item in resolve_tool_surface(profile_id).definitions)
    for path in sorted(Path(root or TOOL_SURFACE_ROOT).rglob("*definitions*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tools = document.get("tools") if isinstance(document, dict) else document
        for item in tools if isinstance(tools, list) else []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
    return frozenset(names)


def _free_text(tool: ComposedTool) -> Iterable[tuple[str, str]]:
    declaration = tool.declaration
    yield "summary", declaration.summary
    yield "use_when", declaration.use_when
    for index, text in enumerate(declaration.does_not):
        yield f"does_not[{index}]", text
    for item in declaration.refusals:
        yield f"refusals[{item.code}].when", item.when
    for item in declaration.stages:
        yield f"stages[{item.name}].summary", item.summary
    for item in declaration.inputs:
        if item.description:
            yield f"inputs[{item.name}].description", item.description


def _worst_case_receipt_bytes(tool: ComposedTool) -> tuple[int, int]:
    declaration = tool.declaration
    body = {
        "quantity": "q" * 64, "value": [0.123456789] * sdk.MAX_MEASUREMENT_VECTOR, "unit": "u" * 16,
    }
    stages = []
    for index, item in enumerate(declaration.stages):
        measured = [dict(entry) for entry in sdk.WORST_CASE_OUTCOME_MEASUREMENTS[item.primitive]]
        measured += [dict(body) for _ in range(declaration.budgets.measurements_per_stage)]
        if not item.read_only:
            measured.append({"quantity": "attempts", "value": declaration.budgets.primitive_actions})
        stages.append(skill_receipt.stage(
            item.name, "not_reached", "r" * 64, action_id=f"call-{'0' * 32}:{index:02d}-{item.name}",
            measurements=measured,
        ))
    receipt = skill_receipt.assemble(
        skill=declaration.name, status="aborted", rationale="x" * skill_receipt.MAX_RATIONALE_CHARS,
        candidate_id=None, observation_id="obs-" + "0" * 60, observation_sha256="f" * 64,
        stages=stages, claim_boundary=CLAIM_BOUNDARY,
    )
    receipt["stop_reason"] = "s" * 64
    receipt["cost"] = {"robot_decisions": 1, "dispatched_stage_actions": declaration.budgets.primitive_actions}
    receipt["composed_tool"] = {
        "tool": declaration.name, "version": declaration.version, "arm_profile": "dual_arm",
        "compiled_sha256": "sha256:" + "0" * 64,
    }
    return skill_receipt.body_bytes(receipt), receipt["delivery"]["stages_with_elided_measurements"]


def check_tool(tool: ComposedTool, *, tunables: TunableTable, known_names: frozenset[str]) -> list[Finding]:
    findings: list[Finding] = []
    name = f"{tool.name}/{tool.version}"
    declaration = tool.declaration

    def find(check: str, message: str) -> None:
        findings.append(Finding(name, check, message))

    for arm_profile in declaration.arm_profiles:
        try:
            fresh = compile_tool(declaration, source_sha256=tool.program.digest, tunables=tunables,
                                 arm_profile=arm_profile)
        except CompileError as exc:
            find("compile", f"{arm_profile}: {exc}")
            continue
        try:
            help_sections(fresh["definition"]["description"])
        except CompileError as exc:
            find("help_sections", f"{arm_profile}: {exc}")
        try:
            validate_tool_help_contract([fresh["definition"]], {declaration.name: fresh["result_schema"]})
        except Exception as exc:  # the ToolSurface validator raises its own error type
            find("help_contract", f"{arm_profile}: {exc}")
        path = tool.compiled_path(arm_profile)
        if not path.is_file():
            find("compiled_missing", f"{arm_profile}: run the compile command")
        elif json.loads(path.read_text(encoding="utf-8")) != fresh:
            find("compiled_drift", f"{arm_profile}: committed document differs from a fresh compile")
    stale = sorted(
        path.stem for path in (tool.directory / "compiled").glob("*.json")
        if path.stem not in declaration.arm_profiles
    ) if (tool.directory / "compiled").is_dir() else []
    for arm_profile in stale:
        find("compiled_stale", f"{arm_profile}.json is not a declared arm profile")

    allowed = {declaration.name, *declaration.refers_to}
    for item in declaration.refers_to:
        if item not in known_names:
            find("refers_to", f"{item!r} is not a tool on any registered surface")
    for where, text in _free_text(tool):
        for word in sorted(set(_WORD.findall(text))):
            canonical = word.replace(".", "_")
            if (word in known_names or canonical in known_names) and not {word, canonical} & allowed:
                find("undeclared_tool_name", f"{where} names {word!r}; add it to refers_to or rephrase")

    declared = declaration.stage_primitives
    driven = set(_STAGE_LITERAL.findall(tool.source))
    for literal in sorted(driven | set(_MEASURE_LITERAL.findall(tool.source))):
        if literal not in declared:
            find("stage_attribution", f"the body names undeclared stage {literal!r}")
    for item in declaration.action_stages:
        if item.name not in driven:
            find("stage_attribution", f"action stage {item.name!r} is never driven by the body")

    from .questions import QuestionError, load_question

    for literal in _ASK_LITERAL.findall(tool.source):
        ref = literal.strip()
        if not (ref.startswith('"') and ref.endswith('"')):
            find("ask_question", f"sdk.ask must name a question literal, got {ref!r}")
            continue
        try:
            question = load_question(ref.strip('"'))
        except QuestionError as exc:
            find("ask_question", str(exc))
            continue
        if not question.admitted:
            find("ask_question", f"{question.ref} has not reached its evaluation threshold")

    for literal in _READ_LITERAL.findall(tool.source):
        ref = literal.strip()
        if not (ref.startswith('"') and ref.endswith('"')) or ref.strip('"') not in sdk.READ_TOOLS:
            find("read_tool", f"sdk.read must name one of {list(sdk.READ_TOOLS)} literally, got {ref!r}")

    from .library import LibraryError, nested_closure

    try:
        closure = nested_closure(tool)
    except LibraryError as exc:
        find("nested_tool", str(exc))
        closure = {}
    for ref, nested in closure.items():
        missing = sorted(set(declaration.arm_profiles) - set(nested.declaration.arm_profiles))
        if missing:
            find("nested_tool", f"{ref} does not declare arm profiles {missing}")

    size, elided = _worst_case_receipt_bytes(tool)
    if elided or size > skill_receipt.RECEIPT_BUDGET_BYTES:
        find("receipt_budget", f"worst-case receipt is {size} bytes with {elided} elided stages; "
                               f"budget {skill_receipt.RECEIPT_BUDGET_BYTES}")
    return findings
