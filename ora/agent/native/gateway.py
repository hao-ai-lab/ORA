"""Native provider tool-schema projection and ToolRuntime gateway."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ora.agent.composition.integral_arguments import coerce_integral_arguments
from ora.agent.composition.null_optional_arguments import (
    CAPABILITY as NULL_OMITS, deliver_tool_definitions, omit_null_optional_arguments,
)
from ora.agent.context.policy_delivery import deliver_tool_result
from ora.agent.contracts import PolicySessionError
from ora.agent.tools.repeated_failures import RepeatedFailureTracker
from ora.agent.tools.robot import TOOL_DEFINITIONS

from .constants import MUTATION_TOOL


def _openai_parameters(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project a top-level union into compatible function parameters."""

    value = copy.deepcopy(dict(schema))
    branches = value.get("oneOf")
    if not isinstance(branches, list) or not branches:
        return value
    if not all(isinstance(branch, Mapping) for branch in branches):
        return value
    if all(branch.get("type") == "object" for branch in branches):
        # Strict Chat Completions endpoints require the function's root type,
        # even when every union branch already proves it. Preserve this safe
        # annotation if conflicting branch properties prevent flattening.
        value.setdefault("type", "object")
    properties: dict[str, Any] = {}
    required_sets: list[set[str]] = []
    for branch in branches:
        branch_properties = branch.get("properties")
        branch_required = branch.get("required")
        if not isinstance(branch_properties, Mapping) or not isinstance(
            branch_required, list,
        ):
            return value
        required_sets.append({
            item for item in branch_required if isinstance(item, str)
        })
        for name, raw_property in branch_properties.items():
            if not isinstance(name, str) or not isinstance(raw_property, Mapping):
                return value
            projected = copy.deepcopy(dict(raw_property))
            if "const" in projected:
                projected["enum"] = [projected.pop("const")]
            previous = properties.get(name)
            if previous is None:
                properties[name] = projected
                continue
            if previous == projected:
                continue
            previous_enum = (
                previous.get("enum") if isinstance(previous, Mapping) else None
            )
            projected_enum = projected.get("enum")
            if (
                isinstance(previous_enum, list)
                and isinstance(projected_enum, list)
                and {
                    key: item for key, item in previous.items() if key != "enum"
                }
                == {
                    key: item for key, item in projected.items() if key != "enum"
                }
            ):
                properties[name]["enum"] = list(dict.fromkeys(
                    previous_enum + projected_enum,
                ))
                continue
            return value
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": sorted(set.intersection(*required_sets)),
    }
    if all(branch.get("additionalProperties") is False for branch in branches):
        result["additionalProperties"] = False
    # Flattened branch properties retain local references to the sibling table.
    definitions = value.get("$defs")
    if isinstance(definitions, Mapping) and definitions:
        result["$defs"] = copy.deepcopy(dict(definitions))
    return result


def openai_tool_definitions(
    allowed: Sequence[str],
    *,
    definitions: Sequence[Mapping[str, Any]] = TOOL_DEFINITIONS,
) -> list[dict[str, Any]]:
    selected = set(allowed)
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": _openai_parameters(item["inputSchema"]),
            },
        }
        for item in definitions
        if item["name"] in selected
    ]


@dataclass
class NativeToolGateway:
    runtime: Any
    #: The session owns exactly one gateway, so the count of identical
    #: consecutive failures is scoped to one policy session.
    repeated_failures: RepeatedFailureTracker = field(
        default_factory=RepeatedFailureTracker,
    )
    _delivered: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)

    def record_proposed_result(
        self, name: str, proposal_event: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Pass one completed call's result through before it is recorded.

        The ``tool_call_proposed`` event holds the decoded arguments. A
        proposal without them -- undecodable JSON, a rejected batch -- resets
        the count. Only the returned new result is ever annotated.

        The surface's delivery rules (degrees, conventions stated once) apply
        to what the session records and shows; the ToolRuntime's own
        ``tool_result`` event already holds the canonical result.
        """

        payload = proposal_event.get("payload")
        arguments = payload.get("arguments") if isinstance(payload, Mapping) else None
        recorded = self.repeated_failures.record(name, arguments, result)
        return deliver_tool_result(getattr(self.runtime, "surface", None), name, recorded)

    def delivered(self) -> list[dict[str, Any]]:
        """The final tool list, as the surface's delivery rules project it (computed once)."""

        if self._delivered is None:
            surface = getattr(self.runtime, "surface", None)
            self._delivered = deliver_tool_definitions(
                self.runtime.definitions(), tuple(getattr(surface, "capabilities", ()) or ()),
            )
        return self._delivered

    def arguments(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """The arguments a handler receives: declared null-as-omitted applied at every
        depth, then integral floats at integer positions as ints (every surface)."""

        surface = getattr(self.runtime, "surface", None)
        listed = self.delivered() if callable(getattr(self.runtime, "definitions", None)) else None
        if NULL_OMITS in tuple(getattr(surface, "capabilities", ()) or ()):
            arguments = omit_null_optional_arguments(surface, name, arguments, definitions=listed)
        return coerce_integral_arguments(listed or (), name, arguments)

    def definitions(self) -> Sequence[Mapping[str, Any]]:
        return openai_tool_definitions(
            self.runtime.allowed_tools,
            definitions=self.delivered(),
        )

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        arguments = self.arguments(name, arguments)
        invoke_with_context = getattr(self.runtime, "invoke_with_context", None)
        if call_context is not None and callable(invoke_with_context):
            return invoke_with_context(
                name, arguments, call_context=call_context,
            )
        return self.runtime.invoke(name, arguments)

    def classify_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        classifier = getattr(self.runtime, "classify_call_effect", None)
        if callable(classifier):
            effect = classifier(name, self.arguments(name, arguments))
            if effect not in {
                "read_only",
                "mutation",
                "lifecycle",
            }:
                raise PolicySessionError(
                    f"invalid call effect classification: {effect}"
                )
            return str(effect)
        surface = self.runtime.surface
        if getattr(surface, "call_effect_profile", None) is not None:
            raise PolicySessionError(
                "ToolSurface requires a call-effect classifier"
            )
        if name in getattr(surface, "lifecycle_tools", ()):
            return "lifecycle"
        if name in getattr(surface, "mutation_tools", (MUTATION_TOOL,)):
            return "mutation"
        return "read_only"

    def begin_read_only_feedback_batch(
        self, name: str, calls: Sequence[Mapping[str, Any]],
    ) -> bool:
        begin = getattr(self.runtime, "begin_read_only_feedback_batch", None)
        calls = [self.arguments(name, call) for call in calls]
        return bool(begin(name, calls)) if callable(begin) else False

    def finish_read_only_feedback_batch(
        self, name: str, results: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        finish = getattr(self.runtime, "finish_read_only_feedback_batch", None)
        if not callable(finish):
            raise PolicySessionError(
                "active read-only batch lacks a feedback renderer"
            )
        result = finish(name, results)
        if not isinstance(result, Mapping):
            raise PolicySessionError("read-only batch feedback must be an object")
        return result
