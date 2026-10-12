"""Parse one assistant tool batch and decide which members may run.

A provider reply may propose several tool calls. Each is decoded into an
execution copy -- the assistant message in history stays exact, except that an
id-less call adopts the id the dispatcher uses -- and the batch is admitted as
a whole: a world mutation must be the only call (in a mixed batch only the
mutation is refused, the read-only calls still run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .provider_calls import adopt_synthesized_tool_call_id
from .tool_arguments import _arguments

# (execution call copy, tool name, decoded arguments, typed effect, parse error)
ParsedCall = tuple[
    dict[str, Any], str | None, dict[str, Any] | None,
    str | None, Exception | None,
]
EffectClassifier = Callable[[str, dict[str, Any]], Any]


def parse_tool_calls(
    calls: list[Any],
    *,
    assistant: dict[str, Any],
    model_call_id: str,
    classify_effect: EffectClassifier | None,
) -> tuple[list[ParsedCall], list[str]]:
    """Decode every call; ``calls`` is replaced element-wise by its copies.

    Returns the parsed batch and the names of the calls that decoded. A call
    that fails to decode keeps the name the provider gave it, so its result
    pairs with the same name the assistant message shows.
    """

    parsed: list[ParsedCall] = []
    names: list[str] = []
    for index, raw_call in enumerate(calls):
        call = dict(raw_call) if isinstance(raw_call, Mapping) else {}
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"{model_call_id}-tool-{index:02d}"
            call["id"] = call_id
            adopt_synthesized_tool_call_id(assistant, index=index, call_id=call_id)
        try:
            name, args = _arguments(call)
            function = dict(call.get("function", {}))
            function["name"] = name
            function["arguments"] = json.dumps(
                args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            call["type"] = "function"
            call["function"] = function
            names.append(name)
            effect = (
                classify_effect(name, args)
                if classify_effect is not None else None
            )
            parsed.append((call, name, args, effect, None))
        except Exception as exc:
            # JSON failure does not erase the function identity already
            # present in the assistant message. Context pairs that name
            # with the result before the policy can read the refusal.
            function = call.get("function")
            original_name = function.get("name") if isinstance(function, Mapping) else None
            name = original_name if isinstance(original_name, str) else "invalid_tool_call"
            if not isinstance(original_name, str):
                call["type"] = "function"
                call["function"] = {"name": name, "arguments": "{}"}
            parsed.append((call, name, None, None, exc))
        calls[index] = call
    return parsed, names


@dataclass(frozen=True)
class BatchAdmission:
    """Which members of one parsed batch may execute."""

    refused_mutations: frozenset[int]
    """Indexes of world mutations refused for sharing the batch."""
    reads_run: int
    """How many members still execute next to a refused mutation."""
    typed_effects: bool
    mutation_tools: frozenset[str]

    def is_mutation(self, effect: str | None, name: str | None) -> bool:
        """Typed slots classify by effect; older compositions by tool name."""

        return (
            effect == "mutation" if self.typed_effects
            else name in self.mutation_tools
        )


def admit_tool_batch(
    parsed: Sequence[ParsedCall],
    names: Sequence[str],
    *,
    call_count: int,
    typed_effects: bool,
    mutation_tools: set[str],
) -> BatchAdmission:
    """Apply the one-mutation batch rule."""

    mutation_count = (
        sum(effect == "mutation" for _call, _name, _args, effect, _error in parsed)
        if typed_effects
        else sum(name in mutation_tools for name in names)
    )
    # A world mutation must be alone. In a mixed batch only the mutation
    # is refused: the read-only calls run and the refusal says so.
    mutation_batch = mutation_count > 1 or (mutation_count == 1 and call_count > 1)
    refused_mutations = {
        index for index, (_call, name, _args, effect, _error) in enumerate(parsed)
        if mutation_batch and (
            effect == "mutation" if typed_effects
            else name in mutation_tools
        )
    }
    reads_run = sum(index not in refused_mutations and error is None
                    for index, (_c, _n, _a, _e, error) in enumerate(parsed))
    return BatchAdmission(
        refused_mutations=frozenset(refused_mutations),
        reads_run=reads_run,
        typed_effects=typed_effects,
        mutation_tools=frozenset(mutation_tools),
    )
