"""Native session, engine, request assembly, stage guard and tool loop.

Exports load on first access (PEP 562), so importing a light submodule such as
``ora.agent.native.tui`` does not import the whole Native loop.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NativeSessionEngine": ".engine",
    "MUTATION_TOOL": ".constants",
    "NATIVE_SLOTS_V3": ".constants",
    "NativePlanner": ".session",
    "assemble_native_v3_static_context": ".composition",
    "NativeToolGateway": ".gateway",
    "openai_tool_definitions": ".gateway",
    "NativeRequestAssembler": ".request_assembly",
    "PreparedNativeRequest": ".request_assembly",
    "NativePolicySession": ".policy_session",
    "NativeTurnLoop": ".tool_loop",
    "native_turn_instruction": ".stage_guard",
}

__all__ = [
    "NativePlanner",
    "NativePolicySession",
    "MUTATION_TOOL",
    "NATIVE_SLOTS_V3",
    "NativeRequestAssembler",
    "NativeSessionEngine",
    "NativeToolGateway",
    "NativeTurnLoop",
    "PreparedNativeRequest",
    "assemble_native_v3_static_context",
    "native_turn_instruction",
    "openai_tool_definitions",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
