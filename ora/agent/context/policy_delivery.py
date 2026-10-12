"""Capability-gated delivery of public facts and tool results.

Both runtimes receive the current public robot facts from one delivery site,
:meth:`~ora.agent.context.codex.ContextProjector.publish`, and every
tool result from one gateway each (Native ``NativeToolGateway``, managed MCP).
Each calls here, after the canonical record is written, so the canonical
observation, receipt and ``tool_result`` event keep their exact radian bytes.

Only the surface's own robot tools are delivered this way. A ``compute``
result is the policy's own program output, under names it chose, and artifact
tools (appended outside the surface) return text the policy already received; rewriting any of them would edit the policy's own content.

``within_payload_deduplication_v1`` states shared conventions once;
``policy_angles_degrees_v1`` restates every angle in degrees;
``observation_delivery_trim_v2`` then leaves out what carries no information
(:mod:`.delivery_trim`), last, so its rules see the delivered form. A surface
that declares none of them receives the very object it passed in.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .delivery_trim import CAPABILITY as TRIM, trim_public_facts, trim_tool_result
from .payload_dedup import CAPABILITY as DEDUP, hoist_fact_conventions
from .policy_units import CAPABILITY as DEGREES, policy_degrees

#: Surface tools whose results carry content the policy authored.
POLICY_AUTHORED_RESULT_TOOLS = frozenset({"compute", "report_issue", "finish_episode"})


def deliver_public_facts(facts: Any, *, capabilities: Iterable[str]) -> Any:
    """The current public robot facts as a declaring surface delivers them."""
    declared = frozenset(capabilities) & {DEDUP, DEGREES, TRIM}
    if not declared or not isinstance(facts, Mapping):
        return facts
    delivered = hoist_fact_conventions(facts) if DEDUP in declared else facts
    delivered = policy_degrees(delivered) if DEGREES in declared else delivered
    return trim_public_facts(delivered) if TRIM in declared else delivered


def _robot_tool(surface: Any, name: Any) -> bool:
    from ora.agent.tools import tool_names

    advertised = {item.get("name") for item in getattr(surface, "definitions", ()) or ()}
    return (isinstance(name, str) and name in advertised
            and tool_names.canonical(name) not in POLICY_AUTHORED_RESULT_TOOLS)


def deliver_tool_result(surface: Any, name: Any, result: Any) -> Any:
    """One robot-tool result (or refusal) as a declaring surface delivers it."""
    declared = frozenset(getattr(surface, "capabilities", ()) or ()) & {DEDUP, DEGREES, TRIM}
    if not declared or not isinstance(result, Mapping) or not _robot_tool(surface, name):
        return result
    delivered = dict(result)
    if DEDUP in declared and isinstance(delivered.get("public_robot_facts"), Mapping):
        delivered["public_robot_facts"] = hoist_fact_conventions(delivered["public_robot_facts"])
    delivered = policy_degrees(delivered) if DEGREES in declared else delivered
    return trim_tool_result(delivered) if TRIM in declared else delivered


__all__ = [
    "POLICY_AUTHORED_RESULT_TOOLS", "deliver_public_facts", "deliver_tool_result",
]
