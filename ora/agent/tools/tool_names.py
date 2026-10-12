"""One canonical spelling per tool, and the dotted form frozen surfaces use.

Both provider protocols constrain a function name to ``^[a-zA-Z0-9_-]{1,64}``,
so a dot is not a legal name. Every dotted tool has been travelling through an
alias rewrite on the wire and back again. The convention was also never uniform:
most tools were already snake_case and a minority carried a family prefix
separated by a dot.

Active surfaces move to snake_case with the family first, so a model scanning
the tool table sees a family together and the alias layer becomes an identity
map. Frozen surfaces keep their dotted names, because those are what sealed runs
were shown -- of the 176 files naming a dotted tool, 174 are frozen.

So the runtime answers to both. Normalising at the dispatch boundary rather than
registering each handler twice means a tool added later cannot be reachable
under one spelling and missing under the other.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

#: Dotted spelling -> canonical spelling. The dotted side is what frozen
#: surfaces advertise; the canonical side is what active surfaces advertise and
#: what a handler is registered under.
DOTTED_TOOL_NAMES = {
    "vision.segment": "vision_segment",
    "vision.ground": "vision_ground",
    "vision.offset": "vision_offset",
    "grasp.plan": "grasp_plan",
    "grasp.verify_retention": "grasp_verify_retention",
    "place.plan": "place_plan",
    "place.verify": "place_verify",
    "motion.probe": "motion_probe",
    "payload.bind": "payload_bind",
    "payload.unbind": "payload_unbind",
    "align_gripper.propose": "align_gripper_propose",
    # Dotted like the rest and dispatched from the same table. Leaving it out
    # would be a rule with one unstated exception, which is how a constraint
    # stops being read as one.
    "grasp.query": "grasp_query",
    # The runtime extension reader. It is appended after composition, so the
    # canonical rename never reached it and a surface that spells every other
    # tool in snake_case still advertised one dotted name. That is the unstated
    # exception the comment above says a rule cannot carry.
    "artifact.read": "artifact_read",
}

CANONICAL_TOOL_NAMES = {
    canonical: dotted for dotted, canonical in DOTTED_TOOL_NAMES.items()
}


def canonical(name: str) -> str:
    """The spelling a handler is registered under, whichever was advertised."""

    return DOTTED_TOOL_NAMES.get(name, name)


def advertised(name: str, *, dotted: bool) -> str:
    """The spelling a given surface advertises for a canonical tool."""

    if not dotted:
        return name
    return CANONICAL_TOOL_NAMES.get(name, name)


def declares(advertised_names: Iterable[str], tool: str) -> bool:
    """Whether these advertised names include one tool, in either spelling.

    The runtime asks this of ``allowed_tools``, which is built from whatever
    each surface advertises, so a literal ``"vision.segment" in allowed_tools``
    silently answers False the moment a surface spells it ``vision_segment``.
    Silently: the caller turns a capability off rather than raising, so the
    failure is a feature quietly missing, not an error anyone sees.
    """

    target = canonical(tool)
    return any(canonical(name) == target for name in advertised_names)


def as_advertised(advertised_names: Iterable[str], tool: str) -> str:
    """The spelling these advertised names use for one tool.

    For indexing a mapping the surface owns -- ``result_schemas`` above all,
    which is snapshot verbatim into the RunPlan, so its keys must stay exactly
    what that surface published and cannot be normalised. The lookup moves
    instead of the mapping.

    Returns ``tool`` unchanged when nothing matches, so a caller asking about a
    tool this surface does not declare fails on the lookup it was going to make
    anyway, naming what it asked for.
    """

    target = canonical(tool)
    for name in advertised_names:
        if canonical(name) == target:
            return name
    return tool


def definition_for(
    definitions: Sequence[Mapping[str, Any]], tool: str,
) -> Mapping[str, Any]:
    """The advertised definition of one tool, whichever spelling it carries.

    Three call sites reached for ``next(item for item in definitions if
    item["name"] == name)`` with a name the caller chose rather than one the
    surface gave it. That raises ``StopIteration`` on a mismatch -- not a
    ``ToolError`` -- so it surfaces as an internal fault rather than anything a
    policy could read. Kept here so the pattern has one owner.
    """

    target = canonical(tool)
    for item in definitions:
        if canonical(str(item.get("name"))) == target:
            return item
    raise KeyError(tool)
