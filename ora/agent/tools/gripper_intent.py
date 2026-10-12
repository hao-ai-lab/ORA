"""The close-intent annotation a policy attaches when it sets a gripper.

`close_gripper` carried an optional `target_ref`: the policy's statement of
which object it believes it is closing on. It is an annotation and never
evidence, and the result says so in the same breath it reports the reference --
a closed jaw is not a grasp, and a reference beside one does not make it into a
measurement.

`set_gripper` replaces that tool, so it carries the annotation too. This lives
outside the tool runtime because that file is already past five thousand lines
and every tool that grew there did so twenty lines at a time.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import retention
from .robot import ToolError

ANNOTATION_ROLE = "agent_annotation_only"


def extract(
    arguments: Mapping[str, Any], *, surface: Any,
) -> tuple[dict[str, Any], str | None, bool]:
    """Split an optional target_ref off the arguments the codec will see.

    Returns the arguments without it, the reference, and whether this surface
    publishes the annotation at all. A surface that does not declare the
    advisory capability never sees the field, so passing one there falls
    through to the ordinary unexpected-field refusal rather than being
    silently accepted and dropped.
    """

    if not surface.supports(retention.ADVISORY_CAPABILITY):
        return dict(arguments), None, False
    reference = arguments.get("target_ref")
    if reference is not None and (
        not isinstance(reference, str) or not reference.strip()
    ):
        raise ToolError(
            "target_ref must be a non-empty intent annotation or null",
            code="invalid_arguments", field="target_ref",
        )
    remainder = {
        key: value for key, value in arguments.items() if key != "target_ref"
    }
    return remainder, reference, True


def annotate(
    result: dict[str, Any], *, targets: Mapping[str, Any],
    reference: str | None, published: bool,
) -> dict[str, Any]:
    """Attach the annotation when something actually closed.

    An annotation naming what is being grasped has no meaning on release, so a
    call that only opens carries none. The role is stated beside the reference
    rather than left to the reader.
    """

    if published and any(
        directive == "close" for directive in targets.values()
    ):
        result["intent"] = {"target_ref": reference, "role": ANNOTATION_ROLE}
    return result
