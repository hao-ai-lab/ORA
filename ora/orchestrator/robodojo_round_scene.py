"""The Supervisor's view of a scheduled RoboDojo round's scene.

A RunPlan with ``runtime.layout_schedule`` plans one layout per round. Before
the first observation of every round the Supervisor compares the backend's
reset record with that plan, records the measured scene on the round result
and, only when the arrangement really differs from every earlier round, adds
one neutral sentence to the round notice. The sentence carries no layout id,
position or count. An unscheduled plan is untouched by all three.
"""

from __future__ import annotations

from typing import Any, Mapping

from .episode_bindings import SupervisorError
from .robodojo_layout_evaluation import ROUND_SCENE_FIELDS, scheduled_rounds

LAYOUT_CHANGE_SENTENCE = (
    " This round begins from a different initial arrangement than earlier rounds;"
    " locate every object again from the current observation."
    " Carry over how to do the task, not where things were."
)


def require_round_scene(plan: Mapping[str, Any], reset: Mapping[str, Any], index: int) -> None:
    """Refuse a round whose reset record did not load the planned layout."""

    schedule = scheduled_rounds(plan)
    if schedule is None:
        return
    if not 0 <= index < len(schedule):
        raise SupervisorError(f"round {index} is outside the planned layout schedule")
    entry = schedule[index]
    expected = {"layout_id": entry.get("layout_id"), "layout_file": entry.get("layout_file"),
                "layout_sha256": entry.get("layout_sha256"), "env_seed": entry.get("layout_id")}
    observed = {field: reset.get(field) for field in expected}
    if observed != expected:
        raise SupervisorError(
            f"environment round {index} layout drift: planned {expected}, reset recorded {observed}")


def round_scene_field(plan: Mapping[str, Any], reset: Mapping[str, Any]) -> dict[str, Any]:
    """The round result's ``scene`` entry, or nothing for an unscheduled plan."""

    if scheduled_rounds(plan) is None:
        return {}
    return {"scene": {field: reset.get(field) for field in ROUND_SCENE_FIELDS}}


def layout_change_sentence(plan: Mapping[str, Any], index: int) -> str:
    """The notice sentence for a round whose layout differs from all earlier ones."""

    schedule = scheduled_rounds(plan)
    if schedule is None or not 0 < index < len(schedule):
        return ""
    earlier = {entry.get("layout_sha256") for entry in schedule[:index]}
    return "" if schedule[index].get("layout_sha256") in earlier else LAYOUT_CHANGE_SENTENCE


__all__ = [
    "LAYOUT_CHANGE_SENTENCE", "layout_change_sentence", "require_round_scene", "round_scene_field",
]
