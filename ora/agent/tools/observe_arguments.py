"""``observe`` argument refusals that name the field, what arrived and what is valid.

``history_point_ids`` naming a mark that is not solved, and overlays requested
on a saved step, were refused with a bare sentence: no
field, no received value, and not the marks that could be drawn.
"""

from __future__ import annotations

from typing import Any, Mapping

from .robot import ToolError


def solved_history(points: Mapping[str, Any], history: Any) -> list[tuple[str, list[float], tuple[int, int, int]]]:
    """The overlay entries of ``history``, refusing any id that is not a solved mark."""

    solved = sorted(str(name) for name, point in points.items()
                    if isinstance(point, Mapping) and point.get("status") == "solved")
    if not isinstance(history, list):
        raise ToolError(
            "history_point_ids must be a list of solved mark names",
            field="history_point_ids", received=history,
            constraints={"solved_point_ids": solved},
        )
    unsolved = [name for name in history if name not in solved]
    if unsolved:
        raise ToolError(
            f"history_point_ids names marks that are not solved: {unsolved}",
            field="history_point_ids", received=unsolved,
            constraints={"solved_point_ids": solved},
        )
    return [(str(name), list(points[name]["xyz_m"]), (0, 180, 255)) for name in history]


def refuse_historical_arguments(arguments: Mapping[str, Any]) -> None:
    """A saved-step ``observe`` takes ``step`` and ``views`` and draws no overlay."""

    unknown = sorted(set(arguments) - {"step", "views", "history_point_ids"})
    if unknown:
        raise ToolError(
            f"historical observe does not take {', '.join(unknown)}",
            field=unknown[0], received=unknown,
            constraints={"allowed": ["step", "views"]},
        )
    if arguments.get("history_point_ids"):
        raise ToolError(
            "history_point_ids overlays are drawn only on the current observation; "
            "a saved step returns its camera images without overlays",
            field="history_point_ids", received=arguments["history_point_ids"],
            constraints={"allowed_with_step": ["step", "views"]},
        )


__all__ = ["refuse_historical_arguments", "solved_history"]
