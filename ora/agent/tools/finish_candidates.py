"""The plan candidates a round offered and did not try, for the give_up record (F4).

09-09 condition (a): EXP11A g008/g009 finish records named options the policy
had not tried. User decision 2026-10-04: a give_up is never refused for it;
the persisted finish record says which offered candidates were not tried and
whether the give_up reason names any of them.

Everything is read from the run's own event log since the round's ``reset``:
``grasp_plan`` and ``place_plan`` results offer candidates; a candidate is
tried when

- ``candidate_id``: an accepted ``grasp_execute``/``place_execute`` names it
  (RoboTwin and RoboDojo, which declare those tools). Grasp ids are hashes of
  the candidate's identity, so a re-plan that offers the same grasp offers one
  candidate; ``place_plan`` ids (``P<n>``) are per call, so each call's
  candidates are their own entries and an execute names the latest call's.
  An id is reported untried only when no offering of it was tried.
- ``pose_match_approximate``: an accepted ``move_pose`` target with an absolute
  position within ``POSITION_TOLERANCE_M`` of one of the candidate's poses
  (grasp pregrasp or contact pose, place ``move_pose``), and an absolute
  orientation within ``ORIENTATION_TOLERANCE_DEG`` when both are given. This
  is the only detector on LIBERO, whose surface copies candidate poses into
  ``move_pose`` and has no execute tools; on the others it also counts a
  copied pose.

A refused call (``tool_error``) tries nothing. ``align_plan`` rows carry no
candidate identity and are not counted. The reason is searched for the
literal candidate ids.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA = "ora.policy-finish-untried-candidates.v1"
CANDIDATE_ID = "candidate_id"
POSE_MATCH = "pose_match_approximate"
OFFER_TOOLS = ("grasp_plan", "place_plan")
EXECUTE_TOOLS = ("grasp_execute", "place_execute")
POSITION_TOLERANCE_M = 0.01
ORIENTATION_TOLERANCE_DEG = 10.0
_TOOLS = (*OFFER_TOOLS, *EXECUTE_TOOLS, "move_pose")
_MARKERS = ('"type": "reset"', *(f'"tool": "{tool}"' for tool in _TOOLS))


def methods_for(surface: Any) -> tuple[str, ...]:
    """The detectors a surface supports: identity where it executes candidates by id."""

    names = {item.get("name") for item in surface.definitions}
    return (CANDIDATE_ID, POSE_MATCH) if names & set(EXECUTE_TOOLS) else (POSE_MATCH,)


def _round_events(lines: Iterable[str]) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for line in lines:
        if not any(marker in line for marker in _MARKERS):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, Mapping):
            continue
        if event.get("type") == "reset":
            events = []
        elif event.get("type") == "tool_result" and event.get("tool") in _TOOLS:
            events.append(event)
    return events


def _pose(value: Any) -> tuple[list[float], list[float] | None] | None:
    """An absolute world pose from a candidate pose or a move_pose target."""

    if not isinstance(value, Mapping):
        return None
    mode = value.get("position_mode", "absolute")
    position = value.get("position_m")
    if mode != "absolute" or not isinstance(position, list) or len(position) != 3:
        return None
    try:
        point = [float(item) for item in position]
    except (TypeError, ValueError):
        return None
    orientation = value.get("orientation_xyzw")
    absolute = value.get("orientation_mode", "absolute_quaternion") == "absolute_quaternion"
    quaternion = None
    if absolute and isinstance(orientation, list) and len(orientation) == 4:
        try:
            quaternion = [float(item) for item in orientation]
        except (TypeError, ValueError):
            quaternion = None
    return point, quaternion


def _targets(value: Any) -> Iterator[tuple[list[float], list[float] | None]]:
    """Every absolute pose in a flat or per-arm move_pose payload."""

    if not isinstance(value, Mapping):
        return
    nested = value.get("targets")
    for target in (nested.values() if isinstance(nested, Mapping) else (value,)):
        pose = _pose(target)
        if pose is not None:
            yield pose


def _candidate_poses(tool: str, candidate: Mapping[str, Any]) -> list:
    if tool == "grasp_plan":
        return [pose for key in ("pregrasp_pose", "contact_pose")
                if (pose := _pose(candidate.get(key))) is not None]
    return list(_targets(candidate.get("move_pose")))


def _angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    if norm == 0.0:
        return math.inf
    dot = min(1.0, abs(sum(a * b for a, b in zip(first, second))) / norm)
    return math.degrees(2.0 * math.acos(dot))


def _matches(target, poses) -> bool:
    position, orientation = target
    for point, quaternion in poses:
        if math.dist(position, point) > POSITION_TOLERANCE_M:
            continue
        if orientation is None or quaternion is None:
            return True
        if _angle_deg(orientation, quaternion) <= ORIENTATION_TOLERANCE_DEG:
            return True
    return False


def _offer(entries: list[dict[str, Any]], latest: dict[str, dict[str, Any]],
           event: Mapping[str, Any], call: int) -> None:
    tool = str(event["tool"])
    result = event.get("result")
    candidates = result.get("candidates") if isinstance(result, Mapping) else None
    for candidate in candidates if isinstance(candidates, list) else ():
        identity = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        if not isinstance(identity, str) or not identity:
            continue
        if tool == "grasp_plan" and any(item["candidate_id"] == identity and item["tool"] == tool
                                        for item in entries):
            continue
        entry = {"candidate_id": identity, "tool": tool, "call": call, "tried_by": [],
                 "_poses": _candidate_poses(tool, candidate)}
        entries.append(entry)
        latest[f"{tool}:{identity}"] = entry


def _try(entry: dict[str, Any], method: str) -> None:
    if method not in entry["tried_by"]:
        entry["tried_by"].append(method)


def ledger(lines: Iterable[str], *, reason: str, methods: Sequence[str]) -> dict[str, Any]:
    """The round's offered candidates, which were tried, and which the reason names."""

    entries: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    calls = {tool: 0 for tool in OFFER_TOOLS}
    for event in _round_events(lines):
        tool = event["tool"]
        arguments = event.get("arguments") if isinstance(event.get("arguments"), Mapping) else {}
        if tool in OFFER_TOOLS:
            calls[tool] += 1
            _offer(entries, latest, event, calls[tool])
        elif tool in EXECUTE_TOOLS and CANDIDATE_ID in methods:
            plan = "grasp_plan" if tool == "grasp_execute" else "place_plan"
            entry = latest.get(f"{plan}:{arguments.get('candidate_id')}")
            if entry is not None:
                _try(entry, CANDIDATE_ID)
        elif tool == "move_pose" and POSE_MATCH in methods:
            for target in _targets(arguments):
                for entry in entries:
                    if _matches(target, entry["_poses"]):
                        _try(entry, POSE_MATCH)
    # An id counts as tried once any offering of it was tried: a reason that
    # names "P0" cannot say which place_plan call's P0 it means.
    tried = {entry["candidate_id"] for entry in entries if entry["tried_by"]}
    untried = sorted({entry["candidate_id"] for entry in entries} - tried)
    named = sorted(identity for identity in untried
                   if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(identity)}(?![A-Za-z0-9_-])", reason))
    return {
        "schema": SCHEMA,
        "methods": list(methods),
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
        "offered": [{key: value for key, value in entry.items() if key != "_poses"} for entry in entries],
        "untried_candidate_ids": untried,
        "named_untried_candidate_ids": named,
        "reason_names_untried": bool(named),
    }


def for_host(host: Any, reason: str) -> dict[str, Any]:
    """The ledger of the host's current round, read from its event log."""

    path = Path(host.event_path)
    methods = methods_for(host.surface)
    if not path.is_file():
        return ledger((), reason=reason, methods=methods)
    with path.open(encoding="utf-8") as stream:
        return ledger(stream, reason=reason, methods=methods)


__all__ = [
    "CANDIDATE_ID", "EXECUTE_TOOLS", "OFFER_TOOLS", "ORIENTATION_TOLERANCE_DEG", "POSE_MATCH",
    "POSITION_TOLERANCE_M", "SCHEMA", "for_host", "ledger", "methods_for",
]
