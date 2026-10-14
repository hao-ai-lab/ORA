"""Condensed one-line views of tool calls, tool results and environment receipts.

The rules follow the replay renderer used for the ORA blog (``teaser/terminal/term.py``
on the ``blog`` branch): a call is printed as the model made it, a result is cut to
its key numbers, a receipt to its outcome. Every number comes from the run's own
artifacts. A condenser never raises: an unfamiliar or drifted shape yields fewer
lines, never a crash of the watcher.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

Segment = tuple[str, str]
"""``(text, style)``; styles are named in :mod:`.screen`."""

_ORDER = (
    "text", "camera", "view", "points_uv", "point_ids", "step", "target_ref", "candidate_id",
    "arm", "action", "ticks", "interaction", "position_mode", "position_frame", "position_m",
    "translation_m", "approach_world", "jaw_line_world", "orientation_mode", "orientation_xyzw",
    "max_candidates",
)


def fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    if isinstance(value, str):
        return f'"{value if len(value) <= 70 else value[:68] + "…"}"'
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {fmt(item)}" for key, item in value.items()) + "}"
    return json.dumps(value)


def v3(values: Any) -> str:
    return "(" + ", ".join(f"{float(x):.3f}" for x in list(values)[:3]) + ")"


def _kwargs(arguments: Mapping[str, Any]) -> str:
    keys = [key for key in _ORDER if key in arguments] + [key for key in arguments if key not in _ORDER]
    return ", ".join(f"{key}={fmt(arguments[key])}" for key in keys)


def call_lines(name: str, arguments: Mapping[str, Any]) -> list[list[Segment]]:
    """The call as the model made it; continuation lines for ``compute`` bodies."""
    head: list[Segment] = [(name, "bold")]
    try:
        if name == "compute":
            lines = [head + [(f"(outputs={fmt(arguments.get('outputs', []))})", "arg")]]
            for binding in arguments.get("bindings", []) or []:
                line: list[Segment] = [(f"    {binding['name']} = {fmt(binding['value'])}", "arg")]
                if binding.get("from"):
                    line.append((f"   # {binding['from']}", "faint"))
                lines.append(line)
            expressions = str(arguments.get("expressions", "")).replace("\\n", "\n")
            return lines + [[("    " + line, "arg")] for line in expressions.splitlines()]
        targets = arguments.get("targets")
        if isinstance(targets, dict):
            parts = [
                f"{arm}: {_kwargs(value)}" if isinstance(value, dict) else f"{arm}={fmt(value)}"
                for arm, value in targets.items()
            ]
            rest = {key: value for key, value in arguments.items() if key != "targets"}
            return [head + [("(" + "; ".join(parts) + (", " + _kwargs(rest) if rest else "") + ")", "arg")]]
        return [head + [(f"({_kwargs(arguments)})", "arg")]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return [head + [(f"({json.dumps(arguments, ensure_ascii=False)[:400]})", "arg")]]


def error_line(result: Mapping[str, Any]) -> str | None:
    error = result.get("error")
    if result.get("ok") is False or isinstance(error, Mapping):
        if isinstance(error, Mapping):
            code = error.get("code") or "error"
            message = error.get("message") or ""
            return f"{code}: {message}".rstrip(": ")
        return str(error or "tool call failed")
    return None


def _result_lines(name: str, result: Mapping[str, Any], mutation: bool) -> list[str]:
    out: list[str] = []
    if name in ("vision_ground", "vision_segment"):
        for candidate in (result.get("candidates") or [])[:4]:
            loc = candidate.get("localization") or {}
            centre = loc.get("center_xyz_m")
            height = loc.get("height_above_support_m")
            out.append(
                f'{candidate["candidate_id"]}  score {candidate.get("backend_score") or 0:.2f}'
                + (f"  centre {v3(centre)} m" if centre else "")
                + (f"  height {1000 * height:.0f} mm" if height is not None else "")
            )
    elif name == "grasp_plan":
        if not result.get("candidates"):
            status = result.get("status")
            out.append("no candidates" + (f" ({status})" if isinstance(status, str) and status else ""))
        for index, candidate in enumerate((result.get("candidates") or [])[:3]):
            ik = candidate.get("ik_reachability") or {}
            out.append(
                f'G{index}  score {candidate.get("backend_score", 0):.2f}  '
                f'pregrasp {v3(candidate["pregrasp_pose"]["position_m"])} m  '
                f'opening {1000 * candidate.get("gripper_opening_m", 0):.0f} mm  '
                + ("reachable" if ik.get("reachable") else "unreachable")
            )
    elif name == "mark_point":
        for mark in (result.get("marks") or [])[:6]:
            out.append(
                f'{mark.get("point_id")}  '
                + (f'{v3(mark["xyz_m"])} m' if mark.get("xyz_m") else (mark.get("error_code") or mark.get("status", "")))
            )
    elif name == "compute":
        for output in result.get("outputs") or []:
            out.append(f'{output["name"]} = {fmt(output["value"])}')
    elif name == "grasp_execute":
        sequence = result.get("committed_sequence")
        if sequence:
            out.append("committed: " + " → ".join(sequence))
    elif name == "geometry_query":
        points = [point for point in result.get("points") or [] if isinstance(point, Mapping)]
        for point in points:          # the transcript keeps four lines and counts the rest
            where = fmt(point.get("pixel_uv")) if point.get("pixel_uv") is not None else "?"
            if point.get("valid") and point.get("world_xyz_m"):
                depth = point.get("depth_median_m")
                out.append(f"{where} → {v3(point['world_xyz_m'])} m" + (f"  depth {depth:.3f} m" if depth is not None else ""))
            else:
                out.append(f"{where} → invalid")
    elif not mutation:
        scalars = [
            f"{key} {fmt(value)}" for key, value in result.items()
            if not key.startswith("_") and isinstance(value, (bool, int, float, str))
            and not (isinstance(value, str) and len(value) > 80)
            and not _is_identity(key, value)
        ]
        if scalars:
            out.append(" · ".join(scalars[:4]))
    return out


_HEX_DIGEST = re.compile(r"[0-9a-f]{32,}\Z")


def _is_identity(key: str, value: Any) -> bool:
    """Hashes and digests identify evidence; they carry nothing to read in a transcript line."""
    return "sha256" in key or key.endswith("_hash") or (isinstance(value, str) and _HEX_DIGEST.match(value) is not None)


def result_lines(name: str, result: Mapping[str, Any] | None, *, mutation: bool = False) -> list[str]:
    if not isinstance(result, Mapping):
        return []
    error = error_line(result)
    if error:
        return ["error " + error]
    try:
        return _result_lines(name, result, mutation)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return []


def _receipt_lines(receipt: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[str]:
    rc = receipt.get("receipt", receipt)
    end = " · round ended" if rc.get("terminal_after_action") else ""
    skill = rc.get("skill_receipt")
    if skill:
        verification = skill.get("verification") or {}
        text = f'{skill.get("status", "")} ({str(skill.get("stop_reason", "")).replace("_", " ")})'
        if (verification.get("jaw_hold") or {}).get("held"):
            text += f' · jaw hold: {verification["jaw_hold"]["held"]}'
        lift = (verification.get("lift_probe") or {}).get("measured_tcp_lift_m")
        if lift is not None:
            text += f" · lifted {1000 * lift:.0f} mm"
        return [text + end]
    if rc.get("wait"):
        return [f'waited {rc["wait"].get("ticks_used")} ticks' + end]
    targets = arguments.get("targets") if isinstance(arguments.get("targets"), dict) else None
    arm = next(iter(targets)) if targets else "right"
    arms = rc.get("arms") or {}
    gripper = rc.get("gripper_outcome") or (arms.get(arm) or {}).get("gripper_outcome")
    if gripper:
        text = f'gripper {str(gripper["state"]).replace("_", " ")}'
        if gripper.get("measured_aperture_mm") is not None:
            text += f' at {gripper["measured_aperture_mm"]:.1f} mm'
        if (rc.get("feedback") or {}).get("contact_estimated"):
            text += " · contact"
        if (rc.get("endpoint_contract") or {}).get("stop_reason") == "environment_terminated":
            text += " · environment terminated"
        return [text + end]
    after = ((rc.get("after") or {}).get(arm) or {}).get("pose_xyz_xyzw")
    error = ((arms.get(arm) or {}).get("pose_error") or {}).get("translation_m")
    text = str(rc.get("execution_status", "done")).replace("_", " ")
    if rc.get("failure_stage"):
        text += f' ({str(rc["failure_stage"]).replace("_", " ")})'
    if after:
        text += f"  end effector {v3(after)} m"
    if error is not None:
        text += f"  error {1000 * error:.1f} mm"
    return [text + end]


def receipt_lines(receipt: Mapping[str, Any] | None, arguments: Mapping[str, Any]) -> list[str]:
    if not isinstance(receipt, Mapping):
        return []
    try:
        return _receipt_lines(receipt, arguments)
    except (KeyError, TypeError, ValueError, AttributeError, StopIteration):
        return []


def tokens(count: int | float | None) -> str:
    if not count:
        return "0"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    return f"{count / 1000:.1f}k" if count >= 1000 else str(int(count))


def visible_text(message: Mapping[str, Any] | None) -> str:
    """The assistant's visible words; ``reasoning_content`` is never read."""
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    parts = [
        str(part.get("text", "")) for part in content or []
        if isinstance(part, Mapping) and part.get("type", "text") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()
