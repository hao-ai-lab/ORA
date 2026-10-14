"""Fold a Native run's ``agent-events.jsonl`` into ORA's transcript operations.

:class:`Transcript` consumes canonical agent events one at a time and returns UI
operations (``{"op": "assistant", ...}``, ``{"op": "tool_call", ...}``, …), the same
stream ``ora watch --emit jsonl`` writes for the pi-tui front end. Between events it
exposes :attr:`Transcript.status`, the one thing happening now (a model call, a
tool, the robot executing, a context compaction) with the recorded unix time it
began, and :meth:`Transcript.footer_state`. :class:`TextRenderer` turns operations
into styled lines for the built-in ANSI and plain renderers. Artifacts are read
through :func:`read_artifact`, which opens files inside the run directory only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .condense import Segment, call_lines, receipt_lines, result_lines, tokens, visible_text

Line = list[Segment]
Op = dict[str, Any]
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
TERMINAL_STATES = frozenset({"completed", "incomplete", "invalid", "blocked"})
_ACT_VERB = {
    "move_pose": "Moving", "move_to": "Moving", "set_gripper": "Gripping", "grasp_execute": "Grasping",
    "recover_pose": "Homing", "wait_ticks": "Waiting", "rotate_wrist": "Rotating", "rotate_pitch": "Rotating",
}
BENCHMARK_NAMES = {"robodojo": "RoboDojo", "libero_pro": "LIBERO-Pro", "libero": "LIBERO", "robotwin": "RoboTwin"}


def read_artifact(run_root: Path, ref: Mapping[str, Any] | None) -> Any:
    """Parse one JSON artifact under ``run_root``; ``None`` when absent, oversized or outside the run."""
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        return None
    root = Path(os.path.abspath(run_root))
    path = Path(os.path.normpath(root / ref["path"]))   # lexical: no "..", symlinked storage allowed
    if not path.is_relative_to(root) or path.suffix != ".json":
        return None
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            return None
        with path.open("rb") as stream:
            return json.loads(stream.read())
    except (OSError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = json.loads(stream.read())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass
class Status:
    verb: str
    started_unix: float


@dataclass
class RunInfo:
    """What the header and footer show, from ``plan.json`` and ``session_started``."""

    run_id: str
    benchmark: str
    task: str
    instruction: str
    rounds: int
    experiment: str = ""

    @classmethod
    def from_run(cls, run_root: Path) -> "RunInfo":
        plan = read_json(run_root / "plan.json")
        runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
        session = plan.get("session") if isinstance(plan.get("session"), Mapping) else {}
        kind = str(runtime.get("kind") or "")
        return cls(
            run_id=run_root.name,
            benchmark=BENCHMARK_NAMES.get(kind, kind or "environment"),
            task=str(plan.get("task") or ""),
            instruction=str(plan.get("task_instruction") or ""),
            rounds=int(session.get("rounds") or 1),
            experiment=str(plan.get("experiment") or ""),
        )


class Transcript:
    """Event-by-event transcript state; :meth:`feed` returns UI operations."""

    def __init__(self, run_root: Path, info: RunInfo | None = None) -> None:
        self.run_root = Path(run_root)
        self.info = info or RunInfo.from_run(self.run_root)
        self.session: dict[str, Any] = {}
        self.status: Status | None = None
        self.round = 0
        self.context_tokens = 0
        self.input_total = 0
        self.output_total = 0
        self.last_call: tuple[float, int] | None = None   # (latency s, output tokens)
        self.finished = False
        self.ended: str | None = None
        self._calls: dict[str, dict[str, Any]] = {}
        self._actions: dict[str, dict[str, Any]] = {}
        self._last_call_id: str | None = None
        self._request_unix: float | None = None
        self._images: tuple[str, int] | None = None     # (tool_call_id, count) awaiting flush

    # -- public -----------------------------------------------------------------

    def feed(self, event: Mapping[str, Any]) -> list[Op]:
        kind = str(event.get("type", ""))
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        unix = float(event.get("unix_s") or 0.0)
        out: list[Op] = []
        if kind != "tool_image":
            out += self.flush()
        handler = getattr(self, "_on_" + kind, None)
        if handler is not None:
            out += handler(event, payload, unix) or []
        return out

    def flush(self) -> list[Op]:
        """Operations held back to merge consecutive events (returned images)."""
        if self._images is None:
            return []
        call_id, count = self._images
        self._images = None
        return [{"op": "tool_images", "id": call_id, "count": count, **self._named(call_id)}]

    def header(self, mode: str, version: str) -> Op:
        info, session = self.info, self.session
        return {
            "op": "header", "version": version, "mode": mode, "run_id": info.run_id,
            "run_root": str(self.run_root), "benchmark": info.benchmark, "task": info.task,
            "experiment": info.experiment, "rounds": info.rounds,
            "model": str(session.get("model") or "").split("/")[-1],
            "reasoning_effort": session.get("reasoning_effort"),
            "provider_protocol": session.get("provider_protocol"),
        }

    def user(self) -> Op:
        return {"op": "user", "text": self.info.instruction or "(no task instruction in plan.json)"}

    def footer_state(self) -> Op:
        return {
            "op": "footer", "round": min(self.round + 1, self.info.rounds), "rounds": self.info.rounds,
            "context_tokens": self.context_tokens, "context_window": self.session.get("context_window_tokens"),
            "input_total": self.input_total, "output_total": self.output_total,
            "last_latency_s": round(self.last_call[0], 1) if self.last_call else None,
            "last_output_tokens": self.last_call[1] if self.last_call else None,
            "model": str(self.session.get("model") or "").split("/")[-1],
            "reasoning_effort": self.session.get("reasoning_effort"),
        }

    def footer(self) -> tuple[str, str]:
        state = self.footer_state()
        left = f'round {state["round"]}/{state["rounds"]}'
        window = state["context_window"]
        if self.context_tokens:
            left += f" · context {tokens(self.context_tokens)}" + (f" / {tokens(window)}" if window else "")
        if self.last_call:
            latency, output = self.last_call
            left += f" · last call {latency:.0f}s ↓ {tokens(output)}"
        effort = state["reasoning_effort"]
        right = " · ".join(str(part) for part in (state["model"], f"{effort} effort" if effort else "") if part)
        return left, right

    # -- event handlers -----------------------------------------------------------

    def _on_operator_message(self, event, payload, unix):
        text = payload.get("message")
        if isinstance(text, str) and text:
            return [{"op": "user", "text": text, "operator": True, "id": payload.get("id")}]
        return []

    def _on_session_started(self, event, payload, unix):
        self.session = dict(payload)

    def _on_provider_request(self, event, payload, unix):
        self._request_unix = unix
        self.status = Status("Thinking", unix)

    def _on_provider_response(self, event, payload, unix):
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        output = int(usage.get("output_tokens") or 0)
        self.context_tokens = int(usage.get("input_tokens") or 0) or self.context_tokens
        self.input_total += int(usage.get("input_tokens") or 0)
        self.output_total += output
        if self._request_unix is not None:
            self.last_call = (max(0.0, unix - self._request_unix), output)
        self._request_unix = None
        self.status = None
        text = visible_text(payload.get("message") if isinstance(payload.get("message"), Mapping) else None)
        return [{"op": "assistant", "text": text}] if text else []

    def _on_provider_output_truncated(self, event, payload, unix):
        return [{"op": "notice", "level": "warning", "mark": "⚠",
                 "text": "model output hit the output-token limit; the turn continues"}]

    def _on_tool_call_proposed(self, event, payload, unix):
        call_id = str(event.get("tool_call_id") or len(self._calls))
        name = str(payload.get("name", "?"))
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), Mapping) else {}
        mutation = payload.get("effect") == "mutation"
        self._calls[call_id] = {"name": name, "arguments": arguments, "mutation": mutation, "call_id": call_id}
        self._last_call_id = call_id
        self.status = Status(f"Running {name}", unix)
        lines = call_lines(name, arguments)
        return [{
            "op": "tool_call", "id": call_id, "name": name, "mutation": mutation,
            "args": "".join(text for text, _ in lines[0][1:]),
            "body": ["".join(text for text, _ in line) for line in lines[1:]],
            "comments": ["".join(text for text, style in line if style == "faint") for line in lines[1:]],
        }]

    def _on_tool_result(self, event, payload, unix):
        call_id = str(event.get("tool_call_id") or self._last_call_id)
        call = self._calls.get(call_id, {"name": payload.get("name", "?"), "arguments": {}, "mutation": False})
        result = read_artifact(self.run_root, event.get("artifact_ref"))
        lines = result_lines(call["name"], result, mutation=call["mutation"])
        if result is None and payload.get("ok") is False:
            lines = ["error (result artifact unavailable)"]
        if not call["mutation"]:
            self.status = None
        more = len(lines) - 4
        lines = lines[:4] + ([f"… +{more} more"] if more > 0 else [])
        error = bool(lines) and lines[0].startswith("error ")
        done = not call["mutation"] or error
        return [{"op": "tool_result", "id": call_id, "lines": lines, "error": error, "done": done,
                 **self._named(call_id)}]

    def _on_tool_image(self, event, payload, unix):
        call_id = str(event.get("tool_call_id") or self._last_call_id)
        if self._images and self._images[0] == call_id:
            self._images = (call_id, self._images[1] + 1)
            return []
        out = self.flush()
        self._images = (call_id, 1)
        return out

    def _on_action_proposed(self, event, payload, unix):
        call_id = str(event.get("tool_call_id") or self._last_call_id)
        call = self._calls.get(call_id) or {"name": "action", "arguments": {}}
        self._actions[str(payload.get("action_id"))] = call
        self.status = Status(f'{_ACT_VERB.get(call["name"], "Acting")} on the robot', unix)

    def _on_environment_receipt(self, event, payload, unix):
        call = self._actions.get(str(payload.get("action_id")), {"name": "action", "arguments": {}})
        self.status = Status("Observing", unix)
        receipt = read_artifact(self.run_root, event.get("artifact_ref"))
        return [{"op": "receipt", "id": call.get("call_id"), "lines": receipt_lines(receipt, call["arguments"])}]

    def _on_context_compactor_attempt(self, event, payload, unix):
        self.status = Status("Compacting context", unix)

    def _on_context_compactor_response(self, event, payload, unix):
        self.status = None
        latency = float(payload.get("latency_s") or 0.0)
        return [{"op": "notice", "level": "dim", "mark": "✻", "text": f"Context compacted · {latency:.0f} s"}]

    def _on_context_compactor_failed(self, event, payload, unix):
        self.status = None
        return [{"op": "notice", "level": "warning", "mark": "✻",
                 "text": f'Context compaction failed: {payload.get("error", "")}'}]

    def _on_official_verdict(self, event, payload, unix):
        self.round += 1
        self.status = None
        out = [{"op": "verdict", "round": self.round, "rounds": self.info.rounds,
                "success": payload.get("official_success"), "benchmark": self.info.benchmark}]
        if self.round < self.info.rounds:
            out.append({"op": "rule", "text": f"round {self.round + 1}/{self.info.rounds}: the world is reset"})
        return out

    def _on_run_terminated(self, event, payload, unix):
        self.status = None
        state = payload.get("state", "")
        termination = str(payload.get("termination", "")).replace("_", " ")
        self.ended = f"{state} · {termination}".strip(" ·")
        return [{"op": "rule", "text": f"run {self.ended}"}]

    def _on_session_closed(self, event, payload, unix):
        self.status = None
        self.finished = True

    # -- helpers ----------------------------------------------------------------

    def _named(self, call_id: str) -> dict[str, str]:
        """Name the call when its result does not follow it directly (a batched proposal)."""
        if call_id == self._last_call_id:
            return {}
        return {"of": str(self._calls.get(call_id, {}).get("name", "?"))}


class TextRenderer:
    """Styled lines for the built-in ANSI and plain renderers."""

    def __init__(self) -> None:
        self._answered: set[str | None] = set()

    def lines(self, op: Op) -> list[Line]:
        kind = op.get("op")
        if kind == "user":
            return [[("❯ ", "user_dim"), (op["text"], "user")]]
        if kind == "assistant":
            return [[("● ", "dot"), (" ".join(op["text"].split()), "txt")]]
        if kind == "notice":
            if op["mark"] == "⚠":
                return [[("  ⚠ ", "warn"), (op["text"], "dim")]]
            return [[(op["mark"] + " ", "warn" if op["level"] == "warning" else "dim"), (op["text"], "dim")]]
        if kind == "tool_call":
            first: Line = [("● ", "call_dot"), (op["name"], "bold"), (op["args"], "arg")]
            body = []
            for text, comment in zip(op["body"], op["comments"]):
                code = text[: len(text) - len(comment)] if comment else text
                body.append([("  ", "arg"), (code, "arg")] + ([(comment, "faint")] if comment else []))
            return [first] + body
        if kind == "tool_result":
            texts = list(op["lines"])
            if texts and op.get("of"):
                texts[0] = f'{op["of"]}: {texts[0]}'
            return self._sub(op["id"], texts, "err" if op["error"] else "res")
        if kind == "tool_images":
            count = op["count"]
            text = (f'{op["of"]}: ' if op.get("of") else "") + f'{count} image{"s" if count > 1 else ""} to the model'
            return self._sub(op["id"], [text])
        if kind == "receipt":
            return self._sub(op["id"], op["lines"], label="receipt  ")
        if kind == "verdict":
            success = op["success"]
            label = f'Round {op["round"]}/{op["rounds"]} · ' if op["rounds"] > 1 else ""
            verdict = {True: "success", False: "failure"}.get(success, "none")
            return [[("● ", "ok_dot" if success else "dot"),
                     (f'{label}official verdict ({op["benchmark"]} evaluator): ', "txt"),
                     (verdict, "ok" if success else "txt")]]
        if kind == "rule":
            return [[], [("── ", "faint"), (op["text"], "dim"), (" ──", "faint")]]
        return []

    def _sub(self, call_id: str | None, texts: list[str], style: str = "res", label: str = "") -> list[Line]:
        """Lines indented under a call; only the first line under that call carries the corner."""
        lines: list[Line] = []
        for text in texts:
            first = call_id not in self._answered
            self._answered.add(call_id)
            lead: Line = [("  ⎿  " if first else "     ", "faint")]
            if label:
                lead.append((label, "dim"))
            lines.append(lead + [(text, style)])
        return lines
