"""Follow a live Native run, or replay a finished one, as ORA's terminal transcript.

Read-only by construction: the watcher opens ``agent-events.jsonl``, ``plan.json``,
``status.json`` and referenced artifacts for reading only, never writes under the
run directory, never takes or reads Runtime leases and never signals a process.
A live run is polled (``poll`` seconds, default 0.2) so a new event appears well
under a second after the Runtime appends it. A finished run (terminal
``status.json`` or a ``session_closed`` event) replays with its recorded timing
sped up by ``speed``; gaps longer than ``max_gap`` recorded seconds are shortened
to ``max_gap``. ``speed=0`` prints the whole transcript at once.

Output goes to a sink: :class:`TerminalSink` draws the built-in ANSI or plain
transcript; :class:`JsonSink` writes the same UI operations as JSON lines for the
pi-tui front end (``ora-tui/``), which draws them the way Pi's interactive mode does.
"""

from __future__ import annotations

import json
import re
import sys
import time
from bisect import bisect_right
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .screen import SPINNER, Screen, logo_rows
from .transcript import TERMINAL_STATES, Line, Op, RunInfo, TextRenderer, Transcript, read_json

EVENTS = "agent-events.jsonl"
BLOCK_MARKS = frozenset({"●", "✻", "──", "❯"})


class EventTail:
    """Incremental reader of complete JSON lines appended to one file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._partial = b""

    def read(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        data = self._partial + data
        *complete, self._partial = data.split(b"\n")
        events = []
        for raw in complete:
            if raw.strip():
                try:
                    events.append(json.loads(raw))
                except ValueError:
                    continue          # a torn line is never fatal to a read-only watcher
        return events


class ReplayClock:
    """Map wall seconds since replay start to recorded unix time, compressing long gaps."""

    def __init__(self, recorded: list[float], speed: float, max_gap: float) -> None:
        self.recorded = recorded
        self.due = [0.0]
        for previous, current in zip(recorded, recorded[1:]):
            gap = min(max(0.0, current - previous), max_gap)
            self.due.append(self.due[-1] + (gap / speed if speed > 0 else 0.0))

    def recorded_at(self, wall: float) -> float:
        if not self.recorded:
            return 0.0
        index = bisect_right(self.due, wall) - 1
        if index < 0:
            return self.recorded[0]
        if index >= len(self.recorded) - 1:
            return self.recorded[-1]
        span = self.due[index + 1] - self.due[index]
        fraction = (wall - self.due[index]) / span if span > 0 else 1.0
        return self.recorded[index] + fraction * (self.recorded[index + 1] - self.recorded[index])


def run_is_finished(run_root: Path) -> bool:
    return read_json(run_root / "status.json").get("state") in TERMINAL_STATES


def _version() -> str:
    try:
        return "v" + metadata.version("ora-agent")
    except metadata.PackageNotFoundError:     # a source checkout imported through PYTHONPATH
        try:
            text = (Path(__file__).resolve().parents[4] / "pyproject.toml").read_text(encoding="utf-8")
            match = re.search(r'^version = "([^"]+)"', text, flags=re.MULTILINE)
            return "v" + match.group(1) if match else "dev"
        except OSError:
            return "dev"


def header_lines(op: Op) -> list[Line]:
    model = op["model"] or "model unknown"
    effort = op["reasoning_effort"]
    details: list[Line] = [
        [("ORA", "bold"), (f' {op["version"]}', "dim"), ("  open robot-use agent", "faint")],
        [(" · ".join(part for part in (model, f"{effort} effort" if effort else "",
                                       str(op["provider_protocol"] or "")) if part), "dim")],
        [(" · ".join(part for part in (f'{op["benchmark"]} / {op["task"]}' if op["task"] else op["benchmark"],
                                       op["experiment"]) if part), "dim")],
        [(f'{op["run_id"]} · {op["mode"]}', "faint")],
    ]
    logo = logo_rows()
    lines: list[Line] = []
    for index in range(max(len(logo), len(details))):
        mark = logo[index] if index < len(logo) else " " * len(logo[0])
        lines.append([(" " + mark + "   ", "ora")] + (details[index] if index < len(details) else []))
    return lines


class TerminalSink:
    """The built-in renderer: transcript lines plus, in a terminal, a live status region."""

    def __init__(self, stream: TextIO, *, ansi: bool, width: int | None) -> None:
        self.screen = Screen(stream, ansi=ansi, width=width)
        self.renderer = TextRenderer()
        self.blank = True
        self.tick = 0
        self.replay = False
        self.world_status = ""
        self.browser_message = ""

    def start(self, header: Op, user: Op) -> None:
        self.replay = str(header["mode"]).startswith("replay")
        self.screen.hide_cursor()
        self.screen.append(header_lines(header) + [[]] + self.renderer.lines(user) + [[]])

    def ops(self, ops: list[Op]) -> None:
        spaced: list[Line] = []
        for op in ops:
            if op.get("op") == "world_view":
                state = "connected" if op.get("connected") else op.get("phase", "waiting")
                self.world_status = f"World · {state} · {op.get('url') or op.get('message') or ''}"
                if op.get("browser_message") and op["browser_message"] != self.browser_message:
                    self.browser_message = op["browser_message"]
                    self.screen.append([[(str(self.browser_message), "dim")]])
                if not self.screen.ansi:
                    self.screen.append([[(self.world_status, "dim")]])
                continue
            for line in self.renderer.lines(op):
                if line and line[0][0].strip() in BLOCK_MARKS and not self.blank:
                    spaced.append([])        # a blank line before each message, call and verdict
                spaced.append(line)
                self.blank = not line
        self.screen.append(spaced)

    def live(self, status: str | None, transcript: Transcript) -> None:
        left, right = transcript.footer()
        self.screen.live(status, SPINNER[self.tick % len(SPINNER)], left, right,
                         self.world_status or "read-only: watching this run" + (" (replay)" if self.replay else ""))
        self.tick += 1

    def close(self, transcript: Transcript) -> None:
        self.screen.close(*transcript.footer())


class JsonSink:
    """UI operations as JSON lines; status and footer only when they change."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self._status: Any = ()
        self._footer: Any = None

    def _write(self, op: Op) -> None:
        self.stream.write(json.dumps(op, ensure_ascii=False) + "\n")
        self.stream.flush()

    def start(self, header: Op, user: Op) -> None:
        self._write(header)
        self._write(user)

    def ops(self, ops: list[Op]) -> None:
        for op in ops:
            self._write(op)

    def live(self, status: str | None, transcript: Transcript) -> None:
        if status != self._status:
            self._status = status
            self._write({"op": "status", "text": status})
        footer = transcript.footer_state()
        if footer != self._footer:
            self._footer = footer
            self._write(footer)

    def close(self, transcript: Transcript) -> None:
        self.live(None, transcript)
        self._write({"op": "end", "ended": transcript.ended})


def watch(
    run_root: Path,
    *,
    stream: TextIO | None = None,
    speed: float | None = None,
    plain: bool | None = None,
    emit: str | None = None,
    poll: float = 0.2,
    max_gap: float = 30.0,
    width: int | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    observer_factory=None,
) -> int:
    """Render one run until its session closes (live) or its events end (replay).

    ``emit="jsonl"`` writes UI operations instead of drawing; ``speed`` then defaults to 10.
    """
    run_root = Path(run_root)
    stream = stream or sys.stdout
    if emit == "jsonl":
        sink: TerminalSink | JsonSink = JsonSink(stream)
        interactive = True
    else:
        interactive = (not plain) if plain is not None else stream.isatty()
        sink = TerminalSink(stream, ansi=interactive, width=width)
    transcript = Transcript(run_root, RunInfo.from_run(run_root))
    tail = EventTail(run_root / EVENTS)
    backlog = tail.read()
    finished = run_is_finished(run_root) or any(event.get("type") == "session_closed" for event in backlog)
    if speed is None:
        speed = 10.0 if interactive else 0.0
    if not finished:
        mode = "live"
    elif speed > 0:
        mode = f"replay ×{speed:g}"
    else:
        mode = "sealed run" if (run_root / "seal.json").is_file() else "finished run"
    started = False
    observer = observer_factory(root=run_root, finished=finished, emit=lambda op: sink.ops([op])) if observer_factory else None
    if observer is not None:
        observer.poll()

    def start(first: Mapping[str, Any] | None) -> None:
        nonlocal started
        if first is not None and first.get("type") == "session_started":
            transcript.feed(first)
        sink.start(transcript.header(mode, _version()), transcript.user())
        started = True

    def emit_event(event: Mapping[str, Any]) -> None:
        if not started:
            start(event)
            if event.get("type") == "session_started":
                return
        sink.ops(transcript.feed(event))

    def draw(recorded_now: float) -> None:
        if observer is not None:
            observer.poll()
        status = transcript.status
        text = None
        if status is not None:
            elapsed = max(0.0, recorded_now - status.started_unix)
            text = f"{status.verb}… ({elapsed:.0f}s · ctrl+c stops watching, not the run)"
        sink.live(text, transcript)

    try:
        if finished:
            clock = ReplayClock([float(event.get("unix_s") or 0.0) for event in backlog], speed, max_gap)
            begin = now()
            for index, event in enumerate(backlog):
                while speed > 0 and now() - begin < clock.due[index]:
                    draw(clock.recorded_at(now() - begin))
                    sleep(min(0.1, max(0.0, clock.due[index] - (now() - begin))))
                emit_event(event)
        else:
            for event in backlog:
                emit_event(event)
            idle, last_status_check = 0.0, now()
            while not transcript.finished:
                events = tail.read()
                for event in events:
                    emit_event(event)
                if events:
                    idle = 0.0
                else:
                    idle += poll
                    if idle >= 0.5:
                        sink.ops(transcript.flush())
                    if now() - last_status_check >= 2.0:
                        last_status_check = now()
                        if run_is_finished(run_root):
                            for event in tail.read():   # the last lines before the state flipped
                                emit_event(event)
                            break
                if started or observer is not None:
                    draw(now())
                sleep(poll)
        if not started:
            start(None)
        sink.ops(transcript.flush())
    except KeyboardInterrupt:
        sink.close(transcript)
        return 130
    finally:
        if observer is not None:
            observer.close()
    sink.close(transcript)
    return 0


__all__ = ["EventTail", "JsonSink", "ReplayClock", "TerminalSink", "header_lines", "run_is_finished", "watch"]
