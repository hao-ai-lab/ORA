"""Draw ORA's transcript in a terminal with ANSI escapes only (no TUI dependency).

The layout follows terminal coding agents: transcript lines are written once and
scroll into the terminal's own scrollback; below them a small live region (spinner
status line, prompt box, footer) is erased and redrawn in place. In plain mode
(``--plain`` or a non-terminal stream) styles and the live region are dropped and
only transcript lines are written, followed by one closing footer line.
"""

from __future__ import annotations

import shutil
from typing import TextIO

from .condense import Segment

Line = list[Segment]

ORA_RED = "\x1b[38;2;204;98;84m"
STYLES = {
    "txt": "\x1b[38;5;253m",
    "dim": "\x1b[38;5;245m",
    "faint": "\x1b[38;5;240m",
    "bold": "\x1b[1;97m",
    "arg": "\x1b[38;5;249m",
    "res": "\x1b[38;5;247m",
    "ora": ORA_RED,
    "ok": "\x1b[38;2;127;211;160m",
    "ok_dot": "\x1b[38;2;127;211;160m",
    "warn": "\x1b[33m",
    "err": "\x1b[38;2;224;120;104m",
    "dot": "\x1b[97m",
    "call_dot": "\x1b[38;5;247m",
    "user": "\x1b[97;48;5;236m",
    "user_dim": "\x1b[38;5;245;48;5;236m",
}
RESET = "\x1b[0m"
SPINNER = "·✢✳✶✻✽✻✶✳✢"
# ORA's mark: a small robot head on a neck with a gripper either side (10 x 7 cells).
LOGO = (
    "....##....",
    "..######..",
    "..#.##.#..",
    "..######..",
    "....##....",
    ".##.##.##.",
    ".#......#.",
)


def logo_rows() -> list[str]:
    """The mark in half blocks: two pixel rows per text row."""
    rows = list(LOGO) + ["." * len(LOGO[0])] * (len(LOGO) % 2)
    out = []
    for top, bottom in zip(rows[::2], rows[1::2]):
        out.append("".join(
            {("#", "#"): "█", ("#", "."): "▀", (".", "#"): "▄"}.get((a, b), " ")
            for a, b in zip(top, bottom)
        ))
    return out


def wrap(segments: Line, width: int, indent: int) -> list[Line]:
    """Wrap styled segments to ``width`` columns; continuation lines start with ``indent`` spaces."""
    width = max(width, indent + 8)
    lines: list[Line] = []
    current: Line = []
    used = 0
    for text, style in segments:
        index = 0
        while index < len(text):
            room = width - used
            if room <= 0:
                lines.append(current)
                current, used = [(" " * indent, style)], indent
                room = width - used
            chunk = text[index:index + room]
            soft = False
            if len(text) - index > room:
                cut = max(chunk.rfind(" "), chunk.rfind(","))
                if cut > room * 0.2:
                    chunk, soft = chunk[:cut + 1], True
            current.append((chunk, style))
            used += len(chunk)
            index += len(chunk)
            if soft:
                used = width
    lines.append(current)
    return lines


class Screen:
    def __init__(self, stream: TextIO, *, ansi: bool, width: int | None = None) -> None:
        self.stream = stream
        self.ansi = ansi
        self.fixed_width = width
        self._live_rows = 0

    @property
    def width(self) -> int:
        if self.fixed_width:
            return self.fixed_width
        return max(40, shutil.get_terminal_size((100, 30)).columns)

    def _paint(self, line: Line, fill: str | None = None) -> str:
        if not self.ansi:
            return "".join(text for text, _ in line).rstrip()
        body = "".join(STYLES.get(style, "") + text + RESET for text, style in line)
        if fill:                      # a background that runs to the right edge
            used = sum(len(text) for text, _ in line)
            body += STYLES[fill] + " " * max(0, self.width - used) + RESET
        return body

    def _erase_live(self) -> None:
        if self.ansi and self._live_rows:
            self.stream.write(f"\r\x1b[{self._live_rows}A\x1b[J")
            self._live_rows = 0

    def append(self, lines: list[Line], *, wrap_indent: int = 2) -> None:
        """Write transcript lines above the live region."""
        if not lines:
            return
        self._erase_live()
        width = self.width - 1
        for line in lines:
            user = bool(line) and line[0][1] in ("user", "user_dim")
            indent = wrap_indent
            if line and len(line[0][0]) <= 6 and line[0][0].strip() in ("", "●", "⎿", "⚠", "✻", "❯"):
                indent = len(line[0][0])        # continuation rows line up under the text
            for row in wrap(line, width, indent) if line else [[]]:
                self.stream.write(self._paint(row, "user" if user else None) + "\n")
        self.stream.flush()

    def live(self, status: str | None, glyph: str, left: str, right: str, prompt: str) -> None:
        """Redraw the spinner line, prompt box and footer below the transcript."""
        if not self.ansi:
            return
        self._erase_live()
        width = self.width - 1
        rule = STYLES["faint"] + "─" * width + RESET
        rows = [""]
        if status:
            rows.append(ORA_RED + glyph + RESET + " " + STYLES["txt"] + status[: width - 2] + RESET)
        rows.append(rule)
        rows.append(STYLES["bold"] + "❯ " + RESET + STYLES["faint"] + prompt[: width - 2] + RESET)
        rows.append(rule)
        rest = " · " + left
        footer = ORA_RED + "▸▸ " + RESET + STYLES["txt"] + "robot mode on" + RESET + STYLES["dim"] + rest[: width - 16] + RESET
        pad = width - 16 - len(rest) - len(right)
        if right and pad >= 2:
            footer += " " * pad + STYLES["dim"] + right + RESET
        rows.append(footer)
        self.stream.write("\n".join(rows) + "\n")
        self.stream.flush()
        self._live_rows = len(rows)

    def close(self, left: str, right: str) -> None:
        """Leave the final footer as plain scrollback."""
        self._erase_live()
        text = "▸▸ " + left + (f" · {right}" if right else "")
        self.stream.write((STYLES["dim"] + text + RESET if self.ansi else text) + "\n")
        if self.ansi:
            self.stream.write("\x1b[?25h")
        self.stream.flush()

    def hide_cursor(self) -> None:
        if self.ansi:
            self.stream.write("\x1b[?25l")
            self.stream.flush()
