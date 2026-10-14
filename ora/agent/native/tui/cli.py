"""``ora``: ORA's terminal entry point. Only ``ora watch`` exists so far, and it is read-only."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from .follow import watch

FRONT_END = Path(__file__).resolve().parents[4] / "ora-tui"


def _run_root(value: str) -> Path:
    """A run directory; run IDs are not resolved here (the agent package stays below the orchestrator)."""
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{value} is not a run directory (pass outputs/<experiment>/runs/<run_id>)")
    return path.resolve()


def _node() -> str | None:
    """``ORA_NODE``, else ``node`` on PATH, else the newest nvm-installed node."""
    if os.environ.get("ORA_NODE"):
        return os.environ["ORA_NODE"]
    found = shutil.which("node")
    if found:
        return found
    candidates = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v*/bin/node")),
                        key=lambda path: [int(part) for part in Path(path).parts[-3].lstrip("v").split(".")])
    return candidates[-1] if candidates else None


def _pi_front_end() -> tuple[str, Path] | None:
    node = _node()
    if node and (FRONT_END / "node_modules" / "@earendil-works" / "pi-tui").is_dir():
        return node, FRONT_END / "index.mjs"
    return None


def _exec_pi(run_root: Path, args: argparse.Namespace, front_end: tuple[str, Path]) -> None:
    node, script = front_end
    passed = []
    for flag, value in (("--speed", args.speed), ("--max-gap", args.max_gap), ("--poll", args.poll)):
        if value is not None:
            passed += [flag, str(value)]
    for name in ("world_view", "world_port", "world_browser", "world_public_url"):
        value = getattr(args, name, None)
        if value is not None:
            passed += ["--" + name.replace("_", "-"), str(value)]
    environment = dict(os.environ)
    root = str(FRONT_END.parent)
    environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    os.execve(node, [node, str(script), str(run_root), "--python", sys.executable, *passed], environment)


def add_watch_options(parser: argparse.ArgumentParser) -> None:
    """The display options of ``ora watch``, shared with the top-level ``ora`` command."""

    parser.add_argument("--speed", type=float, default=None,
                        help="replay speed of a finished run (default 10 in a terminal, "
                             "0 = print at once, the default when not a terminal)")
    parser.add_argument("--max-gap", type=float, default=30.0,
                        help="longest recorded pause replayed, in seconds (default 30)")
    parser.add_argument("--plain", action="store_true", help="no colours and no live status region")
    parser.add_argument("--ui", choices=("auto", "pi", "ansi", "plain"), default="auto",
                        help="front end: pi (pi-tui, needs node and `npm ci` in ora-tui/), ansi "
                             "(built in), plain; auto picks pi in a terminal when installed")
    parser.add_argument("--emit", choices=("jsonl",), default=None,
                        help="write UI operations as JSON lines instead of drawing (the pi front end reads them)")
    parser.add_argument("--poll", type=float, default=0.2,
                        help="seconds between reads of a live run's events (default 0.2)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ora", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "watch", help="follow a live Native run, or replay a finished one, as ORA's transcript",
        description="Follow a live Native run, or replay a finished one, as ORA's terminal "
                    "transcript. Read-only: writes nothing under the run and takes no lease; "
                    "Ctrl-C stops watching, not the run.",
    )
    command.add_argument("run", help="run directory (outputs/<experiment>/runs/<run_id>)")
    add_watch_options(command)
    return parser


def main(argv: list[str] | None = None, *, parsed_args=None, observer_factory=None) -> int:
    args = parsed_args or build_parser().parse_args(argv)
    return watch_run(args.run, args, observer_factory=observer_factory)


def watch_run(run: str | Path, args: argparse.Namespace, *, observer_factory=None) -> int:
    """Watch one run directory with parsed ``add_watch_options`` arguments."""
    try:
        run_root = _run_root(str(run))
        if not (run_root / "agent-events.jsonl").is_file() and not (run_root / "plan.json").is_file():
            raise RuntimeError(f"{run_root} has neither agent-events.jsonl nor plan.json")
        ui = "plain" if args.plain else args.ui
        if args.emit is None and ui in ("auto", "pi"):
            front_end = _pi_front_end() if (ui == "pi" or sys.stdout.isatty()) else None
            if front_end:
                _exec_pi(run_root, args, front_end)
            elif ui == "pi":
                raise RuntimeError(f"the pi front end needs node and `npm ci` in {FRONT_END}")
        return watch(run_root, speed=args.speed, plain={"plain": True, "ansi": False}.get(ui),
                     emit=args.emit, poll=max(0.05, args.poll), max_gap=max(0.0, args.max_gap),
                     observer_factory=observer_factory)
    except BrokenPipeError:              # e.g. piped into head: stop quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
