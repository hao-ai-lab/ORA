"""ORA terminal entry point: local CPU MuJoCo, read-only watch and interaction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ora.agent.native.tui import cli as watch_cli
from .ora_world import add_world_options, observer_factory


LOCAL_COMMANDS = {
    "local": "open the local CPU MuJoCo task picker (LIBERO on this computer)",
    "setup": "install the pinned local CPU simulator and terminal UI for `ora local`",
    "interactive": "choose a prepared benchmark RunSpec and steer ORA while it runs",
}


def route(values: list[str]) -> list[str] | None:
    """Arguments for ora_cli when the command belongs to the local/interactive front ends."""
    if values and values[0] in LOCAL_COMMANDS:
        return values
    if values[:1] == ["doctor"]:
        rest = values[1:]
        if "--env=local" in rest:
            rest.remove("--env=local")
            return ["doctor", *rest]
        if "--env" in rest and rest.index("--env") + 1 < len(rest) and rest[rest.index("--env") + 1] == "local":
            index = rest.index("--env")
            return ["doctor", *rest[:index], *rest[index + 2:]]
    return None


def add_forwarded_commands(sub) -> None:
    """List the forwarded commands in ``ora --help``; ``route`` hands them to ``main`` unparsed."""
    import argparse

    for name, text in LOCAL_COMMANDS.items():
        sub.add_parser(name, help=text, add_help=False).add_argument("args", nargs=argparse.REMAINDER)


def build_parser():
    parser = watch_cli.build_parser()
    parser.description = __doc__
    sub = next(action for action in parser._actions if action.dest == "command")
    add_world_options(sub.choices["watch"])
    from .ora_local import add_local_options
    local = sub.add_parser("local", help="open the local CPU MuJoCo task picker (the default command)")
    add_local_options(local)
    add_world_options(local)
    doctor = sub.add_parser("doctor", help="check local inputs without starting a simulator or calling an API")
    doctor.add_argument("--root", type=Path, help="local setup root")
    doctor.add_argument("--json", action="store_true", help="machine-readable readiness report")
    setup = sub.add_parser("setup", help="install the pinned local CPU simulator and terminal UI", add_help=False)
    setup.add_argument("args", nargs="...")
    interactive = sub.add_parser("interactive", help="choose a prepared benchmark task and interact with ORA")
    interactive.add_argument("--spec", type=Path, action="append", required=True,
                             help="prepared Native benchmark RunSpec; repeat to populate the task picker")
    interactive.add_argument("--confirm-launch", action="store_true",
                             help="authorize launching a selected task after its normal preflight passes")
    interactive.add_argument("--emit", choices=("jsonl",), help="JSON-lines RPC backend instead of the Pi UI")
    interactive.add_argument("--poll", type=float, default=0.2, help="event polling interval in seconds")
    add_world_options(interactive)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or (values[0].startswith("-") and values[0] not in {"-h", "--help"}):
        values.insert(0, "local")
    try:
        if values[0] == "setup":
            from .ora_local_setup import main as setup_main
            return setup_main(values[1:])
        args = build_parser().parse_args(values)
        if args.command == "doctor":
            import json
            from .ora_local import doctor, print_doctor
            result = doctor(args.root)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            else:
                print_doctor(result)
            return 0 if result["ready"] else 2
        explicit_world = any(value == "--world-port" or value.startswith("--world-port=")
                             or value == "--world-public-url" or value.startswith("--world-public-url=")
                             for value in values)
        if args.command == "local":
            from .ora_local import interactive_arguments
            # The local route prepares inputs only. The existing interactive
            # backend keeps manual task selection and every launch gate.
            return main(interactive_arguments(args, world_explicit=explicit_world))
        human = (sys.stdout.isatty() and args.emit is None) or os.environ.get("ORA_WORLD_UI") == "1"
        world = observer_factory(args, enabled=(human and not getattr(args, "plain", False)) or explicit_world)
        if args.command == "watch":
            return watch_cli.main(values, parsed_args=args, observer_factory=world)
        if args.emit is None:
            front_end = watch_cli._pi_front_end()
            if front_end is None:
                raise RuntimeError(f"interactive UI needs Node and npm ci --prefix {watch_cli.FRONT_END}; "
                                   "--emit jsonl exposes the RPC backend")
            node, script = front_end
            passed = ["--interactive", "--python", sys.executable, "--poll", str(args.poll)]
            for path in args.spec:
                passed += ["--spec", str(path.expanduser().resolve())]
            if args.confirm_launch:
                passed.append("--confirm-launch")
            passed += ["--world-view", args.world_view, "--world-port", str(args.world_port),
                       "--world-browser", args.world_browser]
            if args.world_public_url:
                passed += ["--world-public-url", args.world_public_url]
            environment = dict(os.environ)
            root = str(watch_cli.FRONT_END.parent)
            environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"]
                                               if environment.get("PYTHONPATH") else "")
            os.execve(node, [node, str(script), *passed], environment)
        from .ora_backend import serve
        return serve(args.spec, confirm_launch=args.confirm_launch, poll=args.poll, observer_factory=world)
    except ModuleNotFoundError as exc:
        print(f"ora: dependency {exc.name!r} is unavailable; run ./scripts/ora setup first", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
