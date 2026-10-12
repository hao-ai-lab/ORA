"""``python -m ora.agent.tools.composed {compile,check} [--root DIR | --tool DIR ...]``.

``compile`` regenerates every ``compiled/<arm_profile>.json``; ``check`` runs
the layer-1 static checks and exits nonzero on any finding. By default both
cover the harness library; ``--root`` names another library root with the
same ``<name>/<version>/tool.star`` layout (a practice agent's workspace) and
``--tool`` one or more tool directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import check_tool, known_tool_names
from .compiler import compile_tool
from .library import load_library, load_tool
from .tunables import load_tunables


def _tools(args: argparse.Namespace) -> list:
    if args.tool:
        return [load_tool(Path(item).resolve()) for item in args.tool]
    return list(load_library(Path(args.root).resolve() if args.root else None).values())


def _compile(args: argparse.Namespace) -> int:
    tunables = load_tunables()
    for tool in _tools(args):
        for arm_profile in tool.declaration.arm_profiles:
            document = compile_tool(tool.declaration, source_sha256=tool.program.digest, tunables=tunables,
                                    arm_profile=arm_profile)
            path = tool.compiled_path(arm_profile)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(tool.directory.parents[1])} {document['compiled_sha256']}")
    return 0


def _check(args: argparse.Namespace) -> int:
    tunables = load_tunables()
    names = known_tool_names()
    findings = [
        finding
        for tool in _tools(args)
        for finding in check_tool(tool, tunables=tunables, known_names=names)
    ]
    for finding in findings:
        print(finding)
    if not findings:
        print("OK composed tool library")
    return 1 if findings else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ora.agent.tools.composed")
    parser.add_argument("command", choices=("compile", "check"))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--root", help="a library root other than the harness library")
    scope.add_argument("--tool", nargs="+", help="tool directories (<root>/<name>/<version>)")
    args = parser.parse_args(argv)
    return _compile(args) if args.command == "compile" else _check(args)


if __name__ == "__main__":
    sys.exit(main())
