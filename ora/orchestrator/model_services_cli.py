"""Operate the resident SAM3 and Contact-GraspNet services from the shared workspace.

`start` reads the `B2_SAM_*` / `B2_GRASP_*` selectors and the `B2_SAM_PORT` /
`B2_GRASP_PORT` ports from the environment, `status` reports readiness,
`stop` refuses while Runtime leases exist, and `qualify` writes the
hash-named readiness report a launch binds. The host layer supplies the
workspace and Runtime state roots; the service module resolves no paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ora.agent.model_services import operator
from ora.orchestrator import paths


def state_directory(explicit: Path | None) -> Path:
    """The `model-services` workspace under the configured output storage."""
    if explicit is not None:
        return explicit
    return paths.experiment_workspace("model-services")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("start", "status", "stop", "qualify"))
    parser.add_argument("--state", type=Path,
                        help="service state and log directory (default: model-services under the configured "
                             "output storage; services started by the retired scripts/b2 used b2-model-services)")
    args = parser.parse_args(argv)
    state = state_directory(args.state)
    if args.command == "qualify":
        print(operator.qualify(state / "qualification"))
        return 0
    result = operator.lifecycle(args.command, root=paths.PROJECT_ROOT, state=state,
                                runtime_state_root=paths.STATE_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "status":
        return 0 if all(item.get("ready") is True for item in result.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
