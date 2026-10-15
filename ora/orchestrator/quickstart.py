"""A bounded Synthetic episode through the normal plan, preflight and launch path."""
from __future__ import annotations

import argparse
import os
import time
from typing import Any


def synthetic_spec() -> dict[str, Any]:
    protocol = os.environ.get("ORA_PROVIDER", "openai_compatible")
    if protocol not in {"openai_compatible", "anthropic_messages"}:
        raise ValueError("ORA_PROVIDER must be openai_compatible or anthropic_messages")
    fallback_key = "ANTHROPIC_API_KEY" if protocol == "anthropic_messages" else "OPENAI_API_KEY"
    key_env = "ORA_API_KEY" if os.environ.get("ORA_API_KEY") else fallback_key
    if not os.environ.get(key_env):
        raise ValueError(f"Set ORA_API_KEY or {fallback_key} before running the quickstart")
    default_url = "https://api.anthropic.com" if protocol == "anthropic_messages" else "https://api.openai.com/v1"
    endpoint = os.environ.setdefault("ORA_BASE_URL", default_url).rstrip("/")
    model = os.environ.get("ORA_MODEL", "" if protocol == "anthropic_messages" else "gpt-4.1-mini")
    if not model:
        raise ValueError("Set ORA_MODEL to the model ID served by your endpoint")
    from urllib.parse import urlsplit
    parsed = urlsplit(endpoint)
    transport = "loopback_http_v1" if parsed.scheme == "http" and parsed.hostname in {
        "localhost", "127.0.0.1", "::1",
    } else "https_v1"
    return {
        "schema": "ora.run-spec.v1", "experiment": "quickstart", "config": "synthetic-native",
        "milestone": "M0", "task": "move-once", "seed": 0, "evidence_tier": "diagnostic",
        "output_workspace": "outputs/quickstart",
        "task_instruction": "Read the robot manual, then use robot.submit_action to move the left end "
            "effector to position_m [0, 0, 1.02] with orientation_xyzw [0, 0, 0, 1]. "
            "This Synthetic demonstration ends after one accepted action.",
        "policy": {
            "kind": "native_api", "provider": protocol, "model": model, "endpoint": endpoint,
            "endpoint_transport": transport, "api_key_env": key_env, "base_url_env": "ORA_BASE_URL",
            "context_window_tokens": 32768, "max_output_tokens": 2048,
            "supports_images": os.environ.get("ORA_SUPPORTS_IMAGES", "1") != "0",
            "stream": False, "require_live_marker": False,
            "reasoning_effort": os.environ.get("ORA_REASONING_EFFORT", "provider_default"),
        },
        "runtime": {"kind": "synthetic", "terminal_actions": 1,
                    "workspace_bounds_m": {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 6,
                    "max_sim_steps": 1, "wall_time_s": 120},
        "policy_context": {
            "codex_project": ["ora/policy/planners/native/project-v1.md"],
            "episode_prefix": ["ora/policy/core/thin-react-v1.md"],
            "tool_manual": ["ora/policy/embodiments/aloha-agilex/manual-v1.md"],
        },
        "qualification": {"required": False},
    }


def command_run(args: argparse.Namespace) -> int:
    from . import cli

    plan = cli.resolve_run_spec(synthetic_spec())
    preflight = cli._preflight(plan)
    if not preflight["passed"]:
        return 2
    run_id = cli._spawn(plan, preflight, yes=True)
    root = cli._run_root(run_id)
    pid = cli._load(root / "process.json")["pid"]
    print(f"episode: {root}", flush=True)
    deadline = time.monotonic() + plan["budgets"]["wall_time_s"] + 15
    try:
        while time.monotonic() < deadline:
            try:
                finished = os.waitpid(pid, os.WNOHANG)[0] == pid
            except ChildProcessError:
                finished = not cli._pid_alive(pid)
            if (root / "result.json").is_file() and finished:
                result = cli._load(root / "result.json")
                print(f"state: {result['state']}; model calls: {result.get('model_calls', 0)}; "
                      "official_success: null (Synthetic has no benchmark verdict)")
                if (root / "seal.json").is_file():
                    print("evidence: evaluated and sealed")
                else:
                    print(f"evidence was not sealed; inspect {root / 'supervisor.log'}")
                    return 1
                return 0 if result["state"] == "completed" else 1
            if finished and not (root / "result.json").is_file():
                raise RuntimeError(f"Supervisor stopped before writing a result; inspect {root / 'supervisor.log'}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        cli.command_stop(run_id)
        return 130
    cli.command_stop(run_id)
    raise RuntimeError(f"Quickstart exceeded its bounded deadline; inspect {root / 'supervisor.log'}")
