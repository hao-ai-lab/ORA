#!/usr/bin/env python3
"""Operate the Runtime v1 qualification experiment over the shared control CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ora.orchestrator.control_audit import audit_libero_run
from ora.orchestrator.paths import (
    experiment_workspace, iter_run_roots, resolve_run_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITION = Path(__file__).with_name("experiment.yaml")
OUTPUT_WORKSPACE = experiment_workspace("harness-runtime-v1")
SPECS_ROOT = OUTPUT_WORKSPACE / "specs"
GATES_ROOT = OUTPUT_WORKSPACE / "gates"
QUALIFICATION_ROOT = OUTPUT_WORKSPACE / "qualification"
LOCAL_QUALIFICATION_ROOT = OUTPUT_WORKSPACE / "qualification-manifests"
CONTROL_AUDIT_ROOT = OUTPUT_WORKSPACE / "control-audits"
LIBERO_CONTROL_QUALIFICATION_ROOT = QUALIFICATION_ROOT / "libero-pose-control-v2"
LIBERO_CONTROL_REVISIONS = {
    "rlinf": "c90951a0c799a750cb5294ed10587c61cc2af8bf",
    "libero_pro": "eafdb809426b13153aa1e4c42d6601844217dfec",
    "data": "c86fc3b8293185a6f373677018ff3e37f8391602",
}


def _definition() -> dict[str, Any]:
    value = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "ora.runtime-experiment.v1":
        raise RuntimeError("invalid Runtime v1 experiment definition")
    return value


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _milestone(identifier: str) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = _definition()
    item = definition["milestones"].get(identifier.upper())
    if not isinstance(item, Mapping):
        raise RuntimeError(f"unknown milestone: {identifier}")
    return definition, dict(item)


def _gate_path(milestone: str, policy: str) -> Path:
    return GATES_ROOT / f"{milestone.upper()}-{policy}.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _current_source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("cannot resolve current Harness source commit")
    return result.stdout.strip()


def _preserved_python_entry(value: str) -> Path:
    """Canonicalize spelling without dereferencing a virtualenv entry."""

    return Path(os.path.abspath(os.path.expanduser(value)))


def _gate_source_commit(gate_path: Path) -> str:
    gate = _load_json(gate_path)
    recorded = gate.get("source_commit")
    if isinstance(recorded, str) and recorded:
        return recorded
    run_id = gate.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f"gate has no source identity: {gate_path.name}")
    plan = _load_json(resolve_run_root(run_id) / "plan.json")
    source = plan.get("source", {})
    recorded = source.get("commit") if isinstance(source, Mapping) else None
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError(f"gate run has no source identity: {gate_path.name}")
    return recorded


def _require_current_gate(gate_path: Path) -> None:
    if not gate_path.is_file():
        raise RuntimeError(f"required gate is not complete: {gate_path.name}")
    expected = _current_source_commit()
    observed = _gate_source_commit(gate_path)
    if observed != expected:
        raise RuntimeError(
            f"required gate is stale for current source: {gate_path.name} "
            f"records {observed}, current {expected}"
        )


def _check_gate(milestone: str, policy: str, prerequisite: str | None) -> None:
    required = (
        _gate_path(milestone, "scripted")
        if policy == "native_api"
        else (_gate_path(prerequisite, "native_api") if prerequisite else None)
    )
    if required is not None:
        _require_current_gate(required)
    accepted = GATES_ROOT / "M5-accepted.json"
    if (
        accepted.is_file()
        and milestone.upper() == "M5"
        and _gate_source_commit(accepted) == _current_source_commit()
    ):
        raise RuntimeError("M5 already has an accepted official-success trajectory")


def _qualification_path(item: Mapping[str, Any]) -> str | None:
    mode = item.get("qualification")
    if mode is None:
        return None
    if mode == "committed":
        return str(item["qualification_manifest"])
    if mode == "local":
        path = LOCAL_QUALIFICATION_ROOT / f"{item['task']}.json"
        if not path.is_file():
            raise RuntimeError(
                f"{item['task']} is not qualified; run "
                f"`uv run --locked python experiments/runtime_v1/run.py qualify {item['task']}`"
            )
        return str(path)
    raise RuntimeError(f"unsupported qualification mode: {mode}")


def compose(
    milestone: str, *, policy: str, seed: int | None, preview: bool = False,
) -> tuple[dict[str, Any], Path]:
    definition, item = _milestone(milestone)
    milestone = milestone.upper()
    if policy not in {"scripted", "native_api"}:
        raise RuntimeError("policy must be scripted or native_api")
    seeds = list(item["seeds"])
    selected_seed = seeds[0] if seed is None else seed
    if selected_seed not in seeds:
        raise RuntimeError(f"seed {selected_seed} is not frozen for {milestone}: {seeds}")
    if not preview:
        _check_gate(milestone, policy, item.get("prerequisite"))
    defaults = dict(definition["runtime_defaults"])
    runtime_kind = item["runtime"]
    runtime = {
        "kind": runtime_kind,
        "transport": (
            defaults["transport"]
            if runtime_kind == "robotwin"
            else "in_process_direct"
        ),
        "workspace_bounds_m": defaults["workspace_bounds_m"],
    }
    if runtime_kind == "robotwin":
        runtime.update({key: defaults[key] for key in (
            "root_env", "python_env", "required_revision", "task_config",
            "require_gpu", "startup_timeout_s",
            "request_timeout_s",
        )})
        # The live RoboTwin rungs run on the registered host named by the marker.
        host = os.environ.get("ORA_EXECUTION_HOST")
        if host:
            runtime["required_host"] = host
    if policy == "scripted":
        policy_block = {
            "kind": "scripted", "model": "scripted-policy-v1",
            "provider": "trusted_scripted", "reasoning_effort": "none",
            "actions": [item["scripted_action"]],
        }
    else:
        from ora.orchestrator.run_spec_inputs import provider_policy
        policy_block = provider_policy(definition["provider_profile"])
        # Synthetic does not need the simulator provider's live admission marker.
        policy_block["require_live_marker"] = runtime_kind != "synthetic"
    policy_block["tool_surface"] = {
        "profile": "robot_runtime_v1", "component_overrides": {},
    }
    policy_block["action_codec"] = "runtime_action_v3"
    qualification_path = None if preview else _qualification_path(item)
    spec = {
        "schema": "ora.run-spec.v1",
        "experiment": definition["experiment"], "config": definition["config"],
        "output_workspace": definition["output_workspace"],
        "milestone": milestone, "task": item["task"],
        "task_instruction": item["instruction"], "seed": selected_seed,
        "evidence_tier": item["evidence_tier"], "policy": policy_block,
        "runtime": runtime, "budgets": item["budgets"],
        "policy_context": definition["policy_context"],
        "qualification": {
            "required": qualification_path is not None,
            **({"manifest": qualification_path} if qualification_path else {}),
        },
        "prerequisite": item.get("prerequisite"),
    }
    SPECS_ROOT.mkdir(parents=True, exist_ok=True)
    path = SPECS_ROOT / f"{milestone.lower()}-{policy}-seed-{selected_seed}.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec, path


def _control(arguments: list[str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", "ora.orchestrator.cli", *arguments], cwd=PROJECT_ROOT,
        check=False,
    ).returncode


def qualify(task_or_milestone: str) -> int:
    definition = _definition()
    match = None
    for identifier, item in definition["milestones"].items():
        if task_or_milestone.upper() == identifier or task_or_milestone == item["task"]:
            match = (identifier, item)
            break
    if match is None or match[1].get("qualification") != "local":
        raise RuntimeError("qualify accepts M2/M3/M4/M5 or their task name")
    _identifier, item = match
    root_raw = os.environ.get(definition["runtime_defaults"]["root_env"])
    if not root_raw:
        raise RuntimeError(f"missing {definition['runtime_defaults']['root_env']}")
    python_env = definition["runtime_defaults"]["python_env"]
    python_raw = os.environ.get(python_env)
    if not python_raw:
        raise RuntimeError(f"missing {python_env}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = QUALIFICATION_ROOT / item["task"] / stamp
    command = [
        python_raw, "-m", "ora.environment.backends.robotwin.qualification",
        "--robotwin-root", root_raw, "--output-root", str(output),
        "--tasks", item["task"], "--task-config", definition["runtime_defaults"]["task_config"],
        "--required-revision", definition["runtime_defaults"]["required_revision"],
        "--accepted-per-task", "3", "--start-requested-seed", "100",
        "--max-requested-seeds", "105", "--stability-steps", "250",
    ]
    print(" ".join(command))
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(PROJECT_ROOT), process_environment.get("PYTHONPATH"),
    )))
    result = subprocess.run(
        command, cwd=Path(root_raw).expanduser().resolve(),
        env=process_environment, check=False,
    )
    if result.returncode:
        return result.returncode
    ledger = output / item["task"] / "seed-ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    accepted = [record for record in records if record.get("classification") == "accepted"]
    manifest = {
        "schema": "ora.seed-qualification.v1",
        "task": item["task"], "accepted_seeds": [record["requested_seed"] for record in accepted[:3]],
        "reset_predicate_false": all(record.get("reset_official_predicate") is False for record in accepted[:3]),
        "oracle_success": all(record.get("expert", {}).get("post_play_official_predicate") is True for record in accepted[:3]),
        "terminal_stability_steps": min((record.get("stability", {}).get("checks", 0) for record in accepted[:3]), default=0),
        "source_output": str(output),
        "record_sha256": [record.get("record_sha256") for record in accepted[:3]],
        "policy_visible": False,
    }
    target = LOCAL_QUALIFICATION_ROOT / f"{item['task']}.json"
    _atomic(target, manifest)
    print(f"qualification manifest: {target}")
    return 0


def mark_gate(run_id: str) -> int:
    root = resolve_run_root(run_id)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if not (root / "seal.json").is_file():
        raise RuntimeError("seal the run before recording a milestone gate")
    if evaluation.get("status") != "valid" or result.get("state") != "completed":
        raise RuntimeError("only completed evaluator-valid runs satisfy a milestone gate")
    gate = {
        "schema": "ora.milestone-gate.v1",
        "milestone": plan["milestone"], "policy": plan["policy"]["kind"],
        "run_id": run_id, "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source"]["commit"],
        "official_success": result.get("official_success"),
        "evaluation": evaluation["status"], "seal": "seal.json",
    }
    _atomic(_gate_path(plan["milestone"], plan["policy"]["kind"]), gate)
    if (plan["milestone"] == "M5" and plan["policy"]["kind"] == "native_api"
            and result.get("official_success") is True):
        _atomic(GATES_ROOT / "M5-accepted.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def control_audit(run_id: str | None) -> int:
    roots = [resolve_run_root(run_id)] if run_id else list(iter_run_roots())
    records = [
        audit_libero_run(root)
        for root in roots
        if _load_json(root / "plan.json").get("runtime", {}).get("kind")
        == "libero_pro"
    ]
    report = {
        "schema": "ora.libero-control-validity-report.v1",
        "source_commit": _current_source_commit(),
        "record_count": len(records),
        "counts": {
            value: sum(
                record["control_validity"] == value for record in records
            )
            for value in ("control-valid", "contaminated", "indeterminate")
        },
        "records": records,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = CONTROL_AUDIT_ROOT / (
        f"{run_id}.json" if run_id else f"all-libero-{stamp}.json"
    )
    _atomic(target, report)
    print(json.dumps({**report, "output": str(target)}, ensure_ascii=False, indent=2))
    return 0


def libero_pose_control_qualification() -> int:
    from ora.orchestrator.resources import declared_execution_host

    declared_execution_host()
    values = {
        "python": os.environ.get("LIBERO_PRO_PYTHON"),
        "rlinf": os.environ.get("RLINF_ROOT"),
        "libero_pro": os.environ.get("LIBERO_PRO_ROOT"),
        "data": os.environ.get("LIBERO_PRO_DATA_ROOT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"missing LIBERO qualification environment: {', '.join(missing)}"
        )
    # Keep the configured venv/conda entry itself. Resolving its Python
    # symlink would select the base interpreter and lose environment packages.
    python = _preserved_python_entry(str(values["python"]))
    roots = {
        name: Path(str(values[name])).expanduser().resolve()
        for name in ("rlinf", "libero_pro", "data")
    }
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("LIBERO_PRO_PYTHON is not executable")
    for name, root in roots.items():
        if not root.is_dir():
            raise RuntimeError(f"{name} root is unavailable: {root}")
        if _git_external(root, "rev-parse", "HEAD") != LIBERO_CONTROL_REVISIONS[name]:
            raise RuntimeError(f"{name} revision does not match the pinned qualification")
        if _git_external(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"{name} source is modified")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = LIBERO_CONTROL_QUALIFICATION_ROOT / stamp
    report = output / "report.json"
    script = Path(__file__).with_name("libero_pose_control_qualification.py")
    command = [
        str(python), str(script),
        "--project-root", str(PROJECT_ROOT),
        "--output-root", str(output / "scenarios"),
        "--report", str(report),
        "--rlinf-root", str(roots["rlinf"]),
        "--libero-pro-root", str(roots["libero_pro"]),
        "--data-root", str(roots["data"]),
        "--required-rlinf-revision", LIBERO_CONTROL_REVISIONS["rlinf"],
        "--required-libero-pro-revision", LIBERO_CONTROL_REVISIONS["libero_pro"],
        "--required-data-revision", LIBERO_CONTROL_REVISIONS["data"],
    ]
    environment = os.environ.copy()
    environment["LIBERO_TYPE"] = "standard"
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(PROJECT_ROOT), str(roots["rlinf"]), environment.get("PYTHONPATH"),
    )))
    result = subprocess.run(
        command, cwd=roots["rlinf"], env=environment, check=False,
    )
    print(f"LIBERO pose qualification report: {report}")
    return result.returncode


def _git_external(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect source root: {root}")
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime-v1", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("configs", help="list the experiment's M0-M5 configurations")
    explain = sub.add_parser("explain", help="print one milestone configuration")
    explain.add_argument("milestone")
    qualify_parser = sub.add_parser("qualify", help="qualify local M2-M5 seed manifests")
    qualify_parser.add_argument("target")
    actions = {
        "context": "preview exact Native context without model or simulator",
        "plan": "compose and print one milestone RunPlan",
        "preflight": "run milestone launch gates without starting",
        "launch": "launch one gated scripted or Native milestone run",
    }
    for name, help_text in actions.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("milestone")
        command.add_argument("--policy", choices=("scripted", "native_api"), default="scripted")
        command.add_argument("--seed", type=int)
        if name == "launch":
            command.add_argument("--yes", action="store_true")
    status = sub.add_parser("status", help="show one run or the local registry")
    status.add_argument("run_id", nargs="?")
    audit = sub.add_parser(
        "control-audit",
        help="classify LIBERO behavioral verdict and control validity separately",
    )
    audit.add_argument("run_id", nargs="?")
    sub.add_parser(
        "libero-pose-control",
        help="run the source-bound real LIBERO/Panda Pose/Frame v2 matrix",
    )
    forwarded = {
        "watch": "stream one milestone run's causal events",
        "stop": "stop the exact active Runtime v1 run",
        "metadata": "show one run's context and environment manifest",
        "metrics": "rebuild one run's Runtime metrics",
        "evaluate": "evaluate one milestone trace",
        "seal": "seal a valid run and record its milestone gate",
        "export": "export one sealed milestone run",
    }
    for name, help_text in forwarded.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "configs":
            for identifier, item in _definition()["milestones"].items():
                print(f"{identifier}\t{item['task']}\t{item['evidence_tier']}\tseeds={item['seeds']}")
            return 0
        if args.command == "explain":
            _definition_value, item = _milestone(args.milestone)
            print(yaml.safe_dump(item, sort_keys=False, allow_unicode=True))
            return 0
        if args.command == "qualify":
            return qualify(args.target)
        if args.command in {"context", "plan", "preflight", "launch"}:
            _spec, path = compose(
                args.milestone, policy=args.policy, seed=args.seed,
                preview=args.command == "context",
            )
            arguments = [args.command, str(path)]
            if args.command == "launch" and args.yes:
                arguments.append("--yes")
            return _control(arguments)
        if args.command == "status":
            return _control(["status", *([args.run_id] if args.run_id else [])])
        if args.command == "control-audit":
            return control_audit(args.run_id)
        if args.command == "libero-pose-control":
            return libero_pose_control_qualification()
        if args.command == "seal":
            result = _control(["seal", args.run_id])
            return mark_gate(args.run_id) if result == 0 else result
        return _control([args.command, args.run_id])
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"runtime-v1: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
