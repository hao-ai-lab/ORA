"""Bind one submitted composed tool action to the tool runtime that committed it.

The coordinator's half of the join, as ``skill_sequence_job`` is for closed
skills: refuse before any world mutation if the bound tool runtime cannot
serve composed tools or the action does not belong to the current
observation, then hand the runtime's
:class:`~ora.agent.tools.composed.executor.Execution` to
:func:`execute_composed_tool`, the trusted loop below.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.contracts import validate_runtime_action
from ora.environment import EnvironmentSession

from .action_executor import _action_sequence, execute_runtime_action
from .skill_sequence_job import capture_unpublished_observation
from .stage_outcome import classify_stage_action

#: The method a tool runtime publishes to serve a committed composed tool.
DRIVER_PROTOCOL = ("start_composed_tool",)


def run_composed_tool(
    runtime: EnvironmentSession,
    action: Mapping[str, Any],
    *,
    tools: Any,
    current: Mapping[str, Any],
    observation_root: Path,
    main_camera: str,
    record: Callable[[Mapping[str, Any]], None],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], Path]]:
    """Drive one committed composed tool through the bound tool runtime."""

    missing = [name for name in DRIVER_PROTOCOL if not callable(getattr(tools, name, None))]
    if missing:
        raise ValueError("a committed composed tool requires a tool runtime that implements " + ", ".join(missing))
    source = Path(str(current["source_root"]))
    manifest = Path(str(current["observation_manifest"]))
    if not manifest.is_file() or manifest.parent.resolve() != source.resolve():
        raise ValueError("committed composed tool current observation binding is invalid")
    initial = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(initial, Mapping) or initial.get("observation_id") != action.get("observation_id"):
        raise ValueError("committed composed tool initial observation identity drift")
    box: dict[str, Any] = {"latest": (dict(initial), source), "captured": False, "stale": False}

    def capture() -> tuple[dict[str, Any], Path]:
        box["latest"] = capture_unpublished_observation(
            runtime, observation_root=observation_root, main_camera=main_camera, record=record,
            reason="composed_tool_stage",
        )
        box["captured"] = True
        box["stale"] = False
        return box["latest"]

    def fresh() -> tuple[dict[str, Any], Path]:
        """An observation taken after the last dispatched action, for sdk.read and nested plans."""
        return capture() if box["stale"] else box["latest"]

    execution = tools.start_composed_tool(action, capture=capture, latest=lambda: box["latest"], fresh=fresh)
    return execute_composed_tool(
        runtime, action, execution=execution, initial_observation=initial, initial_observation_root=source,
        capture_observation=capture, latest_box=box, should_stop=should_stop,
    )


def execute_composed_tool(
    runtime: EnvironmentSession,
    payload: Mapping[str, Any],
    *,
    execution: Any,
    initial_observation: Mapping[str, Any],
    initial_observation_root: Path,
    capture_observation: Callable[[], tuple[dict[str, Any], Path]],
    latest_box: dict[str, Any],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], Path]]:
    """Drive one committed composed tool without opening an Agent turn.

    The body chooses each next primitive action (``execution.next``), a sparse
    motion v3 move or gripper action; this function owns the same trust boundary as ``_execute_skill_sequence``:
    every child is lowered to the committed parent's observation binding and
    stage-indexed id, dispatched through ``execute_runtime_action``, classified
    by ``classify_stage_action`` and only then shown to the body. An operator
    stop or a terminal episode ends the body from outside, a raised backend
    fault reports before it propagates, and the outer receipt has the closed
    skill shape so projection and evaluation read it unchanged.
    """

    from ora.agent.tools.composed import sdk
    from ora.agent.tools.composed.executor import (
        STOP_ENVIRONMENT_TERMINAL, STOP_OPERATOR_INTERRUPT, Finished,
    )
    from ora.contracts import COMPOSED_TOOL_ACTION_SCHEMA

    action = validate_runtime_action(payload)
    if action["schema"] != COMPOSED_TOOL_ACTION_SCHEMA:
        raise ValueError("execute_composed_tool requires a composed tool action")
    started = time.time()
    latest_box["latest"] = (dict(initial_observation), initial_observation_root)
    latest_box["captured"] = False
    latest_box["stale"] = False
    children: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    with _action_sequence(runtime, action):
        while True:
            step = execution.next()
            if isinstance(step, Finished):
                break
            if should_stop is not None and should_stop():
                execution.stop(STOP_OPERATOR_INTERRUPT)
                break
            if runtime.status().get("terminal") is True:
                execution.stop(STOP_ENVIRONMENT_TERMINAL)
                break
            child = sdk.lower_motion_v3(
                step.request, action_id=step.action_id, observation_id=action["observation_id"],
                observation_sha256=action["bound_observation_sha256"],
            )
            latest_box["captured"] = False
            latest_box["stale"] = True
            try:
                stage_receipt = execute_runtime_action(runtime, child, should_stop=should_stop)
            except Exception:
                execution.fault_pending("trusted_execution_fault")
                execution.finalize()
                raise
            if runtime.status().get("terminal") is True:
                children[step.action_id] = (child, stage_receipt, {})
                execution.stop(STOP_ENVIRONMENT_TERMINAL, dispatched=True)
                break
            record = classify_stage_action(
                {"status": "executed", "reason": "executed", "measurements": []}, child, stage_receipt,
            )
            children[step.action_id] = (child, stage_receipt, record)
            execution.resolve({key: record[key] for key in ("status", "reason", "measurements")})
        if not latest_box["captured"]:
            latest_box["latest"] = capture_observation()
            latest_box["captured"] = True
    finished = execution.finalize()
    latest = latest_box["latest"]
    stage_receipts = []
    for item in finished.stages:
        record = dict(item)
        child = children.get(str(item.get("action_id")))
        if child is not None:
            record["action"] = dict(child[0])
            record["receipt"] = child[1]
            if "pose_residual" in child[2]:
                record["pose_residual"] = child[2]["pose_residual"]
        # A retried stage reports its last attempt; the earlier dispatches stay
        # evidence, in order, so every child the Supervisor executed is joined.
        earlier = [
            entry.action_id for entry in finished.dispatched
            if entry.stage == item["stage"] and entry.action_id != item.get("action_id")
        ]
        if earlier:
            record["attempts"] = [
                {"action_id": action_id, "action": dict(children[action_id][0]), "receipt": children[action_id][1]}
                for action_id in earlier if action_id in children
            ]
        stage_receipts.append(record)
    arguments = action["arguments"]
    return {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": action["action_id"],
        "action": dict(action),
        "skill": arguments["tool"],
        "candidate_id": finished.receipt.get("candidate_id"),
        "bound_observation_id": action["observation_id"],
        "committed_sequence": list(arguments["committed_sequence"]),
        "resumed_from": 0,
        "stage_receipts": stage_receipts,
        "robot_decisions": 1,
        "dispatched_stage_actions": finished.receipt["cost"]["dispatched_stage_actions"],
        "status": finished.status,
        "stop_reason": finished.stop_reason,
        "terminal_after_action": finished.status == "terminated",
        "skill_receipt": dict(finished.receipt),
        "fresh_observation_id": str(latest[0].get("observation_id") or ""),
        "consumed": True,
        "composed_tool": dict(finished.receipt["composed_tool"]),
        "started_unix_s": started,
        "finished_unix_s": time.time(),
    }, latest
