"""Bind one submitted skill sequence to the tool runtime that owns its stages.

This is the coordinator's half of the join and nothing else. The stage loop and
its trust boundary live in ``action_executor.execute_skill_sequence``; the
geometry, the candidate artifacts and the ledger live in the tool runtime. This
module only checks that the runtime bound to this RunPlan actually implements
the driver protocol, and refuses before any world mutation if it does not.

It refuses rather than falling back: a submitted action the Supervisor cannot drive is an admission fault, and
executing the first stage of a sequence nobody can finish is the one outcome
worse than not starting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment import (
    EnvironmentSession, validate_environment_status,
)

from .action_executor import execute_skill_sequence


#: The two methods a tool runtime must publish to serve a committed sequence.
#: ``drive_skill_stage`` produces one stage's action or measurement from the
#: candidate it holds; ``build_skill_receipt`` advances and releases the ledger
#: and returns the staged receipt.
DRIVER_PROTOCOL = ("drive_skill_stage", "build_skill_receipt")


def capture_unpublished_observation(
    runtime: EnvironmentSession, *, observation_root: Path, main_camera: str,
    record: Callable[[Mapping[str, Any]], None], reason: str,
) -> tuple[dict[str, Any], Path]:
    """Capture evidence a driven stage needs without opening an Agent turn.

    Only the last such capture becomes the policy's next observation, through
    ``publish(captured_observation=...)``. The intermediate ones are run
    evidence, which is why each is recorded as not published at capture: a
    reader counting observations must be able to tell which ones the policy
    actually saw.
    """

    captured, _images = runtime.observe(observation_root)
    captured = {**captured, "main_camera": main_camera}
    record({
        "type": "unpublished_observation",
        "observation_id": captured["observation_id"],
        "reason": reason,
        "published_to_agent_at_capture": False,
        "environment_status": validate_environment_status(runtime.status()),
        "time": time.time(),
    })
    return captured, observation_root / str(captured["observation_id"])


def run_skill_sequence(
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
    """Drive one committed sequence through the bound tool runtime."""

    missing = [name for name in DRIVER_PROTOCOL if not callable(getattr(tools, name, None))]
    if missing:
        raise ValueError(
            "a committed skill sequence requires a tool runtime that implements "
            + ", ".join(missing)
        )
    source = Path(str(current["source_root"]))
    manifest = Path(str(current["observation_manifest"]))
    if not manifest.is_file() or manifest.parent.resolve() != source.resolve():
        raise ValueError("committed skill sequence current observation binding is invalid")
    initial = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(initial, Mapping)
        or initial.get("observation_id") != action.get("observation_id")
    ):
        raise ValueError("committed skill sequence initial observation identity drift")
    return execute_skill_sequence(
        runtime, action,
        initial_observation=initial,
        initial_observation_root=source,
        drive_stage=tools.drive_skill_stage,
        build_receipt=tools.build_skill_receipt,
        capture_observation=lambda: capture_unpublished_observation(
            runtime, observation_root=observation_root,
            main_camera=main_camera, record=record,
            reason="skill_sequence_stage",
        ),
        should_stop=should_stop,
    )
