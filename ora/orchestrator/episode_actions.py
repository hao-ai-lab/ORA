"""The Supervisor's execution of one accepted proposal.

`dispatch_action` routes the persisted proposal by schema: a committed skill
sequence runs its stage loop, and every other action goes through
`execute_runtime_action`. `post_action_boundary` decides whether the fresh
observation after that execution ends the round.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ora.contracts import SKILL_SEQUENCE_ACTION_SCHEMA

from .action_executor import execute_runtime_action
from .episode_records import _append
from .skill_sequence_job import run_skill_sequence
from .termination import (
    TerminationDecision,
    environment_boundary,
    infrastructure_boundary,
)

if TYPE_CHECKING:
    from .episode_loop import EpisodeLoop

#: Environment-reported reasons that are infrastructure, not task, boundaries.
_INFRASTRUCTURE_STATUS_REASONS = {
    "libero_control_health_fault",
}


def dispatch_action(
    loop: EpisodeLoop, action: dict[str, Any], current: Mapping[str, Any], *,
    should_stop: Callable[[], bool],
) -> tuple[dict[str, Any], tuple[dict[str, Any], Path] | None]:
    """Execute one proposal; return its receipt and any observation it captured."""

    captured_after_action: tuple[dict[str, Any], Path] | None = None
    if action.get("schema") == SKILL_SEQUENCE_ACTION_SCHEMA:
        receipt, captured_after_action = run_skill_sequence(
            loop.runtime, action, tools=loop.tools, current=current,
            observation_root=loop.observation_root,
            main_camera=loop.projector.main_camera,
            record=lambda event: _append(loop.events, event),
            should_stop=should_stop,
        )
    else:
        receipt = execute_runtime_action(
            loop.runtime, action,
            should_stop=should_stop,
        )
    return receipt, captured_after_action


def post_action_boundary(
    receipt: Mapping[str, Any], environment_status: Mapping[str, Any],
) -> tuple[TerminationDecision, str] | None:
    """The round boundary, if any, that the executed action reached."""

    del receipt
    if not environment_status["terminal"]:
        return None
    environment_reason = str(
        environment_status.get("reason") or "trusted_runtime_boundary"
    )
    decision = (
        infrastructure_boundary(environment_reason, scope="round")
        if environment_reason in _INFRASTRUCTURE_STATUS_REASONS
        else environment_boundary(environment_reason)
    )
    return decision, environment_reason
