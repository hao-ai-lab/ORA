"""What the policy may still see once a round has reached its boundary.

At an environment boundary the Supervisor persists the official verdict, may
offer one final faithful-check turn that cannot execute motion
(`allow_terminal_policy_lifecycle`), and in a multi-round session otherwise
records the terminal observation without a policy turn
(`preserve_terminal_observation_without_policy_turn`) so the persistent
AgentSession keeps a complete history.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ora.agent import ActionProposal, PolicyTurnInput
from ora.agent.native.constants import NATIVE_CAUSAL_SLOTS

from .episode_bindings import SupervisorError
from .episode_records import _append
from .interaction import InteractionStateError
from .policy_finish import read_policy_finish
from .termination import TerminationDecision

if TYPE_CHECKING:
    from .episode_loop import EpisodeLoop


def stop_at_environment_boundary(
    loop: EpisodeLoop, decision: TerminationDecision, environment_reason: str,
    *, terminal_turn_allowed: bool,
) -> None:
    """End the round at a trusted environment or control boundary."""

    loop.record_termination_decision(
        decision, source="environment_status",
    )
    loop.verdict = loop.verdict_guard.persist(loop.round_index)
    loop.termination_authority = decision.authority
    terminal_observation_consumed = False
    if terminal_turn_allowed:
        terminal_observation_consumed = allow_terminal_policy_lifecycle(loop)
    if loop.multi_round and not terminal_observation_consumed:
        preserve_terminal_observation_without_policy_turn(loop)
    loop.interaction.mark_terminal(
        environment_reason
    )
    loop.termination = "trusted_runtime_boundary"


def _terminal_turn(loop: EpisodeLoop, *, terminal_lifecycle: bool) -> PolicyTurnInput:
    return PolicyTurnInput(
        task_id=str(loop.plan["task"]),
        task_instruction=loop.plan.get("task_instruction", loop.plan["task"]),
        context=loop.pointer,
        image_path=loop.workspace / loop.pointer["main_image"],
        observation_id=loop.interaction.observation_id,
        observation_step=loop.interaction.observation_step,
        budget={
            "robot_decisions": (
                loop.plan["budgets"]["max_robot_decisions"] - loop.round_decisions
            ),
            "model_calls": (
                loop.plan["budgets"]["max_model_calls"] - loop.round_model_calls
            ),
            **loop.native_allowance.fields(),
        },
        interaction_phase="terminal",
        fresh_observation_required=False,
        allow_reminder=False,
        terminal_lifecycle=terminal_lifecycle,
    )


def preserve_terminal_observation_without_policy_turn(loop: EpisodeLoop) -> bool:
    callback = getattr(
        loop.planner, "preserve_observation_without_policy_turn", None,
    )
    if not callable(callback):
        loop.record_auxiliary_failure(
            "terminal_observation_preservation",
            SupervisorError(
                "persistent Native session cannot preserve a terminal observation"
            ),
        )
        return False
    terminal_turn = _terminal_turn(loop, terminal_lifecycle=True)
    try:
        callback(**terminal_turn.runtime_kwargs())
        _append(loop.events, {
            "type": "terminal_observation_preserved",
            "observation_id": loop.interaction.observation_id,
            "observation_step": loop.interaction.observation_step,
            "round_index": loop.round_index,
            "time": time.time(),
        })
        return True
    except Exception as exc:
        loop.record_auxiliary_failure("terminal_observation_preservation", exc)
        return False


def allow_terminal_policy_lifecycle(loop: EpisodeLoop) -> bool:
    """Expose one final faithful OpenETA turn without executing new motion."""

    if (
        not loop.tools.surface.supports("faithful_check_task")
        or loop.round_model_calls >= loop.plan["budgets"]["max_model_calls"]
    ):
        return False
    loop.interaction.mark_terminal(
        str(loop.environment_status.get("reason") or "environment_terminal")
    )
    terminal_turn = _terminal_turn(
        loop,
        terminal_lifecycle=(
            loop.policy.get("context_composition") in NATIVE_CAUSAL_SLOTS
        ),
    )
    try:
        loop.interaction.begin_policy(terminal_lifecycle=True)
        outcome = loop.planner.act(terminal_turn)
        call_increment = (
            outcome.model_calls
            if loop.planner.capabilities.runtime_kind != "scripted" else 0
        )
        loop.model_calls += call_increment
        loop.round_model_calls += call_increment
        if isinstance(outcome, ActionProposal):
            loop.interaction.record_model_calls(call_increment)
            loop.interaction.mark_terminal("terminal_mutation_rejected")
            _append(loop.events, {
                "type": "terminal_policy_mutation_rejected",
                "outcome": outcome.as_record(),
                "time": time.time(),
            })
            return True
        loop.interaction.record_policy_outcome(
            kind=outcome.kind,
            model_calls=call_increment,
            reason=getattr(outcome, "reason", "terminal_policy_complete"),
        )
        # A v2 finish here is recorded on the round; the environment already
        # ended it, so the finish changes no authority (ended_round false).
        policy_finish = (
            read_policy_finish(loop.control, loop.tools.surface, ended_round=False)
            if outcome.kind == "lifecycle" else None
        )
        if policy_finish is not None:
            loop.policy_finish = policy_finish
        _append(loop.events, {
            "type": "terminal_policy_lifecycle",
            "outcome": outcome.as_record(),
            "time": time.time(),
            **({"policy_finish": policy_finish} if policy_finish is not None else {}),
        })
        return True
    except Exception as exc:
        loop.record_auxiliary_failure("terminal_policy_lifecycle", exc)
        try:
            loop.interaction.mark_terminal("terminal_policy_lifecycle_failed")
        except InteractionStateError:
            pass
        return False
