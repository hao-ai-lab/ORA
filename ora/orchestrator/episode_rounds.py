"""Closing a round and, in a multi-round session, starting the next one.

`finish_round` runs once a trusted boundary has set the loop's `termination`:
it hands the official verdict to the policy, writes the round result and
reports whether the session ends. Otherwise
`start_next_round` resets the world and every per-round counter, proves the new
round is isolated from the previous one, and publishes its first observation.
"""

from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING, Any

from ora.agent import PolicyRuntimeEvent
from ora.environment import validate_environment_status

from .episode_bindings import SupervisorError, _round_comparability
from .episode_observation import publish_observation
from .episode_records import _append, _atomic_json, _status
from .robodojo_round_scene import round_scene_field
from .stagnation import StagnationGuard
from .termination import TerminationAuthority, infrastructure_boundary
from .verdict_guard import OFFICIAL_VERDICT_CHECKER_DIVERGENCE

if TYPE_CHECKING:
    from .episode_loop import EpisodeLoop


def finish_round(loop: EpisodeLoop) -> bool:
    """Record the round that just ended; return True when the session ends."""

    if loop.multi_round and loop.verdict is not None:
        try:
            loop.environment_status = validate_environment_status(
                loop.runtime.status()
            )
            _append(loop.events, {
                "type": "environment_status",
                "record": loop.environment_status,
                "round_index": loop.round_index,
                "final": True,
                "time": time.time(),
            })
        except Exception as exc:
            loop.record_auxiliary_failure(
                "post_terminal:environment_status", exc,
            )
    loop.deliver_policy_event(
        PolicyRuntimeEvent(
            kind="official_verdict", payload={"value": loop.verdict},
            observation_id=loop.interaction.observation_id,
            observation_step=loop.interaction.observation_step,
        ),
        stage="official_verdict",
    )
    round_finished = time.time()
    round_wall_duration_s = round(
        max(0, time.monotonic_ns() - loop.round_started_monotonic_ns) / 1e9,
        6,
    )
    if loop.verdict_guard.divergence is not None:
        # Checker and record disagree: no behavioral sample this round.
        guard_decision = infrastructure_boundary(
            OFFICIAL_VERDICT_CHECKER_DIVERGENCE, scope="round",
        )
        loop.record_termination_decision(guard_decision, source="verdict_checker_guard")
        loop.termination_authority = guard_decision.authority
    round_result = _round_result(loop, round_finished, round_wall_duration_s)
    loop.round_results.append(round_result)
    if loop.multi_round:
        _write_round_evidence(loop, round_result, round_finished)
    try:
        loop.interaction.close()
    except Exception as exc:
        loop.record_auxiliary_failure(
            "post_terminal:interaction_close", exc,
        )
    return (
        loop.stop_session
        or loop.termination == "operator_interrupt"
        or loop.round_index + 1 >= loop.round_limit
    )


def _round_result(
    loop: EpisodeLoop, round_finished: float, round_wall_duration_s: float,
) -> dict[str, Any]:
    authority = loop.termination_authority
    verdict = loop.verdict
    round_state = (
        "incomplete"
        if authority in {
            TerminationAuthority.OPERATOR,
            TerminationAuthority.INFRASTRUCTURE,
        }
        else "completed"
    )
    round_official = (
        verdict.get("official_success")
        if verdict is not None
        and authority
        is not TerminationAuthority.INFRASTRUCTURE
        else None
    )
    return {
        "schema": "ora.round-result.v1",
        "run_id": loop.run_id,
        "round_index": loop.round_index,
        "seed": loop.expected_round_seed(loop.round_index),
        "state": round_state,
        "termination": loop.termination,
        "termination_authority": authority.value,
        "behavioral_sample_valid": authority not in {
            TerminationAuthority.INFRASTRUCTURE,
            TerminationAuthority.OPERATOR} and not bool(loop.operator_message_count),
        **loop.operator_evidence_fields(),
        "official_success": round_official,
        # Whether this round may share a denominator with another
        # one, carried on the round record so a reader never has to
        # re-derive it from the verdict's abort_reason. The
        # environment that owns the upstream protocol decides; a
        # backend that publishes no comparability leaves it null
        # rather than being assumed comparable.
        **_round_comparability(verdict),
        **round_scene_field(loop.plan, loop.reset_record),
        "decisions": loop.round_decisions,
        "refused_proposals": loop.round_refused_proposals,
        "wait_accounting": loop.native_allowance.wait_accounting(),
        "model_calls": loop.round_model_calls,
        "policy_state_steps": {
            "start": loop.round_start_step,
            "end_exclusive": len(loop.state_store.steps),
        },
        "verdict": verdict,
        "official_verdict_checker_divergence": loop.verdict_guard.divergence,
        "environment_status": loop.environment_status,
        # Only a round a v2 finish contract ended or recorded carries it.
        **({"policy_finish": loop.policy_finish} if loop.policy_finish is not None else {}),
        "started_unix_s": loop.round_started,
        "finished_unix_s": round_finished,
        "wall_duration_s": round_wall_duration_s,
        "wall_duration_clock": "monotonic",
    }


def _write_round_evidence(
    loop: EpisodeLoop, round_result: dict[str, Any], round_finished: float,
) -> None:
    round_root = loop.run_root / "rounds" / f"round-{loop.round_index:04d}"
    try:
        _atomic_json(round_root / "result.json", round_result)
    except Exception as exc:
        loop.record_auxiliary_failure(
            "post_terminal:round_result_write", exc,
        )
    verdict_source = loop.native / "official-verdict.json"
    if verdict_source.is_file():
        verdict_copy = (
            loop.native / "rounds" / f"round-{loop.round_index:04d}"
            / "official-verdict.json"
        )
        try:
            verdict_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verdict_source, verdict_copy)
        except Exception as exc:
            loop.record_auxiliary_failure(
                "post_terminal:verdict_copy", exc,
            )
    try:
        _append(loop.events, {
            "type": "round_finished",
            "round_index": loop.round_index,
            "result": round_result,
            "time": round_finished,
        })
    except Exception as exc:
        loop.record_auxiliary_failure(
            "post_terminal:round_finished_event", exc,
        )


def start_next_round(loop: EpisodeLoop) -> None:
    """Reset the world for the next round and publish its first observation."""

    try:
        loop.projector.close_observation()
    except Exception as exc:
        loop.record_auxiliary_failure(
            "post_terminal:projector_close", exc,
        )
    verdict_source = loop.native / "official-verdict.json"
    try:
        verdict_source.unlink(missing_ok=True)
    except Exception as exc:
        loop.record_auxiliary_failure(
            "post_terminal:verdict_rotation", exc,
        )
    next_round = loop.round_index + 1
    reset_tools = getattr(loop.tools, "reset_round", None)
    if not callable(reset_tools):
        raise SupervisorError(
            "ToolRuntime does not support persistent round reset"
        )
    reset_tools(next_round)
    _reset_round_state(loop, next_round)
    round_index = loop.round_index
    loop.reset_record = loop.runtime.restart_round(
        round_index,
        **(
            {"seed": loop.expected_round_seed(round_index)}
            if loop.round_seeds else {}
        ),
    )
    loop.validate_reset_seed(loop.reset_record, round_index)
    # Before anything of this round reaches the policy: prove the
    # world came back and that no Control file or tool registry
    # still carries the previous round. A refusal here is an
    # infrastructure fault, never a task verdict.
    loop.audit_round_isolation(loop.reset_record, round_index)
    loop.require_reset_descriptor(
        loop.reset_record,
        invalid="environment round reset descriptor is invalid",
        drift="environment runtime descriptor drift after round reset",
    )
    _append(loop.events, {
        "type": "reset", "record": loop.reset_record,
        "round_index": round_index, "time": time.time(),
    })
    loop.round_reset_notice = loop.session_round_notice(round_index)
    _status(
        loop.status_path, run_id=loop.run_id, state="running",
        decisions=loop.decisions, refused_proposals=loop.refused_proposals,
        model_calls=loop.model_calls,
        official_success=None, round_index=round_index,
        rounds=loop.round_limit,
    )
    loop.pointer, loop.environment_status = publish_observation(loop, "reset")


def _reset_round_state(loop: EpisodeLoop, round_index: int) -> None:
    """Every per-round counter and record starts over; session totals do not."""

    loop.round_index = round_index
    loop.interaction = loop.interaction_for(round_index)
    loop.interaction.start()
    loop.round_decisions = loop.round_refused_proposals = 0
    loop.native_allowance.reset()
    loop.round_model_calls = 0
    loop.round_started = time.time()
    loop.round_started_monotonic_ns = time.monotonic_ns()
    loop.round_start_step = len(loop.state_store.steps)
    loop.verdict = None
    loop.verdict_guard.reset()
    loop.termination = None
    loop.termination_authority = TerminationAuthority.NONE
    loop.policy_finish = None
    loop.previous_public_action = None
    loop.pending_execution = None
    loop.last_observation_image = None
    loop.stagnation = StagnationGuard(
        loop.control / "rounds" / f"round-{round_index:04d}"
        / "stagnation.json"
    )
