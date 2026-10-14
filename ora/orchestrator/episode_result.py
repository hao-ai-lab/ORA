"""The two ways an episode ends: a session result or a trusted fault result.

`finish_session` runs after the loop leaves through a trusted boundary. It
derives the session termination and official outcome, closes the policy state and writes `result.json`,
the `run_finished` event and the final status. `fault_result` replaces all of
that when any exception escapes the loop: it classifies the fault, quiesces
the PolicySession and keeps an already persisted behavioral verdict where the
termination authority allows it.
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from ora.agent import PolicyRuntimeEvent
from ora.agent.native.constants import NATIVE_CAUSAL_SLOTS

from .episode_bindings import robodojo_official_complete
from .episode_records import _append, _atomic_json, _status
from .faults import HarnessFault, classify_harness_fault
from .interaction import InteractionStateError
from .round_aggregates import run_round_aggregates
from .run_artifacts import _reconciled_provider_model_calls
from .termination import TerminationAuthority

if TYPE_CHECKING:
    from .episode_loop import EpisodeLoop

_RUN_RESULT_SCHEMA = "ora.run-result.v1"
_SESSION_RESULT_SCHEMA = "ora.session-result.v1"


def _quiesce_policy_after_fault(planner: Any) -> None:
    """Stop the wire sidecar before the fault result counts completions."""

    try:
        planner.close()
    except Exception:
        # A timed-out managed call intentionally leaves its AgentSession step
        # pending, so lifecycle close may fail. The managed adapter still
        # closes its wire proxy in a finally block; the incomplete session remains
        # evidence for evaluation rather than replacing the original fault.
        pass


def _elapsed_s(started_monotonic_ns: int) -> float:
    return round(max(0, time.monotonic_ns() - started_monotonic_ns) / 1e9, 6)


# -- session result ------------------------------------------------------------


def finish_session(loop: EpisodeLoop) -> dict[str, Any]:
    """Write and return the result of a session that ended at a trusted boundary."""

    multi_round = loop.multi_round
    # A single round, or a session stopped early, ends with its round's
    # termination; a multi-round session that ran all rounds ends on its
    # round limit.
    keeps_round_termination = not multi_round or loop.stop_session
    session_termination = (
        loop.termination if keeps_round_termination else "session_round_limit"
    )
    session_authority = (
        loop.termination_authority if keeps_round_termination
        else TerminationAuthority.EXPERIMENT_BUDGET
    )
    if session_authority in {
        TerminationAuthority.OPERATOR,
        TerminationAuthority.INFRASTRUCTURE,
    }:
        state = "incomplete"
    else:
        state = "completed"
    official = (
        None
        if multi_round
        or session_authority is TerminationAuthority.INFRASTRUCTURE
        else loop.verdict.get("official_success") if loop.verdict else None
    )
    if loop.plan["runtime"]["kind"] == "robodojo" and not robodojo_official_complete(
        official, loop.round_results,
        multi_round=multi_round, round_limit=loop.round_limit,
    ):
        state = "incomplete"
    video = _close_rollout_evidence(loop, state, session_termination)
    loop.model_calls = _reconciled_provider_model_calls(
        loop.run_root, loop.plan, loop.model_calls,
    )
    result = _session_result(
        loop, state=state, official=official, video=video,
        session_termination=session_termination,
        session_authority=session_authority,
    )
    _add_round_records(loop, result)
    _publish_session_result(
        loop, result, state=state, official=official,
        session_termination=session_termination,
        session_authority=session_authority,
    )
    return result


def _close_rollout_evidence(
    loop: EpisodeLoop, state: str, session_termination: Any,
) -> Any:
    """Close the policy observation and state; return the operator video."""

    try:
        loop.projector.close_observation()
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:projector_close", exc)
    try:
        loop.state_store.finalize(state=state, termination=str(session_termination))
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:policy_state", exc)
    try:
        return loop.runtime.finalize_operator_video()
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:operator_video", exc)
        return None


def _session_result(
    loop: EpisodeLoop, *, state: str, official: Any, video: Any,
    session_termination: Any, session_authority: TerminationAuthority,
) -> dict[str, Any]:
    multi_round = loop.multi_round
    finished = time.time()
    wall_duration_s = _elapsed_s(loop.started_monotonic_ns)
    return {
        "schema": (
            _SESSION_RESULT_SCHEMA if multi_round else _RUN_RESULT_SCHEMA
        ),
        "run_id": loop.run_id, "plan_sha256": loop.plan["plan_sha256"],
        "state": state, "termination": session_termination,
        "termination_authority": session_authority.value,
        "behavioral_sample_valid": (
            session_authority
            not in {
                TerminationAuthority.INFRASTRUCTURE,
                TerminationAuthority.OPERATOR,
            }
            and not bool(loop.operator_message_count)
        ),
        **(loop.operator_evidence_fields() if "operator_control" in loop.plan else {}),
        "official_success": official, "decisions": loop.decisions,
        "refused_proposals": loop.refused_proposals, "model_calls": loop.model_calls,
        "verdict": None if multi_round else loop.verdict,
        "official_verdict_checker_divergence": None if multi_round else loop.verdict_guard.divergence,
        "operator_video": video,
        "environment_status": loop.environment_status,
        "started_unix_s": loop.started, "finished_unix_s": finished,
        "wall_duration_s": wall_duration_s,
        "wall_duration_clock": "monotonic",
        "auxiliary_failures": list(loop.auxiliary_failures),
    }


def _add_round_records(loop: EpisodeLoop, result: dict[str, Any]) -> None:
    round_results = loop.round_results
    result.update(run_round_aggregates(round_results, multi_round=loop.multi_round))
    if loop.multi_round:
        result["session"] = dict(loop.session_config)
        result["rounds"] = round_results
    if not loop.multi_round and round_results and isinstance(
        round_results[0].get("policy_finish"), Mapping,
    ):
        # A single run carries its one round's v2 finish summary.
        result["policy_finish"] = copy.deepcopy(round_results[0]["policy_finish"])


def _publish_session_result(
    loop: EpisodeLoop, result: dict[str, Any], *, state: str, official: Any,
    session_termination: Any, session_authority: TerminationAuthority,
) -> None:
    """Persist the result, tell the policy, close it, then the final status."""

    try:
        _atomic_json(loop.run_root / "result.json", result)
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:result_write", exc)
    loop.deliver_policy_event(
        PolicyRuntimeEvent(
            kind="run_termination", payload=result,
            observation_id=loop.interaction.observation_id,
            observation_step=loop.interaction.observation_step,
        ),
        stage="run_termination",
    )
    try:
        loop.planner.close()
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:planner_close", exc)
    loop.planner_closed = True
    try:
        _append(loop.events, {
            "type": "run_finished", "result": result, "time": time.time(),
        })
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:run_finished_event", exc)
    status_fields: dict[str, Any] = {}
    if loop.multi_round:
        status_fields = {
            "round_count": len(loop.round_results),
            "round_success_count": result["round_success_count"],
        }
    try:
        _status(
            loop.status_path, run_id=loop.run_id, state=state, decisions=loop.decisions,
            refused_proposals=loop.refused_proposals,
            model_calls=loop.model_calls, official_success=official,
            termination=session_termination,
            termination_authority=session_authority.value,
            **status_fields,
        )
    except Exception as exc:
        loop.record_auxiliary_failure("finalize:status", exc)


# -- trusted fault result ------------------------------------------------------


def fault_result(loop: EpisodeLoop, exc: Exception) -> dict[str, Any]:
    """Write and return the result of an episode that ended in a trusted fault."""

    fault = _classified_fault(loop, exc)
    try:
        _atomic_json(loop.native / "runtime-fault.json", fault.as_dict())
    except Exception as write_exc:
        loop.record_auxiliary_failure("fault:record", write_exc)
    try:
        _append(loop.events, {
            "type": "runtime_fault", "fault": fault.as_dict(), "time": time.time(),
        })
    except Exception as write_exc:
        loop.record_auxiliary_failure("fault:event", write_exc)
    loop.publish_live("runtime_fault", fault.as_dict())
    try:
        loop.projector.close_observation()
    except Exception as close_exc:
        loop.record_auxiliary_failure("fault:projector_close", close_exc)
    try:
        loop.state_store.finalize(state="incomplete", termination="trusted_fault")
    except Exception as finalize_exc:
        loop.record_auxiliary_failure("fault:policy_state", finalize_exc)
    if loop.planner is not None and not loop.planner_closed:
        _quiesce_policy_after_fault(loop.planner)
        loop.planner_closed = True
    loop.model_calls = _reconciled_provider_model_calls(
        loop.run_root, loop.plan, loop.model_calls,
    )
    result, preserved_official = _fault_result_record(loop, exc)
    _publish_fault_result(loop, result, exc, preserved_official=preserved_official)
    return result


def _classified_fault(loop: EpisodeLoop, exc: Exception) -> HarnessFault:
    """Classify the fault and settle the interaction state it leaves behind."""

    interaction = loop.interaction
    fault = classify_harness_fault(exc, phase=interaction.phase)
    if (
        loop.policy.get("context_composition") in NATIVE_CAUSAL_SLOTS
        and interaction.phase == "fresh_observation_required"
        and fault.code == "post_mutation_observation_failure"
    ):
        fault = replace(
            fault, world_state_certainty="unknown", recoverable_at="operator",
        )
    if fault.world_state_certainty == "unknown" and interaction.phase in {
        "executing", "fresh_observation_required",
    }:
        interaction.mark_world_state_unknown(reason=fault.code)
    elif interaction.phase != "closed":
        try:
            interaction.abort("trusted_fault")
        except InteractionStateError:
            pass
    return fault


def _fault_result_record(
    loop: EpisodeLoop, exc: Exception,
) -> tuple[dict[str, Any], Any]:
    """The fault result and the official outcome it preserves, if any."""

    multi_round = loop.multi_round
    verdict = loop.verdict
    authority = loop.termination_authority
    finished = time.time()
    wall_duration_s = _elapsed_s(loop.started_monotonic_ns)
    preserved_official = (
        verdict.get("official_success")
        if isinstance(verdict, Mapping)
        and authority is not TerminationAuthority.INFRASTRUCTURE
        else None
    )
    behavioral_outcome_preserved = (
        verdict is not None
        and authority
        not in {
            TerminationAuthority.NONE,
            TerminationAuthority.INFRASTRUCTURE,
            TerminationAuthority.OPERATOR,
        }
    )
    result = {
        "schema": (
            _SESSION_RESULT_SCHEMA if multi_round else _RUN_RESULT_SCHEMA
        ),
        "run_id": loop.run_id, "plan_sha256": loop.plan["plan_sha256"],
        "state": "incomplete", "termination": "trusted_fault",
        "termination_authority": TerminationAuthority.INFRASTRUCTURE.value,
        "official_success": (
            None if multi_round else preserved_official
        ),
        "decisions": loop.decisions, "refused_proposals": loop.refused_proposals,
        "model_calls": loop.model_calls, "error": str(exc),
        "verdict": None if multi_round else verdict,
        "official_verdict_checker_divergence": None if multi_round else loop.verdict_guard.divergence,
        "behavioral_outcome_preserved": behavioral_outcome_preserved,
        **(loop.operator_evidence_fields() if "operator_control" in loop.plan else {}),
        **({"behavioral_sample_valid": False} if "operator_control" in loop.plan else {}),
        "auxiliary_failures": list(loop.auxiliary_failures),
        "started_unix_s": loop.started, "finished_unix_s": finished,
        "wall_duration_s": wall_duration_s,
        "wall_duration_clock": "monotonic",
    }
    result.update(run_round_aggregates(loop.round_results, multi_round=multi_round))
    if multi_round:
        result["session"] = dict(loop.session_config)
        result["rounds"] = loop.round_results
        if verdict is not None and not any(
            item.get("round_index") == loop.round_index for item in loop.round_results
        ):
            result["pending_round_verdict"] = {
                "round_index": loop.round_index,
                "seed": loop.expected_round_seed(loop.round_index),
                "termination_authority": authority.value,
                "official_success": preserved_official,
                "verdict": verdict,
            }
    return result, preserved_official


def _publish_fault_result(
    loop: EpisodeLoop, result: dict[str, Any], exc: Exception, *,
    preserved_official: Any,
) -> None:
    try:
        _atomic_json(loop.run_root / "result.json", result)
    except Exception as write_exc:
        loop.record_auxiliary_failure("fault:result_write", write_exc)
    if loop.planner is not None:
        try:
            loop.deliver_policy_event(
                PolicyRuntimeEvent(
                    kind="run_termination", payload=result,
                    observation_id=loop.interaction.observation_id,
                    observation_step=loop.interaction.observation_step,
                ),
                stage="fault_run_termination",
            )
        except Exception:
            pass
    try:
        _append(loop.events, {
            "type": "trusted_fault", "error": str(exc), "time": time.time(),
        })
    except Exception as write_exc:
        loop.record_auxiliary_failure("fault:trusted_fault_event", write_exc)
    try:
        _status(
            loop.status_path, run_id=loop.run_id, state="incomplete",
            decisions=loop.decisions, refused_proposals=loop.refused_proposals,
            model_calls=loop.model_calls,
            official_success=(None if loop.multi_round else preserved_official),
            error=str(exc),
            termination="trusted_fault",
            termination_authority=TerminationAuthority.INFRASTRUCTURE.value,
        )
    except Exception as status_exc:
        loop.record_auxiliary_failure("fault:status", status_exc)
