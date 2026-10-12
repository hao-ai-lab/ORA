"""The trusted observe -> policy turn -> execute loop of one episode.

`coordinator.run_episode` binds a RunPlan to its EnvironmentSession,
PolicySession and ToolRuntime (`EpisodeSessions`) and hands them to
`EpisodeLoop.run`. Each pass of `EpisodeLoop._run_rounds` makes at most one
policy turn::

    published observation
      -> trusted stops: environment terminal, operator signal, budgets
      -> policy turn (PolicyTurnInput -> ActionProposal | other outcome)
      -> Supervisor executes the accepted proposal and records its receipt
      -> fresh observation -> environment or control-fault boundary

A trusted boundary sets `termination`; the next pass closes the round
(`episode_rounds.finish_round`) and either resets the world for the next round
of a multi-round session or leaves the loop for
`episode_result.finish_session`. An exception anywhere in that path becomes a
trusted fault result (`episode_result.fault_result`). Observation publishing,
proposal dispatch and the terminal policy lifecycle live in the sibling
`episode_*` modules.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent import ActionProposal, PolicyRuntimeEvent, PolicyTurnInput
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.environment import EnvironmentSession, resolve_runtime_descriptor

from .context_manifest import build_context_manifest, context_placeholder
from .decision_accounting import is_zero_write_refusal
from .episode_actions import dispatch_action, post_action_boundary
from .episode_bindings import SupervisorError
from .episode_observation import publish_observation
from .episode_records import _append, _atomic_json, _status
from .episode_result import fault_result, finish_session
from .episode_rounds import finish_round, start_next_round
from .episode_terminal import (
    preserve_terminal_observation_without_policy_turn,
    stop_at_environment_boundary,
)
from .interaction import EpisodeInteractionState
from .live import HarnessLiveEvent, LiveInteractionPort, NullLiveInteractionPort
from .operator_stop import act_unless_stopped
from .policy_finish import infrastructure_abort, read_policy_finish
from .robodojo_round_scene import layout_change_sentence, require_round_scene
from .round_isolation import RoundIsolationError, require_round_isolation
from .stagnation import StagnationGuard
from .termination import (
    TerminationAuthority,
    TerminationDecision,
    budget_boundary,
    environment_boundary,
    infrastructure_boundary,
    operator_boundary,
    policy_outcome_decision,
)
from .verdict_guard import RoundVerdictGuard
from .wait_accounting import RoundActionLedger


@dataclass(frozen=True)
class EpisodeSessions:
    """What `coordinator.run_episode` binds before the loop starts."""

    runtime_config: Mapping[str, Any]
    runtime: EnvironmentSession
    runtime_descriptor: dict[str, Any]
    compiled_policy: Mapping[str, Any]
    state_store: PolicyStateStore
    projector: ContextProjector
    tools: Any
    planner: Any


class EpisodeLoop:
    """One run's state and its causal loop.

    Session totals (`decisions`, `refused_proposals`, `model_calls`,
    `round_results`) span every round; the `round_*` counters, `verdict`,
    `termination`, `termination_authority`, `interaction` and `stagnation`
    restart with each round of a multi-round session.
    """

    def __init__(
        self, *, run_root: Path, plan: Mapping[str, Any], run_id: str,
        live_port: LiveInteractionPort | None,
    ) -> None:
        """Create the run directories, announce the run and take the stop signals."""

        run_root = run_root.resolve()
        self.run_root = run_root
        self.plan = plan
        self.run_id = run_id
        self.status_path = run_root / "status.json"
        self.events = run_root / "events.jsonl"
        self.workspace = run_root / "sandbox/workspace"
        self.control = run_root / "control"
        self.native = run_root / "native"
        self.observation_root = run_root / "policy-observations"
        for path in (self.workspace, self.control, self.native, self.observation_root):
            path.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.started_monotonic_ns = time.monotonic_ns()
        _append(self.events, {"type": "run_started", "run_id": run_id, "time": self.started})
        session_config = plan.get("session")
        self.session_config = session_config
        self.multi_round = isinstance(session_config, Mapping)
        self.round_limit = int(session_config["rounds"]) if self.multi_round else 1
        self.round_seeds = (
            tuple(session_config.get("round_seeds", ()))
            if self.multi_round else ()
        )
        self.interaction = self.interaction_for(0)
        self.interaction.start()
        self.operator_port = live_port or NullLiveInteractionPort()
        self.live_sequence = 0
        self.auxiliary_failures: list[dict[str, Any]] = []
        self.interrupted = False
        # Set once the PolicySession has started; an operator signal then
        # also interrupts its in-flight turn.
        self.planner: Any = None
        self.previous_term = signal.signal(signal.SIGTERM, self._on_operator_signal)
        self.previous_int = signal.signal(signal.SIGINT, self._on_operator_signal)

    # -- identity, evidence and notification helpers -------------------------

    def expected_round_seed(self, index: int) -> int:
        return int(self.round_seeds[index]) if self.round_seeds else int(self.plan["seed"])

    def validate_reset_seed(self, reset: Mapping[str, Any], index: int) -> None:
        actual = reset.get("seed")
        expected = self.expected_round_seed(index)
        if actual != expected:
            raise SupervisorError(
                "environment round seed drift: "
                f"round {index} expected {expected}, got {actual!r}"
            )
        require_round_scene(self.plan, reset, index)

    def interaction_for(self, round_index: int) -> EpisodeInteractionState:
        evidence_path = (
            self.native / "rounds" / f"round-{round_index:04d}" / "episode-interaction.json"
            if self.multi_round else self.native / "episode-interaction.json"
        )
        return EpisodeInteractionState(
            run_id=(f"{self.run_id}:round-{round_index:04d}" if self.multi_round else self.run_id),
            evidence_path=evidence_path,
        )

    def record_auxiliary_failure(self, stage: str, exc: BaseException) -> None:
        record = {
            "stage": stage,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "time": time.time(),
        }
        self.auxiliary_failures.append(record)
        try:
            _append(self.events, {"type": "auxiliary_failure", **record})
        except Exception:
            pass

    def publish_live(self, kind: str, payload: Mapping[str, Any]) -> None:
        try:
            self.operator_port.publish(HarnessLiveEvent(
                sequence=self.live_sequence, kind=kind, payload=dict(payload),
            ))
        except Exception as exc:
            self.record_auxiliary_failure(f"live_port:{kind}", exc)
        self.live_sequence += 1

    def _on_operator_signal(self, _signum: int, _frame: Any) -> None:
        self.interrupted = True
        current_planner = self.planner
        if current_planner is not None:
            threading.Thread(
                target=current_planner.interrupt,
                args=("operator_interrupt",), daemon=True,
            ).start()

    def _should_stop(self) -> bool:
        return bool(self.interrupted)

    def deliver_policy_event(self, event: PolicyRuntimeEvent, *, stage: str) -> bool:
        try:
            self.planner.accept_runtime_event(event)
            return True
        except Exception as exc:
            self.record_auxiliary_failure(f"policy_callback:{stage}", exc)
            return False

    def record_termination_decision(
        self,
        decision: TerminationDecision,
        *,
        source: str,
        policy_outcome: str | None = None,
        policy_finish: Mapping[str, Any] | None = None,
    ) -> None:
        _append(self.events, {
            "type": "termination_authority_decision",
            "decision": decision.as_dict(),
            "source": source,
            "policy_outcome": policy_outcome,
            "round_index": self.round_index,
            "time": time.time(),
            # Only a v2 finish contract records it; other events keep their shape.
            **({"policy_finish": dict(policy_finish)} if policy_finish is not None else {}),
        })

    def session_round_notice(self, index: int) -> dict[str, Any]:
        instruction = (
            "This is a new independent environment round in the same AgentSession. "
            "The Supervisor resets after trusted success, finish_episode, or a round "
            "budget boundary; do not attempt to reset the environment yourself. "
            "Prior scene state and coordinates are stale. Re-ground from the current "
            "observation and retain useful experience."
        )
        instruction += layout_change_sentence(self.plan, index)
        return {
            "schema": "ora.session-round-notice.v1",
            "kind": "session_round_start",
            "round": index + 1,
            "total_rounds": self.round_limit,
            "budget_scope": "per_round",
            "instruction": instruction,
        }

    def require_reset_descriptor(
        self, reset: Mapping[str, Any], *, invalid: str, drift: str,
    ) -> None:
        """Refuse a reset whose public environment context left the RunPlan."""

        observed_environment = reset.get("public_environment_context")
        if observed_environment is None:
            return
        if not isinstance(observed_environment, Mapping):
            raise SupervisorError(invalid)
        observed_descriptor = resolve_runtime_descriptor(
            self.runtime_config,
            action_codec=self.policy.get("action_codec"),
            observed_public_context=observed_environment,
        ).as_dict()
        if observed_descriptor != self.runtime_descriptor:
            raise SupervisorError(drift)

    def audit_round_isolation(self, reset: Mapping[str, Any], round_index: int) -> None:
        """Prove no Control file or tool registry still carries an earlier round."""

        try:
            report = require_round_isolation(
                self.tools, reset, round_index=round_index,
            )
        except RoundIsolationError as error:
            raise SupervisorError(str(error)) from error
        _append(self.events, {
            "type": "round_start_isolation",
            "record": report,
            "round_index": round_index,
            "time": time.time(),
        })

    # -- lifecycle -------------------------------------------------------------

    def run(self, sessions: EpisodeSessions) -> dict[str, Any]:
        """Run the bound episode and return its result record.

        Any exception after the running status is written becomes a trusted
        fault result; the sessions and signal handlers are always released.
        """

        self._attach(sessions)
        self._init_episode_state()
        _status(self.status_path, run_id=self.run_id, state="running", decisions=0, model_calls=0, official_success=None)
        try:
            self._start_environment()
            self._run_rounds()
            return finish_session(self)
        except Exception as exc:
            return fault_result(self, exc)
        finally:
            self._close()

    def _attach(self, sessions: EpisodeSessions) -> None:
        self.runtime_config = sessions.runtime_config
        self.runtime = sessions.runtime
        self.runtime_descriptor = sessions.runtime_descriptor
        self.state_store = sessions.state_store
        self.projector = sessions.projector
        self.tools = sessions.tools
        self.policy = self.plan["policy"]
        self.planner = sessions.planner
        _atomic_json(
            self.run_root / "context-manifest.json",
            build_context_manifest(
                self.plan, compiled_policy=sessions.compiled_policy,
                context=context_placeholder(self.plan), run_id=self.run_id,
                workspace_root=self.run_root.parents[3],
            ),
        )

    def _init_episode_state(self) -> None:
        plan = self.plan
        self.stagnation = StagnationGuard(
            self.control / "rounds/round-0000/stagnation.json"
            if self.multi_round else self.control / "stagnation.json"
        )
        self.decisions = 0
        self.model_calls = 0
        self.round_decisions = 0
        self.refused_proposals = self.round_refused_proposals = 0
        self.native_allowance = RoundActionLedger(plan)
        self.round_model_calls = 0
        self.round_index = 0
        self.round_started = self.started
        self.round_started_monotonic_ns = self.started_monotonic_ns
        self.round_start_step = 0
        self.round_results: list[dict[str, Any]] = []
        self.round_reset_notice: dict[str, Any] | None = None
        self.verdict: dict[str, Any] | None = None
        self.verdict_guard = RoundVerdictGuard(
            runtime=self.runtime, control=self.control, events=self.events,
            surface=self.tools.surface, write_json=_atomic_json, append=_append,
            publish_live=self.publish_live,
            record_failure=self.record_auxiliary_failure,
        )
        self.termination: str | None = None
        self.termination_authority = TerminationAuthority.NONE
        # The round's v2 finish summary (policy_finish.read_policy_finish).
        self.policy_finish: dict[str, Any] | None = None
        self.stop_session = False
        self.previous_public_action: dict[str, Any] | None = None
        self.last_observation_image: Path | None = None
        self.pending_execution: dict[str, Any] | None = None
        self.planner_closed = False
        self.reset_record: dict[str, Any] | None = None
        self.pointer: dict[str, Any] | None = None
        self.environment_status: dict[str, Any] | None = None

    def _start_environment(self) -> None:
        self.reset_record = self.runtime.start()
        if self.multi_round:
            self.validate_reset_seed(self.reset_record, 0)
        self.require_reset_descriptor(
            self.reset_record,
            invalid="environment reset descriptor is invalid",
            drift="environment runtime descriptor drift",
        )
        _append(self.events, {"type": "reset", "record": self.reset_record, "time": time.time()})
        if self.multi_round:
            # Round 0 of a persistent session is audited on the same rule, so
            # a Control root that a crashed predecessor left dirty is refused
            # before the first turn rather than at the first round boundary.
            self.audit_round_isolation(self.reset_record, 0)

    def _close(self) -> None:
        if self.planner is not None and not self.planner_closed:
            try:
                self.planner.close()
            except Exception as exc:
                self.record_auxiliary_failure("cleanup:planner_close", exc)
        try:
            self.runtime.close()
        except Exception as exc:
            self.record_auxiliary_failure("cleanup:environment_close", exc)
        try:
            self.interaction.close()
        except Exception as exc:
            self.record_auxiliary_failure("cleanup:interaction_close", exc)
        try:
            signal.signal(signal.SIGTERM, self.previous_term)
            signal.signal(signal.SIGINT, self.previous_int)
        except Exception as exc:
            self.record_auxiliary_failure("cleanup:signal_restore", exc)

    # -- the causal loop -------------------------------------------------------

    def _run_rounds(self) -> None:
        """Alternate policy turns and executions until the session ends."""

        if self.multi_round:
            self.round_reset_notice = self.session_round_notice(0)
        self.pointer, self.environment_status = publish_observation(self, "reset")
        while True:
            if self.termination is not None:
                if finish_round(self):
                    break
                start_next_round(self)
                continue
            if self.environment_status["terminal"]:
                self._stop_at_environment_status()
                continue
            if self.interrupted:
                self._stop_on_operator_signal()
                continue
            if self._check_budgets():
                continue
            outcome, call_increment = self._policy_turn()
            if self.round_model_calls > self.plan["budgets"]["max_model_calls"]:
                self._stop_after_model_call_overrun(outcome, call_increment)
                continue
            if not isinstance(outcome, ActionProposal):
                self._settle_policy_outcome(outcome, call_increment)
                continue
            self._execute_proposal(call_increment)

    def _stop_at_environment_status(self) -> None:
        """The environment reported a terminal status before the next turn."""

        environment_reason = str(
            self.environment_status.get("reason") or "trusted_runtime_boundary"
        )
        decision = (
            infrastructure_boundary(environment_reason, scope="round")
            if environment_reason == "libero_control_health_fault"
            else environment_boundary(environment_reason)
        )
        stop_at_environment_boundary(
            self, decision, environment_reason, terminal_turn_allowed=True,
        )

    def _stop_on_operator_signal(self) -> None:
        decision = operator_boundary()
        self.record_termination_decision(decision, source="operator_signal")
        self.interaction.abort("operator_interrupt")
        self.termination = decision.reason
        self.termination_authority = decision.authority
        self.stop_session = True

    def _exhausted_budget(self) -> str | None:
        """Name the first exhausted round budget, checked in a fixed order."""

        budgets = self.plan["budgets"]
        if time.time() - self.round_started >= budgets["wall_time_s"]:
            return "wall_time_budget"
        if self.round_decisions >= budgets["max_robot_decisions"]:
            return "robot_decision_budget"
        if self.round_model_calls >= budgets["max_model_calls"]:
            return "model_call_budget"
        return None

    def _check_budgets(self) -> bool:
        """End the round at an exhausted budget before another policy turn."""

        reason = self._exhausted_budget()
        if reason is None:
            return False
        decision = budget_boundary(reason)
        self.record_termination_decision(decision, source="budget_guard")
        self.interaction.abort(decision.reason)
        if self.multi_round:
            preserve_terminal_observation_without_policy_turn(self)
        self._apply_budget_decision(decision)
        return True

    def _apply_budget_decision(self, decision: TerminationDecision) -> None:
        if decision.abort_environment:
            self.runtime.abort(decision.reason)
        self.termination = decision.reason
        self.termination_authority = decision.authority
        if decision.persist_official_verdict:
            self.verdict = self.verdict_guard.persist(self.round_index)

    def _policy_turn(self) -> tuple[Any, int]:
        """Ask the policy for one outcome on the current observation."""

        budgets = self.plan["budgets"]
        turn = PolicyTurnInput(
            task_id=str(self.plan["task"]),
            task_instruction=self.plan.get("task_instruction", self.plan["task"]),
            context=self.pointer,
            image_path=self.workspace / self.pointer["main_image"],
            observation_id=self.interaction.observation_id,
            observation_step=self.interaction.observation_step,
            budget={
                "robot_decisions": (
                    budgets["max_robot_decisions"] - self.round_decisions
                ),
                "model_calls": (
                    budgets["max_model_calls"] - self.round_model_calls
                ),
                "wall_time_s": max(
                    0,
                    budgets["wall_time_s"]
                    - (time.time() - self.round_started),
                ),
                **self.native_allowance.fields(),
            },
            interaction_phase=self.interaction.phase,
            fresh_observation_required=self.interaction.fresh_observation_required,
            allow_reminder=(
                budgets["max_model_calls"] - self.round_model_calls >= 2
            ),
        )
        self.interaction.begin_policy()
        outcome = act_unless_stopped(
            self.planner, turn, self._should_stop, self.events,
        )
        call_increment = (
            outcome.model_calls
            if self.planner.capabilities.runtime_kind != "scripted" else 0
        )
        self.model_calls += call_increment
        self.round_model_calls += call_increment
        return outcome, call_increment

    def _stop_after_model_call_overrun(self, outcome: Any, call_increment: int) -> None:
        """The turn itself spent past the model-call budget.

        A v2 finish emitted on that turn is still recorded on the round. An
        abort admitted on recorded infrastructure evidence keeps its own
        boundary -- the round is an invalid sample whatever the budget says --
        and any other finish leaves the budget in charge (``ended_round``
        false).
        """

        policy_finish = self._policy_finish_of(outcome, ended_round=True)
        if infrastructure_abort(policy_finish):
            self._settle_policy_outcome(outcome, call_increment)
            return
        if policy_finish is not None:
            self.policy_finish = {**policy_finish, "ended_round": False}
        decision = budget_boundary("model_call_budget")
        self.record_termination_decision(
            decision, source="post_policy_budget_guard",
            policy_finish=self.policy_finish if policy_finish is not None else None,
        )
        self.interaction.record_model_calls(call_increment)
        self.interaction.abort(decision.reason)
        self._apply_budget_decision(decision)

    def _policy_finish_of(self, outcome: Any, *, ended_round: bool) -> dict[str, Any] | None:
        """The v2 finish summary a lifecycle request left in Control, or ``None``."""

        if getattr(outcome, "kind", None) != "lifecycle":
            return None
        return read_policy_finish(self.control, self.tools.surface, ended_round=ended_round)

    def _settle_policy_outcome(self, outcome: Any, call_increment: int) -> None:
        """Apply a turn that returned no action (NoAction or a lifecycle request)."""

        policy_finish = self._policy_finish_of(outcome, ended_round=True)
        if policy_finish is not None:
            self.policy_finish = policy_finish
        decision = policy_outcome_decision(
            outcome, infrastructure_abort=infrastructure_abort(policy_finish),
        )
        self.interaction.record_policy_outcome(
            kind=outcome.kind,
            model_calls=call_increment,
            reason=decision.reason,
            terminal=decision.terminates,
        )
        self.record_termination_decision(
            decision,
            source="policy_outcome",
            policy_outcome=outcome.kind,
            policy_finish=policy_finish,
        )
        if not decision.terminates:
            if call_increment == 0:
                zero_progress = infrastructure_boundary(
                    "zero_progress_policy_no_action"
                )
                self.record_termination_decision(
                    zero_progress,
                    source="policy_progress_watchdog",
                    policy_outcome=outcome.kind,
                )
                self.interaction.abort(zero_progress.reason)
                self.termination = zero_progress.reason
                self.termination_authority = zero_progress.authority
                self.stop_session = True
            return
        self.termination = decision.reason
        self.termination_authority = decision.authority
        # A round-scoped infrastructure boundary (an admitted v2 abort) ends
        # this round only; the environment round is aborted below.
        if decision.scope == "session" and (
            decision.authority is TerminationAuthority.INFRASTRUCTURE
            or decision.authority is TerminationAuthority.OPERATOR
        ):
            self.stop_session = True
            return
        if decision.abort_environment:
            self.runtime.abort(decision.reason)
        if decision.persist_official_verdict:
            self.verdict = self.verdict_guard.persist(self.round_index)

    # -- executing one accepted proposal ---------------------------------------

    def _execute_proposal(self, call_increment: int) -> None:
        """Execute the persisted proposal, record its receipt, observe again."""

        action, current = self._accept_submitted_action(call_increment)
        execute_started_ns = time.monotonic_ns()
        receipt, captured_after_action = dispatch_action(
            self, action, current, should_stop=self._should_stop,
        )
        execute_duration_s = round(
            max(0, time.monotonic_ns() - execute_started_ns) / 1e9, 6
        )
        self._record_receipt(action, receipt, execute_duration_s)
        self.pointer, self.environment_status = publish_observation(
            self, "after_action", captured_observation=captured_after_action,
        )
        boundary = post_action_boundary(receipt, self.environment_status)
        if boundary is not None:
            decision, environment_reason = boundary
            stop_at_environment_boundary(
                self, decision, environment_reason,
                terminal_turn_allowed=(
                    decision.authority != TerminationAuthority.INFRASTRUCTURE
                ),
            )

    def _accept_submitted_action(
        self, call_increment: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the persisted proposal and bind it to the current observation."""

        action_path = self.control / "submitted-action.json"
        action = json.loads(action_path.read_text(encoding="utf-8"))
        self.interaction.record_policy_outcome(
            kind="action",
            model_calls=call_increment,
            action_id=str(action.get("action_id") or "unidentified-action"),
        )
        current = json.loads((self.control / "current-observation.json").read_text(encoding="utf-8"))
        if action["bound_observation_sha256"] != current["observation_sha256"]:
            raise SupervisorError("submitted action provenance mismatch")
        _append(self.events, {"type": "action_submitted", "action": action, "time": time.time()})
        self.interaction.begin_execution()
        self.publish_live("action_submitted", {
            "action_id": action.get("action_id"),
            "observation_id": self.interaction.observation_id,
            "phase": self.interaction.phase,
        })
        return action, current

    def _record_receipt(
        self, action: dict[str, Any], receipt: dict[str, Any],
        execute_duration_s: float,
    ) -> None:
        # A certified zero-write refusal spends no robot decision.
        consumed_decision = not is_zero_write_refusal(receipt)
        self.decisions += consumed_decision
        self.round_decisions += consumed_decision
        self.refused_proposals += not consumed_decision
        self.round_refused_proposals += not consumed_decision
        self.native_allowance.record(receipt, action)
        self.previous_public_action = action
        self.pending_execution = {"action": action, "receipt": receipt}
        _append(self.events, {
            "type": "motion_receipt", "record": receipt, "time": time.time(),
            "duration_s": execute_duration_s, "duration_clock": "monotonic",
            "consumed_decision": consumed_decision,
        })
        receipt_id = receipt.get("receipt_id") or receipt.get("id")
        self.interaction.record_execution(
            receipt_id=str(receipt_id) if receipt_id is not None else None,
            consumed_decision=consumed_decision,
        )
        self.deliver_policy_event(
            PolicyRuntimeEvent(
                kind="environment_receipt", payload=receipt,
                observation_id=self.interaction.observation_id,
                observation_step=self.interaction.observation_step,
            ),
            stage="environment_receipt",
        )
        _status(self.status_path, run_id=self.run_id, state="running", decisions=self.decisions, refused_proposals=self.refused_proposals, model_calls=self.model_calls, official_success=None)
