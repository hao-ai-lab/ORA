"""Provider → tool → result loop for one Native policy observation.

:meth:`NativeTurnLoop.run` handles one fresh observation. It opens the turn
(the observation message and its trace event), then repeats one provider
exchange until an action is submitted, the Episode ends or the call budget is
spent:

1. project the Context and add the request's context-source records;
2. persist the canonical and wire request, send it, and record the reply and
   its usage;
3. answer a reply that cannot act -- rejected or truncated output, or no tool
   call -- with a continuation message, or run its tool batch.

The Session owns every piece of state; this module only sequences it.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes
from ora.agent.tools.robot import ToolError
from ora.agent.providers.continuation import (  # noqa: F401  re-exported
    PROVIDER_CONTINUATION_MESSAGE_KEY,
)

from . import continuations, turn_state
from .constants import (  # noqa: F401
    MUTATION_TOOL, NATIVE_CAUSAL_SLOTS, NATIVE_SLOTS_V3,
)
from .context_sources import annotate_context_sources
from .continuations import Continuation
from .interruption import operator_interrupted
from .mutation_gates import mutation_protocol_error
from .provider_calls import _provider_attempt_evidence
from .provider_calls import adopt_synthesized_tool_call_id  # noqa: F401  re-exported
from .provider_reply import (
    add_usage, call_record, check_reply_completion, check_wire_echo, error_payload,
    execution_tool_calls, reply_message, response_payload,
)
from .tool_arguments import _arguments, _tool_error  # noqa: F401  _arguments re-exported
from .tool_batch import (
    BatchAdmission, EffectClassifier, ParsedCall, admit_tool_batch, parse_tool_calls,
)
from .turn_state import ProviderReply, ProviderRequest, Turn


class _NativeLoopHost(Protocol):
    """Session operations consumed by the provider/tool loop."""


def _provider_estimated_input_tokens(manifest: Mapping[str, Any]) -> int:
    """Select the pressure estimate without eagerly reading a legacy field."""

    value = manifest.get("pressure_estimated_input_tokens")
    if not isinstance(value, int) or isinstance(value, bool):
        value = manifest.get("raw_estimated_input_tokens")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PolicySessionError(
            "native Context projection lacks a positive input-token estimate"
        )
    return value


@dataclass(frozen=True)
class NativeTurnLoop:
    """Own the provider → tool → paired-result loop; Session owns state."""

    def run(
        self,
        session: _NativeLoopHost,
        *,
        task_instruction: str,
        context: Mapping[str, Any],
        image_path: Path,
        allow_reminder: bool = True,
        terminal_lifecycle: bool = False,
    ) -> dict[str, Any]:
        turn = _open_turn(
            session, task_instruction=task_instruction, context=context,
            image_path=image_path, allow_reminder=allow_reminder,
            terminal_lifecycle=terminal_lifecycle,
        )
        attempts = 0
        while attempts < turn.call_limit:
            if session.interrupted:
                return operator_interrupted(attempts)
            projected = _project_context(session, turn)
            if projected is None:
                return turn_state.context_exhausted(attempts)
            if session.interrupted:  # stopped during projection: send nothing
                return operator_interrupted(attempts)
            attempts += 1
            model_call_id = f"model-call-{session.call_sequence:06d}"
            session.call_sequence += 1
            if not _record_context_sources(session, turn, projected, model_call_id):
                return turn_state.context_exhausted(attempts - 1)
            reply = _exchange(session, turn, projected, model_call_id)
            if reply is None:  # the stop closed the provider mid-request
                return operator_interrupted(attempts)
            if session.interrupted:  # reply kept as evidence; no tool may run
                return operator_interrupted(attempts)
            if isinstance(reply.output_rejection, Mapping):
                _continue_same_observation(
                    session, turn, reply,
                    continuations.after_rejected_output(reply.output_rejection),
                )
                continue
            if reply.finish_reason == "length":
                _continue_same_observation(
                    session, turn, reply, continuations.after_truncated_output(),
                )
                continue
            if not reply.calls:
                if turn.terminal_lifecycle or not turn.allow_reminder:
                    return turn_state.no_action(attempts)
                _continue_same_observation(
                    session, turn, reply, continuations.after_no_action(
                        getattr(session.tools.surface, "mutation_tools", (MUTATION_TOOL,))
                    ),
                )
                continue
            outcome = _run_tool_batch(session, turn, reply, attempts)
            if outcome is not None:
                return outcome
        return turn_state.call_limit_reached(attempts)


# -- Opening the turn ------------------------------------------------------


def _open_turn(
    session: Any, *, task_instruction: str, context: Mapping[str, Any],
    image_path: Path, allow_reminder: bool, terminal_lifecycle: bool,
) -> Turn:
    """Append the fresh observation and its trace event; fix the call budget."""

    if session.closed or session.interrupted:
        raise PolicySessionError("native policy session is not active")
    request_deadline = turn_state.request_deadline(context)
    session._bind_system(task_instruction)
    step = session._observation_step(context)
    observation = session._observation_message(
        task_instruction=task_instruction, context=context, image_path=image_path,
        terminal_lifecycle=terminal_lifecycle,
    )
    session.messages.append(observation)
    session.pending_receipt = None
    session.pending_receipt_body = None
    observation_event = _append_observation_event(session, step, observation)
    session._append_round_static_facts(observation_event)
    return Turn(
        step=step,
        observation=observation,
        observation_event=observation_event,
        current_observation_id=observation["_observation"]["observation_id"],
        request_deadline=request_deadline,
        call_limit=turn_state.call_limit(session.max_model_calls_per_observation, context),
        allow_reminder=allow_reminder,
        terminal_lifecycle=terminal_lifecycle,
    )


def _append_observation_event(
    session: Any, step: int, observation: dict[str, Any],
) -> dict[str, Any]:
    """Trace the observation and bind the message to its source events."""

    observation_event = session._append(
        "observation_received",
        observation_step=step,
        payload={
            "observation_id": observation["_observation"]["observation_id"],
            "state_ref": observation["_observation"]["state_ref"],
            **({
                "causal_pair": bool(
                    observation.get("_causal_observation", {}).get("paired")
                ),
                "causal_cameras": list(
                    observation.get("_causal_observation", {}).get("cameras", [])
                ),
            } if session.context_composition in NATIVE_CAUSAL_SLOTS else {}),
        },
    )
    source_event_ids = [str(observation_event["event_id"])]
    receipt_metadata = observation["_observation"].get("receipt")
    if isinstance(receipt_metadata, Mapping) and isinstance(receipt_metadata.get("event_id"), str):
        source_event_ids.append(receipt_metadata["event_id"])
    if session.last_fresh_observation_event_id is not None:
        source_event_ids.append(session.last_fresh_observation_event_id)
    observation["_source_event_ids"] = source_event_ids
    return observation_event


# -- Context projection ----------------------------------------------------


def _project_context(session: Any, turn: Turn) -> dict[str, Any] | None:
    """Project the Context; None when it is exhausted (already traced)."""

    try:
        return session.pipeline.project(
            messages=session.messages, tools=session.tool_schemas, observation_step=turn.step
        )
    except PolicySessionError as exc:
        if not str(exc).startswith("context_exhausted"):
            raise
        session._append(
            "context_exhausted",
            visibility="derived",
            observation_step=turn.step,
            payload={"error": str(exc)},
        )
        return None


def _record_context_sources(
    session: Any, turn: Turn, projected: dict[str, Any], model_call_id: str,
) -> bool:
    """Add the request's context-source records; False when Context is exhausted."""

    try:
        annotate_context_sources(
            projected,
            composition=session.context_composition,
            observation=turn.observation,
            tool_schemas=session.tool_schemas,
            context_sources=session.context_sources,
        )
    except PolicySessionError as exc:
        if not str(exc).startswith("context_exhausted"):
            raise
        session._append(
            "context_exhausted", visibility="derived", observation_step=turn.step,
            model_call_id=model_call_id, payload={"error": str(exc)},
        )
        return False
    return True


# -- One provider exchange -------------------------------------------------


def _exchange(
    session: Any, turn: Turn, projected: dict[str, Any], model_call_id: str,
) -> ProviderReply | None:
    """Send one projected request and record its reply; None if a stop cut it."""

    request = _persist_request(session, turn, projected, model_call_id)
    response = _complete(session, turn, request, model_call_id)
    if response is None:
        return None
    reply = _record_reply(session, turn, request, response, model_call_id)
    _account_call(session, projected, request, response, reply)
    return reply


def _persist_request(
    session: Any, turn: Turn, projected: dict[str, Any], model_call_id: str,
) -> ProviderRequest:
    """Store the canonical and wire request and trace the projection behind them."""

    step = turn.step
    prepared_request = session.request_assembler.prepare(projected)
    request = dict(prepared_request.canonical)
    execution_request = dict(prepared_request.execution)
    request_ref = session.artifacts.put_bytes(
        canonical_bytes(request),
        kind="native_canonical_provider_request",
        suffix=".json",
        media_type="application/json",
    )
    wire_request = dict(prepared_request.wire)
    wire_request_ref = session.artifacts.put_bytes(
        canonical_bytes(wire_request),
        kind="native_provider_wire_request",
        suffix=".json",
        media_type="application/json",
    )
    condensation = projected["manifest"].get("context_condensation")
    if isinstance(condensation, Mapping):
        session._append(
            "context_condensation",
            observation_step=step,
            model_call_id=model_call_id,
            visibility="derived",
            payload={
                **copy.deepcopy(dict(condensation)),
                "next_main_canonical_request_sha256": request_ref["sha256"],
                "next_main_wire_request_sha256": wire_request_ref["sha256"],
            },
        )
    projection = session._append(
        "context_projected",
        causal_parent_id=str(turn.observation_event["event_id"]),
        observation_step=step,
        model_call_id=model_call_id,
        visibility="derived",
        payload={
            **dict(projected["manifest"]),
            "exact_request_sha256": request_ref["sha256"],
            "wire_request_sha256": wire_request_ref["sha256"],
        },
    )
    request_event = session._append(
        "provider_request",
        causal_parent_id=str(projection["event_id"]),
        observation_step=step,
        model_call_id=model_call_id,
        artifact_ref=wire_request_ref,
        payload={
            "model": session.model,
            "provider_protocol": str(
                getattr(session.provider, "protocol", "canonical_unknown")
            ),
            "canonical_request_ref": request_ref,
        },
    )
    return ProviderRequest(
        canonical=request, execution=execution_request, wire=wire_request,
        event=request_event,
    )


def _complete(
    session: Any, turn: Turn, request: ProviderRequest, model_call_id: str,
) -> Mapping[str, Any] | None:
    """Call the provider; a failure is traced, and swallowed only after a stop."""

    try:
        with _provider_attempt_evidence(session, request.event, turn.step, model_call_id,
                                        deadline_monotonic=turn.request_deadline):
            response = session.provider.complete(
                request=request.execution, model_call_id=model_call_id,
            )
    except Exception as exc:
        session._append(
            "provider_error",
            causal_parent_id=str(request.event["event_id"]),
            observation_step=turn.step,
            model_call_id=model_call_id,
            payload=error_payload(exc),
        )
        if session.interrupted:  # the stop closed the provider mid-request
            return None
        raise
    return response


def _record_reply(
    session: Any, turn: Turn, request: ProviderRequest, response: Mapping[str, Any],
    model_call_id: str,
) -> ProviderReply:
    """Trace the reply and keep its assistant message unless it was rejected."""

    check_wire_echo(response, request.wire)
    raw_ref = session.artifacts.put_json(
        response.get("raw_events", []), kind="native_provider_wire_response"
    )
    assistant, output_rejection = reply_message(response)
    usage = response.get("usage")
    response_event = session._append(
        "provider_response",
        causal_parent_id=str(request.event["event_id"]),
        observation_step=turn.step,
        model_call_id=model_call_id,
        payload=response_payload(
            response, assistant=assistant, usage=usage,
            output_rejection=output_rejection,
        ),
        artifact_ref=raw_ref,
    )
    finish_reason = response.get("finish_reason")
    calls = execution_tool_calls(assistant)
    check_reply_completion(finish_reason, calls, output_rejection)
    assistant["_source_event_ids"] = [str(response_event["event_id"])]
    # Rejected output remains exact response evidence, but its edited
    # assistant is not valid provider history (DeepSeek requires the
    # original reasoning on every retained assistant turn).
    if output_rejection is None:
        session.messages.append(assistant)
    return ProviderReply(
        model_call_id=model_call_id, assistant=assistant,
        response_event=response_event, usage=usage, finish_reason=finish_reason,
        output_rejection=output_rejection, calls=calls,
    )


def _account_call(
    session: Any, projected: dict[str, Any], request: ProviderRequest,
    response: Mapping[str, Any], reply: ProviderReply,
) -> None:
    """Record the call, its usage totals and the Provider usage anchor."""

    usage = reply.usage
    session.calls.append(call_record(
        model_call_id=reply.model_call_id, request=request.canonical,
        response=response, usage=usage, context_projection=projected["manifest"],
        output_rejection=reply.output_rejection,
    ))
    add_usage(
        usage, model_call_id=reply.model_call_id,
        reported_totals=session.reported_totals,
        usage_events=session.usage_events,
        context_window_tokens=session.context_window_tokens,
    )
    session.pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(
            projected["manifest"],
        ),
        input_tokens=(
            usage.get("input_tokens")
            if isinstance(usage, Mapping) else None
        ),
        projection_epoch=projected["manifest"].get(
            "projection_epoch",
        ),
        provider_wire_request_sha256=projected["manifest"].get(
            "provider_wire_request_sha256",
        ),
    )
    session._write_session_manifest(exit_reason=None)


# -- Replies that cannot act -----------------------------------------------


def _continue_same_observation(
    session: Any, turn: Turn, reply: ProviderReply, continuation: Continuation,
) -> None:
    """Trace why the reply cannot act and ask again for this observation."""

    event = session._append(
        continuation.event_type,
        causal_parent_id=str(reply.response_event["event_id"]),
        observation_step=turn.step,
        model_call_id=reply.model_call_id,
        visibility="derived",
        payload=continuation.payload,
    )
    session.messages.append({
        "role": "user",
        "content": continuation.content,
        "_source_event_ids": [str(event["event_id"])],
    })


# -- The tool batch --------------------------------------------------------


def _effect_classifier(session: Any) -> EffectClassifier | None:
    """Typed slots classify every call's effect through the gateway."""

    if session.context_composition not in NATIVE_CAUSAL_SLOTS:
        return None
    return lambda name, args: session.gateway.classify_effect(name, args)


def _run_tool_batch(
    session: Any, turn: Turn, reply: ProviderReply, attempts: int,
) -> dict[str, Any] | None:
    """Run one reply's tool calls; the turn's result once one ends the turn."""

    parsed, names = parse_tool_calls(
        reply.calls, assistant=reply.assistant, model_call_id=reply.model_call_id,
        classify_effect=_effect_classifier(session),
    )
    admission = admit_tool_batch(
        parsed, names, call_count=len(reply.calls),
        typed_effects=session.context_composition in NATIVE_CAUSAL_SLOTS,
        mutation_tools=set(getattr(session.tools.surface, "mutation_tools", (MUTATION_TOOL,))),
    )
    if session._execute_aggregated_read_only_batch(
        parsed=parsed, assistant=reply.assistant, response_event=reply.response_event,
        model_call_id=reply.model_call_id, step=turn.step,
    ):
        return None

    tool_images: list[dict[str, Any]] = []
    for index, parsed_call in enumerate(parsed):
        outcome = _run_tool_call(
            session, turn, reply, admission, index, parsed_call, tool_images, attempts,
        )
        if outcome is not None:
            return outcome
    session.messages.extend(tool_images)
    return None


def _run_tool_call(
    session: Any, turn: Turn, reply: ProviderReply, admission: BatchAdmission,
    index: int, parsed_call: ParsedCall, tool_images: list[dict[str, Any]],
    attempts: int,
) -> dict[str, Any] | None:
    """Propose, dispatch and record one call; the turn's result if it ends here."""

    call, name, args, effect, parse_error = parsed_call
    reported_name = name or "invalid_tool_call"
    call_id = str(call["id"])
    proposal_event = session._append(
        "tool_call_proposed",
        causal_parent_id=str(reply.response_event["event_id"]),
        observation_step=turn.step,
        model_call_id=reply.model_call_id,
        tool_call_id=call_id,
        payload={
            "name": reported_name, "arguments": args, "effect": effect,
            "batch_valid": index not in admission.refused_mutations,
        },
    )
    reply.assistant.setdefault("_source_event_ids", []).append(
        str(proposal_event["event_id"])
    )
    result = _dispatch_tool_call(
        session, turn, reply, admission, index, parsed_call,
        call_id=call_id, proposal_event=proposal_event,
    )
    image_message = session._record_tool_result(
        call_id=call_id,
        model_call_id=reply.model_call_id,
        step=turn.step,
        proposal_event=proposal_event,
        name=reported_name,
        result=result,
    )
    if image_message is not None:
        tool_images.append(image_message)
    return _turn_outcome(
        session, turn, reply, admission, name=name, effect=effect,
        call_id=call_id, result=result, attempts=attempts,
    )


def _dispatch_tool_call(
    session: Any, turn: Turn, reply: ProviderReply, admission: BatchAdmission,
    index: int, parsed_call: ParsedCall, *, call_id: str,
    proposal_event: Mapping[str, Any],
) -> dict[str, Any]:
    """The call's result: a refusal, or what the gateway returned."""

    _call, name, args, effect, parse_error = parsed_call
    if index in admission.refused_mutations:
        return _tool_error(parse_error or ToolError(
            "a world-mutating tool must be the only tool call in an assistant "
            "response; this mutation was not executed and the other calls in the "
            "response ran", code="mutation_batch_rejected",
            constraints={"other_calls_executed": admission.reads_run},
        ))
    if parse_error is not None or name is None or args is None:
        return _tool_error(parse_error or ToolError("invalid tool call"))
    protocol_error = mutation_protocol_error(
        composition=session.context_composition, effect=effect,
        terminal_lifecycle=turn.terminal_lifecycle,
    )
    if protocol_error is not None:
        return _tool_error(protocol_error)
    try:
        return dict(session.gateway.execute(
            name,
            args,
            call_context={
                "policy_session_id": session.policy_session_id,
                "model_call_id": reply.model_call_id,
                "tool_call_id": call_id,
                "proposal_event_id": proposal_event["event_id"],
                "observation_step": turn.step,
                "current_observation_event_id": (
                    turn.observation_event["event_id"]
                ),
                "preceding_action_step": session.last_action_step,
                "preceding_action_event_id": (
                    session.last_action_event_id
                ),
                "interaction_phase": "robot_rollout",
            },
        ))
    except Exception as exc:
        return _tool_error(exc)


def _turn_outcome(
    session: Any, turn: Turn, reply: ProviderReply, admission: BatchAdmission, *,
    name: str | None, effect: str | None, call_id: str,
    result: Mapping[str, Any], attempts: int,
) -> dict[str, Any] | None:
    """A submitted action or a finished Episode ends the turn."""

    runtime_metadata = result.get("_runtime", {})
    directive = (
        runtime_metadata.get("directive")
        if isinstance(runtime_metadata, Mapping)
        else None
    )
    if directive is None:
        directive = result.get("directive")
    is_mutation = admission.is_mutation(effect, name)
    if is_mutation and result.get("ok", True) and directive == "action_submitted":
        return _submit_action(session, turn, reply, name=name, call_id=call_id, attempts=attempts)
    if directive == "episode_finished":
        return turn_state.finished_episode(attempts)
    return None


def _submit_action(
    session: Any, turn: Turn, reply: ProviderReply, *, name: str | None, call_id: str,
    attempts: int,
) -> dict[str, Any]:
    """Trace the action the mutation persisted and hand it to the Supervisor."""

    action_path = session.tools.control_root / "submitted-action.json"
    if not action_path.is_file():
        raise PolicySessionError("successful mutation did not persist an action proposal")
    action = json.loads(action_path.read_text(encoding="utf-8"))
    action_ref = session.artifacts.put_json(action, kind="native_action_proposal")
    action_event = session._append(
        "action_proposed",
        observation_step=turn.step,
        model_call_id=reply.model_call_id,
        tool_call_id=call_id,
        payload={
            "action_id": action.get("action_id"),
            "observation_id": session._current_observation_id(),
        },
        artifact_ref=action_ref,
    )
    session.last_action_event_id = str(action_event["event_id"])
    session.last_action_step = turn.step
    session.last_action_id = str(action.get("action_id"))
    session.last_action_public_name = name
    reply.assistant.setdefault("_source_event_ids", []).append(
        str(action_event["event_id"])
    )
    return turn_state.action_submitted(action, attempts)
