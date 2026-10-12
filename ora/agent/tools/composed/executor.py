"""Call and execute a composed tool: one robot decision, one staged receipt.

**Call time** (:func:`submit`): arguments are checked against the compiled
input schema with the same validator the Runtime applies to every tool, the
rationale is checked, tunable parameters are resolved, and the body's
``plan(ctx, inputs, params)`` runs read-only. A ``sdk.refuse`` becomes a
structured ``ToolError`` and nothing is submitted; a ``sdk.commit`` becomes a
:class:`Commitment` and the immediate public result.

**Execution time** (:class:`Execution`): ``execute(ctx, c)`` is interpreted by
deterministic replay. :meth:`Execution.next` returns the next primitive action
as an :class:`ActionRequest`; the trusted executor dispatches it through the
Supervisor, classifies what happened (executed, not_reached, unavailable,
refused) and hands that back with :meth:`Execution.resolve`. When the body
returns, or the executor stops it (operator interrupt, terminal episode,
fault, spent budget), :meth:`Execution.next` returns :class:`Finished`.

The receipt follows the closed-skill conventions so the same projection,
command-tree and evaluation readers apply: child action ids are
``<parent>:<declared stage index:02d>-<stage>`` (a retry within a stage adds
``.<attempt>``), stages advance in declared order, the receipt lists the
stages that ran in that order -- the last attempt's outcome, plus an
``attempts`` count when a stage ran more than once -- and the stages not
reached are left for the projection to name. A stop before a dispatch records
the pending stage as ``skipped`` with the stop reason; a terminal episode
after a dispatch marks that stage ``terminated``.

This module never touches a backend. It is the same state machine whether the
other side is the orchestrator or the contract-test fake in :mod:`.contract`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ora.agent.composition.schema_validation import SchemaViolation, validate_schema_instance
from ora.agent.tools import skill_receipt
from ora.agent.tools.tool_errors import ToolError, schema_refusal

from . import sdk
from .library import ComposedTool
from .nesting import NestedStagesMixin, nested_summary as _nested_summary
from .starlark_runtime import Completed, HostCall, HostRequest, StarlarkProgramError, Suspended
from .tunables import TunableError, TunableTable

#: Why a sequence ended early when the executor, not the body, ended it.
STOP_OPERATOR_INTERRUPT = "operator_interrupt"
STOP_ENVIRONMENT_TERMINAL = "environment_terminal"
STOP_DRIVER_FAULT = "stage_driver_fault"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_COMPLETED = "sequence_completed"
_BUDGET_PREFIX = "budget_exhausted:"
#: The cost sentence closed-skill receipts carry, word for word.
COST_SEMANTICS = (
    "the Harness robot-decision budget counts this whole skill as one "
    "decision; the simulator advanced once per dispatched stage action"
)


@dataclass(frozen=True)
class Port:
    """The host functions behind SDK queries; a missing one is unavailable here.

    ``nested(ref, arguments, *, call_id)`` plans the composed tool a ``tool``
    stage pins on the latest observation and returns ``(tool, commitment,
    port[, attach])`` for its execution (``attach`` receives the nested
    Execution), or raises ToolError when its ``plan()`` refuses.
    """

    queries: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    nested: Callable[..., Any] | None = None


@dataclass(frozen=True)
class Commitment:
    tool: str
    version: str
    arm_profile: str
    compiled_sha256: str
    source_sha256: str
    call_id: str
    observation_id: str
    observation_sha256: str
    embodiment: str | None
    rationale: str
    arm: str
    candidate_id: str | None
    inputs: Mapping[str, Any]
    params: Mapping[str, Any]
    data: Mapping[str, Any]
    evidence: Mapping[str, Any]
    plan_journal: tuple[HostCall, ...]

    @property
    def context(self) -> dict[str, Any]:
        return {
            "tool": self.tool, "version": self.version, "arm_profile": self.arm_profile,
            "arms": list(sdk.ARM_PROFILES[self.arm_profile]), "embodiment": self.embodiment,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "ora.composed-tool-commitment.v1",
            "tool": self.tool, "version": self.version, "arm_profile": self.arm_profile,
            "compiled_sha256": self.compiled_sha256, "source_sha256": self.source_sha256,
            "call_id": self.call_id, "observation_id": self.observation_id,
            "observation_sha256": self.observation_sha256, "embodiment": self.embodiment,
            "rationale": self.rationale, "arm": self.arm, "candidate_id": self.candidate_id,
            "inputs": dict(self.inputs), "params": dict(self.params), "data": dict(self.data),
            "evidence": dict(self.evidence),
            "plan_journal": [
                {"symbol": call.symbol, "args": call.args, "kwargs": call.kwargs, "result": call.result}
                for call in self.plan_journal
            ],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Commitment":
        if value.get("schema") != "ora.composed-tool-commitment.v1":
            raise ValueError("not a composed tool commitment")
        return cls(
            tool=value["tool"], version=value["version"], arm_profile=value["arm_profile"],
            compiled_sha256=value["compiled_sha256"], source_sha256=value["source_sha256"],
            call_id=value["call_id"], observation_id=value["observation_id"],
            observation_sha256=value["observation_sha256"], embodiment=value["embodiment"],
            rationale=value["rationale"], arm=value["arm"], candidate_id=value["candidate_id"],
            inputs=dict(value["inputs"]), params=dict(value["params"]), data=dict(value["data"]),
            evidence=dict(value["evidence"]),
            plan_journal=tuple(
                HostCall(item["symbol"], list(item["args"]), dict(item["kwargs"]), item["result"])
                for item in value["plan_journal"]
            ),
        )


@dataclass(frozen=True)
class ActionRequest:
    index: int
    action_id: str
    stage: str
    stage_index: int
    attempt: int
    request: Mapping[str, Any]  # normalized by sdk.validate_action_request
    host_request: HostRequest
    journal: tuple[HostCall, ...]


@dataclass(frozen=True)
class Dispatched:
    index: int
    action_id: str
    stage: str
    request: Mapping[str, Any]
    outcome: Mapping[str, Any]


@dataclass(frozen=True)
class Finished:
    status: str  # completed | aborted | terminated
    stop_reason: str
    stages: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]
    dispatched: tuple[Dispatched, ...]
    fault: str | None = None


def _internal(message: str) -> ToolError:
    return ToolError(message, code="internal_tool_error", retryable=False)


def _count_queries(journal: Sequence[HostCall]) -> int:
    return sum(1 for call in journal if sdk.HOST.get(call.symbol) == "query")


def bound_queries(
    port: Port, *, entry: str, arms: Sequence[str], used: int, budget: int,
    check_stages: frozenset[str] = frozenset(),
) -> dict[str, Callable[..., Any]]:
    """Every SDK query bound for one interpretation of ``plan`` or ``execute``.

    Arguments are bound against the SDK signature first, an entry may only
    reach its own queries, and every live call spends the shared query budget.
    """

    live = {"count": 0}

    def bind(symbol: str) -> Callable[..., Any]:
        signature = sdk.QUERY_SIGNATURES[symbol]

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sdk.bind_call(symbol, signature, args, kwargs)
            except sdk.SdkCallError as exc:
                raise ValueError(str(exc)) from None
            wrong = sdk.EXECUTE_QUERIES if entry == "plan" else sdk.PLAN_QUERIES
            if symbol in wrong:
                raise ValueError(f"{symbol} is not available in {entry}()")
            if "arm" in bound and bound["arm"] not in arms:
                raise ValueError(f"arm {bound['arm']!r} is not one of {list(arms)}")
            if symbol in sdk.CHECK_QUERIES and bound.get("stage") not in check_stages:
                raise ValueError(f"{symbol} must name a declared check stage, got {bound.get('stage')!r}")
            if symbol == "sdk.capture" and bound["role"] not in sdk.CAPTURE_ROLES:
                raise ValueError(f"capture role must be one of {list(sdk.CAPTURE_ROLES)}")
            if symbol == "sdk.read":
                if bound["tool"] not in sdk.READ_TOOLS:
                    raise ValueError(f"sdk.read reaches only the read tools {list(sdk.READ_TOOLS)}")
                if bound.get("arguments") is not None and not isinstance(bound["arguments"], Mapping):
                    raise ValueError("sdk.read arguments must be a dict of the read tool's inputs")
                bound = {"tool": bound["tool"], "arguments": dict(bound.get("arguments") or {})}
            function = port.queries.get(symbol)
            if function is None:
                raise ValueError(f"{symbol} is unavailable on this host")
            if used + live["count"] >= budget:
                raise StarlarkProgramError("host_refusal", f"{_BUDGET_PREFIX}queries")
            live["count"] += 1
            return function(**bound)

        return call

    return {symbol: bind(symbol) for symbol in sdk.QUERY_SIGNATURES}


def _resolve_params(tool: ComposedTool, arguments: Mapping[str, Any], tunables: TunableTable,
                    embodiment: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for parameter in tool.declaration.parameters:
        entry = tunables[parameter.tunable]
        value = arguments.get(parameter.name) if parameter.expose else None
        if value is None:
            params[parameter.name] = entry.default_for(embodiment)
            continue
        try:
            params[parameter.name] = entry.admit(value)
        except TunableError as exc:
            raise ToolError(str(exc), code="invalid_arguments", field=parameter.name) from None
    return params


def submit(
    tool: ComposedTool,
    *,
    arm_profile: str,
    arguments: Mapping[str, Any],
    port: Port,
    tunables: TunableTable,
    call_id: str,
    observation_id: str,
    observation_sha256: str,
    embodiment: str | None = None,
) -> tuple[Commitment, dict[str, Any]]:
    """Run ``plan()``; return the commitment and the immediate public result, or raise ToolError."""

    compiled = tool.compiled(arm_profile)
    arguments = dict(arguments)
    try:
        validate_schema_instance(arguments, compiled["definition"]["inputSchema"], root_path="$input")
    except SchemaViolation as violation:
        raise schema_refusal(violation) from None
    rationale = skill_receipt.validate_rationale(arguments.get("rationale"))
    arms = sdk.ARM_PROFILES[arm_profile]
    arm = arguments.get("arm") or arms[0]
    inputs: dict[str, Any] = {}
    for item in tool.declaration.inputs:
        if item.kind == "arm_input":
            inputs[item.name] = arm
        elif item.kind == "rationale_input":
            inputs[item.name] = rationale
        else:
            inputs[item.name] = arguments.get(item.name)
    candidate_id = inputs.get("candidate_id") if tool.declaration.input("candidate_input") else None
    params = _resolve_params(tool, arguments, tunables, embodiment)
    context = {
        "tool": tool.name, "version": tool.version, "arm_profile": arm_profile,
        "arms": list(arms), "embodiment": embodiment,
    }
    queries = bound_queries(port, entry="plan", arms=arms, used=0, budget=tool.declaration.budgets.queries)
    try:
        outcome = tool.program.run("plan", args=[context, inputs, params], queries=queries, allow_actions=False)
    except StarlarkProgramError as fault:
        if fault.code == "host_refusal" and fault.detail.startswith(_BUDGET_PREFIX):
            raise _internal(f"{tool.name} plan spent its query budget") from None
        raise _internal(f"{tool.name} plan failed: {fault.code}: {fault.detail}") from None
    assert isinstance(outcome, Completed)
    value = outcome.value
    if not isinstance(value, Mapping) or value.get("kind") not in ("commit", "refuse"):
        raise _internal(f"{tool.name} plan must return sdk.commit(...) or sdk.refuse(...)")
    if value["kind"] == "refuse":
        declared = {item.code: item for item in tool.declaration.refusals}
        if value["code"] not in declared:
            raise _internal(f"{tool.name} plan refused with undeclared code {value['code']!r}")
        raise ToolError(
            value["message"], code=value["code"], field=value["field"],
            retryable=declared[value["code"]].retryable and value["retryable"],
            constraints=value.get("constraints") or None,
        )
    committed_arm = value["data"].get("arm", arm)
    if committed_arm not in arms:
        raise _internal(f"{tool.name} plan committed arm {committed_arm!r}, not one of {list(arms)}")
    arm = committed_arm
    evidence = dict(value.get("evidence") or {})
    undeclared = sorted(set(evidence) - set(tool.declaration.evidence))
    if undeclared:
        raise _internal(f"{tool.name} plan committed undeclared evidence {undeclared}")
    commitment = Commitment(
        tool=tool.name, version=tool.version, arm_profile=arm_profile,
        compiled_sha256=compiled["compiled_sha256"], source_sha256=tool.program.digest,
        call_id=call_id, observation_id=observation_id, observation_sha256=observation_sha256,
        embodiment=embodiment, rationale=rationale, arm=arm, candidate_id=candidate_id,
        inputs=inputs, params=params, data=dict(value["data"]), evidence=evidence,
        plan_journal=outcome.journal,
    )
    result: dict[str, Any] = {
        "success": True,
        "status": "submitted",
        "tool": tool.name,
        "version": tool.version,
        "arm": arm,
        "committed_sequence": [item.name for item in tool.declaration.stages],
        "cost": {"robot_decisions": 1, "max_primitive_actions": tool.declaration.budgets.primitive_actions},
        "claim_boundary": compiled["claim_boundary"],
        **({"candidate_id": candidate_id} if candidate_id is not None else {}),
        **evidence,
    }
    annotations = [item.name for item in tool.declaration.inputs if item.kind == "annotation_input"]
    if annotations:
        result["intent"] = {**{name: inputs[name] for name in annotations}, "role": "agent_annotation_only"}
    validate_schema_instance(result, compiled["result_schema"])
    return commitment, result


def _outcome(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - {"status", "reason", "measurements"}:
        raise ValueError("an outcome is {status, reason, measurements}")
    status = value.get("status")
    if status not in sdk.ACTION_OUTCOMES:
        raise ValueError(f"outcome status must be one of {list(sdk.ACTION_OUTCOMES)}")
    reason = value.get("reason") or status
    if not isinstance(reason, str):
        raise ValueError("outcome reason must be text")
    measurements = list(value.get("measurements") or [])
    for item in measurements:
        if not isinstance(item, Mapping) or "quantity" not in item or "value" not in item:
            raise ValueError("outcome measurements are {quantity, value} items")
    return {"status": status, "reason": reason, "measurements": [dict(item) for item in measurements]}


def _check_outcome(value: Any) -> dict[str, Any]:
    """A check query's result as a stage outcome: status, reason, measurements, verification."""

    if not isinstance(value, Mapping) or value.get("status") not in ("executed", "unavailable", "refused"):
        raise ValueError("a check result has status executed, unavailable or refused")
    out = _outcome({key: value[key] for key in ("status", "reason", "measurements") if key in value})
    if value.get("verification") is not None:
        if not isinstance(value["verification"], Mapping):
            raise ValueError("a check verification is an object")
        out["verification"] = dict(value["verification"])
    return out


@dataclass
class Execution(NestedStagesMixin):
    """Drive one committed body through its primitive actions."""

    tool: ComposedTool
    commitment: Commitment
    port: Port
    clock: Callable[[], float] = time.monotonic
    receipt_context: Callable[[], Mapping[str, Any]] | None = None
    depth: int = 0
    _journal: tuple[HostCall, ...] = ()
    _dispatched: list[Dispatched] = field(default_factory=list)
    _pending: ActionRequest | None = None
    _finished: Finished | None = None
    _started: float | None = None
    _executor_records: list[dict[str, Any]] = field(default_factory=list)
    _finish_args: tuple[str, str, str | None, tuple[Mapping[str, Any], ...]] | None = None
    _child: Any = None  # a nesting._Child while a tool stage runs

    @property
    def finished(self) -> Finished | None:
        return self._finished

    @property
    def dispatched(self) -> tuple[Dispatched, ...]:
        return tuple(self._dispatched)

    def _stage_index(self, stage: str) -> int:
        return [item.name for item in self.tool.declaration.stages].index(stage)

    def _last_stage_index(self) -> int:
        touched = [self._stage_index(stage) for stage in self._touched_stages()]
        return max(touched, default=-1)

    def _touched_stages(self) -> list[str]:
        stages: list[str] = []
        for call in self._journal:
            stage = call.kwargs.get("stage") if (
                sdk.HOST.get(call.symbol) == "action" or call.symbol in sdk.CHECK_QUERIES
            ) else None
            if stage is not None and stage not in stages:
                stages.append(stage)
        return stages

    def next(self) -> ActionRequest | Finished:
        """The next action to dispatch, or the finished sequence."""

        if self._finished is not None:
            return self._finished
        if self._pending is not None:
            raise RuntimeError("resolve the pending action before asking for the next one")
        if self._started is None:
            self._started = self.clock()
        elif self.clock() - self._started > self.tool.declaration.budgets.wall_s:
            return self._end_child_and_finish(STOP_BUDGET_EXHAUSTED)
        if self._child is not None:
            return self._next_child()
        declaration = self.tool.declaration
        arms = sdk.ARM_PROFILES[self.commitment.arm_profile]
        used = _count_queries(self.commitment.plan_journal) + _count_queries(self._journal)
        queries = bound_queries(
            self.port, entry="execute", arms=arms, used=used, budget=declaration.budgets.queries,
            check_stages=frozenset(item.name for item in declaration.stages if item.read_only),
        )
        try:
            result = self.tool.program.run(
                "execute", args=[self.commitment.context, dict(self.commitment.data)], queries=queries,
                allow_actions=True, journal=self._journal,
            )
        except StarlarkProgramError as fault:
            if fault.code == "host_refusal" and fault.detail.startswith(_BUDGET_PREFIX):
                return self._finish("aborted", STOP_BUDGET_EXHAUSTED)
            return self._finish("aborted", STOP_DRIVER_FAULT, fault=f"{fault.code}: {fault.detail}")
        self._journal = result.journal
        fault = self._order_fault()
        if fault:
            return self._finish("aborted", STOP_DRIVER_FAULT, fault=fault)
        if isinstance(result, Suspended):
            try:
                request = sdk.validate_action_request(
                    result.request.symbol, result.request.args, result.request.kwargs,
                    arms=arms, stages=declaration.stage_primitives,
                )
            except sdk.SdkCallError as exc:
                return self._finish("aborted", STOP_DRIVER_FAULT, fault=str(exc))
            stage_index = self._stage_index(request["stage"])
            if stage_index < self._last_stage_index():
                return self._finish("aborted", STOP_DRIVER_FAULT,
                                    fault=f"stage {request['stage']!r} after a later stage ran")
            if request["primitive"] == "tool":
                return self._start_child(request, stage_index, result)
            if len(self._dispatched) >= declaration.budgets.primitive_actions:
                return self._finish("aborted", STOP_BUDGET_EXHAUSTED)
            self._pending = self._request(request, stage_index, result.request, result.journal)
            return self._pending
        return self._complete(result.value)

    def _request(self, request: Mapping[str, Any], stage_index: int, host_request: HostRequest | None,
                 journal: tuple[HostCall, ...]) -> ActionRequest:
        stage = request["stage"]
        attempt = 1 + sum(1 for item in self._dispatched if item.stage == stage)
        action_id = f"{self.commitment.call_id}:{stage_index:02d}-{stage}"
        if attempt > 1:
            action_id += f".{attempt}"
        return ActionRequest(
            index=len(self._dispatched), action_id=action_id, stage=stage, stage_index=stage_index,
            attempt=attempt, request=request, host_request=host_request, journal=journal,  # type: ignore[arg-type]
        )

    def _order_fault(self) -> str | None:
        for call in self._journal:
            if call.symbol in sdk.CHECK_QUERIES:
                stage = call.kwargs.get("stage")
                if self.tool.declaration.stage_primitives.get(stage) != "check":
                    return f"{call.symbol} must name a declared check stage, got {stage!r}"
        indexes = [self._stage_index(stage) for stage in self._touched_stages()]
        if indexes != sorted(indexes):
            return "stages must advance in declared order"
        return None

    def resolve(self, outcome: Mapping[str, Any]) -> None:
        """Hand the classified outcome of the pending action back to the body."""

        if self._pending is None:
            raise RuntimeError("no action is pending")
        try:
            normalized = _outcome(outcome)
        except ValueError as exc:
            raise ValueError(f"{self._pending.action_id}: {exc}") from None
        pending, self._pending = self._pending, None
        self._dispatched.append(Dispatched(pending.index, pending.action_id, pending.stage, pending.request, normalized))
        if self._child is not None:
            self._child.execution.resolve(normalized)
            return
        self._journal = pending.journal + (pending.host_request.resolve(normalized),)

    def stop(self, reason: str, *, dispatched: bool = False) -> Finished:
        """End the sequence from outside: operator interrupt or terminal environment.

        With an action pending and not dispatched, that stage is recorded as
        skipped with the reason. ``dispatched=True`` means the pending action
        ran and the episode ended during it: the stage is reported terminated
        with no arrival judgement, as the closed-skill executor does. A terminal
        environment with nothing pending marks the last dispatched stage.
        """

        if self._finished is not None:
            return self._finished
        if reason not in (STOP_OPERATOR_INTERRUPT, STOP_ENVIRONMENT_TERMINAL):
            raise ValueError(f"stop reason must be {STOP_OPERATOR_INTERRUPT} or {STOP_ENVIRONMENT_TERMINAL}")
        if self._child is not None:
            # The nested tool's pending action is this sequence's pending action; record it below.
            child, self._child = self._child, None
            child.execution._pending = None
            child.execution.stop(STOP_OPERATOR_INTERRUPT)
            if self._pending is None and reason == STOP_ENVIRONMENT_TERMINAL:
                # The episode ended after a nested dispatch: the tool stage ends there.
                last = [item.action_id for item in self._dispatched[child.dispatched_before:]]
                self._executor_records.append({
                    "stage": child.stage, "status": "terminated", "reason": reason,
                    **({"joined_action_id": last[-1]} if last else {}),
                })
                return self._finish("terminated", reason)
        if self._pending is not None and dispatched:
            if reason != STOP_ENVIRONMENT_TERMINAL:
                raise ValueError("only a terminal environment ends a sequence during a dispatch")
            self._executor_records.append({
                "stage": self._pending.stage, "status": "terminated", "reason": reason,
                "action_id": self._pending.action_id,
            })
            self._pending = None
        elif self._pending is not None:
            self._executor_records.append({"stage": self._pending.stage, "status": "skipped", "reason": reason})
            self._pending = None
        elif reason == STOP_ENVIRONMENT_TERMINAL and self._dispatched:
            self._executor_records.append({"terminated": self._dispatched[-1].action_id})
        status = "terminated" if reason == STOP_ENVIRONMENT_TERMINAL else "aborted"
        return self._finish(status, reason)

    def fault_pending(self, reason: str = "trusted_execution_fault") -> Finished:
        """The trusted executor raised while dispatching the pending action.

        The stage is recorded unavailable with the fault as its reason and the
        sequence ends there; the caller re-raises after reporting, as the
        closed-skill executor does.
        """

        if self._pending is None:
            raise RuntimeError("no action is pending")
        self._child = None
        pending, self._pending = self._pending, None
        self._executor_records.append({
            "stage": pending.stage, "status": "unavailable", "reason": reason, "action_id": pending.action_id,
        })
        return self._finish("aborted", reason)

    def _complete(self, value: Any) -> Finished:
        if not isinstance(value, Mapping) or value.get("kind") not in ("complete", "abort"):
            return self._finish("aborted", STOP_DRIVER_FAULT,
                                fault="execute must return sdk.complete(...) or sdk.abort(...)")
        declared = self.tool.declaration.stage_primitives
        measurements = list(value.get("measurements") or [])
        per_stage: dict[str, int] = {}
        for item in measurements:
            if item.get("stage") not in declared:
                return self._finish("aborted", STOP_DRIVER_FAULT,
                                    fault=f"measurement names undeclared stage {item.get('stage')!r}")
            per_stage[item["stage"]] = per_stage.get(item["stage"], 0) + 1
        if any(count > self.tool.declaration.budgets.measurements_per_stage for count in per_stage.values()):
            return self._finish("aborted", STOP_DRIVER_FAULT, fault="body measurements exceed measurements_per_stage")
        if value["kind"] == "complete":
            return self._finish("completed", STOP_COMPLETED, body_measurements=measurements)
        return self._finish("aborted", value["code"], body_measurements=measurements)

    def _attempts(self) -> list[tuple[str, dict[str, Any], str | None]]:
        """(stage, outcome, action_id) for every dispatched action and check, in journal order."""

        ids = iter(item.action_id for item in self._dispatched)
        attempts = []
        for call in self._journal:
            if call.symbol == "sdk.call":
                count = int(_nested_summary(call.result).get("dispatched_actions") or 0)
                taken = [next(ids) for _ in range(count)]
                attempts.append((call.kwargs["stage"], _outcome(call.result), taken[-1] if taken else None))
            elif sdk.HOST.get(call.symbol) == "action":
                attempts.append((call.kwargs["stage"], _outcome(call.result), next(ids)))
            elif call.symbol in sdk.CHECK_QUERIES:
                attempts.append((call.kwargs["stage"], _check_outcome(call.result), None))
        return attempts

    def _stage_records(self, body_measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """One record per touched stage, in declared order; executor outcomes count as the last attempt."""

        by_stage: dict[str, list[tuple[dict[str, Any], str | None]]] = {}
        for stage, outcome, action_id in self._attempts():
            by_stage.setdefault(stage, []).append((outcome, action_id))
        terminated = {item["terminated"] for item in self._executor_records if "terminated" in item}
        for item in self._executor_records:
            if "stage" in item:
                # joined_action_id: the last nested dispatch of a tool stage, already counted as dispatched.
                outcome = {"status": item["status"], "reason": item["reason"], "measurements": []}
                by_stage.setdefault(item["stage"], []).append(
                    (outcome, item.get("action_id") or item.get("joined_action_id")))
        extra: dict[str, list[dict[str, Any]]] = {}
        for entry in body_measurements:
            extra.setdefault(entry["stage"], []).append(
                {key: entry[key] for key in ("quantity", "value", "unit") if key in entry}
            )
        records = []
        for stage in sorted(set(by_stage) | set(extra), key=self._stage_index):
            attempts = by_stage.get(stage, [])
            if not attempts:
                records.append(skill_receipt.stage(stage, "executed", "measured", measurements=extra[stage]))
                continue
            outcome, action_id = attempts[-1]
            measured = list(outcome["measurements"]) + extra.get(stage, [])
            tries = sum(1 for _outcome, attempt_id in attempts if attempt_id is not None)
            if tries > 1:
                measured.append({"quantity": "attempts", "value": tries})
            status, reason = outcome["status"], outcome["reason"]
            if action_id is not None and action_id in terminated:
                status, reason = "terminated", STOP_ENVIRONMENT_TERMINAL
            record = skill_receipt.stage(stage, status, reason, action_id=action_id, measurements=measured)
            if "verification" in outcome:
                record["verification"] = outcome["verification"]
            records.append(record)
        return records

    def _observation_evidence(self) -> list[dict[str, Any]]:
        evidence = []
        for call in self._journal:
            if call.symbol == "sdk.capture" and isinstance(call.result, Mapping):
                evidence.append({
                    "arm": self.commitment.arm,
                    **{key: call.result.get(key) for key in (
                        "role", "observation_root", "observation_id", "observation_sha256",
                    )},
                })
        return evidence

    def finalize(self) -> Finished:
        """Rebuild the finished receipt with the receipt context as it is now.

        The executor captures the final observation after the body ended; the
        receipt must name that observation, as a closed-skill receipt does.
        """

        if self._finish_args is None:
            raise RuntimeError("the sequence has not finished")
        status, stop_reason, fault, body = self._finish_args
        return self._finish(status, stop_reason, fault=fault, body_measurements=body)

    def _finish(self, status: str, stop_reason: str, *, fault: str | None = None,
                body_measurements: Sequence[Mapping[str, Any]] = ()) -> Finished:
        self._finish_args = (status, stop_reason, fault, tuple(body_measurements))
        stages = self._stage_records(body_measurements)
        verification = None
        for item in stages:
            if isinstance(item.get("verification"), Mapping):
                verification = item["verification"]
        context = dict(self.receipt_context() if self.receipt_context else {})
        compiled = self.tool.compiled(self.commitment.arm_profile)
        receipt = skill_receipt.assemble(
            skill=self.tool.name, status=status, rationale=self.commitment.rationale,
            candidate_id=self.commitment.candidate_id,
            observation_id=self.commitment.observation_id, observation_sha256=self.commitment.observation_sha256,
            stages=[
                skill_receipt.stage(item["stage"], item["status"], item["reason"], action_id=item.get("action_id"),
                                    measurements=item.get("measurements", ()))
                for item in stages
            ],
            verification=verification,
            held_object_offset=context.get("held_object_offset"),
            claim_boundary=compiled["claim_boundary"],
        )
        receipt["stop_reason"] = stop_reason
        if "fresh_observation_id" in context:
            receipt["fresh_observation_id"] = str(context["fresh_observation_id"] or "")
        receipt["cost"] = {
            "robot_decisions": 1,
            "dispatched_stage_actions": len(self._dispatched) + sum(
                1 for item in self._executor_records if item.get("action_id")
            ),
            "semantics": COST_SEMANTICS,
        }
        receipt["ledger_released"] = True
        for name in self.tool.declaration.evidence:
            if name in self.commitment.evidence:
                receipt[name] = self.commitment.evidence[name]
        observation_evidence = self._observation_evidence()
        if observation_evidence:
            receipt["observation_evidence"] = observation_evidence
        receipt["composed_tool"] = {
            "tool": self.tool.name, "version": self.tool.version,
            "arm_profile": self.commitment.arm_profile, "compiled_sha256": self.commitment.compiled_sha256,
        }
        _remeasure(receipt)
        self._finished = Finished(status, stop_reason, tuple(stages), receipt, tuple(self._dispatched), fault)
        return self._finished


def _remeasure(receipt: dict[str, Any]) -> None:
    """Record the body size after fields were added, as build_skill_receipt does."""

    for _ in range(4):
        measured = skill_receipt.body_bytes(receipt)
        if receipt["delivery"]["body_bytes"] == measured:
            return
        receipt["delivery"]["body_bytes"] = max(measured, receipt["delivery"]["body_bytes"])
