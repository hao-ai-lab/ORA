"""Tool stages: a composed tool running another composed tool as one of its stages.

The parent body calls ``sdk.call(stage=, arguments=)`` in a declared ``tool``
stage. :class:`NestedStagesMixin` (part of
:class:`~ora.agent.tools.composed.executor.Execution`) plans the
pinned tool through ``port.nested`` on a fresh observation, drives the nested
Execution, and dispatches each of its actions as the next attempt of the parent
stage (``<call>:<index>-<stage>.<n>``), so RoboDojo admits them without a new
rule and the parent's action budget counts them. When the nested tool ends, the
body's ``sdk.call`` resolves to one outcome with a ``nested_tool`` summary and
the parent replays on. Nesting stops at ``MAX_NESTING_DEPTH``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ora.agent.tools.tool_errors import ToolError

from .declaration import MAX_NESTING_DEPTH
from .starlark_runtime import HostCall, HostRequest, Suspended

if TYPE_CHECKING:
    from .executor import ActionRequest, Finished


def nested_summary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    for item in value.get("measurements") or ():
        if isinstance(item, Mapping) and item.get("quantity") == "nested_tool" and isinstance(item.get("value"), Mapping):
            return item["value"]
    return {}


@dataclass
class _Child:
    """A nested composed tool running as one ``tool`` stage of its parent."""

    execution: Any  # the nested Execution
    stage: str
    stage_index: int
    host_request: HostRequest
    journal: tuple[HostCall, ...]
    dispatched_before: int


class NestedStagesMixin:
    """The tool-stage half of ``Execution``; it holds methods only."""

    # ---- nested composed tools (tool stages) ------------------------------------------------------

    def _resolve_call(self, host_request: HostRequest, journal: tuple[HostCall, ...],
                      outcome: Mapping[str, Any]) -> ActionRequest | Finished:
        """Hand a tool stage's outcome to the body without a dispatch of its own and replay on."""

        from .executor import _outcome

        self._journal = journal + (host_request.resolve(_outcome(outcome)),)
        return self.next()

    def _start_child(self, request: Mapping[str, Any], stage_index: int, result: Suspended) -> ActionRequest | Finished:
        stage = request["stage"]
        ref = next(item.tool for item in self.tool.declaration.stages if item.name == stage)
        summary = {"tool": str(ref).split("/")[0], "version": str(ref).split("/")[-1], "status": "refused",
                   "stop_reason": "", "dispatched_actions": 0, "stages": {}}
        if self.depth >= MAX_NESTING_DEPTH:
            summary["stop_reason"] = "nesting_too_deep"
            return self._resolve_call(result.request, result.journal, {
                "status": "refused", "reason": "nesting_too_deep",
                "measurements": [{"quantity": "nested_tool", "value": summary}]})
        if self.port.nested is None:
            summary["stop_reason"] = "nested_tools_unavailable"
            return self._resolve_call(result.request, result.journal, {
                "status": "unavailable", "reason": "nested_tools_unavailable",
                "measurements": [{"quantity": "nested_tool", "value": summary}]})
        try:
            made = self.port.nested(
                ref, dict(request["arguments"]), call_id=f"{self.commitment.call_id}:{stage_index:02d}-{stage}~n",
            )
        except ToolError as error:
            summary["stop_reason"] = error.code
            return self._resolve_call(result.request, result.journal, {
                "status": "refused", "reason": error.code,
                "measurements": [{"quantity": "nested_tool", "value": summary}]})
        tool, commitment, port = made[:3]
        execution = type(self)(tool, commitment, port, clock=self.clock, depth=self.depth + 1)
        if len(made) > 3 and callable(made[3]):
            made[3](execution)
        self._child = _Child(execution, stage, stage_index, result.request, result.journal, len(self._dispatched))
        return self._next_child()

    def _next_child(self) -> ActionRequest | Finished:
        from .executor import STOP_BUDGET_EXHAUSTED, STOP_OPERATOR_INTERRUPT, Finished

        child = self._child
        assert child is not None
        step = child.execution.next()
        if isinstance(step, Finished):
            self._child = None
            dispatched = len(self._dispatched) - child.dispatched_before
            summary = {
                "tool": child.execution.tool.name, "version": child.execution.tool.version,
                "status": step.status, "stop_reason": step.stop_reason, "dispatched_actions": dispatched,
                "stages": {str(item["stage"]): str(item["status"]) for item in step.stages},
            }
            status = "executed" if step.status == "completed" else "not_reached"
            return self._resolve_call(child.host_request, child.journal, {
                "status": status, "reason": step.stop_reason,
                "measurements": [{"quantity": "nested_tool", "value": summary}]})
        if len(self._dispatched) >= self.tool.declaration.budgets.primitive_actions:
            child.execution.stop(STOP_OPERATOR_INTERRUPT)
            return self._end_child_and_finish(STOP_BUDGET_EXHAUSTED)
        request = {**step.request, "stage": child.stage, "nested_stage": step.stage}
        self._pending = self._request(request, child.stage_index, None, ())
        return self._pending

    def _end_child_and_finish(self, reason: str) -> Finished:
        """Stop a running nested tool from outside its body and end this sequence with ``reason``."""

        child, self._child = self._child, None
        if child is not None:
            last = [item.action_id for item in self._dispatched[child.dispatched_before:]]
            self._executor_records.append({
                "stage": child.stage, "status": "not_reached", "reason": reason,
                **({"joined_action_id": last[-1]} if last else {}),
            })
        return self._finish("aborted", reason)
