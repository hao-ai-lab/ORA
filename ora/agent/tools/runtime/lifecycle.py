"""``compute`` and the episode lifecycle tools.

``compute`` (validated in ``compute_arguments``), ``report_issue``,
``check_task`` and ``finish_episode`` with the Runtime's own
remaining-budget record.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.

``load_policy_state`` comes from ``common``,
which reads it from ``core`` at call time.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from ora.agent.composition.schema_validation import ToolSurfaceError, validate_schema_instance
from ora.agent.evidence.policy_state import PolicyStateError

from ora.agent.tools import finish_outcomes, finish_task_check
from ora.agent.tools.runtime.common import load_policy_state
from ora.agent.tools.robot import ToolError, _append
from ora.agent.tools.runtime_artifacts import _atomic_json
from ora.agent.tools.tool_errors import blank_text_refusal, schema_refusal

#: The rules the handler has always enforced -- one ``message`` string of at
#: least one character and nothing else -- written as the schema most surfaces
#: declare. Checked on every surface, so a refusal
#: names its field and rule whether or not the surface validated its declared
#: schema before the handler.
REPORT_ISSUE_INPUT = {
    "type": "object", "additionalProperties": False, "required": ["message"],
    "properties": {"message": {"type": "string", "minLength": 1}},
}


class LifecycleRuntimeMixin:
    """``compute`` and the episode lifecycle tools."""

    def _restricted_compute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Protocol edge only; the work lives in tools/compute_arguments."""

        from ora.agent.tools import compute_arguments

        return compute_arguments.invoke(self, arguments)

    def _report_issue(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        # The declared schema's own rules first (every surface, not only the
        # advisory ones that validate before the handler), then the one rule
        # minLength cannot state: at least one non-whitespace character.
        try:
            validate_schema_instance(arguments, REPORT_ISSUE_INPUT, root_path="$input")
        except ToolSurfaceError as error:
            raise schema_refusal(error) from error
        message = arguments["message"]
        if not message.strip():
            raise blank_text_refusal(
                "message", message,
                message="report_issue requires a non-empty message: the message contains only whitespace",
            )
        _append(self.control_root / "openeta-reported-issues.jsonl", {"message": message.strip(), "time": time.time()})
        return {"success": True}

    def _check_task(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if arguments:
            raise ToolError("check_task accepts no arguments")
        self._current()
        path = self.control_root / "policy-task-check.json"
        if not path.is_file():
            return {"success": False, "reason": "task_checker_unavailable", "retryable": True}
        value = json.loads(path.read_text(encoding="utf-8"))
        return {"success": True, "task_success": bool(value.get("task_success", False))}

    def _runtime_remaining_budget(self) -> dict[str, Any]:
        """The Runtime's own remaining-budget count at this finish boundary.

        Read from the persisted policy state, never from anything the policy
        said. ``_finish`` refuses while an action proposal is pending, so
        the latest published step is the current decision boundary and its
        ``robot_decisions`` is exact. ``model_calls`` is the count as of that
        step and therefore excludes the turn now finishing; the record says so
        rather than implying a freshness it does not have.
        """

        unavailable = {
            "available": False,
            "source": "policy_state_unavailable",
            "robot_decisions": None,
            "model_calls": None,
        }
        if self.state_root is None:
            return unavailable
        try:
            state = load_policy_state(self.state_root.parent, None)
        except (PolicyStateError, OSError, ValueError):
            return unavailable
        budget = state.get("remaining_budget")
        if not isinstance(budget, Mapping):
            return unavailable
        record = {
            "available": True,
            "source": "runtime_policy_state_snapshot",
            "policy_state_step": state.get("step"),
            "policy_state_sha256": state.get("state_sha256"),
            "model_calls_freshness": "as_of_this_step_excludes_the_finishing_turn",
        }
        for key in (
            "robot_decisions", "model_calls", "wall_time_s",
            "session_round", "session_rounds",
        ):
            if key in budget:
                record[key] = budget[key]
        return record

    def _finish(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if (self.control_root / "submitted-action.json").is_file():
            raise ToolError(
                "an action proposal is pending; wait for the fresh observation before finishing",
                code="action_pending_fresh_observation", retryable=False,
            )
        # What the tool ACCEPTS follows this surface's own advertised input
        # schema.
        advertised = set()
        definition = next(
            (item for item in self.surface.definitions
             if item.get("name") == "finish_episode"),
            None,
        )
        if isinstance(definition, Mapping):
            schema = definition.get("inputSchema")
            if isinstance(schema, Mapping):
                properties = schema.get("properties")
                if isinstance(properties, Mapping):
                    advertised = {
                        name for name in properties if isinstance(name, str)
                    }
        accepted = {"outcome", "reason", "attempts"} | advertised
        # A v2 finish contract was admitted in full by finish_outcomes.admit;
        # a released surface keeps its three outcomes.
        outcomes = finish_outcomes.accepted_outcomes(self.surface) or ("success", "failure", "abort")
        if set(arguments) - accepted or arguments.get("outcome") not in outcomes:
            raise ToolError("finish_episode requires outcome success, failure or abort")
        outcome = str(arguments["outcome"])
        if outcome == "failure":
            attempts = arguments.get("attempts")
            if not str(arguments.get("reason", "")).strip():
                raise ToolError("failure requires a reason")
            if (
                not isinstance(attempts, list) or not attempts
                or any(not isinstance(item, str) or not item.strip() for item in attempts)
            ):
                raise ToolError("failure requires non-empty exhausted attempts")
        runtime_budget = self._runtime_remaining_budget()
        remaining_decisions = runtime_budget.get("robot_decisions")
        early_exit = (
            outcome != "success"
            and isinstance(remaining_decisions, int)
            and not isinstance(remaining_decisions, bool)
            and remaining_decisions > 0
        )
        # A latched success was refused by the dispatcher before any other
        # finish check (finish_task_check.refuse_latched).
        finish_task_check.refuse_unconfirmed(self, outcome)
        record = {
            "schema": "ora.openeta-policy-finish.v1",
            **dict(arguments),
            # Runtime truth, stamped unconditionally on every surface: a
            # reviewer can decide from the record alone whether this episode
            # ended because it ran out of allowance or because the policy chose
            # to stop, without believing anything the policy asserted about it.
            "runtime_remaining_budget": runtime_budget,
            "early_exit": early_exit,
            "early_exit_argument_required": False,
            **finish_outcomes.record_fields(self, arguments),
            "time": time.time(),
        }
        _atomic_json(self.control_root / "policy-finish.json", record)
        return {
            "success": True, "outcome": arguments["outcome"],
            "_runtime": {"directive": "episode_finished"},
        }
