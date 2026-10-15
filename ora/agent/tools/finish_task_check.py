"""``finish_episode`` against the round's task check (``faithful_check_task``).

On a declaring surface the Supervisor writes ``policy-task-check.json`` at
every published observation, from the trusted environment boundary:
``terminal`` and ``task_success`` (official success at a terminal boundary;
``orchestrator/episode_observation.py``). Two finish rules read it.

``refuse_latched``
    Once check_task has confirmed success, only ``outcome: success`` is
    admitted. The dispatcher runs it before every other finish check
    (``finish_outcomes.admit``, the input schema, the handler): no other refusal
    can help then, and an abort-evidence or reason refusal would
    send the policy after an outcome that is refused anyway. Retryable --
    ``success`` is admitted at once -- with ``field: outcome``, and the
    unchanged request is refused again.
``refuse_unconfirmed``
    ``success`` needs check_task true. Retryable while the round is not
    terminal; in the terminal turn after an environment boundary without
    official success check_task can no longer become true in this round, so
    the refusal is not retryable there.

Neither code is an infrastructure class: both are the policy's request
against the round's task state.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .robot import ToolError

CAPABILITY = "faithful_check_task"
RECORD = "policy-task-check.json"
LATCHED_CODE = "task_success_latched"
UNCONFIRMED_CODE = "task_success_not_confirmed"


def task_check(host: Any) -> Mapping[str, Any]:
    """The round's task-check record; empty before the Supervisor wrote one."""

    path = host.control_root / RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def refuse_latched(host: Any, arguments: Any) -> None:
    """Refuse any outcome but success once check_task has confirmed success."""

    if not host.surface.supports(CAPABILITY):
        return
    outcome = arguments.get("outcome") if isinstance(arguments, Mapping) else None
    if outcome == "success" or task_check(host).get("task_success") is not True:
        return
    raise ToolError(
        "task success is already confirmed by check_task in this round; finish_episode admits only "
        "outcome success now",
        code=LATCHED_CODE, retryable=True, retry_requires_changed_arguments=True,
        field="outcome", received=outcome,
        constraints={"allowed": ["success"], "task_success": True},
    )


def refuse_unconfirmed(host: Any, outcome: str) -> None:
    """Refuse success that check_task has not confirmed at the current boundary."""

    if outcome != "success" or not host.surface.supports(CAPABILITY):
        return
    checked = task_check(host)
    if checked.get("task_success") is True:
        return
    terminal = checked.get("terminal") is True
    message = (
        "success requires check_task=true, and this round's environment has ended without official "
        "success: check_task cannot become true in this round"
        if terminal else
        "success requires check_task=true at the current state boundary, and check_task is not true"
    )
    raise ToolError(
        message, code=UNCONFIRMED_CODE, retryable=not terminal, field="outcome", received=outcome,
        constraints={"requires": "check_task=true", "task_success": checked.get("task_success"),
                     "terminal": terminal},
    )


__all__ = ["CAPABILITY", "LATCHED_CODE", "RECORD", "UNCONFIRMED_CODE", "refuse_latched",
           "refuse_unconfirmed", "task_check"]
