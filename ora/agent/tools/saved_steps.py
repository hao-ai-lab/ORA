"""Which saved observation a ``step`` argument selects, and how it is refused.

``observe``, ``vision_segment`` and ``geometry_query`` accept a saved
policy-state ``step``. Three defects sat in how they read it (EXP11B census
A3, B7, C2):

* a ``step`` equal to the current step took the historical path, so it was
  labelled historical and its overlays and ``history_point_ids`` were refused,
  although it is the current observation (104 EXP11A calls);
* a step outside the saved range was refused with ``stale_observation`` or
  ``artifact_not_found``, codes that do not say the argument was wrong, and
  ``geometry_query`` stated no range at all;
* the round timeline was read through the placement helper, so a
  ``vision_segment`` refusal spoke of "placement".

``route_current`` sends the current step to the current path,
``round_step_refusal`` and ``saved_step_refusal`` refuse an out-of-range step as
an argument naming its bounds, and ``round_timeline`` is the neutral reader the
placement helper now forwards to.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.evidence.policy_state import verify_policy_state

from .robot import ToolError


def current_step(host: Any) -> int | None:
    """The policy-state step of the current observation, or ``None``.

    ``None`` whenever the current observation cannot be read (closed, absent)
    or names no step, so a saved-step request keeps its historical path.
    """

    try:
        control, _root, _observation = host._current()
    except ToolError:
        return None
    reference = control.get("policy_state") if isinstance(control, Mapping) else None
    step = reference.get("step") if isinstance(reference, Mapping) else None
    return step if type(step) is int else None


def route_current(host: Any, step: Any) -> Any:
    """``None`` (the current path) when ``step`` names the current observation."""

    return None if type(step) is int and step == current_step(host) else step


def round_timeline(
    host: Any, control: Mapping[str, Any], *, subject: str = "saved-step selection",
) -> tuple[list[dict[str, Any]], int]:
    """The run's verified public states and the index of this round's reset."""

    if host.state_root is None:
        raise ToolError(
            f"{subject} requires canonical public-state evidence, and this run has none",
            code="artifact_integrity_fault", retryable=False,
        )
    verified = verify_policy_state(host.state_root.parent, verify_artifacts=False)
    if verified["status"] != "valid":
        raise ToolError(
            f"{subject}: the public-state chain is invalid",
            code="artifact_hash_mismatch", retryable=False,
        )
    states = verified["steps"]
    matches = [index for index, state in enumerate(states)
               if state["observation"]["id"] == control["observation_id"]]
    if (len(matches) != 1 or matches[0] != len(states) - 1
            or states[-1]["observation"]["manifest"]["sha256"] != control["observation_sha256"]):
        raise ToolError(
            f"{subject}: the latest public state is not the current observation",
            code="stale_observation",
        )
    resets = [index for index, state in enumerate(states) if state["cause"] == "reset"]
    if not resets:
        raise ToolError(
            f"{subject}: the round has no reset boundary",
            code="artifact_hash_mismatch", retryable=False,
        )
    return states, resets[-1]


def round_step_refusal(
    step: Any, states: list[Mapping[str, Any]], round_start: int,
) -> ToolError | None:
    """The refusal of a step outside this round's saved range, or ``None``."""

    minimum, maximum = states[round_start]["step"], states[-1]["step"]
    if type(step) is int and minimum <= step <= maximum:
        return None
    return ToolError(
        f"step {step!r} is not a saved observation step of the current round "
        f"[{minimum}, {maximum}]",
        code="invalid_arguments", field="step", received=step,
        constraints={"minimum": minimum, "maximum": maximum, "scope": "current_round"},
    )


def saved_step_refusal(host: Any, step: Any, error: Exception) -> ToolError:
    """A saved policy-state step that did not load: the argument, or the store.

    With a valid chain the step is outside it, an argument refusal with the
    bounds; otherwise the Harness's own evidence failed.
    """

    try:
        verified = verify_policy_state(host.state_root.parent, verify_artifacts=False)
    except Exception:  # noqa: BLE001 - an unreadable chain is the store's fault
        verified = {"status": "invalid"}
    steps = verified.get("steps") if verified.get("status") == "valid" else None
    if steps:
        return ToolError(
            f"step {step!r} is not a saved policy-state step [0, {len(steps) - 1}]",
            code="invalid_arguments", field="step", received=step,
            constraints={"minimum": 0, "maximum": len(steps) - 1, "scope": "run"},
        )
    return ToolError(
        f"saved policy-state evidence is unavailable: {error}",
        code="artifact_integrity_fault", retryable=False, field="step", received=step,
    )


__all__ = [
    "current_step", "round_step_refusal", "round_timeline", "route_current",
    "saved_step_refusal",
]
