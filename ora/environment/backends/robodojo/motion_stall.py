"""Measured joint stall detection for RoboDojo arm motion, one sample per native action.

A released RoboDojo move ended only at measured arrival or when its 20 native
actions ran out, so an arm pressed into the table or a basket held its drive at
the SDK effort limit for the rest of the window and reported
``native_action_budget_exhausted`` (public ``controller_timeout``: "still moving,
call again"). This module feeds the shared, arm-agnostic ``JointBlockedMonitor``
(``environment/controllers/motion_progress.py``, which RoboTwin samples once per
physics step) once per native action -- 10 physics steps, 40 ms -- with the
measured arm joints. Every RoboDojo joint-target segment (global and local
moves, plan references, the moves inside ``grasp_execute`` and every recovery
segment) is driven by ``drive_until_settled`` and ends as soon as each arm has
arrived or stalled. A stall ends the command with ``controller_stalled``; a
window that runs out while an arm still moves keeps
``native_action_budget_exhausted``.

A stalled arm then holds the joints measured when it stalled. The SDK keeps
re-applying the last resolved command on every later native action
(``bridge._fixed_meta_control`` fills each dimension a command does not name
from ``prev_control``), so leaving the unreached IK target there would keep the
arm pushing -- through the next ``set_gripper`` or ``wait`` -- exactly as the
released moves did (review replay of g001-g006: 83 of 84 blocked moves followed
by a jaw command or wait still held the old target, 0.06-0.96 rad away, and
crept 0.010-0.018 rad within three or four native actions). The stalled arm's
command is replaced by its measured joints for the rest of the segment while
other arms continue; an arm that stalls on the segment's last native action
gets one journaled hold native action when the command budget allows, else the
retained SDK command is rewritten. ``session.pose_targets`` follows the held
pose. Each driven arm's receipt carries a compact ``motion_progress``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ora.environment.controllers.motion_progress import JointBlockedMonitor

from .common import pose_reached

#: The rule RoboDojo recovery has applied since EXP11A g003 (three native
#: actions, each moving no joint 0.005 rad), now the one rule of every arm
#: motion. Unobstructed ARX joints move about 0.2 rad per native action.
#: Replayed with this monitor (the EXP11A stall replay) over every recorded EXP11A g001-g006 move_pose/move_free_space
#: command (3,307 driven segments), it fires in 362 of the 403 segments that
#: ended unreached, 4,101 native actions before their window ran out, and in 9
#: of the 2,904 that arrived: an arm creeping at the endpoint tolerance or
#: snagged for a few actions now ends as a stall with its residual. On g005
#: alone: 107 of 119 unreached (1,234 native actions), 1 of 691.
#:
#: RoboTwin met the per-step joint rule's weakness first (integration 0033): a
#: slow Servo approach moving 0.244 mm in 17 physics steps was called blocked,
#: so its Servo adapter confirms a joint stall with cumulative TCP progress.
#: Here the D2 review replay measured that trade: a 1 mm TCP-progress
#: confirmation would spare 5 of the 9 arriving segments but stop 30 of 361 true
#: stalls later. This unit keeps the plain joint rule.
STALL_WINDOW_NATIVE_ACTIONS = 3
STALL_JOINT_STEP_RAD = 0.005
STALL_CODE = "controller_stalled"
BUDGET_CODE = "native_action_budget_exhausted"
PROGRESS_SCHEMA = "ora.robodojo-motion-progress.v1"
TERMINAL_CODE = "episode_terminal_during_execution"
HOLD_AT_STALL = "measured_joints_at_stall"
HOLD_SEGMENT = "stall_hold"
#: Arrival is judged by the executor's own predicate before the stall is read,
#: so every residual counts: an arm standing still short of arrival under an
#: unchanged joint target will not arrive. ``JointBlockedMonitor`` counts a
#: still sample only while the joint residual exceeds this floor.
_RESIDUAL_FLOOR_RAD = 0.0

Arrival = Callable[[Mapping[str, Any]], bool]


def stall_rule() -> dict[str, float | int]:
    """The rule as ``recovery_progress.stall_rule`` has published it since g003."""
    return {"window_native_actions": STALL_WINDOW_NATIVE_ACTIONS, "joint_step_rad": STALL_JOINT_STEP_RAD}


def _max_abs(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(first, second, strict=True))


def pose_arrival(target: Sequence[float]) -> Arrival:
    return lambda state: pose_reached(target, state["pose_xyz_xyzw"])


@dataclass
class ActionStallState:
    """Stall evidence of one sparse action across its driven segments."""

    progress: dict[str, dict[str, Any]] = field(default_factory=dict)  # per arm, latest segment (public)
    holds: dict[str, dict[str, Any]] = field(default_factory=dict)  # per arm held at its stall
    stalled: bool = False  # whether the latest driven segment ended on a stall


class NativeActionStallMonitor:
    """Per-arm stall state of one joint-target segment, fed once per native action."""

    def __init__(self, initial: Mapping[str, Sequence[float]], targets: Mapping[str, Sequence[float]], *,
                 budget: int) -> None:
        self.targets = {arm: [float(value) for value in joints] for arm, joints in targets.items()}
        self.previous = {arm: [float(value) for value in initial[arm]] for arm in self.targets}
        self.last_step: dict[str, float | None] = {arm: None for arm in self.targets}
        self.latched: dict[str, dict[str, Any]] = {}
        self.budget = int(budget)
        self.ticks = 0
        self.monitor = JointBlockedMonitor(
            self.previous, consecutive_physics_steps=STALL_WINDOW_NATIVE_ACTIONS,
            maximum_joint_step_rad=STALL_JOINT_STEP_RAD,
        )

    def sample(self, measured: Mapping[str, Sequence[float]]) -> None:
        """Feed the arm joints measured after one native action."""
        self.ticks += 1
        current = {arm: [float(value) for value in measured[arm]] for arm in self.targets}
        for arm, joints in current.items():
            self.last_step[arm] = _max_abs(joints, self.previous[arm])
        self.previous = current
        # The monitor's report describes RoboTwin waypoint stages; a native
        # segment has none, so only its per-arm consecutive counts are read.
        self.monitor.sample(current, self.targets, tolerance_rad=_RESIDUAL_FLOOR_RAD, stage_index=self.ticks,
                            planned_stages=max(self.budget, 1), waypoint_wait_steps=())

    def frozen(self, arm: str) -> bool:
        """The shared monitor's verdict on the latest samples."""
        return self.monitor.counts[arm] >= self.monitor.required

    def latch(self, arm: str, state: Mapping[str, Any]) -> list[float]:
        """Record the stall and return the measured joints the arm now holds."""
        joints = [float(value) for value in state["arm_joint_qpos"]]
        self.latched[arm] = {"joints": joints, "pose": [float(value) for value in state["pose_xyz_xyzw"]],
                             "tick": self.ticks, "still": int(self.monitor.counts[arm]),
                             "last_step": self.last_step[arm]}
        return list(joints)

    def stalled(self, arm: str) -> bool:
        return arm in self.latched

    def progress(self, arm: str, state: Mapping[str, Any], *, segment: str, arrived: bool,
                 terminal: bool) -> dict[str, Any]:
        """Public end-of-segment evidence; the rule and targets live in help and segments."""
        if arrived:
            return {"schema": PROGRESS_SCHEMA, "segment": segment, "outcome": "arrived"}
        held = self.latched.get(arm)
        joints = held["joints"] if held else [float(value) for value in state["arm_joint_qpos"]]
        outcome = "stalled" if held else "episode_terminal" if terminal else "still_moving_at_window_end"
        result = {
            "schema": PROGRESS_SCHEMA, "segment": segment, "outcome": outcome, "native_actions": self.ticks,
            "still_native_actions": held["still"] if held else int(self.monitor.counts[arm]),
            "last_joint_step_rad": held["last_step"] if held else self.last_step[arm],
            "measured_joints_rad": joints, "joint_residual_max_abs_rad": _max_abs(joints, self.targets[arm]),
        }
        return {**result, "hold": HOLD_AT_STALL} if held else result


def drive_until_settled(
    execution: Any, *, segment: str, joint_targets: Mapping[str, Sequence[float]],
    arrived: Mapping[str, Arrival], budget: int,
) -> tuple[int, NativeActionStallMonitor]:
    """Drive one synchronized joint segment until every arm arrived or stalled, or the budget ends."""
    session = execution.session
    start = session._proprioception()["end_effectors"]
    stall = NativeActionStallMonitor({arm: start[arm]["arm_joint_qpos"] for arm in joint_targets},
                                     joint_targets, budget=budget)
    # The native loop re-sends this one command every native action; a stalled
    # arm's entry is replaced so it holds where it stopped while others move on.
    control = {execution.arm_key(arm): {"position": [float(value) for value in joints]}
               for arm, joints in joint_targets.items()}

    def settled(current: Mapping[str, Any]) -> bool:
        # Called exactly once after every native action.
        stall.sample({arm: current[arm]["arm_joint_qpos"] for arm in joint_targets})
        for arm in joint_targets:
            if arm not in stall.latched and not arrived[arm](current[arm]) and stall.frozen(arm):
                control[execution.arm_key(arm)] = {"position": stall.latch(arm, current[arm])}
        return all(arrived[arm](current[arm]) or stall.stalled(arm) for arm in joint_targets)

    used = session._run_native_control(
        action_id=str(execution.action["action_id"]), control=control, held_targets=execution.held_targets,
        journal=execution.journal, tracking=execution.tracking, segment=segment, reached=settled,
        budget=budget,
    )
    execution.used += used
    _hold_last_stalls(execution, stall)
    return used, stall


def _hold_last_stalls(execution: Any, stall: NativeActionStallMonitor) -> None:
    """Make each arm that stalled on the segment's last native action hold where it stopped."""
    pending = {arm: item for arm, item in stall.latched.items() if item["tick"] == stall.ticks}
    if not pending:
        return
    session = execution.session
    if not session.is_terminal and execution.remaining() > 0:
        execution.used += session._run_native_control(
            action_id=str(execution.action["action_id"]), held_targets=execution.held_targets,
            control={execution.arm_key(arm): {"position": list(item["joints"])} for arm, item in pending.items()},
            journal=execution.journal, tracking=execution.tracking, segment=HOLD_SEGMENT,
            reached=lambda _current: True, budget=1,
        )
        return
    # No native action left: the next command's resolution reads this retained target.
    retained = session.environment.robot_manager.control_manager.prev_control[0]
    for arm, item in pending.items():
        key = execution.arm_key(arm)
        retained[key] = {**retained[key], "position": list(item["joints"])}


def segment_stalled(progress: Iterable[Mapping[str, Any]]) -> bool:
    """Some arm did not arrive, and every arm that did not arrive stood still."""
    outcomes = [item["outcome"] for item in progress]
    return "stalled" in outcomes and all(outcome in {"arrived", "stalled"} for outcome in outcomes)


def record_progress(
    execution: Any, stall: NativeActionStallMonitor, after: Mapping[str, Any], *, segment: str,
    arrived: Mapping[str, Arrival],
) -> None:
    """Keep each driven arm's latest evidence and hold, and whether the segment stalled."""
    terminal = bool(execution.session.is_terminal)
    state: ActionStallState = execution.stall
    latest = {}
    for arm in stall.targets:
        reached = arrived[arm](after[arm])
        latest[arm] = stall.progress(arm, after[arm], segment=segment, arrived=reached, terminal=terminal)
        state.holds.pop(arm, None)
        if stall.stalled(arm) and not reached:
            state.holds[arm] = dict(stall.latched[arm])
            execution.session.pose_targets[arm] = list(stall.latched[arm]["pose"])
    state.progress.update(latest)
    state.stalled = segment_stalled(latest.values())


def pose_target(execution: Any, arm: str, default: Sequence[float]) -> list[float]:
    """The pose an arm now holds: where it stalled, else ``default``."""
    held = execution.stall.holds.get(arm)
    return list(held["pose"]) if held else [float(value) for value in default]


def not_reached_code(execution: Any, otherwise: str) -> str:
    """Why a driven segment did not arrive: official terminal, stall, else ``otherwise``."""
    if execution.session.is_terminal:
        return TERMINAL_CODE
    return STALL_CODE if execution.stall.stalled else otherwise


__all__ = [
    "ActionStallState", "BUDGET_CODE", "HOLD_AT_STALL", "HOLD_SEGMENT", "NativeActionStallMonitor",
    "PROGRESS_SCHEMA", "STALL_CODE", "STALL_JOINT_STEP_RAD", "STALL_WINDOW_NATIVE_ACTIONS", "TERMINAL_CODE",
    "drive_until_settled", "not_reached_code", "pose_arrival", "pose_target", "record_progress",
    "segment_stalled", "stall_rule",
]
