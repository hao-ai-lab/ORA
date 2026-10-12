"""Pinned RoboDojo motion measures (U1 v2 layer L0): definitions and per-run partial counts.

Each measure turns one ``RunJournal`` into a *partial*: integer ``counts``,
lists of ``values`` and lists of ``records``. Partials of several runs merge by
addition, and ``robodojo_motion_metrics.summarize`` renders them as ratios
(numerator, denominator, share) and distributions. Nothing here imports Isaac
or writes a file.

Two cohorts group the arm-ticks and the jaw episodes. *closed_command*: the
arm's recorded gripper directive is ``close``. *holding_mode*: the v1 form of
v2's holding-mode rule, i.e. after a ``set_gripper`` close that ended
``stalled_before_target``, until the next ``set_gripper`` naming that arm, the
round end, or the jaw reaching its floor.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ora.contracts.robodojo_arx import ARX_X5_CONTRACT

from .robodojo_motion_journal import (
    Action, RunJournal, SegmentRun, Tick, applied_steps, arm_target, command_segments, commanded_arms,
    joint_distance, joint_path, position_path, segment_class, segment_runs,
)

METRICS_VERSION = "robodojo-motion-metrics.v2"

#: ARX X5 contract: 4 ms physics step, 25 Hz native action (10 steps).
PHYSICS_DT_S = float(ARX_X5_CONTRACT["physics_dt_s"])
NATIVE_ACTION_S = 1.0 / float(ARX_X5_CONTRACT["control_hz"])
#: Upstream RoboDojo ee67a146 ``env/robot_manager/robot_config/x5.py:37-45``
#: (implicit PD actuator, ``velocity_limit_sim`` 5 rad/s).
X5_JOINT_VELOCITY_LIMIT_RAD_S = 5.0
#: U1 v2 design, L0: "a joint of the commanded arm reaches 0.95 x 5 rad/s".
SATURATION_FRACTION = 0.95
SATURATED_JOINT_SPEED_RAD_S = SATURATION_FRACTION * X5_JOINT_VELOCITY_LIMIT_RAD_S
#: The same limit per native action: 5 rad/s x 40 ms = 0.2 rad per tick.
SATURATED_TICK_DISPLACEMENT_RAD = SATURATION_FRACTION * X5_JOINT_VELOCITY_LIMIT_RAD_S * NATIVE_ACTION_S
#: The ``controller_stalled`` rule owned by
#: ``environment/backends/robodojo/motion_stall.py`` (``STALL_JOINT_STEP_RAD``,
#: ``STALL_WINDOW_NATIVE_ACTIONS``). Host layers may not import the isolated
#: simulator backends (``tests/test_harness_wrapper_refactor.py``), so the
#: values are restated here and ``tests/test_robodojo_motion_metrics.py`` pins
#: them equal to the owner's.
STILL_JOINT_STEP_RAD = 0.005
STILL_WINDOW_NATIVE_ACTIONS = 3
#: U1 v2 design, ARX profile ``in_tick_command``: candidate v_cmd 4.0 rad/s.
V_CMD_CANDIDATE_RAD_S = 4.0
#: U1 v2 design, L0 "out of step": joints moving more than 0.05 rad, progress
#: spread above 0.3; 0.5 is the bottleneck review's own threshold.
OUT_OF_STEP_MINIMUM_EXCURSION_RAD = 0.05
OUT_OF_STEP_SPREADS = (0.3, 0.5)
#: U1 v2 design, L0 "sag": link6 below min(z start, z goal) - 0.02 m.
SAG_MARGIN_M = 0.02
#: Pose segments with a receipt goal pose; recovery segments are not transport.
SAG_SEGMENTS = frozenset({"move", "plan_ref", "precontact"})
#: ARX X5 contract arrival and gripper tolerances.
POSITION_TOLERANCE_M = float(ARX_X5_CONTRACT["position_tolerance_m"])
ORIENTATION_TOLERANCE_RAD = float(ARX_X5_CONTRACT["orientation_tolerance_rad"])
GRIPPER_TOLERANCE = float(ARX_X5_CONTRACT["gripper_tolerance_normalized"])
#: U1 v2 design, L0 "held then lost": private object-link6 displacement (replay only).
HELD_OBJECT_DISPLACEMENT_M = 0.02
#: ``agent/tools/arx_x5_gripper.ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"]``:
#: the pad-face centre along link6 +x, where a held object sits.
JAW_CONTACT_CENTRE_LINK6_X_M = 0.1501
#: Frame sanity bound for replayed object poses: the median distance from the
#: selected object to the jaw contact centre at the close. The pads are 15 mm
#: long and an env-origin offset would be metres, so this separates the two.
OBJECT_FRAME_SANITY_M = 0.05
#: Proposed ARX profile ``waypoint_execution.joint_convergence_tolerance_rad`` (candidate).
TRACKING_TOLERANCE_RAD = 0.03
#: U1 v2 design, L2 proposed bound on measured over planned peak speed.
PEAK_RATIO_BOUND = 1.3
POSE_ACTIONS = ("move_pose", "move_free_space")
TIMEOUT_CODE = "native_action_budget_exhausted"
STALLED_CLOSE = "stalled_before_target"
COHORTS = ("closed_command", "holding_mode")

PlannedWaypoints = Mapping[tuple[int, str], Mapping[str, Any]]
ObjectPoses = Mapping[tuple[int, int], Mapping[str, Sequence[float]]]


@dataclass(frozen=True)
class JawEpisode:
    """One ``set_gripper`` close of one arm and the ticks until the arm's next jaw command."""

    arm: str
    close: Action
    floor: float
    stalled: bool
    ticks: tuple[Tick, ...]


def partial(counts: Mapping[str, int] | None = None, values: Mapping[str, list[float]] | None = None,
            records: Mapping[str, list[Any]] | None = None) -> dict[str, Any]:
    return {"counts": dict(counts or {}), "values": dict(values or {}), "records": dict(records or {})}


def merge_partials(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    values: defaultdict[str, list[Any]] = defaultdict(list)
    records: defaultdict[str, list[Any]] = defaultdict(list)
    for part in parts:
        counts.update(part.get("counts", {}))
        for key, items in part.get("values", {}).items():
            values[key].extend(items)
        for key, items in part.get("records", {}).items():
            records[key].extend(items)
    return partial(counts, dict(values), dict(records))


def _pairs(items: Sequence[Sequence[float]]) -> Iterator[tuple[Sequence[float], Sequence[float]]]:
    return zip(items, items[1:])


def _peak_step(path: Sequence[Sequence[float]]) -> float:
    return max((joint_distance(a, b) for a, b in _pairs(path)), default=0.0)


def _arm_ticks(journal: RunJournal) -> Iterator[tuple[Tick, str]]:
    for item in journal.actions:
        for entry in item.ticks:
            for arm in commanded_arms(entry):
                yield entry, arm


def _names_arm(receipt: Mapping[str, Any], arm: str) -> bool:
    targets = receipt.get("arguments", {}).get("targets", {})
    return receipt.get("action") == "set_gripper" and isinstance(targets, Mapping) and arm in targets


def jaw_episodes(journal: RunJournal) -> tuple[JawEpisode, ...]:
    """Every close command with the ticks until that arm's next jaw command or the round end."""
    episodes = []
    for position, item in enumerate(journal.actions):
        receipt = item.receipt
        if receipt.get("action") != "set_gripper" or not item.ticks:
            continue
        for arm, directive in receipt.get("arguments", {}).get("targets", {}).items():
            if directive != "close":
                continue
            later = []
            for following in journal.actions[position + 1:]:
                if following.round_index != item.round_index or _names_arm(following.receipt, arm):
                    break
                later.extend(following.ticks)
            outcome = receipt.get("arms", {}).get(arm, {}).get("gripper_outcome") or {}
            floor = float(receipt.get("gripper_floors", {}).get(arm, 0.0))
            episodes.append(JawEpisode(arm, item, floor, outcome.get("state") == STALLED_CLOSE, tuple(later)))
    return tuple(episodes)


def _opening(entry: Tick, arm: str) -> float:
    return float(entry.measured[arm]["gripper"])


def jaw_loss(episode: JawEpisode) -> Tick | None:
    """The first tick end at the jaw floor after a close that ended above it; else ``None``."""
    limit = episode.floor + GRIPPER_TOLERANCE
    if _opening(episode.close.ticks[-1], episode.arm) <= limit:
        return None
    return next((entry for entry in episode.ticks if _opening(entry, episode.arm) <= limit), None)


def _holds(episode: JawEpisode) -> bool:
    """The close ended with the jaw above its floor: something is between the pads."""
    return _opening(episode.close.ticks[-1], episode.arm) > episode.floor + GRIPPER_TOLERANCE


def holding_keys(episodes: Sequence[JawEpisode]) -> frozenset[tuple[int, int, str]]:
    """Arm-ticks inside a holding-mode episode before its jaw loss; a close that ended at the floor holds nothing."""
    keys = set()
    for episode in episodes:
        if not episode.stalled or not _holds(episode):
            continue
        loss = jaw_loss(episode)
        for entry in episode.ticks:
            if entry is loss:
                break
            keys.add((entry.action_sequence, entry.index, episode.arm))
    return frozenset(keys)


def _cohorts(entry: Tick, arm: str, holding: frozenset[tuple[int, int, str]]) -> list[str]:
    names = ["all"]
    if entry.measured.get(arm, {}).get("gripper_command_target") == "close":
        names.append("closed_command")
    if (entry.action_sequence, entry.index, arm) in holding:
        names.append("holding_mode")
    return names


def speed_measures(journal: RunJournal, holding: frozenset[tuple[int, int, str]], v_cmd: float) -> dict[str, Any]:
    """Per arm-tick: physics-step saturation, commanded-speed violation and link6 peak speed, by cohort."""
    counts: Counter[str] = Counter()
    values: defaultdict[str, list[float]] = defaultdict(list)
    for entry, arm in _arm_ticks(journal):
        joint_speed = _peak_step(joint_path(entry, arm)) / PHYSICS_DT_S
        link6_speed = max(math.dist(a, b) for a, b in _pairs(position_path(entry, arm))) / PHYSICS_DT_S
        applied = applied_steps(entry, arm)
        commanded = None if applied is None else _peak_step(applied) / PHYSICS_DT_S
        counts["commanded_speed.unrecorded"] += int(commanded is None)
        saturated = int(joint_speed >= SATURATED_JOINT_SPEED_RAD_S)
        for cohort in _cohorts(entry, arm, holding):
            counts.update({f"saturation_physics_step.{cohort}.denominator": 1,
                           f"saturation_physics_step.{cohort}.numerator": saturated})
            values[f"link6_peak_speed.{cohort}"].append(link6_speed)
            if commanded is not None:
                counts.update({f"commanded_speed.{cohort}.denominator": 1,
                               f"commanded_speed.{cohort}.numerator": int(commanded > v_cmd)})
                values[f"commanded_speed.{cohort}.peak"].append(commanded)
    return partial(counts, dict(values))


def tick_displacement(journal: RunJournal) -> dict[str, Any]:
    """Moving arm-ticks whose joint change reaches 0.95 of the 0.2 rad per-tick limit."""
    counts: Counter[str] = Counter()
    for entry, arm in _arm_ticks(journal):
        if entry.predecessor is None:
            continue
        joints = joint_path(entry, arm)
        moved = joint_distance(joints[0], joints[-1])
        if moved >= STILL_JOINT_STEP_RAD:
            counts.update({"saturation_tick_displacement.denominator": 1,
                           "saturation_tick_displacement.numerator": int(moved >= SATURATED_TICK_DISPLACEMENT_RAD)})
    return partial(counts)


def _progress_spread(run: SegmentRun, start: Sequence[float], moving: Sequence[int]) -> float:
    spread = 0.0
    for entry in run.ticks:
        for line in entry.steps:
            joints = line["measured"][run.arm]["arm_joint_qpos"]
            progress = [(joints[j] - start[j]) / (run.target[j] - start[j]) for j in moving]
            spread = max(spread, max(progress) - min(progress))
    return spread


def out_of_step(runs: Sequence[SegmentRun]) -> dict[str, Any]:
    """v1 only: segments with a constant endpoint target; a changing target is not counted."""
    counts: Counter[str] = Counter()
    for run in runs:
        first = run.ticks[0]
        if segment_class(run.segment) is None or first.predecessor is None:
            continue
        if not run.constant_target:
            counts["out_of_step.changing_target"] += 1
            continue
        start = joint_path(first, run.arm)[0]
        moving = [j for j, (a, b) in enumerate(zip(start, run.target))
                  if abs(b - a) > OUT_OF_STEP_MINIMUM_EXCURSION_RAD]
        if len(moving) < 2:
            continue
        spread = _progress_spread(run, start, moving)
        for threshold in OUT_OF_STEP_SPREADS:
            counts.update({f"out_of_step.{threshold}.denominator": 1,
                           f"out_of_step.{threshold}.numerator": int(spread > threshold)})
    return partial(counts)


def _planned_ticks(journal: RunJournal,
                   planned: PlannedWaypoints) -> Iterator[tuple[Tick, str, int, Mapping[str, Any]]]:
    actions = {item.sequence: item for item in journal.actions}
    for (sequence, arm), row in sorted(planned.items()):
        item = actions.get(sequence)
        ticks = [] if item is None else [entry for entry in item.ticks if arm in commanded_arms(entry)]
        for index, entry in enumerate(ticks):
            yield entry, arm, min(index, len(row["positions"]) - 1), row


def tracking_error(journal: RunJournal, planned: PlannedWaypoints | None) -> dict[str, Any]:
    """Tick-end joint error against the waypoint; for v1 the recorded endpoint target stands in."""
    counts: Counter[str] = Counter()
    errors: list[float] = []
    if planned is None:
        pairs = [(entry, arm, arm_target(entry, arm)) for entry, arm in _arm_ticks(journal)]
    else:
        pairs = [(entry, arm, row["positions"][k]) for entry, arm, k, row in _planned_ticks(journal, planned)]
    for entry, arm, waypoint in pairs:
        error = joint_distance(entry.measured[arm]["arm_joint_qpos"], waypoint)
        errors.append(error)
        counts.update({"tracking_error.denominator": 1,
                       "tracking_error.numerator": int(error > TRACKING_TOLERANCE_RAD)})
    return partial(counts, {"tracking_error.error_rad": errors})


def _planned_peak(row: Mapping[str, Any], start: Sequence[float] | None) -> float:
    positions = [list(map(float, waypoint)) for waypoint in row["positions"]]
    return _peak_step(([list(start)] if start is not None else []) + positions) / NATIVE_ACTION_S


def planned_speed_measures(journal: RunJournal, planned: PlannedWaypoints, v_cmd: float) -> dict[str, Any]:
    """Peak ratio per planned segment; impulse saturation where a predicted in-tick command is supplied."""
    counts: Counter[str] = Counter()
    ratios: list[float] = []
    peaks: defaultdict[tuple[int, str], float] = defaultdict(float)
    starts: dict[tuple[int, str], Sequence[float] | None] = {}
    for entry, arm, k, row in _planned_ticks(journal, planned):
        speed = _peak_step(joint_path(entry, arm)) / PHYSICS_DT_S
        key = (entry.action_sequence, arm)
        peaks[key] = max(peaks[key], speed)
        starts.setdefault(key, joint_path(entry, arm)[0] if entry.predecessor is not None else None)
        predicted = row.get("predicted_in_tick_speed_rad_s")
        if predicted is not None and float(predicted[min(k, len(predicted) - 1)]) < v_cmd:
            counts.update({"impulse_saturation.denominator": 1,
                           "impulse_saturation.numerator": int(speed >= SATURATED_JOINT_SPEED_RAD_S)})
    for key, measured in sorted(peaks.items()):
        planned_peak = _planned_peak(planned[key], starts.get(key))
        if planned_peak > 0.0:
            ratios.append(measured / planned_peak)
            counts.update({"peak_ratio.denominator": 1, "peak_ratio.numerator": int(ratios[-1] > PEAK_RATIO_BOUND)})
    return partial(counts, {"peak_ratio.ratio": ratios})


def _rotate(quaternion_xyzw: Sequence[float], vector: Sequence[float], *, inverse: bool) -> list[float]:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    sign = -1.0 if inverse else 1.0
    qx, qy, qz, qw = sign * x / norm, sign * y / norm, sign * z / norm, w / norm
    vx, vy, vz = (float(v) for v in vector)
    tx, ty, tz = 2 * (qy * vz - qz * vy), 2 * (qz * vx - qx * vz), 2 * (qx * vy - qy * vx)
    return [vx + qw * tx + (qy * tz - qz * ty), vy + qw * ty + (qz * tx - qx * tz), vz + qw * tz + (qx * ty - qy * tx)]


def link6_frame_offset(link6_pose_xyz_xyzw: Sequence[float], point: Sequence[float]) -> list[float]:
    """A world point expressed in the link6 frame of ``link6_pose_xyz_xyzw``."""
    offset = [float(p) - float(o) for p, o in zip(point, link6_pose_xyz_xyzw[:3])]
    return _rotate(link6_pose_xyz_xyzw[3:7], offset, inverse=True)


def jaw_contact_centre(link6_pose_xyz_xyzw: Sequence[float]) -> list[float]:
    """The pad-face centre, 0.1501 m along link6 +x, in the pose's frame."""
    along = _rotate(link6_pose_xyz_xyzw[3:7], (JAW_CONTACT_CENTRE_LINK6_X_M, 0.0, 0.0), inverse=False)
    return [float(p) + a for p, a in zip(link6_pose_xyz_xyzw[:3], along)]


def _object_check(episode: JawEpisode, poses: ObjectPoses, run_id: str) -> dict[str, Any] | None:
    """The object nearest the jaw contact centre at the close and its largest later link6-frame drift."""
    anchor = episode.close.ticks[-1]
    objects = poses.get((anchor.action_sequence, anchor.index))
    if not objects:
        return None
    pose = anchor.measured[episode.arm]["pose_xyz_xyzw"]
    centre = jaw_contact_centre(pose)
    name = min(objects, key=lambda key: math.dist(objects[key], centre))
    reference = link6_frame_offset(pose, objects[name])
    drift = max((math.dist(link6_frame_offset(entry.measured[episode.arm]["pose_xyz_xyzw"], row[name]), reference)
                 for entry in episode.ticks
                 for row in [poses.get((entry.action_sequence, entry.index), {})] if name in row), default=0.0)
    return {"run_id": run_id, "action_sequence": anchor.action_sequence, "arm": episode.arm, "object": name,
            "distance_to_jaw_centre_m": math.dist(objects[name], centre), "max_link6_drift_m": drift,
            "lost": drift > HELD_OBJECT_DISPLACEMENT_M, "stalled_close": episode.stalled}


def held_then_lost(journal: RunJournal, episodes: Sequence[JawEpisode], poses: ObjectPoses | None) -> dict[str, Any]:
    """Closes that ended above the jaw floor (closed_command; stalled ones also holding_mode), then lost."""
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for episode in episodes:
        if not _holds(episode):
            continue
        jaw = jaw_loss(episode) is not None
        check = _object_check(episode, poses, journal.run_id) if poses is not None else None
        records.extend([check] if check is not None else [])
        moved = bool(check and check["lost"])
        for cohort in ("closed_command", *(("holding_mode",) if episode.stalled else ())):
            counts.update({f"held_then_lost.{cohort}.denominator": 1,
                           f"held_then_lost.{cohort}.numerator": int(jaw or moved),
                           f"held_then_lost.{cohort}.jaw_floor": int(jaw),
                           f"held_then_lost.{cohort}.object_displacement": int(moved),
                           f"held_then_lost.{cohort}.object_measured": int(check is not None)})
    return partial(counts, records={"held_then_lost.objects": records})


def sag(runs: Sequence[SegmentRun], holding: frozenset[tuple[int, int, str]]) -> dict[str, Any]:
    """Pose segments whose link6 z falls below min(z start, z goal) - 0.02 m, goal z from the receipt."""
    counts: Counter[str] = Counter()
    for run in runs:
        first = run.ticks[0]
        if run.segment not in SAG_SEGMENTS or first.predecessor is None:
            continue
        goal = (run.entry or {}).get("target_pose_xyz_xyzw")
        if goal is None:
            counts["sag.unmatched"] += 1
            continue
        start = float(first.predecessor["measured"][run.arm]["pose_xyz_xyzw"][2])
        lowest = min(float(line["measured"][run.arm]["pose_xyz_xyzw"][2]) for t in run.ticks for line in t.steps)
        state = "holding" if (first.action_sequence, first.index, run.arm) in holding else "free_space"
        counts.update({f"sag.{state}.denominator": 1,
                       f"sag.{state}.numerator": int(lowest < min(start, float(goal[2])) - SAG_MARGIN_M)})
    return partial(counts)


def _pose_error(target: Sequence[float], measured: Sequence[float]) -> tuple[float, float]:
    """``backends.robodojo.common.pose_error`` without NumPy."""
    distance = math.dist([float(v) for v in target[:3]], [float(v) for v in measured[:3]])
    qa, qb = [float(v) for v in target[3:7]], [float(v) for v in measured[3:7]]
    cosine = min(1.0, max(0.0, abs(sum(a * b for a, b in zip(qa, qb))) / (math.hypot(*qa) * math.hypot(*qb))))
    return distance, 2 * math.acos(cosine)


def _final_targets(receipt: Mapping[str, Any]) -> dict[str, list[float]]:
    targets = receipt.get("arguments", {}).get("targets")
    if isinstance(targets, Mapping) and targets:
        return {arm: [*map(float, value["position_m"]), *map(float, value["orientation_xyzw"])]
                for arm, value in targets.items() if isinstance(value, Mapping)}
    result = {}
    for arm in receipt.get("target_arms", []):
        segments = receipt.get("arms", {}).get(arm, {}).get("segments", [])
        if segments and segments[-1].get("target_pose_xyz_xyzw") is not None:
            result[arm] = [float(v) for v in segments[-1]["target_pose_xyz_xyzw"]]
    return result


def arrival(journal: RunJournal) -> dict[str, Any]:
    """Reached share of every pose command; refusals count as not reached; executed subset beside it."""
    counts: Counter[str] = Counter()
    for item in journal.actions:
        receipt = item.receipt
        if receipt.get("action") not in POSE_ACTIONS:
            continue
        counts.update({f"arrival.status.{receipt.get('execution_status')}": 1, "arrival.denominator": 1})
        targets = _final_targets(receipt)
        if not item.ticks or not targets:
            continue
        measured = item.ticks[-1].measured
        errors = [_pose_error(pose, measured[arm]["pose_xyz_xyzw"]) for arm, pose in targets.items()]
        reached = int(all(d <= POSITION_TOLERANCE_M and a <= ORIENTATION_TOLERANCE_RAD for d, a in errors))
        counts.update({"arrival.numerator": reached, "arrival.executed.denominator": 1,
                       "arrival.executed.numerator": reached})
    return partial(counts)


def native_action_cost(journal: RunJournal) -> dict[str, Any]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    per_round: Counter[int] = Counter()
    for item in journal.actions:
        per_round[item.round_index] += len(item.ticks)
        if item.ticks:
            values[f"native_actions.per_command.{item.receipt.get('action')}"].append(len(item.ticks))
        for segment, count in command_segments(item):
            kind = segment_class(segment)
            if kind is not None:
                family = "recover" if segment.startswith("recover") else segment
                values[f"native_actions.per_segment.{kind}"].append(count)
                values[f"native_actions.by_segment.{family}"].append(count)
    limit = journal.native_action_limit or 0
    return partial({"native_actions.per_round.numerator": sum(per_round.values()),
                    "native_actions.per_round.denominator": limit * len(per_round),
                    "native_actions.per_round.rounds": len(per_round)}, dict(values))


def _targeted_arms(item: Action) -> tuple[str, ...]:
    targets = item.receipt.get("arguments", {}).get("targets")
    if isinstance(targets, Mapping) and targets:
        return tuple(targets)
    if item.receipt.get("target_arms"):
        return tuple(item.receipt["target_arms"])
    return tuple(sorted({arm for entry in item.ticks for arm in commanded_arms(entry)}))


def _ended_still(item: Action, arm: str) -> bool:
    first = item.ticks[0].predecessor if item.ticks else None
    states = ([first["measured"][arm]["arm_joint_qpos"]] if first is not None else []) + [
        entry.steps[-1]["measured"][arm]["arm_joint_qpos"] for entry in item.ticks]
    window = [joint_distance(a, b) for a, b in _pairs(states)][-STILL_WINDOW_NATIVE_ACTIONS:]
    return len(window) == STILL_WINDOW_NATIVE_ACTIONS and all(step < STILL_JOINT_STEP_RAD for step in window)


def timeout_stillness(journal: RunJournal) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for item in journal.actions:
        receipt = item.receipt
        if receipt.get("action") in POSE_ACTIONS and receipt.get("failure_code") == TIMEOUT_CODE:
            arms = _targeted_arms(item)
            counts.update({"timeout_stillness.denominator": 1,
                           "timeout_stillness.numerator": int(bool(arms) and all(_ended_still(item, a) for a in arms))})
    return partial(counts)


def object_frame_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether replayed object positions share link6's frame: the held object sits at the jaw centre."""
    distances = [float(record["distance_to_jaw_centre_m"]) for record in records]
    if not distances:
        return {"status": "unmeasured", "median_distance_to_jaw_centre_m": None, "bound_m": OBJECT_FRAME_SANITY_M}
    middle = median(distances)
    return {"status": "consistent" if middle <= OBJECT_FRAME_SANITY_M else "inconsistent",
            "median_distance_to_jaw_centre_m": middle, "bound_m": OBJECT_FRAME_SANITY_M}


def measure_journal(journal: RunJournal, *, planned: PlannedWaypoints | None = None,
                    object_poses: ObjectPoses | None = None, v_cmd: float = V_CMD_CANDIDATE_RAD_S) -> dict[str, Any]:
    runs, episodes = segment_runs(journal), jaw_episodes(journal)
    holding = holding_keys(episodes)
    parts = [speed_measures(journal, holding, v_cmd), tick_displacement(journal), out_of_step(runs),
             tracking_error(journal, planned), held_then_lost(journal, episodes, object_poses), sag(runs, holding),
             arrival(journal), native_action_cost(journal), timeout_stillness(journal),
             partial({"inputs.planned_waypoints": int(planned is not None),
                      "inputs.object_poses": int(object_poses is not None)})]
    if planned is not None:
        parts.append(planned_speed_measures(journal, planned, v_cmd))
    return merge_partials(parts)


__all__ = [
    "COHORTS", "GRIPPER_TOLERANCE", "HELD_OBJECT_DISPLACEMENT_M", "JAW_CONTACT_CENTRE_LINK6_X_M", "JawEpisode",
    "METRICS_VERSION", "NATIVE_ACTION_S", "OBJECT_FRAME_SANITY_M", "ORIENTATION_TOLERANCE_RAD",
    "OUT_OF_STEP_MINIMUM_EXCURSION_RAD", "OUT_OF_STEP_SPREADS", "PEAK_RATIO_BOUND", "PHYSICS_DT_S",
    "POSITION_TOLERANCE_M", "SAG_MARGIN_M", "SAG_SEGMENTS", "SATURATED_JOINT_SPEED_RAD_S",
    "SATURATED_TICK_DISPLACEMENT_RAD", "STILL_JOINT_STEP_RAD", "STILL_WINDOW_NATIVE_ACTIONS",
    "TRACKING_TOLERANCE_RAD", "V_CMD_CANDIDATE_RAD_S", "holding_keys", "jaw_contact_centre", "jaw_episodes",
    "jaw_loss", "link6_frame_offset", "measure_journal", "merge_partials", "object_frame_check", "partial",
]
