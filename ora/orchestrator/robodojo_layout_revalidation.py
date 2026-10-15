"""Scheduled-layout half of a RoboDojo lane gate's scene-smoke revalidation.

A lane whose RunPlan schedules one layout per round is certified by a scene
smoke that ran that exact schedule: after the round-0 checks it restarted once
per later layout, and every restart relaunched the simulation on its planned
layout, passed the worker's scheduled isolation proof, delivered RGB-D, kept
both arms IK-reachable at their current pose and, for every reach object of
the layout (``robodojo_layout_reach``), one arm with a reachable top-down
approach above its centre and every footprint point. The unscheduled smoke
keeps its same-plant identity comparison unchanged in ``robodojo_revalidation``.
"""
from __future__ import annotations

from typing import Any, Mapping

from ora.environment.benchmarks.robodojo import (
    RoboDojoBenchmarkError, validate_scene_registration,
)
from ora.environment.benchmarks.robodojo_layout_reach import FOOTPRINT_POINTS
from .robodojo_layout_evaluation import (
    SCENE_RESET_INITIAL, SCENE_RESET_RELAUNCH, SCHEDULED_ISOLATION_CHECKS,
    SCHEDULED_ROUND_RESET_SCHEMA, SCHEDULED_START_RESET_SCHEMA, render_time_errors, scheduled_reset_invalid,
    source_schedule_errors,
)

#: The smoke checks a scheduled lane adds (``orchestrator/robodojo_layout_smoke.py``; an earlier
#: RoboDojo layout smoke wrote the same checks).
SCHEDULE_CHECKS = ('scheduled_relaunch', 'scheduled_rgbd', 'scheduled_current_pose_ik', 'scheduled_object_reach')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RoboDojoBenchmarkError(message)


def _covering_arms(value: Any) -> list[str] | None:
    """The arms that reached an object's centre and every footprint point, from its per-point records."""
    points = value.get('points') if isinstance(value, Mapping) else None
    if not (isinstance(points, list) and len(points) == FOOTPRINT_POINTS + 1
            and all(isinstance(point, Mapping) and isinstance(point.get('reachable_arms'), list) for point in points)
            and [point.get('point') for point in points] == ['centre', *(f'edge{i}' for i in range(FOOTPRINT_POINTS))]):
        return None
    return [arm for arm in ('left', 'right') if all(arm in point['reachable_arms'] for point in points)]


def _reachable(reach: Any) -> bool:
    """One arm covers every object's centre and footprint, re-derived from the per-point evidence.

    A centre-only record (a smoke from before footprint reach) carries no
    points and is refused, so it cannot certify a lane.
    """
    return isinstance(reach, Mapping) and bool(reach) and all(
        isinstance(value, Mapping) and bool(value.get('reachable_arms'))
        and _covering_arms(value) == value['reachable_arms'] for value in reach.values())


def validate_scheduled_scene(runtime: Mapping[str, Any], report: Mapping[str, Any],
                             source: Mapping[str, Any],
                             synchronizations: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Refuse a smoke that did not run and pass this runtime's layout schedule.

    ``synchronizations`` are the smoke worker's render-synchronization records:
    after every relaunch the cameras must have matched on that epoch's offset.
    """
    schedule = runtime['layout_schedule']['rounds']
    require(report.get('layout_schedule') == [entry['layout_id'] for entry in schedule],
            'scene smoke ran another layout schedule')
    require(all(report.get('checks', {}).get(name) is True for name in SCHEDULE_CHECKS),
            'scheduled-layout smoke checks incomplete')
    require(not source_schedule_errors({'runtime': runtime}, source),
            'scene smoke worker hashed other scheduled layout files')
    reset = report['reset']
    require(reset.get('schema') == SCHEDULED_START_RESET_SCHEMA and reset.get('layout_id') == schedule[0]['layout_id']
            and reset.get('layout_sha256') == schedule[0]['layout_sha256']
            and reset.get('scene_reset_mode') == SCENE_RESET_INITIAL and reset.get('physics_clock_epoch') == 0
            and not scheduled_reset_invalid({'runtime': runtime}, reset, 0),
            'scene smoke round 0 did not start on the scheduled layout')
    require(_reachable(report.get('round0_object_reach')),
            'round-0 layout objects have no reachable top-down approach over their whole footprint')
    rounds = report.get('rounds')
    require(isinstance(rounds, list) and len(rounds) == len(schedule) - 1, 'scheduled rounds missing from the smoke')
    limit = runtime['native_action_limit']
    memory = []
    previous_reset = reset
    for index, (record, entry) in enumerate(zip(rounds, schedule[1:]), start=1):
        restart = record['restart']
        isolation = restart.get('round_isolation') or {}
        checks = isolation.get('checks') or {}
        require(record.get('round_index') == index and restart.get('schema') == SCHEDULED_ROUND_RESET_SCHEMA
                and restart.get('round_index') == index and restart.get('layout_id') == entry['layout_id']
                and restart.get('layout_sha256') == entry['layout_sha256']
                and restart.get('env_seed') == entry['layout_id'] and restart.get('physics_clock_epoch') == index
                and restart.get('scene_reset_mode') == SCENE_RESET_RELAUNCH
                and not scheduled_reset_invalid({'runtime': runtime}, restart, index, previous_reset),
                f'scheduled round {index} did not relaunch its planned layout')
        previous_reset = restart
        require(isolation.get('passed') is True and all(checks.get(name) is True for name in SCHEDULED_ISOLATION_CHECKS),
                f'scheduled round {index} isolation failed')
        validate_scene_registration(restart['scene_registration'], native_action_limit=limit)
        sensors = record.get('sensors') or {}
        require(set(sensors) == {'head', 'left_wrist', 'right_wrist'} and all(
            sensor['shape'] == [480, 640] and sensor['finite_positive_fraction'] > 0.5 for sensor in sensors.values()),
            f'scheduled round {index} RGB-D is invalid')
        require(all(record['alignment'][arm]['candidates'][0]['reachable'] is True for arm in ('left', 'right')),
                f'scheduled round {index} current-pose IK failed')
        require(_reachable(record.get('object_reach')),
                f'scheduled round {index} layout objects have no reachable top-down approach '
                'over their whole footprint')
        memory.append(record.get('worker_gpu_memory_mib'))
    require(not render_time_errors(synchronizations, [reset, *(record['restart'] for record in rounds)]),
            'scene smoke cameras did not match each epoch on its simulation-time offset')
    return {'layout_schedule': [entry['layout_id'] for entry in schedule],
            'worker_gpu_memory_mib_after_relaunch': memory}


__all__ = ['SCHEDULE_CHECKS', 'validate_scheduled_scene']
