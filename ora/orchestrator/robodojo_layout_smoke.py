"""Scheduled-layout rounds of the maintained RoboDojo scene smoke.

After the round-0 scene checks, a session with a per-round layout schedule
restarts once per later scheduled layout. Each restart must relaunch the
simulation on the planned layout and pass the worker's isolation proof; the
fresh scene must deliver valid RGB-D, both arms must be IK-reachable at their
current pose, and every reach object of the layout (its free rigid bodies,
``robodojo_layout_reach``) must have one arm with a reachable top-down approach
above its centre and above every point of its horizontal footprint: a grasp
lands on the object's extent, not on its centre. Object positions come from the
layout file and object sizes from the pinned asset metadata, both on the
trusted side, and are used only to aim this read-only ``align_plan``; nothing
moves. The worker's GPU memory is logged after every relaunch.

No terminal sweep, per-layout card or motion sweep, and no task-success claim.
``robodojo_layout_revalidation.validate_scheduled_scene`` checks what this
writes. Maintained copy (integration 0062 U12) of the earlier B2 RoboDojo
layout smoke; the checks and the report fields are the same.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

from ora.environment.benchmarks.robodojo_layout_reach import layout_reach_objects
from ora.orchestrator.robodojo_layout_revalidation import SCHEDULE_CHECKS

#: Height of link6 above a layout object's spawn centre for the read-only
#: top-down approach: the ARX contact point sits 0.1501 m along link6 +x, so
#: 0.20 m puts it about 5 cm above the centre of a 0.05 m-high bowl, just above
#: its rim. The footprint points are queried at the same height, so the measured
#: reach envelopes (horizontal distance to the nearer arm base) keep their meaning.
TOP_DOWN_HOVER_M = 0.20
ARMS = ('left', 'right')
#: Rotations of the top-down pose about world z; any reachable one counts.
TOP_DOWN_YAWS_DEG = (0.0, 90.0, 180.0, -90.0)


def _multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def top_down_orientation(yaw_deg: float) -> list[float]:
    """xyzw quaternion turning ARX link6 +x (its approach axis) to world -z, then yawing about z."""
    pitch = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
    half = math.radians(yaw_deg) / 2
    return [round(value, 10) for value in _multiply([0.0, 0.0, math.sin(half), math.cos(half)], pitch)]


def _top_down_arms(session: Any, target_ref: str, x: float, y: float, z: float) -> list[str]:
    poses = [[x, y, z, *top_down_orientation(yaw)] for yaw in TOP_DOWN_YAWS_DEG]
    answer = session.align_plan({'schema': 'ora.align-plan-request.v1', 'target_ref': target_ref,
                                 'approach_side': 'top', 'arm': 'both', 'target_poses_xyzw': poses})
    return sorted({candidate['arm'] for candidate in answer['candidates'] if candidate['reachable']})


def object_reach(session: Any, objects: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Read-only top-down IK above each object's centre and footprint points.

    An object counts as reachable only through an arm that reaches the centre
    and every footprint point (any of the four yaws per point); each point
    records which arms reached it, so a failure names the unreachable side.
    """
    result = {}
    for item in objects:
        x, y, z = item['position_m']
        hover = z + TOP_DOWN_HOVER_M
        points = [{'point': 'centre', 'position_m': [x, y, hover],
                   'reachable_arms': _top_down_arms(session, f"smoke-top-down-{item['label']}-centre", x, y, hover)}]
        for edge in item['footprint']['points']:
            px, py = edge['xy_m']
            points.append({'point': edge['point'], 'bearing_deg': edge['bearing_deg'], 'position_m': [px, py, hover],
                           'reachable_arms': _top_down_arms(
                               session, f"smoke-top-down-{item['label']}-{edge['point']}", px, py, hover)})
        result[item['label']] = {
            'position_m': item['position_m'], 'hover_m': TOP_DOWN_HOVER_M, 'metadata': item['metadata'],
            'footprint_half_extents_m': item['footprint']['half_extents_m'], 'points': points,
            'reachable_arms': [arm for arm in ARMS if all(arm in point['reachable_arms'] for point in points)],
            'unreached_points': {arm: [point['point'] for point in points if arm not in point['reachable_arms']]
                                 for arm in ARMS},
        }
    return result


def unreachable_objects(reaches: list[Mapping[str, Any]]) -> list[str]:
    """``round <k> <label>: <arm> misses <point>@<bearing>, ...`` for every object no single arm covers."""
    failures = []
    for index, reach in enumerate(reaches):
        for label, value in reach.items():
            if value['reachable_arms']:
                continue
            bearings = {point['point']: point.get('bearing_deg') for point in value['points']}
            misses = '; '.join(
                f"{arm} misses " + ', '.join(name if bearings[name] is None else f'{name}@{bearings[name]}deg'
                                             for name in names)
                for arm, names in value['unreached_points'].items())
            failures.append(f'round {index} {label}: {misses}')
    return failures


def worker_gpu_memory_mib(pid: int) -> int | None:
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    used = [int(row.split(',')[1]) for row in rows.splitlines() if row.split(',')[0].strip() == str(pid)]
    return sum(used) if used else None


def scheduled_rounds(session: Any, report: dict[str, Any], root: Path, runtime: Mapping[str, Any], *,
                     pid: int, save: Callable[[], None],
                     sensor_evidence: Callable[[Any, Any], dict[str, Any]]) -> None:
    """Round-0 object reach, then one relaunch per later scheduled layout with its checks."""
    assets = Path(os.environ[runtime['asset_root_env']])
    schedule = runtime['layout_schedule']['rounds']
    report['round0_object_reach'] = object_reach(session, layout_reach_objects(assets, schedule[0]['layout_file']))
    save()
    rounds = report.setdefault('rounds', [])
    for index, entry in enumerate(schedule[1:], start=1):
        session.abort('qualification_boundary')
        started = time.monotonic()
        restart = session.restart_round(index, seed=0)
        record = {'round_index': index, 'layout_id': entry['layout_id'],
                  'restart_wall_s': round(time.monotonic() - started, 3),
                  'worker_gpu_memory_mib': worker_gpu_memory_mib(pid), 'restart': restart}
        rounds.append(record)
        if index == 1:
            report['restart'] = restart
        save()
        assert restart['round_isolation']['passed'] is True, f'round {index} isolation failed'
        assert (restart.get('scene_reset_mode') == 'simulation_relaunch'
                and restart.get('layout_id') == entry['layout_id']
                and restart.get('layout_sha256') == entry['layout_sha256']), f'round {index} did not relaunch its layout'
        if index == 1:
            report['checks']['round_reset'] = True
        observation, images = session.observe(root / 'observations')
        record['sensors'] = sensor_evidence(observation, images)
        record['alignment'] = {}
        for arm in ('left', 'right'):
            pose = observation['proprioception']['end_effectors'][arm]['pose_xyz_xyzw']
            record['alignment'][arm] = session.align_plan({
                'schema': 'ora.align-plan-request.v1', 'target_ref': 'smoke-current-pose',
                'approach_side': 'top', 'arm': arm, 'target_poses_xyzw': [pose]})
        record['object_reach'] = object_reach(session, layout_reach_objects(assets, entry['layout_file']))
        save()
    report['checks']['scheduled_relaunch'] = len(rounds) == len(schedule) - 1
    report['checks']['scheduled_rgbd'] = all(
        set(item['sensors']) == {'head', 'left_wrist', 'right_wrist'} for item in rounds)
    report['checks']['scheduled_current_pose_ik'] = all(
        result['candidates'][0]['reachable'] is True for item in rounds for result in item['alignment'].values())
    reaches = [report['round0_object_reach'], *(item['object_reach'] for item in rounds)]
    unreachable = unreachable_objects(reaches)
    report['checks']['scheduled_object_reach'] = all(bool(reach) for reach in reaches) and not unreachable
    save()
    failed = [name for name in SCHEDULE_CHECKS if report['checks'][name] is not True]
    assert not failed, ('scheduled-layout smoke checks failed: ' + ', '.join(failed)
                        + ''.join(f'; {line}' for line in unreachable))
