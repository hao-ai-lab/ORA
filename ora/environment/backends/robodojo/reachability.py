"""Read-only link6 kinematics using the installed SDK's own robot model.

The inverse world/base transform mirrors RobotManager.solve_ik. No task,
collision object, planner trajectory, controller write or physics tick is read.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from .common import array
from ora.environment.controllers.reachability import cartesian_pose_error_twist


def fk_link6(session: Any, arm: str, joints: Sequence[float]) -> list[float]:
    import transforms3d as t3d

    robot = session.robots[arm]
    planner = session.environment.robot_manager.ik_solver[robot.robot_name]
    state = planner._build_joint_state(joints)
    poses = planner.ik_solver.compute_kinematics(state).tool_poses.to_dict()
    pose = poses[planner.ee_link]
    position = array(pose.position).reshape(-1, 3)[0] - np.asarray(planner.frame_bias)
    quaternion = array(pose.quaternion).reshape(-1, 4)[0]
    # calculate_target_pose maps the actual entity base to the planner base;
    # undoing its world->base transform leaves this actual-base multiplication.
    base = np.asarray(robot.entity_origin_pose, dtype=float)
    rotation = t3d.quaternions.quat2mat(base[3:])
    world_position = base[:3] + rotation @ position
    world_quaternion = t3d.quaternions.qmult(base[3:], quaternion)
    result = [*world_position, *world_quaternion[1:], world_quaternion[0]]
    if not np.isfinite(result).all():
        raise ValueError('RoboDojo FK returned non-finite link6 pose')
    return [float(x) for x in result]


def fk_calibrated(session: Any, arm: str, measured: Mapping[str, Any]) -> bool:
    """Whether FK of the measured joints lands on the measured link6 pose."""
    current_fk = fk_link6(session, arm, measured['arm_joint_qpos'])
    calibration = cartesian_pose_error_twist(measured['pose_xyz_xyzw'], current_fk)
    return not (np.linalg.norm(calibration[:3]) > 0.002 or np.linalg.norm(calibration[3:]) > 0.02)


def solve_batch(session: Any, arm: str, poses: Sequence[Sequence[float]], *,
                position_tolerance_m: float, orientation_tolerance_rad: float) -> list[Mapping[str, Any]]:
    from .sparse_motion import solve_ik
    measured = session._proprioception()['end_effectors'][arm]
    if not fk_calibrated(session, arm, measured):
        raise ValueError('RoboDojo read-only FK differs from the current measured link6 frame')
    result = []
    for pose in poses:
        solved = solve_ik(session, arm, pose)
        joints = solved.get('joint_target')
        error = None if joints is None else cartesian_pose_error_twist(pose, fk_link6(session, arm, joints))
        position = None if error is None else float(np.linalg.norm(error[:3]))
        rotation = None if error is None else float(np.linalg.norm(error[3:]))
        jump = solved.get('ik_selection', {}).get('outcome') == 'joint_jump'
        reachable = not jump and error is not None and position <= position_tolerance_m and rotation <= orientation_tolerance_rad
        result.append({'reachable': bool(reachable), 'position_error_m': position,
            'orientation_error_rad': rotation, 'iterations': 1,
            'reason': None if reachable else ('ik_failed' if joints is None else 'ik_branch_jump' if jump else 'fk_residual_exceeded'),
            'ik_selection': solved.get('ik_selection'),
            'claim_boundary': 'SDK IK and FK only; no physics, collision-free path, contact or task-success claim'})
    return result
