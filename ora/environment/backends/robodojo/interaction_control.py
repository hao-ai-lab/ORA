"""Environment-owned support-arm evidence, separate from policy-held targets."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
import numpy as np

from .common import RobodojoError, array


def policy_targets(session: Any, targets: Mapping[str, Any]) -> dict[str, Any]:
    """Project retained commands to the two policy-controlled target robots."""
    manager = session.environment.robot_manager
    keys = {manager.process_name(name) for robot in session.robots.values()
            for name in (robot.arm_name, robot.gripper_name)}
    return {key: deepcopy(value) for key, value in targets.items() if key in keys}


def support_position_targets(session: Any) -> dict[str, list[float]]:
    """Read submitted SDK targets for the environment's auxiliary robots.

    This is private controller evidence. It supplies no policy authority to
    command the auxiliary arm and does not expose its precomputed trajectory.
    """
    manager = session.environment.robot_manager
    result = {}
    for index, robot in enumerate(manager.robot_list):
        if robot.type == "target":
            continue
        targets = array(manager.robot_key[index].data.joint_pos_target[0])
        for name, indices in ((robot.arm_name, robot.arm_joint_indices),
                              (robot.gripper_name, robot.gripper_joint_indices)):
            values = targets[indices]
            if not np.isfinite(values).all():
                raise RobodojoError("SDK applied an invalid auxiliary joint target")
            result[manager.process_name(name)] = values.tolist()
    return result
