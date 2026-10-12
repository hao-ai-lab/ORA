"""Robot-only MuJoCo geometry collector; no task, rendering, step or controller.

Run with the isolated LIBERO interpreter. Loading a gripper asset and calling
forward kinematics is not an episode or a collision/grasp qualification.
"""
from __future__ import annotations

import math

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def collect_panda_gripper(asset: Path) -> dict:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    # Zero is the asset reference, not a newly issued gripper command.
    mujoco.mj_forward(model, data)
    if model.nq != 2:
        raise ValueError("expected the exact standalone two-finger Panda asset")
    geometry = panda_geometry_from_model(model, data, prefix="", reference_qpos=True)
    mesh_path = asset.parent / "meshes/panda_gripper/finger.stl"
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    return {"schema": "ora.robot-asset-geometry.v1", "backend": "libero_pro",
            "scope": "compiled_robot_gripper_only_no_task_no_step", "simulator_version": mujoco.__version__,
            "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "collision_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "local_axes": {"approach": [0., 0., 1.], "closing": [1., 0., 0.]},
            "joint_ranges_m": model.jnt_range.tolist(),
            "controlled_point_offset_m": model.body_pos[model.site_bodyid[site]].tolist(),
            "geometry": geometry}


def panda_geometry_from_model(model, data, *, prefix: str, reference_qpos: bool = False) -> dict:
    """Read only named robot geoms in a loaded model, without changing state.

    Nonzero finger slide offsets are subtracted analytically. No reference
    pose is applied to the live simulator and no scene geom is traversed.
    """
    import mujoco
    import numpy as np
    # mj_step advances qpos after computing cached geom/site transforms. Mixing
    # those transforms with the new finger qpos shifts the reconstructed static
    # bounds by one integration tick. Refresh only scratch kinematics so the
    # reference subtraction uses one coherent state without touching the plant.
    current = mujoco.MjData(model)
    current.qpos[:] = data.qpos
    current.mocap_pos[:] = data.mocap_pos
    current.mocap_quat[:] = data.mocap_quat
    mujoco.mj_kinematics(model, current)
    data = current
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "grip_site")
    if site < 0:
        raise ValueError("loaded Panda does not expose its declared grip site")
    rotation = data.site_xmat[site].reshape(3, 3)
    position = data.site_xpos[site]

    def bounds(name: str, offset) -> dict:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
        if geom < 0:
            raise ValueError(f"missing robot collision geometry {name}")
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.array(list(itertools.product((-1, 1), repeat=3))) * model.geom_size[geom]
        else:
            raise ValueError("unsupported robot-only collision primitive")
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world-position) @ rotation - offset
        return {"minimum": local.min(axis=0).round(10).tolist(), "maximum": local.max(axis=0).round(10).tolist()}

    fingers = []
    for index, sign in ((1, 1), (2, -1)):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + f"finger_joint{index}")
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError("loaded Panda finger is not the expected slide joint")
        measured_qpos = float(data.qpos[model.jnt_qposadr[joint]])
        if reference_qpos and measured_qpos != 0:
            raise ValueError("standalone gripper reference qpos must be zero")
        offset = measured_qpos * data.xaxis[joint] @ rotation
        fingers.append({"id": f"finger_{index}",
                        "opening_direction_local": (sign*data.xaxis[joint] @ rotation).round(10).tolist(),
                        "collision_bounds_m": bounds(f"finger{index}_collision", offset),
                        "pad_bounds_m": bounds(f"finger{index}_pad_collision", offset)})
    return {"controlled_point": "right_grip_site", "unit": "metre",
            "reference_finger_qpos_m": [0., 0.], "fingers": fingers,
            "claim": "robot_geometry_not_contact_or_grasp_guarantee"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect_panda_gripper(args.asset)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()


def panda_arm_joint_limits(model) -> list[list[float]]:
    """Arm joint limits from the loaded model, gripper fingers excluded.

    This file already reads ``model.jnt_range``, but for the standalone gripper
    asset. Reachability needs the arm's: the seven revolute joints that precede
    the two finger slides.
    """

    ranges = model.jnt_range
    if len(ranges) < 7:
        raise ValueError("loaded Panda does not expose seven arm joint ranges")
    limits = [[float(low), float(high)] for low, high in ranges[:7]]
    if any(low >= high for low, high in limits):
        raise ValueError("loaded Panda reports a degenerate arm joint range")
    return limits


def panda_pose_at_joints(model, data, *, prefix: str, joints) -> list[float]:
    """Grip-site pose at a chosen arm configuration, without writing the plant.

    LIBERO published its end-effector pose only at the measured configuration,
    so "could the arm hold this pose" had no answer short of a CuRobo plan --
    and CuRobo has no inverse-kinematics-only mode. MuJoCo is already in this
    process and the pattern is the one ``panda_geometry_from_model`` uses for
    the fingers: copy MjData, set the joints, run kinematics, read the site.

    The live plant is never written, so asking a hypothetical cannot move the
    robot or disturb the observation the policy is reasoning about.
    """

    import mujoco
    import numpy as np

    values = [float(value) for value in joints]
    if len(values) != 7 or not all(np.isfinite(values)):
        raise ValueError("Panda forward kinematics requires seven finite arm joints")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    scratch.qpos[:7] = values
    mujoco.mj_kinematics(model, scratch)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "grip_site")
    if site < 0:
        raise ValueError("loaded Panda does not expose its declared grip site")
    position = np.asarray(scratch.site_xpos[site], dtype=np.float64)
    rotation = np.asarray(scratch.site_xmat[site], dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        axes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][index]
        i, j, k = axes
        scale = math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        w = (rotation[k, j] - rotation[j, k]) / scale
        values_xyz = [0.0, 0.0, 0.0]
        values_xyz[i] = 0.25 * scale
        values_xyz[j] = (rotation[j, i] + rotation[i, j]) / scale
        values_xyz[k] = (rotation[k, i] + rotation[i, k]) / scale
        x, y, z = values_xyz
    return [*(float(value) for value in position), float(x), float(y), float(z), float(w)]
