"""Read only the two loaded ARX finger collision meshes and joint frames.

Reference q=0 is reconstructed from joint-local frames, never from the authored
world placement or the live finger opening. No scene objects, steps or writes.
"""
from __future__ import annotations

from typing import Any


def gripper_geometry(stage: Any, root: str, arm: str) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("ARX geometry requires a metre USD stage")
    fingers = []
    cache = UsdGeom.XformCache()
    for index, number in enumerate((7, 8), 1):
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"{root}/joints/joint{number}"))
        if not joint or joint.GetAxisAttr().Get() != "Y":
            raise ValueError("ARX finger joint topology differs")
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1 or str(body0[0]) != f"{root}/link6" or str(body1[0]) != f"{root}/link{number}":
            raise ValueError("ARX finger joint attachments differ")
        parent_pos = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
        child_pos = Gf.Vec3d(joint.GetLocalPos1Attr().Get())
        parent_rot = Gf.Quatd(joint.GetLocalRot0Attr().Get())
        child_rot_inv = Gf.Quatd(joint.GetLocalRot1Attr().Get()).GetInverse()
        body = stage.GetPrimAtPath(body1[0])
        points = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
                continue
            ancestor = prim
            collision = False
            while ancestor and ancestor != body:
                collision |= ancestor.HasAPI(UsdPhysics.CollisionAPI)
                ancestor = ancestor.GetParent()
            if not collision:
                continue
            relative = cache.ComputeRelativeTransform(prim, body)[0]
            for point in UsdGeom.Mesh(prim).GetPointsAttr().Get():
                local = relative.Transform(Gf.Vec3d(point))
                points.append(parent_pos + parent_rot.Transform(child_rot_inv.Transform(local-child_pos)))
        if not points:
            raise ValueError("ARX loaded finger collision mesh is absent")
        direction = parent_rot.Transform(Gf.Vec3d(0, 1, 0))
        fingers.append({"id": f"finger_{index}",
                        "opening_direction_local": [round(x, 10) for x in direction],
                        "collision_bounds_m": {
                            "minimum": [round(min(p[i] for p in points), 10) for i in range(3)],
                            "maximum": [round(max(p[i] for p in points), 10) for i in range(3)]}})
    return {"controlled_point": f"{arm}_link6", "unit": "metre",
            "reference_finger_qpos_m": [0, 0], "fingers": fingers,
            "claim": "robot_geometry_not_contact_or_grasp_guarantee"}


def loaded_geometry(session: Any) -> dict[str, Any]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    manager = session.environment.robot_manager
    result = {}
    for arm, robot in session.robots.items():
        articulation = manager.robot_key[manager.robot_list.index(robot)]
        paths = articulation.root_physx_view.prim_paths
        if len(paths) != 1:
            raise ValueError("ARX geometry qualification requires exactly one environment")
        result[f"{arm}_gripper"] = gripper_geometry(stage, str(paths[0]), arm)
    return result
