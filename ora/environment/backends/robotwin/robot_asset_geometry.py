"""Load only the pinned robot URDF and inspect cooked collision geometry.

No task import, renderer, simulation step, drive target or controller is used.
The explicit control profile supplies the public TCP convention, never a task.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def collect_aloha_grippers(urdf: Path, profile: dict, *, loaded_robots: dict | None = None) -> dict:
    import numpy as np
    import sapien

    expected = profile["source_binding"]["aloha_urdf_sha256"]
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("robot URDF differs from the actual controller binding")
    if loaded_robots is None:
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(urdf))
        if robot is None:
            raise ValueError("robot asset loader failed")
        loaded_robots = {"left": robot, "right": robot}
        if np.any(robot.get_qpos() != 0):
            raise ValueError("robot loader did not preserve the zero-joint reference")
    xml = ET.parse(urdf).getroot()
    geometries = {}
    sources = {}
    for arm, prefix in (("left", "fl"), ("right", "fr")):
        robot = loaded_robots[arm]
        links = robot.get_links()
        by_name = {link.name: index for index, link in enumerate(links)}
        pin = robot.create_pinocchio_model()
        # Scratch FK only: never set qpos or a drive target on the live entity.
        pin.compute_forward_kinematics(np.zeros_like(robot.get_qpos()))
        binding = profile["arms"][arm]
        link6 = pin.get_link_pose(by_name[binding["pinocchio_link_name"]]).to_transformation_matrix()
        rotation = link6[:3, :3] @ np.asarray(binding["pinocchio_to_sapien_link_post_rotation_matrix"]) @ np.asarray(binding["public_tcp_post_rotation_matrix"])
        origin = link6[:3, 3] + rotation @ np.asarray(binding["public_tcp_translation_local_m"])
        fingers = []
        intersections = []
        for index, number in enumerate((7, 8), 1):
            name = f"{prefix}_link{number}"
            link_id = by_name[name]
            transform = pin.get_link_pose(link_id).to_transformation_matrix()
            local_parts = []
            line_hits = []
            for shape in links[link_id].get_collision_shapes():
                vertices = np.asarray(shape.vertices) * np.asarray(shape.scale)
                frame = transform @ shape.local_pose.to_transformation_matrix()
                local = (vertices @ frame[:3, :3].T + frame[:3, 3] - origin) @ rotation
                local_parts.append(local)
                # Intersection of the cooked robot mesh with the local line
                # x=z=0, not an object/contact query or an inferred grasp width.
                for triangle in np.asarray(shape.triangles).reshape(-1, 3):
                    a, b, c = local[triangle]
                    matrix = np.array([[b[0]-a[0], c[0]-a[0]], [b[2]-a[2], c[2]-a[2]]])
                    if abs(np.linalg.det(matrix)) < 1e-14:
                        continue
                    u, v = np.linalg.solve(matrix, -a[[0, 2]])
                    if u >= -1e-7 and v >= -1e-7 and u+v <= 1+1e-7:
                        line_hits.append(float(a[1]+u*(b[1]-a[1])+v*(c[1]-a[1])))
            vertices = np.concatenate(local_parts)
            joint = xml.find(f"joint[@name='{prefix}_joint{number}']")
            axis = np.array([float(x) for x in joint.find("axis").get("xyz").split()])
            direction = (transform[:3, :3] @ axis) @ rotation
            fingers.append({"id": f"finger_{index}", "opening_direction_local": direction.round(7).tolist(),
                            "collision_bounds_m": {"minimum": vertices.min(0).round(10).tolist(), "maximum": vertices.max(0).round(10).tolist()}})
            if not line_hits:
                raise ValueError("robot finger mesh does not intersect the declared TCP-depth centerline")
            intersections.append([min(line_hits), max(line_hits)])
            mesh = xml.find(f"link[@name='{name}']/collision/geometry/mesh")
            mesh_path = (urdf.parent / mesh.get("filename")).resolve()
            sources[name] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        low, high = sorted(intersections)
        gap = high[0]-low[1]
        if gap < 0:
            raise ValueError("reference cooked finger geometry overlaps on the selected centerline")
        geometries[arm] = {"controlled_point": f"{arm}_tcp", "unit": "metre", "reference_finger_qpos_m": [0., 0.],
                           "fingers": fingers, "centerline_gap_at_tcp_m": round(gap, 10),
                           "claim": "robot_geometry_not_contact_or_grasp_guarantee"}
    return {"schema": "ora.robot-asset-geometry.v1", "backend": "robotwin",
            "scope": "cooked_robot_urdf_only_no_task_no_step", "simulator_version": importlib.metadata.version("sapien"),
            "asset_sha256": digest, "collision_mesh_sha256": sources,
            "control_profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "local_axes": {"approach": [1., 0., 0.], "closing": [0., 1., 0.]}, "geometry": geometries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("control_profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect_aloha_grippers(args.urdf, json.loads(args.control_profile.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
