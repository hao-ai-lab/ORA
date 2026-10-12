"""Public/planner pose frames, measured joints and the Pinocchio arm models.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .common import RoboTwinLiveError, _translate_pose_local_wxyz


class RoboTwinKinematicsMixin:
    """Public/planner pose frames, measured joints and the Pinocchio arm models."""

    def _public_pose_wxyz(self, environment: Any, arm: str) -> list[float]:
        if not self._uses_grip_center_profile():
            return [float(item) for item in environment.get_arm_pose(arm)]
        route = str(self._arm_profile(arm)["public_pose_route"])
        owner, method = route.split(".", 1)
        target = getattr(environment, owner)
        getter = getattr(target, method, None)
        if not callable(getter):
            raise RoboTwinLiveError(f"public TCP pose route is unavailable: {route}")
        return [float(item) for item in getter()]

    def _public_to_planner_wxyz(self, arm: str, public_pose: Any) -> list[float]:
        if not self._uses_grip_center_profile():
            return [float(item) for item in public_pose]
        return _translate_pose_local_wxyz(
            public_pose,
            self._arm_profile(arm)["public_to_planner_translation_local_m"],
        )

    def _measured_arm_joints(self, environment: Any, arm: str) -> list[float]:
        route = str(self._arm_profile(arm).get("measured_joint_route", ""))
        if not route:
            raise RoboTwinLiveError("measured joint hold route is unavailable")
        owner, method = route.split(".", 1)
        getter = getattr(getattr(environment, owner), method, None)
        if not callable(getter):
            raise RoboTwinLiveError(f"measured joint hold route is unavailable: {route}")
        values = [float(item) for item in getter()]
        joints = getattr(environment.robot, f"{arm}_arm_joints", None)
        count = len(joints) if isinstance(joints, (list, tuple)) else 6
        if len(values) < count or not all(math.isfinite(item) for item in values[:count]):
            raise RoboTwinLiveError(f"measured {arm} joint state is invalid")
        return values[:count]

    def _pinocchio_context(self, environment: Any, arm: str) -> Mapping[str, Any]:
        """Build one source-bound, non-mutating FK context for an ALOHA arm."""

        cached = self.pinocchio_models.get(arm)
        if isinstance(cached, Mapping):
            return cached
        import numpy as np

        entity = getattr(environment.robot, f"{arm}_entity", None)
        creator = getattr(entity, "create_pinocchio_model", None)
        if entity is None or not callable(creator):
            raise RoboTwinLiveError(f"Pinocchio FK is unavailable for {arm}")
        active_joints = list(entity.get_active_joints())
        active_names = [str(joint.get_name()) for joint in active_joints]
        links = list(entity.get_links())
        link_names = [str(link.get_name()) for link in links]
        arm_profile = self._arm_profile(arm)
        planner_names = list(arm_profile["planner_joint_names"])
        missing_joints = [name for name in planner_names if name not in active_names]
        required_links = [
            str(arm_profile["pinocchio_link_name"]),
            *[str(name) for name in arm_profile["pinocchio_chain_links"]],
        ]
        missing_links = [name for name in required_links if name not in link_names]
        if missing_joints or missing_links:
            raise RoboTwinLiveError(
                f"source-bound Pinocchio topology drifted for {arm}: "
                f"joints={missing_joints}, links={missing_links}"
            )
        model = creator()
        # SAPIEN's articulation wrapper has already projected both qpos and
        # link indices into articulation order. Reapplying user-facing source
        # names is invalid because the exported Pinocchio URDF uses synthetic
        # joint_N/link_N identifiers.
        joint_limits: list[list[float]] = []
        joint_by_name = dict(zip(active_names, active_joints))
        for name in planner_names:
            raw = np.asarray(joint_by_name[name].get_limits(), dtype=np.float64)
            if raw.size < 2:
                raise RoboTwinLiveError(f"joint limit is unavailable for {name}")
            joint_limits.append([float(raw.reshape(-1)[0]), float(raw.reshape(-1)[1])])
        context = {
            "entity": entity,
            "model": model,
            "active_joint_names": active_names,
            "planner_joint_names": planner_names,
            "planner_joint_indices": [active_names.index(name) for name in planner_names],
            "link_names": link_names,
            "tcp_link_index": link_names.index(str(arm_profile["pinocchio_link_name"])),
            "chain_link_indices": [
                link_names.index(str(name))
                for name in arm_profile["pinocchio_chain_links"]
            ],
            "joint_limits": joint_limits,
        }
        self.pinocchio_models[arm] = context
        return context

    @staticmethod
    def _entity_root_pose(entity: Any) -> Any:
        getter = getattr(entity, "get_pose", None)
        if callable(getter):
            return getter()
        pose = getattr(entity, "pose", None)
        if pose is None:
            raise RoboTwinLiveError("articulation root pose is unavailable")
        return pose

    def _pinocchio_world_link_poses(
        self, environment: Any, arm: str, arm_joints: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        import numpy as np

        context = self._pinocchio_context(environment, arm)
        entity = context["entity"]
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
        values = np.asarray(arm_joints, dtype=np.float64)
        if values.shape != (len(context["planner_joint_names"]),):
            raise RoboTwinLiveError(f"Pinocchio {arm} joint vector is invalid")
        for index, value in zip(context["planner_joint_indices"], values):
            qpos[int(index)] = float(value)
        model = context["model"]
        model.compute_forward_kinematics(qpos)
        root = self._entity_root_pose(entity)

        def world_pose(index: int) -> Any:
            return root * model.get_link_pose(index)

        tcp_link = world_pose(int(context["tcp_link_index"]))
        chain = {
            context["link_names"][index]: world_pose(int(index))
            for index in context["chain_link_indices"]
        }
        return tcp_link, chain

    def _pinocchio_public_pose_xyzw(
        self, environment: Any, arm: str, arm_joints: Any,
    ) -> list[float]:
        import numpy as np
        import transforms3d as t3d

        link_pose, _ = self._pinocchio_world_link_poses(
            environment, arm, arm_joints,
        )
        arm_profile = self._arm_profile(arm)
        rotation = t3d.quaternions.quat2mat(
            np.asarray(link_pose.q, dtype=np.float64)
        ) @ np.asarray(
            arm_profile["public_tcp_post_rotation_matrix"], dtype=np.float64,
        )
        position = np.asarray(link_pose.p, dtype=np.float64) + rotation @ np.asarray(
            arm_profile["public_tcp_translation_local_m"], dtype=np.float64,
        )
        w, qx, qy, qz = t3d.quaternions.mat2quat(rotation)
        return [
            float(position[0]), float(position[1]), float(position[2]),
            float(qx), float(qy), float(qz), float(w),
        ]

    def _pinocchio_link6_pose_xyzw(
        self, environment: Any, arm: str, arm_joints: Any,
    ) -> list[float]:
        import numpy as np
        import transforms3d as t3d

        link_pose, _ = self._pinocchio_world_link_poses(
            environment, arm, arm_joints,
        )
        arm_profile = self._arm_profile(arm)
        rotation = t3d.quaternions.quat2mat(
            np.asarray(link_pose.q, dtype=np.float64)
        ) @ np.asarray(
            arm_profile["pinocchio_to_sapien_link_post_rotation_matrix"],
            dtype=np.float64,
        )
        w, qx, qy, qz = t3d.quaternions.mat2quat(rotation)
        return [
            *[float(value) for value in link_pose.p],
            float(qx), float(qy), float(qz), float(w),
        ]

    def _arm_joint_limits(self, environment: Any, arm: str) -> list[list[float]]:
        return [
            [float(value) for value in interval]
            for interval in self._pinocchio_context(environment, arm)["joint_limits"]
        ]

    def _pinocchio_chain_points(
        self, environment: Any, arm: str, arm_joints: Any,
    ) -> list[list[float]]:
        _, chain = self._pinocchio_world_link_poses(environment, arm, arm_joints)
        return [
            [float(value) for value in pose.p]
            for pose in chain.values()
        ]
