"""Read-only reachability probes and execution of a probed plan reference.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

from ora.environment.controllers.curobo_motion import (
    _curobo_failure_reason,
    _curobo_valid_query,
)
from ora.contracts.workspace import workspace_state

from .pinned_imports import _path_is_within
from .common import (
    LIBERO_SPARSE_RECEIPT_SCHEMA,
    _sparse_planning_failure,
    LiberoProError,
    _file_hash,
    _quaternion_error_rotvec,
)


class LiberoMotionProbeMixin:
    """Read-only reachability probes and execution of a probed plan reference."""

    def _plan_source_identity(self) -> str:
        """Bind references to the exact robot/planner/profile source, not a PID."""

        planner_binding = dict(getattr(self.motion_planner, "binding", {}))
        planner_binding.pop("worker", None)
        body = {
            "source_manifest": getattr(self, "source_manifest", {}),
            "planner_binding": planner_binding,
            "motion_profile": self.motion_profile,
            "qualified_controller_binding": self.qualified_controller_binding,
        }
        return hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Plan reachability queries without writing the simulated world.

        Every query plans through the same qualified CuRobo
        path the executor uses, from one current measured snapshot. Admitted
        plans are stored under an immutable plan_ref that move_pose can
        execute directly; parity freshness lets a plan survive
        gripper-only world writes and expire on anything else. The synthetic
        planner summary carries honest nulls for fields the CuRobo client
        does not report.
        """

        import hashlib as _hashlib
        import math as _math

        import numpy as np

        from ora.environment.controllers.motion_transaction import (
            write_motion_plan_artifact,
        )

        if self.motion_planner is None:
            raise LiberoProError("CuRobo planner is not active")
        if set(request) != {
            "schema", "observation_id", "observation_sha256", "queries",
            "variant_policy",
        } or request.get("schema") != "ora.motion-probe-request.v1":
            raise LiberoProError("motion probe request fields drifted")
        observation_id = request.get("observation_id")
        observation_sha256 = request.get("observation_sha256")
        queries = request.get("queries")
        variant_policy = request.get("variant_policy")
        if (
            not isinstance(observation_id, str) or not observation_id
            or not isinstance(observation_sha256, str) or not observation_sha256
            or not isinstance(queries, list) or not 1 <= len(queries) <= 12
            or variant_policy not in {"exact", "parallel_jaw_equivalents"}
        ):
            raise LiberoProError("motion probe request is invalid")
        motion_state = self._motion_state()
        measured_joints = [
            float(value) for value in motion_state["joint_positions"]
        ]

        def planner_summary(planned: Mapping[str, Any] | None) -> dict[str, Any]:
            status = (
                str(planned.get("planner_status") or planned.get("status"))
                if isinstance(planned, Mapping) else "not_called"
            )
            trajectory = (
                planned.get("trajectory") if isinstance(planned, Mapping)
                else None
            )
            steps = (
                len(trajectory["positions"])
                if isinstance(trajectory, Mapping) else 0
            )
            dt = (
                float(trajectory["dt_s"])
                if isinstance(trajectory, Mapping) else None
            )
            return {
                "called": isinstance(planned, Mapping),
                "status": status,
                "stage": (
                    "SUCCESS"
                    if isinstance(planned, Mapping) and planned.get("reachable")
                    else "FAILED"
                ),
                "valid_query": (
                    _curobo_valid_query(planned) if planned is not None else None
                ),
                "attempts": None,
                "trajopt_attempts": None,
                "used_graph": None,
                "goalset_index": None,
                "path_steps": steps,
                "interpolation_dt_s": dt,
                "errors": {"position": None, "rotation": None, "cspace": None},
                "timings_s": {
                    "solve": None, "ik": None, "graph": None,
                    "trajopt": None, "finetune": None,
                    "total": (
                        float(planned.get("planning_time_s"))
                        if isinstance(planned, Mapping)
                        and isinstance(
                            planned.get("planning_time_s"), (int, float),
                        ) else None
                    ),
                },
                "error": (
                    {"stage": "planner", **dict(planned["error"])}
                    if isinstance(planned, Mapping)
                    and isinstance(planned.get("error"), Mapping) else None
                ),
            }

        results: list[dict[str, Any]] = []
        query_ids: set[str] = set()
        for query in queries:
            if not isinstance(query, Mapping) or set(query) != {
                "query_id", "arm", "pose",
            }:
                raise LiberoProError("motion probe query fields drifted")
            query_id = query.get("query_id")
            arm = query.get("arm")
            pose = query.get("pose")
            if (
                not isinstance(query_id, str) or not query_id
                or query_id in query_ids or arm != "right"
                or not isinstance(pose, Mapping)
                or set(pose) != {"position_m", "orientation_xyzw"}
            ):
                raise LiberoProError("motion probe query is invalid")
            query_ids.add(query_id)
            position = [float(value) for value in pose["position_m"]]
            orientation = [float(value) for value in pose["orientation_xyzw"]]
            if (
                len(position) != 3 or len(orientation) != 4
                or not all(
                    _math.isfinite(value)
                    for value in [*position, *orientation]
                )
            ):
                raise LiberoProError("motion probe pose is invalid")
            if not workspace_state(position, self.workspace_bounds_m)["inside"]:
                raise LiberoProError(
                    "motion probe target is outside the action workspace"
                )
            variant_poses = [("variant-00-exact-+0deg", orientation)]
            if variant_policy == "parallel_jaw_equivalents":
                # 180-degree jaw symmetry about the Panda approach axis
                # (grip-site local +Z): q ⊗ quat(z, pi), xyzw Hamilton
                # product with [0, 0, 1, 0] gives [y, -x, w, -z].
                x, y, z, w = orientation
                variant_poses.append(
                    ("variant-01-jaw-flip-180deg", [y, -x, w, -z])
                )
            variants: list[dict[str, Any]] = []
            admitted: list[dict[str, Any]] = []
            for variant_id, variant_orientation in variant_poses:
                variant_pose = position + list(variant_orientation)
                target_base, _start = self._grip_target_planner_pose(
                    motion_state, variant_pose,
                )
                planned = self.motion_planner.plan({
                    "joint_names": motion_state["joint_names"],
                    "joint_positions": motion_state["joint_positions"],
                    "target_pose_base_xyzw": target_base,
                })
                self._persist_planner_evidence(planned, probe=True)
                summary = planner_summary(planned)
                trajectory = planned.get("trajectory")
                if not planned.get("reachable") or not isinstance(
                    trajectory, Mapping,
                ):
                    variants.append({
                        "variant_id": variant_id, "reachable": False,
                        "planner": summary,
                        "failure": {
                            "code": _curobo_failure_reason(planned),
                            "message": "CuRobo admitted no trajectory",
                        },
                    })
                    continue
                self._validate_planned_start(planned, motion_state)
                stored_trajectory = {
                    "positions": trajectory["positions"],
                    "joint_names": trajectory["joint_names"],
                    "dt_s": trajectory["dt_s"],
                }
                reference_body = {
                    "schema": "ora.libero-plan-reference.v2",
                    "observation_id": observation_id,
                    "observation_sha256": observation_sha256,
                    "world_mutation_sequence": self.world_mutation_sequence,
                    "arm": "right",
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "target_pose_xyzw": variant_pose,
                    "measured_start_joints": measured_joints,
                    "planner_start_fk": list(planned["start_ee_pose_base_xyzw"]),
                    "source_identity": self._plan_source_identity(),
                    "round_index": self.round_index,
                    "trajectory_sha256": hashlib.sha256(json.dumps(
                        stored_trajectory, sort_keys=True, separators=(",", ":"),
                    ).encode()).hexdigest(),
                }
                plan_ref = "plan-" + _hashlib.sha256(json.dumps(
                    reference_body, sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest()[:32]
                artifact = write_motion_plan_artifact(
                    self.output_root / "motion-probes",
                    transaction_id=(
                        f"probe-{self.motion_probe_sequence:04d}-right-"
                        f"{len(variants):03d}"
                    ),
                    backend_id="libero_pro",
                    embodiment_id="franka_panda",
                    binding=reference_body,
                    validation={"status": "kinematics_only"},
                    arrays={
                        "planned_right_joint__position": [
                            [float(v) for v in row]
                            for row in trajectory["positions"]
                        ],
                    },
                )
                # The artifact writer records paths relative to its own
                # root; evaluation resolves against the run native root, so
                # the stored reference re-bases onto motion-probes/.
                artifact = {
                    **artifact,
                    "path": f"motion-probes/{artifact['path']}",
                }
                self.plan_references[plan_ref] = {
                    **reference_body,
                    "plan_ref": plan_ref,
                    "trajectory": stored_trajectory,
                    "artifact": artifact,
                }
                steps = len(trajectory["positions"])
                dt = trajectory.get("dt_s")
                entry = {
                    "variant_id": variant_id, "reachable": True,
                    "target_pose_xyzw": variant_pose,
                    "planner": summary,
                    "endpoint_error": {
                        "translation_m": None, "orientation_rad": None,
                    },
                    "trajectory_steps": steps,
                    "trajectory_duration_s": (
                        round(steps * float(dt), 8)
                        if isinstance(dt, (int, float)) and float(dt) > 0
                        else 0.0
                    ),
                    "plan_ref": plan_ref,
                }
                variants.append(entry)
                admitted.append(entry)
                self.motion_probe_sequence += 1
            admitted.sort(key=lambda row: (
                row["trajectory_duration_s"] == 0.0,
                row["trajectory_duration_s"],
                row["trajectory_steps"],
            ))
            selected = admitted[0] if admitted else None
            results.append({
                "query_id": query_id, "arm": "right",
                "reachable": bool(admitted),
                "selected_variant_id": (
                    selected["variant_id"] if selected else None
                ),
                "plan_ref": selected["plan_ref"] if selected else None,
                "selected_target_pose_xyzw": (
                    selected["target_pose_xyzw"] if selected else None
                ),
                "variants": variants,
            })
        return {
            "schema": "ora.motion-probe-result.v1",
            "observation_id": observation_id,
            "world_mutation_sequence": self.world_mutation_sequence,
            "variant_policy": variant_policy,
            "robot_decisions_consumed": 0,
            "world_state_changed": False,
            "results": results,
        }

    def _execute_plan_reference(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one stored probed plan through the qualified executor."""

        import numpy as np

        plan_ref = str(action["arguments"]["plan_ref"])
        stored = self.plan_references.get(plan_ref)
        if not isinstance(stored, Mapping):
            raise LiberoProError("plan_ref is unknown or tampered")
        if (
            stored.get("schema") != "ora.libero-plan-reference.v2"
            or stored.get("round_index") != self.round_index
            or stored.get("source_identity") != self._plan_source_identity()
        ):
            raise LiberoProError("plan_ref source or round identity drifted")
        if hashlib.sha256(json.dumps(
            stored["trajectory"], sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest() != stored.get("trajectory_sha256"):
            raise LiberoProError("plan_ref trajectory identity drifted")
        recorded_sequence = int(stored["world_mutation_sequence"])
        same_snapshot = (
            recorded_sequence == self.world_mutation_sequence
            and stored["observation_id"] == action["observation_id"]
            and stored["observation_sha256"]
            == action["bound_observation_sha256"]
        )
        intervening = self.world_mutation_actions[recorded_sequence:]
        gripper_only = bool(intervening) and all(
            kind == "set_gripper" for kind in intervening
        )
        if not (same_snapshot or gripper_only):
            raise LiberoProError("plan_ref is stale for the current world state")
        artifact = stored.get("artifact")
        artifact_path = (
            self.output_root / str(artifact.get("path", ""))
        ).resolve() if isinstance(artifact, Mapping) else self.output_root
        if (
            not _path_is_within(artifact_path, self.output_root)
            or not artifact_path.is_file()
            or _file_hash(artifact_path) != artifact.get("sha256")
        ):
            raise LiberoProError("plan_ref artifact identity drifted")
        current_joints = [
            float(value) for value in self._motion_state()["joint_positions"]
        ]
        if len(current_joints) != 7 or len(stored["measured_start_joints"]) != 7 or max(
            abs(a - b) for a, b in zip(
                current_joints, stored["measured_start_joints"], strict=True,
            )
        ) > 0.02:
            raise LiberoProError("plan_ref start joints no longer match")

        target = stored["target_pose_xyzw"]
        target_position = np.asarray(target[:3], dtype=np.float64)
        before_position, before_orientation, _gripper = self._current_state()

        def wrap(inner: dict[str, Any]) -> dict[str, Any]:
            motion_status = str(inner.get("motion_status"))
            feedback = inner.get("feedback")
            feedback = feedback if isinstance(feedback, Mapping) else {}
            if motion_status == "refused":
                status, stage, code = _sparse_planning_failure(
                    str(feedback.get("planning_failure_reason"))
                )
            elif motion_status == "reached":
                status, stage, code = "reached", None, None
            elif motion_status == "stopped":
                status, stage, code = (
                    "not_reached", "execution", feedback.get("stop_reason"),
                )
            else:
                status, stage, code = (
                    "not_reached", "endpoint_verification",
                    "endpoint_residual_exceeded",
                )
            after = inner.get("after", {}).get("right", {})
            after_pose = after.get("pose_xyz_xyzw")
            pose_error = None
            if isinstance(after_pose, list) and len(after_pose) == 7:
                pose_error = {
                    "translation_m": float(np.linalg.norm(
                        target_position
                        - np.asarray(after_pose[:3], dtype=np.float64)
                    )),
                    "orientation_rad": float(np.linalg.norm(
                        _quaternion_error_rotvec(
                            np.asarray(target[3:], dtype=np.float64),
                            np.asarray(after_pose[3:], dtype=np.float64),
                        )
                    )),
                }
            # The applied endpoint bound is compiled by the executing route and
            # only lifted here. This projection maps every non-refused,
            # non-reached, non-stopped result to endpoint_residual_exceeded
            # without re-comparing, so it must never re-derive a bound of its
            # own: an absent contract stays absent.
            endpoint_contract = inner.get("endpoint_contract")
            endpoint_contract = (
                dict(endpoint_contract)
                if isinstance(endpoint_contract, Mapping) else None
            )
            return {
                "schema": LIBERO_SPARSE_RECEIPT_SCHEMA,
                "action_id": action["action_id"],
                "action": "move_pose",
                "arguments": dict(action["arguments"]),
                "consumed": True,
                "state_changed": bool(inner.get("state_changed")),
                "execution_status": status,
                "motion_strategy": "global",
                "failure_stage": stage,
                "failure_code": code,
                "executor": ["libero_curobo_joint_position"],
                "target_arms": ["right"],
                "synchronization": {"mode": "single_arm"},
                "arms": {"right": {
                    "targeted": True,
                    "interaction": "free_space",
                    "pose_error": pose_error,
                    "endpoint_contract": endpoint_contract,
                    "plan_validation": {"status": "kinematics_only"},
                    "plan_ref": plan_ref,
                }},
                "before": inner.get("before"),
                "after": inner.get("after"),
                "steps_used": inner.get("steps_used"),
                "sim_steps_total": inner.get("sim_steps_total"),
                "control_health": inner.get("control_health"),
                "terminal_after_action": bool(
                    self.terminated or self.truncated
                ),
                "segment": inner,
                "finished_unix_s": time.time(),
            }

        previous_transform = getattr(self, "_sparse_receipt_transform", None)
        self._sparse_receipt_transform = wrap
        try:
            return self._execute_planned_motion(
                action,
                preplanned=(
                    {
                        "reachable": True,
                        "trajectory": dict(stored["trajectory"]),
                        "start_ee_pose_base_xyzw": list(stored["planner_start_fk"]),
                        "target_pose_world_xyzw": list(stored["target_pose_xyzw"]),
                    },
                    {
                        "source": "motion_probe_plan_ref",
                        "plan_ref": plan_ref,
                        "artifact": dict(stored["artifact"]),
                    },
                ),
            )
        finally:
            self._sparse_receipt_transform = previous_transform
