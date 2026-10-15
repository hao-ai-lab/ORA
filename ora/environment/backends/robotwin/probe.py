"""Read-only reachability planning and motion probes.

A mixin of ``session.RoboTwinLiveSession``; it holds methods only, no
dataclass fields. ``probe_motion`` validates the request, samples the
action-start snapshot, expands each query into its orientation variants, plans
each arm's variants in one batch and registers a plan_ref for every variant
that validates; the module-level helpers build the per-variant rows and the
per-query selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ora.contracts.pose import canonicalize_quaternion
from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected,
    ValidatedRoboTwinPlan,
    planner_public_summary,
    select_topk_validated_plan,
    split_batch_plan_result,
    validate_inter_arm_clearance,
    validate_single_arm_plan,
)
from ora.environment.controllers.motion_transaction import (
    write_motion_plan_artifact,
)
from ora.environment.controllers.profile_capabilities import (
    IK_CONSTRAINT_FEEDBACK_SINCE,
    profile_has,
)

from .common import (
    RoboTwinLiveError,
    _json_hash,
    _validate_workspace_target,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
    probe_variant_sort_key,
)
from .probe_feedback import probe_request_feedback, with_ik_candidate_feedback


class RoboTwinProbeMixin:
    """Read-only reachability planning and motion probes."""

    @staticmethod
    def _orientation_variants_xyzw(
        orientation_xyzw: Any, *, policy: str,
    ) -> list[tuple[str, list[float]]]:
        qx, qy, qz, qw = canonicalize_quaternion(
            orientation_xyzw, "orientation_xyzw",
        )
        degrees = [0.0] if policy == "exact" else [0.0, -10.0, 10.0, 180.0, 170.0, 190.0]
        variants: list[tuple[str, list[float]]] = []
        for index, angle_deg in enumerate(degrees):
            half = math.radians(angle_deg) / 2.0
            dx, dy, dz, dw = math.sin(half), 0.0, 0.0, math.cos(half)
            orientation = canonicalize_quaternion([
                qw * dx + qx * dw + qy * dz - qz * dy,
                qw * dy - qx * dz + qy * dw + qz * dx,
                qw * dz + qx * dy - qy * dx + qz * dw,
                qw * dw - qx * dx - qy * dy - qz * dz,
            ], "variant_orientation_xyzw")
            variants.append((
                f"variant-{index:02d}-approach-x-{angle_deg:+g}deg",
                orientation,
            ))
        return variants

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Delegate read-only alignment planning to its own module."""

        from ora.contracts.alignment import AlignmentRequestDrift
        from ora.environment.controllers.alignment import plan_alignment

        try:
            return plan_alignment(
                request,
                adapter=self._motion_backend_adapter(
                    session=self, environment=self._require_environment(),
                ),
                control_profile=self.control_profile,
                available_arms=("left", "right"),
            )
        except AlignmentRequestDrift as error:
            raise RoboTwinLiveError(str(error)) from error

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Plan from the current snapshot without writing the simulated world."""

        environment = self._require_environment()
        if not self._uses_validated_planner_profile():
            raise RoboTwinLiveError("motion probing requires control profile v4")
        observation_id, observation_sha256, variant_policy, normalized = (
            self._normalized_motion_probe_request(request)
        )
        ik_feedback = probe_request_feedback(request)
        probe = self._motion_probe_start(
            environment,
            observation_id=observation_id,
            observation_sha256=observation_sha256,
        )
        expanded = self._expanded_motion_probe_variants(
            probe, normalized, variant_policy,
        )
        candidates: dict[str, list[dict[str, Any]]] = {
            query["query_id"]: [] for query in normalized
        }
        probe.validation_profile = self.control_profile["profile"]["plan_validation"]  # type: ignore[index]
        probe.include_ik_constraints = profile_has(
            self.control_profile, since_version=IK_CONSTRAINT_FEEDBACK_SINCE,
        )
        if ik_feedback and not probe.include_ik_constraints:
            raise RoboTwinLiveError(
                "motion probe IK candidate feedback requires control profile v10"
            )
        probe.ik_feedback = ik_feedback
        for arm in ("left", "right"):
            variants = expanded[arm]
            if not variants:
                continue
            raw_batch = self._plan_motion_probe_batch(probe, arm, variants)
            for index, variant in enumerate(variants):
                self._probe_motion_variant(
                    probe, arm, index, variant, raw_batch, candidates,
                )
        results = _motion_probe_results(normalized, candidates)
        return {
            "schema": "ora.motion-probe-result.v1",
            "observation_id": observation_id,
            "world_mutation_sequence": self.world_mutation_sequence,
            "variant_policy": variant_policy,
            "robot_decisions_consumed": 0,
            "world_state_changed": False,
            "results": results,
        }

    def _normalized_motion_probe_request(
        self, request: Mapping[str, Any],
    ) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
        """Validate the probe request; return its identity, policy and queries."""

        try:
            probe_request_feedback(request)  # v1, or v2 naming its feedback
        except ValueError as exc:
            raise RoboTwinLiveError(str(exc)) from exc
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
            raise RoboTwinLiveError("motion probe request is invalid")
        normalized: list[dict[str, Any]] = []
        query_ids: set[str] = set()
        for query in queries:
            if not isinstance(query, Mapping) or set(query) != {"query_id", "arm", "pose"}:
                raise RoboTwinLiveError("motion probe query fields drifted")
            query_id = query.get("query_id")
            arm = query.get("arm")
            pose = query.get("pose")
            if (
                not isinstance(query_id, str) or not query_id or query_id in query_ids
                or arm not in {"left", "right"}
                or not isinstance(pose, Mapping)
                or set(pose) != {"position_m", "orientation_xyzw"}
            ):
                raise RoboTwinLiveError("motion probe query is invalid")
            position = [float(value) for value in pose["position_m"]]
            orientation = [float(value) for value in pose["orientation_xyzw"]]
            if (
                len(position) != 3 or len(orientation) != 4
                or not all(math.isfinite(value) for value in [*position, *orientation])
            ):
                raise RoboTwinLiveError("motion probe pose is invalid")
            _validate_workspace_target(position, self.workspace_bounds_m,
                                       argument_field=f"queries.{query_id}.pose.position_m")
            norm = math.sqrt(sum(value * value for value in orientation))
            if not 0.98 <= norm <= 1.02:
                raise RoboTwinLiveError("motion probe orientation is not normalized")
            orientation = [value / norm for value in orientation]
            query_ids.add(query_id)
            normalized.append({
                "query_id": query_id, "arm": arm,
                "pose": {"position_m": position, "orientation_xyzw": orientation},
            })
        return observation_id, observation_sha256, variant_policy, normalized

    def _motion_probe_start(
        self, environment: Any, *, observation_id: str, observation_sha256: str,
    ) -> _MotionProbe:
        """Sample both arms and capture the action-start snapshot to plan from."""

        measured = {
            arm: self._measured_arm_joints(environment, arm)
            for arm in ("left", "right")
        }
        measured_pose = {
            arm: _wxyz_to_xyzw(self._public_pose_wxyz(environment, arm))
            for arm in ("left", "right")
        }
        motion_adapter = (
            self._motion_backend_adapter(session=self, environment=environment)
            if self._uses_motion_transaction_profile() else None
        )
        motion_action_start = (
            motion_adapter.capture_action_start()
            if motion_adapter is not None else None
        )
        return _MotionProbe(
            environment=environment,
            observation_id=observation_id,
            observation_sha256=observation_sha256,
            measured=measured,
            measured_pose=measured_pose,
            motion_adapter=motion_adapter,
            motion_action_start=motion_action_start,
        )

    def _expanded_motion_probe_variants(
        self,
        probe: _MotionProbe,
        normalized: list[dict[str, Any]],
        variant_policy: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Expand every query into its orientation variants, grouped by arm."""

        motion_adapter = probe.motion_adapter
        expanded: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        for query in normalized:
            for variant_id, orientation in self._orientation_variants_xyzw(
                query["pose"]["orientation_xyzw"], policy=variant_policy,
            ):
                public_wxyz = _xyzw_to_wxyz(
                    query["pose"]["position_m"], orientation,
                )
                public_xyzw = [
                    *public_wxyz[:3], *public_wxyz[4:], public_wxyz[3],
                ]
                expanded[query["arm"]].append({
                    "query_id": query["query_id"],
                    "variant_id": variant_id,
                    "public_wxyz": public_wxyz,
                    "public_xyzw": public_xyzw,
                    "planner_wxyz": (
                        motion_adapter.compile_public_target(
                            query["arm"], public_xyzw,
                        )
                        if motion_adapter is not None
                        else self._public_to_planner_wxyz(
                            query["arm"], public_wxyz,
                        )
                    ),
                })
        return expanded

    def _plan_motion_probe_batch(
        self, probe: _MotionProbe, arm: str, variants: list[dict[str, Any]],
    ) -> Any:
        """Plan every variant of one arm in a single planner batch."""

        if probe.motion_adapter is not None:
            return probe.motion_adapter.plan_batch_global(
                arm,
                [variant["planner_wxyz"] for variant in variants],
                probe.motion_action_start,
            )
        planner = getattr(probe.environment.robot, f"{arm}_plan_multi_path")
        return planner([
            variant["planner_wxyz"] for variant in variants
        ])

    def _probe_motion_variant(
        self,
        probe: _MotionProbe,
        arm: str,
        index: int,
        variant: Mapping[str, Any],
        raw_batch: Any,
        candidates: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Validate one planned variant; a reachable one gets a plan_ref."""

        raw: Mapping[str, Any] | None = None
        try:
            if (
                isinstance(raw_batch, Mapping)
                and raw_batch.get("schema")
                == "robotwin.curobo-topk-batch-result.v1"
            ):
                raw = raw_batch["results"][index]
            else:
                raw = split_batch_plan_result(raw_batch, index)
            self._record_ik_candidate_evidence(raw, arm=arm, identity={
                "kind": "probe", "observation_id": probe.observation_id,
                "query_id": variant["query_id"], "variant_id": variant["variant_id"],
            })
            raw, validated, clearance = self._select_motion_probe_candidate(
                probe, arm, variant, raw,
            )
        except RoboTwinPlanRejected as exc:
            candidates[variant["query_id"]].append(_rejected_motion_probe_row(
                variant, raw, exc,
                include_ik_constraints=probe.include_ik_constraints,
                ik_feedback=probe.ik_feedback,
            ))
            return
        plan_ref = self._register_motion_probe_plan(
            probe, arm, index, variant, raw, validated, clearance,
        )
        candidates[variant["query_id"]].append(_reachable_motion_probe_row(
            variant, raw, validated, plan_ref,
            include_ik_constraints=probe.include_ik_constraints,
            ik_feedback=probe.ik_feedback,
        ))
        self.motion_probe_sequence += 1

    def _select_motion_probe_candidate(
        self,
        probe: _MotionProbe,
        arm: str,
        variant: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> tuple[Any, ValidatedRoboTwinPlan, dict[str, Any]]:
        """Select the first top-k candidate that validates and clears the peer arm."""

        environment = probe.environment
        measured = probe.measured
        measured_pose = probe.measured_pose
        motion_adapter = probe.motion_adapter
        motion_action_start = probe.motion_action_start
        validation_profile = probe.validation_profile

        def validate_probe_candidate(
            candidate: Mapping[str, Any], *, selected_arm: str = arm,
        ) -> ValidatedRoboTwinPlan:
            validated = validate_single_arm_plan(
                candidate,
                expected_joint_names=self._arm_profile(selected_arm)[
                    "planner_joint_names"
                ],
                measured_start_joints=measured[selected_arm],
                measured_start_pose_xyzw=measured_pose[selected_arm],
                target_pose_xyzw=variant["public_xyzw"],
                joint_limits=(
                    motion_adapter.joint_limits(selected_arm)
                    if motion_adapter is not None
                    else self._arm_joint_limits(environment, selected_arm)
                ),
                fk_public_pose=(
                    (
                        lambda joints: motion_adapter.fk_public_tcp_pose(
                            selected_arm, joints,
                        )
                    )
                    if motion_adapter is not None
                    else (
                        lambda joints: self._pinocchio_public_pose_xyzw(
                            environment, selected_arm, joints,
                        )
                    )
                ),
                fk_link6_pose=(
                    (
                        lambda joints: motion_adapter.fk_control_pose(
                            selected_arm, joints,
                        )
                    )
                    if motion_adapter is not None
                    else (
                        lambda joints: self._pinocchio_link6_pose_xyzw(
                            environment, selected_arm, joints,
                        )
                    )
                ),
                workspace_bounds_m=(
                    motion_adapter.workspace_bounds_m()
                    if motion_adapter is not None
                    else self.workspace_bounds_m
                ),
                profile=validation_profile,
            )

            clearance = (
                motion_adapter.validate_collision(
                    {arm: validated}, motion_action_start,
                )
                if motion_adapter is not None
                else validate_inter_arm_clearance(
                    {arm: validated},
                    measured_joints=measured,
                    fk_chain_points=lambda selected, joints: (
                        self._pinocchio_chain_points(
                            environment, selected, joints,
                        )
                    ),
                    link_names={arm: spec["pinocchio_chain_links"] for arm, spec in self.control_profile["profile"]["arms"].items()},
                    profile=self.control_profile["profile"][  # type: ignore[index]
                        "dual_arm_collision_check"
                    ],
                )
            )
            candidate_clearance[id(validated)] = clearance
            return validated

        candidate_clearance: dict[int, dict[str, Any]] = {}
        raw, validated, _selection = select_topk_validated_plan(
            raw, validator=validate_probe_candidate,
        )
        clearance = candidate_clearance[id(validated)]
        return raw, validated, clearance

    def _register_motion_probe_plan(
        self,
        probe: _MotionProbe,
        arm: str,
        index: int,
        variant: Mapping[str, Any],
        raw: Any,
        validated: ValidatedRoboTwinPlan,
        clearance: Mapping[str, Any],
    ) -> str:
        """Write the probe's plan artifact and register its session plan_ref."""

        observation_id = probe.observation_id
        observation_sha256 = probe.observation_sha256
        measured = probe.measured
        binding = {
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "world_mutation_sequence": self.world_mutation_sequence,
            "query_id": variant["query_id"],
            "variant_id": variant["variant_id"],
            "target_pose_xyzw": variant["public_xyzw"],
            "collision_validation": clearance,
        }
        artifact = write_motion_plan_artifact(
            self.output_root / "motion-probes",
            transaction_id=(
                f"probe-{self.motion_probe_sequence:04d}-{arm}-{index:03d}"
            ),
            backend_id="robotwin",
            embodiment_id="aloha_agilex",
            binding={
                **binding,
                "source_revision": self.source_manifest[
                    "robotwin_revision"
                ],
                "control_profile_sha256": self.control_profile["sha256"],
                "measured_start_joints": measured[arm],
            },
            validation=validated.summary,
            arrays={
                f"planned_{arm}_joint__position": validated.position,
                f"planned_{arm}_joint__velocity": validated.velocity,
                f"planned_{arm}_control__pose_xyzw": (
                    validated.planned_link6_pose_xyzw
                ),
                f"planned_{arm}_public_tcp__pose_xyzw": (
                    validated.planned_tcp_pose_xyzw
                ),
            },
        )
        reference_body = {
            "schema": "ora.motion-plan-reference.v1",
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "world_mutation_sequence": self.world_mutation_sequence,
            "robotwin_revision": self.source_manifest["robotwin_revision"],
            "control_profile_sha256": self.control_profile["sha256"],
            "arm": arm,
            "query_id": variant["query_id"],
            "variant_id": variant["variant_id"],
            "target_pose_xyzw": variant["public_xyzw"],
            "measured_start_joints": measured[arm],
            "trajectory_sha256": artifact["artifact_sha256"],
            "backend_id": "robotwin",
            "embodiment_id": "aloha_agilex",
            "action_start_snapshot_sha256": _json_hash({
                "world_mutation_sequence": self.world_mutation_sequence,
                "measured_joints": measured,
                "measured_public_tcp_pose_xyzw": probe.measured_pose,
            }),
        }
        plan_ref = "plan-" + _json_hash(reference_body)[:32]
        self.plan_references[plan_ref] = {
            **reference_body,
            "plan_ref": plan_ref,
            "validated": validated,
            "planner_result": raw,
            "collision_validation": clearance,
            "artifact": artifact,
            # Session-local only (not part of the hashed reference
            # body): both-arm start joints for the gripper-only
            # carve-out freshness check.
            "measured_start_joints_by_arm": {
                str(key): [float(value) for value in joints]
                for key, joints in measured.items()
            },
        }
        return plan_ref


@dataclass
class _MotionProbe:
    """One probe call's observation identity and action-start snapshot."""

    environment: Any
    observation_id: str
    observation_sha256: str
    measured: dict[str, Any]
    measured_pose: dict[str, Any]
    motion_adapter: Any
    motion_action_start: Any
    validation_profile: Any = None
    include_ik_constraints: bool = False
    ik_feedback: bool = False


def _rejected_motion_probe_row(
    variant: Mapping[str, Any],
    raw: Mapping[str, Any] | None,
    exc: RoboTwinPlanRejected,
    *,
    include_ik_constraints: bool,
    ik_feedback: bool = False,
) -> dict[str, Any]:
    planner = planner_public_summary(
        raw, include_topk_selection=False,
        include_constraint_rejection_counts=include_ik_constraints,
    )
    return {
        "variant_id": variant["variant_id"],
        "reachable": False,
        "planner": (
            with_ik_candidate_feedback(planner, raw, admitted=False)
            if ik_feedback else planner
        ),
        "failure": {"code": exc.code, "message": str(exc)},
    }


def _reachable_motion_probe_row(
    variant: Mapping[str, Any],
    raw: Mapping[str, Any],
    validated: ValidatedRoboTwinPlan,
    plan_ref: str,
    *,
    include_ik_constraints: bool,
    ik_feedback: bool = False,
) -> dict[str, Any]:
    planner = planner_public_summary(
        raw, include_topk_selection=False,
        include_constraint_rejection_counts=include_ik_constraints,
    )
    return {
        "variant_id": variant["variant_id"],
        "reachable": True,
        "target_pose_xyzw": variant["public_xyzw"],
        "planner": (
            with_ik_candidate_feedback(planner, raw, admitted=True)
            if ik_feedback else planner
        ),
        "endpoint_error": validated.summary["planning_endpoint_error"],
        "trajectory_steps": int(validated.position.shape[0]),
        # A missing dt is reported as 0.0 to honor the published
        # result schema (number >= 0); the selection sort treats
        # 0.0 as "untimed" and ranks it behind every timed plan,
        # so a missing dt can never win as a zero-duration plan.
        "trajectory_duration_s": (
            round(
                int(validated.position.shape[0])
                * float(raw["interpolation_dt_s"]),
                8,
            )
            if isinstance(
                raw.get("interpolation_dt_s"), (int, float),
            )
            and not isinstance(raw.get("interpolation_dt_s"), bool)
            and float(raw["interpolation_dt_s"]) > 0.0
            else 0.0
        ),
        "plan_ref": plan_ref,
    }


def _motion_probe_results(
    normalized: list[dict[str, Any]],
    candidates: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Select each query's best reachable variant; keep every variant row."""

    results: list[dict[str, Any]] = []
    for query in normalized:
        rows = candidates[query["query_id"]]
        successful = [row for row in rows if row["reachable"]]
        successful.sort(key=probe_variant_sort_key)
        selected = successful[0] if successful else None
        results.append({
            "query_id": query["query_id"],
            "arm": query["arm"],
            "reachable": selected is not None,
            "selected_variant_id": (
                selected["variant_id"] if selected is not None else None
            ),
            "plan_ref": selected["plan_ref"] if selected is not None else None,
            "selected_target_pose_xyzw": (
                selected["target_pose_xyzw"] if selected is not None else None
            ),
            "variants": rows,
        })
    return results
