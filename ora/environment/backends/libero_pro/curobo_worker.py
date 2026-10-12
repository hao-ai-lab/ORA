"""CuRobo v0.7.8 JSON-lines worker; imported only by the isolated planner Python."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from typing import Any, Mapping

from ora.environment.backends.libero_pro.planning_diagnostics import (
    install_ik_observer, reset_ik_observer, summarize,
)


REQUEST_SCHEMA = "ora.curobo-motion-request.v1"
RESPONSE_SCHEMA = "ora.curobo-motion-response.v1"
IK_RESPONSE_SCHEMA = "ora.curobo-ik-response.v1"
START_STATE_SCHEMA = "ora.curobo-start-state.v1"


def _finite_tensor(value: Any, *, shape: tuple[int, ...] | None = None) -> Any:
    """Copy diagnostics off-device without importing Torch on the host."""
    import numpy as np

    array = np.asarray(value.detach().cpu().tolist(), dtype=np.float64)
    if not array.size or not np.isfinite(array).all() or (
        shape is not None and array.shape != shape
    ):
        raise ValueError("CuRobo constraint diagnostic is non-finite or malformed")
    return array


def _check_robot_only_start_state(
    motion_gen: Any, start_state: Any, status_type: Any,
) -> tuple[bool, Any]:
    """Classify the pinned robot-only constraints without changing their weights.

    Upstream v0.7.8's diagnostic branch assumes a world constraint exists and
    temporarily disables collision terms. Evaluate the same graph-safety state
    and terms directly instead; even a diagnostic exception leaves every cost
    enabled and weighted exactly as it was before the request.
    """
    diagnostic: dict[str, Any] = {
        "schema": START_STATE_SCHEMA, "status": "checking",
        "joint_names": list(start_state.joint_names or []),
        "joint_positions": None, "dtype": str(start_state.position.dtype),
        "position_limits": None, "activation_distance": None,
        "constraints": {}, "aggregate": None, "valid": None,
    }
    motion_gen.start_state_diagnostics = diagnostic
    position = start_state.position
    if len(position.shape) == 1:
        position = position.unsqueeze(0)
    diagnostic["joint_positions"] = _finite_tensor(position, shape=(1, 7)).tolist()[0]
    rollout = motion_gen.rollout_fn
    if rollout.cuda_graph_instance:
        raise ValueError("CuRobo start checker cannot share a captured CUDA graph")
    if rollout.constraint_cfg.primitive_collision_cfg is not None:
        raise ValueError("robot-only CuRobo start checker received a world constraint")
    diagnostic["constraints"]["world_collision"] = {
        "configured": False, "enabled": False, "cost": None, "feasible": None,
    }
    terms = (
        ("joint_limits", "bound_cfg", "bound_constraint"),
        ("self_collision", "self_collision_cfg", "robot_self_collision_constraint"),
    )
    constraints = []
    for name, config_name, attribute in terms:
        constraint = getattr(rollout, attribute, None)
        if getattr(rollout.constraint_cfg, config_name, None) is None or constraint is None:
            raise ValueError(f"required CuRobo {name} constraint is missing")
        weight = _finite_tensor(constraint.weight)
        if constraint.enabled is not True or (weight <= 0).any():
            raise ValueError(f"required CuRobo {name} constraint is disabled")
        diagnostic["constraints"][name] = {
            "configured": True, "enabled": True, "weight": weight.tolist(),
            "cost": None, "feasible": None,
        }
        constraints.append(constraint)
    limits = _finite_tensor(constraints[0].joint_limits.position, shape=(2, 7))
    if (limits[0] >= limits[1]).any():
        raise ValueError("CuRobo joint position limits are malformed")
    diagnostic["position_limits"] = limits.tolist()
    activation = _finite_tensor(constraints[0].activation_distance, shape=(4,))
    if (activation < 0).any() or (activation >= 0.5).any():
        raise ValueError("CuRobo bound activation distances are malformed")
    diagnostic["activation_distance"] = activation.tolist()
    state = rollout.dynamics_model.forward(rollout.start_state, position.unsqueeze(1))
    for (name, _, _), constraint, argument in zip(
        terms, constraints, (state.state_seq, state.robot_spheres), strict=True,
    ):
        cost = _finite_tensor(constraint.forward(argument), shape=(1, 1))
        if (cost < 0).any():
            raise ValueError("CuRobo constraint cost is negative")
        diagnostic["constraints"][name].update(
            cost=float(cost.item()), feasible=bool((cost == 0).all()),
        )
    # Check agreement against the original aggregate evaluator on this exact
    # state. Never classify missing/changed constraint semantics as a refusal.
    metrics = rollout.constraint_fn(state, use_batch_env=False)
    aggregate = _finite_tensor(metrics.constraint, shape=(1, 1))
    feasible = _finite_tensor(metrics.feasible, shape=(1, 1))
    if (aggregate < 0).any() or feasible.item() not in (0, 1):
        raise ValueError("CuRobo aggregate constraint evidence is malformed")
    diagnostic["aggregate"] = {
        "cost": float(aggregate.item()), "feasible": bool(feasible.item()),
    }
    valid = all(diagnostic["constraints"][name]["feasible"] for name, _, _ in terms)
    if bool(feasible.item()) != valid or (
        bool((aggregate == 0).all()) != valid
    ):
        raise ValueError("CuRobo aggregate and component start constraints disagree")
    diagnostic.update(status="valid" if valid else "invalid", valid=valid)
    if not diagnostic["constraints"]["joint_limits"]["feasible"]:
        return False, status_type.INVALID_START_STATE_JOINT_LIMITS
    if not diagnostic["constraints"]["self_collision"]["feasible"]:
        return False, status_type.INVALID_START_STATE_SELF_COLLISION
    return True, None


def _motion_gen_class(base: type, status_type: Any) -> type:
    """Instance-local subclass factory, also usable with a CPU fake boundary."""
    class HarnessMotionGen(base):
        def check_start_state(self, start_state: Any) -> tuple[bool, Any]:
            return _check_robot_only_start_state(self, start_state, status_type)

    return HarnessMotionGen


class Worker:
    def __init__(self, *, device: str) -> None:
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenStatus

        self.tensor_args = TensorDeviceType(device=torch.device(device))
        with contextlib.redirect_stdout(sys.stderr):
            config = MotionGenConfig.load_from_robot_config(
                "franka.yml", None, self.tensor_args,
                interpolation_dt=0.02,
                evaluate_interpolated_trajectory=True,
                use_cuda_graph=False,
            )
            self.motion_gen = _motion_gen_class(MotionGen, MotionGenStatus)(config)
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
            install_ik_observer(self.motion_gen)

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ora.environment.controllers.curobo_motion import validate_curobo_request

        self.motion_gen.start_state_diagnostics = None
        reset_ik_observer(self.motion_gen)
        validated = None
        started = time.time()
        try:
            validated = validate_curobo_request({
                key: value for key, value in request.items()
                if key not in {"schema", "command", "request_id"}
            })
            body = self._plan(validated)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            body = {
                "schema": RESPONSE_SCHEMA, "status": "planner_error",
                "reachable": False, "world_model": "none", "valid_query": None,
                "planning_time_s": time.time() - started,
                "error": {
                    "type": type(exc).__name__, "message": str(exc) or type(exc).__name__,
                },
            }
        body["request"] = validated
        body["start_state"] = self.motion_gen.start_state_diagnostics
        body["planning_diagnostics"], body["planning_diagnostic_evidence"] = summarize(
            self.motion_gen, body, validated,
        )
        return body

    def _plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        started = time.time()
        joint_names = request.get("joint_names")
        joint_positions = request.get("joint_positions")
        target = request.get("target_pose_base_xyzw")
        goal_joint_positions = request.get("goal_joint_positions")
        allowed = {
            "schema", "command", "request_id", "joint_names", "joint_positions",
            "target_pose_base_xyzw", "goal_joint_positions",
        }
        if (
            bool(set(request) - allowed)
            or
            not isinstance(joint_names, list) or len(joint_names) != 7
            or not isinstance(joint_positions, list) or len(joint_positions) != 7
            or (target is None) == (goal_joint_positions is None)
            or target is not None and (
                not isinstance(target, list) or len(target) != 7
            )
            or goal_joint_positions is not None and (
                not isinstance(goal_joint_positions, list)
                or len(goal_joint_positions) != 7
            )
        ):
            raise ValueError(
                "plan requires seven joints and a target pose or joint goal"
            )
        q = JointState.from_position(
            self.tensor_args.to_device([joint_positions]), joint_names=joint_names,
        )
        start_kinematics = self.motion_gen.compute_kinematics(q)
        start_position = start_kinematics.ee_pose.position[0].detach().cpu().numpy()
        start_wxyz = start_kinematics.ee_pose.quaternion[0].detach().cpu().tolist()
        with contextlib.redirect_stdout(sys.stderr):
            plan_config = MotionGenPlanConfig(
                max_attempts=4, enable_graph_attempt=2,
                enable_graph=True, enable_opt=True,
            )
            if goal_joint_positions is not None:
                goal_state = JointState.from_position(
                    self.tensor_args.to_device([goal_joint_positions]),
                    joint_names=joint_names,
                )
                result = self.motion_gen.plan_single_js(q, goal_state, plan_config)
            else:
                goal = Pose(
                    position=self.tensor_args.to_device([target[:3]]),
                    quaternion=self.tensor_args.to_device([[
                        target[6], target[3], target[4], target[5],
                    ]]),
                )
                result = self.motion_gen.plan_single(q, goal, plan_config)
        success = bool(result.success.item())
        body: dict[str, Any] = {
            "schema": RESPONSE_SCHEMA,
            "status": "planned" if success else "not_reachable",
            "reachable": success,
            "world_model": "none",
            "planner_status": "success" if success else str(result.status),
            "valid_query": bool(result.valid_query),
            "planning_time_s": time.time() - started,
            "start_ee_pose_base_xyzw": start_position.tolist() + [
                start_wxyz[1], start_wxyz[2], start_wxyz[3], start_wxyz[0],
            ],
            "requested_target_pose_base_xyzw": target,
            "requested_goal_joint_positions": goal_joint_positions,
        }
        if success:
            trajectory = result.get_interpolated_plan()
            positions = trajectory.position.detach().cpu().numpy()
            velocities = (
                trajectory.velocity.detach().cpu().numpy()
                if trajectory.velocity is not None else np.zeros_like(positions)
            )
            body["trajectory"] = {
                "joint_names": list(trajectory.joint_names or joint_names),
                "positions": positions.tolist(),
                "velocities": velocities.tolist(),
                "dt_s": float(result.interpolation_dt),
            }
        return body


    def solve_ik(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Read-only IK for exact base-frame panda_hand poses (``controllers/curobo_ik.py``).

        Each query is one ``solve_single`` of this MotionGen's own IK solver --
        the solver, seed count and convergence thresholds ``plan_single`` uses
        for its IK stage -- seeded and regularised the way that stage is: the
        start configuration (the measured joints, or a chained query's
        solution) as retract configuration and as one seed beside the solver's
        random ones. No trajectory is optimised and nothing is executed; the
        solver's seed sampler advances, as it does for a ``probe_motion`` plan.
        """
        from ora.environment.controllers.curobo_ik import validate_curobo_ik_request

        started = time.time()
        try:
            validated = validate_curobo_ik_request({
                key: value for key, value in request.items()
                if key not in {"schema", "command", "request_id"}
            })
            return self._solve_ik(validated, started)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {
                "schema": IK_RESPONSE_SCHEMA, "status": "planner_error", "world_model": "none",
                "solve_time_s": time.time() - started,
                "error": {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__},
            }
        finally:
            # A reach solve is never part of the next plan's IK diagnostics.
            reset_ik_observer(self.motion_gen)

    def _solve_ik(self, request: Mapping[str, Any], started: float) -> dict[str, Any]:
        import numpy as np
        from curobo.types.math import Pose
        from curobo.types.robot import JointState

        names = request["joint_names"]
        solver = self.motion_gen.ik_solver

        def flat(value: Any) -> Any:
            return np.asarray(value.detach().cpu().tolist(), dtype=np.float64).reshape(-1)

        def error(value: Any) -> float | None:
            # A failed solve's best-seed error is evidence only: a non-finite
            # value is unknown, never a malformed answer that retires the worker.
            number = float(flat(value)[0])
            return number if np.isfinite(number) and number >= 0 else None

        def forward(joints: list[float]) -> list[float]:
            state = JointState.from_position(self.tensor_args.to_device([joints]), joint_names=names)
            pose = self.motion_gen.compute_kinematics(state).ee_pose
            position = pose.position[0].detach().cpu().tolist()
            w, x, y, z = pose.quaternion[0].detach().cpu().tolist()
            return [*map(float, position), float(x), float(y), float(z), float(w)]

        results: list[dict[str, Any]] = []
        ends: list[list[float] | None] = []
        for query in request["queries"]:
            start = query["from_index"]
            seed = request["joint_positions"] if start is None else ends[start]
            if seed is None:
                results.append({"solved": False, "skipped": True, "joint_positions": None,
                                "ee_pose_base_xyzw": None, "position_error_m": None,
                                "rotation_error": None})
                ends.append(None)
                continue
            target = query["target_pose_base_xyzw"]
            goal = Pose(position=self.tensor_args.to_device([target[:3]]),
                        quaternion=self.tensor_args.to_device([[target[6], *target[3:6]]]))
            with contextlib.redirect_stdout(sys.stderr):
                answer = solver.solve_single(
                    goal, retract_config=self.tensor_args.to_device([seed]),
                    seed_config=self.tensor_args.to_device([[seed]]), return_seeds=1,
                    use_nn_seed=False,
                )
            solved = bool(flat(answer.success)[0])
            joints = [float(item) for item in flat(answer.solution)[:7]] if solved else None
            results.append({
                "solved": solved, "skipped": False, "joint_positions": joints,
                "ee_pose_base_xyzw": forward(joints) if solved else None,
                "position_error_m": error(answer.position_error),
                "rotation_error": error(answer.rotation_error),
            })
            ends.append(joints)
        limits = np.asarray(
            self.motion_gen.rollout_fn.bound_constraint.joint_limits.position.detach().cpu().tolist(),
            dtype=np.float64,
        )
        return {
            "schema": IK_RESPONSE_SCHEMA, "status": "solved", "world_model": "none",
            "solver": {"num_seeds": int(solver.num_seeds),
                       "position_threshold_m": float(solver.position_threshold),
                       "rotation_threshold": float(solver.rotation_threshold),
                       "attempts_per_query": 1},
            "joint_names": list(names), "joint_limits": limits.tolist(),
            "start_ee_pose_base_xyzw": forward(list(request["joint_positions"])),
            "results": results, "solve_time_s": time.time() - started,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    worker = Worker(device=args.device)
    for line in sys.stdin:
        request_id: Any = None
        request: Any = {}
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema drift")
            command = request.get("command")
            if command == "ready":
                response = {"status": "ready", "version": "v0.7.8"}
            elif command == "plan":
                response = worker.plan(request)
            elif command == "ik":
                response = worker.solve_ik(request)
            elif command == "close":
                print(json.dumps({"request_id": request_id, "status": "closed"}), flush=True)
                return 0
            else:
                raise ValueError("unsupported command")
            print(json.dumps({"request_id": request_id, **response}), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({
                "request_id": request_id,
                "schema": RESPONSE_SCHEMA,
                "status": "planner_error",
                "reachable": False,
                "world_model": "none",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
