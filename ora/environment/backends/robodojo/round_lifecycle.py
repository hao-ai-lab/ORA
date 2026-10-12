"""Measured RoboDojo reset and round-isolation lifecycle."""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping
from .bridge import RobodojoError
from .common import file_hash, pose_error, write_json
from . import layout_rounds
from ora.environment.benchmarks.robodojo import (
    RoboDojoBenchmarkError, public_context, validate_scene_registration,
)
from ora.environment.benchmarks.robodojo_layouts import (
    configured_layout_schedule, scheduled_layout,
)

ROUND_RESET_SCHEMA = "ora.robodojo-round-reset.v1"
ROUND_ISOLATION_SCHEMA = "ora.robodojo-round-isolation.v1"
#: Every session field that would carry one round's value into the next if it
#: were not rebound, mapped to the empty value a fresh round starts from.
#: ``_bind_reset_state`` clears exactly these and ``_round_isolation_audit``
#: proves it afterwards, so a new per-episode field is declared once and can
#: never be silently left behind by a round reset.
PER_ROUND_CLEARED_CHANNELS: dict[str, Any] = {
    "action_sequences": [],
    "abort_reason": None,
    "boundary_reason": None,
    "fault_reason": None,
    "last_observation_id": None,
    "last_receipt": None,
    "official_terminal": None,
    "plan_references": {},
    "sim_steps": 0,
    "world_mutation_actions": [],
    "world_mutation_sequence": 0,
}
#: How far the arms may sit from the pose the *first* reset of this session
#: measured before a round reset counts as an incomplete world restore. The
#: plant reset is deterministic (``set_robot_init_state`` re-applies the
#: layout's own initial state), so this is a fault threshold, not a budget;
#: the scripted round-reset isolation gate asserts the same 0.02 m.
ROUND_HOME_POSITION_TOLERANCE_M = 0.02
ROUND_HOME_ROTATION_TOLERANCE_DEG = 5.0
HOME_JOINT_WRITE_SCHEMA = "ora.robodojo-pre-reset-home-joint-write.v1"


class RobodojoRoundMixin:
    def _init_layout_schedule(self) -> None:
        """Bind round 0 to the RunPlan's one layout (0 when unset), or to the first of a per-round schedule."""
        try:
            self.layout_schedule = configured_layout_schedule({**self.config, "task": self.task})
        except RoboDojoBenchmarkError as exc:
            raise RobodojoError(f"unsupported RoboDojo layout schedule: {exc}") from exc
        first = (self.config.get("layout_id", 0) if self.layout_schedule is None
                 else self.layout_schedule["rounds"][0]["layout_id"])
        if (type(first) is not int or first < 0 or self.config.get("layout_id", first) != first
                or isinstance(self.config.get("layout_id"), bool)):
            raise RobodojoError("unsupported RoboDojo first-stage binding: layout_id")
        self.config["layout_id"] = self.layout_id = first
        self.physics_clock_epoch = 0
        self.session_plant_facts = self.session_task_instruction = self.round0_layout_verification = None

    def _reset_environment(self, stack_log_name: str, *, relaunch_layout: int | None = None) -> Any:
        import faulthandler

        schedule = self.layout_schedule
        self.environment.harness_layout_ids = (
            (self.layout_id,) if schedule is None else tuple(entry["layout_id"] for entry in schedule["rounds"]))
        if schedule is not None and getattr(self.environment, "harness_simulation_time_offset_s", None) is None:
            # Epoch 0 starts the annotator clock with the context clock.
            self.environment.harness_simulation_time_offset_s = 0.0
        relaunch = None
        with (self.output_root / stack_log_name).open("w") as stack_log:
            faulthandler.dump_traceback_later(60, file=stack_log)
            try:
                if relaunch_layout is None:
                    self.environment.reset(seed=[self.layout_id])
                else:
                    relaunch = self.environment.relaunch_for_layout(relaunch_layout)
            finally:
                faulthandler.cancel_dump_traceback_later()
        if self.environment.step_lim != self.native_action_limit or not self.environment.reward_registered:
            raise RobodojoError("official selected-task lifecycle does not match the binding")
        return relaunch

    def _write_session_home_joints(self) -> dict[str, Any]:
        """Teleport every arm's joints to the first reset's measured home state."""
        manager = self.environment.robot_manager
        arms: dict[str, Any] = {}
        for arm, robot in self.robots.items():
            home = self.session_home_joints.get(arm)
            if home is None:
                raise RobodojoError(f"no session home joints for arm {arm}")
            articulation = manager.robot_key[manager.robot_list.index(robot)]
            names = list(articulation.joint_names)
            ids = [names.index(name) for name in robot.arm_joints_name]
            before = [float(value) for value in manager.get_joint(robot, env_idx_list=[0])[0]]
            position, velocity = _joint_state_rows(articulation, [float(value) for value in home])
            articulation.write_joint_state_to_sim(position, velocity, joint_ids=ids)
            after = [float(value) for value in manager.get_joint(robot, env_idx_list=[0])[0]]
            arms[arm] = {
                "joints_before": before, "home_joints": [float(value) for value in home],
                "joints_after_write": after,
                "joint_distance_before_rad": math.dist(before, home),
            }
        return {"schema": HOME_JOINT_WRITE_SCHEMA, "method": "articulation_joint_state_write",
                "claim_boundary": "joint positions written, velocities zeroed; the upstream reset and the "
                                  "isolation audit that follow decide whether the world is restored",
                "arms": arms}

    def _scene_registration(self) -> dict[str, Any]:
        """Read back the scene/evaluator registration this reset published.

        The per-scene half of RoboDojo qualification is exactly this object,
        and one boot establishes all of it: which layout instances exist and
        resolved to scene objects, which official evaluator check lists the
        task registered, and the native-action limit the environment itself
        reports. It is validated against this session's own binding rather
        than trusted, and a plant that publishes none fails closed.
        """

        value = getattr(self.environment, "scene_registration", None)
        if value is None:
            raise RobodojoError("the official scene/evaluator registration is unavailable")
        return validate_scene_registration(
            value, native_action_limit=self.native_action_limit,
        )

    def _bind_reset_state(self) -> dict[str, Any]:
        """Bind retained targets and recovery anchors to the freshly reset plant."""
        # A reset is a new command lineage. Do not label its fresh SDK buffers
        # with a directive retained from the previous round.
        self.gripper_directives = {}
        state = self._proprioception()
        self.pose_targets, self.gripper_directives = {}, {}
        self.home_joints, self.home_poses = {}, {}
        for arm, value in state["end_effectors"].items():
            self.pose_targets[arm] = list(value["pose_xyz_xyzw"])
            self.gripper_directives[arm] = value["gripper_command"]
            self.home_joints[arm] = list(value["arm_joint_qpos"])
            self.home_poses[arm] = list(value["pose_xyz_xyzw"])
        self.last_safe_joints = deepcopy(self.home_joints)
        self.last_safe_poses = deepcopy(self.home_poses)
        if not self.session_home_poses:
            self.session_home_poses = deepcopy(self.home_poses)
            self.session_home_joints = deepcopy(self.home_joints)
            if self.layout_schedule is not None:
                # Round 0 must be its scheduled layout, and every later round,
                # which rebuilds the scene, must rebuild this plant.
                self.round0_layout_verification = layout_rounds.round0_layout_verification(self)
                if not self.round0_layout_verification["passed"]:
                    raise RobodojoError("round 0 did not load its scheduled layout: "
                                        + json.dumps(self.round0_layout_verification["checks"], sort_keys=True))
                self.session_plant_facts = layout_rounds.plant_facts(self)
                self.session_task_instruction = self._task_instruction()
        for name, empty in PER_ROUND_CLEARED_CHANNELS.items():
            setattr(self, name, deepcopy(empty))
        self.reset_physics_steps = self._clock()
        return self._proprioception()

    def restart_round(self, round_index: int, *, seed: int | None = None) -> dict[str, Any]:
        """Reset the same isolated plant for another independent round.

        Artifact sequences stay session-global so no observation, receipt,
        motion probe or plan artifact can collide across rounds; every
        per-episode target, anchor, plan reference and lifecycle field is
        rebound to the freshly reset plant.
        """
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise RobodojoError("RoboDojo round index must be an integer")
        if round_index != self.round_index + 1:
            raise RobodojoError("RoboDojo round reset must advance by exactly one")
        self._require_started()
        if seed is not None and (isinstance(seed, bool) or seed != 0):
            raise RobodojoError("RoboDojo admits only evaluation seed 0")
        if (self.output_root / "official-verdict.json").exists():
            raise RobodojoError("previous round verdict must be archived before environment reset")
        self.environment.harness_round_index = round_index
        if self.layout_schedule is not None:
            return self._restart_scheduled_round(round_index)
        # The upstream reset drives the arms home under their position drives
        # and cannot pass an obstruction on the home side (g004 classify_objects
        # and build_tower: 100 N m saturated, joints 1-3 motionless), so the
        # isolation audit below failed. Writing the session-home joint state is
        # the restore that audit already demands; it gives the next round
        # nothing the first did not have.
        home_write = self._write_session_home_joints()
        self._reset_environment(f"reset-initialization-stack-round-{round_index:04d}.txt")
        state = self._bind_reset_state()
        self.round_index = round_index
        isolation = self._round_isolation_audit(state)
        reset = {
            "schema": ROUND_RESET_SCHEMA, "task": self.task, "eval_seed": 0, "seed": 0, "layout_id": self.layout_id,
            "round_index": round_index, "action_codec": self.action_codec,
            "source_manifest": {"path": "source-manifest.json", "sha256": self.source_hash},
            "physics_steps_before_policy": self.reset_physics_steps,
            "reward_registered": isolation["measured"]["reward_registered"],
            "task_instruction": self._task_instruction(), "public_environment_context": public_context(),
            # Republished on every round: comparing it with round 0's is how a
            # task qualification record shows that a repeated reset of the same
            # plant kept this scene's object identities.
            "scene_registration": self._scene_registration(),
            "artifact_sequences_preserved": {
                "observation_sequence": self.observation_sequence,
                "action_sequence": self.action_sequence,
                "motion_probe_sequence": self.motion_probe_sequence,
            },
            "pre_reset_home_joint_write": home_write,
            "round_isolation": isolation,
        }
        # The record is written before the refusal so a failed restore leaves
        # readable evidence of *which* check failed rather than only a stack.
        write_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        if not isolation["passed"]:
            self._fault("robodojo_round_reset_isolation_failed")
            raise RobodojoError(
                "RoboDojo round reset did not restore an isolated world: "
                + "; ".join(isolation["failures"])
            )
        self._start_execution_capture()
        return reset

    def _restart_scheduled_round(self, round_index: int) -> dict[str, Any]:
        """Start the round's scheduled layout as a new episode in a relaunched simulation.

        The previous simulation is destroyed, so the session-home joint write
        has nothing to write to; the relaunched robot starts from its
        configured initial state and the home audit below still measures it.
        """
        try:
            entry = scheduled_layout(self.layout_schedule, round_index)
        except RoboDojoBenchmarkError as exc:
            raise RobodojoError(str(exc)) from exc
        previous_epoch, previous_layout_id = self.physics_clock_epoch, self.layout_id
        previous_offset = layout_rounds.simulation_time_offset(self)
        relaunch = self._reset_environment(
            f"reset-initialization-stack-round-{round_index:04d}.txt", relaunch_layout=entry["layout_id"])
        self.layout_id, self.physics_clock_epoch = entry["layout_id"], previous_epoch + 1
        state = self._bind_reset_state()
        self.round_index = round_index
        layout_checks = layout_rounds.scheduled_layout_checks(
            self, entry, relaunch, previous_epoch=previous_epoch, previous_layout_id=previous_layout_id,
            previous_offset=previous_offset)
        isolation = self._round_isolation_audit(state, layout_checks=layout_checks)
        reset = layout_rounds.round_reset_v2(self, {
            "task": self.task, "eval_seed": 0, "seed": 0, "round_index": round_index,
            "action_codec": self.action_codec,
            "source_manifest": {"path": "source-manifest.json", "sha256": self.source_hash},
            "physics_steps_before_policy": self.reset_physics_steps,
            "reward_registered": isolation["measured"]["reward_registered"],
            "task_instruction": self._task_instruction(), "public_environment_context": public_context(),
            "scene_registration": self._scene_registration(),
            "artifact_sequences_preserved": {
                "observation_sequence": self.observation_sequence,
                "action_sequence": self.action_sequence,
                "motion_probe_sequence": self.motion_probe_sequence,
            },
            "round_isolation": isolation,
        }, relaunch)
        write_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        if not isolation["passed"]:
            self._fault("robodojo_round_reset_isolation_failed")
            raise RobodojoError(
                "RoboDojo scheduled round did not start an isolated new layout: "
                + "; ".join(isolation["failures"])
            )
        self._start_execution_capture()
        return reset

    def _official_lifecycle_state(self) -> dict[str, Any]:
        """Read the upstream episode-lifecycle latches the wrapper owns.

        These are lifecycle counters and latches, not the success predicate:
        ``success`` is the upstream *failure* latch, which starts at ``True``
        and is only ever set ``False`` by ``RewardManager._mark_env_failed``,
        and ``end_flag`` / ``take_action_cnt`` are the episode's own step
        bookkeeping. They stay in Control evidence and are never published to
        the policy, exactly like ``status()``.
        """

        environment = self.environment
        try:
            return {
                "readable": True,
                "step_lim": int(environment.step_lim),
                "reward_registered": bool(environment.reward_registered),
                "take_action_cnt": int(environment.take_action_cnt[0]),
                "end_flag": bool(environment.end_flag[0]),
                "failure_latch_clear": bool(environment.success[0]),
                "wrapper_official_terminal": getattr(
                    environment, "_official_terminal", None,
                ) is None,
            }
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # A plant that cannot report its own lifecycle cannot prove it was
            # restored, so every lifecycle check fails rather than being
            # skipped: unreadable is refused, not assumed clean.
            return {
                "readable": False,
                "step_lim": None,
                "reward_registered": False,
                "take_action_cnt": None,
                "end_flag": None,
                "failure_latch_clear": False,
                "wrapper_official_terminal": False,
            }

    def _round_isolation_audit(
        self, state: Mapping[str, Any], *,
        layout_checks: tuple[Mapping[str, bool], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Prove the world was restored and no channel kept a stale value.

        Called after ``_reset_environment`` and ``_bind_reset_state`` and
        before the Supervisor publishes the round's first observation, so the
        policy cannot act on a half-restored world. Every check is a
        measurement of the plant or of this session's own carried state; none
        of them is a literal. A failing check is an infrastructure fault --
        the caller raises and the round keeps a null official verdict --
        rather than a task outcome.
        """

        official = self._official_lifecycle_state()
        failures: list[str] = []
        checks: dict[str, bool] = {}

        checks["official_lifecycle_rebound"] = (
            official["readable"] is True
            and official["step_lim"] == self.native_action_limit
            and official["reward_registered"] is True
        )
        checks["episode_counters_zeroed"] = (
            official["take_action_cnt"] == 0 and official["end_flag"] is False
        )
        checks["failure_latch_cleared"] = official["failure_latch_clear"] is True
        checks["official_terminal_cleared"] = (
            official["wrapper_official_terminal"] is True
            and self.official_terminal is None
            and self.status()["terminal"] is False
        )

        stale = sorted(
            name for name, empty in PER_ROUND_CLEARED_CHANNELS.items()
            if getattr(self, name) != empty
        )
        checks["carried_channels_cleared"] = not stale

        if layout_checks is None:
            layout = self.source.get("layout_file", {})
            recorded_hash = str(layout.get("sha256", ""))
            try:
                current_hash = file_hash(Path(str(layout.get("path"))))
            except (OSError, TypeError):
                current_hash = ""
            checks["layout_identity_unchanged"] = bool(
                recorded_hash
            ) and current_hash == recorded_hash
            layout_measured: Mapping[str, Any] = {
                "layout_sha256": current_hash, "layout_sha256_at_start": recorded_hash,
            }
        else:
            # A scheduled round loads a different layout by design; its own
            # file, relaunch, plant and scene facts replace the identity check.
            checks.update(layout_checks[0])
            layout_measured = layout_checks[1]

        position_error = rotation_error = 0.0
        for arm, value in state["end_effectors"].items():
            anchor = self.session_home_poses.get(arm)
            if anchor is None:
                checks["arm_home_restored"] = False
                failures.append(f"no session home anchor for arm {arm}")
                break
            distance, angle = pose_error(anchor, value["pose_xyz_xyzw"])
            position_error = max(position_error, distance)
            rotation_error = max(rotation_error, math.degrees(angle))
        else:
            checks["arm_home_restored"] = (
                position_error <= ROUND_HOME_POSITION_TOLERANCE_M
                and rotation_error <= ROUND_HOME_ROTATION_TOLERANCE_DEG
            )

        failures.extend(
            f"{name} did not hold" for name, passed in sorted(checks.items())
            if not passed
        )
        if stale:
            failures.append(
                "channels still carrying a previous round: " + ", ".join(stale)
            )
        return {
            "schema": ROUND_ISOLATION_SCHEMA,
            "passed": not failures,
            "checks": dict(sorted(checks.items())),
            "measured": {
                "arms": sorted(state["end_effectors"]),
                "cleared_channels": sorted(PER_ROUND_CLEARED_CHANNELS),
                "end_flag": official["end_flag"],
                "failure_latch_clear": official["failure_latch_clear"],
                **layout_measured,
                "max_home_position_error_m": round(position_error, 6),
                "max_home_rotation_error_deg": round(rotation_error, 4),
                "native_action_limit": self.native_action_limit,
                "physics_steps_before_policy": self.reset_physics_steps,
                "reward_registered": official["reward_registered"],
                "step_lim": official["step_lim"],
                "take_action_cnt": official["take_action_cnt"],
                "stale_channels": stale,
            },
            "tolerances": {
                "home_position_m": ROUND_HOME_POSITION_TOLERANCE_M,
                "home_rotation_deg": ROUND_HOME_ROTATION_TOLERANCE_DEG,
            },
            "failures": failures,
        }


def _joint_state_rows(articulation: Any, home: list[float]) -> tuple[Any, Any]:
    """One-environment position/velocity rows in the articulation's own array type."""
    device = getattr(articulation, "device", None)
    if device is not None:
        import torch

        position = torch.tensor([home], dtype=torch.float32, device=device)
        return position, torch.zeros_like(position)
    import numpy as np

    position = np.asarray([home], dtype=float)
    return position, np.zeros_like(position)
