"""Trusted scripted qualification for the RoboDojo sparse motion v3 dispatch.

This is the RoboDojo counterpart of the LIBERO sparse-motion gate: a closed
case table exercised against the real dual ARX X5 plant in the isolated Isaac
worker through the exact public v3 contract (never a policy model). A passing
device-keyed record is the admission evidence for adding ``robodojo`` to the
sparse-motion qualification table — the table row and the record land in the
same change.

The judgement layer consumes only public receipt fields and is unit tested
with a fake boundary; the CLI assembles the live worker from the explicit
runtime configuration through the same bootstrap formal runs use.  The
rotation cases are the 2π-alias monitors: a large rotation may honestly end
``not_reached`` but must never realize a translation drift bound violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT

from .sparse_motion import (
    GLOBAL_EXECUTOR, GRIPPER_EXECUTOR, JAW_SYMMETRY_AXIS, LOCAL_EXECUTOR, RECOVERY_EXECUTOR,
    ROUTE_REFUSAL_CODE, SPARSE_RECEIPT_SCHEMA, rotate_about_local_axis,
)


from .motion_checks import (
    SCHEMA,
    MOTION_ACTION_V3_SCHEMA,
    ROTATION_ONLY_DRIFT_BOUND_M,
    REACH_POSITION_TOLERANCE_M,
    REACH_ORIENTATION_TOLERANCE_RAD,
    INACTIVE_ARM_DRIFT_BOUND_M,
    GRIPPER_TOLERANCE,
    ARM,
    OTHER_ARM,
    JAW_APERTURE_PROBE_ID,
    JAW_CALIBRATION_SCHEMA,
    JAW_COMMAND_JOINT_RANGE_M,
    JAW_MESH_PAD_OFFSET_M,
    JAW_CLOSED_PAD_GAP_M,
    JAW_APERTURE_PER_JOINT_M,
    JAW_NOMINAL_MAXIMUM_APERTURE_M,
    JAW_FINGER_JOINT_LIMIT_M,
    JAW_OPEN_APERTURE_TOLERANCE_M,
    JAW_CLOSED_APERTURE_BOUND_M,
    JAW_MIMIC_BOUND_M,
    JAW_SWEEP_AGREEMENT_M,
    JAW_FINGER_UNDERSHOOT_BOUND_M,
    JAW_SWEEPS,
    JAW_APERTURE_CASE_ID,
    JAW_CONTACT_LEG_STATUS,
    JAW_CONTACT_LEG_REASON,
    X5_LINK6_REACH_BOUND_M,
    X5_BASE_MOUNTS_M,
    UNREACHABLE_POSITION,
    PLANNER_IK_REFUSAL_CODE,
    _rotate_approach,
    _pose,
    _target,
    build_case_arguments,
    MOTION_CASES,
    SEQUENCE_CASES,
    ALL_CASE_IDS,
    _pose_error,
    _translation_drift,
    judge_motion_case,
    judge_gripper_case,
    judge_recovery_case,
    nominal_jaw_aperture_m,
    _jaw_axis_world,
    measured_pad_face_separation_m,
    derive_jaw_step,
    judge_jaw_aperture_sweeps,
    jaw_aperture_calibration,
    summarize_receipt,
)


def build_qualification_record(
    *, source_commit: str, host: Mapping[str, Any], session_identity: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executed = {str(result.get("case_id")) for result in case_results}
    missing = [case_id for case_id in ALL_CASE_IDS if case_id not in executed]
    record = {
        "schema": SCHEMA, "source_commit": source_commit, "host": dict(host),
        "session": dict(session_identity), "evidence": dict(evidence or {}),
        "cases": [dict(result) for result in case_results], "missing_cases": missing,
        "passed": not missing and all(bool(result.get("passed")) for result in case_results),
        "created_unix_s": time.time(),
    }
    body = {key: value for key, value in record.items() if key != "created_unix_s"}
    record["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return record


def _v3_action(step: int, observation: Mapping[str, Any], arguments: Mapping[str, Any],
               name: str = "move_pose") -> dict[str, Any]:
    return {
        "schema": MOTION_ACTION_V3_SCHEMA, "observation_id": str(observation["observation_id"]),
        "action_id": f"qualification-{step:04d}",
        "bound_observation_sha256": str(observation["observation_sha256"]),
        "action": name, "arguments": dict(arguments),
    }


def _probe_request(observation: Mapping[str, Any], pose: Sequence[float], policy: str,
                   offset: Sequence[float] = (-0.06, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "schema": "ora.motion-probe-request.v1",
        "observation_id": str(observation["observation_id"]),
        "observation_sha256": str(observation["observation_sha256"]),
        "variant_policy": policy,
        "queries": [{"query_id": "q0", "arm": ARM, "pose": _pose(
            [float(pose[i]) + float(offset[i]) for i in range(3)], pose[3:],
        )}],
    }


class _Runner:
    """Drive every case through the injected live boundary, recording each result."""

    def __init__(self, *, observe: Callable[[], Mapping[str, Any]],
                 execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 restart_round: Callable[[int], Mapping[str, Any]],
                 status: Callable[[], Mapping[str, Any]],
                 persist_verdict: Callable[[], Mapping[str, Any]],
                 current_poses: Callable[[Mapping[str, Any]], Mapping[str, Sequence[float]]],
                 native_action_limit: int,
                 measure_jaw_state: Callable[[], Mapping[str, Any]] | None = None) -> None:
        self.observe, self.execute, self.probe = observe, execute, probe
        self.restart_round, self.status, self.persist_verdict = restart_round, status, persist_verdict
        self.current_poses = current_poses
        self.measure_jaw_state = measure_jaw_state
        self.native_action_limit = int(native_action_limit)
        self.results: list[dict[str, Any]] = []
        self.step = 0
        self.home_poses: dict[str, list[float]] = {}

    def record(self, case_id: str, judgement: Mapping[str, Any], receipt: Mapping[str, Any] | None,
               **extra: Any) -> None:
        self.results.append({
            "case_id": case_id, "passed": bool(judgement.get("passed")),
            "notes": list(judgement.get("notes", [])),
            "receipt": summarize_receipt(receipt) if receipt is not None else None, **extra,
        })

    def fail(self, case_id: str, error: BaseException, stage: str = "execution") -> None:
        self.results.append({"case_id": case_id, "passed": False,
                             "notes": [f"{stage} error: {error}"], "receipt": None})

    def act(self, observation: Mapping[str, Any], arguments: Mapping[str, Any], name: str = "move_pose") -> Mapping[str, Any]:
        self.step += 1
        result = self.execute(_v3_action(self.step, observation, arguments, name))
        return result["receipt"] if isinstance(result, Mapping) and "receipt" in result else result

    def run_motion_matrix(self) -> None:
        first = self.observe()
        self.home_poses = {arm: [float(v) for v in pose] for arm, pose in self.current_poses(first).items()}
        observation = first
        for case in MOTION_CASES:
            try:
                poses = self.current_poses(observation)
                arguments = build_case_arguments(str(case["case_id"]), poses)
                receipt = self.act(observation, arguments, str(case["action"]))
                self.record(str(case["case_id"]), judge_motion_case(case, receipt), receipt)
            except Exception as error:  # recorded, never silently dropped
                self.fail(str(case["case_id"]), error)
            observation = self.observe()

    def run_sequence(self) -> None:
        observation = self.observe()
        try:
            receipt = self.act(observation, {"targets": {ARM: "home", OTHER_ARM: "home"}}, "recover_motion")
            judgement = judge_recovery_case(receipt, self.home_poses[ARM])
            other = judge_recovery_case(receipt, self.home_poses[OTHER_ARM], arm=OTHER_ARM)
            self.record("recover_home_after_matrix",
                        {"passed": judgement["passed"] and other["passed"],
                         "notes": judgement["notes"] + other["notes"]}, receipt)
        except Exception as error:
            self.fail("recover_home_after_matrix", error)
        for case_id, directive in (("gripper_close_holds_pose", "close"), ("gripper_open_holds_pose", "open")):
            try:
                observation = self.observe()
                receipt = self.act(observation, {"targets": {ARM: directive}}, "set_gripper")
                self.record(case_id, judge_gripper_case(directive, receipt), receipt)
            except Exception as error:
                self.fail(case_id, error)
        # Held out of the matrix with its case id; see SEQUENCE_CASES.
        # self.run_jaw_aperture_probe()
        try:
            observation = self.observe()
            anchor = [float(v) for v in self.current_poses(observation)[ARM]]
            lifted = self.act(observation, _target({ARM: [anchor[0], anchor[1], anchor[2] + 0.05, *anchor[3:]]},
                                                   "free_space"))
            observation = self.observe()
            receipt = self.act(observation, {"targets": {ARM: "backtrack_last_safe"}}, "recover_motion")
            judgement = judge_recovery_case(receipt, anchor)
            if lifted.get("execution_status") != "reached":
                judgement = {"passed": False, "notes": judgement["notes"] + ["lift before backtrack did not reach"]}
            self.record("recover_backtrack_after_move", judgement, receipt)
        except Exception as error:
            self.fail("recover_backtrack_after_move", error)
        self.run_plan_reference_cases()
        try:
            verdict = self.persist_verdict()
            terminal = self.status()
            passed = (verdict.get("terminal") is False and verdict.get("official_success") is None
                      and terminal.get("terminal") is False)
            self.record("official_preterminal_boundary", {"passed": passed, "notes": [] if passed else [
                f"preterminal verdict {verdict.get('official_success')!r} terminal {verdict.get('terminal')!r}",
            ]}, None, verdict={key: verdict.get(key) for key in ("terminal", "official_success", "native_actions")})
        except Exception as error:
            self.fail("official_preterminal_boundary", error)

    def run_jaw_aperture_probe(self) -> None:
        """Drive ``robodojo_x5_jaw_aperture_probe_v1``'s free-air legs.

        Both arms, both commands the public ``set_gripper`` contract admits,
        twice, in free air. Every command goes through the exact public v3
        boundary a policy uses; only the *reading* is private, and it comes
        from the worker's qualification-only jaw-state method.
        """

        if self.measure_jaw_state is None:
            self.results.append({
                "case_id": JAW_APERTURE_CASE_ID, "passed": False,
                "notes": ["no jaw-state measurement boundary was injected"],
                "receipt": None,
            })
            return
        try:
            arms = (ARM, OTHER_ARM)
            sweeps: dict[str, list[list[dict[str, Any]]]] = {
                arm: [] for arm in arms
            }
            for _sweep in range(JAW_SWEEPS):
                for arm in arms:
                    steps: list[dict[str, Any]] = []
                    for directive, command in (("open", 1.0), ("close", 0.0)):
                        observation = self.observe()
                        self.act(
                            observation, {"targets": {arm: directive}},
                            "set_gripper",
                        )
                        reading = self.measure_jaw_state()
                        steps.append(derive_jaw_step(
                            command=command, reading=reading["arms"][arm],
                        ))
                    sweeps[arm].append(steps)
            judgements = {
                arm: judge_jaw_aperture_sweeps(sweeps[arm]) for arm in arms
            }
            notes = [
                f"{arm}: {note}"
                for arm in arms for note in judgements[arm]["notes"]
            ]
            self.results.append({
                "case_id": JAW_APERTURE_CASE_ID,
                "passed": not notes,
                "notes": notes,
                "receipt": None,
                "probe_id": JAW_APERTURE_PROBE_ID,
                "arms": list(arms),
                "sweeps": [sweep for arm in arms for sweep in sweeps[arm]],
                "contact_leg": {
                    "status": JAW_CONTACT_LEG_STATUS,
                    "reason": JAW_CONTACT_LEG_REASON,
                },
                "command_boundary": (
                    "public_set_gripper_open_close_only_no_intermediate_"
                    "normalized_command_is_expressible"
                ),
            })
        except Exception as error:
            self.fail(JAW_APERTURE_CASE_ID, error)

    def run_plan_reference_cases(self) -> None:
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            row = self.probe(_probe_request(observation, pose, "exact"))["results"][0]
            receipt = self.act(observation, {"plan_ref": row.get("plan_ref")})
            judgement = judge_motion_case({"case_id": "probe_then_plan_ref_execution", "expected_status": {"reached"},
                                           "expected_strategy": "global", "expected_executor": GLOBAL_EXECUTOR}, receipt)
            notes = list(judgement["notes"])
            if not row.get("reachable"):
                notes.append("probe reported unreachable")
            if receipt.get("arms", {}).get(ARM, {}).get("plan_ref") != row.get("plan_ref"):
                notes.append("receipt does not bind the probed plan_ref")
            self.record("probe_then_plan_ref_execution", {"passed": not notes, "notes": notes}, receipt)
        except Exception as error:
            self.fail("probe_then_plan_ref_execution", error)
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            plan_ref = self.probe(_probe_request(observation, pose, "exact", offset=(0.06, 0.0, 0.0)))["results"][0].get("plan_ref")
            self.act(observation, {"targets": {ARM: "close"}}, "set_gripper")
            fresh = self.observe()
            receipt = self.act(fresh, {"plan_ref": plan_ref})
            passed = str(receipt.get("execution_status")) == "reached"
            self.record("plan_ref_survives_gripper_only",
                        {"passed": passed, "notes": [] if passed else ["plan did not survive set_gripper"]}, receipt)
        except Exception as error:
            self.fail("plan_ref_survives_gripper_only", error)
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            plan_ref = self.probe(_probe_request(observation, pose, "exact"))["results"][0].get("plan_ref")
            self.act(observation, _target({ARM: [pose[0], pose[1], pose[2] + 0.006, *pose[3:]]}, "precision"))
            stale = self.observe()
            try:
                self.act(stale, {"plan_ref": plan_ref})
                self.record("plan_ref_expires_after_motion", {"passed": False, "notes": ["stale plan executed"]}, None)
            except Exception:
                self.record("plan_ref_expires_after_motion", {"passed": True, "notes": ["stale plan_ref failed closed"]}, None)
        except Exception as error:
            self.fail("plan_ref_expires_after_motion", error, stage="setup")
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            variants = self.probe(_probe_request(observation, pose, "parallel_jaw_equivalents"))["results"][0].get("variants", [])
            passed = len(variants) == 2
            self.record("jaw_flip_variants_probed", {"passed": passed, "notes": [] if passed else [f"{len(variants)} variants"]},
                        None, variants=[{"variant_id": v.get("variant_id"), "reachable": v.get("reachable")} for v in variants])
        except Exception as error:
            self.fail("jaw_flip_variants_probed", error)
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            probed = self.probe({
                **_probe_request(observation, pose, "exact"),
                "queries": [{"query_id": "q0", "arm": ARM, "pose": _pose(UNREACHABLE_POSITION, pose[3:])}],
            })["results"][0]
            receipt = self.act(observation, _target({ARM: [*UNREACHABLE_POSITION, *pose[3:]]}, "free_space"))
            notes: list[str] = []
            if probed.get("reachable") is not False:
                notes.append("probe admitted the unreachable pose")
            if receipt.get("execution_status") != "planner_failed" or receipt.get("native_actions"):
                notes.append(f"execution ended {receipt.get('execution_status')} with {receipt.get('native_actions')} native actions")
            if receipt.get("failure_code") != PLANNER_IK_REFUSAL_CODE:
                # The probe and the executor must agree on *why*: an envelope
                # rejection would answer workspace_target_<axis> without ever
                # asking the planner, and prove nothing about reachability.
                notes.append(f"execution refused with {receipt.get('failure_code')!r}, not the planner's IK refusal")
            self.record("probe_consistency_unreachable", {"passed": not notes, "notes": notes}, receipt)
        except Exception as error:
            self.fail("probe_consistency_unreachable", error)

    def run_round_isolation(self) -> None:
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            stale_ref = self.probe(_probe_request(observation, pose, "exact"))["results"][0].get("plan_ref")
            terminal_before = self.status().get("terminal")
            reset = self.restart_round(1)
            notes: list[str] = []
            if reset.get("schema") != "ora.robodojo-round-reset.v1" or reset.get("round_index") != 1:
                notes.append(f"round reset record {reset.get('schema')} / {reset.get('round_index')}")
            if reset.get("seed") != 0 or reset.get("reward_registered") is not True:
                notes.append("round reset did not rebind seed 0 with a registered evaluator")
            if terminal_before is not True or self.status().get("terminal") is not False:
                notes.append("round reset did not revoke the previous round's terminal state")
            fresh = self.observe()
            if fresh.get("round_index") != 1 or fresh.get("previous_execution_receipt") is not None:
                notes.append("first observation of the new round carries previous-round state")
            distance, _angle = _pose_error(self.home_poses[ARM], self.current_poses(fresh)[ARM])
            if distance > 0.02:
                notes.append(f"round reset left the arm {distance:.3f} m from its round-0 home")
            try:
                self.act(fresh, {"plan_ref": stale_ref})
                notes.append("previous-round plan_ref executed")
            except Exception:
                pass
            fresh = self.observe()
            pose = self.current_poses(fresh)[ARM]
            receipt = self.act(fresh, _target({ARM: [pose[0], pose[1], pose[2] + 0.04, *pose[3:]]}, "free_space"))
            if receipt.get("round_index") != 1 or receipt.get("execution_status") != "reached":
                notes.append(f"round-1 move ended {receipt.get('execution_status')} in round {receipt.get('round_index')}")
            verdict = self.persist_verdict()
            if verdict.get("round_index") != 1 or verdict.get("native_actions") != receipt.get("native_actions"):
                notes.append("round-1 verdict counters did not restart with the round")
            self.record("round_reset_isolation", {"passed": not notes, "notes": notes}, receipt,
                        reset={key: reset.get(key) for key in ("schema", "round_index", "seed", "artifact_sequences_preserved")})
        except Exception as error:
            self.fail("round_reset_isolation", error)

    def run_terminal_boundary(self) -> None:
        try:
            holds = 0
            last_receipt: Mapping[str, Any] | None = None
            while not self.status().get("terminal") and holds < self.native_action_limit + 5:
                observation = self.observe()
                last_receipt = self.act(observation, {"targets": {ARM: "open"}}, "set_gripper")
                holds += 1
            terminal = self.status()
            verdict = self.persist_verdict()
            notes: list[str] = []
            if terminal.get("terminal") is not True or not str(terminal.get("reason", "")).startswith("official_"):
                notes.append(f"episode did not end on the official boundary: {terminal}")
            if verdict.get("terminal") is not True or not isinstance(verdict.get("official_success"), bool):
                notes.append(f"official verdict is not a terminal Boolean: {verdict.get('official_success')!r}")
            if verdict.get("native_actions") != self.native_action_limit and verdict.get("official_success") is not True:
                notes.append(f"native actions {verdict.get('native_actions')} differ from the task limit")
            if last_receipt is not None and last_receipt.get("terminal_after_action") is not True:
                notes.append("final receipt did not observe the terminal transition")
            try:
                observation = self.observe()
                self.act(observation, {"targets": {ARM: "open"}}, "set_gripper")
                notes.append("terminated environment accepted another action")
            except Exception:
                pass
            for _ in range(3):
                if self.status().get("terminal") is not True:
                    notes.append("terminal status changed on a read")
            after = self.persist_verdict()
            for key in ("official_success", "terminal", "verdict_basis", "fault_reason",
                        "native_actions", "physics_steps", "abort_reason"):
                if after.get(key) != verdict.get(key):
                    notes.append(f"post-terminal verdict changed: {key}")
            self.record("official_terminal_boundary_intact", {"passed": not notes, "notes": notes}, last_receipt,
                        holds=holds, verdict={key: verdict.get(key) for key in (
                            "terminal", "official_success", "native_actions", "physics_steps", "round_index")})
        except Exception as error:
            self.fail("official_terminal_boundary_intact", error)


def run_cases(
    *, observe: Callable[[], Mapping[str, Any]],
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    restart_round: Callable[[int], Mapping[str, Any]],
    status: Callable[[], Mapping[str, Any]],
    persist_verdict: Callable[[], Mapping[str, Any]],
    current_poses: Callable[[Mapping[str, Any]], Mapping[str, Sequence[float]]],
    native_action_limit: int,
    measure_jaw_state: Callable[[], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Drive every case through the injected live boundary.

    ``observe`` returns the fresh public observation with an added
    ``observation_sha256``; ``current_poses`` extracts both measured link6
    poses from it. Round 0 runs the motion matrix and the sequence cases and
    then drives the official native-action boundary to its terminal
    transition with hold commands; the round reset that a persistent session
    performs after that boundary is checked last, in round 1.
    """
    runner = _Runner(observe=observe, execute=execute, probe=probe, restart_round=restart_round,
                     status=status, persist_verdict=persist_verdict, current_poses=current_poses,
                     native_action_limit=native_action_limit,
                     measure_jaw_state=measure_jaw_state)
    runner.run_motion_matrix()
    runner.run_sequence()
    runner.run_terminal_boundary()
    runner.run_round_isolation()
    return runner.results


def _free_memory_mib(device: str) -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    rows = [line.split(",") for line in output.splitlines()]
    selected = [int(row[1]) for row in rows if row[0].strip() == device]
    if len(selected) != 1:
        raise RuntimeError(f"GPU {device} is not visible to nvidia-smi")
    return selected[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robodojo-motion-qualification",
        description=(
            "Run the scripted RoboDojo sparse-motion v3 qualification on the qualified host. "
            "Requires the explicit runtime configuration JSON (the compatibility package's "
            "runtime_configuration) and writes one device-keyed record."
        ),
    )
    parser.add_argument("--runtime-config", type=Path, required=True, help="RoboDojo runtime configuration JSON")
    parser.add_argument("--output", type=Path, required=True, help="evidence directory for the worker and record")
    parser.add_argument("--gate", type=Path, default=None,
                        help="optional device-keyed gate path receiving the identical record bytes")
    arguments = parser.parse_args(argv)
    runtime = json.loads(arguments.runtime_config.read_text(encoding="utf-8"))
    if os.environ.get("ORA_EXECUTION_HOST") != runtime.get("required_host"):
        raise SystemExit("the host marker must name the Runtime's RoboDojo host")
    device = str(runtime["cuda_visible_devices"])
    free_mib = _free_memory_mib(device)
    if free_mib < 20000:
        raise SystemExit(f"GPU {device} has {free_mib} MiB free; the RoboDojo worker needs 20000")
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))

    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
    from ora.environment.factory import build_robodojo_worker_session

    task = str(runtime["task_variant"])
    native_action_limit = NATIVE_ACTION_LIMITS[task]
    arguments.output.mkdir(parents=True, exist_ok=True)
    environment_root = arguments.output / "environment"
    session = build_robodojo_worker_session(
        runtime=runtime, output_root=environment_root, task=task, seed=0,
        max_sim_steps=native_action_limit * 10, action_codec="motion_action_v3",
    )
    observation_root = arguments.output / "observations"
    started = time.time()
    reset = dict(session.start())

    def observe() -> Mapping[str, Any]:
        observation, _images = session.observe(observation_root)
        # observation_sha256 is Runtime-layer identity: the hash of the
        # persisted observation.json, exactly as the context projector
        # computes it for live runs.
        manifest = observation_root / str(observation["observation_id"]) / "observation.json"
        return {**observation, "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    def current_poses(observation: Mapping[str, Any]) -> dict[str, list[float]]:
        return {arm: list(state["pose_xyz_xyzw"])
                for arm, state in observation["proprioception"]["end_effectors"].items()}

    def restart_round(index: int) -> Mapping[str, Any]:
        # The Supervisor rotates the previous round's verdict before a reset;
        # the gate keeps it as round evidence instead of discarding it.
        verdict = environment_root / "official-verdict.json"
        if verdict.exists():
            archive = environment_root / "rounds" / f"round-{index - 1:04d}" / "official-verdict.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            verdict.replace(archive)
        return session.restart_round(index, seed=0)

    close_record: dict[str, Any] = {}
    try:
        results = run_cases(
            observe=observe, execute=session.execute, probe=session.probe_motion,
            restart_round=restart_round,
            status=session.status, persist_verdict=session.persist_official_verdict,
            current_poses=current_poses, native_action_limit=native_action_limit,
            measure_jaw_state=session.measure_jaw_state,
        )
    finally:
        close_started = time.monotonic()
        session.close()
        close_record["close_seconds"] = round(time.monotonic() - close_started, 3)
        try:
            process = json.loads((environment_root / "environment-session.json").read_text(encoding="utf-8"))
            close_record["worker_returncode"] = process.get("worker_returncode")
            close_record["close_error"] = process.get("close_error")
        except (OSError, ValueError):
            close_record["worker_returncode"] = None
    record = build_qualification_record(
        source_commit=source_commit,
        host={
            "execution_host": os.environ.get("ORA_EXECUTION_HOST", "unknown"),
            "cuda_visible_devices": device, "free_memory_mib_at_start": free_mib,
            "device_binding": {key: value for key, value in (reset.get("device_binding") or {}).items()
                               if key != "startup_log_prefix"},
            "runtime_lease": None,
        },
        session_identity={
            "task": task, "eval_seed": 0, "layout_id": 0, "layout_file": runtime.get("layout_file"),
            "layout_sha256": runtime.get("layout_sha256"), "action_codec": "motion_action_v3",
            "native_action_limit": native_action_limit, "rounds": 2,
            "source_revisions": (reset.get("source_binding") or {}).get("source_revisions"),
            "source_manifest_sha256": (reset.get("source_manifest") or {}).get("sha256"),
            "source_dirty": source_dirty,
        },
        case_results=results,
        evidence={
            "environment_root": str(environment_root), "observation_root": str(observation_root),
            "wall_seconds": round(time.time() - started, 3), **close_record,
        },
    )
    body = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target = arguments.output / "robodojo-sparse-motion-qualification.json"
    target.write_text(body, encoding="utf-8")
    written = {"record": str(target)}
    # A gate path carries only passing admission evidence; failing runs stay
    # in their qualification directory so an old passing gate is never
    # overwritten by a later regression.
    if arguments.gate is not None and record["passed"]:
        arguments.gate.parent.mkdir(parents=True, exist_ok=True)
        arguments.gate.write_text(body, encoding="utf-8")
        written["gate"] = str(arguments.gate)
    print(json.dumps({
        "passed": record["passed"], "cases": len(record["cases"]),
        "failed": [case["case_id"] for case in record["cases"] if not case["passed"]],
        "missing": record["missing_cases"],
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), **written,
    }, indent=2))
    return 0 if record["passed"] else 1


__all__ = [
    "ALL_CASE_IDS", "JAW_APERTURE_CASE_ID", "JAW_APERTURE_PROBE_ID",
    "JAW_CALIBRATION_SCHEMA", "MOTION_CASES", "SCHEMA", "SEQUENCE_CASES",
    "build_case_arguments", "build_qualification_record", "derive_jaw_step",
    "jaw_aperture_calibration", "judge_gripper_case", "judge_jaw_aperture_sweeps",
    "judge_motion_case", "judge_recovery_case", "measured_pad_face_separation_m",
    "nominal_jaw_aperture_m", "run_cases", "summarize_receipt", "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised on the GPU host
    import sys as _sys

    _sys.exit(main())
