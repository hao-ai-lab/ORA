"""RoboDojo qualification probe episode: one bounded scene, sensor and motion check.

Split from the robodojo-compat qualification producer on 2026-09-28
(relocation U6); ``robodojo_card_qualification`` drives it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ora.environment.benchmarks.robodojo import source_revisions
from ora.environment.factory import build_environment_session
from ora.environment.rpc import EnvironmentRpcError
from ora.orchestrator.spec import json_sha256


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


# Host copies of worker facts. The host layer may not import the
# process-isolated RoboDojo backend, and the worker's own definitions sit in
# worker-kernel files that qualification records hash, so they are copied
# here rather than moved. tests/environments/robodojo/test_robodojo_operator.py checks each copy
# against the worker source.

#: ``backends/robodojo/motion_checks.py``: X5A.urdf's base_link->link6 chain
#: length, and the two fixed base mounts from ``env_cfg/robot/dual_x5.yml``.
X5_LINK6_REACH_BOUND_M = 0.822
X5_BASE_MOUNTS_M = ((-0.3, -0.45, 0.765), (0.3, -0.45, 0.765))


def workspace_violation(position: Any, bounds: Any) -> str | None:
    """The first axis on which ``position`` leaves ``bounds`` (``backends/robodojo/common.py``)."""
    _require(isinstance(bounds, Mapping), "workspace_bounds_m is required")
    for index, axis in enumerate(("x", "y", "z")):
        if not bounds[axis][0] <= float(position[index]) <= bounds[axis][1]:
            return axis
    return None


#: The commanded arm of the unreachable-pose probe.  ``dual_x5.yml`` bolts the
#: two fixed ARX X5 bases to the table's -y edge at (-/+0.3, -0.45, 0.765) and
#: the left one is the -x mount; the worker's ``motion_checks`` owns that pair and
#: the 0.822 m ``base_link``->``link6`` chain length that bounds how far link6
#: can stand from its own base at any joint vector.
PROBE_ARM = "left"
LEFT_X5_BASE_MOUNT_M = min(X5_BASE_MOUNTS_M, key=lambda mount: mount[0])


#: How far inside every ``workspace_bounds_m`` face the derived probe target
#: sits.  One centimetre is far larger than any float or serialization slack
#: on the envelope comparison and far smaller than the reach margins below.
UNREACHABLE_PROBE_MARGIN_M = 0.01


def unreachable_pose_probe_target(bounds: Mapping[str, Any]) -> list[float]:
    """Derive an in-envelope, provably out-of-reach link6 target.

    The probe exists to show that the *planner* refuses a pose it cannot
    solve, and that the refusal costs no native action.  That only holds if
    the request survives the hard action envelope: ``RobodojoSession.execute``
    raises ``pose target exceeds the qualified workspace`` before it ever
    calls ``solve_ik``, so a target outside ``workspace_bounds_m`` aborts the
    episode instead of exercising the IK.  The target is therefore derived
    from the Runtime this probe is actually handed rather than fixed to one
    envelope: on each axis take the bound face farther from the commanded
    arm's fixed base and step ``UNREACHABLE_PROBE_MARGIN_M`` inside it, which
    is the corner of the box diagonally opposite that mount.

    Both halves are then required, not assumed.  ``workspace_violation``
    confirms the envelope admits the point, and every mount must be farther
    from it than ``X5_LINK6_REACH_BOUND_M`` -- the sum of the URDF's
    ``base_link``->``link6`` translations, an upper bound on link6's distance
    from its base at *any* joint vector, so beyond it the pose is
    kinematically impossible rather than merely hard.  A Runtime whose
    envelope cannot host such a point has no valid probe and is refused here,
    before anything boots.

    Under ``robodojo_runtime_config``'s +/-1.5 m compatibility
    box this reproduces the package's previous constant ``[1.49, 1.49, 1.99]``
    exactly, so the compat operator surface is unchanged.  Under the
    source-owned ``robodojo_arx_x5_tabletop_v1`` envelope the Native default resolves
    (x [-0.75, 0.75], y [-0.65, 0.55], z [0.72, 1.40]) it yields
    ``[0.74, 0.54, 1.39]``, 1.566 m from the left mount and 1.251 m from the
    right one -- the same corner the RoboDojo sparse-motion gate settled on
    when it repaired this defect in its own refusal cases.
    """

    target: list[float] = []
    for index, axis in enumerate(("x", "y", "z")):
        low, high = (float(value) for value in bounds[axis])
        mount = float(LEFT_X5_BASE_MOUNT_M[index])
        far, inward = ((high, -1.0) if abs(high - mount) >= abs(mount - low)
                       else (low, 1.0))
        target.append(round(far + inward * UNREACHABLE_PROBE_MARGIN_M, 6))
    _require(workspace_violation(target, bounds) is None,
             f"the derived unreachable-pose probe target {target} is outside "
             "the action envelope, so it would be refused before the planner")
    _require(all(math.dist(target, mount) > X5_LINK6_REACH_BOUND_M
                 for mount in X5_BASE_MOUNTS_M),
             f"the derived unreachable-pose probe target {target} is within "
             f"{X5_LINK6_REACH_BOUND_M} m of an X5 mount, so an IK refusal is "
             "not provable")
    return target


def _sensor_evidence(observation: Mapping[str, Any], images: list[Path]) -> dict[str, Any]:
    _require(len(images) == 3, "three public cameras are required")
    report = {}
    for name, rgb_path in zip(("head", "left_wrist", "right_wrist"), images, strict=True):
        camera = observation["cameras"][name]
        root = rgb_path.parent.parent
        for reference in camera["files"].values():
            path = root / reference["path"]
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"], "RGB-D artifact hash drift")
        rgb = np.asarray(Image.open(rgb_path))
        depth = np.load(root / camera["files"]["depth_numpy_m_f32"]["path"], allow_pickle=False)
        calibration = json.loads((root / camera["files"]["calibration"]["path"]).read_text())
        intrinsic = np.asarray(calibration["intrinsic_cv"])
        transform = np.asarray(calibration["cam2world_cv"])
        inverse = np.asarray(calibration["extrinsic_cv"])
        valid = np.isfinite(depth) & (depth > 0)
        _require(rgb.shape == (*depth.shape, 3) and valid.any(), "unpaired or empty RGB-D")
        _require(np.isfinite(intrinsic).all() and intrinsic[0, 0] > 1 and intrinsic[1, 1] > 1,
                 "invalid focal calibration")
        _require(np.allclose(transform @ inverse, np.eye(4), atol=1e-6), "camera transforms are inconsistent")
        rendered = calibration["renderer_calibration"]
        _require(rendered["current_physical_pose_verified"] is True
                 and np.allclose(rendered["cam2world_cv"], transform, rtol=0, atol=1e-4),
                 "rendered optical pose is stale relative to the physical camera")
        _require(np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5),
                 "camera rotation is not orthonormal")
        _require(calibration["depth_unit"] == "metres" and calibration["depth_kind"] == "distance_to_image_plane",
                 "depth semantics differ from the admitted sensor profile")
        report[name] = {
            "shape": list(depth.shape), "finite_positive_fraction": float(valid.mean()),
            "depth_quantiles_m": np.quantile(depth[valid], [0, .5, .99, 1]).tolist(),
            "intrinsic_cv": intrinsic.tolist(), "cam2world_cv": transform.tolist(),
            "rgb_sha256": camera["files"]["rgb_lossless_png"]["sha256"],
            "rendering_time_s": rendered["simulation_time_s"],
            "reference_time": rendered["reference_time"],
        }
    return report


#: The command index after which the long probe stops proving anything new.
#: Indices 0-10 carry every substantive assertion: the current-pose IK
#: positive probes (0, 4, 6), the unreachable-target IK refusal with zero
#: native actions (2), gripper close and open holding pose (3, 5) and the four
#: moving-camera probes on both arms (7-10). Index 11 onward is a gripper hold
#: repeated until the task's native-action limit, which exists only to reach
#: the official terminal transition.
MOTION_PREFIX_COMMANDS = 11


def _episode(
    runtime: Mapping[str, Any], root: Path, *, scope: str = "card",
) -> dict[str, Any]:
    """Exercise current-pose/current-gripper holds and one unreachable pose.

    ``scope`` selects how far the same probe runs.

    ``card`` is the long form, unchanged: it drives the episode to the task's
    own native-action limit, which is the only way to witness the official
    terminal transition, and it is what a card record is measured from.

    ``task`` stops after ``MOTION_PREFIX_COMMANDS`` -- every substantive
    assertion -- and instead spends its remaining time on the two facts that
    are genuinely per scene and that the long form never establishes: the
    scene/evaluator registration this layout produces, and whether a *second
    reset of the same live plant* keeps its object identities. It does not
    reach the official terminal boundary and does not claim to.
    """
    if scope not in {"card", "task"}:
        raise RuntimeError(f"unknown infrastructure probe scope {scope!r}")
    environment_root = root / "environment"
    native_limit = int(runtime["native_action_limit"])
    # Resolve the planner-refusal probe against this Runtime's own envelope
    # first: a Runtime that cannot host a provably unreachable in-envelope
    # target has no valid probe, and finding that out is free before boot.
    unreachable = unreachable_pose_probe_target(runtime["workspace_bounds_m"])
    # This bootstrap uses the same factory/transport and explicit lease, while
    # the formal RunSpec separately requires this completed qualification.
    session = build_environment_session(
        runtime_config=runtime, output_root=environment_root, task=runtime["task_variant"], seed=0,
        max_sim_steps=native_limit * 10, action_codec="runtime_action_v3",
    )
    # Each named check is set only where its own assertions have actually
    # completed. They were previously one blanket flag written after the fact,
    # which meant the five names carried no independent evidence; a layered
    # contract cannot say which layer proves what unless they are separate.
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {"passed": False, "scope": scope, "actions": [],
                              "checks": checks,
                              "unreachable_probe_target_m": unreachable,
                              "probe_arm": PROBE_ARM}
    try:
        reset = dict(session.start())
        report["reset"] = reset
        registration = reset["scene_registration"]
        _require(registration["native_action_limit"] == native_limit,
                 "the environment registered another task's native-action limit")
        _require(any(registration["evaluator_registration"].values()),
                 "the official evaluator registered no success check list")
        report["scene_registration"] = registration
        checks["scene_registration"] = True
        checks["evaluator_registration"] = True
        worker_pid = session.delegate.process.pid
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        worker_gpus = sorted({
            row.split(",")[1].strip() for row in processes.splitlines()
            if len(row.split(",")) == 2 and row.split(",")[0].strip() == str(worker_pid)
        })
        _require(worker_gpus == [reset["device_binding"]["selected_gpu_uuid"]],
                 "actual worker GPU residency differs from the selected CUDA/renderer identity")
        report["worker_device_residency"] = {
            "worker_pid": worker_pid, "gpu_uuids": worker_gpus,
            "captured_unix_s": time.time(), "source": "nvidia-smi query-compute-apps",
        }
        _require(reset["reward_registered"] is True, "reset did not register the official evaluator")
        if scope == "card":
            # Lifecycle facts about the worker, not about this scene: they
            # also write official-verdict.json, which ``restart_round``
            # refuses to reset past, so the task scope leaves them to the card
            # record that already proves them on this exact installation.
            initial_verdict = session.persist_official_verdict()
            _require(initial_verdict["official_success"] is None and initial_verdict["terminal"] is False,
                     "preterminal eligibility was exposed as a task verdict")
        observation, images = session.observe(root / "observations")
        report["initial_sensors"] = _sensor_evidence(observation, images)
        repeated, repeated_images = session.observe(root / "observations")
        report["repeated_sensors"] = _sensor_evidence(repeated, repeated_images)
        if scope == "card":
            _require(session.persist_official_verdict()["physics_steps"] == 0, "observing advanced action physics")
        # Measure control invariants through worker receipts; physical contact
        # may change joint measurements even when targets remain unchanged.
        observation = repeated
        command_index = 0
        total_native = 0
        home_poses = {arm: list(state["pose_xyz_xyzw"])
                      for arm, state in observation["proprioception"]["end_effectors"].items()}
        motion_camera_checks = []
        command_ceiling = (native_limit + 3 if scope == "card"
                           else MOTION_PREFIX_COMMANDS)
        while not session.is_terminal and command_index < command_ceiling:
            left = observation["proprioception"]["end_effectors"]["left"]
            moving_arm = "right" if command_index in {9, 10} else "left"
            if command_index in {7, 8, 9, 10}:
                kind = "move_pose"
                pose = home_poses[moving_arm]
                position = list(pose[:3])
                if command_index in {7, 9}:
                    position[2] += 0.06
                arguments = {"arm": moving_arm, "position_m": position, "orientation_xyzw": pose[3:]}
            elif command_index in {0, 2, 4, 6}:
                kind = "move_pose"
                pose = left["pose_xyz_xyzw"]
                position = list(unreachable) if command_index == 2 else pose[:3]
                arguments = {"arm": PROBE_ARM, "position_m": position, "orientation_xyzw": pose[3:]}
            else:
                kind = "set_gripper"
                opening = {3: 0.0, 5: 1.0}.get(command_index, left["gripper_command"])
                arguments = {"arm": "left", "gripper": opening}
            action = {
                "schema": "ora-action.v3", "observation_id": observation["observation_id"],
                "action_id": f"infrastructure-{command_index:04d}", "action": kind,
                "arguments": arguments, "bound_observation_sha256": json_sha256(observation),
            }
            result = session.execute(action)
            receipt = result["receipt"]
            _require(receipt["physics_steps"] == 10 * receipt["native_actions"], "native/physics cadence drift")
            _require(0 <= receipt["native_actions"] <= 20, "per-command native budget exceeded")
            if command_index in {0, 4, 6}:
                _require(receipt["motion_status"] == "reached", "current-pose IK/control positive probe did not converge")
            if command_index == 2:
                _require(receipt["motion_status"] == "ik_failed" and receipt["native_actions"] == 0,
                         "unreachable-pose probe did not expose an action-free IK failure")
                # The envelope check raises before ``solve_ik`` is called, so
                # pin the SDK's own answer and the untouched retained target:
                # neither an envelope refusal nor a pose the arm can actually
                # reach may ever pass as a planner refusal.
                _require(np.allclose(receipt["commanded_pose_xyz_xyzw"][:3], unreachable,
                                     rtol=0, atol=1e-9),
                         "unreachable-pose probe did not command its derived target")
                _require(receipt["ik_status"] not in {"Success", "not_requested"},
                         f"unreachable-pose probe never reached the SDK IK: {receipt['ik_status']!r}")
                _require(receipt["pose_targets_before"] == receipt["pose_targets_after"],
                         "a refused unreachable pose still wrote a retained pose target")
            else:
                _require(receipt["native_actions"] >= 1, "a hold command failed to advance its native control tick")
            total_native += receipt["native_actions"]
            _require(total_native <= native_limit, "official action limit was exceeded")
            before_targets, after_targets = receipt["command_targets_before"], receipt["command_targets_after"]
            _require(receipt["command_invariant"]["verified"] is True, "worker command-preservation check failed")
            other_arm = "left" if moving_arm == "right" else "right"
            held_keys = [f"{other_arm}_arm_joint_state", f"{other_arm}_ee_joint_state"]
            held_keys.append(f"{moving_arm}_ee_joint_state" if kind == "move_pose" else f"{moving_arm}_arm_joint_state")
            for key in held_keys:
                _require(before_targets[key] == after_targets[key], f"action changed a held command: {key}")
            trace = json.loads((environment_root / receipt["control_trace"]["path"]).read_text())
            samples = [json.loads(line) for line in
                       (environment_root / trace["physics_journal"]["path"]).read_text().splitlines()]
            _require(len(samples) == receipt["physics_steps"], "SDK target journal missed physics steps")
            for sample in samples:
                for key, target in sample["command"].items():
                    applied = sample["applied_position_targets"][key]
                    _require(np.allclose(applied, target["position"], rtol=0, atol=1e-6),
                             "SDK targets differ from the recorded command")
                    if key in held_keys:
                        _require(np.allclose(applied, before_targets[key]["position"], rtol=0, atol=1e-6),
                                 f"SDK target journal changed a held dimension: {key}")
            if kind == "set_gripper":
                _require(receipt["pose_targets_before"] == receipt["pose_targets_after"], "gripper action changed retained pose targets")
            _require(receipt["before"][other_arm]["gripper_command"] == receipt["after"][other_arm]["gripper_command"],
                     "action changed the other gripper directive")
            if kind == "move_pose":
                _require(receipt["before"][moving_arm]["gripper_command"] == receipt["after"][moving_arm]["gripper_command"],
                         "pose-only action changed its gripper directive")
            report["actions"].append({"index": command_index, **result})
            command_index += 1
            if command_index % 25 == 0:
                print(f"{root.name}: {total_native}/{native_limit} native actions checked", flush=True)
            if not session.is_terminal:
                observation, images = session.observe(root / "observations")
                sensors = _sensor_evidence(observation, images)
                if command_index - 1 in {7, 8, 9, 10}:
                    displacement = float(np.linalg.norm(np.asarray(receipt["reached_pose_xyz_xyzw"][:3])
                                                        - np.asarray(receipt["before_pose_xyz_xyzw"][:3])))
                    _require(receipt["motion_status"] == "reached" and displacement >= 0.03,
                             "moving-camera probe did not execute a distinct physical pose")
                    motion_camera_checks.append({"index": command_index - 1, "arm": moving_arm,
                                                 "displacement_m": displacement, "sensors": sensors})
        _require(command_index >= MOTION_PREFIX_COMMANDS and len(motion_camera_checks) == 4,
                 "official termination prevented completion of the moving-camera probes")
        report["gripper_transition_probes"] = True
        report["applied_sdk_target_checks"] = True
        report["moving_camera_probes"] = motion_camera_checks
        checks["motion_contract" if scope == "card" else "scene_motion_prefix"] = True
        if scope == "task":
            # The only fact a second reset of a *live* plant can establish,
            # and the one the long two-fresh-worker probe never touches:
            # upstream ``SceneManager.reload_env_scene`` regenerates instance
            # names from LayoutManager's monotone counter when it reloads a
            # saved layout, and ``bridge._keep_layout_identities`` works
            # around it. That workaround only runs on this path, so whether it
            # holds for this scene is measured here or nowhere.
            round_reset = dict(session.restart_round(1))
            repeated_registration = round_reset["scene_registration"]
            _require(repeated_registration == registration,
                     "a repeated reset of the same plant did not preserve this "
                     "scene's object identities")
            report["round_reset"] = round_reset
            report["repeated_reset_scene_registration"] = repeated_registration
            checks["repeated_reset_identity"] = True
            observation, images = session.observe(root / "observations")
            report["repeated_reset_sensors"] = _sensor_evidence(observation, images)
        synchronization = [json.loads(line) for line in
                           (environment_root / "render-synchronization.jsonl").read_text().splitlines()]
        _require(synchronization and all(row["matched"] for row in synchronization),
                 "a rendered observation did not match the physical simulation time")
        report["render_synchronization_checks"] = {
            "observations": len(synchronization), "all_matched": True,
            "max_render_calls": max(len(row["samples"]) for row in synchronization),
        }
        checks["rgbd_calibration" if scope == "card" else "scene_rgbd_calibration"] = True
        if scope == "card":
            verdict = session.persist_official_verdict()
            _require(verdict["terminal"] is True and type(verdict["official_success"]) is bool,
                     "the official environment did not provide its terminal verdict")
            _require(verdict["native_actions"] == total_native and verdict["physics_steps"] == total_native * 10,
                     "final native/physics counters differ from the receipts")
            # The fixed hold probe checks the native timeout, without requiring
            # a task success. An unexpectedly successful probe is recorded
            # verbatim.
            report["native_timeout_exercised"] = total_native == native_limit
            for _ in range(3):
                _require(session.status()["terminal"] is True, "terminal state changed on a status read")
            try:
                session.execute(action)
            except EnvironmentRpcError:
                report["post_terminal_action_rejected"] = True
            else:
                raise RuntimeError("terminated environment accepted another action")
            after = session.persist_official_verdict()
            for key in ("official_success", "terminal", "native_actions", "physics_steps", "abort_reason"):
                _require(after[key] == verdict[key], f"postterminal verdict changed: {key}")
            report["verdict"] = after
            checks["official_terminal"] = True
        report["passed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        start_close = time.monotonic()
        try:
            session.close()
            first_record = (environment_root / "environment-session.json").read_bytes()
            session.close()
            _require((environment_root / "environment-session.json").read_bytes() == first_record,
                     "repeated close changed exit evidence")
            report["close_seconds"] = time.monotonic() - start_close
            process = json.loads(first_record)
            report["close_record"] = process
            report["clean_exit"] = process.get("worker_returncode") == 0 and process.get("close_error") is None
            report["bounded_close"] = report["close_seconds"] <= 60 and process.get("worker_returncode") is not None
            _require(report["bounded_close"], "worker cleanup exceeded its independent bound")
            checks["bounded_close"] = True
        except BaseException as exc:
            report["passed"] = False
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _write(root / "probe-report.json", report)
    return report


def _episode_binding(episodes: list[Mapping[str, Any]], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """What two independent fresh workers on one card must agree on."""

    first, second = [item["reset"] for item in episodes]
    for key in ("task_instruction", "actual_source_fingerprint", "control_binding",
                "runtime_embodiment"):
        _require(first[key] == second[key], f"repeated-reset binding drift: {key}")
    for reset in (first, second):
        _require(reset["source_binding"]["source_revisions"] == source_revisions(runtime),
                 "fresh-worker source revisions drifted")
    device = {key: value for key, value in first["device_binding"].items() if key != "startup_log_prefix"}
    _require(device == {key: value for key, value in second["device_binding"].items() if key != "startup_log_prefix"},
             "fresh workers did not use the same actual GPU identities")
    return {"device_binding": device, "task_instruction": first["task_instruction"],
            "actual_source_fingerprint": first["actual_source_fingerprint"],
            "runtime_embodiment": first["runtime_embodiment"]}


def _measured_checks(episodes: list[Mapping[str, Any]], names) -> dict[str, bool]:
    """Only names every episode actually measured, never a blanket flag."""

    return {name: all(item["checks"].get(name) is True for item in episodes)
            for name in names}
