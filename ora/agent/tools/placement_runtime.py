"""Public-state binding and evidence adapter for read-only placement tools.

Numerical proposals live in placement.py. This adapter owns source/freshness
checks, observed action-stage evidence and immutable review artifacts. It has
no EnvironmentSession and never submits an action.
"""

from __future__ import annotations

from ora.agent.tools.placement_review import (
    _review_image,
)

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ora.agent.composition.tool_surface import ToolSurfaceError, validate_schema_instance
from ora.agent.evidence.receipt_binding import (
    public_motion_arguments_match,
    public_worker_arguments_match,
)
from ora.contracts import target_state_phases, validate_runtime_action
from .grasp import (
    DEPTH_INLIER_GEOMETRY_CAPABILITY, TARGET_REF_PATTERN, _segmentation_source_identity,
    load_segmented_target, segmented_target_geometry,
)
from .placement_receipts import _command_states, _stage_receipt
from .robot import ToolError, _append
from . import motion_arguments, perception_feedback, placement_arguments, tool_names


CAPABILITY = "public_rgbd_placement_support_v1"
DUAL_CAPABILITY = "public_rgbd_dual_arm_placement_support_v1"
ASSESSMENT_PATTERN = re.compile(r"place-check-[0-9a-f]{32}")
CLAIM = (
    "Agent-selected public regions and correspondences, not verified semantic "
    "identity; observed surfaces only, not hidden shape, complete containment, "
    "contact, stability, release permission, attachment, reachability, "
    "collision-free motion or official success. No world mutation."
)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist(path: Path, value: Any) -> None:
    data = _bytes(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ToolError("placement evidence identity drift", code="artifact_hash_mismatch", retryable=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def measurements(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Project numerical diagnostics into a closed, source-free public shape."""
    result: list[dict[str, Any]] = []
    omitted = {"observed_points_world_m", "occupied_cells", "voxel_points_world_m"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key not in omitted:
                result.extend(measurements(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (list, tuple)):
        if value and all(type(x) in (int, float) and math.isfinite(x) for x in value):
            if len(value) > 16:
                raise ValueError("unbounded placement diagnostic vector")
            result.append({"name": prefix, "scalar": None, "vector": list(value), "text": None})
        else:
            for index, item in enumerate(value):
                result.extend(measurements(item, f"{prefix}[{index}]"))
    elif type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("non-finite placement diagnostic")
        result.append({"name": prefix, "scalar": value, "vector": None, "text": None})
    elif isinstance(value, (str, bool)) or value is None:
        result.append({"name": prefix, "scalar": None, "vector": None,
                       "text": None if value is None else str(value)})
    else:
        raise ValueError("unsupported placement diagnostic")
    return result


def public_geometry(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """The projected geometry block, including the held-object offset.

    ``assess_geometry`` computes ``object_visible_offset_from_tcp_world_m`` as a
    sibling of ``geometry`` rather than a member of it, so projecting
    ``assessment["geometry"]`` alone left it in the audit record and out of every
    result. It is the only delivered quantity that states where the visible held
    object sits relative to the controlled point, and the policy cannot derive it
    from its own channels; a release residual is otherwise unattributable.

    Read defensively: an assessment written before this projection carries no such
    key, and its absence must not fail the call.
    """

    geometry = dict(assessment.get("geometry") or {})
    offset = assessment.get("object_visible_offset_from_tcp_world_m")
    if offset is not None:
        geometry["object_visible_offset_from_tcp_world_m"] = offset
    return geometry


def stage_evidence(
    states: list[Mapping[str, Any]], stage: str, *, run_root: Path | None = None,
    arm: str = "right",
) -> dict[str, Any]:
    """A stage is corroborated by consumed public receipts, not its argument.

    ``place_plan`` is declared only by the RoboTwin dual-arm placement surface,
    so the stage reads RoboTwin's sparse command stream.
    """
    return _robotwin_stage_evidence(states, stage, run_root=run_root, arm=arm)


def _robotwin_receipt_arguments_match(action: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    """Share the canonical argument join; action identity is checked separately."""
    return public_worker_arguments_match(action, receipt)


def _robotwin_stage_evidence(states: list[Mapping[str, Any]], stage: str, *,
                            run_root: Path | None, arm: str) -> dict[str, Any]:
    """Interpret the selected sparse-v7 command stream, not physical release.

    A RoboTwin gripper action completes as ``executed`` rather than Cartesian
    ``reached``. Opposite-arm writes neither release this arm nor constitute
    its exit, although their fresh observations invalidate numeric proposals.
    """
    if arm not in {"left", "right"}:
        raise ToolError("placement requires an explicit left or right arm", code="invalid_arguments", field="arm")
    directive, action_id, motions, last_status = "unknown", None, 0, None
    incomplete = False
    for state in _command_states(states, run_root):
        action = state.get("previous_public_action")
        if action is None:
            continue
        receipt = _stage_receipt(state.get("previous_execution_receipt"), run_root)
        if (state.get("_skill_command") is True and isinstance(receipt, Mapping)
            and receipt.get("schema") == "ora.robotwin-motion-receipt.v1"
            and action.get("schema") == "ora-action.v3"
            and receipt.get("action_id") == action.get("action_id")
            and receipt.get("arguments") == action.get("arguments")
            and receipt.get("action") == action.get("action")
            and receipt.get("consumed") is True):
            # Historical legacy skill commands remain command evidence, never
            # evidence of the measured jaw opening or correct public-TCP arrival.
            if action["arguments"].get("arm") != arm:
                continue
            name = action["action"]
            if name == "set_gripper" and receipt.get("state_changed") is True:
                selected = {0.0: "close", 1.0: "open"}.get(action["arguments"].get("gripper"))
                if selected is not None:
                    directive, action_id, motions, last_status = selected, action["action_id"], 0, None
                    continue
            elif name == "move_pose":
                last_status = "unknown"
                if directive == "open" and receipt.get("state_changed") is True:
                    motions += 1
                continue
            incomplete = True
            continue
        if (not isinstance(action, Mapping) or not isinstance(receipt, Mapping)
            or receipt.get("schema") != "ora.robotwin-sparse-motion-receipt.v7"
            or receipt.get("action_id") != action.get("action_id")
            or receipt.get("action") != action.get("action")
            or not _robotwin_receipt_arguments_match(action, receipt)
            or receipt.get("consumed") is not True):
            incomplete = True
            continue
        if action.get("action") == "wait" and receipt.get("target_arms") == []:
            continue  # the shared hold names no arm and changes no command
        selected = receipt.get("target_arms")
        targets = action.get("arguments", {}).get("targets")
        if (not isinstance(selected, list) or not selected
            or len(set(selected)) != len(selected) or not set(selected) <= {"left", "right"}
            or isinstance(targets, Mapping) and set(selected) != set(targets)):
            incomplete = True
            continue
        if arm not in selected:
            continue
        effect = receipt.get("arms", {}).get(arm)
        if not isinstance(effect, Mapping) or effect.get("targeted") is not True:
            incomplete = True
            continue
        name, status = action.get("action"), receipt.get("execution_status")
        if name == "set_gripper":
            requested = targets.get(arm) if isinstance(targets, Mapping) else None
            actual = effect.get("command_preservation", {}).get("gripper_command_target_after")
            if (requested not in {"open", "close"} or actual != requested
                or receipt.get("state_changed") is not True or status != "executed"):
                incomplete = True
                directive, action_id, motions = "unknown", None, 0
                continue
            directive, action_id, motions = requested, action["action_id"], 0
            last_status = None
        elif name in {"move_pose", "move_joints", "servo_delta", "follow_joint_path", "recover_motion"}:
            last_status = status
            if status not in {"reached", "not_reached", "planner_failed", "plan_rejected"}:
                incomplete = True
            elif directive == "open" and receipt.get("state_changed") is True:
                if status in {"reached", "not_reached"}:
                    motions += 1
                else:
                    incomplete = True
        else:
            incomplete = True
    matched = (stage == "pre_release" and directive == "close"
               or stage == "post_release" and directive == "open" and motions == 0
               or stage == "post_exit" and directive == "open" and motions > 0)
    status = "unknown" if incomplete or directive == "unknown" else ("confirmed" if matched else "contradicted")
    return {"status": status,
            "reason": "consumed_command_history_only_not_physical_release" if status == "confirmed" else (
                "public_action_receipt_history_incomplete" if status == "unknown" else
                "requested_stage_disagrees_with_consumed_action_history"),
            "latest_gripper_action_id": action_id, "latest_gripper_directive": directive,
            "post_open_pose_actions": motions, "last_pose_execution_status": last_status}


def _timeline(host: Any, control: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The placement tools' round timeline; the reader is ``saved_steps.round_timeline``."""
    from .saved_steps import round_timeline

    return round_timeline(host, control, subject="placement")


def _target(host: Any, target_ref: str, *, role: str, states: list[dict[str, Any]],
            round_start: int, current: bool) -> dict[str, Any]:
    manifest, mask, _overlay = load_segmented_target(artifact_root=host.artifact_root, target_ref=target_ref)
    identity = manifest["identity"]
    matching = [state for state in states[round_start:]
                if state["observation"]["id"] == identity["observation_id"]]
    if len(matching) != 1 or current and matching[0]["step"] != states[-1]["step"]:
        raise ToolError("placement target is stale or belongs to another round", code="stale_observation", field=role)
    state = matching[0]
    _step, selected, root, observation = host._select_observation(state["step"])
    if selected is None or selected["state_sha256"] != state["state_sha256"]:
        raise ToolError("placement target state identity drift", code="artifact_hash_mismatch", retryable=False)
    camera = identity["camera"]
    _record, rgb, depth, calibration = host._camera(root, observation, camera)
    actual = _segmentation_source_identity(observation_id=identity["observation_id"], camera=camera,
        rgb_path=rgb, depth_path=depth, calibration_path=calibration)
    legacy = {key + "_sha256": actual["files"][key]["sha256"] for key in ("rgb", "depth", "calibration")}
    if identity.get("source") != actual and identity.get("source") != legacy:
        raise ToolError("placement target RGB-D source identity drift", code="artifact_hash_mismatch", retryable=False)
    from .placement_depth import CAPABILITY as depth_capability, project_geometry
    if host.surface.supports(depth_capability) and role in {"object", "support", "boundary"}:
        import numpy as np
        from .placement_refusal import RetainedPointBoundsError, retained_point_error
        calibration_value = json.loads(calibration.read_text())
        try:
            geometry = project_geometry(np.load(mask, allow_pickle=False), np.load(depth, allow_pickle=False),
                calibration_value["intrinsic_cv"], calibration_value["extrinsic_cv"],
                observation_id=identity["observation_id"], target_ref=target_ref, camera=camera)
        except RetainedPointBoundsError as error:
            raise retained_point_error(error, role=role, camera=camera) from error
        geometry["depth_boundary_evidence"].update(source_mask_sha256=_hash(mask),
            **{"source_" + key: value for key, value in legacy.items()})
        if host.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY):
            # Keep every H-retained placement point. The correctness source
            # contributes input-quality diagnostics, not a second silhouette
            # filter that could silently discard a handle or corner.
            quality_geometry = segmented_target_geometry(
                artifact_root=host.artifact_root, target_ref=target_ref,
                depth_path=depth, calibration_path=calibration, voxel_size_m=0.005,
                use_depth_inliers=True, rgb_path=rgb)
            for key in ("status", "reason", "source_quality", "quality_minimums",
                        "quality_failures", "semantic_identity_verified"):
                geometry[key] = quality_geometry[key]
            # This binding describes the separate quality sample, not the
            # full H-retained placement point cloud above.
            geometry["quality_source_binding"] = quality_geometry["source_binding"]
    else:
        geometry = segmented_target_geometry(artifact_root=host.artifact_root, target_ref=target_ref,
            depth_path=depth, calibration_path=calibration, voxel_size_m=0.005,
            use_depth_inliers=host.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY), rgb_path=rgb)
    return {"geometry": geometry, "state": state, "rgb": rgb, "depth": depth,
            "calibration": calibration, "mask": mask, "manifest": manifest,
            "source": {"role": role, "target_ref": target_ref, "camera": camera, **legacy}}


def rim_points(rim: Mapping[str, Any] | None, target: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Back-project explicit current rim samples; never substitute an AABB axis."""
    if rim is None:
        return None, "rim_samples_not_supplied"
    if rim["camera"] != target["source"]["camera"]:
        raise ToolError("rim samples must use the object reference's current source camera", code="invalid_arguments", field="rim.camera")
    import numpy as np
    depth = np.load(target["depth"], allow_pickle=False)
    mask = np.load(target["mask"], allow_pickle=False).astype(bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("placement rim source dimensions disagree")
    calibration = json.loads(target["calibration"].read_text())
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=float)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=float)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
    if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
        or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
        or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("placement rim calibration is invalid")
    if len({tuple(point) for point in rim["points_uv"]}) != len(rim["points_uv"]):
        raise ToolError("rim samples must be distinct current pixels", code="invalid_arguments", field="rim.points_uv")
    camera_to_world = np.linalg.inv(extrinsic)
    points = []
    for x, y in rim["points_uv"]:
        if x >= depth.shape[1] or y >= depth.shape[0]:
            raise ToolError("rim pixel is outside its current camera", code="invalid_arguments", field="rim.points_uv")
        # Small image-space association tolerance, not geometric completion.
        near = mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        if not near.any():
            return None, "rim_sample_not_near_selected_object_mask"
        z = float(depth[y, x])
        if not math.isfinite(z) or not 1e-6 < z <= 2.5:
            return None, "rim_depth_unavailable"
        camera_point = [(x - intrinsic[0, 2]) * z / intrinsic[0, 0],
                        (y - intrinsic[1, 2]) * z / intrinsic[1, 1], z, 1.0]
        points.append((camera_to_world @ camera_point)[:3].tolist())
    return points, None


def invoke_placement(host: Any, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not host.surface.supports(CAPABILITY) or not host.surface.supports(DUAL_CAPABILITY):
        raise ToolError("placement support is not selected", code="tool_unavailable", retryable=False)
    definition = tool_names.definition_for(host.surface.definitions, name)
    try:
        validate_schema_instance(dict(arguments), definition["inputSchema"])
        destination = placement_arguments.destination(arguments)
        protected = placement_arguments.protected_pairs(arguments)
        refs = [arguments["object_ref"], destination["support_ref"]]
        if "boundary_ref" in destination:
            refs.append(destination["boundary_ref"])
        for pair in protected:
            refs.extend((pair["before_target_ref"], pair["current_target_ref"]))
        if any(not TARGET_REF_PATTERN.fullmatch(ref) for ref in refs):
            raise ValueError("placement references must be canonical target references")
        if "rim" in arguments and any(value < 0 for pixel in arguments["rim"]["points_uv"] for value in pixel):
            raise ValueError("rim pixel coordinates must be non-negative integers")
        if "prior_assessment_ref" in arguments and not ASSESSMENT_PATTERN.fullmatch(arguments["prior_assessment_ref"]):
            raise ValueError("malformed placement assessment reference")
    except (ToolSurfaceError, ValueError, TypeError, KeyError) as error:
        raise ToolError(str(error), code="invalid_arguments") from error
    try:
        return _invoke(host, name, arguments, destination, protected)
    except ToolError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise ToolError(str(error), code="placement_evidence_unavailable", retryable=True) from error


def _invoke(host: Any, name: str, arguments: Mapping[str, Any],
            destination: Mapping[str, str],
            protected_refs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    from . import placement
    from .placement_depth import CAPABILITY as depth_capability
    depth_active = host.surface.supports(depth_capability)
    claim = CLAIM + (" Depth-boundary pixels remain uncertain, not certified background; numerical "
        "candidates are conditional on unresolved silhouette, never a full-object fit certificate."
        if depth_active else "")
    control, _root, observation, _pose, _gripper = host._current_grip_state()
    arm = arguments["arm"]
    pose = host._surface_arm_states(observation)[arm]
    states, round_start = _timeline(host, control)
    specs = [("object", arguments["object_ref"]), ("support", destination["support_ref"])]
    if "boundary_ref" in destination:
        specs.append(("boundary", destination["boundary_ref"]))
    from . import placement_refusal
    duplicate = placement_refusal.duplicate_reference_error(specs)
    if duplicate is not None:
        raise duplicate
    targets = [_target(host, ref, role=role, states=states, round_start=round_start, current=True) for role, ref in specs]
    protected = []
    for index, pair in enumerate(protected_refs):
        before = _target(host, pair["before_target_ref"], role=f"protected_{index}_before",
                         states=states, round_start=round_start, current=False)
        current = _target(host, pair["current_target_ref"], role=f"protected_{index}_current",
                          states=states, round_start=round_start, current=True)
        if before["state"]["step"] >= current["state"]["step"]:
            raise ToolError("protected object comparison requires an earlier state in this round", code="invalid_arguments")
        protected.append({"before_geometry": before["geometry"], "current_geometry": current["geometry"]})
        targets.extend((before, current))
    points, rim_reason = rim_points(arguments.get("rim"), targets[0])
    # The handler names this tool canonically; the surface may advertise either
    # spelling. The assessment identity, the event and the result-schema key all
    # use the spelling the policy was shown -- the collector joins the first two
    # by equality -- so they resolve once here rather than three times.
    published = str(tool_names.definition_for(host.surface.definitions, name)["name"])
    phase = arguments["phase"]
    stage_name = "post_release" if phase == "exit" else "pre_release"
    stage = stage_evidence(states[round_start:], stage_name, run_root=host.state_root.parent, arm=arm)
    kwargs = {"destination_kind": destination["kind"], "tcp_pose": pose,
              "rim_points_world": points if arguments["orientation_requirement"] == "upright" else None,
              "protected_geometries": protected}
    boundary = targets[2]["geometry"] if "boundary_ref" in destination else None
    computation = placement.propose_placement(targets[0]["geometry"], targets[1]["geometry"], boundary,
        phase=phase, gripper_open=stage["latest_gripper_directive"] == "open" and (
            perception_feedback.enabled(host.surface) or stage["status"] == "confirmed"), **kwargs)
    if stage["status"] != "confirmed":
        computation.update(status="unknown", reason="requested_phase_command_history_unconfirmed")
        if not perception_feedback.enabled(host.surface):
            computation["candidates"] = []
    assessment = computation["assessment"]
    if arguments["orientation_requirement"] == "none":
        assessment["checks"]["orientation"] = {"status": "consistent", "reason": "orientation_not_requested", "metrics": {}, "thresholds": {}}
    elif rim_reason:
        assessment["checks"]["orientation"] = {"status": "unknown", "reason": rim_reason, "metrics": {}, "thresholds": {}}
    quality_failures = []
    if host.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY):
        for target in targets:
            geometry = target["geometry"]
            role = target["source"]["role"]
            quality = geometry["source_quality"]
            partial_object = (role == "object" or role.startswith("protected_")) and quality["touches_image_boundary"]
            if geometry["status"] == "unknown" or partial_object:
                quality_failures.append({
                    "role": role, "target_ref": geometry["target_ref"],
                    "source_quality": quality, "quality_failures": geometry["quality_failures"],
                    "partial_object_comparability_unknown": bool(partial_object),
                })
        if quality_failures:
            # Preserve the measured diagnostics, but none of the dependent
            # placement checks or proposals can outrank uncertain inputs.
            for check in assessment["checks"].values():
                check.update(status="unknown", reason="selected_mask_depth_or_object_comparability_insufficient")
                check.setdefault("metrics", {})["input_quality_failures"] = quality_failures
            assessment["status"] = "unknown"
            computation.update(status="unknown", reason="selected_mask_depth_or_object_comparability_insufficient")
            computation["candidates"] = []
    artifact_root = host.artifact_root / "placement"
    prior = None
    if "prior_assessment_ref" in arguments:
        ref = arguments["prior_assessment_ref"]
        if not ASSESSMENT_PATTERN.fullmatch(ref):
            raise ToolError("malformed placement assessment reference", code="invalid_arguments")
        prior_path = artifact_root / f"{ref}.json"
        if prior_path.is_symlink():
            raise ToolError("placement reference is not an immutable file", code="artifact_hash_mismatch")
        prior = json.loads(prior_path.read_text())
        if "place-check-" + hashlib.sha256(_bytes(prior)).hexdigest()[:32] != ref:
            raise ToolError("placement assessment content identity drift", code="artifact_hash_mismatch", retryable=False)
        identity = prior["identity"]
        if (identity["run_id"] != states[-1]["run_id"] or identity["round_start_step"] != round_start
            or not identity["state_step"] < states[-1]["step"]
            or identity.get("arguments", {}).get("arm") != arm):
            raise ToolError("prior assessment must precede this observation in the same round", code="stale_observation")
        before = prior["object_geometry"]["centroid_world_m"]
        current = targets[0]["geometry"]["centroid_world_m"]
        assessment["checks"]["region"].setdefault("metrics", {})["prior_visible_centroid_shift_m"] = math.dist(before, current)
    sources = [target["source"] for target in targets]
    record = {"schema": "ora.place-assessment-evidence.v2" if depth_active else
              "ora.place-assessment-evidence.v1", "identity": {
        "run_id": states[-1]["run_id"], "round_start_step": round_start, "state_step": states[-1]["step"],
        "state_sha256": states[-1]["state_sha256"], "observation_id": control["observation_id"],
        "observation_sha256": control["observation_sha256"], "tool": published, "arguments": dict(arguments)},
        "object_geometry": targets[0]["geometry"], "selected_regions": sources,
        "rim_points_world_m": points, "stage_evidence": stage, "computation": computation,
        "claim_boundary": claim}
    if host.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY):
        record["geometry_inputs"] = [{"role": target["source"]["role"],
            "geometry": target["geometry"]} for target in targets]
    if depth_active:
        record["depth_boundary_projections"] = {target["source"]["role"]:
            target["geometry"]["depth_boundary_evidence"] for target in targets
            if "depth_boundary_evidence" in target["geometry"]}
    # Retained so the observation can re-show this offset with its age. A policy
    # that stops calling place.plan otherwise loses the number entirely: the
    # sealed 2026-09-11 LIBERO-10 round two never called it once and fell back
    # to re-grounding both objects after every move.
    offset = assessment.get("object_visible_offset_from_tcp_world_m")
    if offset is not None:
        host.last_held_object_offset = {
            "frame": "world",
            "translation_m": [float(v) for v in offset],
            "basis": "visible_object_points_median_minus_tcp_position",
            "observation_id": control["observation_id"],
            "observation_step": int(states[-1]["step"]),
            "claim_boundary": (
                "Derived from the visible points of that observation and the "
                "controlled-point position then. It is not a measurement of the "
                "object pose, and it only stays valid while the object moves "
                "with the gripper."
            ),
        }
    assessment_ref = "place-check-" + hashlib.sha256(_bytes(record)).hexdigest()[:32]
    path = artifact_root / f"{assessment_ref}.json"
    _persist(path, record)
    overlay = artifact_root / f"{assessment_ref}.png"
    _review_image(overlay, targets=targets, computation=computation, rim=arguments.get("rim"),
                  phase=f"{phase} / arm: {arm}", stage=stage)
    checks = {key: {"status": value["status"], "reason": value["reason"],
                   "measurements": measurements({"metrics": value.get("metrics", {}), "thresholds": value.get("thresholds", {})})}
              for key, value in assessment["checks"].items()}
    result = {"schema": "ora.place-plan.v1",
        "status": computation["status"],
        "reason": computation.get("reason", "visible_geometry_checks_do_not_establish_physical_placement"),
        "observation_id": control["observation_id"], "observation_sha256": control["observation_sha256"],
        "assessment_ref": assessment_ref, "selected_regions": sources, "stage_evidence": stage,
        "checks": checks, "geometry_measurements": measurements(public_geometry(assessment)),
        "evidence": {"path": path.relative_to(host.control_root.parent).as_posix(), "sha256": _hash(path),
                     "overlay_path": overlay.relative_to(host.control_root.parent).as_posix(), "overlay_sha256": _hash(overlay)},
        "claim_boundary": claim}
    result.update(arm=arm, schema="ora.robotwin-place-plan.v1")
    interaction = {"preplace": "free_space", "lower": "contact_approach", "exit": "precision"}[phase]
    # These candidates are published to be copied into move_pose unchanged,
    # so they are written in whichever spelling that surface advertises.
    # A copyable bundle its own surface refuses reads as an instruction and
    # fails at the schema gate before the handler ever sees it.
    move_pose_schema = next(
        (item.get("inputSchema", {}) for item in host.surface.definitions
         if tool_names.canonical(str(item.get("name"))) == "move_pose"), {},
    )

    # Every phase has a copyable pose. Closed execution additionally
    # consumes lower-phase IDs; preplace and exit have no ID executor.

    def _bundle(target_pose: Mapping[str, Any]) -> dict[str, Any]:
        body = motion_arguments.arm_target_fields(
            move_pose_schema,
            position={"mode": "absolute", "position_m": target_pose["position_m"]},
            orientation={
                "mode": "absolute_quaternion",
                "quaternion_xyzw": target_pose["orientation_xyzw"],
            },
            interaction=interaction,
        )
        if motion_arguments.declares_flat_targets(move_pose_schema):
            return {"targets": {arm: body}}
        # The nested writer returns position/orientation objects; the
        # canonical bare pose is what frozen surfaces published here.
        return {"targets": {arm: {**target_pose, "interaction": interaction}}}

    result.update(phase=phase, candidates=[{
        "candidate_id": candidate["candidate_id"],
        "move_pose": _bundle(candidate["target_pose"]),
        "measurements": measurements(candidate.get("metrics", {})), "claim_boundary": claim,
    } for candidate in computation["candidates"]])
    if perception_feedback.enabled(host.surface):
        result["geometry_status"] = assessment["status"]
        result["candidate_prerequisites"] = [
            "selected_gripper_open" if phase == "exit" else "selected_gripper_closed",
            "same_observation_selected_visible_geometry",
            "hidden_shape_and_contact_remain_unverified",
        ]
        result["uncertainty_reasons"] = list(dict.fromkeys(
            [check["reason"] for check in checks.values() if check["status"] == "unknown"]
            + ([stage["reason"]] if stage["status"] != "confirmed" else [])))
    # result_schemas is keyed by what this surface published and is snapshot
    # verbatim into the RunPlan, so the lookup moves to that spelling rather
    # than the mapping being normalised.
    validate_schema_instance(result, host.surface.result_schemas[published])
    _append(host.event_path, {"type": "placement_assessment", "tool": published,
        "observation_id": control["observation_id"], "assessment_ref": assessment_ref,
        "result": result})
    result["images_base64"] = [base64.b64encode(overlay.read_bytes()).decode("ascii")]
    return result
