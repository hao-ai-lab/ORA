"""Bind the reviewed ARX card only after actual compiled-model qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition.embodiment import (
    audit_embodiment_sources, compile_embodiment_card, descriptor_activation_ready,
    load_embodiment_descriptor, validate_activation_record,
)
from ora.environment.controllers.robodojo_arx import controller_binding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARD_ROOT = PROJECT_ROOT / "ora/policy/embodiments/arx-x5"
#: The ARX descriptor the activation compiles and binds; its card lives in ``compiled-v1``.
DESCRIPTOR_PATH = CARD_ROOT / "descriptor-v1.json"
BINDING_SCHEMA = "ora.robodojo-embodiment-activation-binding.v1"


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"ARX embodiment activation rejected: {detail}")


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _immutable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
    path.chmod(0o444)


def create_activation(root: Path, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write separate review, Runtime evidence and activation records.

    Call only after both fresh-worker infrastructure probes complete. This
    function additionally reads their immutable reset/source/report files and
    compares initialized articulation and optical-prim facts with the exact
    compiled descriptor. A missing collector or an incomplete probe fails.
    """
    root = root.resolve()
    _require(len(episodes) == 2 and all(item.get("passed") is True for item in episodes),
             "two completed fresh-worker probes are required")
    descriptor_path = DESCRIPTOR_PATH
    card_path = CARD_ROOT / "compiled-v1/manual.md"
    manifest_path = CARD_ROOT / "compiled-v1/manifest.json"
    descriptor = load_embodiment_descriptor(descriptor_path)
    card, expected = compile_embodiment_card(descriptor)
    manifest = json.loads(manifest_path.read_text())
    _require(card_path.read_text() == card, "compiled card differs from descriptor")
    for key in ("descriptor_sha256", "card_sha256", "projected_facts_sha256"):
        _require(manifest[key] == expected[key], f"compiled manifest drift: {key}")
    audit = audit_embodiment_sources(descriptor, project_root=PROJECT_ROOT)
    _require(audit["passed"] is True, "source audit failed")
    external = next(item for item in descriptor["provenance"]["sources"]
                    if item["source_id"] == "robodojo_x5_robot_config")
    runtime_records = []
    for index, report in enumerate(episodes):
        episode_root = root / f"reset-{index}"
        reset_path = episode_root / "environment/reset.json"
        source_path = episode_root / "environment/source-manifest.json"
        report_path = episode_root / "probe-report.json"
        _require(json.loads(report_path.read_text()) == report, "persisted probe report differs")
        reset = json.loads(reset_path.read_text())
        source = json.loads(source_path.read_text())
        _require(reset == report["reset"], "reset report differs from worker evidence")
        _require(hashlib.sha256(source_path.read_bytes()).hexdigest() == reset["source_manifest"]["sha256"],
                 "source manifest hash drift")
        _require(source["loaded_files"]["Assets/Robots/x5/robot_config.yml"] == external["sha256"],
                 "loaded robot configuration differs from descriptor provenance")
        _require(source["source_revisions"]["assets"] == external["revision"], "asset revision drift")
        _require(reset["control_binding"] == controller_binding(), "control contract drift")
        runtime = reset.get("runtime_embodiment", {})
        _require(runtime.get("collector") == "initialized_isaac_articulation_and_optical_camera_prim",
                 "actual compiled-model collector is missing")
        _require(runtime.get("position_frame") == "episode_local_world" and runtime.get("quaternion_order") == "xyzw",
                 "observed frame convention differs")
        _require(set(runtime["arms"]) == {"left", "right"}, "dual-arm topology differs")
        for manipulator in descriptor["topology"]["manipulators"]:
            arm = runtime["arms"][manipulator["side"]]
            point = next(item for item in descriptor["control"]["controlled_points"]
                         if item["manipulator_id"] == manipulator["id"])
            _require(arm["fixed_base"] is True and len(arm["arm_joint_names"]) == manipulator["joint_dof"],
                     "compiled arm topology differs")
            _require(set(arm["arm_joint_names"] + arm["gripper_joint_names"]).issubset(arm["compiled_joint_names"]),
                     "declared joints are missing from initialized articulation")
            _require(arm["controlled_link"] == point["source_name"] and point["source_name"] in arm["compiled_body_names"],
                     "controlled point is missing from initialized articulation")
        _require(set(runtime["cameras"]) == {"head", "left_wrist", "right_wrist"}, "sensor inventory differs")
        for camera in descriptor["sensing"]["cameras"]:
            name = camera["public_name"]
            observed = runtime["cameras"][name]
            mount = None if name == "head" else f"robot{0 if name == 'left_wrist' else 1}/camera"
            _require(observed["source_name"] == camera["source_name"] and observed["mount_link"] == mount,
                     "camera attachment differs")
            for phase in ("initial_sensors", "repeated_sensors"):
                sensor = report[phase][name]
                _require(sensor["finite_positive_fraction"] > 0 and len(sensor["shape"]) == 2,
                         "qualified metric RGB-D evidence is missing")
        runtime_records.append({"reset": _reference(reset_path, root), "source_manifest": _reference(source_path, root),
                                "probe_report": _reference(report_path, root), "compiled_model": runtime})
    _require(runtime_records[0]["compiled_model"] == runtime_records[1]["compiled_model"],
             "compiled model drifted across fresh workers")
    review = {
        "schema": "ora.embodiment-card-review.v1", "decision": "accepted",
        "reviewer": "Codex source-and-wording review for RoboDojo compatibility",
        "card_sha256": expected["card_sha256"], "descriptor_sha256": expected["descriptor_sha256"],
        "scope": "two fixed 6-DoF arms, model link6 control points, three RGB-D views, Runtime-owned limits",
        "evidence_limits": "No physical jaw aperture, grasp, contact, collision-free path or task-success guarantee.",
    }
    review_path = root / "embodiment/review.json"
    evidence_path = root / "embodiment/runtime-evidence.json"
    activation_path = root / "embodiment/activation.json"
    _immutable(review_path, review)
    _immutable(evidence_path, {
        "schema": "ora.robodojo-embodiment-runtime-evidence.v1",
        "descriptor_sha256": expected["descriptor_sha256"],
        "projected_facts_sha256": expected["projected_facts_sha256"],
        "source_audit": audit, "fresh_worker_evidence": runtime_records,
        "controller_binding": controller_binding(),
        "claim": "Runtime embodiment agreement; no behavioral success claim",
    })
    activation = {
        "schema": "ora.embodiment-activation-record.v1",
        "descriptor_sha256": expected["descriptor_sha256"], "card_sha256": expected["card_sha256"],
        "review": {"decision": "accepted", "card_sha256": expected["card_sha256"],
                   "record_sha256": _reference(review_path, root)["sha256"]},
        "runtime_verification": {"schema": "ora.embodiment-runtime-verification.v1",
                                 "descriptor_sha256": expected["descriptor_sha256"],
                                 "projected_facts_sha256": expected["projected_facts_sha256"],
                                 "evidence_sha256": _reference(evidence_path, root)["sha256"]},
    }
    validate_activation_record(activation, descriptor=descriptor, card=card)
    _require(descriptor_activation_ready(descriptor, activation_record=activation), "shared activation validation failed")
    _immutable(activation_path, activation)
    return {"schema": BINDING_SCHEMA, "activation_ready": True,
            "descriptor": _reference(descriptor_path, PROJECT_ROOT),
            "compiled_card": _reference(card_path, PROJECT_ROOT),
            "compiled_manifest": _reference(manifest_path, PROJECT_ROOT),
            "activation_record": _reference(activation_path, root),
            "review_record": _reference(review_path, root), "runtime_evidence": _reference(evidence_path, root)}
