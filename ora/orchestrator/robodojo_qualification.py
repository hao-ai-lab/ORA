"""Cross-domain admission for RoboDojo infrastructure and policy embodiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robodojo import (
    CARD_QUALIFICATION_CHECKS, CARD_QUALIFICATION_SCHEMA,
    QUALIFICATION_CHECKS, QUALIFICATION_SCHEMA, REVISIONS,
    TASK_QUALIFICATION_CHECKS, TASK_QUALIFICATION_SCHEMA, TASK_UNPROVEN_FACTS,
    WORKER_PACKAGE_VERSIONS,
    RoboDojoBenchmarkError, card_configuration_binding, configuration_binding,
    qualification_binding_layers, source_revisions, task_configuration_binding,
    validate_scene_registration,
)

#: The measured facts a task record carries so a pairing with the wrong card,
#: a different worker environment or drifted worker-side source content is
#: refused rather than inherited. The task probe measures every one of them in
#: its own two fresh workers, and admission compares them with the card
#: record's -- so the join is evidence against evidence, not a copied header.
CARD_JOIN_FIELDS = (
    "worker_kernel_sha256", "source_revisions", "cuda_visible_devices",
    "sensor_profile", "device_binding", "environment",
    "actual_source_fingerprint", "card_configuration", "runtime_embodiment",
)
#: The one ``loaded_files`` entry that is a per-task fact rather than a
#: per-card one. The worker hashes the selected task's own config at boot, so
#: an otherwise card-scoped fingerprint carries the task it was measured on.
#: A card record measured on ``hang_mugs`` would therefore pair with no other
#: task's record at all -- the layering would admit exactly the task it was
#: measured on, which is the whole thing it exists to avoid. ``_task.yml`` is
#: the shared base and stays a card fact.
TASK_CONFIG_PREFIX = "task/RoboDojo/config/"
TASK_CONFIG_SHARED = f"{TASK_CONFIG_PREFIX}_task.yml"
#: The fingerprint members that are per-task facts, not per-card ones, and so
#: are projected out of the card join and asserted against the record's own
#: task instead. ``resolved_environment_config_sha256`` hashes the resolved
#: environment configuration, which names the selected task, so it moves with
#: the task exactly as the loaded task config does.
TASK_SCOPED_FINGERPRINT_FIELDS = ("resolved_environment_config_sha256",)
#: What ``load_layered_qualification`` returns beside the two records, so a
#: caller reads the same top-level keys a combined record exposes.
LAYERED_SCHEMA = "ora.robodojo-layered-qualification.v1"


from .robodojo_kernel import (
    WORKER_KERNEL_PATHS, ROBODOJO_MOTION_KERNEL_PATHS, worker_kernel_extension,
    worker_kernel_sha256,
)


# Same bound as the worker's own readback (graphics_compat.READBACK_TIMEOUT_S).
GRAPHICS_READBACK_TIMEOUT_S = 120


def inspect_graphics_compatibility(python: Path, *, runner=subprocess.run) -> dict[str, Any]:
    """Read the process-local repair in a credential-free external interpreter."""
    from ora.environment.rpc import _worker_environment

    root = Path(__file__).resolve().parents[2]
    script = root / "ora/environment/backends/robodojo/graphics_compat.py"
    try:
        process = runner(
            [str(python), str(script), str(python.parent.parent.parent)],
            cwd=root, env=_worker_environment(root), text=True,
            capture_output=True, timeout=GRAPHICS_READBACK_TIMEOUT_S, check=False,
        )
        if process.returncode != 0:
            raise ValueError(process.stderr.strip() or "external readback failed")
        value = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(value, dict):
            raise ValueError("external readback is not an object")
        return value
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        raise RoboDojoBenchmarkError(f"Vulkan driver compatibility readback failed: {exc}") from exc


def validate_runtime_driver(version: str, *, graphics_compatibility: Mapping[str, Any] | None = None) -> None:
    """Require a verified finite-limit repair for 595.x; live qualification follows."""
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version) is None:
        raise RoboDojoBenchmarkError("NVIDIA driver version is unavailable or malformed")
    if version.startswith("595."):
        value = graphics_compatibility or {}
        before, after = value.get("before"), value.get("after")
        files = value.get("files", {})
        if (value.get("schema") != "ora.robodojo-graphics-compatibility.v1"
                or value.get("enabled") is not True
                or value.get("max_memory_allocation_size") != 0xFFE00000
                or value.get("source_revision") != "a400d0b10059b2cc3cd285475593eee79975a5c4"
                or not isinstance(before, list) or not before
                or not isinstance(after, list) or len(before) != len(after)
                or not isinstance(files, Mapping) or set(files) != {"layer", "profile", "manifest", "patch"}):
            raise RoboDojoBenchmarkError(f"NVIDIA driver {version} requires verified Vulkan allocation-limit compatibility")
        for reference in files.values():
            if (not isinstance(reference, Mapping)
                    or re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", ""))) is None):
                raise RoboDojoBenchmarkError("Vulkan driver compatibility source fingerprint is invalid")
        for original, repaired in zip(before, after, strict=True):
            if (not isinstance(original, Mapping) or not isinstance(repaired, Mapping)
                    or repaired.get("max_memory_allocation_size") != 0xFFE00000
                    or {k: v for k, v in original.items() if k != "max_memory_allocation_size"}
                    != {k: v for k, v in repaired.items() if k != "max_memory_allocation_size"}):
                raise RoboDojoBenchmarkError("Vulkan driver compatibility changed device identity or failed to repair the limit")


def _read_record(
    reference: Mapping[str, Any], *, workspace_root: Path, label: str,
) -> tuple[dict[str, Any], Path]:
    """Read one hash-bound record out of the shared RoboDojo workspace."""

    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("outputs", "robodojo-compat"):
        raise RoboDojoBenchmarkError(f"{label} must live in the shared RoboDojo output workspace")
    path = (workspace_root / relative).resolve()
    if not path.is_relative_to((workspace_root / "outputs" / "robodojo-compat").resolve()):
        raise RoboDojoBenchmarkError(f"{label} path escapes its workspace")
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, ValueError) as exc:
        raise RoboDojoBenchmarkError(f"{label} is unavailable") from exc
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise RoboDojoBenchmarkError(f"{label} hash mismatch")
    if not isinstance(value, dict):
        raise RoboDojoBenchmarkError(f"{label} is not a record")
    return value, path


def _validate_record_identity(
    value: Mapping[str, Any], *, schema: str, harness_commit: str, label: str,
) -> dict[str, Any]:
    """Schema, pass, commit lineage and worker-kernel content, for any layer.

    Content, not commit identity. ``harness_commit`` stays recorded lineage --
    which commit the live probes actually ran at -- while admission compares
    the worker kernel's content at that measurement against its content at the
    RunPlan's own commit. An unrelated commit between measurement and launch
    (documentation, an experiment budget, another backend) leaves the digest
    untouched; any edit inside ``WORKER_KERNEL_PATHS`` changes it. A record
    written before this binding existed carries no digest and is refused
    rather than silently accepted. When only the manifest grew (a file added
    to the list, not edited), the record is admitted exactly when every
    measured and every added file is byte-identical between its measurement
    commit and the plan commit (``worker_kernel_extension``).

    Returns how the kernel was admitted, for the caller to report beside the
    record: ``mode`` is ``identical_digest`` or
    ``manifest_extension_byte_identical``, with both commits, both digests and
    the added paths. After an extension the recorded digest no longer equals
    the plan digest, and this is where a reader sees why it was accepted.
    """

    if value.get("schema") != schema or value.get("passed") is not True:
        raise RoboDojoBenchmarkError(f"{label} did not pass")
    lineage = value.get("harness_commit")
    if not isinstance(lineage, str) or re.fullmatch(r"[0-9a-f]{40}", lineage) is None:
        raise RoboDojoBenchmarkError(f"{label} records no Harness commit lineage")
    recorded_kernel = value.get("worker_kernel_sha256")
    if not isinstance(recorded_kernel, str) or re.fullmatch(r"[0-9a-f]{64}", recorded_kernel) is None:
        raise RoboDojoBenchmarkError(
            f"{label} records no RoboDojo worker kernel content hash"
        )
    plan_kernel = worker_kernel_sha256(harness_commit)
    added: tuple[str, ...] | None = ()
    if recorded_kernel != plan_kernel:
        added = worker_kernel_extension(
            recorded_kernel, measured_commit=lineage, plan_commit=harness_commit,
        )
    if added is None:
        raise RoboDojoBenchmarkError(
            f"RoboDojo worker kernel changed since its {label} ({lineage})"
        )
    return {
        "mode": "manifest_extension_byte_identical" if added else "identical_digest",
        "measured_commit": lineage, "plan_commit": harness_commit,
        "recorded_worker_kernel_sha256": recorded_kernel,
        "plan_worker_kernel_sha256": plan_kernel, "added_paths": list(added),
    }


def _validate_source_fingerprint(fingerprint: Any, *, label: str) -> None:
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != {
        "loaded_files", "installed_packages", "resolved_environment_config_sha256",
    }:
        raise RoboDojoBenchmarkError(f"{label} actual source fingerprint is missing")
    loaded = fingerprint["loaded_files"]
    packages = fingerprint["installed_packages"]
    if (
        not isinstance(loaded, Mapping) or not loaded
        or any(
            not isinstance(name, str) or not name
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in loaded.items()
        )
        or not isinstance(packages, Mapping)
        or any(packages.get(name) != version for name, version in WORKER_PACKAGE_VERSIONS.items())
        or not isinstance(fingerprint["resolved_environment_config_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint["resolved_environment_config_sha256"]) is None
    ):
        raise RoboDojoBenchmarkError(f"{label} actual source fingerprint is invalid")


def _validate_environment_readback(environment: Any, *, label: str) -> None:
    if not isinstance(environment, Mapping) or set(environment) != {
        "python_executable", "python_binary_sha256", "versions",
    }:
        raise RoboDojoBenchmarkError(f"{label} interpreter readback is missing")
    versions = environment["versions"]
    if (
        not isinstance(environment["python_executable"], str)
        or not Path(environment["python_executable"]).is_absolute()
        or not isinstance(environment["python_binary_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", environment["python_binary_sha256"]) is None
        or not isinstance(versions, Mapping)
        or any(versions.get(name) != version for name, version in WORKER_PACKAGE_VERSIONS.items())
        or any(not isinstance(name, str) or not name or not isinstance(version, str) or not version
               for name, version in versions.items())
    ):
        raise RoboDojoBenchmarkError(f"{label} interpreter readback is invalid")


def _validate_checks(value: Mapping[str, Any], names, *, label: str) -> None:
    checks = value.get("checks", {})
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in names):
        raise RoboDojoBenchmarkError(f"{label} checks are incomplete")


def _validate_task_instruction(value: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(value.get("task_instruction"), str) or not value["task_instruction"].strip():
        raise RoboDojoBenchmarkError(f"{label} task instruction is missing")


def _validate_compiled_model(model: Any, *, label: str) -> None:
    """The initialized articulation/optical-prim facts both layers read back.

    The card layer proves this compiled model agrees with the tracked ARX
    descriptor, through the full embodiment activation. The task layer
    re-reads it in its own workers and must land on the same object, so a task
    record measured against a different installation cannot pair.
    """

    if (
        not isinstance(model, Mapping)
        or model.get("collector") != "initialized_isaac_articulation_and_optical_camera_prim"
        or not isinstance(model.get("arms"), Mapping)
        or set(model["arms"]) != {"left", "right"}
        or not isinstance(model.get("cameras"), Mapping)
        or set(model["cameras"]) != {"head", "left_wrist", "right_wrist"}
    ):
        raise RoboDojoBenchmarkError(f"{label} compiled embodiment readback is invalid")


def load_infrastructure_qualification(
    runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str,
) -> dict[str, Any]:
    """Verify combined infrastructure evidence, without a behavioral oracle.

    This is the single-record contract: one 35-56 minute measurement that
    pins the card, the installation, the task and the scene together. It is
    unchanged, it is what ``robodojo_card_qualification.qualify`` still
    produces, and it stays the reference for what the two
    layers must jointly prove. Layout, device, sensor profile, configuration
    binding, external source revisions, worker interpreter and package
    fingerprint, task language, device UUID and embodiment activation are all
    as strict as before.
    """

    value, path = _read_record(
        runtime["infrastructure_qualification"], workspace_root=workspace_root,
        label="infrastructure qualification",
    )
    admission = _validate_record_identity(
        value, schema=QUALIFICATION_SCHEMA, harness_commit=harness_commit,
        label="infrastructure qualification",
    )
    if value.get("source_revisions") != source_revisions(runtime):
        raise RoboDojoBenchmarkError("infrastructure qualification external source drift")
    expected = {
        "task": runtime["task_variant"], "eval_seed": runtime["eval_seed"],
        "layout_id": runtime["layout_id"], "layout_file": runtime["layout_file"],
        "layout_sha256": runtime["layout_sha256"],
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RoboDojoBenchmarkError("infrastructure qualification layout/device/sensor drift")
    if value.get("configuration") != configuration_binding(runtime):
        raise RoboDojoBenchmarkError("infrastructure qualification control/configuration drift")
    _validate_source_fingerprint(
        value.get("actual_source_fingerprint"), label="infrastructure qualification",
    )
    _validate_task_instruction(value, label="infrastructure qualification")
    _validate_environment_readback(value.get("environment"), label="infrastructure qualification")
    _validate_checks(value, QUALIFICATION_CHECKS, label="infrastructure qualification")
    _validate_device_binding(value.get("device_binding"), runtime=runtime)
    _validate_embodiment_activation(
        value.get("embodiment_activation"), root=path.parent, device_binding=value["device_binding"],
    )
    return {**value, "kernel_admission": admission}


def load_card_qualification(
    runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit the per-card, per-installation half of the same evidence.

    Everything the combined record proves about the card and the installation
    and nothing about the task's own scene: the GPU identity the run will read
    back, the worker interpreter and package fingerprint, the worker-side
    loaded-file digests, the external source revisions, the RoboDojo worker
    kernel's content, the card half of the configuration binding, the sensor
    profile, and the embodiment activation against the tracked ARX card.

    ``measured_task`` records which task the card's own full-length episodes
    actually drove to its native-action limit. It is deliberately *not*
    compared with the RunPlan's task: reusing this record across tasks is the
    whole point. It is kept, reported and documented so the operator can see
    which task's boundary the ``official_terminal`` check was witnessed on.
    """

    binding = reference if reference is not None else (
        runtime["infrastructure_qualification"]["card"]
    )
    value, path = _read_record(
        binding, workspace_root=workspace_root, label="card qualification",
    )
    admission = _validate_record_identity(
        value, schema=CARD_QUALIFICATION_SCHEMA, harness_commit=harness_commit,
        label="card qualification",
    )
    if value.get("source_revisions") != source_revisions(runtime):
        raise RoboDojoBenchmarkError("card qualification external source drift")
    expected = {
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RoboDojoBenchmarkError("card qualification device/sensor drift")
    if value.get("card_configuration") != card_configuration_binding(runtime):
        raise RoboDojoBenchmarkError("card qualification control/configuration drift")
    _validate_source_fingerprint(value.get("actual_source_fingerprint"), label="card qualification")
    _validate_environment_readback(value.get("environment"), label="card qualification")
    _validate_checks(value, CARD_QUALIFICATION_CHECKS, label="card qualification")
    _validate_compiled_model(value.get("runtime_embodiment"), label="card qualification")
    measured = value.get("measured_task")
    if (
        not isinstance(measured, Mapping)
        or not isinstance(measured.get("task"), str) or not measured["task"]
        or type(measured.get("native_action_limit")) is not int
        or measured.get("native_timeout_exercised") is not True
    ):
        # The one fact the whole 35-56 minute length buys. A card record that
        # did not actually reach a task's official native-action limit has not
        # witnessed the terminal transition, so it may not carry
        # ``official_terminal`` for any task that rests on it.
        raise RoboDojoBenchmarkError(
            "card qualification did not drive any task to its official "
            "native-action limit, so it does not establish the official "
            "terminal boundary"
        )
    _validate_device_binding(value.get("device_binding"), runtime=runtime)
    _validate_embodiment_activation(
        value.get("embodiment_activation"), root=path.parent,
        device_binding=value["device_binding"],
    )
    return {**value, "kernel_admission": admission}


def load_task_qualification(
    runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str,
    card: Mapping[str, Any], card_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit the per-task, per-scene half and its binding to one card record.

    The task record must name *this* card record by path and hash, and every
    field in ``CARD_JOIN_FIELDS`` -- which the task probe measured itself, in
    its own two fresh workers -- must equal the card record's. So a task
    record paired with a different card, measured against a different worker
    interpreter or package set, or produced while the worker-side loaded files
    or the RoboDojo worker kernel had drifted, is refused rather than carried
    by the card's evidence.

    What it proves about the scene is exactly what one boot establishes: the
    layout the cell names and its content hash, every layout instance
    resolving to a scene object, the official evaluator registering its check
    lists, this task's own registered native-action limit read back at reset,
    the upstream public instruction, RGB-D calibration inside this scene, the
    eleven-command motion prefix, and one identity-preserving repeated reset
    of a live plant. What it explicitly does not prove is named in
    ``not_established``.
    """

    value, _path = _read_record(
        runtime["infrastructure_qualification"]["task"],
        workspace_root=workspace_root, label="task qualification",
    )
    admission = _validate_record_identity(
        value, schema=TASK_QUALIFICATION_SCHEMA, harness_commit=harness_commit,
        label="task qualification",
    )
    if value.get("source_revisions") != source_revisions(runtime):
        raise RoboDojoBenchmarkError("task qualification external source drift")
    expected = {
        "task": runtime["task_variant"], "eval_seed": runtime["eval_seed"],
        "layout_id": runtime["layout_id"], "layout_file": runtime["layout_file"],
        "layout_sha256": runtime["layout_sha256"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RoboDojoBenchmarkError("task qualification layout drift")
    if value.get("task_configuration") != task_configuration_binding(runtime):
        raise RoboDojoBenchmarkError("task qualification task/scene configuration drift")
    _validate_task_instruction(value, label="task qualification")
    _validate_checks(value, TASK_QUALIFICATION_CHECKS, label="task qualification")
    # The disclosure is part of the contract, not commentary: a task record
    # that drops or edits it cannot be admitted, so no consumer can read a
    # layered pair without being told what the task layer did not establish.
    if value.get("not_established") != TASK_UNPROVEN_FACTS:
        raise RoboDojoBenchmarkError(
            "task qualification does not carry the exact record of what the "
            "task layer leaves unproven"
        )
    registration = validate_scene_registration(
        value.get("scene_registration"),
        native_action_limit=runtime["native_action_limit"],
    )
    repeated = validate_scene_registration(
        value.get("repeated_reset_scene_registration"),
        native_action_limit=runtime["native_action_limit"],
    )
    if repeated != registration:
        # Upstream ``SceneManager.reload_env_scene`` regenerates instance
        # names from LayoutManager's monotone counter on a later reset of the
        # same plant, and the bridge's ``_keep_layout_identities`` works
        # around it. A round-1 registration that differs from round 0 means
        # that workaround did not hold for this scene, and every later round
        # of a K-round run would address objects the evaluator cannot resolve.
        raise RoboDojoBenchmarkError(
            "a repeated reset of the same plant did not preserve this scene's "
            "object identities"
        )
    join = value.get("card_join")
    if not isinstance(join, Mapping) or set(join) != set(CARD_JOIN_FIELDS):
        raise RoboDojoBenchmarkError("task qualification carries no card join")
    _validate_compiled_model(join.get("runtime_embodiment"), label="task qualification")
    named = value.get("card_qualification")
    if not isinstance(named, Mapping) or set(named) != {"path", "sha256"} or {
        "path": str(named.get("path")), "sha256": str(named.get("sha256")),
    } != {"path": str(card_reference["path"]), "sha256": str(card_reference["sha256"])}:
        raise RoboDojoBenchmarkError(
            "task qualification was measured against a different card record"
        )
    measured = card.get("measured_task")
    if not isinstance(measured, Mapping) or not isinstance(measured.get("task"), str):
        raise RoboDojoBenchmarkError("card qualification names no measured task")
    card_measured_task = str(measured["task"])
    drifted = sorted(
        field for field in CARD_JOIN_FIELDS
        if _joined(field, join[field]) != _joined(field, card.get(field))
    )
    if drifted:
        raise RoboDojoBenchmarkError(
            "task qualification does not pair with its card qualification: "
            f"{drifted}"
        )
    # The per-task entry projected out above is not discarded: each record must
    # carry exactly its own task's config and no other, so the projection can
    # never hide a fingerprint measured on the wrong scene.
    for label, record, task in (
        ("task", value, str(value.get("task", ""))),
        ("card", card, card_measured_task),
    ):
        fingerprint = (
            record.get("card_join", {}).get("actual_source_fingerprint")
            if label == "task" else record.get("actual_source_fingerprint")
        )
        loaded = (fingerprint or {}).get("loaded_files")
        if not isinstance(loaded, Mapping):
            raise RoboDojoBenchmarkError(
                f"{label} qualification records no loaded worker files"
            )
        selected = sorted(name for name in loaded if _is_task_config(name))
        if selected != [f"{TASK_CONFIG_PREFIX}{task}.yml"]:
            raise RoboDojoBenchmarkError(
                f"{label} qualification loaded {selected} for task {task!r}"
            )
        for name in TASK_SCOPED_FINGERPRINT_FIELDS:
            digest = (fingerprint or {}).get(name)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise RoboDojoBenchmarkError(
                    f"{label} qualification records no {name}"
                )
    return {**value, "kernel_admission": admission}


def _is_task_config(name: str) -> bool:
    """Whether one ``loaded_files`` key is the selected task's own config."""

    return name.startswith(TASK_CONFIG_PREFIX) and name != TASK_CONFIG_SHARED


def _joined(field: str, value: Any) -> Any:
    """Project one card-join field down to what is genuinely a card fact.

    Only ``actual_source_fingerprint`` needs projecting, and only its
    ``loaded_files``: the worker hashes the selected task's config there, so
    comparing it whole would make a card record pair with exactly one task.
    Every other field is compared unchanged.
    """

    if field != "actual_source_fingerprint" or not isinstance(value, Mapping):
        return value
    loaded = value.get("loaded_files")
    if not isinstance(loaded, Mapping):
        return value
    projected = {
        key: item for key, item in value.items()
        if key not in TASK_SCOPED_FINGERPRINT_FIELDS
    }
    projected["loaded_files"] = {
        name: digest for name, digest in loaded.items()
        if not _is_task_config(name)
    }
    return projected


def load_layered_qualification(
    runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str,
) -> dict[str, Any]:
    """Admit the card/task pair and project it as one qualification view.

    Fails closed if either layer is missing, drifted or mismatched. The result
    carries the same top-level keys a combined record exposes -- so preflight's
    interpreter, task-language, device and graphics comparisons are unchanged
    -- plus both records and the explicit ``not_established`` disclosure.
    """

    reference = runtime["infrastructure_qualification"]
    card = load_card_qualification(
        runtime, workspace_root=workspace_root, harness_commit=harness_commit,
        reference=reference["card"],
    )
    task = load_task_qualification(
        runtime, workspace_root=workspace_root, harness_commit=harness_commit,
        card=card, card_reference=reference["card"],
    )
    return {
        "schema": LAYERED_SCHEMA, "passed": True,
        "layers": ["card", "task"],
        "card_qualification": card, "task_qualification": task,
        # The combined view every consumer already reads. The card supplies
        # the installation and device half; the task supplies its own scene
        # half and the official language the run is bound to.
        "harness_commit": card["harness_commit"],
        "worker_kernel_sha256": card["worker_kernel_sha256"],
        "kernel_admission": {"card": card["kernel_admission"], "task": task["kernel_admission"]},
        "source_revisions": card["source_revisions"],
        "environment": card["environment"],
        # The TASK record's fingerprint, not the card's. The worker rebuilds
        # this dict live from the source manifest of the task it is about to
        # run (`backends/robodojo/session.py`) and compares it whole, so a
        # card fingerprint measured on another task fails that readback with
        # an infrastructure fault -- after a ~90 s Isaac boot and after the
        # cell's single-attempt claim is spent. The join above already proved
        # the two agree on every card-scoped member; the per-task members are
        # exactly the ones that must come from the task.
        "actual_source_fingerprint": (
            task["card_join"]["actual_source_fingerprint"]
        ),
        "device_binding": card["device_binding"],
        "graphics_compatibility": card.get("graphics_compatibility"),
        "nvidia_driver_version": card.get("nvidia_driver_version"),
        "configuration": {
            **card["card_configuration"], **task["task_configuration"],
        },
        "task": task["task"], "task_instruction": task["task_instruction"],
        "layout_file": task["layout_file"], "layout_sha256": task["layout_sha256"],
        "cuda_visible_devices": card["cuda_visible_devices"],
        "sensor_profile": card["sensor_profile"],
        "checks": {
            "card": dict(card["checks"]), "task": dict(task["checks"]),
        },
        "card_measured_task": dict(card["measured_task"]),
        "not_established": dict(TASK_UNPROVEN_FACTS),
    }


def load_qualification(
    runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str,
) -> dict[str, Any]:
    """Load whichever qualification contract this Runtime's binding names."""

    layers = qualification_binding_layers(runtime.get("infrastructure_qualification"))
    if layers == ("combined",):
        from .robodojo_revalidation import SCHEMA, read_reference, load
        record, _ = read_reference(runtime["infrastructure_qualification"], workspace_root)
        if record.get("schema") == SCHEMA:
            return load(runtime, workspace_root=workspace_root, harness_commit=harness_commit)
        return load_infrastructure_qualification(
            runtime, workspace_root=workspace_root, harness_commit=harness_commit,
        )
    return load_layered_qualification(
        runtime, workspace_root=workspace_root, harness_commit=harness_commit,
    )


def _activation_reference(reference: Any, *, root: Path, expected_path: str) -> bytes:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise RoboDojoBenchmarkError("embodiment activation reference is invalid")
    if reference["path"] != expected_path:
        raise RoboDojoBenchmarkError("embodiment activation reference path drift")
    path = (root / expected_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RoboDojoBenchmarkError("embodiment activation reference escapes its root")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise RoboDojoBenchmarkError("embodiment activation reference hash mismatch")
    return data


def _validate_device_binding(binding: Any, *, runtime: Mapping[str, Any]) -> None:
    fields = {"schema", "renderer_vulkan_index", "renderer_cuda_index", "renderer_uuid_prefix",
              "renderer_pci_bus_hex", "selected_physical_index", "selected_gpu_uuid",
              "physics_cuda_logical_index", "physics_cuda_uuid", "selected_pci_bus_id",
              "renderer_identity_resolution"}
    if not isinstance(binding, Mapping) or set(binding) != fields:
        raise RoboDojoBenchmarkError("infrastructure qualification device binding is missing or invalid")
    try:
        for key, expected in {"renderer_vulkan_index": runtime["renderer_gpu"], "renderer_cuda_index": 0,
                              "selected_physical_index": int(runtime["cuda_visible_devices"]),
                              "physics_cuda_logical_index": 0}.items():
            if type(binding[key]) is not int or binding[key] != expected:
                raise ValueError(f"{key} drift")
        selected = binding["selected_gpu_uuid"]
        physics = binding["physics_cuda_uuid"]
        prefix = binding["renderer_uuid_prefix"]
        bus = binding["renderer_pci_bus_hex"]
        pci = binding["selected_pci_bus_id"]
        if (
            binding["schema"] != "ora.robodojo-device-binding.v1"
            or binding["renderer_identity_resolution"] != "unique_nvml_uuid_prefix_and_pci_bus"
            or not isinstance(selected, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", selected) is None
            or not isinstance(physics, str)
            or selected.lower().removeprefix("gpu-").replace("-", "") != physics.lower().removeprefix("gpu-").replace("-", "")
            or not isinstance(prefix, str) or re.fullmatch(r"[0-9a-f]{8,32}", prefix) is None
            or not selected.lower().removeprefix("gpu-").replace("-", "").startswith(prefix)
            or not isinstance(bus, str) or re.fullmatch(r"(?:0x)?[0-9a-fA-F]{1,2}", bus) is None
            or not isinstance(pci, str) or re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci) is None
            or int(bus, 16) != int(pci.split(":")[-2], 16)
        ):
            raise ValueError("GPU UUID/PCI identity drift")
    except (ValueError, TypeError, KeyError) as exc:
        raise RoboDojoBenchmarkError(f"infrastructure qualification device binding rejected: {exc}") from exc


def _validate_embodiment_activation(binding: Any, *, root: Path, device_binding: Mapping[str, Any]) -> None:
    """Recheck immutable activation joins against the current tracked card."""
    from ora.agent.composition.embodiment import (
        EmbodimentDescriptorError,
        audit_embodiment_sources, compile_embodiment_card, descriptor_activation_ready,
        validate_activation_record, validate_embodiment_descriptor,
    )
    from ora.environment.controllers.robodojo_arx import controller_binding

    fields = {"schema", "activation_ready", "descriptor", "compiled_card", "compiled_manifest",
              "activation_record", "review_record", "runtime_evidence"}
    if (
        not isinstance(binding, Mapping) or set(binding) != fields
        or binding["schema"] != "ora.robodojo-embodiment-activation-binding.v1"
        or binding["activation_ready"] is not True
    ):
        raise RoboDojoBenchmarkError("infrastructure qualification embodiment activation is missing or invalid")
    project_root = Path(__file__).resolve().parents[2]
    card_root = "ora/policy/embodiments/arx-x5"
    try:
        descriptor_bytes = _activation_reference(
            binding["descriptor"], root=project_root, expected_path=f"{card_root}/descriptor-v1.json",
        )
        descriptor = validate_embodiment_descriptor(json.loads(descriptor_bytes))
        card = _activation_reference(
            binding["compiled_card"], root=project_root, expected_path=f"{card_root}/compiled-v1/manual.md",
        ).decode("utf-8")
        manifest = json.loads(_activation_reference(
            binding["compiled_manifest"], root=project_root, expected_path=f"{card_root}/compiled-v1/manifest.json",
        ))
        expected_card, expected_manifest = compile_embodiment_card(
            descriptor, source_file_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        )
        audit = audit_embodiment_sources(descriptor, project_root=project_root)
        if card != expected_card or manifest != {**expected_manifest, "source_audit": audit}:
            raise RoboDojoBenchmarkError("embodiment compiled card/source manifest drift")
        activation = json.loads(_activation_reference(
            binding["activation_record"], root=root, expected_path="embodiment/activation.json",
        ))
        review = json.loads(_activation_reference(
            binding["review_record"], root=root, expected_path="embodiment/review.json",
        ))
        evidence = json.loads(_activation_reference(
            binding["runtime_evidence"], root=root, expected_path="embodiment/runtime-evidence.json",
        ))
        validate_activation_record(activation, descriptor=descriptor, card=card)
        if not descriptor_activation_ready(descriptor, activation_record=activation):
            raise RoboDojoBenchmarkError("embodiment descriptor is not activation ready")
        if (
            activation["review"]["record_sha256"] != binding["review_record"]["sha256"]
            or activation["runtime_verification"]["evidence_sha256"] != binding["runtime_evidence"]["sha256"]
            or review.get("schema") != "ora.embodiment-card-review.v1"
            or review.get("decision") != "accepted"
            or review.get("descriptor_sha256") != expected_manifest["descriptor_sha256"]
            or review.get("card_sha256") != expected_manifest["card_sha256"]
            or evidence.get("schema") != "ora.robodojo-embodiment-runtime-evidence.v1"
            or evidence.get("descriptor_sha256") != expected_manifest["descriptor_sha256"]
            or evidence.get("projected_facts_sha256") != expected_manifest["projected_facts_sha256"]
            or evidence.get("source_audit") != audit
            or evidence.get("controller_binding") != controller_binding()
        ):
            raise RoboDojoBenchmarkError("embodiment activation review/runtime evidence binding drift")
        workers = evidence.get("fresh_worker_evidence")
        if not isinstance(workers, list) or len(workers) != 2:
            raise RoboDojoBenchmarkError("embodiment activation requires two fresh-worker records")
        for index, worker in enumerate(workers):
            if not isinstance(worker, Mapping) or set(worker) != {"reset", "source_manifest", "probe_report", "compiled_model"}:
                raise RoboDojoBenchmarkError("embodiment activation worker evidence is invalid")
            reset = json.loads(_activation_reference(
                worker["reset"], root=root, expected_path=f"reset-{index}/environment/reset.json",
            ))
            source = json.loads(_activation_reference(
                worker["source_manifest"], root=root, expected_path=f"reset-{index}/environment/source-manifest.json",
            ))
            report = json.loads(_activation_reference(
                worker["probe_report"], root=root, expected_path=f"reset-{index}/probe-report.json",
            ))
            device = source.get("device_binding")
            if not isinstance(device, Mapping) or set(device) != {*device_binding, "startup_log_prefix"}:
                raise RoboDojoBenchmarkError("embodiment activation worker device evidence is invalid")
            prefix = device["startup_log_prefix"]
            _activation_reference(prefix, root=root / f"reset-{index}/environment", expected_path="kit-startup-readback.log")
            residency = report.get("worker_device_residency")
            if (
                not isinstance(residency, Mapping)
                or set(residency) != {"worker_pid", "gpu_uuids", "captured_unix_s", "source"}
                or type(residency["worker_pid"]) is not int or residency["worker_pid"] <= 0
                or residency["gpu_uuids"] != [device_binding["selected_gpu_uuid"]]
                or residency["source"] != "nvidia-smi query-compute-apps"
                or isinstance(residency["captured_unix_s"], bool)
                or not isinstance(residency["captured_unix_s"], (int, float))
                or not math.isfinite(residency["captured_unix_s"]) or residency["captured_unix_s"] <= 0
            ):
                raise RoboDojoBenchmarkError("embodiment activation worker device residency is invalid")
            if (
                report.get("passed") is not True or report.get("reset") != reset
                or reset.get("source_manifest", {}).get("sha256") != worker["source_manifest"]["sha256"]
                or reset.get("runtime_embodiment") != worker["compiled_model"]
                or source.get("source_revisions") != REVISIONS
                or reset.get("control_binding") != controller_binding()
                or worker["compiled_model"] != workers[0]["compiled_model"]
                or {key: device[key] for key in device_binding} != device_binding
                or reset.get("device_binding") != device
            ):
                raise RoboDojoBenchmarkError("embodiment activation worker evidence join drift")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EmbodimentDescriptorError) as exc:
        raise RoboDojoBenchmarkError(f"infrastructure qualification embodiment activation rejected: {exc}") from exc
