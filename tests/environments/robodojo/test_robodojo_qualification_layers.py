"""Two-layer RoboDojo qualification: what each layer proves, and what it refuses.

A card record proves the card and the installation once and is reused across
tasks. A task record proves one task and its scene and binds the exact card
record it was measured against. The pair must prove jointly everything the
single combined record proved, and must refuse everything it refuses -- plus
the new failure the layering makes possible: a task record paired with the
wrong card, a different worker environment or drifted source content.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ora.environment.benchmarks import robodojo
from ora.orchestrator import robodojo_qualification as qualification_module
from ora.orchestrator.robodojo_qualification import (
    CARD_JOIN_FIELDS, load_card_qualification, load_layered_qualification,
    load_qualification, load_task_qualification,
)

#: The qualified RoboDojo worker-kernel content every fixture record claims.
WORKER_KERNEL_DIGEST = "f" * 64
CARD_RELATIVE = "outputs/robodojo-compat/qualification/card/card-qualification.json"
TASK_RELATIVE = "outputs/robodojo-compat/qualification/task/task-qualification.json"


@pytest.fixture(autouse=True)
def _qualified_worker_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content binding without a git history behind the fixture commits."""

    monkeypatch.setattr(
        qualification_module, "worker_kernel_sha256",
        lambda commit, **_kwargs: WORKER_KERNEL_DIGEST,
    )


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict:
    return {
        "kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
        "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        "task_variant": "hang_mugs", "eval_seed": 0, "layout_id": 0,
        "layout_file": "Eval_Layout/RoboDojo/arx_x5/0/hang_mugs_0.json",
        "layout_sha256": "a" * 64, "num_envs": 1, "env_config": "arx_x5",
        "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": 800, "native_actions_per_command": 20,
        "main_camera": "head", "policy_cameras": list(robodojo.PUBLIC_CAMERAS),
        "required_host": "test-host", "require_gpu": True, "physics_device": "cuda:0",
        "renderer_gpu": 2, "cuda_visible_devices": "2", "startup_timeout_s": 600,
        "request_timeout_s": 180, "close_timeout_s": 30,
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        "environment_signal_policy": "lifecycle_only",
        "expected_task_language": "Hang all the mugs on the mug rack.",
    }


def _fingerprint(task: str = "hang_mugs") -> dict:
    """One worker source fingerprint, measured while running ``task``.

    The worker hashes the selected task's own config and the resolved
    environment configuration, so both move with the task. A fixture that
    omitted them made a card record measured on one task pair with every
    other task's record, which is exactly the drift the loader must catch.
    """

    return {
        "loaded_files": {
            "env_cfg/arx_x5.yml": "a" * 64,
            "task/RoboDojo/config/_task.yml": "c" * 64,
            f"task/RoboDojo/config/{task}.yml": "d" * 64,
        },
        "installed_packages": dict(robodojo.WORKER_PACKAGE_VERSIONS),
        "resolved_environment_config_sha256": (
            hashlib.sha256(task.encode()).hexdigest()
        ),
    }


def _environment() -> dict:
    return {
        "python_executable": "/fixture/bin/python", "python_binary_sha256": "e" * 64,
        "versions": {**robodojo.WORKER_PACKAGE_VERSIONS, "warp-lang": "1.9.1"},
    }


def _compiled_model() -> dict:
    return {
        "collector": "initialized_isaac_articulation_and_optical_camera_prim",
        "position_frame": "episode_local_world", "quaternion_order": "xyzw",
        "arms": {"left": {"fixed_base": True}, "right": {"fixed_base": True}},
        "cameras": {"head": {}, "left_wrist": {}, "right_wrist": {}},
    }


def _registration(runtime: dict) -> dict:
    return robodojo.build_scene_registration(
        {"mug_0": "rigid", "rack_0": "geometry"},
        evaluator_registration={
            "check_list": True, "final_check_list": False,
            "trigger_check_list": False,
        },
        native_action_limit=int(runtime["native_action_limit"]),
    )


def _card(runtime: dict) -> dict:
    """A passing card record: everything about this card and installation."""

    return {
        "schema": robodojo.CARD_QUALIFICATION_SCHEMA, "passed": True,
        "harness_commit": "c" * 40, "worker_kernel_sha256": WORKER_KERNEL_DIGEST,
        "source_revisions": robodojo.source_revisions(runtime),
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
        "card_configuration": robodojo.card_configuration_binding(runtime),
        "environment": _environment(),
        "actual_source_fingerprint": _fingerprint("arrange_largest_number"),
        "runtime_embodiment": _compiled_model(),
        "checks": {name: True for name in robodojo.CARD_QUALIFICATION_CHECKS},
        # A card record is measured on one task and reused by others; the task
        # it drove to its own native-action limit is named, never compared.
        "measured_task": {
            "task": "arrange_largest_number",
            "layout_file": "Eval_Layout/RoboDojo/arx_x5/0/arrange_largest_number_0.json",
            "layout_sha256": "d" * 64, "native_action_limit": 1050,
            "native_timeout_exercised": True,
            "task_instruction": "Arrange the largest number.",
        },
        "nvidia_driver_version": "580.65.06", "graphics_compatibility": None,
    }


def _task(runtime: dict, *, card_reference: dict) -> dict:
    """A passing task record, joined to one exact card record."""

    registration = _registration(runtime)
    return {
        "schema": robodojo.TASK_QUALIFICATION_SCHEMA, "passed": True,
        "harness_commit": "c" * 40, "worker_kernel_sha256": WORKER_KERNEL_DIGEST,
        "source_revisions": robodojo.source_revisions(runtime),
        "task": runtime["task_variant"], "eval_seed": runtime["eval_seed"],
        "layout_id": runtime["layout_id"], "layout_file": runtime["layout_file"],
        "layout_sha256": runtime["layout_sha256"],
        "task_configuration": robodojo.task_configuration_binding(runtime),
        "task_instruction": runtime["expected_task_language"],
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "sensor_profile": runtime["sensor_profile"],
        "scene_registration": registration,
        "repeated_reset_scene_registration": deepcopy(registration),
        "checks": {name: True for name in robodojo.TASK_QUALIFICATION_CHECKS},
        "not_established": robodojo.TASK_UNPROVEN_FACTS,
        "card_qualification": dict(card_reference),
        "card_join": {
            "worker_kernel_sha256": WORKER_KERNEL_DIGEST,
            "source_revisions": robodojo.source_revisions(runtime),
            "cuda_visible_devices": runtime["cuda_visible_devices"],
            "sensor_profile": runtime["sensor_profile"],
            "device_binding": None,          # filled in by _layered
            "environment": _environment(),
            "actual_source_fingerprint": _fingerprint(str(runtime["task_variant"])),
            "card_configuration": robodojo.card_configuration_binding(runtime),
            "runtime_embodiment": _compiled_model(),
        },
        "nvidia_driver_version": "580.65.06", "graphics_compatibility": None,
    }


def _layered(tmp_path: Path, activation, *, runtime: dict | None = None):
    """Write a matched card/task pair and bind it into the Runtime."""

    runtime = runtime if runtime is not None else _runtime()
    card = _card(runtime)
    card["device_binding"] = deepcopy(activation.device_binding)
    card["embodiment_activation"] = activation((tmp_path / CARD_RELATIVE).parent)
    card_sha = _write(tmp_path / CARD_RELATIVE, card)
    card_reference = {"path": CARD_RELATIVE, "sha256": card_sha}
    task = _task(runtime, card_reference=card_reference)
    task["card_join"]["device_binding"] = deepcopy(card["device_binding"])
    task_sha = _write(tmp_path / TASK_RELATIVE, task)
    runtime["infrastructure_qualification"] = {
        "card": card_reference,
        "task": {"path": TASK_RELATIVE, "sha256": task_sha},
    }
    return runtime, card, task


def _rebind(tmp_path: Path, runtime: dict, *, card=None, task=None) -> None:
    """Rewrite one or both records and refresh their hashes in the binding."""

    if card is not None:
        runtime["infrastructure_qualification"]["card"]["sha256"] = _write(
            tmp_path / CARD_RELATIVE, card,
        )
    if task is not None:
        runtime["infrastructure_qualification"]["task"]["sha256"] = _write(
            tmp_path / TASK_RELATIVE, task,
        )


# ---------------------------------------------------------------------------
# The pair is admitted, and jointly proves what one record proved
# ---------------------------------------------------------------------------


def test_a_task_record_paired_with_its_card_record_is_admitted(
    tmp_path: Path, robodojo_activation,
) -> None:
    runtime, card, task = _layered(tmp_path, robodojo_activation)
    view = load_layered_qualification(
        runtime, workspace_root=tmp_path, harness_commit="c" * 40,
    )
    assert view["layers"] == ["card", "task"]
    # Each layer is its stored record plus how admission bound its kernel,
    # and the view reports both bindings at the top level.
    admission = view["kernel_admission"]
    assert {layer: admission[layer]["mode"] for layer in admission} == {
        "card": "identical_digest", "task": "identical_digest"}
    assert view["card_qualification"] == {**card, "kernel_admission": admission["card"]}
    assert view["task_qualification"] == {**task, "kernel_admission": admission["task"]}
    # The two halves reconstruct exactly the configuration binding the single
    # combined record compared field for field: nothing fell out of the split.
    assert view["configuration"] == robodojo.configuration_binding(runtime)
    assert (set(robodojo.CARD_CONFIGURATION_FIELDS)
            | set(robodojo.TASK_CONFIGURATION_FIELDS)) == set(
        robodojo.CONFIGURATION_FIELDS
    )
    # The consumer-facing keys a combined record exposes are unchanged, so
    # preflight's interpreter, task-language and device comparisons still work.
    assert view["environment"] == _environment()
    assert view["task_instruction"] == runtime["expected_task_language"]
    assert view["device_binding"] == card["device_binding"]
    assert view["layout_sha256"] == runtime["layout_sha256"]
    # The disclosure travels with the pair, and names the evidence that does
    # establish the fact the task layer does not.
    unproven = view["not_established"]["official_terminal_at_native_action_limit"]
    assert "native-action limit" in unproven["fact"]
    assert "card record" in unproven["established_by"]
    # And the card names which task's limit it actually witnessed it on.
    assert view["card_measured_task"]["task"] == "arrange_largest_number"
    assert view["card_measured_task"]["native_timeout_exercised"] is True
    # The generic entry point dispatches on the binding shape.
    assert load_qualification(
        runtime, workspace_root=tmp_path, harness_commit="c" * 40,
    ) == view


def test_the_card_record_is_reusable_across_tasks_and_the_task_record_is_not(
    tmp_path: Path, robodojo_activation,
) -> None:
    """The whole point of the split, stated as an admission fact.

    The same card record admits a Runtime for a different task; a task record
    does not, because its layout, hash, limit and instruction are the task.
    """

    runtime, _card, _task = _layered(tmp_path, robodojo_activation)
    other = dict(runtime)
    other.update({
        "task_variant": "pack_objects_into_box",
        "layout_file": "Eval_Layout/RoboDojo/arx_x5/0/pack_objects_into_box_0.json",
        "layout_sha256": "9" * 64, "native_action_limit": 1300,
    })
    # The card half still admits it: nothing in it is scene-derived.
    assert load_card_qualification(
        other, workspace_root=tmp_path, harness_commit="c" * 40,
    )["card_configuration"] == robodojo.card_configuration_binding(other)
    # The task half does not.
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="layout drift"):
        load_layered_qualification(
            other, workspace_root=tmp_path, harness_commit="c" * 40,
        )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_missing_card_record_is_refused(
    tmp_path: Path, robodojo_activation,
) -> None:
    runtime, _card, _task = _layered(tmp_path, robodojo_activation)
    (tmp_path / CARD_RELATIVE).unlink()
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="card qualification is unavailable"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


def test_a_missing_task_record_is_refused(
    tmp_path: Path, robodojo_activation,
) -> None:
    runtime, _card, _task = _layered(tmp_path, robodojo_activation)
    (tmp_path / TASK_RELATIVE).unlink()
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="task qualification is unavailable"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


@pytest.mark.parametrize("layer", ["card", "task"])
def test_half_a_layered_binding_is_refused_rather_than_completed(layer: str) -> None:
    """A binding naming one layer never falls back to the other contract."""

    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="only part of it"):
        robodojo.qualification_binding_layers(
            {layer: {"path": CARD_RELATIVE, "sha256": "a" * 64}},
        )


@pytest.mark.parametrize("field", CARD_JOIN_FIELDS)
def test_a_task_record_measured_against_another_card_cannot_pair(
    tmp_path: Path, robodojo_activation, field: str,
) -> None:
    """Every join field is measured by the task probe and compared, not copied.

    A different card, a different worker interpreter or package set, and
    drifted worker-side source content each move one of these, so the pair
    fails closed instead of the task record inheriting the card's evidence.
    """

    runtime, _card, task = _layered(tmp_path, robodojo_activation)
    drifted = deepcopy(task)
    original = drifted["card_join"][field]
    if isinstance(original, str):
        drifted["card_join"][field] = "0" * len(original)
    elif isinstance(original, dict):
        drifted["card_join"][field] = {**original, "fixture_drift": True}
    else:
        drifted["card_join"][field] = None
    _rebind(tmp_path, runtime, task=drifted)
    with pytest.raises(robodojo.RoboDojoBenchmarkError) as error:
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )
    assert field in str(error.value) or "compiled embodiment" in str(error.value)


def test_a_task_record_that_names_a_different_card_record_is_refused(
    tmp_path: Path, robodojo_activation,
) -> None:
    """Naming another card file is refused even when the join fields agree."""

    runtime, _card, task = _layered(tmp_path, robodojo_activation)
    renamed = deepcopy(task)
    renamed["card_qualification"] = {
        "path": "outputs/robodojo-compat/qualification/other/card-qualification.json",
        "sha256": renamed["card_qualification"]["sha256"],
    }
    _rebind(tmp_path, runtime, task=renamed)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="different card record"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


def test_a_card_record_measured_on_another_card_is_refused(
    tmp_path: Path, robodojo_activation,
) -> None:
    """The device the record binds must be the device the RunPlan selects."""

    runtime, card, _task = _layered(tmp_path, robodojo_activation)
    moved = deepcopy(card)
    moved["cuda_visible_devices"] = "5"
    _rebind(tmp_path, runtime, card=moved)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="device/sensor drift"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


@pytest.mark.parametrize("mutation", [
    "layout_hash", "layout_file", "task", "limit", "instruction",
    "instances", "instance_count", "identity_digest", "evaluator",
    "registered_limit", "repeated_reset_renaming", "checks",
    "not_established", "schema", "kernel", "legacy_record", "sources",
    "hash", "escape",
])
def test_a_task_record_that_stops_proving_its_scene_is_refused(
    tmp_path: Path, robodojo_activation, mutation: str,
) -> None:
    runtime, _card, task = _layered(tmp_path, robodojo_activation)
    value = deepcopy(task)
    if mutation == "layout_hash":
        value["layout_sha256"] = "9" * 64
    elif mutation == "layout_file":
        value["layout_file"] = "Eval_Layout/RoboDojo/arx_x5/0/hang_mugs_7.json"
    elif mutation == "task":
        value["task"] = "pack_objects_into_box"
    elif mutation == "limit":
        value["task_configuration"]["native_action_limit"] = 1300
    elif mutation == "instruction":
        value["task_instruction"] = "   "
    elif mutation == "instances":
        # Object registration differs: this scene registered another object.
        instances = {"mug_0": "rigid", "rack_0": "geometry", "tray_0": "rigid"}
        value["scene_registration"]["instances"] = instances
        value["scene_registration"]["instance_identities_sha256"] = (
            robodojo.scene_identities_sha256(instances)
        )
        value["scene_registration"]["registered_instance_count"] = 3
    elif mutation == "instance_count":
        value["scene_registration"]["registered_instance_count"] = 3
    elif mutation == "identity_digest":
        value["scene_registration"]["instance_identities_sha256"] = "0" * 64
    elif mutation == "evaluator":
        value["scene_registration"]["evaluator_registration"] = {
            "check_list": False, "final_check_list": False,
            "trigger_check_list": False,
        }
    elif mutation == "registered_limit":
        value["scene_registration"]["native_action_limit"] = 1300
    elif mutation == "repeated_reset_renaming":
        # Exactly upstream's monotone-counter renaming on a later reset.
        renamed = {"mug_1": "rigid", "rack_1": "geometry"}
        value["repeated_reset_scene_registration"] = {
            **value["repeated_reset_scene_registration"],
            "instances": renamed,
            "instance_identities_sha256": robodojo.scene_identities_sha256(renamed),
        }
    elif mutation == "checks":
        value["checks"]["repeated_reset_identity"] = False
    elif mutation == "not_established":
        # A record may not quietly drop the disclosure it is required to carry.
        value.pop("not_established")
    elif mutation == "schema":
        value["schema"] = robodojo.CARD_QUALIFICATION_SCHEMA
    elif mutation == "kernel":
        value["worker_kernel_sha256"] = "a" * 64
    elif mutation == "legacy_record":
        value.pop("worker_kernel_sha256")
    elif mutation == "sources":
        value["source_revisions"]["isaaclab"] = "d" * 40
    _rebind(tmp_path, runtime, task=value)
    if mutation == "hash":
        path = tmp_path / TASK_RELATIVE
        path.write_text(path.read_text() + " ")
    elif mutation == "escape":
        runtime["infrastructure_qualification"]["task"]["path"] = "../outside.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


@pytest.mark.parametrize("mutation", [
    "schema", "not_passed", "kernel", "legacy_record", "lineage", "sources",
    "configuration", "sensor", "fingerprint", "environment", "checks",
    "compiled_model", "no_timeout", "no_measured_task", "activation", "hash",
])
def test_a_card_record_that_stops_proving_its_card_is_refused(
    tmp_path: Path, robodojo_activation, mutation: str,
) -> None:
    runtime, card, _task = _layered(tmp_path, robodojo_activation)
    value = deepcopy(card)
    if mutation == "schema":
        value["schema"] = robodojo.QUALIFICATION_SCHEMA
    elif mutation == "not_passed":
        value["passed"] = False
    elif mutation == "kernel":
        value["worker_kernel_sha256"] = "a" * 64
    elif mutation == "legacy_record":
        value.pop("worker_kernel_sha256")
    elif mutation == "lineage":
        value["harness_commit"] = "measured-somewhere"
    elif mutation == "sources":
        value["source_revisions"]["curobo"] = "d" * 40
    elif mutation == "configuration":
        value["card_configuration"]["native_actions_per_command"] = 19
    elif mutation == "sensor":
        value["sensor_profile"] = "rgb_only_v1"
    elif mutation == "fingerprint":
        value["actual_source_fingerprint"]["installed_packages"]["torch"] = "2.9.0"
    elif mutation == "environment":
        value["environment"]["python_executable"] = "python"
    elif mutation == "checks":
        value["checks"]["official_terminal"] = False
    elif mutation == "compiled_model":
        value["runtime_embodiment"]["arms"] = {"left": {}}
    elif mutation == "no_timeout":
        # A card record that never reached a task's official native-action
        # limit has not witnessed the terminal transition it is credited with.
        value["measured_task"]["native_timeout_exercised"] = False
    elif mutation == "no_measured_task":
        value.pop("measured_task")
    elif mutation == "activation":
        value["embodiment_activation"]["activation_ready"] = False
    _rebind(tmp_path, runtime, card=value)
    if mutation == "hash":
        path = tmp_path / CARD_RELATIVE
        path.write_text(path.read_text() + " ")
    with pytest.raises(robodojo.RoboDojoBenchmarkError):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )


def test_the_layered_binding_survives_an_unrelated_commit_and_not_a_kernel_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robodojo_activation,
) -> None:
    """Both layers are content-bound exactly the way the combined record is."""

    kernels = {
        "c" * 40: WORKER_KERNEL_DIGEST,   # where the probes actually ran
        "d" * 40: WORKER_KERNEL_DIGEST,   # a later, unrelated commit
        "e" * 40: "a" * 64,               # a commit that moved the kernel
    }
    runtime, _card, _task = _layered(tmp_path, robodojo_activation)
    monkeypatch.setattr(
        qualification_module, "worker_kernel_sha256",
        lambda commit, **_kwargs: kernels[commit],
    )
    for commit in ("c" * 40, "d" * 40):
        assert load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit=commit,
        )["layers"] == ["card", "task"]
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="worker kernel changed"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="e" * 40,
        )


def test_the_layered_pair_is_an_admissible_runtime_binding_shape() -> None:
    """``validate_runtime_extension`` admits the pair and only a whole pair."""

    runtime = _runtime()
    runtime["infrastructure_qualification"] = {
        "card": {"path": CARD_RELATIVE, "sha256": "a" * 64},
        "task": {"path": TASK_RELATIVE, "sha256": "b" * 64},
    }
    robodojo.validate_runtime_extension(runtime)
    assert robodojo.qualification_binding_layers(
        runtime["infrastructure_qualification"],
    ) == ("card", "task")
    # The combined single record stays admissible unchanged.
    runtime["infrastructure_qualification"] = {
        "path": CARD_RELATIVE, "sha256": "a" * 64,
    }
    robodojo.validate_runtime_extension(runtime)
    assert robodojo.qualification_binding_layers(
        runtime["infrastructure_qualification"],
    ) == ("combined",)
    runtime["infrastructure_qualification"] = {
        "card": {"path": CARD_RELATIVE, "sha256": "not-a-digest"},
        "task": {"path": TASK_RELATIVE, "sha256": "b" * 64},
    }
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="path/sha256"):
        robodojo.validate_runtime_extension(runtime)


def test_the_task_layer_is_not_credited_with_the_official_terminal_boundary() -> None:
    """The named unproven fact is a member of neither layer's check set."""

    unproven = set(robodojo.TASK_UNPROVEN_FACTS)
    assert unproven == {"official_terminal_at_native_action_limit"}
    assert "official_terminal" in robodojo.CARD_QUALIFICATION_CHECKS
    assert "official_terminal" not in robodojo.TASK_QUALIFICATION_CHECKS
    # Every substantive name in the combined contract is carried by exactly
    # one layer, so the split proves what the single record proved.
    covered = set(robodojo.CARD_QUALIFICATION_CHECKS) | set(
        robodojo.TASK_QUALIFICATION_CHECKS
    )
    assert {"rgbd_calibration", "motion_contract", "official_terminal",
            "bounded_close"} <= covered
    assert {"scene_rgbd_calibration", "scene_motion_prefix",
            "repeated_reset_identity", "scene_registration"} <= covered


def test_load_task_qualification_refuses_a_card_it_was_not_handed(
    tmp_path: Path, robodojo_activation,
) -> None:
    """The task loader never resolves its own card; the layered loader pairs."""

    runtime, card, _task = _layered(tmp_path, robodojo_activation)
    foreign = deepcopy(card)
    foreign["environment"] = {**_environment(), "python_binary_sha256": "9" * 64}
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="does not pair"):
        load_task_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
            card=foreign,
            card_reference=runtime["infrastructure_qualification"]["card"],
        )


def test_the_layered_view_is_handed_to_the_worker_like_a_combined_record(
    tmp_path: Path, robodojo_activation,
) -> None:
    """The Supervisor accepts both contracts, and the worker rechecks both.

    ``coordinator._qualified_runtime_binding`` carries the admitted record
    into the isolated worker, which compares it with its own live readback at
    reset (``RobodojoSession.start``). The layered projection must therefore
    supply exactly the fields that comparison reads, or a run that preflighted
    clean would die at its first boot.
    """

    from ora.orchestrator import coordinator

    runtime, card, task = _layered(tmp_path, robodojo_activation)
    view = load_layered_qualification(
        runtime, workspace_root=tmp_path, harness_commit="c" * 40,
    )
    _write(tmp_path / "preflight.json", {"checks": [{
        "check": "robodojo_infrastructure_qualification",
        "passed": True, "detail": view,
    }]})
    assert coordinator._qualified_runtime_binding(tmp_path, runtime) == view
    # Presence is not the contract. RobodojoSession.start rebuilds each of
    # these live from the source manifest of the task it is about to run and
    # compares the VALUE, so assert the values the worker would compute --
    # a view that merely carries the keys still dies at the first boot, which
    # is exactly what the earlier version of this test failed to catch.
    assert view["passed"] is True
    assert view["source_revisions"] == task["card_join"]["source_revisions"]
    assert view["layout_sha256"] == task["layout_sha256"]
    assert view["task_instruction"] == task["task_instruction"]
    assert view["device_binding"] == task["card_join"]["device_binding"]
    assert view["configuration"] == {
        **card["card_configuration"], **task["task_configuration"],
    }
    # The one the worker rebuilds from THIS task's own loaded files and
    # resolved environment configuration. The card measured another task, so
    # taking the card's copy here hands the worker a fingerprint it cannot
    # reproduce.
    assert view["actual_source_fingerprint"] == (
        task["card_join"]["actual_source_fingerprint"]
    )
    assert card["measured_task"]["task"] != task["task"]
    assert view["actual_source_fingerprint"] != card["actual_source_fingerprint"]

    _write(tmp_path / "preflight.json", {"checks": [{
        "check": "robodojo_infrastructure_qualification",
        "passed": True, "detail": {**view, "schema": "unqualified"},
    }]})
    with pytest.raises(coordinator.SupervisorError, match="infrastructure binding"):
        coordinator._qualified_runtime_binding(tmp_path, runtime)


@pytest.mark.parametrize("mutation", [
    None, "missing_task_record", "missing_card_record", "unpaired_card",
    "renamed_scene",
])
def test_preflight_requires_both_layers_and_reports_what_the_task_layer_omits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robodojo_activation,
    mutation: str | None,
) -> None:
    """The gate itself: both layers, or the check fails.

    A missing layer, a card record measured elsewhere and a repeated reset
    that lost this scene's identities each fail the
    ``robodojo_infrastructure_qualification`` check rather than degrading to
    the other layer's evidence. When the pair is admitted, preflight's own
    report names the fact the task layer does not establish.
    """

    import subprocess
    import sys

    from ora.orchestrator import preflight

    runtime, card, task = _layered(tmp_path, robodojo_activation)
    root, assets = tmp_path / "RoboDojo", tmp_path / "Assets"
    root.mkdir()
    layout = assets / runtime["layout_file"]
    layout.parent.mkdir(parents=True)
    layout.write_text('{"fixture": true}')
    runtime["layout_sha256"] = hashlib.sha256(layout.read_bytes()).hexdigest()
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    for name, value in {"ROBODOJO_ROOT": root, "ROBODOJO_ASSET_ROOT": assets,
                        "ROBODOJO_PYTHON": python,
                        "ORA_EXECUTION_HOST": "test-host"}.items():
        monkeypatch.setenv(name, str(value))
    observed = {"python_executable": str(python), "python_binary_sha256": "e" * 64,
                "versions": {"isaacsim": "5.1.0.0", "torch": "2.7.0+cu128",
                             "numpy": "1.26.0", "warp-lang": "1.9.1"}}
    card["environment"] = deepcopy(observed)
    card["layout_sha256"] = runtime["layout_sha256"]
    task["layout_sha256"] = runtime["layout_sha256"]
    task["task_configuration"] = robodojo.task_configuration_binding(runtime)
    task["card_join"]["environment"] = deepcopy(observed)
    if mutation == "unpaired_card":
        card["actual_source_fingerprint"] = {
            **_fingerprint("arrange_largest_number"),
            "loaded_files": {
                **_fingerprint("arrange_largest_number")["loaded_files"],
                "env_cfg/arx_x5.yml": "9" * 64,
            },
        }
    if mutation == "renamed_scene":
        renamed = {"mug_1": "rigid", "rack_1": "geometry"}
        task["repeated_reset_scene_registration"] = {
            **task["repeated_reset_scene_registration"], "instances": renamed,
            "instance_identities_sha256": robodojo.scene_identities_sha256(renamed),
        }
    _rebind(tmp_path, runtime, card=card, task=task)
    # The task record's stored card reference must follow the card's new hash.
    task["card_qualification"] = dict(runtime["infrastructure_qualification"]["card"])
    _rebind(tmp_path, runtime, task=task)
    if mutation == "missing_task_record":
        (tmp_path / TASK_RELATIVE).unlink()
    elif mutation == "missing_card_record":
        (tmp_path / CARD_RELATIVE).unlink()

    roots = {root: "robodojo", root / "third_party/IsaacLab": "isaaclab",
             root / "third_party/curobo": "curobo",
             root / "XPolicyLab": "xpolicylab", assets: "assets"}

    def fake_git(path, *args):
        revision = robodojo.REVISIONS[roots[path]]
        return subprocess.CompletedProcess(
            args, 0, revision if args[0] == "rev-parse" else "", "",
        )

    def runner(command, **kwargs):
        if command[0] == "nvidia-smi":
            uuid = robodojo_activation.device_binding["selected_gpu_uuid"]
            return subprocess.CompletedProcess(
                command, 0, f"2, {uuid}, 40000, 580.65.06\n", "",
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(observed), "")

    monkeypatch.setattr(preflight, "_git", fake_git)
    monkeypatch.setattr(preflight, "resolve_workspace_root", lambda _root: tmp_path)
    monkeypatch.setattr(
        qualification_module, "worker_kernel_sha256",
        lambda commit, **_kwargs: WORKER_KERNEL_DIGEST,
    )
    checks = preflight._robodojo_runtime({
        "runtime": runtime, "task": runtime["task_variant"], "seed": 0,
        "task_instruction": runtime["expected_task_language"],
        "source": {"commit": "c" * 40},
    }, runner)
    by_name = {check["check"]: check for check in checks}
    if mutation is None:
        assert by_name["robodojo_infrastructure_qualification"]["passed"] is True
        # The preflight detail says how each layer's kernel was admitted, so a
        # manifest-extension admission is never mistaken for digest equality.
        admitted = by_name["robodojo_infrastructure_qualification"]["detail"]["kernel_admission"]
        assert {layer: admitted[layer]["mode"] for layer in admitted} == {
            "card": "identical_digest", "task": "identical_digest"}
        assert admitted["card"]["plan_commit"] == "c" * 40
        layers = by_name["robodojo_qualification_layers"]
        assert layers["passed"] is True
        detail = layers["detail"]
        assert detail["layers"] == ["card", "task"]
        assert set(detail["not_established_by_the_task_layer"]) == {
            "official_terminal_at_native_action_limit",
        }
        assert detail["card_measured_task"]["native_timeout_exercised"] is True
    else:
        assert by_name["robodojo_infrastructure_qualification"]["passed"] is not True
        # A failed pair never reports itself as a layered success.
        assert "robodojo_qualification_layers" not in by_name


def test_a_card_record_pairs_with_a_task_it_was_not_measured_on(
    tmp_path: Path, robodojo_activation,
) -> None:
    """The defect that made the whole split admit exactly one task.

    The worker's source fingerprint carries two per-task facts: the selected
    task's own loaded config, and the resolved environment configuration hash.
    Comparing the fingerprint whole meant a card record measured on one task
    paired with that task and nothing else -- the layering silently degraded
    to a per-task qualification that still cost a card measurement. Both are
    projected out of the join and asserted against each record's own task
    instead.
    """

    runtime, card, task = _layered(tmp_path, robodojo_activation)
    # The fixture already models it: the card was measured on
    # arrange_largest_number, the task record on hang_mugs.
    assert card["measured_task"]["task"] != task["task"]
    fingerprints = (
        card["actual_source_fingerprint"],
        task["card_join"]["actual_source_fingerprint"],
    )
    assert fingerprints[0] != fingerprints[1]
    view = load_layered_qualification(
        runtime, workspace_root=tmp_path, harness_commit="c" * 40,
    )
    assert set(view["layers"]) == {"card", "task"}
    # Admission succeeding is only half of it. The view is handed to the
    # isolated worker, which rebuilds the fingerprint from the task it runs,
    # so the view must carry the TASK's copy -- the card's would preflight
    # clean and then fail the boot readback.
    assert view["actual_source_fingerprint"] == fingerprints[1]
    assert view["actual_source_fingerprint"] != fingerprints[0]


@pytest.mark.parametrize("mutation", ["wrong_task_config", "no_task_config", "two_task_configs"])
def test_a_fingerprint_measured_on_the_wrong_scene_is_still_refused(
    tmp_path: Path, robodojo_activation, mutation: str,
) -> None:
    """Projecting the per-task members out may not hide a wrong measurement."""

    runtime, _card, task = _layered(tmp_path, robodojo_activation)
    loaded = dict(task["card_join"]["actual_source_fingerprint"]["loaded_files"])
    own = f"task/RoboDojo/config/{task['task']}.yml"
    if mutation == "wrong_task_config":
        loaded.pop(own)
        loaded["task/RoboDojo/config/push_T.yml"] = "d" * 64
    elif mutation == "no_task_config":
        loaded.pop(own)
    else:
        loaded["task/RoboDojo/config/push_T.yml"] = "d" * 64
    task["card_join"]["actual_source_fingerprint"] = {
        **task["card_join"]["actual_source_fingerprint"], "loaded_files": loaded,
    }
    _rebind(tmp_path, runtime, task=task)
    task["card_qualification"] = dict(runtime["infrastructure_qualification"]["card"])
    _rebind(tmp_path, runtime, task=task)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="loaded"):
        load_layered_qualification(
            runtime, workspace_root=tmp_path, harness_commit="c" * 40,
        )
