"""Host-safe identities and admission for the first RoboDojo integration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from ora.environment.workspace_profiles import (
    ROBODOJO_ARX_X5_TABLETOP_V1,
)


REVISIONS = {
    "robodojo": "ee67a1468510da7624a089164402359f2afc72c8",
    "isaaclab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
    "curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e",
    "xpolicylab": "432f82b1758c5b1202e42a3dfe014546dbc50871",
    "assets": "a14409d7fae673c00499e01fd88b4457df6351b1",
}
REVISION_FIELDS = {
    "robodojo": "required_revision",
    "isaaclab": "required_isaaclab_revision",
    "curobo": "required_curobo_revision",
    "xpolicylab": "required_xpolicylab_revision",
    "assets": "required_asset_revision",
}
QUALIFICATION_SCHEMA = "ora.robodojo-infrastructure-qualification.v1"
QUALIFICATION_CHECKS = (
    "repeated_reset", "rgbd_calibration", "motion_contract",
    "official_terminal", "bounded_close",
)
#: The per-card/per-installation half of the same evidence, measured once for
#: a card plus worker environment plus source content and reused by every task
#: that runs on it. Same five substantive names as the combined record, with
#: ``repeated_reset`` renamed to what the two *fresh workers* actually show --
#: that two independent worker processes on this card agree on task language,
#: source fingerprint, control binding, external revisions and GPU identity.
#: Scene-object identity across two resets of one live plant is a scene fact
#: and moved to the task layer, where it is genuinely measured.
CARD_QUALIFICATION_SCHEMA = "ora.robodojo-card-qualification.v1"
CARD_QUALIFICATION_CHECKS = (
    "worker_determinism", "rgbd_calibration", "motion_contract",
    "official_terminal", "bounded_close",
)
#: The per-task/per-scene half: the layout the cell names, its content hash,
#: every layout instance resolving to a scene object, the official evaluator
#: registering its checks, the task's own registered native-action limit read
#: back at reset, the upstream public instruction, RGB-D calibration inside
#: this scene, the eleven-command motion prefix, one identity-preserving
#: repeated reset of a live plant, and a bounded close.
TASK_QUALIFICATION_SCHEMA = "ora.robodojo-task-qualification.v1"
TASK_QUALIFICATION_CHECKS = (
    "scene_registration", "evaluator_registration", "repeated_reset_identity",
    "scene_rgbd_calibration", "scene_motion_prefix", "bounded_close",
)
#: What the task layer explicitly does *not* establish, named in the record,
#: in preflight's report and in the reference documentation rather than
#: quietly inherited from the card record.
#:
#: ``RobodojoSession._official_outcome`` reports ``terminal: True`` only when
#: ``bridge.advance_official_terminal`` has latched, and that latch needs
#: ``take_action_cnt >= step_lim``, an official success, or an official
#: failure. Lowering ``max_sim_steps`` does not substitute: the session then
#: reports ``max_sim_steps_per_episode_exhausted`` with ``terminal: False``.
#: So the only way to witness the transition is to spend the task's full
#: native-action budget, which is the entire 86% of the long probe's wall
#: clock. The task layer refuses to claim it.
TASK_UNPROVEN_FACTS = {
    "official_terminal_at_native_action_limit": {
        "fact": (
            "that the official RoboDojo terminal transition actually fires at "
            "this task's own registered native-action limit, and that the "
            "final native/physics counters reconcile with the receipts there"
        ),
        "why_not": (
            "witnessing the transition requires spending the task's whole "
            "native-action budget, which is what makes the long probe cost "
            "35-56 minutes; a lowered max_sim_steps reports "
            "max_sim_steps_per_episode_exhausted with terminal false and "
            "proves nothing about the official boundary"
        ),
        "established_by": (
            "the card record's own full-length episodes, which drive one task "
            "on this exact card, worker environment and source content to its "
            "limit and record the transition, the counter reconciliation, the "
            "post-terminal action refusal and the immutable post-terminal "
            "verdict; and, for this task specifically, by the run itself, "
            "whose official verdict is read from the same boundary"
        ),
        "task_layer_proves_instead": (
            "that this task's own registered native-action limit is the one "
            "the environment reports at reset (step_lim == "
            "native_action_limit) and that the official evaluator registered "
            "its success checks for this scene"
        ),
    },
}
#: The 21 ``CONFIGURATION_FIELDS`` split by what a record can actually reuse.
#: A card record measured for one task is admissible for another task on the
#: same card only if every field here is unchanged; the four task fields are
#: what a new task necessarily moves.
TASK_CONFIGURATION_FIELDS = (
    "task_variant", "eval_seed", "layout_id", "layout_file", "layout_sha256",
    "native_action_limit",
)
#: The published per-scene registration a reset produces: which layout
#: instances exist, that each resolved to a scene object, which official
#: evaluator check lists the task registered, and the native-action limit the
#: environment itself reports. This is the entire per-scene surface the task
#: qualification layer rests on, and it costs one boot.
SCENE_REGISTRATION_SCHEMA = "ora.robodojo-scene-registration.v1"
#: The instance kinds upstream actually registers as scene objects on CUDA.
#: Fluid objects are disabled upstream on CUDA and never registered, so they
#: are outside the identity set rather than a missing object. On CPU physics
#: (``robodojo_physics.py``) upstream does register Fluid; it stays outside this
#: set until a batched kernel refresh, because editing the tuple restales every
#: card and scene gate. ``physics_scene_probe`` checks the liquid meanwhile.
SCENE_TRACKED_INSTANCE_KINDS = (
    "rigid", "dynamic", "articulation", "garment", "geometry",
)
SCENE_REGISTRATION_FIELDS = (
    "schema", "instances", "instance_identities_sha256",
    "registered_instance_count", "evaluator_registration",
    "native_action_limit",
)
PUBLIC_CAMERAS = ("head", "left_wrist", "right_wrist")
#: Every first-stage action codec the isolated worker may be bound to. The
#: worker's own source manifest reports the codec it was actually configured
#: with, so a cell that changes codec cannot be attested under another one.
ACTION_CODECS = ("runtime_action_v3", "motion_action_v3")
#: The single sensor profile the pinned RoboDojo camera configuration produces.
SENSOR_PROFILE = "rgbd_metric_v1"
#: The pinned upstream simulation cadence. These are worker facts, not RunPlan
#: choices: the worker verifies a RunPlan that declares anything else instead
#: of reporting the declaration back as if it had been measured.
PHYSICS_DT_S = 0.004
CONTROL_HZ = 25
#: Every admitted fixed-layout RoboDojo task, mapped to the official
#: ``step_lim`` its own upstream task module sets in ``__init__``. The value
#: is never chosen here: ``RobodojoSession._reset_environment`` compares the
#: freshly reset ``environment.step_lim`` against the binding and refuses the
#: episode when they differ, so a wrong number is a worker that will not run
#: rather than a looser budget.
#:
#: Source of every number: the single literal ``self.step_lim = N`` in
#: ``task/RoboDojo/tasks/<task>.py`` at RoboDojo ``REVISIONS["robodojo"]``
#: (ee67a146), at the line named beside each entry. The task set is the 42
#: base simulation tasks of ``scripts/internal/task_inventory.py``
#: ``DIMENSION_TASKS`` at that revision (the ``*_random`` variants are other
#: task modules and are not admitted). Admission does not need a reviewed
#: task contract; ``agent/composition/robodojo_task_contract.py`` records an
#: absent one, and every scene still needs its own task qualification.
#:
#: All 42 official tasks are admitted. The last two, ``pour_by_language``
#: (step_lim 800) and ``pour_liquid_into_cup`` (400), joined on 2026-10-04
#: (0059 pour admission): both are judged on a Fluid object, which upstream
#: ``SceneManager`` never creates on a CUDA physics device
#: (``env/scene_manager/scene_manager.py``), so both resolve CPU physics
#: (``robodojo_physics.py``). Each has a reviewed task contract (registry-v7)
#: and a passing fluid ``physics_scene_probe``
#: (``robodojo-physics-scene-probe-20261004T012747Z-pour-r2``).
#:
#: Every entry lives inside this one literal map, the region the card and
#: scene dependency projection (``robodojo_revalidation._select_task_limit``)
#: reduces to the measured task's own entry, so adding a task here moves no
#: other task's card or scene-gate dependency digest. The worker-kernel digest
#: hashes this file's bytes and does move.
#: The RunSpec may bind ``runtime.required_host`` to the execution host whose
#: profile registers the RoboDojo installation and lanes
#: (``orchestrator/hosts/<id>.yaml``); preflight checks the host marker against it.

NATIVE_ACTION_LIMITS = {
    "align_blocks": 200,                      # tasks/align_blocks.py:9
    "arrange_largest_number": 1050,           # tasks/arrange_largest_number.py:11
    "build_tower": 1050,                      # tasks/build_tower.py:9
    "classify_objects": 1100,                 # tasks/classify_objects.py:9
    "classify_objects_by_language": 1100,     # tasks/classify_objects_by_language.py:11
    "cover_blocks": 800,                      # tasks/cover_blocks.py:9
    "deposit_coin": 300,                      # tasks/deposit_coin.py:9
    "fasten_screws": 1900,                    # tasks/fasten_screws.py:11
    "fill_egg_holder": 700,                   # tasks/fill_egg_holder.py:11
    "fill_pen_holder": 1100,                  # tasks/fill_pen_holder.py:11
    "fold_clothes": 500,                      # tasks/fold_clothes.py:9
    "general_pickup": 200,                    # tasks/general_pickup.py:9
    "hang_mugs": 800,                         # tasks/hang_mugs.py:11
    "imitate_sorting_sequence": 1600,         # tasks/imitate_sorting_sequence.py:15
    "insert_key": 300,                        # tasks/insert_key.py:11
    "insert_tubes": 500,                      # tasks/insert_tubes.py:11
    "make_kong": 600,                         # tasks/make_kong.py:15
    "make_toast": 1400,                       # tasks/make_toast.py:15
    "match_and_pick_from_conveyor": 700,      # tasks/match_and_pick_from_conveyor.py:9
    "organize_table": 1000,                   # tasks/organize_table.py:11
    "pack_objects_into_box": 1300,            # tasks/pack_objects_into_box.py:11
    "pick_from_conveyor_by_image": 700,       # tasks/pick_from_conveyor_by_image.py:9
    "play_stacking_toy": 1200,                # tasks/play_stacking_toy.py:11
    "play_tic_tac_toe": 1100,                 # tasks/play_tic_tac_toe.py:17
    "play_Xylophone": 500,                    # tasks/play_Xylophone.py:9
    "plug_in_charger": 400,                   # tasks/plug_in_charger.py:9
    "pour_balls_into_vase": 600,              # tasks/pour_balls_into_vase.py:9
    "pour_by_language": 800,                  # tasks/pour_by_language.py:10
    "pour_liquid_into_cup": 400,              # tasks/pour_liquid_into_cup.py:10
    "press_by_number": 700,                   # tasks/press_by_number.py:9
    "push_T": 600,                            # tasks/push_T.py:9
    "put_bottles_into_dustbin": 700,          # tasks/put_bottles_into_dustbin.py:11
    "solve_equation": 300,                    # tasks/solve_equation.py:9
    "sort_nesting_dolls_by_size": 1050,       # tasks/sort_nesting_dolls_by_size.py:9
    "stack_blocks": 550,                      # tasks/stack_blocks.py:19
    "stack_blocks_by_language": 400,          # tasks/stack_blocks_by_language.py:9
    "stack_bowls": 800,                       # tasks/stack_bowls.py:9
    "store_laptop_and_headphones": 800,       # tasks/store_laptop_and_headphones.py:9
    "store_tools_in_toolbox": 900,            # tasks/store_tools_in_toolbox.py:11
    "swap_blocks": 700,                       # tasks/swap_blocks.py:9
    "swap_T": 400,                            # tasks/swap_T.py:9
    "sweep_blocks": 1000,                     # tasks/sweep_blocks.py:9
}
#: Physics steps RoboDojo advances per native action, at the pinned
#: ``physics_dt: 0.004`` / ``control_hz: 25`` pair: one control tick is
#: ``1 / 25`` s and ``0.04 / 0.004`` is ten PhysX steps. Both settings are
#: exact members of ``CONFIGURATION_FIELDS``, so this ratio cannot drift
#: without the configuration binding changing too.
PHYSICS_STEPS_PER_NATIVE_ACTION = 10
WORKER_PACKAGE_VERSIONS = {
    "isaacsim": "5.1.0.0", "torch": "2.7.0+cu128", "numpy": "1.26.0",
}
CONFIGURATION_FIELDS = (
    "task_variant", "eval_seed", "layout_id", "layout_file", "layout_sha256",
    "num_envs", "env_config", "sensor_profile", "physics_dt", "control_hz",
    "native_action_limit", "native_actions_per_command", "physics_device",
    "renderer_gpu", "cuda_visible_devices", "main_camera", "policy_cameras",
    "workspace_bounds_m", "startup_timeout_s", "request_timeout_s", "close_timeout_s",
)
CARD_CONFIGURATION_FIELDS = tuple(
    field for field in CONFIGURATION_FIELDS
    if field not in TASK_CONFIGURATION_FIELDS
)
#: Nothing may fall out of the split: the two halves partition the exact set
#: the combined record compares field for field, so a card record plus a task
#: record still pins every configuration field a single record pinned.
assert set(CARD_CONFIGURATION_FIELDS) | set(TASK_CONFIGURATION_FIELDS) == set(
    CONFIGURATION_FIELDS
) and not set(CARD_CONFIGURATION_FIELDS) & set(TASK_CONFIGURATION_FIELDS)


#: The only non-default contact route; absent keeps the released refusal of a
#: contact approach beyond the local envelope (backends/robodojo/staged_contact.py).
CONTACT_APPROACH_STAGED = "global_precontact_then_local"


class RoboDojoBenchmarkError(ValueError):
    """A RoboDojo source, layout or infrastructure binding is invalid."""


def physics_step_ceiling(task: str) -> int:
    """The most physics steps one RoboDojo episode of ``task`` can admit.

    ``RobodojoSession.__init__`` refuses any ``max_sim_steps`` outside
    ``[10, native_action_limit * 10]`` or off a ten-step boundary, because the
    episode itself ends at the task's official native-action limit and each
    native action is exactly ``PHYSICS_STEPS_PER_NATIVE_ACTION`` PhysX steps.
    A physics budget above this number is therefore not a looser budget, it is
    a worker that refuses to start. Every caller that sizes ``max_sim_steps``
    for a RoboDojo cell reads it here rather than restating a constant, so the
    ceiling has one owner and follows the task's own limit.
    """

    if task not in NATIVE_ACTION_LIMITS:
        raise RoboDojoBenchmarkError(
            "RoboDojo task must be one of the reviewed fixed-layout tasks: "
            + ", ".join(sorted(NATIVE_ACTION_LIMITS))
        )
    return NATIVE_ACTION_LIMITS[task] * PHYSICS_STEPS_PER_NATIVE_ACTION


def source_revisions(runtime: Mapping[str, Any]) -> dict[str, str]:
    return {name: str(runtime.get(field, "")) for name, field in REVISION_FIELDS.items()}


def configuration_binding(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {key: runtime[key] for key in CONFIGURATION_FIELDS}


def card_configuration_binding(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The card/installation half a record measured for one task can reuse."""

    return {key: runtime[key] for key in CARD_CONFIGURATION_FIELDS}


def task_configuration_binding(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The task/scene half a new task necessarily moves."""

    return {key: runtime[key] for key in TASK_CONFIGURATION_FIELDS}


def scene_identities_sha256(instances: Mapping[str, str]) -> str:
    """Hash the exact instance-name/kind map a reset registered."""

    return hashlib.sha256(json.dumps(
        dict(sorted(instances.items())), sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def build_scene_registration(
    instances: Mapping[str, str], *, evaluator_registration: Mapping[str, Any],
    native_action_limit: int,
) -> dict[str, Any]:
    """Assemble the per-scene registration a reset publishes."""

    tracked = {
        str(name): str(kind) for name, kind in sorted(instances.items())
        if str(kind) in SCENE_TRACKED_INSTANCE_KINDS
    }
    return validate_scene_registration({
        "schema": SCENE_REGISTRATION_SCHEMA,
        "instances": tracked,
        "instance_identities_sha256": scene_identities_sha256(tracked),
        "registered_instance_count": len(tracked),
        "evaluator_registration": {
            key: bool(evaluator_registration.get(key))
            for key in ("check_list", "final_check_list", "trigger_check_list")
        },
        "native_action_limit": int(native_action_limit),
    }, native_action_limit=int(native_action_limit))


def validate_scene_registration(
    value: Any, *, native_action_limit: int,
) -> dict[str, Any]:
    """Refuse a scene registration that proves nothing about this scene.

    An empty instance map, a stale identity digest, a count that disagrees
    with the map, an evaluator that registered no check list at all, or a
    limit other than the task's own registered one all fail closed. The
    identity digest is what a repeated reset is compared on, so it is
    recomputed here rather than trusted.
    """

    if not isinstance(value, Mapping) or set(value) != set(SCENE_REGISTRATION_FIELDS):
        raise RoboDojoBenchmarkError("scene registration is missing or malformed")
    instances = value["instances"]
    registration = value["evaluator_registration"]
    if (
        value["schema"] != SCENE_REGISTRATION_SCHEMA
        or not isinstance(instances, Mapping) or not instances
        or any(
            not isinstance(name, str) or not name
            or not isinstance(kind, str) or kind not in SCENE_TRACKED_INSTANCE_KINDS
            for name, kind in instances.items()
        )
        or value["instance_identities_sha256"] != scene_identities_sha256(instances)
        or type(value["registered_instance_count"]) is not int
        or value["registered_instance_count"] != len(instances)
        or not isinstance(registration, Mapping)
        or set(registration) != {"check_list", "final_check_list", "trigger_check_list"}
        or any(type(item) is not bool for item in registration.values())
        or not any(registration.values())
        or type(value["native_action_limit"]) is not int
        or value["native_action_limit"] != int(native_action_limit)
    ):
        raise RoboDojoBenchmarkError(
            "scene registration does not establish this scene's objects, its "
            "official evaluator or its registered native-action limit"
        )
    return dict(value)


def public_context() -> dict[str, Any]:
    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "robodojo",
        "embodiment_id": "arx_x5",
        "robot_models": ["ARX_X5", "ARX_X5"],
        "controller": "CUROBO_IK_JOINT_INTERPOLATION",
        "public_cameras": list(PUBLIC_CAMERAS),
    }


def _reference(value: Any) -> bool:
    return (
        isinstance(value, Mapping) and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str) and bool(value["path"])
        and re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is not None
    )


def qualification_binding_layers(qualification: Any) -> tuple[str, ...]:
    """Which record kinds a well-formed ``infrastructure_qualification`` names.

    Two shapes are admitted and nothing in between. ``{path, sha256}`` is the
    combined single record the compat package still produces and the tracked
    unbound EXP08A marker still carries. ``{card: {...}, task: {...}}`` is the
    layered pair: a card record measured once per card, worker environment and
    source content, and a task record measured per task and scene that binds
    that exact card record by hash. A binding that names only one of the two
    layers is refused here rather than half-loaded later, so preflight fails
    closed on a missing layer instead of falling back to the other one.
    """

    if _reference(qualification):
        return ("combined",)
    if isinstance(qualification, Mapping) and set(qualification) == {"card", "task"}:
        if not _reference(qualification["card"]) or not _reference(qualification["task"]):
            raise RoboDojoBenchmarkError(
                "each RoboDojo qualification layer needs a path/sha256 pair"
            )
        return ("card", "task")
    if isinstance(qualification, Mapping) and set(qualification) & {"card", "task"}:
        raise RoboDojoBenchmarkError(
            "a layered RoboDojo qualification binding needs both the card and "
            f"the task record; {sorted(set(qualification))} names only part of it"
        )
    raise RoboDojoBenchmarkError(
        "RoboDojo requires an infrastructure qualification path/hash, or a "
        "card/task layered pair"
    )


def validate_qualification_binding(qualification: Any) -> tuple[str, ...]:
    """Admit either the combined record or the card/task layered pair."""

    return qualification_binding_layers(qualification)


def validate_runtime_extension(runtime: Mapping[str, Any]) -> None:
    """Admit only the reviewed source, its round-0 layout and explicit settings.

    Without ``layout_schedule`` every round replays layout 0. With one, round 0
    is the schedule's first layout and each later round a different reviewed
    layout of the same evaluation seed (``robodojo_layouts``); which layouts
    are reviewed at all is checked against the catalog at RunSpec resolution.

    ``physics_device`` is pinned per task: ``cpu`` for the CPU-physics scenes
    of ``robodojo_physics.CPU_PHYSICS_SCENES`` (user decision 2026-10-04) and
    ``cuda:0`` for every other task. The table is imported here, lazily, so it
    stays outside the worker kernel: this function is stripped from every card
    and scene dependency projection (``robodojo_revalidation``).
    """
    from .robodojo_physics import physics_device_for

    allowed = {
        "kind", "transport", "root_env", "python_env", "asset_root_env",
        *REVISION_FIELDS.values(), "required_host", "require_gpu",
        "eval_seed", "layout_id", "layout_file", "layout_sha256", "num_envs",
        "task_variant", "env_config", "sensor_profile", "physics_dt",
        "control_hz", "native_action_limit", "native_actions_per_command",
        "physics_device", "renderer_gpu", "cuda_visible_devices",
        "main_camera", "policy_cameras", "startup_timeout_s", "request_timeout_s",
        "close_timeout_s", "workspace_bounds_m", "environment_signal_policy", "execution_capture",
        "public_environment_context", "public_geometry_descriptor", "infrastructure_qualification",
        "resource_binding", "expected_task_language", "layout_schedule",
        # Unified-instance bindings (EXP08A robodojo_x5): the resolver replaces
        # a source-owned workspace profile selector with its resolved body and
        # keeps the literal bounds beside it, and composition records which
        # environment/embodiment prompt profile the public context selected.
        "workspace_profile", "environment_prompt_profile",
        # Derived by the RunSpec from robodojo_staged_contact_approach_v1 and
        # robodojo_retract_then_unwind_recovery_v1.
        "contact_approach_route", "recovery_route",
    }
    unexpected = set(runtime) - allowed
    if unexpected:
        raise RoboDojoBenchmarkError(f"unsupported RoboDojo runtime fields: {sorted(unexpected)}")
    if runtime.get("transport") != "local_rpc":
        raise RoboDojoBenchmarkError("RoboDojo requires local_rpc")
    for field in ("root_env", "python_env", "asset_root_env"):
        if not isinstance(runtime.get(field), str) or re.fullmatch(
            r"[A-Z][A-Z0-9_]*", runtime[field],
        ) is None:
            raise RoboDojoBenchmarkError(f"runtime.{field} must name an environment variable")
    from .robodojo_asset_revision import asset_revision_for
    expected_revisions = {**REVISIONS, "assets": asset_revision_for(runtime.get("task_variant"))}
    if source_revisions(runtime) != expected_revisions:
        raise RoboDojoBenchmarkError("RoboDojo requires the reviewed code/fork/asset revisions")
    task = runtime.get("task_variant")
    if not isinstance(task, str) or task not in NATIVE_ACTION_LIMITS:
        raise RoboDojoBenchmarkError(
            "RoboDojo task must be one of the reviewed fixed-layout tasks: "
            + ", ".join(sorted(NATIVE_ACTION_LIMITS))
        )
    exact = {
        "task_variant": task, "eval_seed": 0,
        "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1",
        "physics_dt": 0.004, "control_hz": 25, "native_action_limit": NATIVE_ACTION_LIMITS[task],
        "native_actions_per_command": 20, "main_camera": "head",
        # CPU physics still renders, and may solve PhysX dynamics and CuRobo,
        # on the worker's one visible GPU: require_gpu stays exact for every task.
        "require_gpu": True,
        "physics_device": physics_device_for(task),
    }
    host = runtime.get("required_host")
    if host is not None and (not isinstance(host, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host) is None):
        raise RoboDojoBenchmarkError("runtime.required_host must be a stable host id")
    for key, expected in exact.items():
        value = runtime.get(key)
        if value != expected or isinstance(value, bool) != isinstance(expected, bool):
            raise RoboDojoBenchmarkError(f"runtime.{key} must be {expected!r}")
    if runtime.get("contact_approach_route", CONTACT_APPROACH_STAGED) != CONTACT_APPROACH_STAGED:
        raise RoboDojoBenchmarkError(f"runtime.contact_approach_route must be {CONTACT_APPROACH_STAGED!r} or absent")
    if runtime.get("recovery_route", "retract_then_unwind") != "retract_then_unwind":
        raise RoboDojoBenchmarkError("runtime.recovery_route must be 'retract_then_unwind' or absent")
    if runtime.get("policy_cameras") != list(PUBLIC_CAMERAS):
        raise RoboDojoBenchmarkError("RoboDojo requires the qualified three-camera order")
    profile = runtime.get("workspace_profile")
    if profile is not None and (
        not isinstance(profile, Mapping)
        or profile.get("profile_id") != ROBODOJO_ARX_X5_TABLETOP_V1
        or profile.get("environment_id") != "robodojo"
    ):
        raise RoboDojoBenchmarkError(
            "RoboDojo admits only the source-owned "
            f"{ROBODOJO_ARX_X5_TABLETOP_V1} action workspace profile"
        )
    device = runtime.get("cuda_visible_devices")
    if not isinstance(device, str) or re.fullmatch(r"[0-7]", device) is None:
        raise RoboDojoBenchmarkError("runtime.cuda_visible_devices must select one GPU")
    if type(runtime.get("renderer_gpu")) is not int or runtime["renderer_gpu"] != int(device):
        raise RoboDojoBenchmarkError("runtime.renderer_gpu must select the physical GPU")
    for field in ("startup_timeout_s", "request_timeout_s", "close_timeout_s"):
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RoboDojoBenchmarkError(f"runtime.{field} must be positive and finite")
    if runtime["close_timeout_s"] != 30:
        raise RoboDojoBenchmarkError("RoboDojo close_timeout_s must be 30")
    path = runtime.get("layout_file")
    if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
        raise RoboDojoBenchmarkError("layout_file must be relative to the pinned Assets root")
    if re.fullmatch(rf"Eval_Layout/RoboDojo/arx_x5/0/{re.escape(task)}_\d+\.json", path) is None:
        raise RoboDojoBenchmarkError("layout_file does not belong to the selected task/seed")
    if re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("layout_sha256", ""))) is None:
        raise RoboDojoBenchmarkError("RoboDojo requires layout_sha256")
    if runtime.get("layout_schedule") is None:
        # One episode on one official layout ordinal (layout 0 by default);
        # whether the bytes exist and match is the preflight's and worker's.
        from .robodojo_layouts import layout_file_for
        layout_id = runtime.get("layout_id")
        if type(layout_id) is not int or layout_id < 0 or path != layout_file_for(task, 0, layout_id):
            raise RoboDojoBenchmarkError(
                "runtime.layout_id must be one official layout ordinal whose file is layout_file "
                "without a reviewed layout_schedule")
    else:
        from .robodojo_layouts import validate_layout_schedule
        validate_layout_schedule(
            runtime["layout_schedule"], task=task, eval_seed=0, layout_id=runtime.get("layout_id"),
            layout_file=path, layout_sha256=runtime.get("layout_sha256"),
        )
    public = runtime.get("public_environment_context")
    if public is not None and public != public_context():
        raise RoboDojoBenchmarkError("RoboDojo public environment context drift")
    validate_qualification_binding(runtime.get("infrastructure_qualification"))


def selected_layout(asset_root: Path, *, task: str, eval_seed: int, layout_id: int) -> Path:
    """Mirror the upstream numeric-sort ordinal without loading private state.

    Only ``<task>_<n>.json`` files count, exactly as ``SeedManager.init_eval``
    matches them, so ``<task>_random_<n>.json`` (another task module) never
    enters the ordinal. Which ordinals a RunPlan may select is decided by
    ``validate_runtime_extension`` and the reviewed layout catalog.
    """
    if (task not in NATIVE_ACTION_LIMITS or type(eval_seed) is not int or eval_seed != 0
            or type(layout_id) is not int or layout_id < 0):
        raise RoboDojoBenchmarkError(
            "only the reviewed tasks / eval_seed 0 / a non-negative layout ordinal are admitted"
        )
    folder = asset_root / "Eval_Layout" / "RoboDojo" / "arx_x5" / str(eval_seed)
    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    files = sorted(
        (p for p in folder.glob(f"{task}_*.json") if pattern.fullmatch(p.name)),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if layout_id >= len(files):
        raise RoboDojoBenchmarkError("the selected official layout is unavailable")
    selected = files[layout_id].resolve()
    if not selected.is_relative_to(asset_root.resolve()) or not selected.is_file():
        raise RoboDojoBenchmarkError("layout escapes the pinned Assets root")
    return selected


# --- Round-ending protocol comparability -------------------------------------
#
# Upstream RoboDojo has exactly three ways to end an episode: the reward
# manager's own success latch, its failure latch (``RewardManager.
# _mark_env_failed`` clears ``success`` on a task's permanent failure predicate,
# and ``EvalEnv`` then ends the episode false before the limit), and the task's
# native action limit. It has no notion of a policy conceding, and no notion of
# a Harness decision or model-call allowance. A round that ended any other way
# is a Harness-invented truncation with no upstream counterpart, and pooling it
# into a rate with a round that ran to one of upstream's own limits silently
# changes what the rate measures. The failure predicates' thresholds are
# private; only which condition ended the round is recorded.
#
# One sentence a reader can apply: **a RoboDojo round is protocol-comparable
# only when it ended on one of upstream's own ending conditions -- the official
# success latch, the official failure latch or the native action limit -- and
# any reported RoboDojo rate must group rounds by ``round_end_basis`` instead of averaging
# across it, so a round the policy ended with finish_episode never shares a
# denominator with a round that ran to a limit.**
ROUND_END_BASES = (
    "official_terminal_transition",
    "policy_finished_episode",
    "harness_budget_exhausted",
    "harness_boundary_other",
    "operator_interrupt",
    "infrastructure_fault",
    "episode_incomplete",
)
UPSTREAM_ENDING_CONDITIONS = (
    "official_success_latch", "official_failure_latch", "native_action_limit",
)
#: Control-only environment status reason for each upstream ending condition.
#: Before the failure latch was named, both false endings shared
#: ``official_native_action_limit_or_failure``; it survives only for a terminal
#: record whose native-action counts are unreadable.
OFFICIAL_TERMINAL_REASONS = {
    "official_success_latch": "official_success",
    "official_failure_latch": "official_failure_latch",
    "native_action_limit": "official_native_action_limit",
    "official_episode_end": "official_native_action_limit_or_failure",
}
#: The Supervisor's own round boundaries. Every one of them stops a round
#: before upstream would have stopped it, so none of them is comparable.
HARNESS_BUDGET_REASONS = (
    "memory_teacher_soft_horizon",
    "model_call_budget",
    "robot_decision_budget",
    "simulation_step_cap",
    "wall_time_budget",
)
POLICY_LIFECYCLE_REASONS = ("policy_finished_episode",)
OPERATOR_REASONS = ("operator_interrupt",)


def classify_round_ending(
    *,
    official_terminal: bool,
    fault_reason: str | None,
    abort_reason: str | None,
    official_success: bool | None = None,
    native_actions: int | None = None,
    native_action_limit: int | None = None,
) -> dict[str, Any]:
    """Name how one RoboDojo round ended and whether that is comparable.

    Conservative by construction: only the simulator's own terminal transition
    is comparable, and an ending this table does not recognize is
    ``harness_boundary_other`` with ``protocol_comparable`` false rather than
    an optimistic default. ``protocol_comparable`` is null, never false, when
    the round produced no behavioural result at all -- an infrastructure fault,
    an operator interrupt or an episode that had not stopped -- because those
    are excluded from every denominator anyway and calling them "not
    comparable" would imply they were a truncated behaviour sample.
    """

    if official_terminal:
        counted = isinstance(native_actions, int) and isinstance(
            native_action_limit, int,
        )
        condition = (
            "official_success_latch"
            if official_success is True
            else "native_action_limit"
            if counted and native_actions >= native_action_limit
            else "official_failure_latch"
            if counted and official_success is False
            # Unreadable counts cannot attribute a false ending.
            else "official_episode_end"
        )
        return {
            "round_end_basis": "official_terminal_transition",
            "protocol_comparable": True,
            "upstream_ending_condition": condition,
            "comparability_basis": (
                "the simulator declared the episode over on its own predicate"
            ),
        }
    if fault_reason is not None:
        basis, comparable, note = (
            "infrastructure_fault", None,
            "an infrastructure fault is never a behavioural result",
        )
    elif abort_reason in OPERATOR_REASONS:
        basis, comparable, note = (
            "operator_interrupt", None,
            "an operator interrupt is never a behavioural result",
        )
    elif abort_reason is None:
        basis, comparable, note = (
            "episode_incomplete", None,
            "the episode had not stopped when the verdict was read",
        )
    elif abort_reason in POLICY_LIFECYCLE_REASONS:
        basis, comparable, note = (
            "policy_finished_episode", False,
            "the policy ended the round itself; upstream has no such ending",
        )
    elif abort_reason in HARNESS_BUDGET_REASONS:
        basis, comparable, note = (
            "harness_budget_exhausted", False,
            f"a Harness allowance ({abort_reason}) stopped the round before "
            "either upstream ending condition",
        )
    else:
        basis, comparable, note = (
            "harness_boundary_other", False,
            f"an unclassified Harness boundary ({abort_reason}) stopped the "
            "round; unrecognized endings are never treated as comparable",
        )
    return {
        "round_end_basis": basis,
        "protocol_comparable": comparable,
        "upstream_ending_condition": None,
        "comparability_basis": note,
    }


def official_terminal_reason(
    official_terminal: Mapping[str, Any], native_action_limit: int | None,
) -> str:
    """Name the upstream condition behind a terminal RoboDojo status."""

    native_actions = official_terminal.get("native_actions")
    ending = classify_round_ending(
        official_terminal=True, fault_reason=None, abort_reason=None,
        official_success=bool(official_terminal.get("official_success")),
        native_actions=None if native_actions is None else int(native_actions),
        native_action_limit=native_action_limit,
    )
    return OFFICIAL_TERMINAL_REASONS[ending["upstream_ending_condition"]]
