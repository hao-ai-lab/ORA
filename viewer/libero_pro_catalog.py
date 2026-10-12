"""Pinned, read-only LIBERO / LIBERO-Pro task and Runtime identity catalog.

The catalog preserves the complete official 130-task hierarchy and benchmark
order without loading the simulator or an external dataset at Viewer runtime.
The 40 source-paired LIBERO-Pro anchors additionally retain exact BDDL
instructions and Runtime status; the Viewer never invents missing variants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import quote

from ora.environment.benchmarks.libero import LIBERO_BASE_SUITES
from ora.environment.benchmarks.libero_pro import (
    CORE_VARIANT_ORDER as PERTURBATION_ORDER,
    CORE_VARIANT_SUFFIXES,
    FAMILY_ORDER as PRO_FAMILY_ORDER,
    LIBERO_PRO_CODE_REVISION,
    LIBERO_PRO_DATA_REVISION,
    ROBUSTNESS_CASE_FOLDERS,
    base_suite,
    core_suite,
    robustness_suite,
)
from ora.environment.benchmarks.libero_pro_language import TASK_ROWS

CATALOG_SCHEMA = "ora.libero-pro-task-catalog.v2"
MIGRATED_FROM_COMMIT = "a762defefdecff6286ba3079cb02a77cd5327141"
CODE_ROOT = (
    "https://github.com/Zxy-MLlab/LIBERO-PRO/tree/"
    f"{LIBERO_PRO_CODE_REVISION}"
)
DATA_ROOT = (
    "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/blob/"
    f"{LIBERO_PRO_DATA_REVISION}"
)

FAMILY_ORDER = ("spatial", "object", "goal", "libero_90", "libero_10")
COLLECTION_ORDER = ("controlled_transfer", "libero_100")
COLLECTIONS = {
    "controlled_transfer": {
        "label_zh": "受控知识迁移",
        "label_en": "Controlled transfer",
        "description": "三个十任务套件分别隔离空间、物体和目标变化。",
    },
    "libero_100": {
        "label_zh": "LIBERO-100",
        "label_en": "Entangled transfer",
        "description": "90 个短程预训练任务与 10 个相对长程下游任务。",
    },
}
FAMILY_LABELS = {
    "spatial": {
        "label_zh": "空间关系",
        "label_en": "LIBERO-Spatial",
        "description": "同类黑碗由相对位置区分，目标统一为放到盘子上。",
        "collection": "controlled_transfer",
        "task_range": "task-00—09",
        "role": "spatial-relation transfer",
        "pro_supported": True,
    },
    "object": {
        "label_zh": "物体识别",
        "label_en": "LIBERO-Object",
        "description": "同一篮筐场景中改变要操作的食品物体。",
        "collection": "controlled_transfer",
        "task_range": "task-00—09",
        "role": "object-identity transfer",
        "pro_supported": True,
    },
    "goal": {
        "label_zh": "目标技能",
        "label_en": "LIBERO-Goal",
        "description": "同一厨房场景中改变抽屉、容器或目标状态。",
        "collection": "controlled_transfer",
        "task_range": "task-00—09",
        "role": "goal/procedure transfer",
        "pro_supported": True,
    },
    "libero_90": {
        "label_zh": "短程预训练",
        "label_en": "LIBERO-90",
        "description": "LIBERO-100 中用于预训练的 90 个短程厨房、客厅和书房任务。",
        "collection": "libero_100",
        "task_range": "task-00—89",
        "role": "short-horizon pretraining split",
        "pro_supported": False,
    },
    "libero_10": {
        "label_zh": "相对长程",
        "label_en": "LIBERO-Long / LIBERO-10",
        "description": "LIBERO-100 的十个多阶段下游任务。",
        "collection": "libero_100",
        "task_range": "task-00—09",
        "role": "long-horizon downstream split",
        "pro_supported": True,
    },
}

PERTURBATIONS = {
    "position": {
        "suffix": CORE_VARIANT_SUFFIXES["position"],
        "label_zh": "位置",
        "label_en": "Position",
        "changes": "初始物体位置；任务目标和语言通常保持不变。",
        "changed_fields": ["initial_state"],
    },
    "language": {
        "suffix": CORE_VARIANT_SUFFIXES["language"],
        "label_zh": "语义",
        "label_en": "Language",
        "changes": "改写自然语言指令；任务目标保持不变。",
        "changed_fields": ["instruction"],
    },
    "object": {
        "suffix": CORE_VARIANT_SUFFIXES["object"],
        "label_zh": "物体外观",
        "label_en": "Object appearance",
        "changes": "改变物体外观、颜色或尺度；语言通常保持不变。",
        "changed_fields": ["appearance", "color", "scale"],
    },
    "task": {
        "suffix": CORE_VARIANT_SUFFIXES["task"],
        "label_zh": "任务",
        "label_en": "Task logic",
        "changes": "改变指令、目标状态或目标物体；同一 task_id 不再代表同一目标。",
        "changed_fields": ["instruction", "goal", "target_object"],
    },
}

LEGACY_ENVIRONMENT = {
    "id": "environment",
    "label_zh": "环境",
    "label_en": "Environment",
    "changes": "替换工作环境；当前固定数据没有独立 *_env BDDL，需上游运行时生成。",
    "runtime_registered": False,
    "static_assets": False,
}

_ROBUSTNESS_PRESENTATION = {
    "visual_noise_glare": {
        "label_zh": "光照与视觉噪声",
        "label_en": "Lighting + observation noise",
        "changes": "改变照明，并向观测加入 glare / noise。",
    },
    "camera_view_angle": {
        "label_zh": "相机视角",
        "label_en": "Camera view angle",
        "changes": "改变相机位置与朝向。",
    },
    "runtime_object_move": {
        "label_zh": "运行中移动目标",
        "label_en": "Runtime target movement",
        "changes": "接近抓取时移动目标；0.09 m 近抓取触发，step 160 兜底。",
    },
    "object_texture": {
        "label_zh": "物体纹理",
        "label_en": "Object texture",
        "changes": "改变任务相关物体的外观与纹理。",
    },
    "view_occlusion": {
        "label_zh": "视野遮挡",
        "label_en": "View occlusion",
        "changes": "用场景物体遮挡关键目标或目标区域。",
    },
    "object_shape": {
        "label_zh": "物体形状",
        "label_en": "Object shape",
        "changes": "缩放目标物体形状，改变视觉与抓取几何。",
    },
    "initial_pose_position_angle": {
        "label_zh": "初始位姿",
        "label_en": "Initial pose + yaw",
        "changes": "改变初始平面位置与偏航角。",
    },
}
ROBUSTNESS_CASES = tuple(
    {
        "id": folder,
        "case": case,
        **_ROBUSTNESS_PRESENTATION[case],
    }
    for case, folder in ROBUSTNESS_CASE_FOLDERS.items()
)


@dataclass(frozen=True)
class CoreVariant:
    perturbation: str
    suite: str
    instruction: str
    changes: str
    changed_fields: tuple[str, ...]
    bddl_url: str
    init_url: str
    runtime_registered: bool = True


@dataclass(frozen=True)
class LiberoProTask:
    family: str
    task_id: int
    name: str
    scene: str | None
    canonical_instruction: str
    core_variants: tuple[CoreVariant, ...]

    @property
    def base_suite(self) -> str:
        return self.family if self.family.startswith("libero_") else f"libero_{self.family}"

    @property
    def collection(self) -> str:
        return str(FAMILY_LABELS[self.family]["collection"])

    @property
    def pro_anchor(self) -> bool:
        return bool(FAMILY_LABELS[self.family]["pro_supported"])

    @property
    def base_runtime_supported(self) -> bool:
        return self.base_suite in LIBERO_BASE_SUITES

    @property
    def key(self) -> str:
        return f"{self.base_suite}:task-{self.task_id:02d}"

    def variant(self, perturbation: str) -> CoreVariant:
        if not self.pro_anchor:
            raise KeyError(f"{self.base_suite} is catalog-only and has no LIBERO-Pro core variants")
        return next(item for item in self.core_variants if item.perturbation == perturbation)


# family, task_id, canonical filename, task/swap/language/object BDDL language,
# in benchmark order; owned by ora/environment/benchmarks/libero_pro_language.py.
_TASK_ROWS = TASK_ROWS

# Official benchmark order from LIBERO's pinned suite task map.  LIBERO-90 is
# part of the original 130-task catalog, but it is not a source-paired
# LIBERO-Pro perturbation family in the active dataset.
_LIBERO_90_NAMES = (
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE6_close_the_microwave",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    "KITCHEN_SCENE7_open_the_microwave",
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
    "KITCHEN_SCENE8_turn_off_the_stove",
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_turn_on_the_stove",
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
)

# Four pinned BDDL language fields intentionally differ from their scene
# filenames.  Keep the exact policy-facing text rather than normalizing it from
# the filename; the other 86 language fields are exact underscore-to-space
# projections of their official task names.
_LIBERO_90_LANGUAGE_OVERRIDES = {
    14: "put the black bowl in the middle on the plate",
    17: "stack the black bowl in the middle on the black bowl at the front",
    84: "pick up the red mug and place it to the right compartment of the caddy",
    85: "pick up the white mug and place it to the right compartment of the caddy",
}


def _scene(name: str) -> str | None:
    prefix = name.partition("_")[0]
    if prefix not in {"KITCHEN", "LIVING", "STUDY"}:
        return None
    # LIVING_ROOM_SCENE2 needs three components; KITCHEN/STUDY need two.
    parts = name.split("_")
    count = 3 if parts[:2] == ["LIVING", "ROOM"] else 2
    candidate = "_".join(parts[:count])
    return candidate.replace("_", " ") if "SCENE" in candidate else None


def _asset_url(*parts: str) -> str:
    return "/".join((DATA_ROOT, *(quote(part, safe="_") for part in parts)))


def _build_task(row: tuple) -> LiberoProTask:
    family, task_id, name, task_language, swap_language, lan_language, object_language = row
    instructions = {
        "task": task_language,
        "swap": swap_language,
        "lan": lan_language,
        "object": object_language,
    }
    variants = []
    for perturbation in PERTURBATION_ORDER:
        definition = PERTURBATIONS[perturbation]
        suite = core_suite(family, perturbation)
        variants.append(CoreVariant(
            perturbation=perturbation,
            suite=suite,
            instruction=instructions[definition["suffix"]],
            changes=definition["changes"],
            changed_fields=tuple(definition["changed_fields"]),
            bddl_url=_asset_url("bddl_files", suite, f"{name}.bddl"),
            init_url=_asset_url("init_files", suite, f"{name}.pruned_init"),
        ))
    return LiberoProTask(
        family=family,
        task_id=task_id,
        name=name,
        scene=_scene(name),
        canonical_instruction=swap_language,
        core_variants=tuple(variants),
    )


def _instruction_from_scene_name(name: str) -> str:
    parts = name.split("_")
    try:
        scene_index = next(
            index for index, part in enumerate(parts) if part.startswith("SCENE")
        )
    except StopIteration:
        return " ".join(parts)
    return " ".join(parts[scene_index + 1 :])


def _build_libero_90_task(task_id: int, name: str) -> LiberoProTask:
    return LiberoProTask(
        family="libero_90",
        task_id=task_id,
        name=name,
        scene=_scene(name),
        canonical_instruction=_LIBERO_90_LANGUAGE_OVERRIDES.get(
            task_id, _instruction_from_scene_name(name),
        ),
        core_variants=(),
    )


TASKS = tuple(sorted(
    (
        *(_build_task(row) for row in _TASK_ROWS),
        *(
            _build_libero_90_task(task_id, name)
            for task_id, name in enumerate(_LIBERO_90_NAMES)
        ),
    ),
    key=lambda task: (FAMILY_ORDER.index(task.family), task.task_id),
))
TASK_BY_KEY = {(task.family, task.task_id): task for task in TASKS}


def robustness_assets(task: LiberoProTask) -> list[dict]:
    """Return the seven registered BDDL-driven Runtime cells for one task."""

    if not task.pro_anchor:
        return []
    runtime_suite = base_suite(task.family)
    return [
        {
            **case,
            "bddl_url": _asset_url(
                "bddl_files", case["id"], "bddl", runtime_suite,
                f"{task.name}.bddl",
            ),
            "init_source": f"init_files/{runtime_suite}/{task.name}.pruned_init",
            "runtime_suite": runtime_suite,
            "experiment_config": robustness_suite(task.family, case["case"]),
            "runtime_registered": True,
            "status": "runtime_registered_bddl_config",
            "init_state_contract": (
                "shared_clean_state_with_named_occluder_joint_insertion_v2"
                if case["case"] == "view_occlusion" else "shared_clean_state"
            ),
        }
        for case in ROBUSTNESS_CASES
    ]


def task_payload(task: LiberoProTask) -> dict:
    family = FAMILY_LABELS[task.family]
    return {
        "key": task.key,
        "family": task.family,
        "suite": task.base_suite,
        "collection": task.collection,
        "collection_label": COLLECTIONS[task.collection]["label_en"],
        "suite_label": family["label_en"],
        "suite_role": family["role"],
        "base_runtime_supported": task.base_runtime_supported,
        "pro_anchor": task.pro_anchor,
        "task_id": task.task_id,
        "name": task.name,
        "scene": task.scene,
        "canonical_instruction": task.canonical_instruction,
        "core_variants": [asdict(item) for item in task.core_variants],
        "robustness_assets": robustness_assets(task),
    }


def catalog_payload() -> dict:
    """Return the stable read-only JSON representation served by the Viewer."""

    return {
        "schema": CATALOG_SCHEMA,
        "libero_pro_code_revision": LIBERO_PRO_CODE_REVISION,
        "libero_pro_data_revision": LIBERO_PRO_DATA_REVISION,
        "migrated_from": {
            "repository": "ora-agent",
            "commit": MIGRATED_FROM_COMMIT,
        },
        "evidence_boundary": {
            "kind": "read_only_catalog_with_runtime_identity",
            "contains_demonstrations": False,
            "contains_policy_rollouts": False,
            "runtime_registered_base_suites": len(LIBERO_BASE_SUITES),
            "runtime_registered_core_suites": 16,
            "runtime_registered_robustness_suites": 28,
            "robustness_extension_runtime_registered": True,
        },
        "counts": {
            "base_tasks": len(TASKS),
            "controlled_transfer_tasks": 30,
            "libero_100_tasks": 100,
            "libero_90_tasks": 90,
            "libero_long_tasks": 10,
            "libero_pro_anchor_tasks": sum(task.pro_anchor for task in TASKS),
            "core_configurations": sum(
                len(task.core_variants) for task in TASKS
            ),
            "core_suites": len(PRO_FAMILY_ORDER) * len(PERTURBATION_ORDER),
            "robustness_cases": len(ROBUSTNESS_CASES),
            "robustness_configurations": sum(
                len(robustness_assets(task)) for task in TASKS
            ),
            "init_states_per_task": 50,
        },
        "collections": [
            {"id": identifier, **COLLECTIONS[identifier]}
            for identifier in COLLECTION_ORDER
        ],
        "families": [
            {"id": identifier, **FAMILY_LABELS[identifier]}
            for identifier in FAMILY_ORDER
        ],
        "perturbations": [
            {"id": identifier, **PERTURBATIONS[identifier]}
            for identifier in PERTURBATION_ORDER
        ],
        "legacy_environment": LEGACY_ENVIRONMENT,
        "robustness_cases": list(ROBUSTNESS_CASES),
        "tasks": [task_payload(task) for task in TASKS],
    }


if len(TASKS) != 130 or len(TASK_BY_KEY) != 130:
    raise RuntimeError("LIBERO gallery snapshot must contain 130 unique base tasks")
for family, expected in {
    "spatial": 10, "object": 10, "goal": 10,
    "libero_90": 90, "libero_10": 10,
}.items():
    task_ids = [task.task_id for task in TASKS if task.family == family]
    if task_ids != list(range(expected)):
        raise RuntimeError(f"{family} must preserve official task order")
if sum(task.pro_anchor for task in TASKS) != 40:
    raise RuntimeError("LIBERO-Pro must retain exactly 40 source-paired anchors")
if len({variant.suite for task in TASKS for variant in task.core_variants}) != 16:
    raise RuntimeError("LIBERO-Pro gallery snapshot must contain 16 core suites")
