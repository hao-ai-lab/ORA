"""Closed-contract tests for BDDL-driven LIBERO-Pro robustness."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from ora.environment.backends.libero_pro.robustness import (
    LiberoProRobustnessError,
    RobustnessEnvironment,
    validate_robustness_config,
)


def _config(case: str) -> dict:
    value = {
        "case": case,
        "category": case,
        "source_suite": "libero_object",
        "source_task": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "bddl_only": True,
        "detail": "fixture",
        "lighting": {
            "enabled": False, "mode": "none", "ambient": 1.0,
            "diffuse": 1.0, "specular": 1.0,
        },
        "observation": {
            "enabled": False, "mode": "none", "brightness": 1.0,
            "sigma": 0.0,
        },
        "camera": {
            "enabled": False, "mode": "none", "name": "agentview",
            "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
        },
        "runtime_move": {
            "enabled": False, "mode": "none", "object": "alphabet_soup_1",
            "step": 0, "delta_xy": [0.0, 0.0], "min_delta_norm": 0.0,
        },
        "object_appearance": {
            "enabled": False, "mode": "none", "targets": "basket_1",
            "rgba": [0.3, 0.3, 0.3, 1.0],
            "override_material_rgba": False,
            "disable_material_texture": False,
            "apply_visible_geom_rgba": False,
        },
        "object_placement": {
            "enabled": False, "mode": "none", "reference": "alphabet_soup_1",
            "front_direction": [1.0, 0.0], "targets": "occluder_1",
            "offsets_xy": [0.0, 0.0], "target_quat": [1.0, 0.0, 0.0, 0.0],
            "z_offset": 0.0, "mesh_scale": [1.0, 1.0, 1.0],
            "min_reference_distance": 0.0, "pin_targets": False,
        },
        "object_shape": {
            "enabled": False, "mode": "none", "target": "alphabet_soup_1",
            "mesh_scale": [1.0, 1.0, 1.0],
        },
        "initial_pose": {
            "enabled": False, "mode": "none", "target": "alphabet_soup_1",
            "delta_xy": [0.0, 0.0], "min_delta_norm": 0.0, "yaw": 0.0,
        },
    }
    if case == "visual_noise_glare":
        value["lighting"].update(enabled=True, mode="dark_lighting")
        value["observation"].update(
            enabled=True, mode="dark_noise", brightness=0.68, sigma=12.0,
        )
    elif case == "runtime_object_move":
        value["runtime_move"].update(
            enabled=True, mode="qpos_delta_xy", trigger="near_grasp",
            distance_threshold=0.09, fallback_step=160,
            delta_xy=[0.17, 0.06], min_delta_norm=0.18,
        )
    elif case == "camera_view_angle":
        value["camera"].update(
            enabled=True, mode="agentview_angle", pos=[0.7, 0.0, 1.4],
            quat=[0.5, 0.5, 0.5, 0.5],
        )
    elif case == "object_texture":
        value["object_appearance"].update(
            enabled=True, mode="table_like_material",
            override_material_rgba=True, disable_material_texture=True,
            apply_visible_geom_rgba=True,
        )
    elif case == "view_occlusion":
        value["object_placement"].update(
            enabled=True, mode="front_occlusion", pin_targets=True,
            offsets_xy=[[0.1, 0.0]],
        )
    elif case == "object_shape":
        value["object_shape"].update(
            enabled=True, mode="mesh_scale", mesh_scale=[1.1, 0.9, 1.0],
        )
    elif case == "initial_pose_position_angle":
        value["initial_pose"].update(
            enabled=True, mode="delta_xy_yaw", delta_xy=[0.08, -0.03],
            min_delta_norm=0.08, yaw=0.35,
        )
    return value


def _validate(config: dict, case: str) -> dict:
    return validate_robustness_config(
        config, case=case, suite="libero_object",
        task_name="pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    )


def test_robustness_config_is_closed_and_source_bound() -> None:
    value = _config("visual_noise_glare")
    assert _validate(value, "visual_noise_glare") == value
    unknown = deepcopy(value)
    unknown["observation"]["silent_default"] = True
    with pytest.raises(LiberoProRobustnessError, match="invalid fields"):
        _validate(unknown, "visual_noise_glare")
    drift = deepcopy(value)
    drift["source_task"] = "another_task"
    with pytest.raises(LiberoProRobustnessError, match="suite/task binding"):
        _validate(drift, "visual_noise_glare")


def test_runtime_move_preserves_nine_centimeter_and_step_160_contract() -> None:
    value = _config("runtime_object_move")
    assert _validate(value, "runtime_object_move")["runtime_move"] == value[
        "runtime_move"
    ]
    value["runtime_move"]["distance_threshold"] = 0.091
    with pytest.raises(LiberoProRobustnessError, match=r"\(0,0.09\]"):
        _validate(value, "runtime_object_move")
    value = _config("runtime_object_move")
    value["runtime_move"]["fallback_step"] = 159
    with pytest.raises(LiberoProRobustnessError, match="fallback_step"):
        _validate(value, "runtime_object_move")


@pytest.mark.parametrize(
    "case",
    (
        "visual_noise_glare", "camera_view_angle", "runtime_object_move",
        "object_texture", "view_occlusion", "object_shape",
        "initial_pose_position_angle",
    ),
)
def test_all_seven_robustness_case_schemas_are_admitted(case: str) -> None:
    assert _validate(_config(case), case)["case"] == case


def test_occlusion_init_inserts_free_joint_without_assuming_suffix_order() -> None:
    model = SimpleNamespace(
        nq=14, nv=12, jnt_type=np.asarray([0]),
        jnt_qposadr=np.asarray([0]), jnt_dofadr=np.asarray([0]),
        joint_name2id=lambda name: 0 if name == "occluder_joint" else -1,
    )
    current = np.arange(27, dtype=np.float64) + 100.0
    delegate = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(model=model),
            objects_dict={
                "occluder_1": SimpleNamespace(joints=["occluder_joint"]),
            },
        ),
        get_sim_state=lambda: current.copy(),
    )
    environment = RobustnessEnvironment.__new__(RobustnessEnvironment)
    environment._case = "view_occlusion"
    environment._delegate = delegate
    environment._config = {
        "object_placement": {"targets": ["occluder_1"]},
    }
    environment._init_state_adaptation = None
    environment._model_recompile = None
    source = np.arange(14, dtype=np.float64)
    adapted = environment._adapt_shared_init_state(source)
    assert np.array_equal(adapted[1:8], current[1:8])
    assert np.array_equal(adapted[8:15], source[1:8])
    assert np.array_equal(adapted[15:21], current[15:21])
    assert np.array_equal(adapted[21:27], source[8:14])
    assert environment._init_state_adaptation["mode"] == (
        "insert_bddl_reset_free_joints_v1"
    )


def test_mesh_scale_is_compiled_into_mjcf_before_simulation() -> None:
    model = SimpleNamespace(
        njnt=1, nq=1, nv=1, jnt_qposadr=np.asarray([0]),
        jnt_dofadr=np.asarray([0]), joint_id2name=lambda joint_id: "joint_0",
        nbody=1, body_id2name=lambda body_id: "world",
        body_name2id=lambda name: 0,
        body_pos=np.asarray([[1.0, 2.0, 3.0]]),
        body_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        mesh_scale=np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64),
        mesh_id2name=lambda mesh_id: "target_mesh" if mesh_id == 0 else None,
        mesh_name2id=lambda name: 0 if name == "target_mesh" else -1,
    )
    task_model = SimpleNamespace(
        get_xml=lambda: (
            '<mujoco><asset><mesh name="target_mesh" '
            'scale="0.01 0.02 0.03"/></asset></mujoco>'
        ),
    )
    inner = SimpleNamespace(
        model=task_model,
        sim=SimpleNamespace(model=model, forward=lambda: None),
    )

    def reset_from_xml_string(xml: str) -> None:
        mesh = ElementTree.fromstring(xml).find("asset/mesh")
        model.mesh_scale[0] = np.fromstring(mesh.get("scale"), sep=" ")
        model.body_pos[0] = 0.0

    environment = RobustnessEnvironment.__new__(RobustnessEnvironment)
    environment._case = "object_shape"
    environment._config = {
        "object_shape": {
            "target": "target_1", "mesh_scale": [2.0, 0.5, 3.0],
        },
    }
    environment._runtime = SimpleNamespace(
        body_mesh_ids=lambda observed, target: [0]
        if observed is inner and target == "target_1" else [],
    )
    environment._delegate = SimpleNamespace(
        env=inner, reset_from_xml_string=reset_from_xml_string,
        get_sim_state=lambda: np.asarray([0.0, 0.5, 0.0]),
        set_state=lambda state: None, _post_process=lambda: None,
        _update_observables=lambda force: None,
    )
    evidence = environment._recompile_configured_mesh_scale()
    assert np.allclose(model.mesh_scale[0], [0.02, 0.01, 0.09])
    assert np.allclose(model.body_pos[0], [1.0, 2.0, 3.0])
    assert evidence["configured_mesh_assets_recompiled"] is True
    assert evidence["meshes"][0]["verified"] == pytest.approx(
        [0.02, 0.01, 0.09]
    )


def test_recompiled_init_state_maps_by_joint_name_across_reordering() -> None:
    source_layout = [
        {"joint": "target_free", "qpos_address": 0, "qpos_width": 7,
         "qvel_address": 0, "qvel_width": 6},
        {"joint": "joint_a", "qpos_address": 7, "qpos_width": 1,
         "qvel_address": 6, "qvel_width": 1},
        {"joint": "joint_b", "qpos_address": 8, "qpos_width": 1,
         "qvel_address": 7, "qvel_width": 1},
    ]
    current_layout = [
        {"joint": "joint_b", "qpos_address": 0, "qpos_width": 1,
         "qvel_address": 0, "qvel_width": 1},
        {"joint": "target_free", "qpos_address": 1, "qpos_width": 7,
         "qvel_address": 1, "qvel_width": 6},
        {"joint": "joint_a", "qpos_address": 8, "qpos_width": 1,
         "qvel_address": 7, "qvel_width": 1},
    ]
    current = np.arange(18, dtype=np.float64) + 100.0
    environment = RobustnessEnvironment.__new__(RobustnessEnvironment)
    environment._case = "view_occlusion"
    environment._config = {
        "object_placement": {"target": "occluder_1"},
    }
    environment._source_joint_layout = source_layout
    environment._recompiled_joint_layout = current_layout
    environment._model_recompile = {"mode": "fixture"}
    environment._delegate = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(nq=9, nv=8)),
            objects_dict={
                "occluder_1": SimpleNamespace(joints=["target_free"]),
            },
        ),
        get_sim_state=lambda: current.copy(),
    )
    source = np.asarray([0.5, 10.0, 20.0, 30.0, 40.0])
    adapted = environment._adapt_shared_init_state(source)
    assert adapted[1] == 20.0
    assert adapted[9] == 10.0
    assert adapted[10] == 40.0
    assert adapted[17] == 30.0
    assert np.array_equal(adapted[2:9], current[2:9])
    assert np.array_equal(adapted[11:17], current[11:17])
    assert environment._init_state_adaptation["reordered_joint_count"] == 2


def test_official_core_evaluation_is_sixteen_suites_at_fifty_states():
    from ora.environment.benchmarks.libero_pro import (
        CORE_SUITES, LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION,
        LiberoProBenchmarkError, OFFICIAL_EVALUATION_SCHEMA, official_core_evaluation,
        validate_runtime_extension,
    )

    protocol = official_core_evaluation()
    assert protocol["schema"] == OFFICIAL_EVALUATION_SCHEMA
    assert protocol["libero_pro_code_revision"] == LIBERO_PRO_CODE_REVISION
    assert protocol["libero_pro_data_revision"] == LIBERO_PRO_DATA_REVISION
    cells = protocol["cells"]
    assert protocol["episode_count"] == len(cells) == 4 * 4 * 10 * 50 == 8000
    assert {cell["suite"] for cell in cells} == set(CORE_SUITES)
    assert {cell["label"] for cell in cells} == {"Obj", "Pos", "Sem", "Task"}
    assert len({(c["suite"], c["task_id"], c["seed"]) for c in cells}) == len(cells)
    assert {cell["seed"] for cell in cells} == set(range(50))
    assert "env" in protocol["excluded"]
    for cell in cells[::997]:
        validate_runtime_extension({"suite": cell["suite"], "task_id": cell["task_id"]})
    subset = official_core_evaluation(families=["object"], perturbations=["position"], init_states=10)
    assert subset["episode_count"] == 100
    assert [c["suite"] for c in subset["cells"][:1]] == ["libero_object_swap"]
    for kwargs in ({"families": ["object", "object"]}, {"families": ["kitchen"]},
                   {"perturbations": ["env"]}, {"init_states": 0}, {"init_states": 51},
                   {"init_states": True}):
        with pytest.raises(LiberoProBenchmarkError):
            official_core_evaluation(**kwargs)
