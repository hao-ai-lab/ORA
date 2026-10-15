"""V2 source compilation, before-policy geometry gates and fresh defaults."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.composition.embodiment import (
    EmbodimentDescriptorError, audit_embodiment_sources, load_embodiment_descriptor,
    render_embodiment_card, validate_embodiment_descriptor,
)
from ora.agent.composition.embodiment_geometry import load_public_geometry_binding, public_local_axes
from ora.agent.composition.tool_surface import resolve_tool_surface, validate_schema_instance
from ora.contracts.robot_geometry import verify_loaded_robot_geometry
from ora.orchestrator.spec import resolve_run_spec
from tests.support.run_specs import native_spec

ROOT = Path(__file__).resolve().parents[3]


DEFAULT_SURFACES = {"libero": "libero_placement_direct_overhead_v1", "robotwin": "robotwin_toolchain_b2_closed_noservo_v2"}


def binding(backend):
    return load_public_geometry_binding(resolve_tool_surface(DEFAULT_SURFACES[backend]).view_config["public_geometry_descriptor"], project_root=ROOT)


@pytest.mark.parametrize("robot,convention", [("franka-panda", "panda_grip_site"), ("aloha-agilex", "aloha_grip_center_tcp")])
def test_current_geometry_is_closed_source_bound_and_deterministically_delivered(robot, convention):
    base = ROOT / "ora/policy/embodiments"
    descriptor = load_embodiment_descriptor(base / robot / (
        "descriptor-public-geometry-v8-controller-feedback-v1-session-split-v1.json" if robot == "franka-panda"
        else "descriptor-public-geometry-v5-session-split-v1.json"))
    schema = json.loads((base / "descriptor-schema-v2.json").read_text())
    validate_schema_instance(descriptor, schema)
    assert audit_embodiment_sources(descriptor, project_root=ROOT)["passed"]
    card = render_embodiment_card(descriptor)
    assert card == (base / robot / "manual-public-geometry-v3.md").read_text()
    assert (len(card.encode())+3)//4 <= 700
    assert public_local_axes(convention) == descriptor["control"]["controlled_points"][0]["local_axes"]
    assert all(term not in card for term in ("sha256", "robot-asset://", "source_name", "c9f5288"))
    assert "world-Z corrections" in card
    if robot == "franka-panda":
        assert "[-11.6, 4.4]" in card and "9.249" in card
    else:
        assert "4.8964" in card and "not a general minimum grasp width" in card


@pytest.mark.parametrize("bad", ["private_field", "nan", "reversed_bounds", "nonunit", "nonorthogonal", "unknown_point"])
def test_geometry_schema_rejects_privilege_and_invalid_numbers(bad):
    value = validate_embodiment_descriptor(json.loads((ROOT / (
        "ora/policy/embodiments/franka-panda/"
        "descriptor-public-geometry-v8-controller-feedback-v1-session-split-v1.json")).read_text()))
    geometry = value["geometry"]["grippers"][0]["contact_geometry"]
    if bad == "private_field":
        geometry["object_pose"] = [1, 2, 3]
    elif bad == "nan":
        geometry["fingers"][0]["collision_bounds_m"]["minimum"][0] = float("nan")
    elif bad == "reversed_bounds":
        geometry["fingers"][0]["collision_bounds_m"]["minimum"][0] = 100
    elif bad == "nonunit":
        value["control"]["controlled_points"][0]["local_axes"]["approach"] = [0, 0, 2]
    elif bad == "nonorthogonal":
        value["control"]["controlled_points"][0]["local_axes"]["closing"] = [0, 0, 1]
    else:
        geometry["controlled_point"] = "unknown"
    with pytest.raises(EmbodimentDescriptorError):
        validate_embodiment_descriptor(value)


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_loaded_geometry_is_a_real_comparison_not_a_source_flag(backend):
    expected = binding(backend)
    observed = copy.deepcopy(expected["geometry"])
    original = copy.deepcopy(observed)
    result = verify_loaded_robot_geometry(expected, observed)
    assert result["passed"] and observed == original
    assert result["simulation_steps"] == result["controller_writes"] == 0
    assert result["descriptor_sha256"] == expected["descriptor_sha256"]
    finger = next(iter(observed.values()))["fingers"][0]
    finger["collision_bounds_m"]["maximum"][0] += 0.0001
    with pytest.raises(ValueError, match="numeric mismatch"):
        verify_loaded_robot_geometry(expected, observed)


def test_libero_session_checks_loaded_geometry_before_caching_a_public_proof(tmp_path):
    from ora.environment.backends.libero_pro.session import LiberoProError, LiberoProSession
    expected = binding("libero")
    observed = copy.deepcopy(expected["geometry"])
    calls = []
    session = SimpleNamespace(public_robot_geometry=expected, output_root=tmp_path, round_index=2,
                              _worker_attr=lambda name: calls.append(name) or observed)
    LiberoProSession._verify_loaded_robot_geometry(session)
    assert calls == ["harness_robot_geometry"]
    assert session.robot_geometry_verification["round_index"] == 2
    saved = json.loads((tmp_path / "robot-geometry/round-2.json").read_text())
    assert saved["observed_geometry"] == observed
    assert saved["expected_binding"] == expected
    assert "observed_geometry" not in session.robot_geometry_verification
    observed["right_gripper"]["fingers"][0]["collision_bounds_m"]["maximum"][0] += 0.01
    with pytest.raises(LiberoProError, match="differs"):
        LiberoProSession._verify_loaded_robot_geometry(session)
    assert session.robot_geometry_verification is None


def test_new_surface_cannot_pair_a_geometry_descriptor_with_an_old_manual():
    from ora.orchestrator.defaults import expand_native_defaults
    from ora.orchestrator.spec import RunSpecError
    spec = expand_native_defaults(native_spec("libero_pro"), root=ROOT)
    spec["policy"].pop("baseline")
    spec["policy_context"]["project_instructions"][-1] = "ora/policy/embodiments/franka-panda/manual-v6.md"
    with pytest.raises(RunSpecError, match="exact compiled embodiment"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("selected", [False, True])
def test_robotwin_encoder_delivery_does_not_require_a_geometry_binding(tmp_path, monkeypatch, selected):
    from tests.environments.robotwin.test_robotwin_correctness_repair import Plant, live_session
    from ora.environment.backends.robotwin import gripper_state
    plant = Plant()
    session = live_session(tmp_path, plant)
    session.public_robot_geometry = binding("robotwin") if selected else None
    states = {arm: {"commanded_opening_normalized": 0., "drive_target_opening_normalized": 0.,
                    "measured_opening_normalized": 2/11, "measured_primary_joint_position": 0.}
              for arm in ("left", "right")}
    calls = []
    monkeypatch.setattr(gripper_state, "public_gripper_state", lambda robot: calls.append(robot) or states)
    observation, _ = session.observe(tmp_path / "observations")
    # The measured jaw separation is public state, so the encoder is read on
    # every observation regardless of any binding. What a binding still selects
    # is the normalized actuator block, which is model-internal training state
    # and not grasp evidence.
    assert len(calls) == 1
    assert observation["proprioception"]["gripper_evidence_semantics"] == (
        "command_target_and_measured_jaw_separation_m"
    )
    for arm in ("left", "right"):
        eef = observation["proprioception"]["end_effectors"][arm]
        assert "gripper_opening_m" in eef
        if selected:
            assert eef["gripper_state"] == states[arm]
        else:
            assert "gripper_state" not in eef
    assert plant.steps == 0 and plant.calls == []


def test_panda_geometry_uses_coherent_scratch_kinematics_without_live_writes(monkeypatch):
    """An integrated qpos must not be subtracted from older cached transforms."""
    import sys
    from types import SimpleNamespace
    import numpy as np
    from ora.environment.backends.libero_pro.robot_asset_geometry import panda_geometry_from_model
    from ora.contracts.robot_geometry import verify_loaded_robot_geometry, GEOMETRY_BINDING_SCHEMA

    names = {'grip_site': 0, 'finger_joint1': 0, 'finger_joint2': 1,
             'finger1_collision': 0, 'finger1_pad_collision': 1,
             'finger2_collision': 2, 'finger2_pad_collision': 3}
    model = SimpleNamespace(geom_type=np.full(4, 1), geom_size=np.tile([.004, .008, .003], (4, 1)),
                            jnt_type=np.full(2, 2), jnt_qposadr=np.array([0, 1]))

    def data_factory(_model):
        return SimpleNamespace(qpos=np.zeros(2), mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)),
            site_xmat=np.eye(3).reshape(1, 9), site_xpos=np.zeros((1, 3)),
            geom_xmat=np.tile(np.eye(3).reshape(1, 9), (4, 1)), geom_xpos=np.zeros((4, 3)),
            xaxis=np.array([[1., 0, 0], [1., 0, 0]]))

    def kinematics(_model, data):
        data.geom_xpos[:, 0] = [-.0035 + data.qpos[0], -.0035 + data.qpos[0],
                                .0035 + data.qpos[1], .0035 + data.qpos[1]]

    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(MjData=data_factory, mj_kinematics=kinematics,
        mj_name2id=lambda _model, _kind, name: names[name],
        mjtObj=SimpleNamespace(mjOBJ_SITE=0, mjOBJ_GEOM=1, mjOBJ_JOINT=2),
        mjtGeom=SimpleNamespace(mjGEOM_MESH=0, mjGEOM_BOX=1), mjtJoint=SimpleNamespace(mjJNT_SLIDE=2)))
    live = data_factory(model)
    live.qpos[:] = [.0199983, -.0199983]
    kinematics(model, live)
    live.qpos[:] = [.02, -.02]  # One integration tick ahead of cached geometry.
    before = {name: value.copy() for name, value in vars(live).items()}
    geometry = panda_geometry_from_model(model, live, prefix='')
    assert geometry['fingers'][0]['pad_bounds_m']['minimum'] == [-.0075, -.008, -.003]
    assert geometry['fingers'][0]['pad_bounds_m']['maximum'] == [.0005, .008, .003]
    assert geometry['fingers'][1]['pad_bounds_m']['minimum'] == [-.0005, -.008, -.003]
    for name, value in vars(live).items():
        np.testing.assert_array_equal(value, before[name])
    # Real asset changes still fail the unchanged micrometre geometry guard.
    binding = {'schema': GEOMETRY_BINDING_SCHEMA, 'descriptor_sha256': '0' * 64,
               'geometry': {'right_gripper': geometry}}
    model.geom_size[0, 0] += .001
    changed = panda_geometry_from_model(model, live, prefix='')
    with pytest.raises(ValueError, match='numeric mismatch'):
        verify_loaded_robot_geometry(binding, {'right_gripper': changed})


def test_libero_forward_kinematics_answers_a_chosen_joint_vector_without_touching_the_plant(monkeypatch):
    """The one piece LIBERO was missing for planner-free reachability.

    The shared solver needs end-effector pose at a *chosen* joint vector, not
    at the measured one. LIBERO published only the measured pose, so the only
    way to ask "could the arm hold this pose" was CuRobo, which has no IK-only
    mode and always returns a full trajectory.

    MuJoCo is already in the environment process and this exact pattern already
    runs there for the gripper: copy MjData, set qpos, mj_kinematics, read the
    site. This sets the arm joints instead. The live plant is never written.
    """
    import sys
    from types import SimpleNamespace
    import numpy as np
    from ora.environment.backends.libero_pro.robot_asset_geometry import (
        panda_pose_at_joints,
    )

    names = {"grip_site": 0}
    model = SimpleNamespace(
        jnt_range=np.array([[-2.9, 2.9]] * 7 + [[0.0, 0.04]] * 2),
        jnt_qposadr=np.arange(9),
    )

    def data_factory(_model):
        return SimpleNamespace(
            qpos=np.zeros(9), mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)),
            site_xmat=np.eye(3).reshape(1, 9), site_xpos=np.zeros((1, 3)),
        )

    def kinematics(_model, data):
        # A stand-in chain: the site moves with the first arm joint only.
        data.site_xpos[0] = [data.qpos[0], 0.0, 0.9]

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(
        MjData=data_factory, mj_kinematics=kinematics,
        mj_name2id=lambda _model, _kind, name: names[name],
        mjtObj=SimpleNamespace(mjOBJ_SITE=0),
    ))

    live = data_factory(model)
    live.qpos[:] = [0.5] + [0.0] * 8
    kinematics(model, live)
    before = {name: value.copy() for name, value in vars(live).items()}

    pose = panda_pose_at_joints(model, live, prefix="", joints=[0.25] + [0.0] * 6)
    assert len(pose) == 7
    assert pose[0] == pytest.approx(0.25)
    assert pose[2] == pytest.approx(0.9)

    # The plant is untouched: asking a hypothetical must not move the robot.
    for name, value in before.items():
        assert np.array_equal(getattr(live, name), value), name


def test_libero_arm_joint_limits_are_read_from_the_loaded_model(monkeypatch):
    """The second missing piece: limits for the arm, not the gripper fingers.

    robot_asset_geometry already reads model.jnt_range, but only for the
    standalone gripper asset. A reachability answer needs the arm's.
    """
    import sys
    from types import SimpleNamespace
    import numpy as np
    from ora.environment.backends.libero_pro.robot_asset_geometry import (
        panda_arm_joint_limits,
    )

    model = SimpleNamespace(
        jnt_range=np.array([[-2.9, 2.9], [-1.76, 1.76], [-2.9, 2.9], [-3.07, -0.07],
                            [-2.9, 2.9], [-0.02, 3.75], [-2.9, 2.9], [0.0, 0.04], [0.0, 0.04]]),
    )
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace())
    limits = panda_arm_joint_limits(model)
    assert len(limits) == 7, "seven arm joints, gripper fingers excluded"
    assert limits[3] == [-3.07, -0.07]
    assert all(low < high for low, high in limits)


def test_the_shared_solver_accepts_libero_kinematics_unchanged(monkeypatch):
    """The claim the whole two-backend plan rests on.

    A design note said LIBERO could not do planner-free reachability because it
    lacks a Jacobian. It does lack one -- and does not need one: the solver
    finite-differences a Jacobian from the same forward-kinematics callable it
    already takes. So the only LIBERO-specific pieces are the two functions
    above, and the solver itself is used unchanged.
    """
    import sys
    from types import SimpleNamespace
    import numpy as np
    from ora.environment.controllers.reachability import solve_reachability
    from ora.environment.backends.libero_pro.robot_asset_geometry import (
        panda_arm_joint_limits, panda_pose_at_joints,
    )

    names = {"grip_site": 0}
    model = SimpleNamespace(
        jnt_range=np.array([[-2.9, 2.9]] * 7 + [[0.0, 0.04]] * 2),
        jnt_qposadr=np.arange(9),
    )

    def data_factory(_model):
        return SimpleNamespace(
            qpos=np.zeros(9), mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)),
            site_xmat=np.eye(3).reshape(1, 9), site_xpos=np.zeros((1, 3)),
        )

    def kinematics(_model, data):
        data.site_xpos[0] = [data.qpos[0], data.qpos[1], 0.9 + data.qpos[2]]

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(
        MjData=data_factory, mj_kinematics=kinematics,
        mj_name2id=lambda _model, _kind, name: names[name],
        mjtObj=SimpleNamespace(mjOBJ_SITE=0),
    ))
    live = data_factory(model)

    solved = solve_reachability(
        target_pose_xyzw=[0.12, -0.05, 0.97, 0.0, 0.0, 0.0, 1.0],
        seed_joints=[0.0] * 7,
        fk_public_pose=lambda joints: panda_pose_at_joints(
            model, live, prefix="", joints=joints,
        ),
        joint_limits=panda_arm_joint_limits(model),
        position_tolerance_m=0.001,
        orientation_tolerance_rad=np.deg2rad(1.0),
    )
    assert solved["reachable"] is True
    assert solved["position_error_m"] <= 0.001
    assert "not a collision-free path" in solved["claim_boundary"]
