"""The 0059 ports on the RoboTwin and LIBERO default surfaces.

Every unit is capability-gated and the current defaults
(``robotwin_toolchain_b2_closed_noservo_v2``,
``libero_placement_direct_overhead_v1``) declare them. This module checks:

* every port help component is bound by a registered surface;
* M5 ``world_axes`` compiles each embodiment's own declared axes (ALOHA TCP
  approach +X, jaw line +Y; Panda grip site approach +Z, jaw line +X), and the
  help names exactly those axes;
* V1 draws the 1024x800 depth-profile card with each backend's camera names,
  with the support line only where ``target_support_height_v1`` is declared,
  and the LIBERO help names that card size;
* V2's observed-extent cell reads RoboTwin (3x4 extrinsic) and LIBERO (4x4)
  calibrations and its legend names the backend's real review cameras.

On LIBERO, M2 (the CuRobo reach backend) also replaces grasp_plan's help with
the released text plus the IK-selection paragraph and adds the pooled
``ik_reachability`` fields; ``test_libero_reach_ports.py`` drives them. The
real dispatcher delivers every port through result-schema admission.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.embodiment_geometry import public_local_axes
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.world_axes_schema import MODE, TARGET_AXES_PROPERTY
from ora.agent.tools import grounding_preview, scene_extent_review, world_axes
from ora.contracts.pose import quaternion_matrix

ROOT = Path(__file__).resolve().parents[3]
PORTS = ROOT / "ora/policy/tool_surfaces/closed_skills_ports_v1"
#: The default surface of each backend, which declares every 0059 unit.
DEFAULTS = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}
POSE_TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
#: Result fields the ports add under compiled_pose.targets.<arm>.
ADDED_TARGET_FIELDS = {
    "robotwin": {"reach_along_request", "observed_clearance", "target_axes_world"},
    "libero_pro": {"reach_along_request", "observed_clearance", "target_axes_world"},
}


@pytest.fixture(params=["robotwin", "libero_pro"])
def backend(request):
    return request.param


def _descriptions(surface):
    return {item["name"]: item["description"] for item in surface.definitions}


def _target_properties(schema):
    return schema["properties"]["compiled_pose"]["properties"]["targets"]["additionalProperties"]["properties"]


# --------------------------------------------------------------------------
# M5: world_axes on each embodiment's declared axes
# --------------------------------------------------------------------------

AXIS_NAMES = {(1, 0, 0): "+X", (0, 1, 0): "+Y", (0, 0, 1): "+Z"}


def test_world_axes_compiles_the_embodiment_s_own_axes(backend):
    surface = resolve_tool_surface(DEFAULTS[backend])
    convention = surface.view_config["controlled_frame_convention"]
    axes = public_local_axes(convention)
    arm = "left" if backend == "robotwin" else "right"
    request = {"position": {"mode": "absolute", "position_m": [0.1, 0.0, 1.0]},
               "orientation": {"mode": MODE, "approach_world": [0, 0, -1], "jaw_line_world": [0, 1, 0]}}
    compiled = world_axes.resolve_request(surface, request, arm=arm)
    rotation = np.asarray(quaternion_matrix(compiled["orientation"]["quaternion_xyzw"]))
    assert rotation @ np.asarray(axes["approach"], float) == pytest.approx([0, 0, -1], abs=1e-9)
    assert rotation @ np.asarray(axes["closing"], float) == pytest.approx([0, 1, 0], abs=1e-9)
    echo = world_axes.echo(surface, {"orientation_xyzw": compiled["orientation"]["quaternion_xyzw"]})
    assert echo["target_axes_world"]["approach"] == pytest.approx([0, 0, -1], abs=1e-6)
    for tool in POSE_TOOLS:
        assert _target_properties(surface.result_schemas[tool])["target_axes_world"] == TARGET_AXES_PROPERTY
    move_pose = _descriptions(surface)["move_pose"]
    stated = re.search(r"approach_world \(.*?local (\+[XYZ])\) and jaw_line_world \(its local (\+[XYZ])\)", move_pose)
    assert stated, move_pose[-400:]
    assert (stated[1], stated[2]) == (AXIS_NAMES[tuple(int(v) for v in axes["approach"])],
                                      AXIS_NAMES[tuple(int(v) for v in axes["closing"])])
    assert "link6" not in move_pose
    assert "world_axes takes approach_world and jaw_line_world;" in move_pose


# --------------------------------------------------------------------------
# V1: the depth-profile card in each backend's camera vocabulary
# --------------------------------------------------------------------------

def _write_observation(root: Path, cameras: dict[str, np.ndarray], *, extrinsic_rows: int) -> tuple[Path, dict]:
    """One flat-floor RGB-D observation; ``cameras`` maps names to cam2world."""
    target = root / "observation-0001"
    target.mkdir(parents=True)
    intrinsic = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    records = {}
    for name, cam2world in cameras.items():
        rgb = np.full((48, 64, 3), [52, 58, 66], dtype=np.uint8)
        depth = np.full((48, 64), 1.2, dtype=np.float32)
        rgb[14:34, 20:40] = [200, 60, 40]
        depth[14:34, 20:40] = 0.8
        folder = target / name
        folder.mkdir()
        Image.fromarray(rgb).save(folder / "rgb.png")
        np.save(folder / "depth-m.npy", depth, allow_pickle=False)
        extrinsic = np.linalg.inv(cam2world)[:extrinsic_rows]
        (folder / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": extrinsic.tolist(),
            "cam2world_cv": cam2world.tolist(), "depth_unit": "metres",
            "depth_array_order": "opencv_top_left_row_major", "policy_quaternion_order": "xyzw"}))
        records[name] = {"width": 64, "height": 48, "files": {
            role: {"path": f"{name}/{file}",
                   "sha256": hashlib.sha256((folder / file).read_bytes()).hexdigest()}
            for role, file in (("rgb_lossless_png", "rgb.png"), ("depth_numpy_m_f32", "depth-m.npy"),
                               ("calibration", "calibration.json"))}}
    return target, {"observation_id": "port:observation-0001", "cameras": records}


def _shifted(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, 0.05, 0.02]
    return pose


#: backend -> (primary camera, review cameras, extrinsic rows the backend writes)
VOCABULARY = {
    "robotwin": ("head_camera", ("head_camera", "left_camera"), 3),
    "libero_pro": ("agentview", ("agentview", "wrist"), 4),
}


def test_the_depth_profile_card_follows_the_default_surface(backend, tmp_path):
    from ora.agent.tools.grasp import materialize_grounded_targets
    from tests.environments.robodojo.test_robodojo_perception import FakeHeadGrounding

    surface = resolve_tool_surface(DEFAULTS[backend])
    options = grounding_preview.options(surface)
    # The support line follows target_support_height_v1.
    assert options == {"renderer": grounding_preview.PROFILE_RENDERER,
                       "support_line": surface.supports(grounding_preview.SUPPORT_CAPABILITY)}
    primary, _review, rows = VOCABULARY[backend]
    root, observation = _write_observation(tmp_path / "obs", {primary: np.eye(4)}, extrinsic_rows=rows)
    folder = root / primary
    grounded, paths = materialize_grounded_targets(
        artifact_root=tmp_path / "artifacts", observation_id=observation["observation_id"], camera=primary,
        rgb_path=folder / "rgb.png", depth_path=folder / "depth-m.npy",
        calibration_path=folder / "calibration.json", text="red mug", max_candidates=1,
        backend=FakeHeadGrounding(), depth_profile=options)
    with Image.open(paths[1]) as image:
        assert image.size == (grounding_preview.CARD_WIDTH, grounding_preview.PROFILE_CARD_HEIGHT)
    record = grounded["visualization"]["candidate_previews"][0]["depth_profile"]
    assert record["status"] == "rendered" and record["options"] == options
    assert grounded["identity"]["visualization"]["depth_profile"] == options


def test_the_libero_help_names_the_profiled_card_size():
    layer = json.loads((PORTS / "libero-grounding-depth-profile-help-v1.json").read_text(encoding="utf-8"))
    assert set(layer) == {"replace"} and set(layer["replace"]) == {"vision_ground"}
    delivered = _descriptions(resolve_tool_surface(DEFAULTS["libero_pro"]))["vision_ground"]
    legacy = f"{grounding_preview.CARD_WIDTH}x{grounding_preview.LEGACY_CARD_HEIGHT}"
    profiled = f"{grounding_preview.CARD_WIDTH}x{grounding_preview.PROFILE_CARD_HEIGHT}"
    for text in (layer["replace"]["vision_ground"], delivered):
        assert text.count(profiled) == 1 and legacy not in text


# --------------------------------------------------------------------------
# V2: the observed-extent cell with each backend's review cameras
# --------------------------------------------------------------------------

def test_the_observed_extent_cell_reads_each_backend_s_review_cameras(backend, tmp_path):
    surface = resolve_tool_surface(DEFAULTS[backend])
    assert surface.supports(scene_extent_review.CAPABILITY)
    primary, review, rows = VOCABULARY[backend]
    root, observation = _write_observation(
        tmp_path / "obs", {review[0]: np.eye(4), review[1]: _shifted(0.3)}, extrinsic_rows=rows)
    workspace = {"x": [-0.2, 0.2], "y": [-0.2, 0.2], "z": [0.72, 1.4]}
    result = scene_extent_review.cell(
        observation_root=root, observation=observation, camera_names=review, max_depth_m=2.5,
        workspace_bounds_m=workspace, target_world=np.zeros((0, 3)), grip_sites=[])
    assert result["record"]["status"] == "available"
    assert result["record"]["camera_order"] == list(review)
    assert result["record"]["points_outside_workspace"] > 0  # the walls/floor the clipped panels drop
    assert scene_extent_review.captions(
        True, scene_sources=[{"camera": name} for name in review], backend_cameras=[primary],
        camera=primary) == (f"fused RGB-D · {review[0]} + {review[1]}", f"backend input · {primary}")
    assert scene_extent_review.identity(True) == {"review_renderer": scene_extent_review.RENDERER}


def test_every_port_component_is_bound_by_a_registered_surface():
    bound = set()
    for path in PROFILE_CATALOG.values():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for ref in manifest.get("components", {}).values():
            if isinstance(ref, str) and (path.parent / ref).resolve().parent == PORTS.resolve():
                bound.add(Path(ref).name)
    assert bound == {path.name for path in PORTS.glob("*.json")}
    for name in bound:
        layer = json.loads((PORTS / name).read_text(encoding="utf-8"))
        assert set(layer) <= {"append", "replace"}


# --------------------------------------------------------------------------
# The real dispatcher: every port through result-schema admission
# --------------------------------------------------------------------------

def _dispatcher(backend, tmp_path):
    from ora.agent.tools.runtime import ManipulationToolRuntime
    from tests.agent.motion.test_observed_clearance_v2 import ALOHA_DOWN, LIBERO_CAMERAS, PANDA_DOWN, ROBOTWIN_CAMERAS, _aloha, \
        _panda, _write_scene
    from tests.environments.robodojo.test_robodojo_closed_skills_default import BothArmsProposer
    from tests.environments.robodojo.test_robodojo_perception import FakeHeadGrounding

    source = tmp_path / "observation"
    if backend == "robotwin":
        observation = _write_scene(source, {**ROBOTWIN_CAMERAS, "front_camera": ((0.0, -0.8, 1.2), (0.0, 1.0, -0.5))},
                                   {"left": _aloha([-0.3, -0.3, 1.1, *ALOHA_DOWN]),
                                    "right": _aloha([0.3, -0.3, 1.1, *ALOHA_DOWN])})
    else:
        observation = _write_scene(source, LIBERO_CAMERAS, {"right": _panda([0.3, 0.3, 1.1, *PANDA_DOWN])})
    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=Path(__file__).resolve().parents[3]
        / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.4, 0.4],
                                                                  "z": [0.7, 1.45]},
        state_root=None, tool_surface_profile=DEFAULTS[backend], controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=BothArmsProposer())
    pointer = {"open": True, "observation_id": observation["observation_id"],
               "observation_path": str(source / "observation.json"), "generation": 3,
               "observation_sha256": hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    return runtime


def _robotwin_answer(request):
    from ora.contracts.reach_check import RESULT_SCHEMA

    return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": "robotwin_single_seed_damped_least_squares_v1",
            "matches_move_pose_admission": False, "motion_planner_called": False,
            "claim_boundary": "One damped-least-squares descent; not move_pose's admission.",
            "results": [{"reachable": True, "reason": None, "route": None, "position_error_m": 0.001,
                         "orientation_error_rad": 0.01, "joint_limit_margin_rad": 0.5, "joint_delta_rad": 0.2}
                        for _query in request["queries"]]}


def _libero_answer(request):
    from ora.environment.backends.libero_pro.reach_check import CLAIM_BOUNDARY, IK_MODEL

    return {**_robotwin_answer(request), "ik_model": IK_MODEL, "claim_boundary": CLAIM_BOUNDARY}


def test_the_dispatcher_delivers_every_port_through_result_admission(backend, tmp_path):
    runtime = _dispatcher(backend, tmp_path)
    requests = []
    answer = _robotwin_answer if backend == "robotwin" else _libero_answer
    runtime.align_plan_callback = lambda request: requests.append(request) or answer(request)
    arm, jaw_line = ("left", [0, 1, 0]) if backend == "robotwin" else ("right", [1, 0, 0])
    request = {"interaction": "free_space", "position_mode": "absolute", "position_m": [0.0, 0.0, 0.95],
               "orientation_mode": "world_axes", "approach_world": [0, 0, -1], "jaw_line_world": jaw_line}
    result = runtime.invoke("move_pose", {"targets": {arm: request}} if backend == "robotwin" else request)
    target = result["compiled_pose"]["targets"][arm]
    assert set(target) >= ADDED_TARGET_FIELDS[backend]
    assert target["target_axes_world"] == {"approach": [0.0, 0.0, -1.0], "jaw_line": [float(v) for v in jaw_line]}
    clearance = target["observed_clearance"]
    tip = 0.03557 if backend == "robotwin" else 0.00925  # fingertips beyond the grip centre along the approach
    assert clearance["status"] == "measured"
    assert clearance["footprint_lowest_m"][2] == pytest.approx(0.95 - tip, abs=2e-4)
    assert clearance["clearance_m"] == pytest.approx(0.95 - tip - 0.842, abs=0.005)
    reach = target["reach_along_request"]
    assert reach["matches_move_pose_admission"] is False and len(requests) == 1
    assert requests[0]["arm"] == arm
    # RoboTwin's check is measured against move_pose's admission; LIBERO's is not yet.
    assert (reach["measured_disagreement"] is not None) is (backend == "robotwin")
    # No new policy-visible key collides with the agent-trace privileged-field denylist.
    from ora.agent.evidence.trajectory import _forbidden_path

    assert _forbidden_path(target) is None
    for tool in POSE_TOOLS:
        assert _forbidden_path(_target_properties(runtime.surface.result_schemas[tool])) is None, tool
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())["arguments"]["targets"]
    rotation = np.asarray(quaternion_matrix(submitted[arm]["orientation_xyzw"]))
    approach = public_local_axes(runtime.surface.view_config["controlled_frame_convention"])["approach"]
    assert rotation @ np.asarray(approach, float) == pytest.approx([0, 0, -1], abs=1e-9)
    assert submitted[arm]["position_m"] == pytest.approx([0.0, 0.0, 0.95])
