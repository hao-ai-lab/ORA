"""Contact-GraspNet sees each distinct crop point once, in seeded random order.

Upstream pads a sparse target crop to 20000 points by appending random
duplicates while keeping the scene's scan order; the ball query then builds
one-sided neighbourhoods and the width head collapses to its narrowest bin. The
resident adapter shadows the crop and per-region inference on its own
estimator instance. These tests drive that adapter against a stub estimator
that reproduces upstream's control flow and resampling semantics, so no
checkpoint, CUDA or upstream source tree is needed.
"""

from __future__ import annotations

import base64
import io
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from ora.agent.model_services import resident
from ora.agent.model_services.resident import (
    CONTACT_INPUT_POLICY,
    CONTACT_INPUT_REGIME_SCHEMA,
    GraspService,
    _encode_npy,
    contact_input_regime,
)
from ora.agent.tools.grasp import (
    CONTACT_GRASPNET_MODEL_IDENTITY,
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    ContactGraspNetApiProposer,
)

CONFIGURED = 20000


def _regularize(pc, npoints, use_farthest_point=False):
    """Upstream ``data.regularize_pc_point_count`` without farthest sampling."""

    assert use_farthest_point is False
    if pc.shape[0] > npoints:
        return pc[np.random.choice(range(pc.shape[0]), size=npoints, replace=False)]
    required = npoints - pc.shape[0]
    if required > 0:
        index = np.random.choice(range(pc.shape[0]), size=required)
        pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def _reject_outliers(data, m=0.4, z_only=False):
    distance = np.abs(data - np.median(data, axis=0, keepdims=True))
    return data[np.sum(distance, axis=1) < m]


class StubEstimator:
    """Upstream ``GraspEstimator`` control flow with a recording network."""

    def __init__(self):
        self._num_input_points = CONFIGURED
        self._contact_grasp_cfg = {"DATA": {
            "raw_num_points": CONFIGURED, "use_farthest_point": False,
        }}
        self.fed = []

    def extract_3d_cam_boxes(self, full_pc, pc_segments, min_size=0.3, max_size=0.6):
        raise AssertionError("the upstream padding crop must be shadowed")

    def predict_grasps(self, pc, constant_offset=False, convert_cam_coords=True,
                       forward_passes=1):
        # Upstream preprocess_pc_for_inference resamples to _num_input_points.
        fed = pc if len(pc) == self._num_input_points else _regularize(
            pc, self._num_input_points,
        )
        self.fed.append(np.array(fed))
        count = 3
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        poses[:, :3, 3] = fed[:count]
        return (poses, np.asarray([0.2, 0.29, 0.1], dtype=np.float32),
                fed[:count].astype(np.float32),
                np.asarray([0.025, 0.033, 0.009], dtype=np.float32))

    def predict_scene_grasps(self, pc_full, pc_segments={}, local_regions=False,
                             filter_grasps=False, forward_passes=1):
        assert local_regions and filter_grasps
        grasps, scores, contacts, openings = {}, {}, {}, {}
        regions, _ = self.extract_3d_cam_boxes(pc_full, pc_segments, min_size=0.2)
        for key, region in regions.items():
            grasps[key], scores[key], contacts[key], openings[key] = (
                self.predict_grasps(region, convert_cam_coords=True,
                                    forward_passes=forward_passes)
            )
        return grasps, scores, contacts, openings


def _service():
    service = object.__new__(GraspService)
    service.np = np
    service.torch = SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda _seed: None,
                             synchronize=lambda: None),
        inference_mode=nullcontext,
    )
    service.estimator = StubEstimator()
    service._install_input_policy(
        regularize=_regularize, reject_outliers=_reject_outliers,
    )
    service.model_identity = dict(CONTACT_GRASPNET_MODEL_IDENTITY)
    service.adapter_sha256 = "a" * 64
    service.instance_id = "model-instance-test"
    service.load_count = 1
    service.load_ms = service.warmup_ms = 0.0
    service.started_unix_s = 1.0
    service.requests = 0
    return service


def _scene(unique_count, *, duplicates=0, seed=0):
    rng = np.random.default_rng(seed)
    target = np.column_stack([
        rng.uniform(-0.015, 0.015, 400), rng.uniform(-0.015, 0.015, 400),
        rng.uniform(0.70, 0.73, 400),
    ])
    table = np.column_stack([
        rng.uniform(-0.09, 0.09, unique_count - 400),
        rng.uniform(-0.09, 0.09, unique_count - 400),
        np.full(unique_count - 400, 0.74),
    ])
    scene = np.concatenate([target, table]).astype(np.float32)
    if duplicates:
        scene = np.concatenate([scene, scene[:duplicates]])
    return scene, target.astype(np.float32)


def test_sparse_crop_runs_at_its_distinct_point_count_in_seeded_order():
    service = _service()
    scene, target = _scene(2029)
    poses, scores, _contacts, openings, regime = service._predict(
        scene, target, forward_passes=1, seed=7,
    )
    (fed,) = service.estimator.fed
    assert len(fed) == 2029 == len(np.unique(fed, axis=0))
    # Same points as the scene crop, but not in the scene's scan order.
    np.testing.assert_array_equal(
        np.unique(fed, axis=0), np.unique(scene, axis=0),
    )
    assert not np.array_equal(fed, scene)
    assert service.estimator._num_input_points == CONFIGURED  # restored
    assert regime == {
        "schema": CONTACT_INPUT_REGIME_SCHEMA, "policy": CONTACT_INPUT_POLICY,
        "status": "region", "crop_points": 2029, "unique_points": 2029,
        "configured_input_points": CONFIGURED, "input_points": 2029,
        "padding_factor": 1.0, "upstream_padding_factor": 9.8571,
        "point_order": "request_seeded_permutation",
        "upstream_scene_order": True,
    }
    # The permutation is a function of the request seed alone.
    service._predict(scene, target, forward_passes=1, seed=7)
    service._predict(scene, target, forward_passes=1, seed=8)
    _first, again, other = service.estimator.fed
    np.testing.assert_array_equal(again, fed)
    assert not np.array_equal(other, fed)
    # Score ordering and aligned outputs are unchanged by the adapter.
    assert scores.tolist() == pytest.approx([0.29, 0.2, 0.1])
    assert openings.tolist() == pytest.approx([0.033, 0.025, 0.009])
    assert poses.shape == (3, 4, 4)


def test_unpatched_flow_would_have_padded_the_same_crop_tenfold():
    estimator = StubEstimator()
    scene, _target = _scene(2029)
    estimator.predict_grasps(_regularize(scene, CONFIGURED))
    (fed,) = estimator.fed
    assert len(fed) == CONFIGURED and len(np.unique(fed, axis=0)) == 2029


def test_duplicate_scene_rows_are_fed_once():
    service = _service()
    scene, target = _scene(3000, duplicates=500)
    *_, regime = service._predict(scene, target, forward_passes=1, seed=3)
    (fed,) = service.estimator.fed
    assert regime["crop_points"] == 3500 and regime["unique_points"] == 3000
    assert len(fed) == regime["input_points"] == 3000
    assert len(np.unique(fed, axis=0)) == 3000


def test_dense_crop_keeps_the_upstream_downsample_exactly():
    service = _service()
    scene, target = _scene(26000)
    *_, regime = service._predict(scene, target, forward_passes=1, seed=11)
    (fed,) = service.estimator.fed
    np.random.seed(11)
    expected = _regularize(scene, CONFIGURED)
    np.testing.assert_array_equal(fed, expected)
    assert regime["input_points"] == CONFIGURED
    assert regime["upstream_padding_factor"] == 1.0
    assert regime["padding_factor"] == pytest.approx(1.0)
    assert regime["upstream_scene_order"] is False


def test_too_few_distinct_points_yield_no_region_instead_of_padding():
    service = _service()
    target = np.asarray([[0.001 * i, 0.0, 0.72] for i in range(8)], dtype=np.float32)
    scene = np.concatenate([target, target, target])
    poses, scores, contacts, openings, regime = service._predict(
        scene, target, forward_passes=1, seed=1,
    )
    assert service.estimator.fed == []
    assert len(poses) == len(scores) == len(contacts) == len(openings) == 0
    assert regime["status"] == "too_few_unique_points"
    assert regime["input_points"] == 0 and regime["padding_factor"] is None
    assert regime["point_order"] is None


def test_empty_crop_reports_no_region():
    service = _service()
    _scene_points, target = _scene(1000)
    far = target + np.asarray([5.0, 5.0, 5.0], dtype=np.float32)
    *_, regime = service._predict(far, target, forward_passes=1, seed=1)
    assert regime["status"] == "no_region"
    assert regime["unique_points"] == 0 and regime["upstream_padding_factor"] is None


def test_infer_response_and_health_expose_the_regime():
    service = _service()
    scene, target = _scene(2029)
    result = service.infer({
        "schema": "ora.model-service.grasp-proposals-request.v1",
        "scene_points_npy_base64": _encode_npy(scene),
        "target_points_npy_base64": _encode_npy(target),
        "max_candidates": 2, "forward_passes": 1, "seed": 7,
    })
    assert result["input_regime"]["input_points"] == 2029
    assert result["input_regime"]["upstream_padding_factor"] == 9.8571
    health = service.health()
    assert "distinct_shuffled_crop_input_v1" in health["capabilities"]
    assert health["input_policy"] == {
        "schema": "ora.contact-graspnet-input-policy.v1",
        "policy": CONTACT_INPUT_POLICY, "configured_input_points": CONFIGURED,
        "duplicate_padding": False, "point_order": "request_seeded_permutation",
        "minimum_region_unique_points": resident.CONTACT_MIN_REGION_UNIQUE_POINTS,
    }


def test_regime_helper_marks_downsampling_and_absence():
    downsampled = contact_input_regime(
        status="region", crop_points=30000, unique_points=30000,
        configured_input_points=CONFIGURED, input_points=CONFIGURED,
    )
    assert downsampled["padding_factor"] == 1.0
    assert downsampled["upstream_padding_factor"] == 1.0


def _encoded(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.mark.parametrize("regime", [
    None,
    {"schema": CONTACT_INPUT_REGIME_SCHEMA, "input_points": 2029},
])
def test_client_keeps_the_regime_private_and_tolerates_old_services(regime):
    pose = np.eye(4)[None]
    pose[0, 2, 3] = 0.8

    def respond(request):
        body = {
            "schema": "ora.model-service.grasp-proposals.v1",
            "request_id": "regime-fixture", "model": CONTACT_GRASPNET_MODEL_IDENTITY,
            "service_identity": {
                "schema": "ora.model-service-response-identity.v1",
                "instance_id": "fixture", "pid": 1, "started_unix_s": 1.0,
                "adapter_sha256": "a" * 64,
            },
            "poses_camera_npy_base64": _encoded(pose),
            "scores_npy_base64": _encoded([0.2]),
            "openings_m_npy_base64": _encoded([0.03]),
            "contact_points_camera_npy_base64": _encoded(np.zeros((1, 3))),
            "metrics": {"queue_ms": 0.0, "inference_ms": 1.0, "total_ms": 1.0},
        }
        if regime is not None:
            body["input_regime"] = regime
        return httpx.Response(200, json=body)

    proposer = ContactGraspNetApiProposer()
    proposer.close()
    object.__setattr__(
        proposer, "_client", httpx.Client(transport=httpx.MockTransport(respond)),
    )
    points = np.column_stack([
        np.linspace(-0.02, 0.02, 32), np.zeros(32), np.full(32, 0.8),
    ])
    try:
        _candidates, diagnostics = proposer.propose(
            scene_points_camera_m=points, target_points_camera_m=points,
            camera_to_world=np.eye(4), target_ref="target-" + "b" * 24,
            workspace_bounds_m={axis: [-2.0, 2.0] for axis in "xyz"},
            max_candidates=1, compiler_identity=PANDA_GRIP_SITE_COMPILER_IDENTITY,
        )
    finally:
        proposer.close()
    assert diagnostics["_backend_input_regime"] == regime
