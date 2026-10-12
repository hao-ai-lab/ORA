"""Raw grasp rejection evidence survives current result schemas on all backends."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from ora.agent.composition import grasp_rejection_schema, tool_surface
from ora.agent.composition.grasp_rejection_schema import CAPABILITY, LIMIT_SEMANTICS
from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG, ToolSurfaceError, resolve_tool_surface, validate_schema_instance,
)
from ora.agent.tools.grasp import (
    ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES, ARX_X5_LINK6_COMPILER_IDENTITIES,
    CONTACT_GRASPNET_MODEL_IDENTITY, PANDA_GRIP_SITE_COMPILER_IDENTITY,
    ContactGraspNetApiProposer, PcaTopDownGraspProposer,
)
from surface_copies import rebase_manifest


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROFILES = (
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v2", "robotwin_toolchain_b2_closed_composed_v1",
    "robotwin_toolchain_b2_closed_noservo_v1",
    "robodojo_placement_closed_skills_v2", "robodojo_placement_closed_composed_v1",
)
COMPILERS = {
    "libero": PANDA_GRIP_SITE_COMPILER_IDENTITY,
    "robotwin": ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES["left"],
    "robodojo": ARX_X5_LINK6_COMPILER_IDENTITIES["left"],
}


def _proposal_arguments(compiler):
    x, y = np.meshgrid(np.linspace(-0.20, 0.20, 24), np.linspace(-0.12, 0.12, 20))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.8)))
    return {
        "scene_points_camera_m": points, "target_points_camera_m": points,
        "camera_to_world": np.eye(4), "target_ref": "target-" + "a" * 24,
        "workspace_bounds_m": {axis: [-2.0, 2.0] for axis in "xyz"},
        "max_candidates": 3, "compiler_identity": compiler,
    }


def _encoded(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _learned_propose(compiler, openings):
    poses = np.repeat(np.eye(4)[None], len(openings), axis=0)
    poses[:, 2, 3] = 0.8

    def respond(request):
        assert request.url.path == "/v1/grasp-proposals"
        return httpx.Response(200, json={
            "schema": "ora.model-service.grasp-proposals.v1",
            "request_id": "offline-rejection-fixture", "model": CONTACT_GRASPNET_MODEL_IDENTITY,
            "service_identity": {
                "schema": "ora.model-service-response-identity.v1",
                "instance_id": "offline-cgn-fixture", "pid": 123, "started_unix_s": 1.0,
                "adapter_sha256": "a" * 64,
            },
            "poses_camera_npy_base64": _encoded(poses),
            "scores_npy_base64": _encoded(np.ones(len(openings))),
            "openings_m_npy_base64": _encoded(openings),
            "contact_points_camera_npy_base64": _encoded(np.zeros((len(openings), 3))),
            "metrics": {"queue_ms": 0.0, "inference_ms": 1.0, "total_ms": 1.0},
        })

    proposer = ContactGraspNetApiProposer()
    proposer.close()
    object.__setattr__(proposer, "_client", httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        return proposer.propose(**_proposal_arguments(compiler))
    finally:
        proposer.close()


@pytest.fixture(scope="module")
def raw_diagnostics():
    result = {}
    for backend, compiler in COMPILERS.items():
        pca_candidates, pca = PcaTopDownGraspProposer().propose(**_proposal_arguments(compiler))
        learned_candidates, learned = _learned_propose(compiler, [0.2, 0.0, -0.01])
        assert not pca_candidates and not learned_candidates
        assert pca["rejected"] and len(learned["rejected"]) == 3
        for row in pca["rejected"]:
            assert row["required_opening_m"] > row["maximum_opening_m"]
        assert [row["predicted_opening_m"] for row in learned["rejected"]] == [0.2, 0.0, -0.01]
        for row in [*pca["rejected"], *learned["rejected"]]:
            assert row["maximum_opening_m"] == compiler.get("maximum_opening_m", 0.08)
            assert row["limit_semantics"] == LIMIT_SEMANTICS
        result[backend] = (pca, learned)
    return result


@pytest.mark.parametrize("profile", CURRENT_PROFILES)
def test_raw_pca_and_learned_rejections_fit_each_current_surface(profile, raw_diagnostics):
    surface = resolve_tool_surface(profile)
    assert surface.supports(CAPABILITY)
    schema = surface.result_schemas["grasp_plan"]
    row_schema = {"$ref": "#/$defs/rejected_backend_candidate", "$defs": schema["$defs"]}
    pca, learned = raw_diagnostics[profile.split("_", 1)[0]]
    for row in [*pca["rejected"], *learned["rejected"]]:
        before = copy.deepcopy(row)
        validate_schema_instance(row, row_schema)
        assert row == before  # No feedback fields are removed to satisfy a schema.
    # The materializer separately retains these three private backend records;
    # all public diagnostics, including the complete rejection rows, survive.
    public = {key: value for key, value in learned.items()
              if key not in {"_service_identity", "_canonical_backend_output",
                             "_backend_input_regime"}}
    public.update(compiler_rejected=[], compiler_returned_candidate_count=0)
    validate_schema_instance(public, {**schema["$defs"]["diagnostics"], "$defs": schema["$defs"]})
    assert public["rejected"] == learned["rejected"]
    derived = next(item for item in surface.components if item["slot"] == f"derived:{CAPABILITY}")
    assert derived["sha256"] == hashlib.sha256((ROOT / derived["ref"]).read_bytes()).hexdigest()
    assert surface.snapshot()["result_schemas"]["grasp_plan"] == schema


@pytest.mark.parametrize("profile", CURRENT_PROFILES)
def test_optional_unknown_openings_and_strict_diagnostic_bounds(profile):
    defs = resolve_tool_surface(profile).result_schemas["grasp_plan"]["$defs"]
    schema = {"$ref": "#/$defs/rejected_backend_candidate", "$defs": defs}
    base = {"candidate_id": "grasp-unknown", "reasons": ["aperture_exceeds_embodiment_limit"]}
    validate_schema_instance(base, schema)  # Historical rows need no invented measurements.
    for field in ("predicted_opening_m", "required_opening_m"):
        validate_schema_instance({**base, field: None, "maximum_opening_m": 0.055,
                                  "limit_semantics": LIMIT_SEMANTICS}, schema)
    validate_schema_instance({**base, "predicted_opening_m": -0.01}, schema)
    for field, value in (("predicted_opening_m", float("inf")), ("predicted_opening_m", True),
                         ("required_opening_m", -0.01), ("required_opening_m", float("nan")),
                         ("maximum_opening_m", 0), ("maximum_opening_m", -0.01),
                         ("maximum_opening_m", None), ("limit_semantics", "measured_jaw_aperture"),
                         ("unexpected_measurement_m", 0.1)):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance({**base, field: value}, schema)
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance({"candidate_id": "grasp-missing-reasons"}, schema)


@pytest.mark.parametrize("opening", [float("nan"), float("inf")])
def test_learned_nonfinite_arrays_still_fail_before_diagnostics(opening):
    with pytest.raises(ValueError, match="non-finite candidate arrays"):
        _learned_propose(COMPILERS["robotwin"], [opening])


@pytest.mark.parametrize("profile", ["libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v1"])
def test_historical_manifest_does_not_inherit_the_new_schema(monkeypatch, tmp_path, profile):
    """A manifest that does not declare the capability keeps the closed historical row.

    The predecessors that never declared it are no longer in this tree, so a
    test-local copy of a kept surface without the capability stands in.
    """
    path = PROFILE_CATALOG[profile]
    before = path.read_bytes()
    manifest = rebase_manifest(json.loads(before), path.parent, tmp_path)
    manifest["capabilities"] = [name for name in manifest["capabilities"] if name != CAPABILITY]
    copy_id = manifest["profile_id"] = f"{profile}__without_rejection_feedback"
    copied = tmp_path / f"{copy_id}.json"
    copied.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setitem(PROFILE_CATALOG, copy_id, copied)
    monkeypatch.setattr(tool_surface, "PROFILE_ROOT", Path(copied.anchor))
    surface = resolve_tool_surface(copy_id)
    assert not surface.supports(CAPABILITY)
    assert not any(item["slot"] == f"derived:{CAPABILITY}" for item in surface.components)
    defs = surface.result_schemas["grasp_plan"]["$defs"]
    with pytest.raises(ToolSurfaceError, match="unknown result fields"):
        validate_schema_instance({"candidate_id": "grasp-old", "reasons": ["rejected"],
                                  "predicted_opening_m": None},
                                 {"$ref": "#/$defs/rejected_backend_candidate", "$defs": defs})
    assert path.read_bytes() == before


def test_derived_contract_requires_the_existing_closed_rejection_shape():
    with pytest.raises(ToolSurfaceError, match="closed backend-rejection contract"):
        grasp_rejection_schema.bind_grasp_rejection_diagnostics({})
