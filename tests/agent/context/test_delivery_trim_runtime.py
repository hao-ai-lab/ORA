"""``observation_delivery_trim_v2`` at the delivery sites, and what stays recoverable.

The direct_v4 surfaces declare it. A Native observation built by the real
session shows the trimmed facts; the policy-state observation, the hash-bound
calibration and the canonical ``tool_result`` keep every byte; and an aged
observation reopens to exactly the text the policy was first shown.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from ora.agent.composition.baseline import load_native_baseline
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.context import delivery_trim as trim
from ora.agent.context import policy_delivery
from ora.agent.context.current_visual import camera_geometry
from tests.agent.context.test_current_visual_delivery import ROOT, planner_for, published

BACKENDS = ("libero_pro", "robotwin", "robodojo")


def _bound_context(tmp_path, backend):
    context, projector, store, surface, source = published(tmp_path, backend)
    observation = json.loads((source / "observation.json").read_text(encoding="utf-8"))
    bound = dataclasses.replace(projector, workspace=tmp_path / "bound-workspace",
                                control_root=tmp_path / "bound-control", capabilities=surface.capabilities)
    delivered = bound.publish(observation=observation, observation_root=source, task_instruction="Inspect",
                              remaining_budget={}, policy_state=store.steps[-1])
    return context, delivered, store, surface, source, observation


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_default_surface_declares_the_capability(backend):
    instance = load_native_baseline(ROOT)["instances"][backend]
    surface = resolve_tool_surface(instance["tool_surface"])
    assert surface.supports(trim.CAPABILITY)
    assert trim.declared_rules(surface.capabilities) == trim.RULES


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_native_message_keeps_the_calibration_and_trims_the_facts(tmp_path, backend):
    """The small rules apply; the comparison sentence, both pose matrices, the
    exact calibration digits and the calibration owner reach the policy."""
    legacy, delivered, store, surface, source, observation = _bound_context(tmp_path, backend)
    planner = planner_for(tmp_path, delivered, surface)
    message = planner._causal_observation_message(context=delivered, terminal_lifecycle=False)
    envelope = json.loads(message["content"][0]["text"])
    current = envelope["current_actual"]
    assert isinstance(envelope["comparison"]["instruction"], str)
    assert "terminal_lifecycle" not in envelope
    facts = current["public_robot_facts"]
    for name, entry in facts["cameras"].items():
        assert "image_view" not in entry and "native_pixel_to_image_offset_xy" not in entry, name
    assert all("image_view" in entry for entry in legacy["public_robot_facts"]["cameras"].values())
    assert json.loads((source / "observation.json").read_text(encoding="utf-8")) == observation
    for name, geometry in current["camera_geometry"].items():
        full = camera_geometry(source, observation["cameras"][name])
        assert {key: geometry[key] for key in full if key in geometry} == {
            key: full[key] for key in full if key in geometry}, name
        assert "cam2world_cv" in geometry and len(set(geometry) - {"intrinsic_cv"}) > 1, name
    assert facts.get("calibration_owner") == legacy["public_robot_facts"].get("calibration_owner")


@pytest.mark.parametrize("backend", ("libero_pro", "robotwin"))
def test_a_surface_without_the_capability_keeps_the_legacy_message(tmp_path, backend):
    legacy, _, _, _, _, _ = _bound_context(tmp_path, backend)
    # The kept surface that predates delivery trim (runtime_v1 RoboTwin qualification).
    frozen = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    assert not frozen.supports(trim.CAPABILITY)
    assert policy_delivery.deliver_public_facts(
        legacy["public_robot_facts"], capabilities=("observation_delivery_trim_v0",)) is legacy["public_robot_facts"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_canonical_tool_result_keeps_every_byte(backend):
    from ora.agent.native.gateway import NativeToolGateway

    instance = load_native_baseline(ROOT)["instances"][backend]
    surface = resolve_tool_surface(instance["tool_surface"])
    canonical = {"status": "ok", "candidates": [{"position_m": [0.20466682314872742, -0.1, 0.9],
                                                 "orientation_error_deg": 2.864789}]}
    before = copy.deepcopy(canonical)

    class Runtime:
        def __init__(self):
            self.surface = surface

    delivered = NativeToolGateway(Runtime()).record_proposed_result("move_pose", {"payload": {"arguments": {}}},
                                                                    canonical)
    assert canonical == before
    # The delivered copy keeps the exact digits.
    assert delivered["candidates"] == canonical["candidates"]
    # A tool result the policy authored is never rewritten.
    authored = {"outputs": [{"name": "x", "value": 0.20466682314872742}]}
    assert NativeToolGateway(Runtime()).record_proposed_result(
        "compute", {"payload": {"arguments": {}}}, authored) == authored


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_aged_trimmed_observation_reopens_to_exactly_what_the_policy_read(tmp_path, backend):
    """Lossless aging stores the delivered text; reopening its pointer returns those bytes.

    The policy never held more than the trimmed copy, so the recoverable
    observation is that copy byte for byte, with every marker and the round
    slot listing it was delivered with.
    """
    from tests.agent.context.test_context_near_window import TOOLS, history, pipeline_at

    golden = json.loads((ROOT / f"tests/data/delivery_trim/{backend}-delivered.json").read_text(encoding="utf-8"))
    pipeline = pipeline_at(tmp_path, trigger_estimated_tokens=1)
    messages = history(5)
    delivered = {}
    for message in messages:
        if "_observation" not in message:
            continue
        step = message["_observation"]["step"]
        envelope = copy.deepcopy(golden["observation"])
        envelope["current_actual"] = {**envelope["current_actual"], "step": step}
        text = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
        message["content"][0]["text"] = text
        delivered[step] = text
    projection = pipeline.project(messages=messages, tools=TOOLS, observation_step=5)
    manifest = projection["manifest"]
    assert manifest["lossless_invariants_before_sha256"] == manifest["lossless_invariants_after_sha256"]
    reopened = 0
    for message in projection["messages"]:
        if message["role"] != "user" or not isinstance(message["content"], list):
            continue
        body = json.loads(message["content"][0]["text"])
        if body.get("observation_externalized"):
            _, recovered = pipeline.context_artifacts.resolve(body["recoverable_observation"]["ref"])
            step = body["current_actual"]["step"]
            assert recovered.decode("utf-8") == delivered[step]
            assert json.loads(recovered)["last_action_effect"] == golden["observation"]["last_action_effect"]
            reopened += 1
    assert reopened >= 2


def test_the_full_image_marker_is_unchanged(tmp_path):
    from ora.agent.memory.artifacts import decode_recoverable_artifact_marker
    from tests.agent.context.test_context_near_window import TOOLS, history, pipeline_at

    pipeline = pipeline_at(tmp_path)
    projection = pipeline.project(messages=history(2), tools=TOOLS, observation_step=2)
    markers = [block["text"] for message in projection["messages"] if isinstance(message.get("content"), list)
               for block in message["content"]
               if block.get("type") == "text" and block["text"].startswith("<recoverable_artifact>")]
    assert markers and all(set(decode_recoverable_artifact_marker(marker)) == {
        "v", "ref", "mime", "bytes", "at", "open"} for marker in markers)
