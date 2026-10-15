"""One run emits one action codec, whichever backend the surface is for.

`set_gripper` routed to its v3 implementation only when a surface declared
`dual_arm_sparse_targets_v1`. The active RoboTwin surface declares that; the
active LIBERO surface declares `single_arm_flat_targets_v1` instead. So on
LIBERO `set_gripper` fell through to the flat path and emitted
`ora-motion-action.v1` while `open_gripper` on the same surface
emitted v3 -- two codecs inside one run, which `orchestrator/control_audit.py`
compares as `action_codec_identity`.

The two helpers that read the surface's declared topology already existed and
already handled both shapes. The routing gate was the only thing that did not.
"""
from __future__ import annotations

from typing import Any

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.tools.runtime import ManipulationToolRuntime

ACTIVE = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}


def _runtime(profile: str) -> tuple[Any, dict[str, Any]]:
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(profile)
    submitted: dict[str, Any] = {}
    runtime._ensure_mutation_available = lambda _name: None
    runtime._current_grip_state = lambda: (
        {"observation_id": "observation-0"},
        None,
        {"proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": [0.0] * 7},
            "right": {"pose_xyz_xyzw": [0.0] * 7},
        }}},
        None,
        None,
    )
    runtime._atomic_arm = lambda _observation: "right"

    def submit(**kwargs: Any) -> dict[str, Any]:
        submitted.update(kwargs)
        return {"status": "submitted", "action": kwargs["public_action_name"]}

    runtime._atomic_submit = submit
    return runtime, submitted


def test_the_two_active_surfaces_declare_different_topology_capabilities():
    """The premise. Widening one gate is only safe because these differ."""

    robotwin = resolve_tool_surface(ACTIVE["robotwin"])
    libero = resolve_tool_surface(ACTIVE["libero_pro"])
    assert robotwin.supports("dual_arm_sparse_targets_v1")
    assert not robotwin.supports("single_arm_flat_targets_v1")
    assert libero.supports("single_arm_flat_targets_v1")
    assert not libero.supports("dual_arm_sparse_targets_v1")


@pytest.mark.parametrize("backend", sorted(ACTIVE))
def test_set_gripper_emits_the_same_codec_as_the_pair_it_replaces(backend):
    """set_gripper and open_gripper must not disagree about the codec."""

    runtime, submitted = _runtime(ACTIVE[backend])
    arguments = (
        {"action": "close"} if backend == "libero_pro"
        else {"targets": {"right": "close"}}
    )
    runtime._motion_set_gripper(arguments)
    via_set_gripper = submitted["schema"]

    runtime, submitted = _runtime(ACTIVE[backend])
    # A dual-arm surface makes open_gripper name its arm; a single-arm one has
    # only the one. That asymmetry is the pair's limitation, not a codec choice.
    runtime._atomic_open_gripper({} if backend == "libero_pro" else {"arm": "right"})
    assert submitted["schema"] == via_set_gripper, (
        "one run must not carry two action codecs"
    )


@pytest.mark.parametrize("backend", sorted(ACTIVE))
def test_set_gripper_compiles_to_the_shared_sparse_codec(backend):
    """Whatever shape the surface takes, the submitted action is one shape."""

    runtime, submitted = _runtime(ACTIVE[backend])
    arguments = (
        {"action": "close"} if backend == "libero_pro"
        else {"targets": {"left": "close", "right": "open"}}
    )
    runtime._motion_set_gripper(arguments)
    expected = (
        {"right": "close"} if backend == "libero_pro"
        else {"left": "close", "right": "open"}
    )
    assert submitted["arguments"] == {"targets": expected}


def test_one_call_sets_both_grippers_where_the_embodiment_has_two():
    """The coordination the pair could not express, and the reason for it."""

    runtime, submitted = _runtime(ACTIVE["robotwin"])
    runtime._motion_set_gripper({"targets": {"left": "close", "right": "open"}})
    assert submitted["arguments"]["targets"] == {"left": "close", "right": "open"}


@pytest.mark.parametrize("backend", sorted(ACTIVE))
def test_closing_carries_the_intent_annotation_the_pair_published(backend):
    """close_gripper's target_ref survives the tool that replaces it.

    It is an annotation and never evidence: the result says so in the same
    breath it reports the reference. Dropping it while retiring close_gripper
    would remove a contract both default surfaces publish, quietly.
    """

    runtime, _ = _runtime(ACTIVE[backend])
    arguments: dict[str, Any] = (
        {"action": "close"} if backend == "libero_pro"
        else {"targets": {"right": "close"}}
    )
    plain = runtime._motion_set_gripper(dict(arguments))
    assert plain["intent"] == {"target_ref": None, "role": "agent_annotation_only"}

    runtime, _ = _runtime(ACTIVE[backend])
    annotated = runtime._motion_set_gripper({**arguments, "target_ref": "target-abc"})
    assert annotated["intent"] == {
        "target_ref": "target-abc", "role": "agent_annotation_only",
    }


@pytest.mark.parametrize("backend", sorted(ACTIVE))
def test_opening_alone_carries_no_close_intent(backend):
    """An annotation about what is being grasped has no meaning on release."""

    runtime, _ = _runtime(ACTIVE[backend])
    arguments = (
        {"action": "open"} if backend == "libero_pro"
        else {"targets": {"right": "open"}}
    )
    assert "intent" not in runtime._motion_set_gripper(arguments)


@pytest.mark.parametrize("backend", sorted(ACTIVE))
def test_an_empty_target_ref_is_refused_by_name(backend):
    """A blank annotation is a mistake, not an absent one."""

    runtime, _ = _runtime(ACTIVE[backend])
    arguments = (
        {"action": "close"} if backend == "libero_pro"
        else {"targets": {"right": "close"}}
    )
    with pytest.raises(Exception, match="target_ref"):
        runtime._motion_set_gripper({**arguments, "target_ref": "   "})
