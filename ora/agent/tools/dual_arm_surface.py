"""Backend-aware camera/compiler selection and closed result projection."""
from __future__ import annotations
from typing import Any, Mapping

def dual_arm_grasp_scene_cameras(surface: Any, arm: str) -> tuple[str, str]:
    """Head plus the planning arm's wrist under the surface's camera names."""

    if surface.supports(ROBODOJO_RGBD_CAPABILITY):
        return ("head", f"{arm}_wrist")
    return ("head_camera", f"{arm}_camera")


def dual_arm_grasp_compiler_id(surface: Any, arm: str) -> str | None:
    """Select the registered per-arm compiler the surface declares."""

    if surface.supports("robodojo_x5_public_rgbd_topdown_grasp_v1"):
        return f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2"
    if surface.supports(ARX_X5_GRASP_COMPILER_CAPABILITY):
        return f"contact_graspnet_to_robodojo_x5_{arm}_link6_v1"
    if surface.supports("contact_graspnet_robotwin_aloha_grip_center_compiler_v3"):
        return f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v3"
    if surface.supports("dual_arm_grasp_plan_v1"):
        return f"contact_graspnet_to_robotwin_aloha_{arm}_end_effector_v1"
    return None


def conform_to_result_schema(
    schema: Any, result: dict[str, Any],
) -> dict[str, Any]:
    """Drop fields a closed result schema does not advertise.

    A tool that returns a field its own advertised schema forbids is a dead
    tool: `validate_schema_instance` rejects the call as `unknown result
    fields` and the policy gets a non-retryable internal error rather than an
    answer. `vision.ground` was in exactly that state on 19 of the 23
    registered surfaces exposing it, because the builder emits
    `candidate_previews` unconditionally while most surfaces never declared
    it. Dropping is the honest direction -- a surface that never promised a
    field has no policy relying on one -- and it keeps the fix in the runtime
    rather than in nineteen hash-pinned schema components.
    """

    if (
        not isinstance(schema, Mapping)
        or schema.get("additionalProperties") is not False
    ):
        return result
    advertised = schema.get("properties")
    if not isinstance(advertised, Mapping):
        return result
    return {
        name: value for name, value in result.items() if name in advertised
    }



ROBODOJO_RGBD_CAPABILITY = "robodojo_rgbd_v1"

ARX_X5_GRASP_COMPILER_CAPABILITY = (
    "contact_graspnet_robodojo_x5_link6_compiler_v1"
)

#: In-memory per-round registries of the perception/motion surface. They are
#: cleared by ``reset_round`` and audited by ``round_start_isolation``.


PER_ROUND_RUNTIME_REGISTRIES: dict[str, Any] = {
    "_active_mark_point_batch": None,
    "_known_plan_refs": {},
    "_submitted_mutations": [],
}

#: Every field `grasp.grounding_public_result` builds. Named here so the
#: catalog contract test can check each registered surface against the real
#: served shape instead of a shape the test invented.


GROUNDING_RESULT_FIELDS = (
    "success", "status", "observation_id", "camera", "text", "grounding_ref",
    "backend", "candidates", "diagnostics", "candidate_previews", "overlay",
)
