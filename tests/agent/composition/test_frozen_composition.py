"""Compositions this tree no longer composes are refused before launch.

The RunSpec resolver refuses a RoboTwin control profile older than the oldest
one composed, and policy values the slimming stages removed from its accepted
sets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.environment.controllers.profile_capabilities import (
    FROZEN_CONTROL_PROFILE_SHA256, OLDEST_COMPOSED_CONTROL_PROFILE, PROFILE_VERSION_IDS,
)
from ora.orchestrator.defaults import expand_native_defaults
from ora.orchestrator.spec import resolve_run_spec
from tests.support.run_specs import native_spec

ROOT = Path(__file__).resolve().parents[3]


def _explicit_robotwin_spec(version: int) -> dict:
    """The default RoboTwin composition as an explicit-ToolSurface RunSpec bound to profile ``version``.

    Without ``policy.baseline`` the resolver keeps the spec's own
    ``runtime.control_profile``, so any registered profile can be named.
    """
    spec = expand_native_defaults(native_spec("robotwin"), root=ROOT)
    del spec["policy"]["baseline"]
    spec["runtime"]["control_profile"] = {
        "path": f"ora/policy/embodiments/aloha-agilex/control-profile-v{version}.json",
        "sha256": FROZEN_CONTROL_PROFILE_SHA256[version],
    }
    return spec


@pytest.mark.parametrize("version", [v for v in sorted(PROFILE_VERSION_IDS) if v < OLDEST_COMPOSED_CONTROL_PROFILE])
def test_a_control_profile_older_than_the_oldest_composed_is_refused_before_launch(version, tmp_path):
    from ora.environment.backends.robotwin.session import RoboTwinLiveError, RoboTwinLiveSession
    from ora.environment.controllers.embodiment_control import load_embodiment_control_profile
    from ora.orchestrator.spec_values import RunSpecError

    spec = _explicit_robotwin_spec(version)
    with pytest.raises(RunSpecError, match=(
        f"runtime.control_profile v{version} is older than the oldest control profile "
        f"this tree composes \\(v{OLDEST_COMPOSED_CONTROL_PROFILE}\\)"
    )):
        resolve_run_spec(spec, root=ROOT)
    # The worker's session refuses the profile too, whatever bound it. Only the
    # oldest composed profile's file is in this tree; the session reads the
    # loaded profile's identity, so the older one is that file under its id.
    current = load_embodiment_control_profile(
        ROOT / f"ora/policy/embodiments/aloha-agilex/control-profile-v{OLDEST_COMPOSED_CONTROL_PROFILE}.json",
        FROZEN_CONTROL_PROFILE_SHA256[OLDEST_COMPOSED_CONTROL_PROFILE],
    )
    binding = {**current, "profile": {**current["profile"], "profile_id": PROFILE_VERSION_IDS[version]}}
    with pytest.raises(RoboTwinLiveError, match=f"control profile of version >= {OLDEST_COMPOSED_CONTROL_PROFILE}"):
        RoboTwinLiveSession(
            robotwin_root=tmp_path, output_root=tmp_path / "output", task="blocks_ranking_size",
            actual_seed=0, episode=0, action_codec=spec["policy"]["action_codec"],
            control_profile=binding, task_factory=lambda _task: None,
        )


def test_the_oldest_composed_control_profile_resolves_and_binds(tmp_path):
    from ora.environment import factory
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession

    plan = resolve_run_spec(_explicit_robotwin_spec(OLDEST_COMPOSED_CONTROL_PROFILE), root=ROOT)
    assert plan["runtime"]["control_profile"]["path"].endswith(
        f"control-profile-v{OLDEST_COMPOSED_CONTROL_PROFILE}.json")
    binding = factory.validate_robotwin_control_binding(
        plan["runtime"], action_codec=plan["policy"]["action_codec"], project_root=ROOT,
    )
    assert binding["profile"]["profile_id"] == PROFILE_VERSION_IDS[OLDEST_COMPOSED_CONTROL_PROFILE]
    live = RoboTwinLiveSession(
        robotwin_root=tmp_path, output_root=tmp_path / "output", task="blocks_ranking_size",
        actual_seed=0, episode=0, action_codec=plan["policy"]["action_codec"],
        control_profile=binding, task_factory=lambda _task: None,
    )
    assert live._uses_motion_transaction_profile()


#: Values the slimming stage removed from the resolver's accepted sets, as
#: (policy field, value, pattern of the RunSpecError the resolver raises).
#: Each is refused at resolution.
REMOVED_POLICY_VALUES: list[tuple[str, str, str]] = [
    # Unit B2: the old Native Context.
    ("context_retention", "append_only_epoch_v1", "unsupported native context retention"),
    ("context_retention", "action_step_pressure_prune_v1", "unsupported native context retention"),
    ("context_retention", "action_step_pressure_compact_v1", "unsupported native context retention"),
    ("context_profile", "native_context_v1", "unsupported native context profile"),
    ("provider_continuation_replay", "visible_only_v1", "provider_continuation_replay must be exact_v1"),
    # Unit B3: Active Task Memory; ORA removed Task Memory altogether.
    ("memory_policy", "active_task_memory_v1", "policy.memory_policy must be disabled"),
    # Unit B5: the motion_action_v1/v2 policy codecs.
    ("action_codec", "motion_action_v1", "policy.action_codec must be"),
    ("action_codec", "motion_action_v2", "policy.action_codec must be"),
]


def _explicit_default_spec() -> dict:
    """The default RoboTwin RunSpec with every field explicit and no baseline marker.

    The baseline layer refuses any override of a field it binds before the
    accepted-set check runs, so the removed value is set on the explicit spec.
    """
    spec = expand_native_defaults(native_spec("robotwin"), root=ROOT)
    del spec["policy"]["baseline"]
    return spec


@pytest.mark.parametrize("field,value,refusal", REMOVED_POLICY_VALUES)
def test_a_removed_policy_value_is_refused(field, value, refusal):
    from ora.orchestrator.spec_values import RunSpecError

    spec = _explicit_default_spec()
    spec["policy"][field] = value
    with pytest.raises(RunSpecError, match=refusal):
        resolve_run_spec(spec, root=ROOT)


def test_the_provider_image_window_of_the_pressure_retentions_is_refused():
    from ora.orchestrator.spec_values import RunSpecError

    for spec in (native_spec("robotwin"), _explicit_default_spec()):
        spec["policy"]["max_inline_images_per_request"] = 4
        with pytest.raises(RunSpecError, match="max_inline_images_per_request is not supported"):
            resolve_run_spec(spec, root=ROOT)
