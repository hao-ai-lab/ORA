"""ARX X5 embodiment card v5: the public jaw reading is both fingers' mean (E11G12-001).

The v4 card said "The public encoder reading is normalized joint position",
which the backend published from the first finger alone. Since EXP11A g013 the
public opening is the mean of both finger joints on the same normalized command
scale, and the finger pair can shift to one side under load. Descriptor v5
changes only the card projection (``full_finger_reference_v3`` = v2 plus that
sentence); geometry, axes, controlled points and provenance are v4's, so the
loaded-geometry evidence stays reusable. The live Dojo bundle, environment
profile v6 and the closed-skills surface pin v5. The v4 predecessor left the
open-source tree, so the v4-equivalence comparisons are no longer pinned here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card

ROOT = Path(__file__).resolve().parents[1]
ARX = ROOT / "ora/policy/embodiments/arx-x5"
V5 = ARX / "descriptor-public-geometry-v5.json"
CARD_V5 = ARX / "manual-public-geometry-v5.md"
LIVE_BUNDLE = ROOT / "ora/policy/prompt_bundles/current_v4/robodojo-x5-native_api-v4.yaml"
ENVIRONMENTS = ROOT / "ora/policy/environments/robodojo"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_compiled_card_states_the_two_finger_encoder_sentence():
    card = render_embodiment_card(load_embodiment_descriptor(V5))
    assert card == CARD_V5.read_text(encoding="utf-8")
    text = _flat(card)
    assert "normalized joint position" not in text
    assert ("The public encoder reading is the mean of both finger joints on the normalized command scale "
            "(empty-close reference about 0.185, full-open 1); it is not measured pad separation, and a pair "
            "shifted to one side is no evidence of holding or release.") in text
    assert (len(card.encode()) + 3) // 4 <= 1200


def test_the_live_referrers_pin_v5():
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.embodiment_geometry import PUBLIC_GEOMETRY_DESCRIPTORS
    from ora.agent.composition.environment_context import resolve_environment_prompt_profile
    from ora.environment.benchmarks.robodojo import public_context

    bundle = yaml.safe_load(LIVE_BUNDLE.read_text(encoding="utf-8"))
    assert bundle["layers"]["embodiment"]["source"] == str(CARD_V5.relative_to(ROOT))
    assert bundle["layers"]["embodiment"]["descriptor"] == str(V5.relative_to(ROOT))
    profile_id = bundle["bindings"]["environment_prompt_profile"]["profile_id"]
    assert profile_id == "robodojo_public_explicit_v6"
    profile = json.loads((ENVIRONMENTS / "context-profile-public-explicit-v6.json").read_text())
    compiled = profile["compiled_embodiment"]
    # Named by path; the resolved bindings carry the sha256 of the bytes read.
    assert compiled["descriptor"] == str(V5.relative_to(ROOT))
    assert compiled["card_sha256"] == _sha(CARD_V5)
    review = json.loads((ROOT / compiled["review"]).read_text())
    # The review attests the card text; the descriptor's bytes are policy.lock's.
    assert "descriptor_sha256" not in review and review["card_sha256"] == _sha(CARD_V5)
    surface = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    assert surface.view_config["public_geometry_descriptor"] == {"path": compiled["descriptor"], "sha256": _sha(V5)}
    assert PUBLIC_GEOMETRY_DESCRIPTORS["arx_x5_link6"] == "arx-x5/descriptor-public-geometry-v5.json"
    resolved = resolve_environment_prompt_profile(public_context(), project_root=ROOT, profile_id=profile_id)
    assert resolved["embodiment_policy"] == str(CARD_V5.relative_to(ROOT))
