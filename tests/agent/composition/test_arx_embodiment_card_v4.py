"""ARX X5 embodiment card v4: one controlled-point name, a named pad axis, the contact centre.

The v3 card called link6 the "controlled reference" in one section and the
"controlled point" in the next, gave the nominal pad interval "along local
lateral" although it runs along link6 local Z (the jaw-closing line is local
Y), and stated the jaw contact centre as "150.1 mm along approach" while the
tool help says "0.1501 m ahead of link6 along the approach axis". Descriptor v4
fixed the card projection (``full_finger_reference_v2``). The v3 and v4
descriptors are no longer in this tree; the live v5 card keeps v4's projection
plus one encoder sentence (test_arx_embodiment_card_v5), so the v4 wording is
checked on the live card.
"""

from __future__ import annotations

from pathlib import Path

from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
from ora.agent.tools.arx_x5_gripper import ARX_X5_GRIPPER_GEOMETRY

ROOT = Path(__file__).resolve().parents[3]
ARX = ROOT / "ora/policy/embodiments/arx-x5"
LIVE = ARX / "descriptor-public-geometry-v5.json"
LIVE_CARD = ARX / "manual-public-geometry-v5.md"


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_compiled_card_names_one_controlled_point_the_pad_axis_and_the_contact_centre():
    card = render_embodiment_card(load_embodiment_descriptor(LIVE))
    assert card == LIVE_CARD.read_text(encoding="utf-8")
    text = _flat(card)
    assert "controlled reference" not in text and "local lateral" not in text
    assert "`left_link6` is the left arm's controlled point, its wrist flange" in text
    assert "Local Z is perpendicular to both." in text
    assert "142.6–157.6 mm along the approach axis and -10.4–8 mm along local Z" in text
    # The same number the grasp compiler and the observe help use.
    centre = ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"]
    assert f"The jaw contact centre is +{centre:.4f} m along the approach axis from `left_link6`" in text
    assert ARX_X5_GRIPPER_GEOMETRY["pad_inner_face_span_link6_z_m"] == [-0.0104, 0.0080]
    for side in ("left", "right"):
        assert f"`{side}_link6`" in text

