"""Head-image distance recorded across every executed action boundary.

This module once owned a repeated-action guard. It judged whether an action
had produced public progress, whether it repeated the previous action,
accumulated a streak from the two, raised a warning, and offered the
coordinator a `is_high_confidence` verdict that became an
`action_stagnation_advisory` event and a "progress has stalled, change your
approach" prompt stage. All of that was retired. Three reasons, in order of
weight:

1. Forming that judgement and delivering it to the policy is the Harness
   deciding for the policy. `AGENTS.md` gives tools measurement, not
   decisions, and the surviving instruction told the policy what to conclude
   and what to do next.
2. Repairing it was not a one-line fix. It resolved the acting arm through
   `arguments["arm"]`, and the sparse motion codec's arguments are
   `{interactions, targets}` with no `arm` key, so both lookups returned an
   empty record and both derived quantities were null on every executed
   action. Restoring that read would have brought back the displacement
   everywhere but the aperture change only where the receipt carries a
   numeric one: `libero-sparse-motion-receipt.v1`,
   `libero-pro-hybrid-motion-receipt.v2` and
   `robodojo-sparse-motion-receipt.v1` do, while
   `robotwin-sparse-motion-receipt.v7` carries only a categorical
   `gripper_command_target`. `no_public_progress` requires all three terms,
   so the repaired guard would have been alive on LIBERO and still silently
   constant-false on RoboTwin.
3. Its seven thresholds were bare dataclass defaults, bound to no profile,
   named in no manifest and covered by no test, unlike every other numeric
   constant in this repository.

What survives is the one quantity that was actually measured rather than
judged: the perceptual-hash distance between the main public image before and
after the action. It is recorded as Harness-owned run evidence. It is
deliberately not delivered to the policy, because a
zero distance is not evidence that nothing moved. Measured over 7,056 paired
records from 91 runs, 1,594 action boundaries left the head image bit
identical, and 863 of those had the gripper actuate or the end effector move
more than 5 mm.

The artifact name `stagnation.json` and the event type
`stagnation_observation` are retained so runs sealed before this change stay
readable with one vocabulary.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


OBSERVATION_DELTA_SCHEMA = "ora.observation-delta.v3"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _phash(path: Path) -> int:
    """Reduce one image to the 64 lowest-frequency DCT bits.

    The repository's only image comparison across an action boundary. A
    whole-scene camera at working distance does not resolve the gripper jaws,
    and the retained 8x8 block discards everything finer, so equality here
    means the coarse scene layout is unchanged and nothing more.
    """

    import numpy as np
    from PIL import Image

    pixels = np.asarray(
        Image.open(path).convert("L").resize((32, 32)), dtype=np.float64
    )
    indexes = np.arange(32, dtype=np.float64)
    basis = np.cos((2 * indexes[:, None] + 1) * indexes[None, :] * math.pi / 64.0)
    coefficients = basis.T @ pixels @ basis
    low = coefficients[:8, :8]
    median = float(np.median(low[1:]))
    bits = low >= median
    result = 0
    for value in bits.flat:
        result = (result << 1) | int(bool(value))
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass
class StagnationGuard:
    """Record the main-image distance across each executed action boundary.

    The name is retained with the artifact and event it writes. Nothing is
    guarded: there is no threshold, no boolean and no advisory, and the
    Supervisor reads no verdict from this class.
    """

    state_path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state_path = self.state_path.resolve()
        self._persist()

    def observe(self, *, before_image: Path, after_image: Path) -> dict[str, Any]:
        record = {
            "head_phash_hamming": _hamming(
                _phash(before_image), _phash(after_image)
            ),
        }
        self.records.append(record)
        self._persist()
        return record

    def _persist(self) -> None:
        _atomic_json(self.state_path, {
            "schema": OBSERVATION_DELTA_SCHEMA,
            "records": list(self.records),
        })
