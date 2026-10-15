"""New work does not make an oversized module larger.

The repository has files of 8,000 and 5,700 lines. Nobody set out to write
them: each change added the twenty lines that were obviously easiest to add
where the surrounding code already was. This ratchet makes that path the harder
one without demanding the files be split today.

It deliberately does not enforce a global limit. Splitting `robotwin/session.py`
is its own reviewed change with its own risk; failing every commit until someone
does it would only teach people to raise the threshold.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Above this, a module is past the point where a reader holds it in mind, and
# the repository's own guidance calls for extraction (200-400 typical, 800 max).
OVERSIZED_LINES = 800

# Files already past the line when this ratchet was written. They may shrink and
# they may stay still; they may not grow. Shrinking one below OVERSIZED_LINES
# means deleting its entry -- a stale exemption is a lie about the codebase.
#
# A raised entry carries `(lines, reason)` instead of a bare count. Raising one
# is the move that quietly kills a ratchet, so it costs a sentence that a
# reviewer can disagree with. A bare int means the number has never been raised.
BASELINE = {
    "ora/agent/context/native.py": (1133, (
        "The deferred-eager fix for cache_prefix_rewritten. Full-current "
        "delivery defers L1/L2 into the open ActionStep, so closing that step "
        "rewrites already-delivered results while L3 retains every image and "
        "L4 has no pressure trigger -- a rewrite nothing else announces, which "
        "killed a run after nine clean append-only requests. Two pieces did "
        "move out to context/thresholds.py and context/deferred_eager.py: the "
        "layer thresholds and their ordering, and the coverage predicate with "
        "its transition record. What stays is the decision to re-render and "
        "compare, which closes over ten locals inside _project_action_step_"
        "lossless_v4; lifting that would pass those ten across a boundary to "
        "save eleven lines."
    )),
    "ora/agent/context/projection.py": (1339, (
        "episode_index_keyframe moved here from native/session.py with the "
        "docstring the heuristic never had. Same change removed 235 lines from "
        "orchestrator/stagnation.py and 17 from session.py -- 225 fewer lines "
        "across the three. The heuristic reads only what _collapse_old_groups "
        "in this file consumes, so its own module would be one function with "
        "one caller."
    )),
    "ora/agent/evidence/trajectory.py": 988,
    "ora/agent/memory/artifacts.py": 985,
    "ora/agent/native/session.py": 1790,
    "ora/agent/providers/native.py": 1827,
    "ora/agent/tools/grasp.py": (3621, (
        "The target's principal frame. The segmentation localizer already ran a "
        "full eigendecomposition and kept one of three vectors, so a policy "
        "could read a target's position and size but never its orientation -- "
        "and a QR board faces the way its least-extent axis points, a fan's "
        "facing needs two axes, an upright bottle is its greatest-extent axis. "
        "All three stand at zero across every cohort. What was added is the "
        "measurement that was already being computed and discarded, plus the "
        "capability gate that keeps it off surfaces whose schemas forbid "
        "unknown properties; the diagnostics assembly this file used to hold "
        "left for grasp_diagnostics.py in the same batch."
    )),
    "ora/agent/tools/robot.py": 1224,
    "ora/environment/backends/libero_pro/robustness.py": 1075,
    "ora/environment/backends/robotwin/planning.py": (1122, (
        "Record the latest target 83943397 planning-diagnostics repair as the "
        "integration baseline. This file is byte-identical to that target; "
        "0048 introduces no growth or unrelated controller refactor here."
    )),
    "ora/environment/backends/robotwin/vendored/planner.py": (1489, (
        "Record the latest target 83943397 planning-diagnostics repair as the "
        "integration baseline. This file is byte-identical to that target; "
        "0048 introduces no growth or unrelated controller refactor here."
    )),
    "ora/environment/controllers/motion_transaction.py": 1553,
    "ora/environment/replay_worker.py": 1098,
    "ora/orchestrator/cli.py": 1013,
}


def _tracked_python_files() -> list[str]:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "ora/**/*.py"],
        capture_output=True, text=True, check=True,
    )
    return [name for name in listed.stdout.split() if name.endswith(".py")]


def _line_count(name: str) -> int:
    return len((ROOT / name).read_text(encoding="utf-8").splitlines())


def _recorded(name: str) -> int:
    entry = BASELINE[name]
    return entry[0] if isinstance(entry, tuple) else entry


# Registering one read-only environment tool costs a dataclass field, a factory
# passthrough and a coordinator binding -- each a handful of lines that cannot
# live anywhere else. A ratchet that forbids those forbids adding tools, so it
# would be raised or deleted rather than obeyed. This allowance is per file and
# small enough that it cannot absorb a feature: 24 lines is one thin forwarder,
# not an implementation.
WIRING_ALLOWANCE_LINES = 25


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_an_oversized_module_never_grows_beyond_unavoidable_wiring(name):
    """The ratchet turns one way, with room only for a forwarder."""
    current = _line_count(name)
    assert current <= _recorded(name) + WIRING_ALLOWANCE_LINES, (
        f"{name} grew from {_recorded(name)} to {current} lines, past the "
        f"{WIRING_ALLOWANCE_LINES}-line wiring allowance. It is already over "
        f"{OVERSIZED_LINES}; put the work in its own module and leave a forwarder, "
        f"as controllers/alignment.py and agent/tools/alignment.py do."
    )


def test_no_module_crosses_the_line_without_being_recorded():
    """A file that newly exceeds the limit must be a deliberate, visible choice."""
    crossed = {
        name: _line_count(name)
        for name in _tracked_python_files()
        if name not in BASELINE and _line_count(name) > OVERSIZED_LINES
    }
    assert not crossed, (
        f"these modules crossed {OVERSIZED_LINES} lines: {crossed}. Split them, or "
        "add them to BASELINE with a reason in the commit message."
    )


def test_every_raised_baseline_states_why():
    """Raising a number is what kills a ratchet, so it costs a sentence."""
    unreasoned = {
        name: entry for name, entry in BASELINE.items()
        if isinstance(entry, tuple) and not (len(entry) == 2 and str(entry[1]).strip())
    }
    assert not unreasoned, (
        f"these raised baselines carry no reason: {sorted(unreasoned)}. A raised "
        "entry is (lines, reason); the reason is what a reviewer disagrees with."
    )


def test_a_shrunk_module_loses_its_exemption():
    """An exemption that no longer applies must be deleted, not left to rot."""
    stale = {
        name: _line_count(name)
        for name in BASELINE
        if _line_count(name) <= OVERSIZED_LINES
    }
    assert not stale, (
        f"these modules no longer need an exemption: {stale}. Remove them from "
        "BASELINE so it keeps describing the codebase that exists."
    )
