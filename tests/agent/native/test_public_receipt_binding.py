from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.agent.evidence.receipt_binding import canonical_public_receipt


@pytest.mark.parametrize("damage", [None, "pose_change", "boolean", "nonfinite", "unknown_numeric", "extra_key", "action_id", "source_hash"])
def test_public_action_join_tolerates_only_known_pose_normalization(damage):
    import copy
    from ora.agent.evidence.receipt_binding import public_actions_match

    # Exact archived G Goal-3 step142 values, differing by one final float bit.
    expected = {"schema": "ora-motion-action.v3", "action": "move_pose",
        "action_id": "joined-action", "observation_id": "source-observation",
        "bound_observation_sha256": "a" * 64,
        "arguments": {"interactions": {"right": "precision"}, "targets": {"right": {
            "position_m": [0.0, 0.051862973884, 1.0132046114],
            "orientation_xyzw": [0.9867773877625678, 0.09426295972795873,
                                  0.13021380006894298, 0.020718293737990928]}}}}
    actual = copy.deepcopy(expected)
    actual["arguments"]["targets"]["right"]["orientation_xyzw"][:2] = [0.9867773877625677, 0.09426295972795871]
    if damage == "pose_change":
        actual["arguments"]["targets"]["right"]["position_m"][0] = 1e-6
    elif damage == "boolean":
        actual["arguments"]["targets"]["right"]["position_m"][0] = False
    elif damage == "nonfinite":
        actual["arguments"]["targets"]["right"]["position_m"][0] = float("inf")
    elif damage == "unknown_numeric":
        expected["arguments"]["unregistered"] = 0.0
        actual["arguments"]["unregistered"] = 1e-16
    elif damage == "extra_key":
        actual["arguments"]["targets"]["right"]["extra"] = None
    elif damage == "action_id":
        actual["action_id"] = "different-action"
    elif damage == "source_hash":
        actual["bound_observation_sha256"] = "b" * 64
    assert public_actions_match(expected, actual) is (damage is None)


def _wrapper(root: Path) -> tuple[dict, dict]:
    receipt = {
        "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action_id": "close-1", "action": "set_gripper", "consumed": True,
        "motion_status": "not_reached", "state_changed": True, "steps_used": 80,
    }
    path = root / "native/receipts/action-0000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))
    return {"previous_execution_receipt": {
        "path": "receipts/action-0000.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "receipt": receipt,
    }}, receipt


def test_direct_and_environment_wrapped_receipts_resolve_to_the_same_evidence(tmp_path: Path) -> None:
    state, receipt = _wrapper(tmp_path)
    assert canonical_public_receipt(state, tmp_path) == receipt
    direct = {"previous_execution_receipt": receipt}
    resolved = canonical_public_receipt(direct, tmp_path)
    assert resolved == receipt
    resolved["steps_used"] = 0
    assert receipt["steps_used"] == 80


@pytest.mark.parametrize("corruption", [
    "hash", "inline", "missing", "absolute", "traversal", "ambiguous", "symlink",
    "parent_symlink", "nul_path", "invalid_utf8",
])
def test_wrapper_must_bind_one_unchanged_regular_receipt(tmp_path: Path, corruption: str) -> None:
    state, receipt = _wrapper(tmp_path)
    wrapper = state["previous_execution_receipt"]
    path = tmp_path / "native/receipts/action-0000.json"
    if corruption == "hash":
        wrapper["sha256"] = "0" * 64
    elif corruption == "inline":
        wrapper["receipt"] = {**receipt, "steps_used": 0}
    elif corruption == "missing":
        path.unlink()
    elif corruption == "absolute":
        wrapper["path"] = str(path)
    elif corruption == "traversal":
        wrapper["path"] = "../native/receipts/action-0000.json"
    elif corruption == "ambiguous":
        second = tmp_path / "receipts/action-0000.json"
        second.parent.mkdir()
        second.write_bytes(path.read_bytes())
    elif corruption == "parent_symlink":
        target = tmp_path.parent / f"{tmp_path.name}-outside"
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    elif corruption == "nul_path":
        wrapper["path"] = "receipts/invalid\0.json"
    elif corruption == "invalid_utf8":
        path.write_bytes(b"\xff")
        wrapper["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        target = path.with_name("canonical.json")
        path.rename(target)
        path.symlink_to(target)
    assert canonical_public_receipt(state, tmp_path) is None


def test_verified_wrapper_returns_only_the_sanitized_public_copy(tmp_path: Path) -> None:
    from ora.agent.evidence.policy_state import _sanitize_embedded_receipt

    state, receipt = _wrapper(tmp_path)
    receipt["controller_tracking"] = {"phases": {"global_precontact": {"steps": 1}}}
    path = tmp_path / "native/receipts/action-0000.json"
    path.write_text(json.dumps(receipt))
    state["previous_execution_receipt"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state["previous_execution_receipt"]["receipt"] = _sanitize_embedded_receipt(receipt)
    public = canonical_public_receipt(state, tmp_path)
    assert public == _sanitize_embedded_receipt(receipt)
    assert "global_precontact" not in public["controller_tracking"]["phases"]
