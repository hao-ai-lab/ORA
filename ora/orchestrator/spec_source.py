"""RunPlan identity blocks: ``source`` and ``policy_identity``.

``source`` stays exactly ``{commit, dirty}``: launchers, schedulers and
collectors compare it whole across cells and recompute plan hashes from it.
What the plan's policy content is goes in a separate hashed block::

    policy_identity:
      schema: ora.run-plan-policy-identity.v1
      policy_lock_sha256: <sha256 of policy.lock bytes, or null>
      policy_lock_absent_reason: <text>   # only when policy_lock_sha256 is null
      model_visible_sha256: <model_visible.model_visible_sha256 of the plan>

Plans resolved before this block existed omit it and still validate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .model_visible import model_visible_sha256
from .policy_lock import LOCK_PATH, lock_sha256

POLICY_IDENTITY_SCHEMA = "ora.run-plan-policy-identity.v1"
ABSENT_LOCK_REASON = f"{LOCK_PATH} is absent from the resolving source tree"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEYS = frozenset({"schema", "policy_lock_sha256", "model_visible_sha256"})


def bind_policy_identity(portable: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """The ``policy_identity`` block of a plan whose other blocks are resolved."""

    lock = lock_sha256(root)
    identity: dict[str, Any] = {"schema": POLICY_IDENTITY_SCHEMA, "policy_lock_sha256": lock}
    if lock is None:
        identity["policy_lock_absent_reason"] = ABSENT_LOCK_REASON
    identity["model_visible_sha256"] = model_visible_sha256(portable, root=root)
    return identity


def valid_plan_identity(plan: Mapping[str, Any]) -> bool:
    """True when ``source`` is ``{commit, dirty}`` and any ``policy_identity`` is well formed."""

    source = plan.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"commit", "dirty"}
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
        or not isinstance(source.get("dirty"), bool)
    ):
        return False
    if "policy_identity" not in plan:
        return True
    identity = plan["policy_identity"]
    if not isinstance(identity, Mapping) or identity.get("schema") != POLICY_IDENTITY_SCHEMA:
        return False
    lock = identity.get("policy_lock_sha256")
    expected = _KEYS | ({"policy_lock_absent_reason"} if lock is None else set())
    if set(identity) != expected or not _hex(identity.get("model_visible_sha256")):
        return False
    if lock is None:
        reason = identity.get("policy_lock_absent_reason")
        return isinstance(reason, str) and bool(reason)
    return _hex(lock)


def _hex(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
