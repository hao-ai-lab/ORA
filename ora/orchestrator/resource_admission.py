"""Resource-class coexistence and measured GPU-memory admission."""
from __future__ import annotations
from typing import Any, Mapping
from .resources import ExecutionResourceError, resolve_execution_resource, resource_lease_key


def _visible_gpu_ids(value: Mapping[str, Any]) -> set[str]:
    environment = value.get("worker_environment")
    raw = (
        environment.get("CUDA_VISIBLE_DEVICES")
        if isinstance(environment, Mapping) else None
    )
    if not isinstance(raw, str):
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _lease_key(value: Mapping[str, Any]) -> str:
    """Return the registry key of one binding or lease record."""

    return f"{value.get('resource_class')}--{value.get('slot_id')}"


def _memory_floor_mib(value: Mapping[str, Any]) -> int | None:
    """Return the declared free-memory floor of one binding or lease record.

    Lease records written before this field existed carry no floor of their
    own, so the host profile they name is re-resolved instead. That fallback
    is only trusted when the record's ``profile_sha256`` still matches what
    this source resolves: a checkout that repriced the profile would otherwise
    silently reprice a live lease held by another checkout. A record whose
    profile no longer resolves — renamed, retired, repriced, or registered
    only by the source commit that wrote it — has no trustworthy floor and is
    reported as unknown rather than silently priced at zero.
    """

    declared = value.get("minimum_free_memory_mib")
    if (
        not isinstance(declared, bool)
        and isinstance(declared, int)
        and declared > 0
    ):
        return declared
    profile_id = value.get("profile_id")
    runtime_kind = value.get("runtime_kind")
    if not isinstance(profile_id, str) or not isinstance(runtime_kind, str):
        return None
    host_id = value.get("host_id")
    try:
        resolved = resolve_execution_resource(
            profile_id, runtime_kind=runtime_kind,
            required_host=host_id if isinstance(host_id, str) else None,
        )
    except ExecutionResourceError:
        return None
    recorded_sha256 = value.get("profile_sha256")
    if (
        isinstance(recorded_sha256, str)
        and recorded_sha256 != resolved.get("profile_sha256")
    ):
        return None
    floor = resolved.get("minimum_free_memory_mib")
    if not isinstance(floor, bool) and isinstance(floor, int) and floor > 0:
        return floor
    return None


def _lease_refusal(rule: str, lease: Mapping[str, Any]) -> str:
    if lease.get("schema") == "ora.legacy-active-run-lease.v1":
        holder = "legacy active-run lease"
    else:
        holder = f"{lease.get('resource_class')}/{lease.get('slot_id')}"
    return (
        "execution resource is already leased: "
        f"{lease.get('run_id')} ({holder}); rule={rule}"
    )


def _resource_refusal(
    binding: Mapping[str, Any], active: list[Mapping[str, Any]], *,
    inventory: Mapping[str, Mapping[str, Any]] | None = None,
    probe_error: str | None = None,
) -> str | None:
    """Return why this binding cannot be claimed now, or None to admit it.

    Exclusivity is decided first and is unchanged: a legacy lease record, an
    ``exclusive_host`` plan on either side, and a second claim on one
    ``resource_class``/``slot_id`` each refuse outright, so no slot is ever
    double-claimed and unprofiled or physical plans stay host-exclusive.

    Everything after that is capacity, not class (operator ruling 2026-09-08:
    a single GPU must not be reserved for one experiment). A claim on a device
    that no active lease shares keeps its previous admission path exactly —
    the per-slot lease plus preflight's own free-memory floor — and never
    touches the probe. A claim that would be **co-resident with any active
    lease** is priced, whatever the two classes are, because a same-class
    neighbour consumes the same memory a cross-class one does:

        sum(declared floor of every lease already on the device)
        + this binding's declared floor
        <= free_mib + sum(min(declared floor, memory that lease's own
                              processes currently hold))

    The right-hand side is what the device can still give to leased work. Free
    memory is admissible outright; memory our own priced leases already hold is
    admissible again because their declared floors are charged on the left, and
    crediting it back at most their floor keeps a worker that overshoots from
    buying capacity it does not have. Everything else in use — a foreign job, a
    resident service holding no lease, driver reserve — is charged in full,
    which is exactly what the previous ``held + floor <= total_mib`` arithmetic
    could not see.

    Preflight's own rule, ``free_mib >= floor``, needs no separate branch here:
    every credit is capped at that lease's floor, so the credited sum never
    exceeds the held sum, and the inequality above therefore already implies
    it. Preflight still applies it independently for the claims this rule does
    not price.

    A missing, failing or unperformed device probe refuses a co-resident claim
    rather than admitting it blind: without the live reading the Harness cannot
    see a foreign process already holding the card, and preflight fails a GPU
    run under the same condition anyway. A neighbouring lease this source
    cannot price refuses for the same reason, and a device this source cannot
    identify as an nvidia-smi index (a MIG or UUID form) is refused rather than
    guessed.
    """

    resource_class = str(binding["resource_class"])
    key = resource_lease_key(binding)
    for lease in active:
        if lease.get("schema") == "ora.legacy-active-run-lease.v1":
            return _lease_refusal("legacy_lease", lease)
        lease_class = str(lease.get("resource_class"))
        if resource_class == "exclusive_host" or lease_class == "exclusive_host":
            return _lease_refusal("exclusive_host", lease)
        if _lease_key(lease) == key:
            return _lease_refusal("slot_lease", lease)
    devices = _visible_gpu_ids(binding)
    if not devices:
        return None
    co_resident = [
        lease for lease in active if _visible_gpu_ids(lease) & devices
    ]
    if not co_resident:
        return None
    unindexed = sorted(item for item in devices if not item.isdigit())
    if unindexed:
        return (
            "co-residency refused: CUDA_VISIBLE_DEVICES names "
            f"{', '.join(unindexed)}, which is not an nvidia-smi device index; "
            "rule=device_identity_unsupported"
        )
    floor = _memory_floor_mib(binding)
    if floor is None:
        return (
            "co-residency refused: "
            f"{binding.get('profile_id')} declares no usable "
            "minimum_free_memory_mib; rule=missing_declared_floor"
        )
    if inventory is None:
        return (
            "co-residency refused: cannot read GPU memory "
            f"({probe_error or 'no claim-time device probe was performed'}); "
            "rule=device_probe_unavailable"
        )
    for device in sorted(devices, key=int):
        entry = inventory.get(device)
        if entry is None:
            return (
                f"co-residency refused: GPU {device} is not visible "
                "to nvidia-smi; rule=device_probe_unavailable"
            )
        leased_mib = entry.get("leased_mib") or {}
        held = 0
        credited = 0
        holders = 0
        for lease in co_resident:
            if device not in _visible_gpu_ids(lease):
                continue
            lease_floor = _memory_floor_mib(lease)
            if lease_floor is None:
                return (
                    f"co-residency refused: GPU {device} holds lease "
                    f"{lease.get('run_id')} "
                    f"({lease.get('resource_class')}/{lease.get('slot_id')}, "
                    f"profile {lease.get('profile_id')}) whose declared floor "
                    "cannot be resolved; rule=unpriced_active_lease"
                )
            held += lease_floor
            credited += min(
                lease_floor, int(leased_mib.get(_lease_key(lease), 0)),
            )
            holders += 1
        admissible = int(entry["free_mib"]) + credited
        if held + floor > admissible:
            return (
                f"execution device is full: GPU {device} has "
                f"{entry['free_mib']} MiB free plus {credited} MiB already "
                f"held by its own leases ({admissible} MiB admissible; "
                f"{entry['total_mib']} MiB total, the rest is in unattributed "
                f"use); {holders} active lease(s) declare {held} MiB and "
                f"{binding.get('profile_id')} needs {floor} MiB more "
                f"({held + floor} MiB required); rule=device_capacity"
            )
    return None
