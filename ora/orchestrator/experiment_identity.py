"""Source-bound experiment families and lineage; never a launch or run registry."""
from __future__ import annotations

import copy
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.paths import PROJECT_ROOT

SCHEMA = "ora.experiment-identity.v1"
CATALOG_SCHEMA = "ora.experiment-lineage.v1"
CATALOG_PATH = "ora/orchestrator/identity-registry.json"
KEY = re.compile(r"EXP[0-9]{2,}[A-Z][0-9]*/[a-z][a-z0-9-]*/g(?:[0-9]{3}|[1-9][0-9]{3,})")


def validate_identity(value: Any) -> dict[str, Any]:
    """Validate a self-contained identity without consulting today's catalog."""
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "family", "branch", "generation", "parents"
    } or value.get("schema") != SCHEMA:
        raise ValueError("invalid experiment_identity schema/fields")
    key = identity_key(value)
    if not KEY.fullmatch(key) or int(value["generation"][1:]) < 1:
        raise ValueError("experiment_identity must name EXP family/branch/gNNN")
    parents = value["parents"]
    if (not isinstance(parents, list) or any(
        not isinstance(p, str) or not KEY.fullmatch(p) or p.endswith("/g000") or p == key for p in parents
    ) or len(set(parents)) != len(parents)):
        raise ValueError("experiment_identity parents must be unique other generation IDs")
    return copy.deepcopy(dict(value))


def identity_key(value: Mapping[str, Any]) -> str:
    return "/".join(str(value.get(k, "")) for k in ("family", "branch", "generation"))


def load_catalog(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    value = json.loads((root / CATALOG_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CATALOG_SCHEMA:
        raise ValueError("invalid experiment lineage catalog schema")
    if (any(not isinstance(value.get(k), dict) for k in ("families", "generations", "namespaces"))
            or not isinstance(value.get("legacy_aliases"), list)):
        raise ValueError("invalid experiment lineage catalog collections")
    families = value["families"]
    nodes = value["generations"]
    for family, entry in families.items():
        if (not re.fullmatch(r"EXP[0-9]{2,}[A-Z][0-9]*", family)
                or not isinstance(entry, dict)
                or any(not isinstance(entry.get(k), str) or not entry[k] for k in ("title", "description"))):
            raise ValueError("invalid experiment family")
    for key, node in nodes.items():
        if (not isinstance(node, dict) or node.get("status") not in {"development", "historical"}
                or not isinstance(node.get("label"), str) or not node["label"]):
            raise ValueError("invalid generation metadata")
        identity = validate_identity(node.get("identity"))
        if key != identity_key(identity) or identity["family"] not in families:
            raise ValueError("generation key/family mismatch")
        if any(p not in nodes for p in identity["parents"]):
            raise ValueError("unknown parent generation")
    remaining = {k: len(v["identity"]["parents"]) for k, v in nodes.items()}
    children: dict[str, list[str]] = {k: [] for k in nodes}
    for key, node in nodes.items():
        for parent in node["identity"]["parents"]:
            children[parent].append(key)
    ready = deque(k for k, count in remaining.items() if count == 0)
    visited = 0
    while ready:
        key = ready.popleft()
        visited += 1
        for child in children[key]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("experiment lineage cycle")
    for namespace, family in value["namespaces"].items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", namespace) or family not in families:
            raise ValueError("invalid experiment namespace/family")
    seen: set[tuple[str, str, str]] = set()
    for alias in value["legacy_aliases"]:
        if (not isinstance(alias, dict) or set(alias) != {"experiment", "cohort_id", "source_commit", "generation_id"}
                or any(not isinstance(v, str) for v in alias.values())):
            raise ValueError("invalid legacy alias fields")
        selector = (alias["experiment"], alias["cohort_id"], alias["source_commit"])
        key = alias["generation_id"]
        if (selector in seen or key not in nodes
                or not re.fullmatch(r"[0-9a-f]{40}", selector[2])
                or value["namespaces"].get(selector[0], nodes[key]["identity"]["family"]) != nodes[key]["identity"]["family"]):
            raise ValueError("ambiguous or invalid legacy experiment alias")
        seen.add(selector)
    _validate_reviewed_migrations(value)
    return value


MIGRATION_FIELDS = frozenset({
    "experiment", "cohort_id", "source_commit", "recorded_generation_id", "generation_id", "decided", "review"})


def _validate_reviewed_migrations(catalog: Mapping[str, Any]) -> None:
    """Reviewed relabels of exact recorded runs; optional, and never a binding for new plans.

    The recorded generation must belong to the namespace's family, so a
    migration can only move what that namespace actually recorded.
    """
    migrations = catalog.get("reviewed_migrations", [])
    if not isinstance(migrations, list):
        raise ValueError("invalid reviewed migration collection")
    nodes = catalog["generations"]
    seen: set[tuple[str, str, str, str]] = set()
    for entry in migrations:
        if (not isinstance(entry, dict) or set(entry) != MIGRATION_FIELDS
                or any(not isinstance(v, str) for v in entry.values())):
            raise ValueError("invalid reviewed migration fields")
        selector = (entry["experiment"], entry["cohort_id"], entry["source_commit"], entry["recorded_generation_id"])
        recorded, target = entry["recorded_generation_id"], entry["generation_id"]
        if (selector in seen or recorded not in nodes or target not in nodes or recorded == target
                or not re.fullmatch(r"[0-9a-f]{40}", entry["source_commit"])
                or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", entry["decided"]) or not entry["review"].strip()
                or catalog["namespaces"].get(entry["experiment"], nodes[recorded]["identity"]["family"])
                != nodes[recorded]["identity"]["family"]):
            raise ValueError("ambiguous or invalid reviewed experiment migration")
        seen.add(selector)


def bind_identity(spec: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> dict | None:
    """Bind new plans from explicit RunSpec or source experiment declaration.

    Historical aliases are deliberately read-only and cannot select new treatment.
    """
    import yaml

    namespace = str(spec.get("experiment", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", namespace):
        raise ValueError("invalid experiment execution namespace")
    declaration = root / "experiments" / namespace.replace("-", "_") / "experiment.yaml"
    declared = None
    if declaration.is_file():
        document = yaml.safe_load(declaration.read_text())
        if isinstance(document, Mapping) and document.get("experiment") == namespace:
            declared = document.get("experiment_identity")
    explicit = spec.get("experiment_identity")
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError("RunSpec experiment_identity conflicts with source declaration")
    selected = explicit if explicit is not None else declared
    if selected is None:
        return None
    identity = validate_identity(selected)
    catalog = load_catalog(root)
    node = catalog["generations"].get(identity_key(identity))
    if not node or node["identity"] != identity:
        raise ValueError("experiment_identity is not registered at this source")
    if catalog["namespaces"].get(namespace) != identity["family"]:
        raise ValueError("experiment namespace is not registered to this family")
    return identity


def project_identity(plan: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Project archival labels without modifying plans or inventing unknown ancestry.

    A reviewed migration relabels only the exact (experiment, cohort, source,
    recorded generation) it names and keeps the recorded projection beside it.
    """
    recorded = _recorded_identity(plan, catalog)
    key = recorded.get("generation_id")
    if not key:
        return recorded
    if recorded["binding"] == "plan" and recorded["identity"] != catalog["generations"].get(key, {}).get("identity"):
        return recorded
    admission = plan.get("launch_admission")
    source = plan.get("source")
    admission = admission if isinstance(admission, Mapping) else {}
    source = source if isinstance(source, Mapping) else {}
    for entry in catalog.get("reviewed_migrations", []):
        if (entry["recorded_generation_id"] == key and entry["experiment"] == plan.get("experiment")
                and entry["cohort_id"] == admission.get("cohort_id", "")
                and entry["source_commit"] == source.get("commit")):
            target = catalog["generations"][entry["generation_id"]]["identity"]
            return {"binding": "reviewed_migration", "family": target["family"],
                    "generation_id": entry["generation_id"], "identity": copy.deepcopy(target),
                    "recorded": recorded}
    return recorded


def _recorded_identity(plan: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """The plan-bound identity, else the exact legacy alias, else the namespace family."""
    if "experiment_identity" in plan:
        try:
            identity = validate_identity(plan["experiment_identity"])
        except ValueError:
            return {"binding": "invalid", "family": None, "generation_id": None}
        return {"binding": "plan", "family": identity["family"],
                "generation_id": identity_key(identity), "identity": identity}
    namespace = plan.get("experiment")
    if not isinstance(namespace, str):
        return {"binding": "unmapped", "family": None, "generation_id": None}
    family = catalog["namespaces"].get(namespace)
    admission = plan.get("launch_admission")
    source = plan.get("source")
    admission = admission if isinstance(admission, Mapping) else {}
    source = source if isinstance(source, Mapping) else {}
    for alias in catalog["legacy_aliases"]:
        if (alias["experiment"] == namespace and alias["cohort_id"] == admission.get("cohort_id", "")
                and alias["source_commit"] == source.get("commit")):
            key = alias["generation_id"]
            return {"binding": "historical_alias", "family": catalog["generations"][key]["identity"]["family"], "generation_id": key,
                    "identity": copy.deepcopy(catalog["generations"][key]["identity"])}
    return {"binding": "family_only" if family else "unmapped",
            "family": family, "generation_id": None}


def command_experiments(args: Any, *, root: Path = PROJECT_ROOT) -> int:
    try:
        catalog = load_catalog(root)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    family = args.family
    if family and family not in catalog["families"]:
        raise RuntimeError(f"unknown experiment family: {family}")
    selected = {k: v for k, v in catalog["generations"].items()
                if not family or v["identity"]["family"] == family}
    if args.json:
        view = {"schema": "ora.experiment-lineage-view.v1", "family": family,
                "families": {k: v for k, v in catalog["families"].items() if not family or k == family},
                "generations": selected,
                "namespaces": {k: v for k, v in catalog["namespaces"].items() if not family or v == family},
                "legacy_aliases": [a for a in catalog["legacy_aliases"] if a["generation_id"] in selected],
                "reviewed_migrations": [m for m in catalog.get("reviewed_migrations", [])
                                        if m["generation_id"] in selected or m["recorded_generation_id"] in selected]}
        print(json.dumps(view, indent=2, ensure_ascii=False))
    else:
        for key, node in selected.items():
            parents = ",".join(node["identity"]["parents"]) or "root"
            print(f"{key}\t{node['status']}\tparents={parents}\t{node['label']}")
    return 0
