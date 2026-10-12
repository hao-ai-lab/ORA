"""Host-side reviewed public completion layer binding; private records never enter policy context."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REGISTRY_PATH = "ora/policy/environments/robodojo/task-contracts/registry-v8.json"
REGISTRY_SCHEMA = "ora.robodojo-task-contract-registry.v1"


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError("RoboDojo contract source is missing or escapes its root")
    return path


def resolve_reviewed_contract(
    runtime: Mapping[str, Any], *, project_root: Path, disclosure: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Host-safe source/layout binding used by the shared RunSpec guard.

    Pinned revision/layout checks require no simulator install. Fresh preflight
    independently verifies the exact evaluator and asset dependency bytes.
    """
    disclosure = dict(disclosure, registry=REGISTRY_PATH)
    task = runtime.get("task_variant")
    if not isinstance(task, str) or not task:
        disclosure["reason"] = "runtime_task_identity_unavailable"
        return None, disclosure
    registry = json.loads(_source_path(project_root, REGISTRY_PATH).read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("invalid RoboDojo task contract registry")
    entry = registry["tasks"].get(task)
    disclosure["task"] = task
    scheduled = runtime.get("layout_schedule") is not None
    if entry is None and scheduled:
        raise RuntimeError("a layout-scheduled RoboDojo task needs a reviewed contract and its layout appendix")
    if entry is None:
        disclosure["reason"] = "task_absent_from_verified_contract_registry"
        return None, disclosure
    if not isinstance(entry, Mapping) or set(entry) != {"record", "layer"}:
        # The registry names record and layer by path; policy.lock carries their
        # integrity and the record carries the reviewed layer-text hash.
        raise RuntimeError(f"RoboDojo registry entry for {task} must be exactly {{record, layer}} paths")
    from .locked_assets import locked_asset
    record_sha256 = locked_asset(project_root, entry["record"], label="RoboDojo contract record",
                                 error=RuntimeError).sha256
    record = load_task_contract(task, bindings={task: entry}, project_root=project_root)["record"]
    source, layout = record["source"], record["layout"]
    expected = {"required_revision": source["robodojo_revision"],
                "required_asset_revision": layout["assets_revision"],
                "layout_file": layout["layout_file"], "layout_sha256": layout["layout_sha256"],
                "eval_seed": layout["eval_seed"], "layout_id": layout["ordinal"]}
    compatibility = portability = None
    if scheduled:
        # The record reviews one layout; its hash-bound appendix extends the
        # same contract text to every layout the schedule may select.
        from .robodojo_layout_compatibility import verify_scheduled_contract_layouts
        compatibility = verify_scheduled_contract_layouts(
            runtime, record_path=entry["record"], project_root=project_root)
        for field in ("layout_file", "layout_sha256", "layout_id"):
            expected.pop(field)
    elif runtime.get("layout_id", layout["ordinal"]) != layout["ordinal"]:
        # One episode on another official layout: the reviewed portability
        # table must list this record on exactly that layout's bytes.
        from .robodojo_layout_compatibility import verify_single_layout_contract
        portability = verify_single_layout_contract(
            runtime, record_path=entry["record"], record=record, project_root=project_root,
            registry=REGISTRY_PATH)
        for field in ("layout_file", "layout_sha256", "layout_id"):
            expected.pop(field)
    for field, value in expected.items():
        if runtime.get(field) != value:
            raise RuntimeError(f"RoboDojo reviewed contract differs from runtime {field}")
    # A layout-readback instruction is generated from that layout; the lane
    # gate binds it and the worker refuses another, so only a fixed one is
    # compared with the record here.
    readback = portability is not None and portability["instruction_source"] == "layout_readback"
    if not readback and runtime.get("expected_task_language") not in (None, record["instruction"]["text"]):
        raise RuntimeError("RoboDojo reviewed contract task instruction differs")
    disclosure.update(delivered=True, reason="reviewed_contract_matches_the_pinned_source",
        runtime_revision=source["robodojo_revision"],
        contract_sha256=record["contract_layer"]["sha256"],
        check_success_sha256=source.get("run_reward_sha256", source["file_sha256"]),
        evaluator_file_sha256=source["file_sha256"],
        record=entry["record"], record_sha256=record_sha256,
        layout_sha256=layout["layout_sha256"],
        clause_count=len(record.get("evaluator_clauses", record["semantic_contract"])))
    if compatibility is not None:
        disclosure["layout_compatibility"] = compatibility
    if portability is not None:
        disclosure["layout_portability"] = portability
    return entry["layer"], disclosure


def verify_contract_dependencies(
    disclosure: Mapping[str, Any], *, project_root: Path, source_root: Path, asset_root: Path,
) -> dict[str, Any]:
    """Fresh, read-only preflight over evaluator/asset bytes; never policy text."""
    if disclosure.get("backend") != "robodojo" or not disclosure.get("delivered"):
        return {"applicable": False}
    path = _source_path(project_root, str(disclosure["record"]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != disclosure["record_sha256"]:
        raise RuntimeError("RoboDojo preflight contract record drift")
    record = json.loads(path.read_text())
    # A layout-scheduled plan adds the evaluator assets its scheduled layouts
    # read (each bottle model's metadata, for example); they fail closed too.
    dependencies = [{"root": "robodojo", "path": record["source"]["source_path"],
                     "sha256": record["source"]["file_sha256"]},
                    *record.get("evaluator_dependencies", []),
                    *(disclosure.get("layout_compatibility") or {}).get("evaluator_dependencies", [])]
    roots = {"robodojo": source_root, "assets": asset_root}
    for dependency in dependencies:
        file = _source_path(roots[dependency["root"]], dependency["path"])
        if hashlib.sha256(file.read_bytes()).hexdigest() != dependency["sha256"]:
            raise RuntimeError(f"RoboDojo reviewed evaluator dependency drift: {dependency['path']}")
    return {"applicable": True, "record_sha256": disclosure["record_sha256"],
            "checked_dependencies": len(dependencies)}

def load_task_contract(task: str, *, bindings: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Resolve one reviewed RoboDojo completion contract, fail-closed.

    The treatment's whole claim is that the policy received exactly the
    reviewed restatement of that task's official criteria, so a RoboDojo cell
    that reaches ``compose`` without a hash-matched record and a layer that
    renders it must not launch. A task with no reviewed record refuses here
    rather than silently composing a bare-instruction cell into a cohort whose
    other cells carry a contract.
    """

    binding = bindings.get(task)
    if not isinstance(binding, Mapping):
        raise RuntimeError(
            f"RoboDojo task has no reviewed completion contract: {task}"
        )
    record_path = project_root / str(binding["record"])
    layer_path = project_root / str(binding["layer"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != "ora.robodojo-task-contract.v1"
        or record.get("task") != task
    ):
        raise RuntimeError(
            f"RoboDojo contract record is not this task's record: {record_path}"
        )
    bound = record.get("contract_layer")
    layer_bytes = layer_path.read_bytes()
    if (
        not isinstance(bound, Mapping)
        or bound.get("path") != str(binding["layer"])
        or bound.get("sha256") != hashlib.sha256(layer_bytes).hexdigest()
    ):
        raise RuntimeError(
            f"RoboDojo contract layer changed after review: {layer_path}"
        )
    # The delivered bullets are hard-wrapped, so compare on collapsed
    # whitespace rather than on the wrapped bytes.
    text = layer_bytes.decode("utf-8")
    bullets, current = [], None
    for line in text.splitlines():
        if line.startswith("- "):
            if current is not None:
                bullets.append(" ".join(current.split()))
            current = line[2:]
        elif current is not None and line.strip():
            current += " " + line.strip()
        elif current is not None:
            bullets.append(" ".join(current.split()))
            current = None
    if current is not None:
        bullets.append(" ".join(current.split()))
    if bullets != [
        " ".join(str(line).split()) for line in record["semantic_contract"]
    ]:
        raise RuntimeError(
            f"RoboDojo contract layer does not render its record: {layer_path}"
        )
    return {"record": record, "layer": str(binding["layer"])}
