"""Rebuildable seed-0 LIBERO/LIBERO-Pro gallery previews.

Preview compilation is an explicit offline operator action.  The Viewer only
reads the resulting cache and never imports or contacts a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.environment.benchmarks.libero_pro import (
    LIBERO_PRO_CODE_REVISION,
    LIBERO_PRO_DATA_REVISION,
    ROBUSTNESS_SCHEMA,
)
from ora.orchestrator.paths import OUTPUTS_ROOT
from viewer.libero_pro_catalog import (
    ROBUSTNESS_CASES,
    TASKS,
)


PREVIEW_SCHEMA = "ora.libero-pro-preview-cache.v2"
# Keep the specimen identity stable for the original 480 images.  The v2
# manifest adds 90 LIBERO-90 base scenes; it does not redefine an existing
# source/render tuple.
PREVIEW_IDENTITY_SCHEMA = "ora.libero-pro-preview-cache.v1"
PREVIEW_ROOT = OUTPUTS_ROOT / "viewer-derived/libero-pro-previews"
MANIFEST_PATH = PREVIEW_ROOT / "manifest.json"
RENDER_CONFIG = {
    "camera": "agentview",
    "width": 256,
    "height": 256,
    "seed": 0,
    "robots": ["Panda"],
    "controller": "OSC_POSE",
    "action_count": 0,
    "robustness_adapter": ROBUSTNESS_SCHEMA,
}
EXPECTED_BASE_PREVIEWS = 130
EXPECTED_CORE_PREVIEWS = 160
EXPECTED_ROBUSTNESS_PREVIEWS = 280
EXPECTED_PREVIEWS = (
    EXPECTED_BASE_PREVIEWS + EXPECTED_CORE_PREVIEWS
    + EXPECTED_ROBUSTNESS_PREVIEWS
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        check=False, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _language(path: Path) -> str:
    match = re.search(
        r"\(:language\s+([^\r\n)]*?)\s*\)",
        path.read_text(encoding="utf-8"), re.IGNORECASE,
    )
    if match is None or not match.group(1).strip():
        raise RuntimeError(f"BDDL has no language field: {path}")
    return match.group(1).strip()


def preview_descriptors(
    *, libero_pro_root: Path, data_root: Path,
) -> list[dict[str, Any]]:
    """Resolve all 570 base, core and robustness seed-0 preview inputs."""

    code_bddl_root = libero_pro_root / "libero/libero/bddl_files"
    code_init_root = libero_pro_root / "libero/libero/init_files"
    data_bddl_root = data_root / "bddl_files"
    data_init_root = data_root / "init_files"
    descriptors: list[dict[str, Any]] = []
    for task in TASKS:
        runtime_suite = task.base_suite
        base_bddl = code_bddl_root / runtime_suite / f"{task.name}.bddl"
        if task.pro_anchor:
            base_init = (
                data_init_root / runtime_suite / f"{task.name}.pruned_init"
            )
            base_init_source = (
                f"libero-pro-data:init_files/{runtime_suite}/"
                f"{task.name}.pruned_init"
            )
        else:
            base_init = code_init_root / runtime_suite / f"{task.name}.pruned_init"
            base_init_source = (
                f"libero-pro-code:libero/libero/init_files/{runtime_suite}/"
                f"{task.name}.pruned_init"
            )
        descriptors.append({
            "key": f"base:{runtime_suite}:task-{task.task_id:02d}",
            "layer": "base",
            "family": task.family,
            "suite": runtime_suite,
            "task_id": task.task_id,
            "task_name": task.name,
            "perturbation": None,
            "instruction": _language(base_bddl),
            "bddl_path": base_bddl,
            "bddl_source": (
                f"libero-pro-code:libero/libero/bddl_files/{runtime_suite}/"
                f"{task.name}.bddl"
            ),
            "init_path": base_init,
            "init_source": base_init_source,
        })
        for variant in task.core_variants:
            bddl = data_bddl_root / variant.suite / f"{task.name}.bddl"
            init = data_init_root / variant.suite / f"{task.name}.pruned_init"
            actual_language = _language(bddl)
            if actual_language != variant.instruction:
                raise RuntimeError(
                    f"catalog language drift for {variant.suite}/task-{task.task_id:02d}"
                )
            descriptors.append({
                "key": f"core:{variant.suite}:task-{task.task_id:02d}",
                "layer": "core",
                "family": task.family,
                "suite": variant.suite,
                "task_id": task.task_id,
                "task_name": task.name,
                "perturbation": variant.perturbation,
                "instruction": actual_language,
                "bddl_path": bddl,
                "bddl_source": (
                    f"libero-pro-data:bddl_files/{variant.suite}/"
                    f"{task.name}.bddl"
                ),
                "init_path": init,
                "init_source": (
                    f"libero-pro-data:init_files/{variant.suite}/"
                    f"{task.name}.pruned_init"
                ),
            })
        for case in ROBUSTNESS_CASES if task.pro_anchor else ():
            case_name = str(case["case"])
            experiment_config = f"{runtime_suite}_robust_{case_name}"
            bddl = (
                data_bddl_root / str(case["id"]) / "bddl" / runtime_suite
                / f"{task.name}.bddl"
            )
            actual_language = _language(bddl)
            if actual_language.casefold() != task.canonical_instruction.casefold():
                raise RuntimeError(
                    f"robustness language drift for {experiment_config}/"
                    f"task-{task.task_id:02d}"
                )
            descriptors.append({
                "key": (
                    f"robustness:{experiment_config}:task-{task.task_id:02d}"
                ),
                "layer": "robustness",
                "family": task.family,
                "suite": experiment_config,
                "runtime_suite": runtime_suite,
                "task_id": task.task_id,
                "task_name": task.name,
                "perturbation": case_name,
                "robustness_case": case_name,
                "instruction": actual_language,
                "bddl_path": bddl,
                "bddl_source": (
                    f"libero-pro-data:bddl_files/{case['id']}/bddl/"
                    f"{runtime_suite}/{task.name}.bddl"
                ),
                "init_path": base_init,
                "init_source": (
                    f"libero-pro-data:init_files/{runtime_suite}/"
                    f"{task.name}.pruned_init"
                ),
            })
    if len(descriptors) != EXPECTED_PREVIEWS:
        raise RuntimeError("LIBERO preview plan must contain exactly 570 records")
    if len({item["key"] for item in descriptors}) != EXPECTED_PREVIEWS:
        raise RuntimeError("LIBERO preview plan contains duplicate identities")
    return descriptors


def _resolved_specs(
    *, descriptors: Iterable[Mapping[str, Any]], output_root: Path,
    old_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs = []
    for descriptor in descriptors:
        bddl = Path(descriptor["bddl_path"]).resolve()
        init = Path(descriptor["init_path"]).resolve()
        if not bddl.is_file() or not init.is_file() or init.stat().st_size == 0:
            raise RuntimeError(f"preview source is unavailable: {bddl} / {init}")
        bddl_sha = _sha256(bddl)
        init_sha = _sha256(init)
        identity = _json_hash({
            "schema": PREVIEW_IDENTITY_SCHEMA,
            "render": RENDER_CONFIG,
            "code_revision": LIBERO_PRO_CODE_REVISION,
            "data_revision": LIBERO_PRO_DATA_REVISION,
            "key": descriptor["key"],
            "bddl_sha256": bddl_sha,
            "init_sha256": init_sha,
        })
        relative = (
            Path("sets") / identity[:16] / str(descriptor["layer"])
            / str(descriptor["suite"])
            / f"task-{int(descriptor['task_id']):02d}-seed-0.png"
        )
        old = old_records.get(str(descriptor["key"]), {})
        specs.append({
            **descriptor,
            "bddl_path": str(bddl),
            "init_path": str(init),
            "bddl_sha256": bddl_sha,
            "init_sha256": init_sha,
            "preview_identity_sha256": identity,
            "image": relative.as_posix(),
            "output_path": str((output_root / relative).resolve()),
            "expected_image_sha256": (
                old.get("image_sha256")
                if old.get("preview_identity_sha256") == identity else None
            ),
        })
    return specs


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_preview_manifest(root: Path = PREVIEW_ROOT) -> dict[str, Any]:
    """Load and validate the presentation cache without contacting LIBERO."""

    root = root.resolve()
    manifest = _load_json(root / "manifest.json")
    records = manifest.get("records")
    source = manifest.get("source")
    valid_identity = (
        manifest.get("schema") == PREVIEW_SCHEMA
        and isinstance(source, Mapping)
        and source.get("libero_pro_code_revision") == LIBERO_PRO_CODE_REVISION
        and source.get("libero_pro_data_revision") == LIBERO_PRO_DATA_REVISION
        and manifest.get("render") == RENDER_CONFIG
        and isinstance(records, list)
        and len(records) == EXPECTED_PREVIEWS
    )
    index: dict[str, dict[str, Any]] = {}
    missing = []
    if valid_identity:
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("key"), str):
                valid_identity = False
                break
            relative = Path(str(record.get("image", "")))
            path = (root / relative).resolve()
            if relative.is_absolute() or root not in path.parents:
                valid_identity = False
                break
            value = dict(record)
            if not path.is_file():
                missing.append(value["key"])
            index[value["key"]] = value
    ready = valid_identity and len(index) == EXPECTED_PREVIEWS and not missing
    return {
        "ready": ready,
        "manifest": manifest if valid_identity else {},
        "index": index if valid_identity else {},
        "missing": missing,
        "root": root,
        "expected": EXPECTED_PREVIEWS,
        "available": len(index) - len(missing) if valid_identity else 0,
    }


def _verify_source(root: Path, expected_revision: str, label: str) -> None:
    if not (root / ".git").exists():
        raise RuntimeError(f"{label} is not a Git checkout: {root}")
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_revision:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected_revision}, got {actual}"
        )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"{label} source is modified: {dirty}")


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    project = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        project, environment.get("PYTHONPATH"),
    )))
    return environment


def _compile_batch(
    *, python: Path, input_path: Path, result_path: Path,
) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            str(python), "-m",
            "ora.environment.backends.libero_pro.preview",
            str(input_path), str(result_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_worker_environment(), capture_output=True, text=True,
        check=False,
    )
    if result.returncode:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"LIBERO preview worker failed:\n{tail}")
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("LIBERO preview worker returned an invalid result")
    return value


def compile_previews(
    *, python: Path, libero_pro_root: Path, data_root: Path,
    output_root: Path = PREVIEW_ROOT, jobs: int = 4,
) -> dict[str, Any]:
    """Compile the complete 570-image seed-0 cache and publish one manifest."""

    # Preserve the virtual-environment entry path. Resolving its symlink would
    # execute the bare uv-managed interpreter without the environment's
    # site-packages.
    python = Path(os.path.abspath(os.path.expanduser(str(python))))
    libero_pro_root = libero_pro_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"invalid LIBERO-Pro Python: {python}")
    _verify_source(libero_pro_root, LIBERO_PRO_CODE_REVISION, "LIBERO-Pro code")
    _verify_source(data_root, LIBERO_PRO_DATA_REVISION, "LIBERO-Pro data")
    jobs = max(1, min(int(jobs), 8, EXPECTED_PREVIEWS))
    output_root.mkdir(parents=True, exist_ok=True)
    old = load_preview_manifest(output_root)
    # A valid v1 manifest is deliberately reusable during the one-way catalog
    # expansion.  The image identity remains v1, so its verified hashes let the
    # isolated renderer skip all 480 previously compiled specimens.
    previous_manifest = _load_json(output_root / "manifest.json")
    previous_records = previous_manifest.get("records", [])
    old_records = {
        item["key"]: item
        for item in previous_records
        if isinstance(item, Mapping) and isinstance(item.get("key"), str)
    }
    descriptors = preview_descriptors(
        libero_pro_root=libero_pro_root, data_root=data_root,
    )
    specs = _resolved_specs(
        descriptors=descriptors, output_root=output_root,
        old_records=old_records,
    )
    catalog_identity = _json_hash({
        "schema": PREVIEW_SCHEMA,
        "source": [LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION],
        "render": RENDER_CONFIG,
        "previews": [item["preview_identity_sha256"] for item in specs],
    })
    if (
        old.get("ready")
        and old.get("manifest", {}).get("catalog_identity_sha256") == catalog_identity
    ):
        print(f"LIBERO preview cache already complete: {output_root}")
        return old["manifest"]

    chunks = [specs[index::jobs] for index in range(jobs)]
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output_root) as build_name:
        build = Path(build_name)
        futures = {}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for index, chunk in enumerate(chunks):
                input_path = build / f"worker-{index:02d}-input.json"
                result_path = build / f"worker-{index:02d}-result.json"
                input_path.write_text(json.dumps({
                    "libero_pro_root": str(libero_pro_root),
                    "data_root": str(data_root),
                    "output_root": str(output_root),
                    "render": RENDER_CONFIG,
                    "specs": chunk,
                }), encoding="utf-8")
                future = pool.submit(
                    _compile_batch, python=python, input_path=input_path,
                    result_path=result_path,
                )
                futures[future] = (index, len(chunk))
            completed = 0
            for future in as_completed(futures):
                index, size = futures[future]
                batch = future.result()
                if len(batch) != size:
                    raise RuntimeError(f"preview worker {index} returned {len(batch)}/{size}")
                records.extend(batch)
                completed += size
                print(f"LIBERO previews: {completed}/{EXPECTED_PREVIEWS}", flush=True)

    records.sort(key=lambda item: item["key"])
    if len(records) != EXPECTED_PREVIEWS or len({item["key"] for item in records}) != EXPECTED_PREVIEWS:
        raise RuntimeError("compiled LIBERO preview set is incomplete")
    manifest = {
        "schema": PREVIEW_SCHEMA,
        "catalog_identity_sha256": catalog_identity,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "libero_pro_code_revision": LIBERO_PRO_CODE_REVISION,
            "libero_pro_data_revision": LIBERO_PRO_DATA_REVISION,
        },
        "render": RENDER_CONFIG,
        "counts": {
            "base": EXPECTED_BASE_PREVIEWS,
            "core": EXPECTED_CORE_PREVIEWS,
            "robustness_extension": EXPECTED_ROBUSTNESS_PREVIEWS,
            "total": EXPECTED_PREVIEWS,
        },
        "evidence_boundary": (
            "viewer-derived zero-action initial observations; not demonstrations, "
            "policy rollouts, evaluations, or sealed run evidence"
        ),
        "records": records,
    }
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root / "manifest.json")
    print(f"LIBERO preview cache complete: {output_root}")
    return manifest


def _required_path(value: str | None, environment_name: str) -> Path:
    raw = value or os.environ.get(environment_name)
    if not raw:
        raise RuntimeError(f"missing --{environment_name.lower().replace('_', '-')} or {environment_name}")
    return Path(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="external LIBERO_PRO_PYTHON")
    parser.add_argument("--libero-pro-root", help="pinned LIBERO_PRO_ROOT")
    parser.add_argument("--data-root", help="pinned LIBERO_PRO_DATA_ROOT")
    parser.add_argument("--output-root", type=Path, default=PREVIEW_ROOT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.status:
            status = load_preview_manifest(args.output_root)
            print(json.dumps({
                key: status[key] for key in ("ready", "available", "expected", "missing")
            }, indent=2, sort_keys=True))
            return 0 if status["ready"] else 1
        compile_previews(
            python=_required_path(args.python, "LIBERO_PRO_PYTHON"),
            libero_pro_root=_required_path(args.libero_pro_root, "LIBERO_PRO_ROOT"),
            data_root=_required_path(args.data_root, "LIBERO_PRO_DATA_ROOT"),
            output_root=args.output_root, jobs=args.jobs,
        )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"compile-libero-previews: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
