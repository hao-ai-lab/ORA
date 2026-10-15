"""Lossless post-seal archive of provider requests (agent/evidence/request_archive.py)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from ora.agent.evidence import request_archive
from ora.orchestrator import evaluation, metrics

pytestmark = pytest.mark.skipif(not request_archive.available(), reason="tar and zstd executables are required")


def _sealed_run(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """A run whose agent trace references two provider requests and one receipt, all sealed."""
    run = tmp_path / "runs" / "run-a"
    artifacts = run / "native" / "agent-artifacts"
    artifacts.mkdir(parents=True)
    image = "A" * 50_000  # the repeated inline image that makes requests large
    bodies = {
        "request-1": json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image}"}}]}]}).encode(),
        "request-2": json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image}"}}, {"type": "text", "text": "again"}]}]}).encode(),
        "receipt": b'{"receipt": true}',
    }
    files: dict[str, bytes] = {}
    for name, data in bodies.items():
        relative = f"native/agent-artifacts/{hashlib.sha256(data).hexdigest()}.json"
        (run / relative).write_bytes(data)
        files[name] = relative
    events = [
        {"type": "provider_request", "model_call_id": "m0", "artifact_ref": {
            "kind": "native_provider_wire_request", "path": files["request-1"]},
         "payload": {"canonical_request_ref": {"kind": "native_canonical_provider_request", "path": files["request-1"]}}},
        {"type": "provider_request", "model_call_id": "m1", "artifact_ref": {
            "kind": "native_provider_wire_request", "path": files["request-2"]}, "payload": {}},
        {"type": "environment_receipt", "artifact_ref": {"kind": "environment_receipt", "path": files["receipt"]}},
    ]
    (run / "agent-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    inventory = []
    for relative in sorted({*files.values(), "agent-events.jsonl"}):
        data = (run / relative).read_bytes()
        inventory.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        (run / relative).chmod(0o444)
    (run / "plan.json").write_text("{}")
    (run / "seal.json").write_text(json.dumps({"inventory": inventory}))
    return run, {relative: (run / relative).read_bytes() for relative in files.values()}


def test_sealed_requests_are_archived_losslessly_and_read_back(tmp_path: Path) -> None:
    run, originals = _sealed_run(tmp_path)
    requests = sorted(request_archive.request_paths(run))
    result = request_archive.compact(run)
    assert result["status"] == "archived" and result["members"] == 2
    assert all(not (run / path).exists() for path in requests)
    receipt = next(path for path in originals if path not in requests)
    assert (run / receipt).is_file()  # only provider requests are archived
    manifest = json.loads((run / request_archive.MANIFEST).read_text())
    assert {member["path"] for member in manifest["members"]} == set(requests)
    for path, data in originals.items():
        assert request_archive.read_bytes(run, path) == data
    seal = json.loads((run / "seal.json").read_text())
    for record in seal["inventory"]:
        if record["path"] in requests:
            assert request_archive.inventory_record_present(run, record)
    assert request_archive.compact(run)["status"] == "already_archived"


def test_export_restores_archived_inventory_byte_identical(tmp_path: Path) -> None:
    run, originals = _sealed_run(tmp_path)
    request_archive.compact(run)
    exported = evaluation.export(run, tmp_path / "export.tar.gz")
    with tarfile.open(exported) as archive:
        for path, data in originals.items():
            assert archive.extractfile(f"{run.name}/{path}").read() == data


def test_request_metrics_read_through_the_archive(tmp_path: Path) -> None:
    run, _originals = _sealed_run(tmp_path)
    events = [json.loads(line) for line in (run / "agent-events.jsonl").read_text().splitlines()]
    before = metrics._native_request_inventory(run, events)
    request_archive.compact(run)
    assert metrics._native_request_inventory(run, events) == before


def test_the_operator_all_json_format_converts_to_requests_only(tmp_path: Path) -> None:
    run, originals = _sealed_run(tmp_path)
    artifacts = run / "native" / "agent-artifacts"
    names = sorted(path.name for path in artifacts.iterdir())
    manifest = {"files": [{"name": name, "bytes": (artifacts / name).stat().st_size,
                           "sha256": hashlib.sha256((artifacts / name).read_bytes()).hexdigest()} for name in names]}
    subprocess.run(f"tar -C '{artifacts}' -cf - {' '.join(names)} | zstd -q --long=31 -o "
                   f"'{run / request_archive.LEGACY_ARCHIVE}'", shell=True, check=True)
    (run / request_archive.LEGACY_MANIFEST).write_text(json.dumps(manifest))
    for name in names:
        (artifacts / name).unlink()
    for path, data in originals.items():  # the legacy format is readable as it is
        assert request_archive.read_bytes(run, path) == data
    result = request_archive.convert_legacy(run)
    assert result["status"] == "restored" and result["compact"]["status"] == "archived"
    assert not (run / request_archive.LEGACY_MANIFEST).exists()
    requests = request_archive.request_paths(run)
    for path, data in originals.items():
        assert (run / path).is_file() == (path not in requests)
        assert request_archive.read_bytes(run, path) == data


def test_without_zstd_or_a_seal_nothing_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, originals = _sealed_run(tmp_path)
    monkeypatch.setattr(request_archive, "available", lambda: False)
    assert request_archive.compact(run)["status"] == "unavailable"
    (run / "seal.json").unlink()
    monkeypatch.undo()
    assert request_archive.compact(run)["status"] == "unsealed"
    assert all((run / path).is_file() for path in originals)
