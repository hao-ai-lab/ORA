"""Lossless post-seal archive of a run's provider request artifacts, and reads that see through it.

Every model call stores its exact provider request under ``native/agent-artifacts/<sha256>.json``, and each
request carries the whole projected context again, inline images included as base64. Consecutive requests
therefore repeat the same megabytes: on one GPU cluster on 2026-10-09 these requests were about 95% of all run bytes,
974 GB across 2,823 runs, and filled the shared directory quota. Packed per run into one zstd long-range
archive they took 38 GB (about 25x), with nothing lost.

:func:`compact` runs after a run is sealed. It packs the sealed request files (event ``artifact_ref`` or a
payload ``*_request_ref`` whose kind is in :data:`REQUEST_KINDS`) into :data:`ARCHIVE`, streams the archive
back to check every member's sha256 against the seal inventory, writes :data:`MANIFEST`, and only then
removes the originals. Everything else in the run stays a plain file. The seal is not rewritten: a sealed
inventory record whose file is absent is satisfied by an archived member with the same sha256 and size
(:func:`inventory_record_present`), and :func:`read_bytes` returns the original bytes either way.

:func:`read_bytes` also reads the earlier operator format (``native/agent-artifacts-json.tar.zst`` with
``agent-artifacts-json.manifest.json``, every JSON artifact of the run), so runs compacted by hand before
this module existed stay readable. Compaction needs the ``tar`` and ``zstd`` executables; without them it
returns ``unavailable`` and leaves the run as it was.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

ARCHIVE = "native/provider-requests.tar.zst"
MANIFEST = "native/provider-requests.manifest.json"
SCHEMA = "ora.provider-request-archive.v1"
#: Artifact kinds that hold a whole provider request.
REQUEST_KINDS = frozenset({"native_provider_wire_request", "native_canonical_provider_request"})
#: zstd level; the long-range window is what removes the repetition between requests.
LEVEL = 15
#: The operator format of 2026-10-09: every JSON artifact of the run, names relative to the artifact dir.
LEGACY_ARCHIVE = "native/agent-artifacts-json.tar.zst"
LEGACY_MANIFEST = "native/agent-artifacts-json.manifest.json"
LEGACY_PREFIX = "native/agent-artifacts/"


class RequestArchiveError(RuntimeError):
    """A request archive is missing, unreadable or disagrees with the seal."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def available() -> bool:
    return shutil.which("zstd") is not None and shutil.which("tar") is not None


def _refs(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and value.get("kind") in REQUEST_KINDS:
            yield value
        for child in value.values():
            yield from _refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _refs(child)


def request_paths(run_root: Path) -> set[str]:
    """Run-relative paths of every provider request artifact the agent trace references."""
    events = Path(run_root) / "agent-events.jsonl"
    paths: set[str] = set()
    if not events.is_file():
        return paths
    with events.open(encoding="utf-8") as stream:
        for line in stream:
            if '_request' not in line:
                continue
            event = json.loads(line)
            for ref in _refs({"artifact_ref": event.get("artifact_ref"), "payload": event.get("payload")}):
                paths.add(str(ref["path"]))
    return paths


def _manifest(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def compact(run_root: Path, *, level: int = LEVEL) -> dict[str, Any]:
    """Archive the sealed provider requests of ``run_root``; a no-op when there is nothing to do."""
    run_root = Path(run_root).resolve()
    seal_path = run_root / "seal.json"
    if not seal_path.is_file():
        return {"status": "unsealed"}
    if (run_root / MANIFEST).is_file():
        return {"status": "already_archived"}
    if not available():
        return {"status": "unavailable", "reason": "tar and zstd executables are required"}
    records = json.loads(seal_path.read_text(encoding="utf-8")).get("inventory")
    if not isinstance(records, list):
        return {"status": "no_inventory"}
    inventory = {str(item["path"]): item for item in records}
    members = sorted(path for path in request_paths(run_root)
                     if path in inventory and (run_root / path).is_file())
    if not members:
        return {"status": "no_requests"}
    for path in members:
        record = inventory[path]
        if (run_root / path).stat().st_size != record["bytes"] or _sha256_file(run_root / path) != record["sha256"]:
            raise RequestArchiveError(f"sealed request drifted before archiving: {path}")
    temporary = run_root / (ARCHIVE + ".tmp")
    with temporary.open("wb") as out:
        tar = subprocess.Popen(["tar", "-C", str(run_root), "-cf", "-", "--", *members], stdout=subprocess.PIPE)
        subprocess.run(["zstd", "-q", f"-{level}", "--long=31", "-T2", "-c"], stdin=tar.stdout, stdout=out,
                       check=True)
        tar.stdout.close()
        if tar.wait() != 0:
            temporary.unlink(missing_ok=True)
            raise RequestArchiveError("tar failed while archiving provider requests")
        out.flush()
        os.fsync(out.fileno())
    try:
        found = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _members(temporary)}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    expected = {path: (inventory[path]["bytes"], inventory[path]["sha256"]) for path in members}
    if found != expected:
        temporary.unlink(missing_ok=True)
        raise RequestArchiveError("provider request archive does not reproduce the sealed files")
    os.replace(temporary, run_root / ARCHIVE)
    manifest = {
        "schema": SCHEMA, "created_unix_s": time.time(), "archive": ARCHIVE,
        "archive_sha256": _sha256_file(run_root / ARCHIVE), "archive_bytes": (run_root / ARCHIVE).stat().st_size,
        "compressor": f"zstd -{level} --long=31 (read with --long=31)",
        "claim_boundary": "lossless storage of sealed inventory files; each member's sha256 and size equal its "
                          "seal.json inventory record",
        "members": [{"path": path, "bytes": inventory[path]["bytes"], "sha256": inventory[path]["sha256"]}
                    for path in members],
    }
    temporary_manifest = run_root / (MANIFEST + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, run_root / MANIFEST)
    for path in members:
        (run_root / path).unlink()
    _archived.cache_clear()
    return {"status": "archived", "members": len(members),
            "bytes_before": sum(inventory[path]["bytes"] for path in members),
            "archive_bytes": manifest["archive_bytes"]}


def _members(archive: Path) -> Iterator[tuple[str, bytes]]:
    process = subprocess.Popen(["zstd", "-q", "-d", "--long=31", "-c", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                if member.isfile():
                    handle = stream.extractfile(member)
                    yield member.name, handle.read() if handle is not None else b""
    finally:
        process.stdout.close()
        if process.wait() != 0:
            raise RequestArchiveError(f"cannot decompress {archive}")


@functools.lru_cache(maxsize=2)
def _archived(run_root: str, stamp: tuple[float, ...]) -> dict[str, bytes]:
    """Every archived member of one run, keyed by run-relative path (both formats)."""
    root = Path(run_root)
    members: dict[str, bytes] = {}
    if (root / MANIFEST).is_file():
        members.update(_members(root / ARCHIVE))
    if (root / LEGACY_MANIFEST).is_file():
        members.update((LEGACY_PREFIX + name, data) for name, data in _members(root / LEGACY_ARCHIVE))
    return members


def _stamp(run_root: Path) -> tuple[float, ...]:
    return tuple((run_root / name).stat().st_mtime if (run_root / name).is_file() else 0.0
                 for name in (MANIFEST, LEGACY_MANIFEST))


def archived_bytes(run_root: Path, relative: str) -> bytes | None:
    """The archived bytes of ``relative`` in ``run_root``, or None if no archive holds it."""
    run_root = Path(run_root).resolve()
    stamp = _stamp(run_root)
    if not any(stamp):
        return None
    return _archived(str(run_root), stamp).get(str(relative))


def read_bytes(run_root: Path, relative: str) -> bytes:
    """The bytes of a run artifact, from its file or, after archiving, from the run's archive."""
    run_root = Path(run_root).resolve()
    path = (run_root / relative).resolve()
    if not path.is_relative_to(run_root):
        raise RequestArchiveError(f"artifact path escapes the run: {relative}")
    if path.is_file():
        return path.read_bytes()
    data = archived_bytes(run_root, relative)
    if data is None:
        raise FileNotFoundError(path)
    return data


def exists(run_root: Path, relative: str) -> bool:
    try:
        read_bytes(run_root, relative)
    except (FileNotFoundError, RequestArchiveError):
        return False
    return True


def inventory_record_present(run_root: Path, record: Mapping[str, Any]) -> bool:
    """A sealed inventory record whose file is absent is present if an archive holds identical bytes."""
    data = archived_bytes(run_root, str(record["path"]))
    return (data is not None and len(data) == record["bytes"]
            and hashlib.sha256(data).hexdigest() == record["sha256"])


def verified_bytes(run_root: Path, ref: Mapping[str, Any]) -> bytes | None:
    """The bytes ``ref`` names if they exist (as a file or archived) and match its sha256; otherwise None."""
    root = Path(run_root).resolve()
    if not (root / str(ref.get("path"))).resolve().is_relative_to(root):
        return None
    try:
        data = read_bytes(root, str(ref["path"]))
    except (FileNotFoundError, RequestArchiveError, KeyError):
        return None
    return data if hashlib.sha256(data).hexdigest() == ref.get("sha256") else None


def compact_after_seal(run_root: Path) -> None:
    """Landing hook: archive the requests of a just-sealed run, silently and fail-open.

    The supervisor log is sealed inventory, so nothing is printed; a failure leaves the sealed files as they
    are and is recorded in ``native/provider-requests.refusal.json``.
    """
    try:
        compact(run_root)
    except Exception as exc:  # noqa: BLE001 - recorded beside the run, never raised into landing
        try:
            refusal = Path(run_root) / "native" / "provider-requests.refusal.json"
            refusal.parent.mkdir(parents=True, exist_ok=True)
            refusal.write_text(json.dumps({"schema": "ora.provider-request-archive-refusal.v1",
                                           "run_id": Path(run_root).name, "reason": str(exc)}) + "\n",
                               encoding="utf-8")
        except OSError:
            pass


def convert_legacy(run_root: Path) -> dict[str, Any]:
    """Put a run compacted in the 2026-10-09 operator format back into plain files, then archive its requests.

    Every member is restored byte-identical (checked against the legacy manifest) and read-only, as sealed;
    then :func:`compact` archives only the provider requests, so every other reader finds its files again.
    """
    run_root = Path(run_root).resolve()
    legacy = run_root / LEGACY_MANIFEST
    if not legacy.is_file():
        return {"status": "not_legacy"}
    expected = {item["name"]: (item["bytes"], item["sha256"])
                for item in json.loads(legacy.read_text(encoding="utf-8"))["files"]}
    target = run_root / LEGACY_PREFIX
    target.mkdir(parents=True, exist_ok=True)
    restored = 0
    for name, data in _members(run_root / LEGACY_ARCHIVE):
        if expected.get(name) != (len(data), hashlib.sha256(data).hexdigest()):
            raise RequestArchiveError(f"legacy archive member does not match its manifest: {name}")
        path = target / name
        if not path.exists():
            temporary = target / f".{name}.restore"
            temporary.write_bytes(data)
            os.chmod(temporary, 0o444)
            os.replace(temporary, path)
        restored += 1
    if restored != len(expected):
        raise RequestArchiveError("legacy archive is missing members")
    legacy.unlink()
    (run_root / LEGACY_ARCHIVE).unlink()
    _archived.cache_clear()
    return {"status": "restored", "files": restored, "compact": compact(run_root)}


def _run_roots(arguments: list[str]) -> list[Path]:
    roots: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if (path / "plan.json").is_file():
            roots.append(path)
        elif (path / "runs").is_dir():
            roots.extend(sorted(child for child in (path / "runs").iterdir() if child.is_dir()))
    return roots


def main(argv: list[str] | None = None) -> int:
    """Operator entry: archive the provider requests of sealed runs (run roots or experiment directories)."""
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--convert-legacy", action="store_true",
                        help="restore runs in the 2026-10-09 all-JSON operator format, then archive requests only")
    args = parser.parse_args(argv)
    roots = _run_roots(args.paths)
    work = convert_legacy if args.convert_legacy else compact
    counts: dict[str, int] = {}
    before = after = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        for root, future in [(root, pool.submit(work, root)) for root in roots]:
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - reported per run
                failures.append(f"{root}: {exc}")
                continue
            final = result.get("compact", result)
            counts[final["status"]] = counts.get(final["status"], 0) + 1
            before += final.get("bytes_before", 0)
            after += final.get("archive_bytes", 0)
    print(json.dumps({"runs": len(roots), "status": counts, "request_bytes": before, "archive_bytes": after,
                      "failures": failures[:20], "failure_count": len(failures)}, indent=1))
    return 1 if failures else 0


__all__ = ["ARCHIVE", "LEGACY_ARCHIVE", "LEGACY_MANIFEST", "MANIFEST", "REQUEST_KINDS", "RequestArchiveError",
           "SCHEMA", "archived_bytes", "available", "compact", "compact_after_seal", "convert_legacy", "exists",
           "inventory_record_present", "main", "read_bytes", "request_paths", "verified_bytes"]


if __name__ == "__main__":
    raise SystemExit(main())
