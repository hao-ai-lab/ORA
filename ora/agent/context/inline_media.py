"""PolicySession cache for recoverable payload admission during projection.

Lossless projection re-materializes the whole raw history on every call. Before
this cache, every historical image was base64-decoded, re-read from the
Artifact store and re-hashed on each projection. The first admission of a
payload still runs the complete catalog path (decode, content hash, store
collision check, index registration or resolve-and-compare). Later
projections reuse that verified result while the backing file keeps the same
``(inode, size, mtime_ns)``; any change to that file identity drops the entry
and repeats the complete admission, which then fails closed on drift.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.memory.artifacts import ContextArtifactCatalog

from .request_size import GenerationalMemo


def _decode_image_data_url(value: Any) -> tuple[str, bytes] | None:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return None
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header:
        raise PolicySessionError("inline image is not a base64 data URL")
    media_type = header[5:].split(";", 1)[0]
    try:
        return media_type, base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise PolicySessionError("inline image base64 is invalid") from exc


@dataclass(frozen=True)
class _InlineImage:
    media_type: str
    sha256: str
    byte_count: int


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return status.st_ino, status.st_size, status.st_mtime_ns


class InlineMediaCache:
    """Verified admissions bound to one ContextArtifactCatalog instance."""

    def __init__(self) -> None:
        self._catalog: ContextArtifactCatalog | None = None
        self._images = GenerationalMemo()
        self._resolved = GenerationalMemo()
        self._stored: dict[tuple[Any, ...], tuple[dict[str, Any], Path, Any]] = {}

    def bind(self, catalog: ContextArtifactCatalog) -> None:
        if self._catalog is not catalog:
            self._images, self._resolved = GenerationalMemo(), GenerationalMemo()
            self._stored = {}
            self._catalog = catalog
        self._images.rotate()
        self._resolved.rotate()

    def _image(self, url: Any) -> tuple[_InlineImage, bytes | None]:
        cached = self._images.get(url) if type(url) is str else None
        if cached is not None:
            return cached, None
        decoded = _decode_image_data_url(url)
        if decoded is None:
            raise PolicySessionError("lossless Context supports only inline image data URLs")
        media_type, data = decoded
        image = _InlineImage(media_type, hashlib.sha256(data).hexdigest(), len(data))
        self._images.put(url, image)
        return image, data

    def put_bytes(
        self, data: bytes, *, kind: str, suffix: str, media_type: str,
        round_index: int | None, action_step: int | None,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        return self._put(
            digest, len(data), lambda: data, kind=kind, suffix=suffix,
            media_type=media_type, round_index=round_index, action_step=action_step,
        )

    def _put(
        self, digest: str, byte_count: int, data: Any, *, kind: str, suffix: str,
        media_type: str, round_index: int | None, action_step: int | None,
    ) -> dict[str, Any]:
        catalog = self._catalog
        assert catalog is not None
        key = (digest, byte_count, kind, suffix, media_type, round_index, action_step)
        cached = self._stored.get(key)
        if cached is not None and _file_identity(cached[1]) == cached[2]:
            return copy.deepcopy(cached[0])
        reference = catalog.put_bytes(
            data(), kind=kind, suffix=suffix, media_type=media_type,
            round_index=round_index, action_step=action_step,
        )
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = catalog.artifacts.root / f"{digest}{safe_suffix}"
        identity = _file_identity(path)
        if identity is not None:
            self._stored[key] = (copy.deepcopy(reference), path, identity)
        return reference

    def admit_image(
        self, image: Any, *, existing_refs: Any, image_index: int,
        round_index: int | None, action_step: int | None,
    ) -> dict[str, Any]:
        """Return the public ref exactly as the uncached admission path would."""

        catalog = self._catalog
        assert catalog is not None
        url = image.get("url") if isinstance(image, Mapping) else None
        identity, data = self._image(url)

        def payload() -> bytes:
            if data is not None:
                return data
            decoded = _decode_image_data_url(url)
            assert decoded is not None
            return decoded[1]

        if not isinstance(existing_refs, list) or image_index >= len(existing_refs):
            suffix = ".png" if identity.media_type == "image/png" else ".img"
            return self._put(
                identity.sha256, identity.byte_count, payload,
                kind="context_image", suffix=suffix, media_type=identity.media_type,
                round_index=round_index, action_step=action_step,
            )
        existing = existing_refs[image_index]
        if not isinstance(existing, Mapping) or not isinstance(existing.get("ref"), str):
            raise PolicySessionError("inline media Artifact reference is invalid")
        key = (str(existing["ref"]), url)
        cached = self._resolved.get(key)
        entry = None
        if cached is not None and _file_identity(cached[1]) == cached[2]:
            entry = cached[0]
        else:
            self._resolved.discard(key)
            try:
                entry, stored = catalog.resolve(str(existing["ref"]))
            except Exception as exc:
                raise PolicySessionError(
                    "inline media Artifact reference cannot be resolved"
                ) from exc
            if stored != payload():
                raise PolicySessionError("inline media differs from its Artifact reference")
            storage = entry.get("storage_ref")
            path = catalog.run_root / str(storage.get("path"))
            file_identity = _file_identity(path)
            if file_identity is not None:
                self._resolved.put(key, (copy.deepcopy(entry), path, file_identity))
        if (
            entry.get("media_type") != identity.media_type
            or existing.get("sha256") != entry.get("sha256")
        ):
            raise PolicySessionError("inline media differs from its Artifact reference")
        return catalog.public_ref(
            entry, round_index=round_index, action_step=action_step,
        )
