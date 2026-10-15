"""Read and update the repository ``.env`` that ``ora setup --env`` owns keys in.

Only plain ``KEY=value`` lines are understood (the format ``.env.example``
uses and ``set -a; . ./.env`` sources). Updates keep every other line, replace
owned keys in place and append new ones under one marked block. Credentials
are never written: a key that names an API key, token or secret is refused,
and the template's empty ``DEEPSEEK_*_API_KEY=`` lines are removed because
sourcing them after a secrets file blanks the key.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_CREDENTIAL = re.compile(r"(API_KEY|_TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
#: The template lines that blank a key when ``.env`` is sourced after the secrets.
EMPTY_CREDENTIAL_LINES = ("DEEPSEEK_LOW_API_KEY", "DEEPSEEK_LOW_API_KEYS", "DEEPSEEK_API_KEY")
BLOCK_HEADER = "# --- written by ./scripts/ora setup --env (single-GPU); edit freely, setup rewrites these keys ---"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def read(path: Path) -> dict[str, str]:
    """The ``KEY=value`` assignments of ``path`` (later lines win); empty when absent."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def _render(key: str, value: str) -> str:
    if "'" in value or "\n" in value:
        raise ValueError(f"refusing to write {key}: its value cannot be single-quoted ({value!r})")
    return f"{key}='{value}'" if re.search(r"[\s#\"$`\\;&|<>()]", value) else f"{key}={value}"


def update(path: Path, owned: Mapping[str, str], *, defaults: Mapping[str, str] | None = None) -> dict[str, str]:
    """Write ``owned`` keys, add ``defaults`` only where unset; return what changed.

    The file is created with mode 0600 when absent and replaced atomically.
    """
    for key in (*owned, *(defaults or {})):
        if _CREDENTIAL.search(key):
            raise ValueError(f"ora setup never writes credentials ({key})")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    current = read(path)
    wanted = dict(owned)
    for key, value in (defaults or {}).items():
        if not current.get(key):
            wanted[key] = value
    changed = {key: value for key, value in wanted.items() if current.get(key) != value}
    result: list[str] = []
    written: set[str] = set()
    for line in lines:
        match = _LINE.match(line) if not line.lstrip().startswith("#") else None
        key = match.group(1) if match else None
        if key in EMPTY_CREDENTIAL_LINES and not _unquote(match.group(2)):
            continue
        if key in wanted:
            if key not in written:
                result.append(_render(key, wanted[key]))
                written.add(key)
            continue
        result.append(line)
    missing = [key for key in wanted if key not in written]
    if missing:
        if BLOCK_HEADER not in result:
            if result and result[-1].strip():
                result.append("")
            result.append(BLOCK_HEADER)
        result.extend(_render(key, wanted[key]) for key in missing)
    text = "\n".join(result) + "\n"
    if lines and text == "\n".join(lines) + "\n":
        return changed
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    descriptor, temporary = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return changed


def load_into(environ: dict[str, str] | os._Environ, path: Path) -> list[str]:
    """Set each non-empty ``path`` value that ``environ`` does not already set; return the names."""
    applied = []
    for key, value in read(path).items():
        if value and not environ.get(key):
            environ[key] = value
            applied.append(key)
    return applied


__all__ = ["BLOCK_HEADER", "EMPTY_CREDENTIAL_LINES", "load_into", "read", "update"]
