"""Source-tracked execution-resource profiles for concurrent Runtime sessions.

An execution host profile is ``<hosts>/<host id>.yaml``. ``<hosts>`` is this
package's ``hosts/`` directory, or the directory named by
``ORA_HOSTS_DIR``; ``examples/hosts/example-host.yaml`` is a
documented template. Profiles shipped in the package (``local-cpu``) remain
available when ``ORA_HOSTS_DIR`` has no profile of that id.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from ora.orchestrator import yaml_cache


RESOURCE_BINDING_SCHEMA = "ora.execution-resource-binding.v1"
HOST_PROFILE_SCHEMA = "ora.execution-host.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOSTS_DIR_ENV = "ORA_HOSTS_DIR"
HOSTS_ROOT = Path(os.environ.get(HOSTS_DIR_ENV) or Path(__file__).resolve().parent / "hosts").resolve()
#: Profiles shipped with ORA (``local-cpu`` for ``ora local``); a user's ``ORA_HOSTS_DIR``
#: profile of the same id takes precedence.
BUILTIN_HOSTS_ROOT = (Path(__file__).resolve().parent / "hosts").resolve()


def _host_profile_path(stable_host: str) -> Path | None:
    for root in dict.fromkeys((HOSTS_ROOT, BUILTIN_HOSTS_ROOT)):
        path = (root / f"{stable_host}.yaml").resolve()
        if path.is_file() and path.is_relative_to(root):
            return path
    return None
HOST_MARKER_ENV = "ORA_EXECUTION_HOST"
WORKER_ENVIRONMENT_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_EGL_DEVICE_ID",
    # A software-rendered (Mesa llvmpipe) LIBERO lane selects its EGL vendor and
    # thread budget per lane: NVIDIA EGL on driver 595 corrupts or hangs when
    # several processes render on one GPU or SAPIEN Vulkan runs on the node.
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LIBGL_ALWAYS_SOFTWARE",
    "LP_NUM_THREADS",
}


class ExecutionResourceError(ValueError):
    """An execution resource profile is absent, malformed, or incompatible."""


def _stable_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise ExecutionResourceError(f"{label} must be a stable id")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_host_profile(host_id: str) -> tuple[Mapping[str, Any], Path]:
    """The host profile's shared frozen parse, reused while its bytes are unchanged.

    ``resolve_execution_resource`` only reads it and copies scalars out, so the
    frozen view never reaches a caller. ``ValueError`` covers non-UTF-8 bytes
    and a recursive alias.
    """

    stable_host = _stable_id(host_id, "execution host")
    path = _host_profile_path(stable_host)
    if path is None:
        raise ExecutionResourceError(
            f"execution host profile is unavailable: {stable_host}"
        )
    try:
        value = yaml_cache.load_frozen(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ExecutionResourceError(
            f"cannot load execution host profile: {exc}"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != HOST_PROFILE_SCHEMA
        or value.get("id") != stable_host
    ):
        raise ExecutionResourceError("execution host profile identity is invalid")
    return value, path


def require_registered_host(host_id: str) -> str:
    """Return ``host_id`` when it names a source-tracked host profile; raise otherwise."""
    _load_host_profile(host_id)
    return host_id


def declared_execution_host(expected: str | None = None) -> str:
    """The registered execution host this process runs on, from its marker.

    Qualification and evaluation tools bind their records to this id instead of
    assuming one machine: the marker must be set, name a source-tracked host
    profile, and equal ``expected`` when a plan already names a host.
    """

    marker = os.environ.get(HOST_MARKER_ENV)
    if not marker:
        raise ExecutionResourceError(f"{HOST_MARKER_ENV} is not set")
    _load_host_profile(marker)
    if expected is not None and marker != expected:
        raise ExecutionResourceError(
            f"execution host marker {marker} does not match required host {expected}"
        )
    return marker


def _implicit_exclusive_binding(host_id: str, runtime_kind: str) -> dict[str, Any]:
    body = {
        "schema": RESOURCE_BINDING_SCHEMA,
        "profile_id": "implicit-exclusive-host",
        "host_id": host_id,
        "runtime_kind": runtime_kind,
        "resource_class": "exclusive_host",
        "slot_id": "default",
        "worker_environment": {},
        "minimum_free_memory_mib": None,
        "source": "runtime-default",
    }
    return {**body, "profile_sha256": _canonical_sha256(body)}


def resolve_execution_resource(
    profile_id: Any,
    *,
    runtime_kind: str,
    required_host: str | None,
) -> dict[str, Any]:
    """Resolve a RunSpec resource selector into one hash-bound RunPlan binding."""

    host_id = _stable_id(required_host or "local", "required host")
    kind = _stable_id(runtime_kind, "runtime kind")
    if profile_id is None:
        return _implicit_exclusive_binding(host_id, kind)
    selected = _stable_id(profile_id, "runtime.resource_profile")
    host, host_path = _load_host_profile(host_id)
    resources = host.get("execution_resources")
    if not isinstance(resources, Mapping):
        raise ExecutionResourceError(
            f"execution host {host_id} does not register resource profiles"
        )
    profile = resources.get(selected)
    if not isinstance(profile, Mapping):
        raise ExecutionResourceError(
            f"execution resource profile is unavailable: {host_id}.{selected}"
        )
    allowed = {
        "runtime_kind", "resource_class", "slot_id", "worker_environment",
        "minimum_free_memory_mib",
    }
    if set(profile) != allowed:
        raise ExecutionResourceError(
            f"execution resource {selected} fields must be {sorted(allowed)}"
        )
    profile_kind = _stable_id(profile.get("runtime_kind"), "resource runtime kind")
    if profile_kind != kind:
        raise ExecutionResourceError(
            f"execution resource {selected} is for {profile_kind}, not {kind}"
        )
    resource_class = _stable_id(
        profile.get("resource_class"), "resource class"
    )
    slot_id = _stable_id(profile.get("slot_id"), "resource slot")
    raw_environment = profile.get("worker_environment")
    if not isinstance(raw_environment, Mapping):
        raise ExecutionResourceError("resource worker_environment must be an object")
    if not set(raw_environment).issubset(WORKER_ENVIRONMENT_NAMES):
        raise ExecutionResourceError(
            "resource worker_environment contains an untrusted name"
        )
    worker_environment: dict[str, str] = {}
    for name, raw_value in sorted(raw_environment.items()):
        if not isinstance(raw_value, str) or not raw_value or "\x00" in raw_value:
            raise ExecutionResourceError(
                f"resource worker_environment.{name} must be a non-empty string"
            )
        worker_environment[str(name)] = raw_value
    vendor_files = worker_environment.get("__EGL_VENDOR_LIBRARY_FILENAMES")
    if vendor_files is not None and not all(
        item.startswith("/") for item in vendor_files.split(":")
    ):
        raise ExecutionResourceError(
            "resource __EGL_VENDOR_LIBRARY_FILENAMES must name absolute files"
        )
    cuda = worker_environment.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None:
        cuda_devices = cuda.split(",")
        if not all(item.isdigit() for item in cuda_devices) or len(set(cuda_devices)) != len(cuda_devices):
            raise ExecutionResourceError(
                "resource CUDA_VISIBLE_DEVICES must list distinct numeric GPUs"
            )
    minimum = profile.get("minimum_free_memory_mib")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum <= 0
    ):
        raise ExecutionResourceError(
            "resource minimum_free_memory_mib must be a positive integer"
        )
    profile_body = {
        "schema": RESOURCE_BINDING_SCHEMA,
        "profile_id": selected,
        "host_id": host_id,
        "runtime_kind": kind,
        "resource_class": resource_class,
        "slot_id": slot_id,
        "worker_environment": worker_environment,
        "minimum_free_memory_mib": minimum,
        "source": _source_path(host_path),
    }
    return {**profile_body, "profile_sha256": _canonical_sha256(profile_body)}


def _source_path(path: Path) -> str:
    """A host profile's path, relative to the project when it lives inside it."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def resource_lease_key(binding: Mapping[str, Any]) -> str:
    if binding.get("schema") != RESOURCE_BINDING_SCHEMA:
        raise ExecutionResourceError("execution resource binding schema is invalid")
    resource_class = _stable_id(binding.get("resource_class"), "resource class")
    slot_id = _stable_id(binding.get("slot_id"), "resource slot")
    return f"{resource_class}--{slot_id}"


def load_host_profile(host_id: str) -> dict[str, Any] | None:
    """Read one source-tracked host profile; None when the host has none."""

    stable_host = _stable_id(host_id, "execution host")
    if _host_profile_path(stable_host) is None:
        return None
    return _load_host_profile(stable_host)[0]


__all__ = [
    "ExecutionResourceError", "RESOURCE_BINDING_SCHEMA",
    "load_host_profile", "resolve_execution_resource", "resource_lease_key",
]
