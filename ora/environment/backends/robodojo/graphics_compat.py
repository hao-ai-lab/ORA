"""Process-local Vulkan allocation-limit compatibility for Isaac Sim 5.1.

Only the isolated interpreter calls this module. It installs no dependencies
and never changes a driver, renderer feature, task or physics setting.
"""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


CAP = 0xFFE00000
PROFILE = "VP_ROBODOJO_RTX_ALLOCATION_COMPAT"
SOURCE_REVISION = "a400d0b10059b2cc3cd285475593eee79975a5c4"
SCHEMA = "ora.robodojo-graphics-compatibility.v1"
PATCH_SHA256 = "4189d74dda895d78786dac467021f70084714c68cefc9826406225c797b6193e"


class GraphicsCompatibilityError(ValueError):
    pass


def installation(runtime_directory: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    # Resolve the compatibility directory itself, so a relocated (linked)
    # directory still contains its own files.
    root = (runtime_directory / "graphics-compat").resolve()
    receipt = json.loads((root / "installation.json").read_text())
    if (receipt.get("schema") != "ora.robodojo-graphics-compat-installation.v1"
            or receipt.get("source_revision") != SOURCE_REVISION
            or str(receipt.get("source_url", "")).removesuffix(".git")
            != "https://github.com/KhronosGroup/Vulkan-Profiles"):
        raise GraphicsCompatibilityError("Vulkan compatibility installation source drift")
    patches = receipt.get("patches", [])
    if len(patches) != 1 or patches[0].get("sha256") != PATCH_SHA256:
        raise GraphicsCompatibilityError("Vulkan compatibility core-identity patch drift")
    receipt["patch"] = patches[0]
    paths = {}
    for role in ("layer", "profile", "manifest", "patch"):
        reference = receipt[role]
        path = (root / reference["path"]).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]):
            raise GraphicsCompatibilityError(f"Vulkan compatibility {role} hash/path drift")
        paths[role] = path
    profile = json.loads(paths["profile"].read_text())
    profiles = profile.get("profiles", {})
    capabilities = profile.get("capabilities", {})
    expected = {"properties": {"VkPhysicalDeviceMaintenance3Properties": {
        "maxMemoryAllocationSize": CAP}}}
    if (set(profiles) != {PROFILE} or len(capabilities) != 1
            or list(capabilities.values()) != [expected]
            or profiles[PROFILE].get("capabilities") != list(capabilities)):
        raise GraphicsCompatibilityError("Vulkan profile must override only the finite allocation limit")
    manifest = json.loads(paths["manifest"].read_text())
    layer = manifest.get("layer", {})
    if (layer.get("name") != "VK_LAYER_KHRONOS_profiles" or layer.get("type") != "GLOBAL"
            or Path(layer.get("library_path", "")).resolve() != paths["layer"]
            or layer.get("enable_environment") != {"ROBODOJO_VULKAN_COMPAT_ENABLE": "1"}
            or layer.get("disable_environment") != {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""}):
        raise GraphicsCompatibilityError("Vulkan compatibility layer manifest drift")
    return receipt, paths


class _App(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("pApplicationName", c.c_char_p), ("applicationVersion", c.c_uint32),
                ("pEngineName", c.c_char_p), ("engineVersion", c.c_uint32), ("apiVersion", c.c_uint32)]


class _Create(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                ("pApplicationInfo", c.c_void_p), ("enabledLayerCount", c.c_uint32),
                ("ppEnabledLayerNames", c.c_void_p), ("enabledExtensionCount", c.c_uint32),
                ("ppEnabledExtensionNames", c.c_void_p)]


class _Properties(c.Structure):
    _fields_ = [("apiVersion", c.c_uint32), ("driverVersion", c.c_uint32),
                ("vendorID", c.c_uint32), ("deviceID", c.c_uint32), ("deviceType", c.c_uint32),
                ("deviceName", c.c_char * 256), ("pipelineCacheUUID", c.c_uint8 * 16),
                ("remaining_properties", c.c_uint8 * 2048)]


class _Properties2(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("properties", _Properties)]


class _Maintenance3(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("maxPerSetDescriptors", c.c_uint32), ("maxMemoryAllocationSize", c.c_uint64)]


class _Identity(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("deviceUUID", c.c_uint8 * 16),
                ("driverUUID", c.c_uint8 * 16), ("deviceLUID", c.c_uint8 * 8),
                ("deviceNodeMask", c.c_uint32), ("deviceLUIDValid", c.c_uint32)]


def properties(*, expected_layer: Path | None = None) -> list[dict[str, Any]]:
    """Query Vulkan properties without creating a device, surface or renderer."""
    vk = c.CDLL("libvulkan.so.1")
    vk.vkCreateInstance.argtypes = [c.POINTER(_Create), c.c_void_p, c.POINTER(c.c_void_p)]
    vk.vkCreateInstance.restype = c.c_int32
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = c.c_int32
    vk.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(_Properties2)]
    vk.vkGetPhysicalDeviceProperties2.restype = None
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkDestroyInstance.restype = None
    app = _App(sType=0, pApplicationName=b"robodojo-graphics-readback", apiVersion=(1 << 22) | (1 << 12))
    create = _Create(sType=1, pApplicationInfo=c.c_void_p(c.addressof(app)))
    instance = c.c_void_p()
    if vk.vkCreateInstance(c.byref(create), None, c.byref(instance)) != 0:
        raise GraphicsCompatibilityError("Vulkan property readback could not create an instance")
    result = []
    try:
        count = c.c_uint32()
        if vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) != 0 or count.value == 0:
            raise GraphicsCompatibilityError("Vulkan property readback found no devices")
        devices = (c.c_void_p * count.value)()
        if vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) != 0:
            raise GraphicsCompatibilityError("Vulkan device enumeration failed")
        for device in devices:
            identity = _Identity(sType=1000071004)
            maintenance = _Maintenance3(sType=1000168000, pNext=c.c_void_p(c.addressof(identity)))
            props = _Properties2(sType=1000059001, pNext=c.c_void_p(c.addressof(maintenance)))
            vk.vkGetPhysicalDeviceProperties2(device, c.byref(props))
            if props.properties.vendorID != 0x10DE:
                continue
            result.append({
                "device_uuid": bytes(identity.deviceUUID).hex(),
                "driver_uuid": bytes(identity.driverUUID).hex(),
                "device_luid": bytes(identity.deviceLUID).hex(),
                "device_node_mask": int(identity.deviceNodeMask),
                "device_luid_valid": int(identity.deviceLUIDValid),
                "device_name": props.properties.deviceName.decode(),
                "driver_version": int(props.properties.driverVersion),
                "physical_properties_sha256": hashlib.sha256(bytes(props.properties)).hexdigest(),
                "max_per_set_descriptors": int(maintenance.maxPerSetDescriptors),
                "max_memory_allocation_size": int(maintenance.maxMemoryAllocationSize),
            })
        if not result:
            raise GraphicsCompatibilityError("Vulkan property readback found no NVIDIA devices")
        if expected_layer is not None and str(expected_layer) not in Path("/proc/self/maps").read_text():
            raise GraphicsCompatibilityError("the verified Vulkan profiles layer was not loaded")
    finally:
        vk.vkDestroyInstance(instance, None)
    return sorted(result, key=lambda value: value["device_uuid"])


def _configure(paths: dict[str, Path]) -> None:
    if "ROBODOJO_VULKAN_COMPAT_DISABLE" in os.environ:
        raise GraphicsCompatibilityError("Vulkan compatibility is explicitly disabled")
    os.environ.update({
        "ROBODOJO_VULKAN_COMPAT_ENABLE": "1",
        "VK_KHRONOS_PROFILES_PROFILE_NAME": PROFILE,
        "VK_KHRONOS_PROFILES_PROFILE_DIRS": str(paths["profile"].parent),
        "VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES": "SIMULATE_PROPERTIES_BIT",
        "VK_KHRONOS_PROFILES_DEBUG_REPORTS": "DEBUG_REPORT_ERROR_BIT",
        "VK_KHRONOS_PROFILES_EMULATE_PORTABILITY": "false",
        "VK_ADD_IMPLICIT_LAYER_PATH": str(paths["manifest"].parent),
    })
    data_root = paths["manifest"].parent.parent.parent
    os.environ["XDG_DATA_DIRS"] = str(data_root) + os.pathsep + os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")


def readback(runtime_directory: Path) -> dict[str, Any]:
    """Compare native and repaired properties in a disposable interpreter."""
    receipt, paths = installation(runtime_directory)
    before = properties()
    _configure(paths)
    after = properties(expected_layer=paths["layer"])
    without_limit = lambda rows: [{k: v for k, v in row.items() if k != "max_memory_allocation_size"} for row in rows]
    if (without_limit(before) != without_limit(after)
            or any(row["max_memory_allocation_size"] != CAP for row in after)):
        raise GraphicsCompatibilityError("Vulkan compatibility property readback drift")
    return {
        "schema": SCHEMA, "enabled": True, "profile_name": PROFILE,
        "max_memory_allocation_size": CAP, "source_revision": SOURCE_REVISION,
        "files": {role: receipt[role] for role in ("layer", "profile", "manifest", "patch")},
        "before": before, "after": after,
    }


# The readback reads and hashes installed files; on the shared HDD a
# writeback storm has pushed it past 30 s while healthy. A real hang still
# fails here, well inside the worker start limit.
READBACK_TIMEOUT_S = 120


def _start_gate():
    """``start_gate`` as a package module, or by path when this file runs as a script."""
    if __package__:
        from . import start_gate
        return start_gate
    import importlib.util
    spec = importlib.util.spec_from_file_location("robodojo_start_gate", Path(__file__).with_name("start_gate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_readback(runtime_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The readback, measured once per node, boot, driver and installation (RDJ-10).

    The value is the readback itself, byte-identical to a fresh one; every
    worker still proves in its own process that SDK startup loaded the verified
    layer (``verify_activation``).
    """
    receipt, _ = installation(runtime_directory)
    inputs = {"files": {role: receipt[role]["sha256"] for role in ("layer", "profile", "manifest", "patch")},
              "python": sys.executable, "readback_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    gate = _start_gate()
    return gate.cached("graphics-readback", inputs, lambda: readback(runtime_directory),
                       timeout_s=READBACK_TIMEOUT_S - 10)


def activate(runtime_directory: Path) -> dict[str, Any]:
    """Verify in a child, then configure this worker before Isaac loads Vulkan."""
    _, paths = installation(runtime_directory)
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(runtime_directory)],
        text=True, capture_output=True, timeout=READBACK_TIMEOUT_S, check=False,
    )
    if process.returncode != 0:
        raise GraphicsCompatibilityError(process.stderr.strip() or "Vulkan compatibility readback failed")
    value = json.loads(process.stdout.strip().splitlines()[-1])
    _configure(paths)
    return value


def verify_activation(runtime_directory: Path, binding: dict[str, Any]) -> None:
    """After SDK startup, bind the actual loaded library and unchanged files."""
    receipt, paths = installation(runtime_directory)
    current = {role: receipt[role] for role in paths}
    if (current != binding.get("files")
            or str(paths["layer"]) not in Path("/proc/self/maps").read_text()):
        raise GraphicsCompatibilityError("SDK did not load the verified Vulkan compatibility installation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_directory", type=Path)
    arguments = parser.parse_args()
    try:
        value, provenance = cached_readback(arguments.runtime_directory)
        print(json.dumps(provenance, sort_keys=True), file=sys.stderr)
        print(json.dumps(value, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"Vulkan driver compatibility rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
