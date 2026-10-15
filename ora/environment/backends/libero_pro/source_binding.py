"""Source revisions, runtime dependency files and the qualified controller binding.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ora.environment.controllers.libero_osc import libero_osc_action_contract

from .construction import build_environment as _build_environment
from .pinned_imports import _benchmark_data_binding, _path_is_within
from .common import LiberoProError, _jsonable, _file_hash, _git


class LiberoSourceBindingMixin:
    """Source revisions, runtime dependency files and the qualified controller binding."""

    def _verify_source(self) -> dict[str, Any]:
        if self.environment_factory is not _build_environment:
            return {
                "libero_pro_revision": self.required_libero_pro_revision,
                "data_revision": self.required_data_revision,
                "test_double": True,
            }
        bindings = (
            ("libero_pro", self.libero_pro_root, self.required_libero_pro_revision),
            ("data", self.data_root, self.required_data_revision),
        )
        manifest: dict[str, Any] = {"environment_driver": self.environment_driver}
        for name, root, expected in bindings:
            revision = _git(root, "rev-parse", "HEAD")
            if revision != expected:
                raise LiberoProError(
                    f"{name} revision mismatch: expected {expected}, got {revision}"
                )
            dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                raise LiberoProError(f"{name} source is modified: {dirty}")
            manifest[f"{name}_revision"] = revision
        return manifest

    def _bind_dependency_files(self) -> None:
        candidates: list[Path] = []
        for key in ("bddl_file", "init_states_file"):
            path = Path(str(self.metadata.get(key, ""))).expanduser().resolve()
            if path.is_file():
                candidates.append(path)
        bindings = self.metadata.get("runtime_bindings", {})
        if self.environment_factory is _build_environment:
            if not isinstance(bindings, Mapping):
                raise LiberoProError("runtime source bindings are unavailable")
            benchmark_data_root, _benchmark_data_source = _benchmark_data_binding(
                libero_pro_root=self.libero_pro_root,
                data_root=self.data_root,
                suite=self.suite,
            )
            required = {
                "libero_module": self.libero_pro_root / "libero",
                "benchmark_module": self.libero_pro_root / "libero",
                "direct_driver_module": Path(__file__).resolve().parent,
                "bddl_root": benchmark_data_root / "bddl_files",
                "init_states_root": benchmark_data_root / "init_files",
            }
            for key, expected_root in required.items():
                path = Path(str(bindings.get(key, ""))).resolve()
                if not _path_is_within(path, expected_root):
                    raise LiberoProError(
                        f"runtime binding {key} escaped its verified source: {path}"
                    )
            for key in ("bddl_file", "init_states_file"):
                path = Path(str(self.metadata.get(key, ""))).resolve()
                expected = (
                    benchmark_data_root / "bddl_files"
                    if key == "bddl_file"
                    else benchmark_data_root / "init_files"
                )
                if not _path_is_within(path, expected):
                    raise LiberoProError(
                        f"runtime {key} escaped the pinned data checkout: {path}"
                    )
            if self.robustness_case is not None:
                for key in (
                    "robustness_runtime_module", "robustness_parser_module",
                ):
                    path = Path(str(bindings.get(key, ""))).resolve()
                    expected_root = self.libero_pro_root / "libero/libero/envs"
                    if (
                        not path.is_file()
                        or not _path_is_within(path, expected_root)
                    ):
                        raise LiberoProError(
                            f"runtime binding {key} escaped the paired "
                            f"LIBERO-Pro source: {path}"
                        )
            robosuite_module = Path(str(bindings.get(
                "robosuite_module", "",
            ))).resolve()
            controller_config = Path(str(bindings.get(
                "controller_config_file", "",
            ))).resolve()
            panda_gripper = Path(str(bindings.get(
                "panda_gripper_file", "",
            ))).resolve()
            panda_gripper_module = Path(str(bindings.get(
                "panda_gripper_module", "",
            ))).resolve()
            if not robosuite_module.is_file():
                raise LiberoProError("bound RoboSuite module is unavailable")
            robosuite_root = robosuite_module.parent
            for label, path in (
                ("OSC controller config", controller_config),
                ("Panda gripper module", panda_gripper_module),
                ("Panda gripper", panda_gripper),
            ):
                if not path.is_file() or not _path_is_within(
                    path, robosuite_root,
                ):
                    raise LiberoProError(
                        f"bound {label} escaped the imported RoboSuite package"
                    )
            try:
                bound_config = json.loads(
                    controller_config.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise LiberoProError(
                    "bound OSC controller config is not readable JSON"
                ) from exc
            if (
                not isinstance(bound_config, Mapping)
                or libero_osc_action_contract(bound_config)
                != self.controller_action_contract
            ):
                raise LiberoProError(
                    "bound OSC controller config differs from the active contract"
                )
        for value in bindings.values() if isinstance(bindings, Mapping) else ():
            path = Path(str(value)).resolve()
            if path.is_file():
                candidates.append(path)
        candidates = sorted(set(candidates))
        dependencies = [
            {"path": str(path), "sha256": _file_hash(path), "bytes": path.stat().st_size}
            for path in candidates
        ]
        self.source_manifest = {
            **self.source_manifest,
            "runtime_bindings": _jsonable(bindings),
            "dependencies": dependencies,
        }

    def _validate_qualified_controller_binding(self) -> None:
        """Require reset-time control sources to equal the passing preflight."""

        if self.environment_factory is not _build_environment:
            return
        expected = self.qualified_controller_binding
        if (
            not isinstance(expected, Mapping)
            or expected.get("schema")
            != "ora.libero-controller-qualification.v1"
            or expected.get("contract_sha256")
            != self.controller_action_contract.get("contract_sha256")
            or not isinstance(expected.get("binding"), Mapping)
        ):
            raise LiberoProError(
                "LIBERO controller qualification is absent or differs at reset"
            )
        bindings = self.metadata.get("runtime_bindings")
        if not isinstance(bindings, Mapping):
            raise LiberoProError("runtime source bindings are unavailable")
        expected_binding = expected["binding"]
        for path_field, digest_field in (
            ("robosuite_module", "robosuite_module_sha256"),
            ("controller_config_file", "controller_config_sha256"),
            ("panda_gripper_module", "panda_gripper_module_sha256"),
            ("panda_gripper_file", "panda_gripper_sha256"),
        ):
            path = Path(str(bindings.get(path_field, ""))).resolve()
            if (
                not path.is_file()
                or str(path) != expected_binding.get(path_field)
                or _file_hash(path) != expected_binding.get(digest_field)
            ):
                raise LiberoProError(
                    f"LIBERO controller source changed after preflight: {path_field}"
                )
