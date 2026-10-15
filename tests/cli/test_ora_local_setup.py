"""Isolated CPU setup and read-only readiness; no simulator or downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ora.orchestrator import ora_local_setup as setup


def test_bootstrap_uses_only_stdlib_and_canonical_source_pins():
    from ora.environment.benchmarks.libero_pro import (
        LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION,
    )
    assert (setup.CODE_REVISION, setup.DATA_REVISION) == (
        LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION,
    )
    code = ("import sys; sys.path.insert(0," + repr(str(setup.PROJECT_ROOT)) + "); "
            "from ora.orchestrator.ora_local_setup import main; main(['--help'])")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "separate simulator runtime" in result.stdout


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    # A Linux host without an NVIDIA EGL vendor file, whatever the test machine has.
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_DIRS", raising=False)
    monkeypatch.setattr(setup, "EGL_VENDOR_DIRECTORIES", ())


@pytest.fixture
def installed(tmp_path, monkeypatch, linux):
    paths = setup.local_paths(tmp_path / "local environment")
    for root, pattern in (
        (paths["source_root"], "libero/libero/assets/scene.xml"),
        (paths["source_root"], "libero/libero/bddl_files/libero_goal/task.bddl"),
        (paths["source_root"], "libero/libero/init_files/libero_goal/task.pruned_init"),
        (paths["data_root"], "bddl_files/libero_goal_swap/task.bddl"),
        (paths["data_root"], "init_files/libero_goal_swap/task.pruned_init"),
    ):
        path = root / pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source-bound test asset")
    manifest = {"schema": setup.SCHEMA, "code_revision": setup.CODE_REVISION,
                "data_revision": setup.DATA_REVISION,
                "paths": {key: str(value) for key, value in paths.items()},
                "environment": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}}
    paths["manifest"].write_text(json.dumps(manifest))
    def git(root, *args, **kwargs):
        if args[0] == "status":
            return ""
        return setup.CODE_REVISION if root == paths["source_root"] else setup.DATA_REVISION
    monkeypatch.setattr(setup, "_git", git)
    metadata = {"python": [3, 11, 12], "packages": setup.runtime_requirements()}
    monkeypatch.setattr(setup, "_run", lambda *args, **kwargs: json.dumps(metadata))
    monkeypatch.setattr(setup, "_graphics_check", lambda *args: (True, "fixture OSMesa library"))
    return paths, manifest, metadata


def test_default_root_and_explicit_paths_are_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("ORA_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert setup.default_local_root() == tmp_path / "xdg/ora"
    monkeypatch.setenv("ORA_LOCAL_ROOT", str(tmp_path / "chosen"))
    paths = setup.local_paths()
    assert paths["root"] == tmp_path / "chosen"
    assert setup.CODE_REVISION in paths["source_root"].name
    assert setup.DATA_REVISION in paths["data_root"].name
    assert paths["python"] == paths["root"] / "runtime/bin/python"


def test_platform_graphics_defaults_and_rejections(monkeypatch, linux):
    assert setup.graphics_environment() == {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    assert setup.graphics_environment("egl")["MUJOCO_GL"] == "egl"
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: ("13.0", ("", "", ""), "arm64"))
    assert setup.graphics_environment() == {"MUJOCO_GL": "cgl", "PYOPENGL_PLATFORM": "darwin"}
    with pytest.raises(setup.LocalSetupError, match="unsupported"):
        setup.graphics_environment("osmesa")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    with pytest.raises(setup.LocalSetupError, match="supports Apple Silicon"):
        setup.graphics_environment()


def test_linux_auto_graphics_selects_egl_only_with_an_nvidia_vendor_file(tmp_path, monkeypatch, linux):
    vendors = tmp_path / "egl_vendor.d"
    vendors.mkdir()
    (vendors / "50_mesa.json").write_text("{}")
    monkeypatch.setattr(setup, "EGL_VENDOR_DIRECTORIES", (str(vendors),))
    assert setup.graphics_environment()["MUJOCO_GL"] == "osmesa"
    (vendors / "10_nvidia.json").write_text("{}")
    assert setup.nvidia_egl_vendor() == str(vendors / "10_nvidia.json")
    assert setup.graphics_environment() == {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    # The glvnd override variable replaces the default search path.
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_DIRS", str(tmp_path / "elsewhere"))
    assert setup.graphics_environment()["MUJOCO_GL"] == "osmesa"
    # An explicit backend is never second-guessed.
    assert setup.graphics_environment("osmesa")["MUJOCO_GL"] == "osmesa"


def test_dependency_closure_has_exact_cpu_pins(monkeypatch, linux):
    requirements = setup.runtime_requirements()
    assert len(requirements) == 74
    assert requirements["torch"] == "2.6.0+cpu"
    assert requirements["mujoco"] == "3.8.1"
    assert requirements["robosuite"] == "1.4.1"
    assert requirements["opencv-python-headless"] == "4.11.0.86"
    assert "opencv-python" not in requirements
    assert {"setuptools", "wheel", "numba", "llvmlite", "pyopengl"} <= requirements.keys()
    assert not any("nvidia" in key or "cuda" in key or key == "triton" for key in requirements)
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: ("13.0", ("", "", ""), "arm64"))
    requirements = setup.runtime_requirements()
    assert len(requirements) == 77
    assert requirements["torch"] == "2.6.0"
    assert requirements["opencv-python-headless"] == "4.11.0.86"
    assert "opencv-python" not in requirements
    assert "pyobjc-framework-cocoa" in requirements and "evdev" not in requirements


@pytest.mark.parametrize("release,accepted", [("12.7.6", False), ("13.0", True), ("15.6.1", True), ("", False)])
def test_macos_minimum_version_checked_before_download(tmp_path, monkeypatch, release, accepted):
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: (release, ("", "", ""), "arm64"))
    if accepted:
        assert setup._platform() == ("Darwin", "arm64")
    else:
        monkeypatch.setattr(setup, "_run", lambda *args, **kwargs: pytest.fail("must reject before any subprocess"))
        with pytest.raises(setup.LocalSetupError, match="macOS 13 or newer"):
            setup.setup(tmp_path / "new")
        assert not (tmp_path / "new").exists()


def test_missing_doctor_has_no_filesystem_or_simulator_side_effect(tmp_path, linux):
    root = tmp_path / "not-created"
    before = set(sys.modules)
    report = setup.doctor(root)
    assert not report["ready"]
    assert not root.exists()
    assert not any(name.startswith(("mujoco", "robosuite", "libero.libero"))
                   for name in set(sys.modules) - before)
    assert "no simulator or provider" in report["qualification"]


def test_doctor_ready_checks_exact_files_packages_and_metadata(installed):
    paths, manifest, metadata = installed
    before = {str(path): path.read_bytes() for path in paths["root"].rglob("*") if path.is_file()}
    report = setup.doctor(paths["root"])
    assert report["ready"]
    assert setup.environment(paths["root"]) == manifest["environment"]
    assert before == {str(path): path.read_bytes() for path in paths["root"].rglob("*") if path.is_file()}
    metadata["packages"]["torch"] = "2.6.0+cu124"
    report = setup.doctor(paths["root"])
    assert not report["ready"]
    assert not next(check for check in report["checks"] if check["name"] == "runtime_packages")["passed"]


def test_doctor_rejects_dirty_source_and_missing_assets(installed, monkeypatch):
    paths, _, _ = installed
    monkeypatch.setattr(setup, "_git", lambda *args, **kwargs: "modified")
    (paths["source_root"] / "libero/libero/assets/scene.xml").unlink()
    report = setup.doctor(paths["root"])
    assert not report["ready"]
    failures = {check["name"] for check in report["checks"] if not check["passed"]}
    assert {"source_root", "data_root", "assets"} <= failures


def test_doctor_rejects_lfs_pointer_in_required_init(installed):
    paths, _, _ = installed
    (paths["data_root"] / "init_files/libero_goal_swap/task.pruned_init").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64)
    report = setup.doctor(paths["root"])
    assert not report["ready"]
    assert not next(check for check in report["checks"] if check["name"] == "pro_init_states")["passed"]


def test_manifest_cannot_override_paths_or_environment(installed):
    paths, manifest, _ = installed
    manifest["paths"]["python"] = "/usr/bin/python3"
    paths["manifest"].write_text(json.dumps(manifest))
    assert not setup.doctor(paths["root"])["ready"]
    assert setup.environment(paths["root"]) == setup.graphics_environment()
    manifest["paths"] = {key: str(value) for key, value in paths.items()}
    manifest["environment"]["LD_PRELOAD"] = "untrusted"
    paths["manifest"].write_text(json.dumps(manifest))
    assert not setup.doctor(paths["root"])["ready"]
    with pytest.raises(setup.LocalSetupError, match="graphics settings"):
        setup.environment(paths["root"])


def test_setup_uses_only_separate_root_and_pinned_commands(tmp_path, monkeypatch, linux):
    root = tmp_path / "a local root"
    commands, checkouts = [], []
    original_environment = dict(os.environ)
    monkeypatch.setattr(setup.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(setup, "_checkout", lambda path, url, revision: checkouts.append((path, url, revision)))
    def run(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "/python311":
            return "[3, 11, 12]"
        if argv[0] == str(root / "runtime/bin/python"):
            return "[3, 11, 12]"
        if argv[0] == "/tools/node":
            return "v24.4.0"
        if argv[1] == "venv":
            Path(argv[-1]).mkdir(parents=True)
            (Path(argv[-1]) / "pyvenv.cfg").write_text("version = 3.11.12")
        return ""
    monkeypatch.setattr(setup, "_run", run)
    monkeypatch.setattr(setup, "doctor", lambda path: {"ready": True, "root": str(path)})
    assert setup.setup(root, python="/python311")["ready"]
    assert [item[2] for item in checkouts] == [setup.CODE_REVISION, setup.DATA_REVISION]
    assert all(root in item[0].parents for item in checkouts)
    install = commands[-1]
    assert "--no-deps" in install and "--no-build-isolation" in install
    assert install[install.index("--torch-backend") + 1] == "cpu"
    assert all("--system" not in command for command in commands)
    assert any(command[1] == "venv" and "--no-python-downloads" in command for command in commands)
    assert any(command[1] == "sync" and "--locked" in command for command in commands)
    assert any(command[1] == "ci" and "--prefix" in command for command in commands)
    assert dict(os.environ) == original_environment
    manifest = json.loads((root / "setup.json").read_text())
    assert manifest["simulator_started"] is False
    assert manifest["runtime_requirements"] == setup.runtime_requirements()


def test_setup_refuses_wrong_python_before_download(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    monkeypatch.setattr(setup, "_run", lambda argv, **kwargs: "v24.0.0" if argv[0] == "node" else "[3, 12, 0]")
    monkeypatch.setattr(setup, "_checkout", lambda *args: pytest.fail("must not download"))
    with pytest.raises(setup.LocalSetupError, match="Python 3.11"):
        setup.setup(tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_missing_or_old_node_is_actionable_before_download(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None if name == "npm" else name)
    with pytest.raises(setup.LocalSetupError, match="Node.js 20"):
        setup.setup(tmp_path / "new")
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    monkeypatch.setattr(setup, "_run", lambda *args, **kwargs: "v18.0.0")
    with pytest.raises(setup.LocalSetupError, match="found v18"):
        setup.setup(tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_checkout_never_resets_user_modified_or_wrong_commit(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    (path / "kept").write_text("owned")
    subprocess.run(["git", "-C", str(path), "add", "kept"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    with pytest.raises(setup.LocalSetupError, match="wrong-revision"):
        setup._checkout(path, "https://must-not-fetch.invalid", setup.CODE_REVISION)
    assert (path / "kept").read_text() == "owned"


def test_checkout_can_resume_only_empty_same_origin_prefix(tmp_path, monkeypatch):
    commands = []
    path = tmp_path / "partial"
    (path / ".git").mkdir(parents=True)
    def git(root, *args, **kwargs):
        commands.append(args)
        if args == ("rev-parse", "HEAD"):
            raise setup.LocalSetupError("unborn HEAD")
        if args == ("remote", "get-url", "origin"):
            return setup.CODE_URL
        return ""
    monkeypatch.setattr(setup, "_git", git)
    setup._checkout(path, setup.CODE_URL, setup.CODE_REVISION)
    assert ("fetch", "--depth", "1", "origin", setup.CODE_REVISION) in commands
    assert ("checkout", "--detach", setup.CODE_REVISION) in commands


def test_checkout_and_runtime_symlinks_are_refused(tmp_path, monkeypatch, linux):
    other = tmp_path / "other"
    other.mkdir()
    link = tmp_path / "link"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(setup.LocalSetupError, match="symlinked source"):
        setup._checkout(link, setup.CODE_URL, setup.CODE_REVISION)


def test_setup_cli_failure_is_structured_and_never_claims_ready(monkeypatch, capsys):
    monkeypatch.setattr(setup, "setup", lambda *args, **kwargs: (_ for _ in ()).throw(setup.LocalSetupError("missing uv")))
    assert setup.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result == {"ready": False, "error": "missing uv"}
