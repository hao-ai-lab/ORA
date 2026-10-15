"""``ora setup --env``: recipes, idempotence and ``.env`` ownership, without downloads or a GPU."""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ora.orchestrator import ora_env_file, ora_gpu_recipes as recipes, ora_gpu_setup as setup


def test_pins_equal_their_canonical_modules():
    from ora.agent.model_services import resident
    from ora.environment.backends.robotwin import vendored_install
    from ora.environment.benchmarks.libero_pro import LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION
    from ora.orchestrator import robotwin_evaluation

    assert recipes.LIBERO_CODE[1] == LIBERO_PRO_CODE_REVISION
    assert recipes.LIBERO_DATA[1] == LIBERO_PRO_DATA_REVISION
    assert recipes.ROBOTWIN[1] == vendored_install.UPSTREAM_REVISION
    assert recipes.ROBOTWIN_INSTALL_REVISION == vendored_install.INSTALL_REVISION == robotwin_evaluation.REQUIRED_REVISION
    assert recipes.CAPX[1] == resident.CAPX_SOURCE_COMMIT
    assert (recipes.SAM3_SOURCE_COMMIT, recipes.SAM3_SHA256) == (resident.SAM3_SOURCE_COMMIT,
                                                                 resident.SAM3_CHECKPOINT_SHA256)
    assert (recipes.CONTACT_SOURCE_COMMIT, recipes.CONTACT_CHECKPOINT_SHA256, recipes.CONTACT_CONFIG_SHA256) == (
        resident.CONTACT_SOURCE_COMMIT, resident.CONTACT_CHECKPOINT_SHA256, resident.CONTACT_CONFIG_SHA256)


def test_environments_pull_in_only_what_they_run_on():
    assert recipes.components_for(["perception"]) == ["terminal-ui", "perception"]
    # The LIBERO planner reuses RoboTwin's CuRobo interpreter, not its 30 GB of assets.
    assert recipes.components_for(["libero"]) == ["terminal-ui", "perception", "robotwin-python", "libero"]
    assert recipes.components_for(["robotwin", "libero"]) == [
        "terminal-ui", "perception", "robotwin-python", "robotwin-assets", "libero"]
    assert recipes.components_for(["robodojo"]) == ["terminal-ui", "perception", "robodojo"]


def _recording(tmp_path, **kwargs):
    commands = []

    def runner(argv, *, cwd=None, env=None, timeout=None):
        commands.append({"argv": argv, "cwd": cwd, "env": dict(env or {})})
        if argv[:2] == ["uv", "venv"]:
            target = Path(argv[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin/python").write_text("")
            version = argv[argv.index("--python") + 1]
            (target / "pyvenv.cfg").write_text(f"version_info = {version}.4\n")

    context = setup.Context(prefix=tmp_path / "prefix", runner=runner, environ={"PATH": "/usr/bin"}, **kwargs)
    return context, commands


def test_every_simulator_environment_installs_cu128_torch(tmp_path):
    context, commands = _recording(tmp_path)
    steps = [step for component in ("perception", "robotwin-python", "libero")
             for step in recipes.BUILDERS[component](context) if step.name == "torch"]
    setup.execute(context, steps, stream=io.StringIO())
    installs = [command["argv"] for command in commands]
    assert all(argv[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"] for argv in installs)
    assert [argv[5:7] for argv in installs] == [["torch==2.8.0", "torchvision==0.23.0"],
                                                ["torch==2.7.0", "torchvision==0.22.0"],
                                                ["torch==2.7.0", "torchvision==0.22.0"]]


def test_robotwin_requirements_drop_only_torch_and_torchvision(tmp_path):
    context, commands = _recording(tmp_path)
    upstream = context.path("sources/RoboTwin/script/requirements.txt")
    upstream.parent.mkdir(parents=True)
    upstream.write_text("torch==2.4.1\ntorchvision\nsapien==3.0.0b1\nmplib==0.2.1\ntorchaudio==2.4.1\n")
    step = next(step for step in recipes.robotwin_python(context) if step.name == "requirements")
    setup.execute(context, [step], stream=io.StringIO())
    written = context.path("manifests/ora-setup/robotwin-requirements.txt").read_text().split()
    assert written == ["sapien==3.0.0b1", "mplib==0.2.1", "torchaudio==2.4.1"]
    assert commands[-1]["argv"][-2:] == ["numpy==1.26.4", "setuptools==69.5.1"]


def test_curobo_is_built_for_this_gpu_and_rebuilt_when_the_arch_changes(tmp_path):
    cuda = tmp_path / "cuda-12.8"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").write_text("")
    context, commands = _recording(tmp_path, cuda_home=cuda, cuda_arch="12.0")
    python = context.path("envs/robotwin/bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    step = next(step for step in recipes.robotwin_python(context) if step.name == "curobo-build")
    setup.execute(context, [step], stream=io.StringIO())
    build = commands[-1]
    assert build["argv"][-3:] == ["-e", str(context.path("sources/curobo")), "--no-build-isolation"]
    assert build["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0+PTX"
    assert build["env"]["CUDA_HOME"] == str(cuda)
    assert build["env"]["PATH"].startswith(f"{cuda / 'bin'}:")
    assert setup.step_done(context, step)
    hopper = setup.Context(prefix=context.prefix, runner=context.runner, cuda_home=cuda, cuda_arch="9.0")
    assert not setup.step_done(hopper, step)


def test_cuda_arch_comes_from_nvidia_smi_and_is_validated(tmp_path, monkeypatch):
    context = setup.Context(prefix=tmp_path)
    monkeypatch.setattr(setup.Context, "output", lambda self, argv, **kwargs: "12.0\n12.0")
    assert recipes.cuda_arch(context) == "12.0+PTX"
    monkeypatch.setattr(setup.Context, "output", lambda self, argv, **kwargs: "")
    with pytest.raises(setup.GpuSetupError, match="--cuda-arch"):
        recipes.cuda_arch(context)


def _repository(path: Path) -> str:
    path.mkdir()
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@example.com"],
                 ["git", "config", "user.name", "t"]):
        subprocess.run(argv, cwd=path, check=True)
    (path / "file.txt").write_text("pinned\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_clone_step_pins_a_revision_skips_when_done_and_never_resets(tmp_path):
    revision = _repository(tmp_path / "upstream")
    context = setup.Context(prefix=tmp_path / "prefix")
    step = recipes.clone_step("demo", "repo", "sources/repo", str(tmp_path / "upstream"), revision)
    assert not setup.step_done(context, step)
    setup.execute(context, [step], stream=io.StringIO())
    checkout = context.path("sources/repo")
    assert subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() == revision
    output = io.StringIO()
    setup.execute(context, [step], stream=output)
    assert "already done" in output.getvalue()
    (checkout / "file.txt").write_text("edited\n")
    assert not setup.step_done(context, step)
    with pytest.raises(setup.GpuSetupError, match="not a clean checkout"):
        setup.execute(context, [step], stream=io.StringIO())
    assert (checkout / "file.txt").read_text() == "edited\n"


def test_sam3_download_is_verified_before_it_is_kept(tmp_path, monkeypatch):
    context, commands = _recording(tmp_path)

    def fake_curl(argv, **kwargs):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"not the checkpoint")

    context.runner = fake_curl
    step = next(step for step in recipes.perception(context) if step.name == "sam3")
    with pytest.raises(setup.GpuSetupError, match="differs from the pinned"):
        setup.execute(context, [step], stream=io.StringIO())
    assert commands[0][-1] == recipes.SAM3_URL
    assert not context.path("models/sam3.pt").exists()
    assert not context.path("models/sam3.pt.partial").exists()
    monkeypatch.setattr(recipes, "SAM3_SHA256", recipes._sha256(Path(__file__)))
    monkeypatch.setattr(recipes, "_sha256", lambda path: recipes.SAM3_SHA256)
    setup.execute(context, [step], stream=io.StringIO())
    assert context.path("models/sam3.pt").is_file() and setup.step_done(context, step)


def test_robodojo_stages_always_name_the_prefix_root(tmp_path):
    context, commands = _recording(tmp_path)
    setup.execute(context, recipes.robodojo(context), stream=io.StringIO())
    stages = [command["argv"][2] for command in commands]
    assert sorted(stages[:3]) == ["assets", "sdk", "sources"]
    assert stages[3:] == ["resolve", "packages", "bind-assets", "graphics-compat", "doctor", "manifest"]
    root = str(context.path("robodojo-runtime"))
    assert all(command["argv"][-2:] == ["--root", root] for command in commands)
    # The real graphics-compat stage writes this receipt; with it the rerun only re-checks.
    receipt = context.path("robodojo-runtime/graphics-compat/installation.json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    commands.clear()
    setup.execute(context, recipes.robodojo(context), stream=io.StringIO())
    assert [command["argv"][2] for command in commands] == ["doctor", "manifest"]


def test_env_keys_cover_each_environment_and_hold_no_credential_or_host_marker(tmp_path):
    context = setup.Context(prefix=tmp_path)
    keys = recipes.env_keys(context, recipes.COMPONENTS)
    for name in ("B2_SAM_CHECKPOINT", "B2_GRASP_CONFIG", "ROBOTWIN_PYTHON", "CUROBO_PYTHON", "LIBERO_PRO_ROOT",
                 "ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT", "ROBODOJO_PYTHON", "ROBODOJO_RUNTIME_ROOT"):
        assert keys[name].startswith(str(tmp_path)), name
    assert keys["CUROBO_PYTHON"] == keys["ROBOTWIN_PYTHON"]
    assert keys["ROBODOJO_ROOT"] == f"{keys['ROBODOJO_RUNTIME_ROOT']}/source/RoboDojo"
    assert not [key for key in keys if "KEY" in key or "TOKEN" in key]
    assert "ORA_EXECUTION_HOST" not in keys


def test_env_file_update_owns_keys_and_removes_blanking_credential_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# operator notes\nDEEPSEEK_LOW_API_KEY=\nDEEPSEEK_API_KEY=keep-me\nB2_SAM_PORT=8124\n"
                    "ROBOTWIN_ROOT=/old\n")
    os.chmod(path, 0o640)
    changed = ora_env_file.update(path, {"ROBOTWIN_ROOT": "/new root", "B2_SAM_GPU": "0"},
                                  defaults={"DEEPSEEK_BASE_URL": "https://api.deepseek.com", "B2_SAM_PORT": "8114"})
    text = path.read_text()
    assert "DEEPSEEK_LOW_API_KEY=" not in text and "DEEPSEEK_API_KEY=keep-me" in text
    assert "B2_SAM_PORT=8124" in text and "# operator notes" in text
    assert ora_env_file.read(path)["ROBOTWIN_ROOT"] == "/new root"
    assert set(changed) == {"ROBOTWIN_ROOT", "B2_SAM_GPU", "DEEPSEEK_BASE_URL"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert ora_env_file.update(path, {"ROBOTWIN_ROOT": "/new root", "B2_SAM_GPU": "0"}) == {}
    with pytest.raises(ValueError, match="never writes credentials"):
        ora_env_file.update(path, {"DEEPSEEK_LOW_API_KEY": "secret"})
    fresh = tmp_path / "fresh.env"
    ora_env_file.update(fresh, {"A": "1"})
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600


def test_env_file_load_never_overrides_or_blanks_the_environment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_LOW_API_KEY=\nROBOTWIN_ROOT=/from-file\nMUJOCO_GL=egl\n")
    environ = {"DEEPSEEK_LOW_API_KEY": "from-shell", "MUJOCO_GL": "osmesa"}
    assert ora_env_file.load_into(environ, path) == ["ROBOTWIN_ROOT"]
    assert environ == {"DEEPSEEK_LOW_API_KEY": "from-shell", "MUJOCO_GL": "osmesa", "ROBOTWIN_ROOT": "/from-file"}


def test_cli_dry_run_changes_nothing_and_refuses_a_prefix_in_the_checkout(tmp_path, capsys):
    env_file = tmp_path / ".env"
    code = setup.main(["--env", "perception", "--prefix", str(tmp_path / "gpu"), "--env-file", str(env_file),
                       "--dry-run", "--json"])
    assert code == 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["components"] == ["terminal-ui", "perception"]
    assert {item["status"] for item in report["steps"]} <= {"pending", "done"}
    assert report["env_keys_written"] == []
    assert not env_file.exists() and not (tmp_path / "gpu").exists()
    code = setup.main(["--env", "perception", "--prefix", str(setup.PROJECT_ROOT / "gpu-inside"),
                       "--env-file", str(env_file), "--dry-run"])
    assert code == 1
    assert "inside the Harness checkout" in capsys.readouterr().err


def test_ora_cli_routes_setup_env_to_the_gpu_installer(monkeypatch):
    from ora.orchestrator import ora_cli

    seen = []

    def gpu_main(argv):
        seen.append(argv)
        return 0

    monkeypatch.setattr(setup, "main", gpu_main)
    assert ora_cli.main(["setup", "--env", "robotwin", "--dry-run"]) == 0
    assert seen == [["--env", "robotwin", "--dry-run"]]


def test_a_recreated_environment_invalidates_its_package_markers(tmp_path):
    context, commands = _recording(tmp_path)
    steps = [step for step in recipes.libero(context) if step.name in {"venv-libero", "torch"}]
    setup.execute(context, steps, stream=io.StringIO())
    assert setup.step_done(context, steps[1])
    import shutil
    shutil.rmtree(context.path("envs/libero"))
    commands.clear()
    setup.execute(context, steps, stream=io.StringIO())
    assert [command["argv"][:2] for command in commands] == [["uv", "venv"], ["uv", "pip"]]


def test_graphics_compat_is_done_by_its_receipt_and_never_replaced(tmp_path):
    context, commands = _recording(tmp_path)
    step = next(step for step in recipes.robodojo(context) if step.name == "graphics-compat")
    assert not setup.step_done(context, step)
    receipt = context.path("robodojo-runtime/graphics-compat/installation.json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    setup.execute(context, [step], stream=io.StringIO())
    assert commands == []


def test_a_failed_git_status_is_not_a_clean_checkout(tmp_path, monkeypatch):
    revision = _repository(tmp_path / "upstream")
    context = setup.Context(prefix=tmp_path / "prefix")
    step = recipes.clone_step("demo", "repo", "sources/repo", str(tmp_path / "upstream"), revision)
    setup.execute(context, [step], stream=io.StringIO())
    real = setup.Context.probe
    monkeypatch.setattr(setup.Context, "probe", lambda self, argv, **kwargs: None if "status" in argv
                        else real(self, argv, **kwargs))
    assert not setup.step_done(context, step)


def test_robotwin_assets_unpack_with_unzip_after_a_path_check(tmp_path):
    import zipfile

    context, commands = _recording(tmp_path)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in recipes.ROBOTWIN_ASSET_ZIPS:
        with zipfile.ZipFile(downloads / name, "w") as archive:
            archive.writestr(f"{name[:-4]}/file.txt", "x")
    manifest = context.path("manifests/robotwin-assets.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"directory": str(downloads), "sha256": dict.fromkeys(recipes.ROBOTWIN_ASSET_ZIPS, "0")}))
    step = next(step for step in recipes.robotwin_assets(context) if step.name == "unpack")
    setup.execute(context, [step], stream=io.StringIO())
    assert [command["argv"][:3] for command in commands] == [["unzip", "-o", "-q"]] * 3
    with zipfile.ZipFile(downloads / "objects.zip", "w") as archive:
        archive.writestr("../escape.txt", "x")
    manifest.write_text(json.dumps({"directory": str(downloads), "sha256": {"changed": "1"}}))
    with pytest.raises(setup.GpuSetupError, match="escapes"):
        setup.execute(context, [step], stream=io.StringIO())
