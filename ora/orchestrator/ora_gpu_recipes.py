"""The single-GPU installation recipes behind ``ora setup --env``.

Every command here is a step of ``docs/getting-started/single-gpu.md``
§9 as verified on 2026-10-14: cu128 Torch in every simulator environment,
RoboTwin's requirements without its torch/torchvision pins, CuRobo built for
this GPU's compute capability, SAM3 from ModelScope with the pinned SHA-256
and the ``setup-robodojo`` stages including ``graphics-compat``. Pins are
duplicated from their canonical modules so this file imports nothing heavy;
``tests/test_ora_gpu_setup.py`` keeps them equal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ora_gpu_setup import Context, GpuSetupError, Step

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
LIBERO_CODE = ("https://github.com/Zxy-MLlab/LIBERO-PRO.git", "2b910b5b5f53016bef9907632f6f840f1ce2229c")
LIBERO_DATA = ("https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro", "c86fc3b8293185a6f373677018ff3e37f8391602")
ROBOTWIN = ("https://github.com/RoboTwin-Platform/RoboTwin.git", "c3ddfa8b97d5519efa828b075999bd0006778e5e")
#: The vendored install's deterministic commit on top of ``ROBOTWIN``.
ROBOTWIN_INSTALL_REVISION = "cf4872cc90538fd4bca761c8db912d8f65e49f2b"
CUROBO = ("https://github.com/NVlabs/curobo.git", "d64c4b005459db10c5dd867d8b30a87d5bda9bdb")
CAPX = ("https://github.com/capgym/cap-x.git", "53e9966d7a8e2fa7494676772bccc35280f5c0ed")
SAM3_SOURCE_COMMIT = "6fe87d64a5beb9084923d7a9e002741178635b09"
CONTACT_SOURCE_COMMIT = "da3dcfb2f53e43b186083ee4a9d1e232f73efc98"
SAM3_URL = "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CONTACT_CHECKPOINT_SHA256 = "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
CONTACT_CONFIG_SHA256 = "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5"
ROBOTWIN_ASSET_REPOSITORY = "TianxingChen/RoboTwin2.0"
ROBOTWIN_ASSET_ZIPS = ("background_texture.zip", "embodiments.zip", "objects.zip")
HF_HUB = "huggingface-hub==1.27.0"

#: Install order; an environment pulls in the components it runs on.
COMPONENTS = ("terminal-ui", "perception", "robotwin-python", "robotwin-assets", "libero", "robodojo")
REQUIRES = {
    "perception": ("terminal-ui", "perception"),
    # The LIBERO planner worker reuses RoboTwin's CuRobo interpreter.
    "libero": ("terminal-ui", "perception", "robotwin-python", "libero"),
    "robotwin": ("terminal-ui", "perception", "robotwin-python", "robotwin-assets"),
    "robodojo": ("terminal-ui", "perception", "robodojo"),
}
#: Upper bounds from the verification machine, in GB, including caches and zips.
DISK_GIB = {"terminal-ui": 1, "perception": 12, "robotwin-python": 12, "robotwin-assets": 32,
            "libero": 10, "robodojo": 90}

_GRASP = "sources/cap-x/capx/third_party/contact_graspnet_pytorch"
_GRASP_CHECKPOINT = f"{_GRASP}/checkpoints/contact_graspnet/checkpoints/model.pt"
_GRASP_CONFIG = f"{_GRASP}/checkpoints/contact_graspnet/config.yaml"


def components_for(environments: Sequence[str]) -> list[str]:
    wanted = {component for name in environments for component in REQUIRES[name]}
    return [component for component in COMPONENTS if component in wanted]


def disk_gib(components: Sequence[str]) -> int:
    return sum(DISK_GIB[component] for component in components)


def _python(context: Context, name: str) -> Path:
    return context.path(f"envs/{name}/bin/python")


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _file_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


# --- toolchain ----------------------------------------------------------------------------------

def _cuda_home(context: Context) -> Path:
    candidates = [context.cuda_home] if context.cuda_home else [
        Path(context.environ["CUDA_HOME"]) if context.environ.get("CUDA_HOME") else None,
        Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda")]
    for candidate in candidates:
        if candidate is not None and (candidate / "bin/nvcc").is_file():
            return candidate
    raise GpuSetupError("CuRobo needs the CUDA 12.8 toolkit (nvcc); the driver alone is not enough. "
                        "Install it or pass --cuda-home /path/to/cuda-12.8")


def cuda_arch(context: Context) -> str:
    """``TORCH_CUDA_ARCH_LIST`` for this GPU, e.g. ``12.0+PTX``."""
    value = context.cuda_arch or context.output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).split("\n")[0].strip()
    if not re.fullmatch(r"\d+\.\d+", value or ""):
        raise GpuSetupError("cannot read the GPU compute capability from nvidia-smi; pass --cuda-arch, e.g. 12.0")
    return f"{value}+PTX"


def _system_vulkan_loader(context: Context) -> bool:
    import ctypes

    try:
        ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return False
    return True


def check_toolchain(context: Context, components: Sequence[str]) -> None:
    """Refuse before any download when a required tool is missing."""
    missing = [tool for tool in ("git", "uv") if shutil.which(tool) is None]
    if "perception" in components and shutil.which("curl") is None:
        missing.append("curl")
    if "robotwin-assets" in components and shutil.which("unzip") is None:
        missing.append("unzip")
    if "terminal-ui" in components:
        missing += [tool for tool in ("node", "npm") if shutil.which(tool) is None]
    if "robodojo" in components:
        missing += [tool for tool, found in (("git-lfs", shutil.which("git-lfs")),
                                             ("a C++ compiler (c++)", shutil.which("c++") or shutil.which("g++")))
                    if found is None]
    if missing:
        raise GpuSetupError("missing on PATH: " + ", ".join(missing)
                            + " (docs/getting-started/single-gpu.md §2 installs uv, Node 22 and git-lfs "
                            "into ~/.local without sudo)")
    if "robodojo" in components and not _system_vulkan_loader(context):
        # Not fatal: Isaac Sim ships a loader that only exists after the sdk stage.
        print("[ora setup] warning: no system Vulkan loader (libvulkan.so.1). RoboDojo's graphics-compat "
              "check and workers need one: install libvulkan1, or after the sdk stage put Isaac Sim's bundled "
              "loader directory on LD_LIBRARY_PATH and rerun (docs/getting-started/single-gpu.md §8)",
              flush=True)
    if shutil.which("nvidia-smi") is None:
        raise GpuSetupError("nvidia-smi is unavailable; the single-GPU path needs an NVIDIA driver (R570+)")
    if "terminal-ui" in components:
        version = context.output(["node", "--version"])
        match = re.fullmatch(r"v(\d+)\.\d+\.\d+", version)
        if not match or int(match.group(1)) < 20:
            raise GpuSetupError(f"terminal UI requires Node.js 20 or newer; found {version or 'none'}")
    if "robotwin-python" in components:
        _cuda_home(context)
        cuda_arch(context)


# --- shared step builders -----------------------------------------------------------------------

def _head(context: Context, path: Path) -> str:
    return context.output(["git", "-C", path, "rev-parse", "HEAD"])


def _clean(context: Context, path: Path) -> bool:
    # A failed or timed-out status is not evidence of a clean tree.
    return (path / ".git").exists() and context.probe(
        ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], timeout=600) == ""


def _at(context: Context, path: Path, revisions: Sequence[str]) -> bool:
    return (path / ".git").exists() and _head(context, path) in revisions and _clean(context, path)


def clone_step(component: str, name: str, relative: str, url: str, revision: str, *,
               accepted: Sequence[str] = (), lfs_skip: bool = False) -> Step:
    """Clone ``url`` and detach at ``revision`` (or accept a later pinned state).

    An existing checkout at another revision or with tracked edits is refused,
    never reset.
    """
    revisions = (revision, *accepted)

    def run(context: Context) -> None:
        path = context.path(relative)
        if path.exists():
            raise GpuSetupError(f"{path} exists but is not a clean checkout of {revision[:12]}; "
                                "move it away or fix it, then rerun ora setup")
        path.parent.mkdir(parents=True, exist_ok=True)
        env = {"GIT_LFS_SKIP_SMUDGE": "1"} if lfs_skip else None
        clone = ["git", "clone", *(["--no-checkout"] if lfs_skip else []), url, path]
        context.run(clone, env=env, timeout=7200)
        context.run(["git", "-C", path, "checkout", "--detach", revision], env=env, timeout=3600)
        # Record stat data once so later read-only status checks stay fast.
        context.run(["git", "-C", path, "update-index", "-q", "--refresh"], timeout=3600)

    return Step(component, name, f"clone {Path(relative).name} @ {revision[:12]}", run,
                done=lambda context: _at(context, context.path(relative), revisions), marker=False)


def venv_step(component: str, name: str, python_version: str, *, seed: bool = True) -> Step:
    def run(context: Context) -> None:
        context.run(["uv", "venv", "--python", python_version, *(["--seed"] if seed else []),
                     context.path(f"envs/{name}")])

    def done(context: Context) -> bool:
        config = context.path(f"envs/{name}/pyvenv.cfg")
        pattern = rf"^(version|version_info)\s*=\s*{re.escape(python_version)}(\.|\s*$)"
        return config.is_file() and re.search(pattern, config.read_text(), re.M) is not None

    return Step(component, f"venv-{name}", f"Python {python_version} environment envs/{name}", run,
                done=done, marker=False)


def pip_step(component: str, name: str, env: str, title: str, arguments: Sequence[str], *,
             recipe_extra: Mapping[str, Any] | None = None, environment=None) -> Step:
    """``uv pip install --python <env> <arguments>``; done when the same recipe was installed."""
    def run(context: Context) -> None:
        context.run(["uv", "pip", "install", "--python", _python(context, env), *arguments],
                    env=environment(context) if environment else None)

    def recipe(context: Context) -> Mapping[str, Any]:
        extra = recipe_extra(context) if callable(recipe_extra) else (recipe_extra or {})
        return {"python": str(_python(context, env)), "environment": _venv_identity(context, env),
                "arguments": list(arguments), **extra}

    return Step(component, name, title, run, recipe=recipe,
                done=lambda context: _python(context, env).is_file())


def _venv_identity(context: Context, name: str) -> list[int] | None:
    """Binds package markers to one virtualenv instance: a recreated env reinstalls."""
    try:
        stat = context.path(f"envs/{name}/pyvenv.cfg").stat()
    except OSError:
        return None
    return [stat.st_ino, stat.st_mtime_ns]


def torch_step(component: str, env: str, torch: str, torchvision: str) -> Step:
    return pip_step(component, "torch", env, f"Torch {torch} / torchvision {torchvision} (cu128)",
                    [f"torch=={torch}", f"torchvision=={torchvision}", "--index-url", TORCH_INDEX])


# --- components ---------------------------------------------------------------------------------

def terminal_ui(context: Context) -> list[Step]:
    lock = context.project_root / "ora-tui/package-lock.json"

    def run(inner: Context) -> None:
        inner.run(["npm", "ci", "--prefix", inner.project_root / "ora-tui", "--no-audit", "--no-fund"],
                  timeout=1200)

    return [Step("terminal-ui", "npm-ci", "ORA terminal UI (npm ci)", run,
                 recipe=lambda inner: {"package_lock_sha256": _file_sha256(lock)},
                 done=lambda inner: (inner.project_root / "ora-tui/node_modules").is_dir())]


def perception(context: Context) -> list[Step]:
    url, revision = CAPX
    capx = "sources/cap-x"

    def submodules_done(inner: Context) -> bool:
        root = inner.path(capx)
        return (_head(inner, root / "capx/third_party/sam3") == SAM3_SOURCE_COMMIT
                and _head(inner, root / "capx/third_party/contact_graspnet_pytorch") == CONTACT_SOURCE_COMMIT)

    def submodules(inner: Context) -> None:
        inner.run(["git", "-C", inner.path(capx), "submodule", "update", "--init", "--recursive",
                   "capx/third_party/sam3", "capx/third_party/contact_graspnet_pytorch"], timeout=3600)

    def sam3_done(inner: Context) -> bool:
        path = inner.path("models/sam3.pt")
        try:
            saved = json.loads(inner.path("manifests/ora-setup/sam3.json").read_text())
            stat = path.stat()
        except (OSError, ValueError):
            return False
        return saved == {"sha256": SAM3_SHA256, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def sam3(inner: Context) -> None:
        target = inner.path("models/sam3.pt")
        partial = target.with_name("sam3.pt.partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.is_file() and _sha256(target) == SAM3_SHA256):
            inner.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", partial, SAM3_URL], timeout=7200)
            actual = _sha256(partial)
            if actual != SAM3_SHA256:
                partial.unlink()
                raise GpuSetupError(f"sam3.pt SHA-256 {actual} differs from the pinned {SAM3_SHA256}")
            partial.replace(target)
        stat = target.stat()
        record = inner.path("manifests/ora-setup/sam3.json")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"sha256": SAM3_SHA256, "size": stat.st_size,
                                      "mtime_ns": stat.st_mtime_ns}) + "\n")

    def verify_grasp(inner: Context) -> None:
        for relative, expected in ((_GRASP_CHECKPOINT, CONTACT_CHECKPOINT_SHA256),
                                   (_GRASP_CONFIG, CONTACT_CONFIG_SHA256)):
            actual = _file_sha256(inner.path(relative))
            if actual != expected:
                raise GpuSetupError(f"{inner.path(relative)} SHA-256 is {actual}, expected {expected}")

    requirements = context.project_root / "ops/x86/requirements-perception.txt"
    return [
        clone_step("perception", "cap-x", capx, url, revision),
        Step("perception", "submodules", "SAM3 and Contact-GraspNet submodules", submodules,
             done=submodules_done, marker=False),
        venv_step("perception", "perception", "3.11"),
        torch_step("perception", "perception", "2.8.0", "0.23.0"),
        pip_step("perception", "requirements", "perception", "ops/x86/requirements-perception.txt",
                 ["-r", requirements],
                 recipe_extra=lambda inner: {"requirements_sha256": _file_sha256(requirements)}),
        Step("perception", "sam3", "sam3.pt from ModelScope (3.45 GB, SHA-256 pinned)", sam3,
             done=sam3_done, marker=False),
        Step("perception", "verify-graspnet", "verify Contact-GraspNet checkpoint and config", verify_grasp,
             marker=False),
    ]


def _robotwin_requirements(context: Context) -> Path:
    return context.path("manifests/ora-setup/robotwin-requirements.txt")


def robotwin_python(context: Context) -> list[Step]:
    url, revision = ROBOTWIN
    root = "sources/RoboTwin"
    upstream = lambda inner: inner.path(f"{root}/script/requirements.txt")

    def vendored(inner: Context) -> None:
        inner.run([inner.python, "-m", "ora.environment.backends.robotwin.vendored_install",
                   "--robotwin-root", inner.path(root), "--expected-revision", ROBOTWIN_INSTALL_REVISION],
                  cwd=inner.project_root, env={"PYTHONPATH": str(inner.project_root)})

    def requirements(inner: Context) -> None:
        # Upstream pins torch==2.4.1 and leaves torchvision unpinned; both stay with the cu128 step.
        lines = [line for line in upstream(inner).read_text().splitlines()
                 if not re.match(r"^\s*(torch|torchvision)\s*([=<>!~ ]|$)", line)]
        target = _robotwin_requirements(inner)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
        inner.run(["uv", "pip", "install", "--python", _python(inner, "robotwin"), "-r", target,
                   "numpy==1.26.4", "setuptools==69.5.1"])

    def curobo_environment(inner: Context) -> dict[str, str]:
        home = _cuda_home(inner)
        environment = {"CUDA_HOME": str(home), "TORCH_CUDA_ARCH_LIST": cuda_arch(inner),
                       "PATH": f"{home / 'bin'}:{inner.environ.get('PATH', '')}"}
        # A conda/pip toolkit keeps its headers under targets/; nvcc finds them, the host compiler does not.
        targets = home / "targets/x86_64-linux/include"
        if not (home / "include/cuda_runtime_api.h").is_file() and (targets / "cuda_runtime_api.h").is_file():
            environment["CPATH"] = os.pathsep.join(filter(None, (str(targets), inner.environ.get("CPATH", ""))))
        return environment

    def curobo_recipe(inner: Context) -> Mapping[str, Any]:
        return {"curobo": CUROBO[1], "arch": cuda_arch(inner), "cuda_home": str(_cuda_home(inner))}

    curobo_url, curobo_revision = CUROBO
    return [
        clone_step("robotwin-python", "robotwin", root, url, revision, accepted=(ROBOTWIN_INSTALL_REVISION,)),
        Step("robotwin-python", "vendored-install", f"vendored planner install (HEAD {ROBOTWIN_INSTALL_REVISION[:8]})",
             vendored, done=lambda inner: _at(inner, inner.path(root), (ROBOTWIN_INSTALL_REVISION,)), marker=False),
        venv_step("robotwin-python", "robotwin", "3.10"),
        torch_step("robotwin-python", "robotwin", "2.7.0", "0.22.0"),
        Step("robotwin-python", "requirements", "RoboTwin requirements without torch/torchvision", requirements,
             recipe=lambda inner: {"requirements_sha256": _file_sha256(upstream(inner)),
                                   "environment": _venv_identity(inner, "robotwin"),
                                   "extra": ["numpy==1.26.4", "setuptools==69.5.1"]},
             done=lambda inner: _python(inner, "robotwin").is_file()),
        clone_step("robotwin-python", "curobo", "sources/curobo", curobo_url, curobo_revision),
        pip_step("robotwin-python", "curobo-build", "robotwin", "build CuRobo for this GPU (~6 min)",
                 ["-e", str(context.path("sources/curobo")), "--no-build-isolation"],
                 recipe_extra=curobo_recipe, environment=curobo_environment),
        pip_step("robotwin-python", "warp", "robotwin", "warp-lang 1.12.0", ["warp-lang==1.12.0"]),
    ]


_DOWNLOAD_ASSETS = """import hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root, repo, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(',')
revision = HfApi().dataset_info(repo).sha
folder = root / 'downloads' / ('robotwin-' + revision)
snapshot_download(repo_id=repo, repo_type='dataset', revision=revision, allow_patterns=names, local_dir=folder)
files = {}
for name in names:
    digest = hashlib.sha256()
    with (folder / name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    files[name] = digest.hexdigest()
(root / 'manifests' / 'robotwin-assets.json').write_text(json.dumps(
    {'repository': repo, 'revision': revision, 'directory': str(folder), 'sha256': files}, indent=2) + '\\n')
"""


def _asset_manifest(context: Context) -> dict[str, Any] | None:
    try:
        return json.loads(context.path("manifests/robotwin-assets.json").read_text())
    except (OSError, ValueError):
        return None


def robotwin_assets(context: Context) -> list[Step]:
    root = "sources/RoboTwin"

    def download_done(inner: Context) -> bool:
        manifest = _asset_manifest(inner)
        return bool(manifest) and set(manifest.get("sha256", {})) == set(ROBOTWIN_ASSET_ZIPS) and all(
            (Path(manifest["directory"]) / name).is_file() for name in ROBOTWIN_ASSET_ZIPS)

    def download(inner: Context) -> None:
        inner.path("manifests").mkdir(parents=True, exist_ok=True)
        inner.run([_python(inner, "downloads"), "-c", _DOWNLOAD_ASSETS, inner.prefix,
                   ROBOTWIN_ASSET_REPOSITORY, ",".join(ROBOTWIN_ASSET_ZIPS)], timeout=14400)

    def unpack(inner: Context) -> None:
        manifest = _asset_manifest(inner) or {}
        target = inner.path(f"{root}/assets")
        for name in ROBOTWIN_ASSET_ZIPS:
            archive_path = Path(manifest["directory"]) / name
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    resolved = (target / member).resolve()
                    if not resolved.is_relative_to(target.resolve()):
                        raise GpuSetupError(f"{name} member {member!r} escapes the assets directory")
            # unzip, as the guide: embodiments.zip carries executable files.
            inner.run(["unzip", "-o", "-q", archive_path, "-d", target], timeout=7200)

    def embodiment_paths(inner: Context) -> None:
        inner.run([_python(inner, "robotwin"), "script/update_embodiment_config_path.py"], cwd=inner.path(root))

    assets_recipe = lambda inner: {"assets": (_asset_manifest(inner) or {}).get("sha256"),
                                   "robotwin_root": str(inner.path(root)), "extractor": "unzip -o"}
    return [
        venv_step("robotwin-assets", "downloads", "3.11", seed=False),
        pip_step("robotwin-assets", "huggingface-hub", "downloads", HF_HUB, [HF_HUB]),
        Step("robotwin-assets", "download", "RoboTwin 2.0 asset zips (~14 GB, no token)", download,
             done=download_done, marker=False),
        Step("robotwin-assets", "unpack", "unpack assets into RoboTwin/assets (~16 GB)", unpack,
             recipe=assets_recipe),
        Step("robotwin-assets", "embodiment-paths", "update_embodiment_config_path.py", embodiment_paths,
             recipe=assets_recipe),
    ]


def libero(context: Context) -> list[Step]:
    code_url, code_revision = LIBERO_CODE
    data_url, data_revision = LIBERO_DATA
    requirements = context.project_root / "ops/x86/requirements-libero.txt"
    return [
        clone_step("libero", "libero-pro", "sources/LIBERO-PRO", code_url, code_revision),
        clone_step("libero", "libero-pro-data", "sources/LIBERO-Pro-data", data_url, data_revision, lfs_skip=True),
        venv_step("libero", "libero", "3.11"),
        torch_step("libero", "libero", "2.7.0", "0.22.0"),
        pip_step("libero", "requirements", "libero", "ops/x86/requirements-libero.txt", ["-r", requirements],
                 recipe_extra=lambda inner: {"requirements_sha256": _file_sha256(requirements)}),
    ]


ROBODOJO_INSTALLER = "ora/environment/backends/robodojo/installation/setup_runtime.py"


def robodojo_root(context: Context) -> Path:
    return context.path("robodojo-runtime")


def robodojo(context: Context) -> list[Step]:
    def stage_command(inner: Context, stage: str) -> list[Any]:
        return [inner.python, inner.project_root / ROBODOJO_INSTALLER, stage, "--root", robodojo_root(inner)]

    def downloads(inner: Context) -> None:
        import concurrent.futures

        # Sources, the 66 GB of assets and the Isaac Sim SDK are independent.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(inner.run, stage_command(inner, stage), timeout=6 * 3600,
                                   log_name=f"robodojo.{stage}") for stage in ("sources", "assets", "sdk")]
            for future in futures:
                future.result()

    def stage(name: str):
        return lambda inner: inner.run(stage_command(inner, name), timeout=7200)

    def recipe(inner: Context) -> Mapping[str, Any]:
        text = (inner.project_root / ROBODOJO_INSTALLER).read_text()
        pins = dict(re.findall(r'^(DOJO_REV|ASSET_REV|PYTHON_VERSION|PROFILES_REV) = "([^"]+)"', text, re.M))
        return {"root": str(robodojo_root(inner)), **pins}

    steps = [Step("robodojo", "downloads", "RoboDojo sources, assets (66 GB, git-lfs) and Isaac Sim SDK",
                  downloads, recipe=recipe)]
    for name, title in (("resolve", "resolve"), ("packages", "packages"), ("bind-assets", "bind assets")):
        steps.append(Step("robodojo", name, title, stage(name), recipe=recipe))
    # The stage refuses to replace its receipt, which every worker verifies at
    # start; the receipt itself is the done state (rebuild: remove graphics-compat/).
    steps.append(Step("robodojo", "graphics-compat", "graphics-compat (Vulkan Profiles layer)",
                      stage("graphics-compat"), marker=False,
                      done=lambda inner: (robodojo_root(inner) / "graphics-compat/installation.json").is_file()))
    # Cheap and read-only: always rerun so every setup ends on a fresh verdict.
    steps.append(Step("robodojo", "doctor", "doctor (expect fail=0)", stage("doctor"), marker=False))
    steps.append(Step("robodojo", "manifest", "install manifest", stage("manifest"), marker=False))
    return steps


BUILDERS = {"terminal-ui": terminal_ui, "perception": perception, "robotwin-python": robotwin_python,
            "robotwin-assets": robotwin_assets, "libero": libero, "robodojo": robodojo}


def steps_for(context: Context, components: Sequence[str]) -> list[Step]:
    return [step for component in components for step in BUILDERS[component](context)]


def env_keys(context: Context, components: Sequence[str]) -> dict[str, str]:
    """The ``.env`` keys these components own. Never a credential or the host marker."""
    keys: dict[str, str] = {}
    if "perception" in components:
        python = str(_python(context, "perception"))
        keys.update({
            "B2_SAM_PYTHON": python, "B2_SAM_ROOT": str(context.path("sources/cap-x/capx/third_party/sam3")),
            "B2_SAM_CHECKPOINT": str(context.path("models/sam3.pt")), "B2_SAM_GPU": "0",
            "B2_GRASP_PYTHON": python, "B2_GRASP_ROOT": str(context.path(_GRASP)),
            "B2_GRASP_CHECKPOINT": str(context.path(_GRASP_CHECKPOINT)),
            "B2_GRASP_CONFIG": str(context.path(_GRASP_CONFIG)), "B2_GRASP_GPU": "0",
        })
    if "robotwin-python" in components:
        python = str(_python(context, "robotwin"))
        keys.update({"ROBOTWIN_ROOT": str(context.path("sources/RoboTwin")), "ROBOTWIN_PYTHON": python,
                     "CUROBO_ROOT": str(context.path("sources/curobo")), "CUROBO_PYTHON": python})
    if "libero" in components:
        keys.update({"LIBERO_PRO_PYTHON": str(_python(context, "libero")),
                     "LIBERO_PRO_ROOT": str(context.path("sources/LIBERO-PRO")),
                     "LIBERO_PRO_DATA_ROOT": str(context.path("sources/LIBERO-Pro-data")),
                     "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    if "robodojo" in components:
        root = robodojo_root(context)
        keys.update({"ROBODOJO_RUNTIME_ROOT": str(root), "ROBODOJO_ROOT": str(root / "source/RoboDojo"),
                     "ROBODOJO_ASSET_ROOT": str(root / "assets/Assets"),
                     "ROBODOJO_PYTHON": str(root / "env/bin/python")})
    return keys


__all__ = ["BUILDERS", "COMPONENTS", "REQUIRES", "check_toolchain", "components_for", "cuda_arch",
           "disk_gib", "env_keys", "steps_for"]
