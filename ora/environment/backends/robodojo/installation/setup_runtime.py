"""Prepare isolated RoboDojo dependencies; never start a simulator or GPU job."""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import zipfile

DOJO_REV = "ee67a1468510da7624a089164402359f2afc72c8"
ASSET_REV = "a14409d7fae673c00499e01fd88b4457df6351b1"
PYTHON_VERSION = "3.11.15"
SUBMODULES = {
    "third_party/IsaacLab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
    "third_party/curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e",
    "XPolicyLab": "432f82b1758c5b1202e42a3dfe014546dbc50871",
}
#: Assets at ASSET_REV whose LFS objects Hugging Face no longer serves: since
#: 2026-10-04 every ``swap_T`` evaluation layout (162 objects, eval seeds 0-2)
#: returns 404; upstream regenerated them. They stay pointers and the receipt
#: lists each one, so ``swap_T`` cannot run on an installation made after that
#: date (its layout does not parse); every other task's assets are verified.
UPSTREAM_PURGED_ASSETS = "Assets/Eval_Layout/RoboDojo/arx_x5/*/swap_T_*.json"


#: The worker's verifier, run as the ``graphics-compat`` stage's final readback.
GRAPHICS_COMPAT_MODULE = Path(__file__).resolve().parents[1] / "graphics_compat.py"


def upstream_purged(name: str) -> bool:
    return fnmatch.fnmatchcase(name, UPSTREAM_PURGED_ASSETS)


#: Process-local Vulkan allocation-limit repair for NVIDIA 595 drivers (INSTALL.md,
#: "Process-local graphics compatibility"); must equal graphics_compat.py's pins.
VULKAN_PROFILES_URL = "https://github.com/KhronosGroup/Vulkan-Profiles"
VULKAN_PROFILES_REV = "a400d0b10059b2cc3cd285475593eee79975a5c4"
COMPAT_PATCH_SHA256 = "4189d74dda895d78786dac467021f70084714c68cefc9826406225c797b6193e"
COMPAT_PROFILE_SHA256 = "a94b47098944b7f0ddc95ec46a811c877af582fddee3c02061d5dd4b23ccb1cd"
COMPAT_PROFILE_NAME = "VP_ROBODOJO_RTX_ALLOCATION_COMPAT"
COMPAT_GENERATOR_REQUIREMENTS = ("jsonschema==4.23.0",)


def _single(paths) -> Path:
    found = sorted(paths)
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one build product, found {[str(path) for path in found]}")
    return found[0]


CONSTRAINTS = """numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
setuptools<82
"""


def validate_asset_pointer_blobs(files, tracked_blobs, blobs):
    """Validate exact HEAD Asset blobs, independently of hydrated working files."""
    if {item["name"] for item in files} != set(tracked_blobs):
        raise RuntimeError("HEAD Assets inventory differs from the LFS pointer inventory")
    for item in files:
        canonical = ("version https://git-lfs.github.com/spec/v1\n"
                     f"oid {item['oid_type']}:{item['oid']}\nsize {item['size']}\n").encode()
        if blobs[tracked_blobs[item["name"]]] != canonical:
            raise RuntimeError(f"Noncanonical HEAD Asset pointer: {item['name']}")


class Installer:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.source = self.root / "source/RoboDojo"
        self.assets = self.root / "assets"
        self.python = self.root / "env/bin/python"
        self.env = dict(os.environ)
        for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONPATH", "PYTHONHOME"):
            self.env.pop(key, None)
        for key, suffix in {
            "UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "python",
            "PIP_CACHE_DIR": "cache/pip", "XDG_CACHE_HOME": "cache/xdg",
            "XDG_DATA_HOME": "data/xdg", "OMNI_CONFIG_PATH": "config/omniverse",
            "CUDA_CACHE_PATH": "cache/cuda", "__GL_SHADER_DISK_CACHE_PATH": "cache/gl",
            "TMPDIR": "tmp", "HF_HOME": "cache/huggingface",
        }.items():
            path = self.root / suffix
            path.mkdir(parents=True, exist_ok=True)
            self.env[key] = str(path)
        self.env.update(PYTHONNOUSERSITE="1", PIP_USER="0", OMNI_KIT_ACCEPT_EULA="YES",
                        TERM="xterm-256color", GIT_LFS_SKIP_SMUDGE="1",
                        PIP_INDEX_URL="https://pypi.org/simple",
                        CUROBO_USE_PYBIND="0",
                        VIRTUAL_ENV=str(self.root / "env"))
        self.env["PATH"] = str(self.root / "env/bin") + os.pathsep + self.env["PATH"]
        (self.root / "logs").mkdir(exist_ok=True)
        self.constraints = self.root / "constraints.txt"
        self.constraints.write_text(CONSTRAINTS)
        self.starlette_override = self.root / "starlette-install-override.txt"
        self.starlette_override.write_text("starlette==0.45.3\n")
        omni_paths = {"data_root": self.root / "data/ov", "cache_root": self.root / "cache/ov",
                      "logs_root": self.root / "logs/ov"}
        for path in omni_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        (self.root / "config/omniverse/omniverse.toml").write_text(
            "[paths]\n" + "".join(f"{name} = {json.dumps(str(path))}\n" for name, path in omni_paths.items()))

    def run(self, *args: str | Path, cwd: Path | None = None, capture=False):
        command = [str(arg) for arg in args]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=cwd or self.root, env=self.env,
                                check=True, text=True, capture_output=capture)
        return result.stdout.strip() if capture else None

    def clone(self, url: str, target: Path, revision: str):
        if not (target / ".git").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            clone_flags = ["--no-checkout"]
            if target == self.assets:
                # HF's promisor fetch can fail; LFS smudging is already disabled.
                clone_flags += ["--depth", "1", "--sparse"]
            else:
                clone_flags += ["--filter=blob:none"]
            self.run("git", "clone", *clone_flags, url, target)
        if self.run("git", "status", "--porcelain", cwd=target, capture=True):
            # A --no-checkout clone reports deleted paths until its first checkout.
            if (target / ".git/index").exists():
                raise RuntimeError(f"Refusing to modify dirty checkout {target}")
        self.run("git", "fetch", "origin", revision, cwd=target)
        if target == self.assets:
            self.run("git", "sparse-checkout", "init", "--cone", cwd=target)
            self.run("git", "sparse-checkout", "set", "Assets", cwd=target)
        self.run("git", "checkout", "--detach", revision, cwd=target)

    def sources(self):
        self.clone("https://github.com/RoboDojo-Benchmark/RoboDojo.git", self.source, DOJO_REV)
        self.run("git", "submodule", "sync", "--recursive", cwd=self.source)
        self.run("git", "submodule", "update", "--init", "--recursive", cwd=self.source)
        for name, revision in SUBMODULES.items():
            actual = self.run("git", "rev-parse", "HEAD", cwd=self.source / name, capture=True)
            if actual != revision:
                raise RuntimeError(f"Unexpected {name}: {actual} != {revision}")
        self._ignore_local_paths()
        self.manifest()

    def _ignore_local_paths(self):
        exclude = self.source / ".git/info/exclude"
        current = exclude.read_text() if exclude.exists() else ""
        for pattern in ("/Assets",):
            if pattern not in current.splitlines():
                current += "\n" + pattern + "\n"
        exclude.write_text(current)

    def download_assets(self):
        self.clone("https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo", self.assets, ASSET_REV)
        self.run("git", "sparse-checkout", "init", "--cone", cwd=self.assets)
        self.run("git", "sparse-checkout", "set", "Assets", cwd=self.assets)
        self.run("git", "lfs", "install", "--local", cwd=self.assets)
        self.run("git", "config", "lfs.concurrenttransfers", "32", cwd=self.assets)
        self.run("git", "lfs", "pull", "--include=Assets/**",
                 "--exclude=" + UPSTREAM_PURGED_ASSETS, cwd=self.assets)
        self.verify_assets()

    def verify_assets(self):
        """Verify cached objects and every hydrated Assets path at the fixed commit."""
        actual = self.run("git", "rev-parse", "HEAD", cwd=self.assets, capture=True)
        if actual != ASSET_REV:
            raise RuntimeError(f"Unexpected asset revision: {actual}")
        # fsck sees the full commit even with a sparse checkout. Exclude dataset
        # and checkpoint roots that this asset-only installation did not fetch.
        roots = self.run("git", "ls-tree", "--name-only", "HEAD", cwd=self.assets, capture=True).splitlines()
        excluded = ",".join([name + "/**" for name in roots if name != "Assets"] + [UPSTREAM_PURGED_ASSETS])
        self.run("git", "-c", "lfs.fetchexclude=" + excluded, "lfs", "fsck", "--objects", cwd=self.assets)
        for name in ("Robots", "Object", "Material", "Eval_Layout", "Traj"):
            if not (self.assets / "Assets" / name).is_dir():
                raise RuntimeError(f"Missing asset subtree {name}")
        files = json.loads(self.run("git", "lfs", "ls-files", "--json", cwd=self.assets, capture=True))["files"]
        files = [item for item in files if item["name"].startswith("Assets/")]
        if not files:
            raise RuntimeError("No Assets LFS inventory found")
        tree = self.run("git", "ls-tree", "-rz", "HEAD", "--", "Assets", cwd=self.assets, capture=True)
        tracked_blobs = {}
        for entry in filter(None, tree.split("\0")):
            metadata, name = entry.split("\t", 1)
            mode, kind, oid = metadata.split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise RuntimeError(f"Unsupported tracked Asset entry: {name}")
            tracked_blobs[name] = oid
        unique_blobs = sorted(set(tracked_blobs.values()))
        batch = subprocess.run(["git", "cat-file", "--batch"], cwd=self.assets, env=self.env,
                               input=("\n".join(unique_blobs) + "\n").encode(),
                               capture_output=True, check=True).stdout
        blobs, offset = {}, 0
        for expected_oid in unique_blobs:
            end = batch.index(b"\n", offset)
            oid, kind, size = batch[offset:end].decode().split()
            if oid != expected_oid or kind != "blob":
                raise RuntimeError("Unexpected HEAD Asset blob response")
            offset = end + 1
            blobs[oid] = batch[offset:offset + int(size)]
            offset += int(size) + 1
        validate_asset_pointer_blobs(files, tracked_blobs, blobs)
        purged = [item for item in files if upstream_purged(item["name"]) and not item["downloaded"]]
        for item in files:
            if item in purged:
                continue
            path = self.assets / item["name"]
            if not item["downloaded"] or not item["checkout"] or not path.is_file():
                raise RuntimeError(f"Missing or unhydrated asset: {item['name']}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if item["oid_type"] != "sha256" or digest.hexdigest() != item["oid"] or path.stat().st_size != item["size"]:
                raise RuntimeError(f"Asset payload hash/size mismatch: {item['name']}")
        unique = {item["oid"]: item for item in files}
        status = self.run("git", "status", "--porcelain", cwd=self.assets, capture=True)
        if status:
            raise RuntimeError("Asset checkout must be clean after payload verification")
        receipt = {"schema": "robodojo-assets-verification.v1", "revision": actual,
                   "scope": "HEAD:Assets/**", "lfs_cache_fsck_objects": "passed",
                   "head_pointer_validation": "all_canonical", "head_pointer_blobs": len(unique_blobs),
                   "whole_repository_fsck_claimed": False,
                   "excluded_top_level_paths": [name for name in roots if name != "Assets"],
                   "git_status": "clean",
                   "working_tree_sha256": "all_passed_except_upstream_purged",
                   "upstream_purged_unhydrated": {item["name"]: item["oid"] for item in
                                                  sorted(purged, key=lambda item: item["name"])},
                   "lfs_paths": len(files),
                   "logical_bytes": sum(item["size"] for item in files),
                   "unique_lfs_objects": len(unique),
                   "unique_lfs_bytes": sum(item["size"] for item in unique.values())}
        (self.root / "assets-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        inventory = self.run("git", "lfs", "ls-files", "--long", cwd=self.assets, capture=True)
        (self.root / "assets-lfs.txt").write_text(inventory + "\n")
        self.manifest()

    def python_runtime(self):
        """Select an explicit Data1 base; never let PATH discover an external one."""
        base = self.root / f"python/cpython-{PYTHON_VERSION}-linux-x86_64-gnu/bin/python3.11"
        if not base.exists():
            self.run("uv", "python", "install", PYTHON_VERSION, "--install-dir", self.root / "python", "--no-bin")
        if not base.is_file() or not base.resolve().is_relative_to(self.root / "python"):
            raise RuntimeError(f"Expected independent CPython base missing: {base}")
        if not self.python.exists():
            self.run("uv", "venv", "--python", base, "--seed", self.root / "env")
        code = """import sys, sysconfig, json, _ssl
virtual = {}
for module in list(sys.modules.values()):
    marker = getattr(module, 'PATH_PLACEHOLDER', None)
    if isinstance(marker, str) and marker in sys.path:
        virtual[marker] = {'finder_file': module.__file__, 'mapping': module.MAPPING,
                           'namespaces': module.NAMESPACES}
print(json.dumps({'version': '.'.join(map(str, sys.version_info[:3])),
    'executable': sys.executable, 'base_executable': sys._base_executable,
    'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
    'stdlib': sysconfig.get_path('stdlib'), 'purelib': sysconfig.get_path('purelib'),
    'ssl_module_origin': _ssl.__spec__.origin, 'sys_path': [p for p in sys.path if p],
    'editable_path_hooks': virtual}))
"""
        result = json.loads(self.run(self.python, "-c", code, capture=True))
        paths = [result[key] for key in ("executable", "base_executable", "prefix", "base_prefix",
                                         "stdlib", "purelib")]
        paths += [path for path in result["sys_path"] if path not in result["editable_path_hooks"]]
        for finder in result["editable_path_hooks"].values():
            paths += [finder["finder_file"], *finder["mapping"].values()]
            paths += [path for namespace in finder["namespaces"].values() for path in namespace]
        if result["ssl_module_origin"] != "built-in":
            paths.append(result["ssl_module_origin"])
        if result["version"] != PYTHON_VERSION or any(not Path(path).resolve().is_relative_to(self.root) for path in paths):
            raise RuntimeError(f"Interpreter escapes the independent runtime root: {result}")
        result["base_binary_sha256"] = hashlib.sha256(Path(result["base_executable"]).read_bytes()).hexdigest()
        result["all_python_paths_within_runtime_root"] = True
        (self.root / "python-runtime.json").write_text(json.dumps(result, indent=2) + "\n")

    def sdk(self):
        self.python_runtime()
        self.run(self.python, "-m", "pip", "install", "setuptools<82", "setuptools_scm>=6.2", "wheel==0.45.1")
        # --find-links below needs the directory even when donor-wheels never ran.
        (self.root / "wheels").mkdir(exist_ok=True)
        local_wheels = sorted((self.root / "wheels").glob("*.whl"))
        if local_wheels:
            self.run(self.python, "-m", "pip", "install", "--no-deps", *local_wheels)
        self.run("uv", "pip", "install", "--python", self.python, "torch==2.7.0", "torchvision==0.22.0",
                 "torchaudio==2.7.0", "--index-url", "https://download.pytorch.org/whl/cu128",
                 "--find-links", self.root / "wheels", "-c", self.constraints)
        self.run("uv", "pip", "install", "--python", self.python, "isaacsim[all,extscache]==5.1.0",
                 "--index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match",
                 "--extra-index-url", "https://pypi.nvidia.com", "-c", self.constraints,
                 "--find-links", self.root / "wheels")
        self.manifest()

    def donor_wheels(self, donor: Path):
        """Rebuild only distributions whose hashed installed payload is intact.

        These are derived local wheels, never claimed to have the original ZIP
        digest. Their provenance records distinguish the installed RECORD hashes
        from the new archive hash. Generated scripts/pyc/install metadata are not
        copied; pip recreates entry points in the new interpreter.
        """
        wheel_dir = self.root / "wheels"
        wheel_dir.mkdir(exist_ok=True)
        reports = []
        for info in sorted(donor.glob("*.dist-info")):
            if not info.name.startswith(("isaacsim", "torch-", "torchvision-", "torchaudio-",
                                         "nvidia_", "triton-")):
                continue
            print(f"Validating donor {info.name}", flush=True)
            rows = list(csv.reader((info / "RECORD").open()))
            payload, failures = [], []
            for relative, expected, size in rows:
                if not expected or relative.startswith("../"):
                    continue
                path = donor / relative
                if not path.is_file():
                    failures.append({"path": relative, "reason": "missing"})
                    continue
                algorithm, value = expected.split("=", 1)
                digest = hashlib.new(algorithm)
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                actual = base64.urlsafe_b64encode(digest.digest()).decode().rstrip("=")
                if actual != value or (size and path.stat().st_size != int(size)):
                    failures.append({"path": relative, "reason": "hash_or_size_mismatch"})
                elif not (path.parent == info and path.name in {"INSTALLER", "REQUESTED", "direct_url.json"}):
                    payload.append((relative, expected, str(path.stat().st_size)))
            report = {"donor": str(info), "hashed_payload_files": len(payload), "failures": failures}
            if failures:
                print(f"  rejected: {len(failures)} missing/changed files", flush=True)
                reports.append(report)
                continue
            tags = [line[5:] for line in (info / "WHEEL").read_text().splitlines() if line.startswith("Tag: ")]
            python_tags, abi_tags, platform_tags = zip(*(tag.split("-") for tag in tags))
            compressed_tag = "-".join(".".join(dict.fromkeys(values)) for values in (python_tags, abi_tags, platform_tags))
            name = info.name.removesuffix(".dist-info") + "-" + compressed_tag + ".whl"
            wheel = wheel_dir / name
            record_name = info.name + "/RECORD"
            record = io.StringIO()
            writer = csv.writer(record, lineterminator="\n")
            writer.writerows(payload)
            writer.writerow((record_name, "", ""))
            with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for relative, _, _ in payload:
                    archive.write(donor / relative, relative)
                archive.writestr(record_name, record.getvalue())
            digest = hashlib.sha256()
            with wheel.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            report.update(wheel=str(wheel), derived_wheel_sha256=digest.hexdigest())
            reports.append(report)
            (self.root / "donor-wheel-provenance.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(f"  validated and rebuilt {name}", flush=True)
        (self.root / "donor-wheel-provenance.json").write_text(json.dumps(reports, indent=2) + "\n")

    def packages(self):
        if not self.python.exists() or not self.source.exists():
            raise RuntimeError("Run sources and sdk stages first")
        self.env["PIP_CONSTRAINT"] = str(self.constraints)
        self.env["UV_CONSTRAINT"] = str(self.constraints)
        common = ["uv", "pip", "install", "--python", self.python, "--index-url", "https://pypi.org/simple",
                  "-c", self.constraints, "--override", self.starlette_override]
        lock = self.root / "installation.lock"
        if not lock.exists():
            self.resolve()
        # The resolved file includes tracked editable sources and all selected
        # runtime dependencies, with no RL framework extras or editor mutation.
        self.run(*common, "-r", lock, "--no-build-isolation-package", "nvidia-curobo",
                 "--extra-index-url", "https://pypi.nvidia.com",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                 "--index-strategy", "unsafe-best-match", "--find-links", self.root / "wheels")
        check = subprocess.run([str(self.python), "-m", "pip", "check"], env=self.env,
                               text=True, capture_output=True, check=False)
        messages = [line for line in check.stdout.splitlines() if line.strip()]
        known = all(line.startswith("isaaclab ") and "starlette==0.49.1" in line
                    and "starlette 0.45.3" in line for line in messages)
        result = {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr,
                  "accepted_upstream_starlette_exception": check.returncode == 1 and bool(messages)
                  and known and not check.stderr.strip()}
        (self.root / "pip-check.json").write_text(json.dumps(result, indent=2) + "\n")
        if check.returncode and not result["accepted_upstream_starlette_exception"]:
            raise RuntimeError(f"Unexpected dependency conflicts: {check.stdout}{check.stderr}")
        freeze = self.run(self.python, "-m", "pip", "freeze", "--all", capture=True)
        (self.root / "pip-freeze.txt").write_text(freeze + "\n")
        metadata = self.run(self.python, "-c",
                            "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()},sort_keys=True,indent=2))",
                            capture=True)
        (self.root / "packages.json").write_text(metadata + "\n")
        self.manifest()

    def resolve(self):
        """Record the complete pinned resolver input without launching imports."""
        requirements = ["isaacsim[all,extscache]==5.1.0", "torch==2.7.0+cu128",
                        "torchvision==0.22.0+cu128", "torchaudio==2.7.0+cu128",
                        "opencv-python-headless==4.11.0.86", "pillow", "matplotlib", "scikit-learn"]
        requirements += (self.source / "scripts/requirements.txt").read_text().splitlines()
        for directory in sorted((self.source / "third_party/IsaacLab/source").iterdir()):
            if (directory / "setup.py").exists():
                requirements.append("-e " + str(directory))
        requirements.append("-e " + str(self.source / "third_party/curobo") + "[cu12]")
        inputs = self.root / "installation.in"
        inputs.write_text("\n".join(requirements) + "\n")
        self.run("uv", "pip", "compile", "--python-version", "3.11",
                 "--index-url", "https://pypi.org/simple", "--extra-index-url", "https://pypi.nvidia.com",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                 "--index-strategy", "unsafe-best-match", "--constraint", self.constraints,
                 "--override", self.starlette_override, inputs, "--output-file", self.root / "installation.lock")

    def bind_assets(self):
        self._ignore_local_paths()
        link = self.source / "Assets"
        target = self.assets / "Assets"
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected asset symlink: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace {link}")
        else:
            link.symlink_to(target)
        generated = []
        excluded = self.assets / ".git/info/exclude"
        exclusions = excluded.read_text() if excluded.exists() else ""
        for template in sorted((target / "Robots").rglob("*_tmp.yml")):
            output = template.with_name(template.name.replace("_tmp.yml", ".yml"))
            relative = output.relative_to(self.assets).as_posix()
            tracked = self.run("git", "ls-files", "--", relative, cwd=self.assets, capture=True)
            if tracked:
                raise RuntimeError(f"Refusing to overwrite tracked asset {relative}")
            raw = template.read_bytes()
            content = raw.decode().replace("${ASSETS_PATH}", str(self.source)).replace("$ASSETS_PATH", str(self.source))
            output.write_text(content)
            template_relative = template.relative_to(self.assets).as_posix()
            generated.append({"template": template_relative,
                              "template_git_blob": self.run("git", "rev-parse", "HEAD:" + template_relative,
                                                            cwd=self.assets, capture=True),
                              "template_sha256": hashlib.sha256(raw).hexdigest(),
                              "output": relative, "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                              "replacement_root": str(self.source)})
            pattern = "/" + relative
            if pattern not in exclusions.splitlines():
                exclusions += "\n" + pattern + "\n"
        if not generated:
            raise RuntimeError("No robot templates found")
        excluded.write_text(exclusions)
        (self.root / "asset-bindings.json").write_text(json.dumps(generated, indent=2) + "\n")
        self.manifest()

    def doctor(self):
        """Run upstream static checks with an isolated temporary inventory path."""
        original = self.source / "scripts/internal/verify_install.sh"
        raw = original.read_text()
        root_line = 'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"'
        old_temp = "/tmp/robodojo_tasks_verify.json"
        if raw.count(root_line) != 1 or raw.count(old_temp) != 3:
            raise RuntimeError("Unexpected upstream doctor script; review its path adaptation")
        temp_path = self.root / "tmp/robodojo_tasks_verify.json"
        adapted = raw.replace(root_line, "ROOT_DIR=" + shlex.quote(str(self.source)))
        adapted = adapted.replace(">" + old_temp, ">" + shlex.quote(str(temp_path)))
        adapted = adapted.replace("open('" + old_temp + "',", "open(" + repr(str(temp_path)) + ",")
        adapted = adapted.replace("rm -f " + old_temp, "rm -f " + shlex.quote(str(temp_path)))
        if any(expression in adapted for expression in (
                ">" + old_temp, "open('" + old_temp + "',", "rm -f " + old_temp)):
            raise RuntimeError("Unadapted upstream doctor temporary path")
        script = self.root / "tmp/verify_install.isolated.sh"
        script.write_text(adapted)
        receipt = {"source": str(original), "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                   "adapted_copy": str(script), "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                   "adaptations": {"ROOT_DIR": str(self.source), old_temp: str(temp_path)},
                   "upstream_source_modified": False, "isaac_imports_skipped": True,
                   "conda_and_policy_checks_skipped": True}
        (self.root / "doctor-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        self.run("bash", script, "--skip-conda", "--skip-isaac", "--skip-policy",
                 "--summary", self.root / "doctor.json", cwd=self.source)

    def graphics_compat(self, jobs: int = 16):
        """Build the Khronos Profiles layer and write ``graphics-compat/`` with its receipt.

        Every RoboDojo worker activates the layer, so every installation needs it.
        Needs git, a C++17 compiler and network access for the
        layer's pinned ``known_good.json`` dependencies. Starts no simulator; the
        worker's own readback verifies the repair on the GPU before every start.
        """
        here = Path(__file__).resolve().parent
        patch, profile = here / "vulkan-profiles-core-id.patch", here / f"{COMPAT_PROFILE_NAME}.json"
        for path, expected in ((patch, COMPAT_PATCH_SHA256), (profile, COMPAT_PROFILE_SHA256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Tracked compatibility input drifted: {path.name}")
        if not self.python.exists():
            raise RuntimeError("graphics-compat needs the RoboDojo interpreter; run the sdk stage first")
        work = self.root / "graphics-compat-build"
        source, build, venv = work / "Vulkan-Profiles", work / "build", work / "venv"
        if not (source / ".git").exists() or self.run(
                "git", "rev-parse", "HEAD", cwd=source, capture=True) != VULKAN_PROFILES_REV:
            self.clone(VULKAN_PROFILES_URL, source, VULKAN_PROFILES_REV)
        if self.run("git", "rev-parse", "HEAD", cwd=source, capture=True) != VULKAN_PROFILES_REV:
            raise RuntimeError("Unexpected Vulkan-Profiles revision")
        applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source,
                                 env=self.env, capture_output=True).returncode == 0
        if not applied:
            self.run("git", "apply", patch, cwd=source)
        # CMake and Ninja come from a separate tool environment, so no system build
        # tools or sudo are needed and the simulator environment stays unchanged.
        tools = self.root / "tools/build-env"
        if not (tools / "bin/python").exists():
            self.run("uv", "venv", "--python", "3.11", tools)
        self.run("uv", "pip", "install", "--python", tools / "bin/python", "cmake==4.1.2", "ninja==1.13.0")
        self.env["PATH"] = str(tools / "bin") + os.pathsep + self.env["PATH"]
        if not (venv / "bin/python").exists():
            self.run("uv", "venv", "--python", self.python, venv)
        self.run("uv", "pip", "install", "--python", venv / "bin/python", *COMPAT_GENERATOR_REQUIREMENTS)
        commands = [
            ["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DUPDATE_DEPS=ON",
             "-DBUILD_TESTS=OFF", f"-DPython3_EXECUTABLE={venv / 'bin/python'}"],
            ["cmake", "--build", build, "--target", "ProfilesLayer", "-j", str(jobs)],
        ]
        for command in commands:
            self.run(*command)
        built_layer = _single(build.rglob("libVkLayer_khronos_profiles.so"))
        built_manifest = _single(build.rglob("VkLayer_khronos_profiles.json"))
        out = self.root / "graphics-compat"
        targets = {"layer": out / "lib/libVkLayer_khronos_profiles.so",
                   "profile": out / f"profiles/{COMPAT_PROFILE_NAME}.json",
                   "manifest": out / "share/vulkan/implicit_layer.d/VkLayer_khronos_profiles.json",
                   "patch": out / patch.name}
        for path in targets.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        targets["layer"].write_bytes(built_layer.read_bytes())
        targets["profile"].write_bytes(profile.read_bytes())
        targets["patch"].write_bytes(patch.read_bytes())
        manifest = json.loads(built_manifest.read_text())
        manifest["layer"]["library_path"] = str(targets["layer"])
        manifest["layer"]["enable_environment"] = {"ROBODOJO_VULKAN_COMPAT_ENABLE": "1"}
        manifest["layer"]["disable_environment"] = {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""}
        targets["manifest"].write_text(json.dumps(manifest, indent=4) + "\n")
        versions = {tool: self.run(tool, "--version", capture=True).splitlines()[0] for tool in ("cmake", "c++")}
        reference = {role: {"path": path.relative_to(out).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for role, path in targets.items()}
        receipt = {"schema": "ora.robodojo-graphics-compat-installation.v1",
                   "source_url": VULKAN_PROFILES_URL, "source_revision": VULKAN_PROFILES_REV,
                   "patches": [reference["patch"]], **reference,
                   "build": {"installer": "scripts/setup-robodojo graphics-compat",
                             "generator_requirements": list(COMPAT_GENERATOR_REQUIREMENTS),
                             "tool_versions": versions,
                             "reproduction_commands": [" ".join(map(str, c)) for c in commands]}}
        temp = out / f"installation.{os.getpid()}.json"
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        temp.replace(out / "installation.json")
        print(json.dumps({role: value["sha256"] for role, value in reference.items()}, indent=2))
        # The worker's own verifier: hashes, profile shape and a child readback
        # that compares Vulkan properties with and without the layer.
        self.run(self.python, GRAPHICS_COMPAT_MODULE, self.root)

    def manifest(self):
        data = {"schema": "robodojo-install.v1", "root": str(self.root),
                "python": str(self.python), "source": str(self.source),
                "asset_root": str(self.assets / "Assets"), "expected_source": DOJO_REV,
                "expected_assets": ASSET_REV, "expected_submodules": SUBMODULES,
                "simulator_launched_by_installer": False, "checkouts": {}}
        for name, path in [("RoboDojo", self.source), ("assets", self.assets)] + [
                (name, self.source / name) for name in SUBMODULES]:
            if (path / ".git").exists():
                data["checkouts"][name] = {
                    "head": self.run("git", "rev-parse", "HEAD", cwd=path, capture=True),
                    "status": self.run("git", "status", "--porcelain", cwd=path, capture=True),
                }
        # Separate temporary file per process permits independent download stages.
        temp = self.root / f"install-manifest.{os.getpid()}.json"
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(self.root / "install-manifest.json")


def default_root(environ=None) -> Path | None:
    """The installation root this shell already names, or None.

    ``ROBODOJO_RUNTIME_ROOT`` wins; otherwise ``ROBODOJO_ROOT`` (the
    ``<root>/source/RoboDojo`` checkout) names its installation root. There is
    no host default: a missing root used to fall back to one SSH23 path.
    """
    values = os.environ if environ is None else environ
    if values.get("ROBODOJO_RUNTIME_ROOT"):
        return Path(values["ROBODOJO_RUNTIME_ROOT"]).expanduser()
    source = values.get("ROBODOJO_ROOT")
    if source:
        source_path = Path(source).expanduser()
        if source_path.name == "RoboDojo" and source_path.parent.name == "source":
            return source_path.parent.parent
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sources", "assets", "verify-assets", "python-runtime", "sdk", "packages", "resolve", "bind-assets", "graphics-compat", "doctor", "manifest", "donor-wheels"))
    parser.add_argument("--donor-site-packages", type=Path)
    parser.add_argument("--root", type=Path,
                        help="installation root (default: ROBODOJO_RUNTIME_ROOT, else the root that "
                             "ROBODOJO_ROOT=<root>/source/RoboDojo names)")
    args = parser.parse_args(argv)
    root = args.root or default_root()
    if root is None:
        parser.error("pass --root <installation directory> or set ROBODOJO_RUNTIME_ROOT")
    installer = Installer(root)
    if args.stage == "donor-wheels":
        if args.donor_site_packages is None:
            parser.error("donor-wheels requires --donor-site-packages")
        installer.donor_wheels(args.donor_site_packages.resolve())
        return
    stages = {"assets": installer.download_assets, "verify-assets": installer.verify_assets,
              "bind-assets": installer.bind_assets, "python-runtime": installer.python_runtime,
              "graphics-compat": installer.graphics_compat}
    (stages.get(args.stage) or getattr(installer, args.stage))()


if __name__ == "__main__":
    main()
