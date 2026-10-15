"""The wheel carries every product package and the data files it reads at run time."""

from __future__ import annotations

import fnmatch
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]

#: Tracked non-Python files under a product package that the wheel need not carry.
NOT_SHIPPED = {"ora/agent/README.md"}


def _tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def _discovered() -> set[str]:
    """setuptools' ``packages.find`` rule: a directory with ``__init__.py`` whose
    dotted name matches an ``include`` pattern (the project venv has no setuptools)."""
    find = CONFIG["packages"]["find"]
    where = ROOT / find["where"][0]
    tops = {pattern.split(".")[0] for pattern in find["include"]}
    names = {
        ".".join(init.parent.relative_to(where).parts)
        for top in tops for init in (where / top).rglob("__init__.py")
    }
    return {name for name in names if any(fnmatch.fnmatchcase(name, p) for p in find["include"])}


def test_every_product_package_is_in_the_wheel():
    packages = _discovered()
    for top in ("ora", "viewer"):
        expected = {
            str(Path(path).parent).replace("/", ".")
            for path in _tracked(top) if path.endswith("/__init__.py")
        }
        assert expected - packages == set(), f"{top} packages missing from the wheel"


def test_every_named_experiment_package_exists():
    names = [name for name in CONFIG["packages"]["find"]["include"] if name.startswith("experiments")]
    missing = [name for name in names if not (ROOT / name.replace(".", "/") / "__init__.py").is_file()]
    assert missing == []


def test_runtime_data_files_under_ora_are_package_data():
    base = ROOT / "ora"
    shipped = {
        str(path.relative_to(ROOT))
        for pattern in CONFIG["package-data"]["ora"]
        for path in base.glob(pattern) if path.is_file()
    }
    unshipped = [
        path for path in _tracked("ora")
        if not path.endswith(".py") and path not in shipped and path not in NOT_SHIPPED
    ]
    assert unshipped == []


def test_vendored_robotwin_sources_ship_as_data_beside_their_installer():
    # vendored/ has no __init__.py: vendored_install copies these files into a RoboTwin checkout.
    patterns = CONFIG["package-data"]["ora"]
    assert "environment/backends/robotwin/vendored/*" in patterns
    assert (ROOT / "ora/environment/backends/robotwin/vendored/planner.py").is_file()


def test_live_world_browser_assets_are_in_the_wheel():
    shipped = {
        str(path.relative_to(ROOT / "viewer"))
        for pattern in CONFIG["package-data"]["viewer"]
        for path in (ROOT / "viewer").glob(pattern) if path.is_file()
    }
    assert {"templates/base.html", "templates/live_world.html", "static/base.css",
            "static/viewer.css", "static/live_world.js"} <= shipped
