"""Build public docs from canonical sources without importing the runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PAGES = {
    "developer/contributing.md": "CONTRIBUTING.md",
    "developer/runtime-ladder.md": "experiments/runtime_v1/README.md",
    "usage/viewer.md": "viewer/README.md",
    "usage/examples.md": "examples/README.md",
    "usage/official-suite.md": "examples/official_suite/README.md",
    "environments/robodojo-installation.md": "ora/environment/backends/robodojo/installation/INSTALL.md",
}
_sources: dict[Path, str] = {}
_link = re.compile(r"(?P<start>!?\[[^\n]*?\]\()(?P<url>[^\s)]+)")


def on_config(config):
    for source in EXTERNAL_PAGES.values():
        config.watch.append(str(ROOT / source))
    return config


def on_files(files, config):
    # A second boundary in addition to exclude_docs: no private page or asset.
    for file in list(files):
        if file.src_uri.startswith("design/"):
            files.remove(file)
    for uri, source in EXTERNAL_PAGES.items():
        files.append(File.generated(config, uri, abs_src_path=str(ROOT / source)))
    _sources.clear()
    for file in files:
        if file.src_uri.endswith(".md"):
            source = ROOT / EXTERNAL_PAGES.get(file.src_uri, "docs/" + file.src_uri)
            _sources[source.resolve()] = file.src_uri
    _sources[(ROOT / "README.md").resolve()] = "README.md"
    return files


def on_page_markdown(markdown, page, config, files):
    original = ROOT / EXTERNAL_PAGES.get(page.file.src_uri, "docs/" + page.file.src_uri)

    def rewrite(match):
        url = urlsplit(match["url"])
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return match[0]
        target = (original.parent / unquote(url.path)).resolve()
        if not target.is_relative_to(ROOT):
            raise PluginError(f"Documentation link escapes the repository: {original}: {url.path}")
        relative = target.relative_to(ROOT)
        if relative.parts[:2] == ("docs", "design"):
            raise PluginError(f"Public page links to private planning: {page.file.src_uri}")
        if target in _sources:
            path = os.path.relpath(_sources[target], Path(page.file.src_uri).parent)
            rewritten = urlunsplit(("", "", path, url.query, url.fragment))
        elif target.is_file():
            rewritten = config.extra["source_repository"] + quote(relative.as_posix())
            if url.fragment:
                rewritten += "#" + url.fragment
        else:
            return match[0]  # Strict MkDocs validation reports missing links.
        return match["start"] + rewritten

    lines, fence = [], None
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            lines.append(line)
        else:
            lines.append(line if fence else _link.sub(rewrite, line))
    return "".join(lines)


def on_post_build(config):
    site = Path(config.site_dir)
    if (site / "design").exists():
        raise PluginError("Private planning was copied into the documentation site")
    search = json.loads((site / "search/search_index.json").read_text())
    if not search.get("docs"):
        raise PluginError("Documentation search index is empty")
    for entry in search["docs"]:
        if entry["location"].startswith("design/"):
            raise PluginError("Private planning entered the documentation search index")
