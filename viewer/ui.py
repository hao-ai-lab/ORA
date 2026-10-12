"""The shared HTML shell for every Viewer page, including standalone rendering."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup


ROOT = Path(__file__).parent
TEMPLATES = Environment(
    loader=FileSystemLoader(ROOT / "templates"),
    autoescape=select_autoescape(("html",)),
)


def render_page(title: str, body: str, *, wide: bool = False) -> str:
    """Only the already-rendered, escaped application body is trusted HTML."""
    version = max((ROOT / "static" / name).stat().st_mtime_ns for name in ("base.css", "viewer.css"))
    return TEMPLATES.get_template("base.html").render(
        title=title, body=Markup(body), wide=wide, style_version=version,
    )


def render_style_guide() -> str:
    body = TEMPLATES.get_template("style_guide.html").render()
    return render_page("Frontend template", body)
