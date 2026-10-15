# Maintain the documentation

The documentation uses MkDocs Material. `mkdocs.yml` owns navigation,
search, code-copy controls, theme, Markdown extensions, and API rendering.
The public pages stay in `docs/`; current contracts remain under `docs/reference/`.

## Preview and build

From the source checkout, with uv installed:

```bash
./scripts/docs serve
```

Open `http://127.0.0.1:8000/`. Editing a page reloads the preview. Build a static
site with strict link, anchor, navigation, and private-content checks:

```bash
./scripts/docs build
```

The output is `.cache/docs/site/`, ignored by Git. CI runs this same build.
The commands do not launch a model or simulator.

## Dependencies

`docs/requirements.in` pins the documentation tools; `docs/requirements.txt`
locks their transitive dependencies with hashes. The wrapper uses an isolated
uv environment and leaves the runtime's `uv.lock` unchanged.

When intentionally changing documentation dependencies, regenerate their lock:

```bash
uv pip compile docs/requirements.in --python-version 3.11 \
  --generate-hashes --output-file docs/requirements.txt
```

## Add or change a page

1. Edit its canonical Markdown source. Use repository-relative Markdown links.
2. Register a new public page in `mkdocs.yml` under the reader's task.
3. Run `./scripts/docs build`; fix missing pages, anchors, and omitted navigation.
4. Preview wide and narrow layouts, code blocks, tables, and search.

Contributing, examples, the viewer guide, the Runtime ladder, and the detailed
RoboDojo installer are included from their existing repository READMEs. The build
hook rewrites their relative links for the site and watches their source files.
Edit those originals rather than creating another copy under `docs/`.

## Python reference

Pages under `docs/api/` select the supported symbols using mkdocstrings.
The Python handler uses static analysis with inspection disabled: simulator
dependencies and runtime credentials are unnecessary. Update the matching
current-reference prose when behavior changes; generated signatures do not
replace an authority or evidence contract.

## Hosting

Static output can be hosted under a root URL or a project subdirectory. Set
`ORA_DOCS_URL` to the intended absolute URL when building, and update
`extra.source_repository` in `mkdocs.yml` if the source destination changes.
The default URL is for local preview. Building does not publish or push anything;
the final hosting destination and publication remain the user's decision.
