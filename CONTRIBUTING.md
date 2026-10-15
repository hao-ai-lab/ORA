# Contributing

Read [AGENTS.md](AGENTS.md) and the [code map](docs/code-map.md). Open an issue or draft PR describing the concrete problem and expected behavior. For a new backend or a schema/authority change, describe the contract before implementation.

Install FFmpeg on your system before running the full suite; the video export
tests use both `ffmpeg` and `ffprobe` from `PATH`. On Ubuntu, run
`sudo apt-get install ffmpeg`. The Synthetic quickstart does not need FFmpeg.

```bash
uv sync --all-extras --locked
uv run --locked python -m pytest tests
uvx pre-commit run --all-files
```

Use targeted tests while developing. Simulator qualification is separate from deterministic tests: state the code, controller, assets, environment and coverage actually checked. Independent tests can run concurrently within measured host capacity; do not disturb someone else's simulator or service.

Changes to APIs, defaults, prompts, tools, context, runtime lifecycle or evidence must update the current reference in the same PR. New shared behavior becomes the default for applicable maintained consumers. Regenerate `policy.lock` with `./scripts/ora lock` after policy changes. Compare model-visible content for intentional policy changes.

Keep credentials, generated evidence, simulator assets and media out of Git. Preserve sealed evidence; revalidation writes independent reports. Third-party code must retain its source/license notices. Use semantic versioning within the 0.x series and describe breaking API/schema changes explicitly.

Preview the documentation with `./scripts/docs serve`, and run
`./scripts/docs build` before returning a documentation change. It checks
navigation, links, anchors, API rendering, and private planning exclusions.
See the [documentation guide](docs/developer/documentation.md) for source
ownership and the isolated documentation dependency lock.
