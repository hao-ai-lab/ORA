# Contributing

Read [AGENTS.md](AGENTS.md) and the [code map](docs/code-map.md). Open an issue or draft PR describing the concrete problem and expected behavior. For a new backend or a schema/authority change, describe the contract before implementation.

```bash
uv sync --all-extras --locked
uv run --locked python -m pytest tests
uvx pre-commit run --all-files
```

Use targeted tests while developing. Simulator qualification is separate from deterministic tests: state the code, controller, assets, environment and coverage actually checked. Independent tests can run concurrently within measured host capacity; do not disturb someone else's simulator or service.

Changes to APIs, defaults, prompts, tools, context, runtime lifecycle or evidence must update the current reference in the same PR. New shared behavior becomes the default for applicable maintained consumers. Regenerate `policy.lock` with `./scripts/ora lock` after policy changes, and composed tools with `python -m ora.agent.tools.composed compile` when needed. Compare model-visible content for intentional policy changes.

Keep credentials, generated evidence, simulator assets and media out of Git. Preserve sealed evidence; revalidation writes independent reports. Third-party code must retain its source/license notices. Use semantic versioning within the 0.x series and describe breaking API/schema changes explicitly.
