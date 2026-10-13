# Install ORA

ORA's host environment needs Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Python 3.11. uv can install the pinned Python version. Start with a source
checkout so launches can record an exact Git commit.

```bash
git clone https://github.com/hao-ai-lab/ora-agent.git
cd ora-agent
uv sync --all-extras --locked
uv run --locked ora doctor
```

`--all-extras` also installs the media extra (`imageio`, used by RoboTwin
qualification captures) and the test runner; the same command is used everywhere
in these docs.

`ora doctor` reports Python and ORA versions, credential presence, and optional
graphics/media dependencies. It does not print keys, authenticate a provider,
or start a simulator. Missing simulator dependencies are expected at this step.

## Requirements by step

| Step | Model API | GPU | Simulator / assets |
|---|---|---|---|
| Install and run `ora doctor` | Not required | Not required | Not required |
| Synthetic first episode | Required | Not required | Not required |
| Robot episode | Required, with the selected observation capabilities | See environment guide | Required |
| Build these docs | Not required | Not required | Not required |

Synthetic uses a real model API. Cost and latency depend on the endpoint; a
self-hosted service can use the same interface.

## Next step

[Run the first episode](../quickstart.md), then choose a
[robot environment](../environments/README.md).

## Developer installation

```bash
uv sync --all-extras --locked
uv run --locked python -m pytest tests
```

The full suite needs system FFmpeg for video tests. See
[contributing](../../CONTRIBUTING.md). Preview these docs with
`./scripts/docs serve`; documentation dependencies are isolated from the runtime.
