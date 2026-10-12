# Examples

| Example | Purpose |
|---|---|
| `uv run --locked ora run --env synthetic` | Real model API, no simulator/GPU; [quickstart](../docs/quickstart.md). |
| `uv run --locked python examples/quickstart_smoke.py` | Same CLI with a deterministic local HTTP model; CI uses no API credits. |
| [official_suite](official_suite/README.md) | Expand maintained LIBERO-Pro and RoboTwin suite cells. |
| [hosts/example-host.yaml](hosts/example-host.yaml) | Template for local host/resource registration. |
| [LIBERO recipe](../docs/environments/libero.md) | Install and compose one Panda episode. |
| [RoboTwin recipe](../docs/environments/robotwin.md) | Install and compose one dual-arm episode. |
| [RoboDojo recipe](../docs/environments/robodojo.md) | Install and compose one Isaac episode. |

Keep model keys and generated runs outside Git. Examples use current defaults; they do not reproduce an archived experiment's score.
