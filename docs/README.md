# ORA documentation

Start with the [quickstart](quickstart.md). It runs a real model against Synthetic and produces an inspectable episode without a simulator.

- [Supported environments, providers and platforms](support.md)
- [LIBERO / LIBERO-Pro](environments/libero.md), [RoboTwin](environments/robotwin.md), [RoboDojo](environments/robodojo.md)
- [Host and perception-service setup](environments/common.md)
- [Architecture](architecture.md), [code map](code-map.md), [glossary](glossary.md)
- [Current APIs and evidence](reference/README.md), [CLI reference](guides/cli-reference.md)
- [Context assembly](context-assembly.md), [backend authoring](guides/add-backend.md)
- [Examples](../examples/README.md), [tests](../tests/README.md), [contributing](../CONTRIBUTING.md)

Commands assume the repository root and a clean source checkout. Simulator installation is separate from the `uv.lock` environment. A resolved plan records the exact configuration; an installation or a passing unit test does not constitute simulator qualification.
