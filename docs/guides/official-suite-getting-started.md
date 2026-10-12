# Benchmark setup and official suites

Start with a [single Synthetic episode](../quickstart.md), then install one robot environment:

- [LIBERO / LIBERO-Pro](../environments/libero.md)
- [RoboTwin](../environments/robotwin.md)
- [RoboDojo](../environments/robodojo.md)
- [Shared host, planner and perception setup](../environments/common.md)

Each page describes the isolated interpreter, pinned sources/assets, discovery checks and one-cell launch. Run one qualified cell before scaling up. No delivery branch or historical experiment tag is required.

[examples/official_suite](../../examples/official_suite/README.md) expands maintained LIBERO-Pro and RoboTwin suites from the current default resolver. Official seed selection, task/source versions, episode budget and environment verdict remain explicit. Use `ora.orchestrator.suite_queue --help` for scheduling already composed cells, respecting declared resources and existing leases.

A policy using robot tools is a different treatment from a trained VLA policy. Report its observation/action abstraction, task/seed protocol, attempts and budget alongside results. Installation and infrastructure qualification do not establish model competence or benchmark success.
