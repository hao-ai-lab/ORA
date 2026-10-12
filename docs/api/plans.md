# Plans and benchmark cells

These functions compose or inspect configuration. Resolving a plan does not
launch an episode. Source, host, controller, provider, and qualification gates
still apply at launch.

```python
from ora.orchestrator.spec import (
    load_run_spec,
    resolve_run_spec,
    validate_run_plan,
)
```

::: ora.orchestrator.spec.load_run_spec

::: ora.orchestrator.spec.resolve_run_spec

::: ora.orchestrator.spec.validate_run_plan

## Benchmark builders

```python
from ora.orchestrator.benchmark_spec import (
    build_robotwin_spec,
    build_libero_spec,
    build_libero_pro_spec,
)
```

::: ora.orchestrator.benchmark_spec.build_robotwin_spec

::: ora.orchestrator.benchmark_spec.build_libero_spec

::: ora.orchestrator.benchmark_spec.build_libero_pro_spec
