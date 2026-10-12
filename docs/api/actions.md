# Action contracts

Validate an action against its versioned contract before admission. A valid
envelope still needs Supervisor checks and an executing environment. Target-state
v4 stages a combined request as pose-first, conditional-gripper phases.

```python
from ora.contracts.action import (
    ActionValidationError,
    validate_runtime_action,
    target_state_phases,
)
```

::: ora.contracts.action.ActionValidationError

::: ora.contracts.action.validate_runtime_action

::: ora.contracts.action.target_state_phases

[Runtime lifecycle and authority](../reference/runtime-lifecycle.md)
