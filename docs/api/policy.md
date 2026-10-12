# Policy sessions

The runtime supplies a policy-safe turn; the policy returns a proposal, no
action, or a lifecycle request. A proposal does not itself mutate the environment.

```python
from ora.agent.contracts import (
    PolicySession,
    PolicySessionCapabilities,
    PolicyTurnInput,
    PolicyRuntimeEvent,
    ActionProposal,
    NoAction,
    LifecycleRequest,
)
```

::: ora.agent.contracts.PolicySession

::: ora.agent.contracts.PolicySessionCapabilities

::: ora.agent.contracts.PolicyTurnInput

::: ora.agent.contracts.PolicyRuntimeEvent

::: ora.agent.contracts.ActionProposal

::: ora.agent.contracts.NoAction

::: ora.agent.contracts.LifecycleRequest
