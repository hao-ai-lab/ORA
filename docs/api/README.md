# Python API

These pages render signatures and docstrings from ORA's source with static
analysis. Building them does not import the runtime, contact a model, or load
simulator packages. Import from the canonical subpackage shown for each symbol.

| Area | Interfaces |
|---|---|
| [Plans and benchmark cells](plans.md) | Load/resolve/validate a RunSpec and compose a benchmark task |
| [Policy sessions](policy.md) | Policy turn, proposal, lifecycle, and session contracts |
| [Environment sessions](environment.md) | Transport-neutral session boundary |
| [Action contracts](actions.md) | Versioned action validation and target-state phases |

The generated signatures describe Python entry points. The maintained
[public-interface reference](../reference/public-interfaces.md),
[configuration/evidence contract](../reference/configuration-and-evidence.md),
and [runtime lifecycle](../reference/runtime-lifecycle.md) define their authority
and behavior. Operator launches use the supervised CLI admission path.
