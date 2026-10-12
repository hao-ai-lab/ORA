# Configuration and evidence

## RunSpec and RunPlan

A RunSpec (`ora.run-spec.v1`) names `experiment`, `config`, `milestone`, `task`, `task_instruction`, `seed`, `evidence_tier`, `output_workspace`, `policy`, `runtime` and `budgets`. Additional qualification, session, prerequisite and launch-admission blocks are validated when selected. Resolution is implemented in `ora/orchestrator/spec.py` and its focused `spec_*` modules; the resolved plan is the exact execution contract.

Policy kinds are `native_api` and `scripted`. Native providers are `openai_compatible` and `anthropic_messages`. A Native policy binds an explicit endpoint/model, credential-variable names, context/output limits, tool surface, action codec and observation/context policy. The selected ToolSurface's capabilities and schema compatibility are validated together. `memory_policy` is `disabled`.

Supported runtime kinds are `synthetic`, `robotwin`, `libero_pro` and `robodojo`. LIBERO and LIBERO-Pro both use `libero_pro` runtime kind with different source-paired task catalogs. Runtime inputs identify transport, simulator interpreter/source, task/seed/layout, renderer, control/motion profiles and execution resource as applicable.

Budgets bound robot decisions, model calls, simulation steps and wall time. Backend-specific limits may further restrict actions or physics steps. A model context window must leave a safe input margin after its output budget. Persistent rounds are explicit session inputs; they are not implied by task identity.

## Fresh benchmark specifications

Use the maintained builders rather than copying a resolved plan:

```bash
uv run --locked python -m ora.orchestrator.benchmark_spec tasks robotwin
uv run --locked python -m ora.orchestrator.benchmark_spec tasks libero_pro
uv run --locked python -m ora.orchestrator.robodojo_spec --help
```

The builders take task/provider/host/resource inputs, exact qualification and service reports where required, and an explicit execution namespace/cohort. The default resolver selects `native_full_explicit_motion_direct_v4`. A deliberate custom surface or harness-change overlay is recorded as such; registration alone does not make it the default.

The [environment pages](../support.md) show one-cell recipes. `ora baselines --explain` previews the selected baseline and tool bindings. `ora plan RUN_SPEC` shows the source, hashes, placement and budgets before launch.

## Identity and policy assets

`source` binds the exact Git commit and dirty state. Launch requires clean committed source. `policy_identity` records the policy-lock digest and model-visible digest. The plan identity also binds resolved configuration, source/version contracts and content references. Schema strings and asset paths are serialized data, so namespace migration changes hashes even when motion behavior remains the same.

`policy.lock` pins tracked policy assets and catalogs in canonical sha256sum format. Run `ora lock` after intentional policy edits and commit the lock together. `ora lock --check` detects drift; `ora lock --strip-pins` maintains path-only assets and compiled embodiment manifests. Composed tools have their own compiler identity and must be regenerated with `python -m ora.agent.tools.composed compile` when their source/schema changes.

Host profiles name environment variables, not credential values or machine-local paths. Copy `examples/hosts/example-host.yaml` to an ignored/local host directory and set `ORA_HOSTS_DIR` and `ORA_EXECUTION_HOST`. Exact simulator slots are registered in that profile. Host/resource placement is distinct from policy-visible composition.

## Filesystem roots

Source checkouts contain tracked code/assets. A runtime workspace holds `.harness-runtime/` leases and the reconstructable run index. By default linked Git worktrees share the primary checkout's runtime workspace. `ORA_WORKSPACE_ROOT` explicitly selects an absolute workspace. `ORA_OUTPUTS_ROOT` selects an existing shared evidence root; `ORA_EXTRA_OUTPUT_ROOTS` supplies additional read-only discovery roots. Use `ora workspace` to inspect actual resolution.

Logical run namespaces are `outputs/<experiment>/`; a run is keyed by its exact run ID. Keep generated artifacts, qualification, seeds, credentials and media outside Git. The Synthetic quickstart writes `outputs/quickstart/runs/<run-id>/` beneath its selected evidence root.

## Episode evidence

The run directory binds `plan.json`, `preflight.json`, `launch.json`, process identity, canonical events, policy-state/trajectory evidence, public observations, action proposals and measured receipts. `result.json` records termination/state and the official environment verdict. Provider requests/responses and content-addressed artifacts are archived for reconstruction; exact filenames and schemas are owned by the evidence modules and validated by their tests.

Derived metrics, context manifests/audits and behavioral-ownership reports do not replace raw evidence. SFT export is a derived provider-request projection. Original capture and reconstructed replay are separately labeled in the viewer.

Evaluation validates source/action/trace/artifact boundaries before sealing. A seal binds the immutable inventory and final trace identity. A failed seal is recorded as a refusal. Later revalidation writes separate reports without changing the original run. Synthetic's `official_success` remains `null`; only a benchmark's official boundary may supply a boolean.

## Credentials and launch

Credential environment variables, TLS trust and simulator roots are local operator configuration. Preflight checks configuration/qualification but never turns an installation into benchmark success evidence. `ora doctor` reports credential presence and setup discovery only; it does not authenticate, render, initialize CUDA or qualify physics.

`ora run --env synthetic` uses `ORA_API_KEY` (or the chosen provider's conventional key variable), `ORA_BASE_URL`, `ORA_MODEL`, `ORA_PROVIDER`, optional `ORA_REASONING_EFFORT` and `ORA_SUPPORTS_IMAGES`. Its default limits are explicit in `quickstart.py` and documented in the [quickstart](../quickstart.md). The benchmark builders use their selected provider profiles instead.

`ora launch RUN_SPEC` requires exact preflight and confirmation; `--yes` is explicit non-interactive authorization for that plan. `ora run --env synthetic` starts only its bounded Synthetic plan. Retry cannot rewrite a sealed run or repeat an attempt that already dispatched a robot action.

## Local model APIs and gateways

`loopback_http_v1` admits a direct HTTP model API on localhost. Credentials come from the policy's named environment variables. Set `gateway_client_binding: true` only for a CLIProxyAPI gateway whose config/client files should be loaded and checked; the bundled gateway profiles select it explicitly. The default is no gateway file binding. HTTPS profiles cannot select gateway binding.

Synthetic runs in process, so its scratch directory is checked for capacity and writability without an unused Unix RPC socket path limit. Isolated simulator workers retain that limit.

## Path-only asset references

Composable policy assets name owned components by repository-relative path. `policy.lock` supplies their current bytes and content hashes; do not duplicate mutable owned-asset SHA literals inside a composable record. External source/model pins, reviewed evidence and generated compiled manifests retain their distinct identities. A reference escaping the tree, absent from the lock or using an inadmissible inline pin is refused. Regenerate compiled artifacts and the lock deliberately after changing their inputs.

## Storage admission floors

Synthetic defaults to 1 GiB free in the evidence workspace and 0.25 GiB in scratch. Isolated robot runtimes retain the 20 GiB workspace and 2 GiB scratch defaults. A host profile's explicit `runtime_storage` floors take precedence for either kind; capacity and writability are still probed. These are admission reserves, not predicted episode sizes.
