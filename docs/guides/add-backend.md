# Add an environment backend

An environment implements the session contract under `ora/environment/` and registers a descriptor/factory. Follow [environment compatibility authoring](environment-compatibility-authoring.md) before adding prompts, tools or embodiment assets.

1. Audit the upstream task/evaluator source, units, frames, public constants, reset/step/termination and official verdict. Record source/asset versions. Unknown required inputs block the affected capability.
2. Keep identities, suite expansion and source contracts host-safe in `environment/benchmarks/`. Simulator imports belong in an isolated backend worker.
3. Implement public observations, action execution and measured receipts through the environment session contract. Route mutation through the Supervisor; never expose private evaluator state to the policy.
4. Bind the reviewed embodiment, prompt bundle and ToolSurface, including units, refusal constraints and context projection. Update the default resolver for applicable maintained consumers and the current reference together.
5. Add targeted fake-worker contract tests and the smallest meaningful qualified simulator smoke. Record dependency/version coverage; do not reuse physical evidence across an unreviewed invalidating change.
6. Add an English installation and one-cell recipe, doctor discovery bindings, support-matrix row and issue-template guidance.

Use the existing Synthetic backend for transport/lifecycle contracts and one maintained robot backend for the isolated interpreter pattern. Run the relevant tests, regenerate `policy.lock` when needed, and compare model-visible composition before submitting a PR.
