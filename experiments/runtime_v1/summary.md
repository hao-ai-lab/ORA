# Reading Runtime v1 evidence

See the [ladder README](README.md) for milestones and operator commands. Generated evidence is kept outside Git.

A completed, evaluator-valid, sealed run can satisfy its Runtime gate. `official_success` is a separate value supplied by the selected benchmark; Synthetic uses null. M1 is an infrastructure check, M2 remains diagnostic, and M3–M5 exercise manipulation. The ladder is behaviorally accepted only after a valid completed Native M5 run reports official success.

A gate binds its source commit and exact run; it does not rewrite historical results or independently certify another simulator source, task, seed, controller or asset tree. Review those dependencies when reusing qualification.
