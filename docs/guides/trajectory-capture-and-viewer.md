# Trajectory capture, inspection and diagnostic replay

Use recorded original pixels to show an execution. Running the physics engine again is a separate diagnostic. ORA does not promise generic full-state restore or deterministic physical replay across simulators. See [evidence formats](../reference/configuration-and-evidence.md).

## Sources and authority

| Source | Starts simulation | Claim |
|---|---|---|
| Original continuous capture | No | Camera pixels actually saved during execution, with declared coverage. |
| Original observations | No | Pixels at observation boundaries; intervening motion may be absent. |
| Numeric controller replay | Yes | Diagnostic execution of recorded control inputs; contact/object equivalence still requires validation. |
| Historical semantic replay | Yes | A supported historical action format executed with its source-bound code/dependencies. |

RGB snapshots do not include complete qpos/qvel, contacts, solver/controller state or RNG state. Diagnostic state files are comparison evidence, not a public snapshot/restore protocol. MP4 is a lossy presentation; use source PNG and pixel hashes for pixel comparisons. Reconstructed media never replaces the original official verdict.

## Shared implementation

`ora/environment/execution_capture.py` owns recording; backend execution-capture mixins supply real post-step RGB, simulation clocks and round/action boundaries. RoboTwin, LIBERO and RoboDojo have backend adapters. Synthetic has no physical capture claim. Keep simulator imports in isolated backend workers.

`ora/environment/controller_replay.py` owns numeric journals and their validated readers. `viewer/execution_capture.py` validates original sources; `viewer/trajectory.py` compiles execution video; `viewer/process_timeline.py` and `process_video.py` join policy events by exact identities. Reuse these modules instead of writing another experiment-specific recorder/compiler.

## Capture contract

Maintained robot defaults resolve execution capture as enabled, 10 FPS and a 32-entry queue. FPS accepts integers 1–30 and capacity 4–32; explicit disable is retained. Verify the resolved plan and worker binding. Sampling must not add physics steps, change controls/hold duration or perturb random lighting/RNG.

Each round starts its own simulation clock after reset/warmup; action and terminal boundaries request frames. One recorder retains session-global action IDs across rounds. Do not concatenate rounds by wall clock or filename order. Finalize media before sealing and verify any worker acknowledgment.

Ordinary queue overflow is a recorded drop. Boundary frames reserve two entries and wait at most 250 ms; pixel copies are bounded at 8 MiB. Writer finalization allows ten seconds to drain and does not accept late appended frames. Inspect attempted/written/unwritten counts, drop reasons, errors and reset/terminal coverage. `complete` means the recorder's coverage contract passed, not every physics step was imaged or the task succeeded.

The run retains plan/reset/source bindings, controller journal and NPZ arrays, capture manifest, timeline and lossless PNG frames alongside normal trace, observations, receipts and seal. Read paths from manifests; do not assume every backend uses the same subdirectory.

## Compile original media

```bash
TRAJECTORY_OUTPUTS=/absolute/path/to/outputs
TRAJECTORY_RUN_ID=replace-with-exact-run-id
./scripts/dev/compile-viewer-videos --outputs-root "$TRAJECTORY_OUTPUTS" \
  --run-id "$TRAJECTORY_RUN_ID" --video-mode execution --replay-mode review
./scripts/open-viewer --outputs-root "$TRAJECTORY_OUTPUTS"
```

Source preference is finalized original capture, supported same-session frames, then original observations. A running capture uses observations until finalized. Observation-only video labels missing intermediate motion. Simulation time controls resampling; held frames show their age. Agent thinking often pauses physics, so video length is not episode wall time.

Derived MP4/timelines/manifests live under `outputs/<experiment>/viewer-derived/<run_id>/`, outside sealed run directories. Verify source kind, fidelity label, capture status, hashes and frame count. Normal Viewer playback must not initialize a simulator. An uncached explicit replay may return pending/503 until its separate diagnostic finishes.

To add recorded model output/context cards or select a recorded round:

```bash
./scripts/dev/compile-viewer-videos --outputs-root "$TRAJECTORY_OUTPUTS" \
  --run-id "$TRAJECTORY_RUN_ID" --video-mode process \
  --include-reasoning --include-context --card-seconds 2.5
./scripts/dev/compile-viewer-videos --outputs-root "$TRAJECTORY_OUTPUTS" \
  --run-id "$TRAJECTORY_RUN_ID" --video-mode execution --round 1
```

The round CLI is one-based; stored `round_index` is zero-based. These options display only recorded content. Exact action IDs join process chapters; unmatched native child actions remain explicitly unplaced, never joined by nearby time. Chapter indices refer to resampled output frames.

## Diagnostic replay

Use `--replay-mode forensic` only after inspecting exact source, interpreter, assets, controller and GPU/slot availability. It allows replay but still prefers existing original capture. Do not delete/hide originals to force reconstruction. Numeric replay executes every recorded causal control and settle step in order; camera sampling may be sparse, controls may not. It must not ask the policy or perception service to replan the recorded dense controls.

There is no generic automatic free-slot scheduler for all replay paths. A media file lock is not a Runtime resource lease. Preserve explicit bindings/logs, avoid live-job conflicts and allow one builder per run/media identity. Independent runs may use independently qualified slots; one world's control sequence remains serial.

Copy complete runs with their referenced relative evidence and exact bytes when moving machines. MP4 alone is not an auditable run. Original playback needs no simulator; replay does. Duplicate run IDs across output roots are rejected; select the unique source instead of changing IDs or seals. Missing/corrupt source hashes must fail or display incomplete coverage, never silently trigger simulation or fabricated original media.
