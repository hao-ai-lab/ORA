"""Four single-attempt, public-fixture prepared-grasp development diagnostics.

The producer uses the shared tool runtime, Supervisor executor, capture and
resource leases. The checker never constructs an environment. Imported poses
are explicit operator qualification fixtures, not fresh policy proposals.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

SOURCE = Path(__file__).resolve().parents[2]
SCHEMA = "ora.robotwin-prepared-grasp-qualification.v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, ensure_ascii=False)


def git(*args: str, cwd: Path = SOURCE) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def clean_source() -> str:
    if git("status", "--porcelain"):
        raise ValueError("qualification requires a clean committed checkout")
    return git("rev-parse", "HEAD")


def qualification_root(path: Path, source_run: Path | None = None) -> Path:
    path = path.resolve()
    if "runs" in path.parts or (source_run and path.is_relative_to(source_run.resolve())):
        raise ValueError("qualification output must be outside formal run directories")
    return path


def closed_start_proof(setup: dict[str, Any], effector: dict[str, Any]) -> dict[str, Any]:
    """A closed jaw has zero measured aperture, independent of actuator scale.

    ALOHA's normalized actuator interval includes negative finger travel below
    the physical closed stop. Its measured normalized value at zero aperture is
    therefore not zero. Do not invent an actuator-normalization arrival gate.
    """
    record = setup.get("receipt", setup)
    aperture = effector.get("gripper_opening_m")
    finite_aperture = isinstance(aperture, (int, float)) and not isinstance(aperture, bool) and math.isfinite(aperture)
    passed = (record.get("consumed") is True and record.get("execution_status") == "executed"
              and effector.get("gripper_command_target") == "close"
              and finite_aperture and 0 <= aperture <= 1e-6)
    return {"measured_opening_normalized": effector.get("gripper_state", {}).get("measured_opening_normalized"),
            "measured_jaw_separation_m": aperture, "passed": passed,
            "basis": "executed close directive and measured zero jaw aperture (1 micrometre numerical allowance)"}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from ora.environment.factory import validate_robotwin_control_binding
    from ora.orchestrator.resources import resolve_execution_resource

    commit = clean_source()
    oldroot = args.source_run.resolve()
    output = qualification_root(args.output, oldroot)
    oldplan = json.loads((oldroot / "plan.json").read_text())
    if oldplan["runtime"]["kind"] != "robotwin":
        raise ValueError("source fixture must be a RoboTwin run")
    preflight = json.loads((oldroot / "preflight.json").read_text())
    for name in ("seed_qualification", "robotwin_control_binding", "robotwin_revision"):
        if not any(c["check"] == name and c["passed"] for c in preflight["checks"]):
            raise ValueError(f"source lacks passing {name}")
    runtime = copy.deepcopy(oldplan["runtime"])
    host = runtime.get("required_host")
    marker = os.environ.get("ORA_EXECUTION_HOST")
    if not host or marker != host:
        raise ValueError("qualified execution host marker is required")
    if not os.access(args.robotwin_python, os.X_OK):
        raise ValueError("isolated RoboTwin interpreter is not executable")
    environment = {
        runtime["root_env"]: str(args.robotwin_root.resolve()),
        runtime["python_env"]: str(args.robotwin_python.absolute()),
        "ORA_EXECUTION_HOST": marker,
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    }
    if git("rev-parse", "HEAD", cwd=args.robotwin_root) != runtime["required_revision"]:
        raise ValueError("RoboTwin fixture revision mismatch")
    if git("status", "--porcelain", "--untracked-files=no", cwd=args.robotwin_root):
        raise ValueError("RoboTwin source is dirty")
    validate_robotwin_control_binding(runtime, action_codec="motion_action_v3")
    for key in ("control_profile", "public_geometry_descriptor"):
        if digest(SOURCE / runtime[key]["path"]) != runtime[key]["sha256"]:
            raise ValueError(f"fixture {key} changed")
    dependencies = {
        str(p.relative_to(SOURCE)): digest(p)
        for folder in ("ora",)
        for p in (SOURCE / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".md"}
    }
    for name in (str(Path(__file__).relative_to(SOURCE)),
                 "ora/orchestrator/action_executor.py"):
        dependencies[name] = digest(SOURCE / name)
    for key in ("control_profile", "public_geometry_descriptor"):
        dependencies[runtime[key]["path"]] = runtime[key]["sha256"]
    provider_binding = None
    if args.provider_profile:
        from experiments.runtime_v1.provider_prepared_grasp_probe import load_probe_binding
        provider_binding = load_probe_binding(args.provider_profile)
        if args.provider_profile != provider_binding["profile_id"]:
            raise ValueError("only the selected registered diagnostic profile is supported")
        for name in ("catalog", "probe_source"):
            ref = provider_binding[name]
            dependencies[ref["path"]] = ref["sha256"]
    dependency_hash = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    gpu_rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    resources = {line.split(",")[0].strip(): {
        "free_mib": int(line.split(",")[1]), "utilization_percent": int(line.split(",")[2]),
    } for line in gpu_rows.strip().splitlines()}
    cases = []
    for (arm, query), slots in zip(
        (("left", args.left_query), ("right", args.right_query)),
        (args.slots[:2], args.slots[2:]), strict=True,
    ):
        source = oldroot / "openeta-tool-artifacts/grasp-queries" / query / "manifest.json"
        manifest = json.loads(source.read_text())
        identity = manifest["identity"]
        if identity["compiler"]["arm"] != arm or not identity["observation_id"].endswith("observation-0000"):
            raise ValueError("fixture must bind the selected arm and initial observation")
        for mode, slot in zip(("closed_start", "empty_offset"), slots, strict=True):
            binding = resolve_execution_resource(slot, runtime_kind="robotwin", required_host=host)
            gpu = binding["worker_environment"]["CUDA_VISIBLE_DEVICES"]
            if resources[gpu]["free_mib"] < binding["minimum_free_memory_mib"]:
                raise ValueError(f"insufficient free GPU memory: {slot}")
            candidate = copy.deepcopy(manifest["compiled_candidates"][0])
            original_candidate = copy.deepcopy(candidate)
            if mode == "empty_offset":
                for field in ("contact_pose", "pregrasp_pose"):
                    candidate[field]["position_m"][2] += 0.18
            candidate["candidate_id"] = "grasp-" + hashlib.sha256(
                json.dumps({"candidate": candidate, "mode": mode, "arm": arm}, sort_keys=True).encode()
            ).hexdigest()[:24]
            case = {
                "case_id": f"{arm}-{mode}", "arm": arm, "mode": mode,
                "runtime": {**runtime, "resource_binding": binding},
                "task": oldplan["task"], "seed": oldplan["seed"],
                "source_query": str(source), "source_query_sha256": digest(source),
                "candidate": candidate, "original_candidate": original_candidate,
                "source_identity": identity, "repeats": 1,
                "max_sim_steps": 100, "timeout_s": 1200,
            }
            case["case_key"] = hashlib.sha256((dependency_hash + json.dumps(case, sort_keys=True)).encode()).hexdigest()
            cases.append(case)
    per_gpu: dict[str, int] = {}
    for c in cases:
        gpu = c["runtime"]["resource_binding"]["worker_environment"]["CUDA_VISIBLE_DEVICES"]
        per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
    if any(resources[gpu]["free_mib"] < count * 6144 for gpu, count in per_gpu.items()):
        raise ValueError("aggregate worker memory exceeds measured GPU capacity")
    plan = {
        "schema": SCHEMA, "purpose": "development_diagnostic", "coverage_policy": "representative-v1",
        "producer_source_commit": commit, "producer_dependency_hash": dependency_hash,
        "dependency_closure": dependencies, "checker_hash": digest(Path(__file__)),
        "source_run": str(oldroot), "source_plan_sha256": digest(oldroot / "plan.json"),
        "source_preflight_sha256": digest(oldroot / "preflight.json"),
        "environment": environment, "cases": cases,
        "provider": provider_binding,
        "reuse": "Exact task/seed/controller/assets from passing source preflight; current reset RGB-D binding checked before fixture use. Candidate poses are imported public fixtures, with no new perception-model inference.",
        "parallel_execution": {"cpu_count": os.cpu_count(), "gpus": resources,
            "hostname": subprocess.check_output(["hostname"], text=True).strip(),
            "execution_host_marker": marker,
            "isolated_python": subprocess.check_output([str(args.robotwin_python), "--version"], text=True).strip(),
            "memory": Path("/proc/meminfo").read_text(), "slots": args.slots,
            "shared_setup": "existing isolated interpreter and assets; four independently leased sessions"},
        "estimated_wall_minutes": 22,
        "estimate_basis": "four parallel single-attempt sessions; conservative 1200s per-case cap including 600s startup, plus checking/cleanup; qualified 3.5–5.6GiB residency per worker",
        "nondeterminism_evidence": None,
        "claim_boundary": "operator fixture mechanism diagnostic; optional two-call registered-provider tool diagnostic is not a full PolicySession or task benchmark; no task success or stable holding claim; no automatic replacement",
    }
    save(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "cases": [c["case_id"] for c in cases], "estimated_wall_minutes": 22}


def produce(args: argparse.Namespace) -> dict[str, Any]:
    from ora.agent.context import ContextProjector
    from ora.agent.context.skill_observations import materialize_skill_observations, native_blocks
    from ora.agent.tools.runtime import ManipulationToolRuntime
    from ora.contracts import MOTION_ACTION_V3_SCHEMA
    from ora.environment.factory import build_environment_session
    from ora.orchestrator.action_executor import execute_runtime_action, execute_skill_sequence
    from ora.orchestrator.cli import assign_resource_lease, claim_resource_lease, release_resource_lease

    root = qualification_root(args.output)
    plan = json.loads((root / "plan.json").read_text())
    if clean_source() != plan["producer_source_commit"]:
        raise ValueError("producer source differs from prepared plan")
    for name, expected in plan["dependency_closure"].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"producer dependency drift: {name}")
    case = next(c for c in plan["cases"] if c["case_id"] == args.case)
    if digest(Path(case["source_query"])) != case["source_query_sha256"]:
        raise ValueError("source fixture changed")
    os.environ.update(plan["environment"])
    output = root / case["case_id"]
    output.mkdir(exist_ok=False)
    started = time.time()
    report = {"schema": SCHEMA, "case_id": case["case_id"], "case_key": case["case_key"],
              "producer_source_commit": plan["producer_source_commit"], "status": "incomplete"}
    owner = "robotwin-prepared-grasp-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    lease = session = probe = None

    def interrupted(signum: int, _frame: Any) -> None:
        raise TimeoutError(f"qualification interrupted by signal {signum}")

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.alarm(case["timeout_s"])
    try:
        if plan.get("provider"):
            from experiments.runtime_v1.provider_prepared_grasp_probe import ProviderGraspProbe
            probe = ProviderGraspProbe(plan["provider"], output, case["case_id"])
            report["provider_profile"] = plan["provider"]["profile_id"]
        lease = claim_resource_lease({"runtime": case["runtime"], "plan_sha256": digest(root / "plan.json")}, owner, os.getpid())
        lease = assign_resource_lease(lease, os.getpid())
        save(output / "start.json", {"owner_pid": os.getpid(), "lease_owner": owner, "started_unix_s": started})
        session = build_environment_session(runtime_config=case["runtime"], output_root=output / "native",
            task=case["task"], seed=case["seed"], max_sim_steps=case["max_sim_steps"], action_codec="motion_action_v3")
        save(output / "reset.json", session.start())

        def capture():
            observation, _ = session.observe(output / "observations")
            return observation, output / "observations" / observation["observation_id"]

        observation, observation_root = capture()
        identity = case["source_identity"]
        matches = {key: digest(observation_root / identity["camera"] / ref["name"]) == ref["sha256"]
                   for key, ref in identity["source"]["files"].items()}
        save(output / "initial-source-match.json", matches)
        if not all(matches.values()):
            raise ValueError("historical public candidate does not match fresh reset RGB-D")
        if case["mode"] == "closed_start":
            action = {"schema": MOTION_ACTION_V3_SCHEMA, "action_id": "fixture-initial-close",
                "observation_id": observation["observation_id"],
                "bound_observation_sha256": digest(observation_root / "observation.json"),
                "action": "set_gripper", "arguments": {"targets": {case["arm"]: "close"}}}
            setup = execute_runtime_action(session, action)
            save(output / "initial-close.json", setup)
            observation, observation_root = capture()
            effector = observation["proprioception"]["end_effectors"][case["arm"]]
            proof = closed_start_proof(setup, effector)
            save(output / "setup-proof.json", proof)
            if not proof["passed"]:
                raise ValueError("initial close was not confirmed by public jaw state")
        save(output / "before.json", observation)
        manual = SOURCE / "ora/policy/embodiments/aloha-agilex/manual-public-geometry-v3.md"
        projector = ContextProjector(output / "workspace", output / "control",
            SOURCE / "ora/policy/planners/native/policy-v4.md", manual,
            SOURCE / "ora/policy/core/evidence-action-loop-v5.md", main_camera="head_camera", expose_manual=False)
        projector.publish(observation=observation, observation_root=observation_root,
            task_instruction="Operator-only prepared-grasp qualification fixture",
            remaining_budget={"robot_decisions": 8, "model_calls": 0},
            policy_state={"step": 0, "path": "qualification", "sha256": "qualification", "state_sha256": "qualification"})
        host = ManipulationToolRuntime(control_root=output / "control", manual_path=manual,
            event_path=output / "tool-events.jsonl", workspace_bounds_m=case["runtime"]["workspace_bounds_m"],
            state_root=None, tool_surface_profile="robotwin_toolchain_b2_closed_noservo_v1")
        old_artifacts = Path(plan["source_run"]) / "openeta-tool-artifacts"
        target_ref = identity["target_ref"]
        if case["mode"] == "empty_offset":
            index_path = old_artifacts / "segmented-target-index" / f"{target_ref}.json"
            index = json.loads(index_path.read_text())
            destination = host.artifact_root / index["artifact_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old_artifacts / index["artifact_path"], destination)
            destination = host.artifact_root / "segmented-target-index" / index_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(index_path, destination)
        save(host.artifact_root / "grasp-queries/qualification-fixture/manifest.json", {
            "query_ref": "qualification-fixture", "identity": {"observation_id": observation["observation_id"],
                "target_ref": target_ref, "compiler": identity["compiler"]},
            "compiled_candidates": [case["candidate"]],
            "qualification_provenance": {"source_query": case["source_query"],
                "source_query_sha256": case["source_query_sha256"], "mode": case["mode"], "new_inference": False,
                "target_reference": "original public surface; omitted after closed-start setup to avoid stale geometry"},
        })
        arguments = probe.propose(host, case, observation, observation_root) if probe else {
            "candidate_id": case["candidate"]["candidate_id"],
            "rationale": "Operator qualification fixture: authorized opening, candidate pregrasp and bounded lift",
        }
        save(output / "proposal.json", host.invoke("grasp_execute", arguments))
        from ora.agent.tools.skill_driver import load_commitment
        save(output / "commitment.json", load_commitment(host))
        action = json.loads((output / "control/submitted-action.json").read_text())
        receipt, (observation, observation_root) = execute_skill_sequence(session, action,
            initial_observation=observation, initial_observation_root=observation_root,
            drive_stage=host.drive_skill_stage, build_receipt=host.build_skill_receipt, capture_observation=capture)
        save(output / "receipt.json", receipt)
        save(output / "after.json", observation)
        images = materialize_skill_observations(receipt=receipt, run_root=output,
            workspace=output / "workspace", camera_names=host.surface.view_config["automatic_actual_views"])
        blocks = native_blocks(images, run_root=output)
        # Exercise the real serializer without sending a Provider request.
        from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request
        request = {"model": "offline", "max_tokens": 64, "messages": [{"role": "user", "content": blocks}]}
        wires = [json.dumps(OpenAICompatibleProvider.prepare_request(request)), json.dumps(translate_anthropic_request(request))]
        image_delivery = [{**im, "both_wire_serializers_contain_original": all(
            base64.b64encode((output / im["run_path"]).read_bytes()).decode() in wire for wire in wires)} for im in images]
        save(output / "image-delivery.json", image_delivery)
        if probe:
            report["provider_review"] = probe.review(host, receipt, observation, observation_root, images)
        report.update(status="produced", sequence_status=receipt["status"], stop_reason=receipt["stop_reason"],
            official_task_verdict_claimed=False, final_observation_id=observation["observation_id"])
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        if session is not None:
            try:
                session.close()
                worker = getattr(session, "process", None)
                if worker is not None and worker.poll() is None:
                    raise RuntimeError("owned simulator worker did not exit")
            except Exception as exc:
                report["close_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "incomplete"
        if lease is not None and "close_error" not in report:
            try:
                release_resource_lease(owner, owner_pid=os.getpid(), claim_token=lease["claim_token"])
            except Exception as exc:
                report.update(status="incomplete", close_error=f"lease cleanup: {type(exc).__name__}: {exc}")
        if probe is not None:
            try:
                probe.close()
            except Exception as exc:
                report.update(status="incomplete", provider_close_error=type(exc).__name__)
        report["elapsed_s"] = time.time() - started
        save(output / "result.json", report)
    return report


def check_case(case: dict[str, Any], receipt: dict[str, Any], commitment: dict[str, Any], images: list) -> dict[str, Any]:
    """Check mechanisms separately from whether the selected pose was reachable."""
    expected = ["open", "pregrasp", "approach", "close", "lift", "verify"]
    stages = receipt["stage_receipts"]
    checks = {
        "committed_sequence": receipt["committed_sequence"] == expected,
        "candidate_arm": commitment["arm"] == case["arm"],
        "pregrasp_preserved": commitment["poses"]["pregrasp"] == case["candidate"]["pregrasp_pose"],
        "contact_preserved": commitment["poses"]["approach"] == case["candidate"]["contact_pose"],
        "bounded_lift": abs(commitment["poses"]["lift"]["position_m"][2] - commitment["poses"]["approach"]["position_m"][2] - .03) < 1e-9,
        "ledger_released": receipt["skill_receipt"]["ledger_released"] is True,
    }
    failed = [i for i, s in enumerate(stages) if s["status"] != "executed"]
    checks["stopped_after_unconfirmed_stage"] = not failed or failed[0] == len(stages) - 1
    checks["stage_order"] = [s["stage"] for s in stages] == expected[:len(stages)]
    for s in stages:
        if "action" not in s:
            continue
        action = s["action"]
        checks[f"{s['stage']}_arm_route"] = action["schema"] == "ora-motion-action.v3" and set(action["arguments"]["targets"]) == {case["arm"]}
    checks["baseline_image_presence"] = bool(images) == bool(receipt["skill_receipt"].get("observation_evidence"))
    if images:
        checks["original_baseline_images"] = all(i["both_wire_serializers_contain_original"] for i in images)
        checks["baseline_arm_camera"] = [i["camera"] for i in images] == ["head_camera", f"{case['arm']}_camera"]
        checks["baseline_is_historical"] = all(i["observation_id"] != receipt["fresh_observation_id"] for i in images)
    verification = receipt["skill_receipt"].get("verification") or {}
    probe = verification.get("lift_probe")
    if receipt["status"] == "completed":
        checks["measured_lift_reported"] = isinstance(probe, dict) and isinstance(probe.get("measured_tcp_lift_m"), (int, float))
    return {"case_id": case["case_id"], "checks": checks, "lift_probe": probe,
        "mechanism_checks_passed": all(checks.values()), "sequence_status": receipt["status"],
        "physical_coverage": "all_stages" if receipt["status"] == "completed" else "prefix_only",
        "stop_reason": receipt["stop_reason"], "holding_success_claimed": False}


def check(args: argparse.Namespace) -> dict[str, Any]:
    root = qualification_root(args.output)
    plan = json.loads((root / "plan.json").read_text())
    results = []
    for case in plan["cases"]:
        directory = root / case["case_id"]
        report = json.loads((directory / "result.json").read_text())
        if report["status"] != "produced" or report.get("close_error"):
            result = {"case_id": case["case_id"], "status": "incomplete", "error": report.get("error"),
                      "physical_coverage": "setup_only", "original_result_sha256": digest(directory / "result.json")}
            setup_path = directory / "initial-close.json"
            if case["mode"] == "closed_start" and setup_path.exists():
                frames = sorted((directory / "observations").glob("*/observation.json"))
                if frames:
                    last = json.loads(frames[-1].read_text())
                    result["setup_revalidation"] = closed_start_proof(json.loads(setup_path.read_text()),
                        last["proprioception"]["end_effectors"][case["arm"]])
                    result["setup_evidence_sha256"] = {str(p.relative_to(directory)): digest(p) for p in (setup_path, frames[-1])}
            results.append(result)
            continue
        result = check_case(case, json.loads((directory / "receipt.json").read_text()),
            json.loads((directory / "commitment.json").read_text()), json.loads((directory / "image-delivery.json").read_text()))
        before = json.loads((directory / "before.json").read_text())
        after = json.loads((directory / "after.json").read_text())
        result["public_robot_state"] = {"before": before["proprioception"], "after": after["proprioception"]}
        frames = [json.loads(p.read_text()) for p in (directory / "observations").glob("*/observation.json")]
        result["public_measurements"] = [{"observation_id": frame["observation_id"],
            "selected_arm": frame["proprioception"]["end_effectors"][case["arm"]]} for frame in frames]
        result["original_artifact_sha256"] = {name: digest(directory / name) for name in (
            "receipt.json", "commitment.json", "before.json", "after.json", "image-delivery.json", "initial-source-match.json", "result.json")}
        results.append(result)
    report = {"schema": SCHEMA, "producer_source_commit": plan["producer_source_commit"],
        "checker_source_commit": git("rev-parse", "HEAD"), "checker_hash": digest(Path(__file__)),
        "checker_worktree_dirty": bool(git("status", "--porcelain")),
        "plan_sha256": digest(root / "plan.json"), "cases": results,
        "claim_boundary": "separate mechanism correctness and incomplete physical coverage; no official success claim"}
    destination = qualification_root(args.report_output) if args.report_output else root / "review.json"
    save(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("source-run", "output", "robotwin-root", "robotwin-python"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--left-query", required=True)
    p.add_argument("--right-query", required=True)
    p.add_argument("--slots", nargs=4, required=True)
    p.add_argument("--provider-profile", choices=["kimi_direct_k3_anthropic_high_v1", "deepseek_flash_openai_low_16k_v1"])
    p = sub.add_parser("produce")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", required=True)
    p = sub.add_parser("check")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "produce": produce, "check": check}[args.command](args), ensure_ascii=False))


if __name__ == "__main__":
    main()
