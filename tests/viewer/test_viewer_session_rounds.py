"""Viewer contracts for one persistent AgentSession spanning reset rounds."""

from __future__ import annotations

import json
from pathlib import Path

from viewer.app import _trajectory_page
from viewer.registry import register_run
from viewer.rounds import render_round_matrix
from viewer.trajectory import _render_trajectory_frame, compile_trajectory


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_viewer_projects_persistent_k_session_without_inventing_aggregate_verdict(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "persistent-k2"
    _write(run / "plan.json", {
        "experiment": "session-k-test", "config": "persistent-k2",
        "milestone": "SESSION-K", "task": "pick", "seed": 0,
        "evidence_tier": "behavioral",
        "runtime": {"kind": "libero_pro", "suite": "libero_object_swap", "task_id": 0},
        "policy": {"kind": "native_api", "model": "test-model"},
        "source": {"commit": "source-commit"},
    })
    _write(run / "status.json", {"state": "completed", "official_success": None})
    _write(run / "evaluation.json", {"status": "valid", "official_success": None})
    _write(run / "seal.json", {})
    _write(run / "result.json", {
        "schema": "ora.session-result.v1",
        "state": "completed", "termination": "session_round_limit",
        "official_success": None, "decisions": 3, "model_calls": 7,
        "refused_proposals": 2, "round_count": 2, "round_success_count": 1,
        "session": {
            "cadence": "persistent_agent_k_rounds_v1",
            "reset_owner": "supervisor", "budget_scope": "per_round",
        },
        "rounds": [
            {
                "round_index": 0, "state": "completed",
                "termination": "trusted_runtime_boundary",
                "official_success": True, "decisions": 1, "model_calls": 2,
                "wall_duration_s": 10.0,
                "policy_state_steps": {"start": 0, "end_exclusive": 2},
                "verdict": {"sim_steps": 40},
            },
            {
                "round_index": 1, "state": "completed",
                "termination": "round_budget_exhausted",
                "official_success": False, "decisions": 2, "model_calls": 5,
                "refused_proposals": 2,
                "wall_duration_s": 20.0,
                "policy_state_steps": {"start": 2, "end_exclusive": 4},
                "verdict": {"sim_steps": 80},
            },
        ],
    })
    manifest = []
    for step in range(4):
        relative = f"policy-state/steps/step-{step:04d}.json"
        _write(run / relative, {
            "step": step, "cause": "reset" if step in {0, 2} else "after_action",
            "task_instruction": "Pick the object.", "main_camera": "agentview",
            "observation": {"id": f"obs-{step}"}, "cameras": {},
        })
        manifest.append({"step": step, "path": relative})
    _write(run / "policy-state/manifest.json", {"steps": manifest})

    trajectory = compile_trajectory(run)
    assert trajectory["view_kind"] == "persistent_multi_round_session"
    assert trajectory["summary"]["official_success"] is None
    assert trajectory["session_rounds"]["schema"] == (
        "ora.viewer-session-rounds.v1"
    )
    assert trajectory["session_rounds"]["round_success_count"] == 1
    assert trajectory["session_rounds"]["success_within_k"] is True
    assert trajectory["session_rounds"]["all_rounds_succeeded"] is False
    assert trajectory["session_rounds"]["aggregate_official_success"] is None
    assert trajectory["session_rounds"]["integrity"] == {
        "status": "valid", "issues": [],
    }
    assert [step["round"]["round_ordinal"] for step in trajectory["steps"]] == [
        1, 1, 2, 2,
    ]
    assert [step["round"]["local_step_index"] for step in trajectory["steps"]] == [
        0, 1, 0, 1,
    ]

    # Refused proposals spend no decision and are projected beside them.
    assert trajectory["summary"]["refused_proposals"] == 2
    registration = register_run(run)
    assert registration.category == "formal"
    matrix = render_round_matrix([registration])
    assert "2 actions, 2 refused" in matrix and "1 actions<" in matrix
    assert registration.official_success is None
    assert registration.session_cadence == "persistent_agent_k_rounds_v1"
    assert (registration.round_success_count, registration.round_count) == (1, 2)

    rendered = _trajectory_page(run.name, trajectory)
    assert "Persistent K-session notebook" in rendered
    assert "One AgentSession, 2 reset EnvironmentSessions." in rendered
    assert "1/2 rounds" in rendered
    assert "aggregate <code>official_success</code> intentionally remains null" in rendered
    assert "Supervisor reset follows; AgentSession continues" in rendered
    assert _render_trajectory_frame(run, trajectory, 2).size == (1280, 720)


def test_a_scheduled_round_shows_its_layout_and_the_relaunch(tmp_path: Path) -> None:
    from viewer.process_timeline import _scene_note
    from viewer.rounds import round_ledger, scene_label

    run = tmp_path / "runs" / "scheduled-k2"
    _write(run / "plan.json", {"session": {"cadence": "persistent_agent_k_rounds_v1", "rounds": 2}})
    _write(run / "status.json", {"state": "completed"})
    scenes = [{"layout_id": 21, "layout_sha256": "19f00f56baad" + "0" * 52, "scene_reset_mode": "initial_launch"},
              {"layout_id": 37, "layout_sha256": "d57e33334ced" + "0" * 52,
               "scene_reset_mode": "simulation_relaunch"}]
    _write(run / "result.json", {"state": "completed", "rounds": [
        {"round_index": index, "state": "completed", "official_success": False, "decisions": 1, "scene": scene}
        for index, scene in enumerate(scenes)]})
    rows = round_ledger(run)["rounds"]
    assert [scene_label(row["scene"]) for row in rows] == ["layout 21 · 19f00f56", "layout 37 · d57e3333"]
    assert _scene_note(rows[0]["scene"]) == " · layout 21 · 19f00f56"
    assert _scene_note(rows[1]["scene"]) == " · simulation relaunched on layout 37 · d57e3333"
    # An unscheduled round names no layout.
    assert scene_label(None) is None and _scene_note(None) == ""
