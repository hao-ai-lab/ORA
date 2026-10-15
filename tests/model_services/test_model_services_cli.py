"""The model-services entry dispatches to the shared operator with the host's roots."""
from __future__ import annotations

import json

from ora.orchestrator import model_services_cli as cli
from ora.orchestrator import paths


def test_commands_dispatch_with_output_storage_and_runtime_roots(tmp_path, monkeypatch, capsys):
    calls = []

    def lifecycle(command, **kwargs):
        calls.append((command, kwargs))
        return {"segmentation": {"ready": command != "start"}, "grasp_proposals": {"ready": True}}

    def qualify(directory):
        calls.append(("qualify", directory))
        return directory / "report.json"

    monkeypatch.setattr(cli.operator, "lifecycle", lifecycle)
    monkeypatch.setattr(cli.operator, "qualify", qualify)
    # The state lives under the configured output storage, as every other
    # experiment workspace does, not under the checkout.
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "storage")

    assert cli.main(["start"]) == 0
    started = json.loads(capsys.readouterr().out)
    assert started["segmentation"]["ready"] is False
    assert cli.main(["status"]) == 0
    assert cli.main(["stop", "--state", str(tmp_path / "custom")]) == 0
    assert cli.main(["qualify"]) == 0
    assert capsys.readouterr().out.strip().endswith("qualification/report.json")

    assert [call[0] for call in calls] == ["start", "status", "stop", "qualify"]
    default_state = (tmp_path / "storage" / "model-services").resolve()
    assert calls[0][1] == {"root": paths.PROJECT_ROOT, "state": default_state,
                           "runtime_state_root": paths.STATE_ROOT}
    assert calls[2][1]["state"] == tmp_path / "custom"
    assert calls[3][1] == default_state / "qualification"


def test_status_exits_nonzero_until_both_services_are_ready(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path)
    monkeypatch.setattr(cli.operator, "lifecycle", lambda command, **kwargs: {
        "segmentation": {"ready": False, "error": "ConnectionError"},
        "grasp_proposals": {"ready": True}})
    assert cli.main(["status"]) == 1
    assert json.loads(capsys.readouterr().out)["segmentation"]["error"] == "ConnectionError"
