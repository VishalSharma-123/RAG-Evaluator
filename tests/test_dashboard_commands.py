from __future__ import annotations

from pathlib import Path

from rag_evaluator.commands.dashboard_commands import launch_dashboard


def test_launch_dashboard_invokes_streamlit_with_database_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, check, env):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "rag_evaluator.commands.dashboard_commands.subprocess.run",
        fake_run,
    )

    summary = launch_dashboard(database_path=Path("results.duckdb"))

    command = captured["command"]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert captured["check"] is False
    assert captured["env"]["RAG_EVALUATOR_DATABASE_PATH"] == "results.duckdb"
    assert summary.database_path == Path("results.duckdb")
    assert summary.return_code == 0
