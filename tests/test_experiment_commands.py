from __future__ import annotations

from pathlib import Path

from rag_evaluator.commands.experiment_commands import run_experiment


def test_run_experiment_uses_database_path_from_config_when_cli_not_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: unit",
                "run_settings:",
                "  database_path: storage/from-config.duckdb",
                "datasets: []",
                "pipelines: []",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_experiment_from_config(
        *,
        config_path: Path,
        database_path: Path,
        openai_base_url: str | None,
    ):
        captured["config_path"] = config_path
        captured["database_path"] = database_path
        captured["openai_base_url"] = openai_base_url
        return type(
            "Summary",
            (),
            {
                "experiment_name": "unit",
                "config_path": config_path,
                "database_path": database_path,
                "sample_count": 0,
                "document_count": 0,
                "datasets": [],
                "pipeline_runs": [],
            },
        )()

    monkeypatch.setattr(
        "rag_evaluator.commands.experiment_commands.run_experiment_from_config",
        fake_run_experiment_from_config,
    )

    summary = run_experiment(
        config_path=config_path,
        database_path=None,
    )

    assert captured["config_path"] == config_path
    assert captured["database_path"] == Path("storage/from-config.duckdb")
    assert captured["openai_base_url"] is None
    assert summary.database_path == Path("storage/from-config.duckdb")


def test_run_experiment_prefers_cli_database_path_over_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: unit",
                "run_settings:",
                "  database_path: storage/from-config.duckdb",
                "datasets: []",
                "pipelines: []",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_experiment_from_config(
        *,
        config_path: Path,
        database_path: Path,
        openai_base_url: str | None,
    ):
        captured["database_path"] = database_path
        captured["openai_base_url"] = openai_base_url
        return type(
            "Summary",
            (),
            {
                "experiment_name": "unit",
                "config_path": config_path,
                "database_path": database_path,
                "sample_count": 0,
                "document_count": 0,
                "datasets": [],
                "pipeline_runs": [],
            },
        )()

    monkeypatch.setattr(
        "rag_evaluator.commands.experiment_commands.run_experiment_from_config",
        fake_run_experiment_from_config,
    )

    summary = run_experiment(
        config_path=config_path,
        database_path=Path("storage/from-cli.duckdb"),
    )

    assert captured["database_path"] == Path("storage/from-cli.duckdb")
    assert captured["openai_base_url"] is None
    assert summary.database_path == Path("storage/from-cli.duckdb")


def test_run_experiment_forwards_openai_base_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: unit",
                "datasets: []",
                "pipelines: []",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_experiment_from_config(
        *,
        config_path: Path,
        database_path: Path,
        openai_base_url: str | None,
    ):
        captured["database_path"] = database_path
        captured["openai_base_url"] = openai_base_url
        return type(
            "Summary",
            (),
            {
                "experiment_name": "unit",
                "config_path": config_path,
                "database_path": database_path,
                "sample_count": 0,
                "document_count": 0,
                "datasets": [],
                "pipeline_runs": [],
            },
        )()

    monkeypatch.setattr(
        "rag_evaluator.commands.experiment_commands.run_experiment_from_config",
        fake_run_experiment_from_config,
    )

    run_experiment(
        config_path=config_path,
        database_path=Path("storage/from-cli.duckdb"),
        openai_base_url="https://proxy.example/v1",
    )

    assert captured["database_path"] == Path("storage/from-cli.duckdb")
    assert captured["openai_base_url"] == "https://proxy.example/v1"
