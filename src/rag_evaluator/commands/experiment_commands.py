from __future__ import annotations

from pathlib import Path

from rag_evaluator.application import run_experiment_from_config
from rag_evaluator.application.types import ExperimentRunSummary
from rag_evaluator.io import load_experiment_config

DEFAULT_DATABASE_PATH = Path("storage/results.duckdb")


def run_experiment(
    *,
    config_path: Path,
    database_path: Path | None,
    openai_base_url: str | None = None,
) -> ExperimentRunSummary:
    resolved_database_path = _resolve_database_path(
        config_path=config_path,
        cli_database_path=database_path,
    )
    return run_experiment_from_config(
        config_path=config_path,
        database_path=resolved_database_path,
        openai_base_url=openai_base_url,
    )


def _resolve_database_path(
    *,
    config_path: Path,
    cli_database_path: Path | None,
) -> Path:
    if cli_database_path is not None:
        return cli_database_path

    experiment = load_experiment_config(config_path)
    run_settings_database_path = experiment.run_settings.database_path
    if run_settings_database_path:
        return Path(run_settings_database_path)

    return DEFAULT_DATABASE_PATH
