from __future__ import annotations

from pathlib import Path

from rag_evaluator.application.types import (
    DatasetLoadSummary,
    ExperimentRunSummary,
    PersistedPipelineRunSummary,
    SyntheticGenerationSummary,
)
from rag_evaluator.cli import main as cli_main
from rag_evaluator.commands.handlers import (
    handle_generate_synthetic,
    handle_run_experiment,
)
from rag_evaluator.commands.main import main
from rag_evaluator.commands.parser import build_parser


def test_cli_main_is_compatibility_alias() -> None:
    from rag_evaluator.commands.main import main as commands_main

    assert cli_main is commands_main


def test_build_parser_supports_generate_synthetic() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "generate-synthetic",
            "--chunks",
            "chunks.jsonl",
            "--output",
            "synthetic.jsonl",
            "--question-type",
            "factoid",
            "--question-type",
            "unanswerable",
        ]
    )

    assert args.command == "generate-synthetic"
    assert args.provider == "openrouter"
    assert args.question_types == ["factoid", "unanswerable"]
    assert args.handler is handle_generate_synthetic


def test_build_parser_supports_run_experiment() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "experiment.yaml",
            "--database-path",
            "results.duckdb",
        ]
    )

    assert args.command == "run-experiment"
    assert args.config == Path("experiment.yaml")
    assert args.database_path == Path("results.duckdb")
    assert args.handler is handle_run_experiment


def test_build_parser_leaves_run_experiment_database_path_unset_by_default() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "experiment.yaml",
        ]
    )

    assert args.command == "run-experiment"
    assert args.config == Path("experiment.yaml")
    assert args.database_path is None


def test_main_dispatches_generate_synthetic(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_synthetic(**kwargs):
        captured.update(kwargs)
        return SyntheticGenerationSummary(
            chunks_path=Path("chunks.jsonl"),
            output_path=Path("synthetic.jsonl"),
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            chunk_count=1,
            sample_count=1,
        )

    monkeypatch.setattr(
        "rag_evaluator.commands.handlers.generate_synthetic",
        fake_generate_synthetic,
    )

    exit_code = main(
        [
            "generate-synthetic",
            "--chunks",
            "chunks.jsonl",
            "--output",
            "synthetic.jsonl",
            "--question-type",
            "factoid",
        ]
    )

    assert exit_code == 0
    assert captured["chunks_path"] == Path("chunks.jsonl")
    assert captured["output_path"] == Path("synthetic.jsonl")
    assert captured["question_types"] == ["factoid"]


def test_main_dispatches_run_experiment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_experiment(*, config_path: Path, database_path: Path) -> ExperimentRunSummary:
        captured["config_path"] = config_path
        captured["database_path"] = database_path
        return ExperimentRunSummary(
            experiment_name="unit",
            config_path=config_path,
            database_path=database_path,
            sample_count=3,
            document_count=2,
            datasets=[DatasetLoadSummary(dataset_name="tiny", sample_count=3)],
            pipeline_runs=[
                PersistedPipelineRunSummary(
                    pipeline_name="pipeline-1",
                    run_id="run-1",
                    result_count=3,
                    chunk_count=4,
                    runtime_metadata={"runtime": "ok"},
                )
            ],
        )

    monkeypatch.setattr(
        "rag_evaluator.commands.handlers.run_experiment",
        fake_run_experiment,
    )

    exit_code = main(
        [
            "run-experiment",
            "experiment.yaml",
            "--database-path",
            "results.duckdb",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == Path("experiment.yaml")
    assert captured["database_path"] == Path("results.duckdb")


def test_generate_synthetic_handler_calls_input_service(monkeypatch, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate-synthetic",
            "--chunks",
            "chunks.jsonl",
            "--output",
            "synthetic.jsonl",
            "--question-type",
            "factoid",
            "--max-samples",
            "2",
            "--reasoning-enabled",
        ]
    )
    captured: dict[str, object] = {}

    def fake_generate_synthetic(**kwargs) -> SyntheticGenerationSummary:
        captured.update(kwargs)
        return SyntheticGenerationSummary(
            chunks_path=kwargs["chunks_path"],
            output_path=kwargs["output_path"],
            provider=str(kwargs["provider"]),
            model=str(kwargs["model"]),
            chunk_count=1,
            sample_count=2,
        )

    monkeypatch.setattr(
        "rag_evaluator.commands.handlers.generate_synthetic",
        fake_generate_synthetic,
    )

    exit_code = handle_generate_synthetic(args)

    assert exit_code == 0
    assert captured["question_types"] == ["factoid"]
    assert captured["max_samples"] == 2
    assert "Generated 2 synthetic samples from 1 chunks" in capsys.readouterr().out


def test_run_experiment_handler_calls_experiment_service(monkeypatch, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-experiment",
            "experiment.yaml",
            "--database-path",
            "results.duckdb",
        ]
    )
    captured: dict[str, object] = {}

    def fake_run_experiment(*, config_path: Path, database_path: Path) -> ExperimentRunSummary:
        captured["config_path"] = config_path
        captured["database_path"] = database_path
        return ExperimentRunSummary(
            experiment_name="unit",
            config_path=config_path,
            database_path=database_path,
            sample_count=3,
            document_count=2,
            datasets=[DatasetLoadSummary(dataset_name="tiny", sample_count=3)],
            pipeline_runs=[
                PersistedPipelineRunSummary(
                    pipeline_name="pipeline-1",
                    run_id="run-1",
                    result_count=3,
                    chunk_count=4,
                    runtime_metadata={"runtime": "ok"},
                )
            ],
        )

    monkeypatch.setattr(
        "rag_evaluator.commands.handlers.run_experiment",
        fake_run_experiment,
    )

    exit_code = handle_run_experiment(args)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["config_path"] == Path("experiment.yaml")
    assert captured["database_path"] == Path("results.duckdb")
    assert "Loaded 3 samples from dataset `tiny`" in output
    assert "Completed pipeline `pipeline-1`" in output
