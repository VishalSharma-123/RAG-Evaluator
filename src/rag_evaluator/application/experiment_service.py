from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rag_evaluator.application.llm_overrides import apply_openai_base_url
from rag_evaluator.application.experiment_inputs import load_experiment_inputs
from rag_evaluator.application.types import (
    ExperimentRunSummary,
    PersistedPipelineRunSummary,
)
from rag_evaluator.execution import run_experiment as execute_experiment
from rag_evaluator.io import load_experiment_config
from rag_evaluator.persistence import DuckDBResultsStore


def run_experiment_from_config(
    config_path: Path,
    database_path: Path,
    openai_base_url: str | None = None,
) -> ExperimentRunSummary:
    """
    Load, execute, and persist one configured experiment.
    """

    experiment = apply_openai_base_url(
        load_experiment_config(config_path),
        openai_base_url,
    )
    inputs = load_experiment_inputs(experiment)

    started_at = datetime.now(UTC)
    experiment_output = execute_experiment(
        experiment=inputs.experiment,
        samples=inputs.samples,
        documents=inputs.documents,
    )
    completed_at = datetime.now(UTC)

    results_store = DuckDBResultsStore(database_path=database_path)
    pipeline_summaries: list[PersistedPipelineRunSummary] = []

    for pipeline_run in experiment_output.pipeline_runs:
        run_id = build_run_id(
            experiment_name=inputs.experiment.experiment_name,
            pipeline_name=pipeline_run.pipeline.name,
        )

        results_store.write_run(
            run_id=run_id,
            experiment=inputs.experiment,
            pipeline=pipeline_run.pipeline,
            results=pipeline_run.results,
            metadata={
                "config_path": str(config_path),
                "database_path": str(database_path),
                "sample_count": len(inputs.samples),
                "document_count": len(inputs.documents),
                "chunk_count": len(pipeline_run.chunks),
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "runtime_metadata": pipeline_run.runtime_metadata,
            },
        )
        pipeline_summaries.append(
            PersistedPipelineRunSummary(
                pipeline_name=pipeline_run.pipeline.name,
                run_id=run_id,
                result_count=len(pipeline_run.results),
                chunk_count=len(pipeline_run.chunks),
                runtime_metadata=pipeline_run.runtime_metadata,
            )
        )

    return ExperimentRunSummary(
        experiment_name=inputs.experiment.experiment_name,
        config_path=config_path,
        database_path=database_path,
        sample_count=len(inputs.samples),
        document_count=len(inputs.documents),
        datasets=inputs.datasets,
        pipeline_runs=pipeline_summaries,
    )


def build_run_id(*, experiment_name: str, pipeline_name: str) -> str:
    """
    Create a persistence run identifier for one pipeline execution.
    """

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}__{pipeline_name}__{timestamp}"
