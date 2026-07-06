from __future__ import annotations

from pathlib import Path

from rag_evaluator.application.experiment_inputs import load_experiment_inputs
from rag_evaluator.application.types import IndexBuildPipelineSummary, IndexBuildSummary
from rag_evaluator.config import ExperimentConfig, RetrieverType
from rag_evaluator.execution.runtime import build_pipeline_runtime
from rag_evaluator.io import load_experiment_config


def build_index(
    *,
    config_path: Path,
    pipeline_name: str | None = None,
) -> IndexBuildSummary:
    """
    Build retrieval indexes for configured pipelines without running generation.
    """

    experiment = load_experiment_config(config_path)
    inputs = load_experiment_inputs(experiment)
    selected_pipelines = _select_pipelines(
        experiment=experiment,
        pipeline_name=pipeline_name,
    )

    pipeline_summaries: list[IndexBuildPipelineSummary] = []
    for pipeline in selected_pipelines:
        runtime = build_pipeline_runtime(
            pipeline=pipeline,
            documents=inputs.documents,
        )
        pipeline_summaries.append(
            IndexBuildPipelineSummary(
                pipeline_name=pipeline.name,
                chunk_count=len(runtime.chunks),
                store_provider=pipeline.store.provider.value,
                collection_name=pipeline.store.collection_name,
                persist_directory=(
                    Path(pipeline.store.persist_directory)
                    if pipeline.retriever.type != RetrieverType.BM25
                    else None
                ),
                retriever_type=pipeline.retriever.type.value,
            )
        )

    return IndexBuildSummary(
        experiment_name=experiment.experiment_name,
        config_path=config_path,
        document_count=len(inputs.documents),
        sample_count=len(inputs.samples),
        pipeline_indexes=pipeline_summaries,
    )


def _select_pipelines(
    *,
    experiment: ExperimentConfig,
    pipeline_name: str | None,
):
    if pipeline_name is None:
        return experiment.pipelines

    matching = [pipeline for pipeline in experiment.pipelines if pipeline.name == pipeline_name]
    if not matching:
        raise ValueError(f"Pipeline not found in config: {pipeline_name}")

    return matching
