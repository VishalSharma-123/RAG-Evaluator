from __future__ import annotations

from rag_evaluator.application.types import ExperimentRunSummary, SyntheticGenerationSummary
from rag_evaluator.schemas import Chunk


def render_synthetic_summary(summary: SyntheticGenerationSummary) -> None:
    print(
        f"Generated {summary.sample_count} synthetic samples from {summary.chunk_count} chunks"
    )
    print(f"Wrote synthetic dataset to {summary.output_path}")


def render_experiment_summary(summary: ExperimentRunSummary) -> None:
    for dataset in summary.datasets:
        print(f"Loaded {dataset.sample_count} samples from dataset `{dataset.dataset_name}`")

    print(
        f"Prepared {summary.sample_count} samples and {summary.document_count} source documents "
        f"for experiment `{summary.experiment_name}`"
    )

    for pipeline_run in summary.pipeline_runs:
        print(
            f"Completed pipeline `{pipeline_run.pipeline_name}`: "
            f"{pipeline_run.result_count} results persisted to {summary.database_path} "
            f"as run `{pipeline_run.run_id}`"
        )


def render_chunk_output(chunks: list[Chunk], chunker_type: str) -> None:
    print(f"Chunked document into {len(chunks)} chunks using {chunker_type}")
    for chunk in chunks:
        print(f"--- {chunk.chunk_id} [{chunk.start_char}, {chunk.end_char}] ---")
        print(chunk.text)
        print()
