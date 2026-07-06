from __future__ import annotations

from pathlib import Path

from rag_evaluator.commands.scoring_commands import score_run
from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.persistence import DuckDBResultsStore
from rag_evaluator.schemas import EvalResult, FinalContext, GeneratedAnswer, RetrievalMetrics


def test_score_run_recomputes_persisted_metrics_and_failures(
    tmp_path: Path,
    make_chunk,
    make_retrieved_chunk,
    make_sample,
) -> None:
    database_path = tmp_path / "results.duckdb"
    sample = make_sample(reference_answer="Retrieval augmented generation")
    chunk = make_chunk(text="Retrieval augmented generation uses retrieved context.")
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "bm25", "top_k": 1},
        }
    )
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [],
            "pipelines": [pipeline.model_dump(mode="json")],
        }
    )
    result = EvalResult(
        run_id="run-1",
        sample=sample,
        retrieved_chunks=[make_retrieved_chunk(chunk=chunk)],
        final_context=FinalContext(chunks=[chunk], rendered_text=chunk.text),
        generated_answer=GeneratedAnswer(
            sample_id=sample.sample_id,
            answer="Retrieval augmented generation",
            model_name="unit-model",
        ),
        retrieval_metrics=RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg=0.0,
        ),
    )
    store = DuckDBResultsStore(database_path=database_path)
    store.write_run(
        run_id="run-1",
        experiment=experiment,
        pipeline=pipeline,
        results=[result],
        metadata={"run_status": "completed"},
    )

    summary = score_run(database_path=database_path, run_id="run-1")
    rows = store.fetch_run("run-1")

    assert summary.sample_count == 1
    assert summary.metric_count == 1
    assert rows[0]["precision_at_k"] == 1.0
    assert rows[0]["recall_at_k"] == 1.0
