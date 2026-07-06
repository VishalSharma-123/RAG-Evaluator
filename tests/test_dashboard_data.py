from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.dashboard.data import (
    load_cost_latency_comparison,
    load_failure_breakdown,
    load_question_type_breakdown,
    load_retrieved_chunks,
    load_run_comparison,
    load_run_summary,
)
from rag_evaluator.persistence import DuckDBResultsStore
from rag_evaluator.schemas import (
    EvalResult,
    FailureMode,
    FinalContext,
    GeneratedAnswer,
    GenerationMetrics,
    QuestionType,
    RetrievalMetrics,
)


def test_dashboard_loaders_return_phase_7_views(
    tmp_path: Path,
    make_chunk,
    make_retrieved_chunk,
    make_sample,
) -> None:
    database_path = tmp_path / "results.duckdb"
    experiment = _experiment()
    pipeline_a = experiment.pipelines[0]
    pipeline_b = pipeline_a.model_copy(update={"name": "pipeline-b"})
    chunk = make_chunk()

    store = DuckDBResultsStore(database_path=database_path)
    store.write_run(
        run_id="run-a",
        experiment=experiment,
        pipeline=pipeline_a,
        results=[
            EvalResult(
                run_id="run-a",
                sample=make_sample(sample_id="sample-1", question_type=QuestionType.FACTOID),
                retrieved_chunks=[make_retrieved_chunk(chunk=chunk)],
                final_context=FinalContext(chunks=[chunk], rendered_text=chunk.text),
                generated_answer=GeneratedAnswer(
                    sample_id="sample-1",
                    answer="RAG uses retrieved context.",
                    model_name="test-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                    latency_ms=100,
                    cost_usd=0.01,
                ),
                retrieval_metrics=RetrievalMetrics(
                    precision_at_k=1.0,
                    recall_at_k=1.0,
                    mrr=1.0,
                    ndcg=1.0,
                ),
                generation_metrics=GenerationMetrics(
                    faithfulness=0.9,
                    relevance=0.8,
                    hallucination=0.1,
                    bert_score=0.7,
                ),
                failure_modes=[FailureMode.HALLUCINATION],
            )
        ],
        metadata={"completed_at": datetime.now(UTC).isoformat()},
    )
    store.write_run(
        run_id="run-b",
        experiment=experiment,
        pipeline=pipeline_b,
        results=[
            EvalResult(
                run_id="run-b",
                sample=make_sample(
                    sample_id="sample-2",
                    question_type=QuestionType.MULTI_HOP,
                    evidence_chunk_ids=[],
                ),
                retrieved_chunks=[],
                generated_answer=GeneratedAnswer(
                    sample_id="sample-2",
                    answer="Partial answer.",
                    model_name="test-model",
                    prompt_tokens=8,
                    completion_tokens=4,
                    latency_ms=200,
                    cost_usd=0.02,
                ),
                retrieval_metrics=RetrievalMetrics(
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    mrr=0.0,
                    ndcg=0.0,
                ),
                generation_metrics=GenerationMetrics(
                    faithfulness=0.4,
                    relevance=0.5,
                    hallucination=0.6,
                    bert_score=0.3,
                ),
                failure_modes=[FailureMode.RETRIEVAL_MISS],
            )
        ],
        metadata={"completed_at": datetime.now(UTC).isoformat()},
    )

    comparison_df = load_run_comparison(["run-a", "run-b"], database_path=database_path)
    summary_df = load_run_summary("run-a", database_path=database_path)
    question_type_df = load_question_type_breakdown(
        ["run-a", "run-b"],
        database_path=database_path,
    )
    failure_df = load_failure_breakdown(["run-a", "run-b"], database_path=database_path)
    cost_latency_df = load_cost_latency_comparison(
        ["run-a", "run-b"],
        database_path=database_path,
    )
    retrieved_df = load_retrieved_chunks(
        "run-a",
        "sample-1",
        database_path=database_path,
    )

    assert set(comparison_df["run_id"]) == {"run-a", "run-b"}
    assert comparison_df.set_index("run_id").loc["run-a", "sample_count"] == 1
    assert comparison_df.set_index("run_id").loc["run-b", "total_cost_usd"] == 0.02
    assert summary_df.loc[0, "failure_modes"] == ["HALLUCINATION"]
    assert set(question_type_df["question_type"]) == {"factoid", "multi_hop"}
    assert set(failure_df["failure_mode"]) == {"HALLUCINATION", "RETRIEVAL_MISS"}
    assert cost_latency_df.set_index("run_id").loc["run-a", "prompt_tokens"] == 10
    assert retrieved_df.loc[0, "chunk_id"] == chunk.chunk_id


def test_dashboard_loaders_return_empty_frames_for_empty_run_selection(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "results.duckdb"
    DuckDBResultsStore(database_path=database_path).initialize()

    assert load_run_comparison([], database_path=database_path).empty
    assert load_question_type_breakdown([], database_path=database_path).empty
    assert load_failure_breakdown([], database_path=database_path).empty
    assert load_cost_latency_comparison([], database_path=database_path).empty


def _experiment() -> ExperimentConfig:
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-a",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 1},
        }
    )
    return ExperimentConfig.model_validate(
        {
            "experiment_name": "dashboard-unit",
            "datasets": [],
            "pipelines": [pipeline.model_dump(mode="json")],
        }
    )
