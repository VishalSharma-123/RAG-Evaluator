from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.persistence.base import ResultsStoreError
from rag_evaluator.schemas import EvalResult


def build_run_row(
        *,
        run_id: str,
        experiment: ExperimentConfig,
        pipeline: PipelineConfig,
        metadata: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    
    run_metadata = {
        "experiment_metadata": experiment.metadata,
        "pipeline_metadata": pipeline.metadata,
        **(metadata or {}),
    }
    
    return (
        run_id,
        experiment.experiment_name,
        pipeline.name,
        config_hash(pipeline),
        coerce_timestamp(run_metadata.get("started_at")),
        coerce_timestamp(run_metadata.get("completed_at")),
        json_dumps(run_metadata),
    )

def build_sample_row(
        *,
        run_id: str,
        result: EvalResult,
) -> tuple[Any, ...]:
    sample = result.sample
    return (
        run_id,
        sample.sample_id,
        sample.question,
        sample.question_type.value,
        sample.source_dataset,
        sample.source_split,
        sample.reference_answer,
        sample.is_answerable,
        json_dumps(sample.metadata),
    )

def build_retrieved_chunk_rows(
        *,
        run_id: str,
        result: EvalResult,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    
    for retrieved in result.retrieved_chunks:
        rows.append(
            (
                run_id,
                result.sample.sample_id,
                retrieved.chunk.chunk_id,
                retrieved.chunk.document_id,
                retrieved.rank,
                retrieved.score,
                retrieved.retriever_name,
                json_dumps(
                    {
                        "retrieved_metadata": retrieved.metadata,
                        "chunk_metadata": retrieved.chunk.metadata,
                        "start_char": retrieved.chunk.start_char,
                        "end_char": retrieved.chunk.end_char,
                    }
                ),
            )
        )
    
    return rows

def build_generated_answer_row(
        *,
        run_id: str,
        result: EvalResult,
) -> tuple[Any, ...] | None:
    if result.generated_answer is None:
        return None
    
    answer = result.generated_answer
    return (
        run_id,
        result.sample.sample_id,
        answer.answer,
        answer.model_name,
        answer.prompt_tokens,
        answer.completion_tokens,
        answer.latency_ms,
        answer.cost_usd,
        json_dumps(
            {
                **answer.metadata,
                "result_metadata": result.metadata,
            }
        ),
    )

def build_metric_row(
        *,
        run_id: str,
        result: EvalResult,
) -> tuple[Any, ...]:
    retrieval = result.retrieval_metrics
    generation = result.generation_metrics
    
    return (
        run_id,
        result.sample.sample_id,
        retrieval.precision_at_k,
        retrieval.recall_at_k,
        retrieval.mrr,
        retrieval.ndcg,
        generation.faithfulness if generation is not None else None,
        generation.relevance if generation is not None else None,
        generation.hallucination if generation is not None else None,
        generation.bert_score if generation is not None else None,
    )

def build_failure_rows(
        *,
        run_id: str,
        result: EvalResult,
) -> list[tuple[Any, ...]]:
    return [
        (
            run_id,
            result.sample.sample_id,
            failure_mode.value,
        )
        for failure_mode in result.failure_modes
    ]

def config_hash(pipeline: PipelineConfig) -> str:
    payload = json.dumps(
        pipeline.model_dump(mode = "json"),
        sort_keys = True,
        separators = (",", ":"),
        default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)

def coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    
    raise ResultsStoreError(
        f"Unsupported timestamp value for DuckDB persistence: {value!r}"
    )