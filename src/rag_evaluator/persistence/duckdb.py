from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.persistence.base import ResultsStore, ResultsStoreError
from rag_evaluator.persistence.queries import (
    DELETE_FAILURE_LABELS_BY_RUN_ID,
    DELETE_GENERATED_ANSWERS_BY_RUN_ID,
    DELETE_METRIC_SCORES_BY_RUN_ID,
    DELETE_RETRIEVED_CHUNKS_BY_RUN_ID,
    DELETE_RUNS_BY_RUN_ID,
    DELETE_SAMPLES_BY_RUN_ID,
    FETCH_RETRIEVED_CHUNKS_FOR_RUN,
    FETCH_RUN,
    INSERT_FAILURE_LABEL,
    INSERT_GENERATED_ANSWER,
    INSERT_METRIC_SCORE,
    INSERT_RETRIEVED_CHUNK,
    INSERT_RUN,
    INSERT_SAMPLE,
)
from rag_evaluator.persistence.serializer import (
    build_failure_rows,
    build_generated_answer_row,
    build_metric_row,
    build_retrieved_chunk_rows,
    build_run_row,
    build_sample_row,
)
from rag_evaluator.schemas import Chunk, EvalResult, EvalSample, GeneratedAnswer, RetrievedChunk
from rag_evaluator.scoring.failures import classify_failures
from rag_evaluator.scoring.generation import score_generation
from rag_evaluator.scoring.retrieval import score_retrieval


@dataclass(frozen=True)
class DuckDBResultsStore(ResultsStore):
    """
    DuckDB-backed analytical store for experiment outputs.
    """

    database_path: str | Path = "storage/results.duckdb"
    schema_path: str | Path | None = None
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", Path(self.database_path))
        
        resolved_schema_path = (
            Path(self.schema_path)
            if self.schema_path is not None
            else Path(__file__).with_name("schema.sql")
        )
        object.__setattr__(self, "schema_path", resolved_schema_path)
    
    
    def initialize(self) -> None:
        """
        Initialize the DuckDB database and schema.
        :return:
        """
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            schema_sql = self.schema_path.read_text(encoding="utf-8")
            
            with duckdb.connect(str(self.database_path)) as connection:
                connection.execute(schema_sql)
        
        except OSError as exc:
            raise ResultsStoreError(
                f"Failed to read or prepare DuckDB files: {exc}"
            ) from exc
        
        except duckdb.Error as exc:
            raise ResultsStoreError(
                f"Failed to initialize DuckDB schema: {exc}"
            ) from exc
    
    def write_run(
        self,
        *,
        run_id: str,
        experiment: ExperimentConfig,
        pipeline: PipelineConfig,
        results: list[EvalResult],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist one evaluated run and its pre-sample records.
        :param run_id:
        :param experiment:
        :param pipeline:
        :param results:
        :param metadata:
        :return:
        """
        self.initialize()

        connection: duckdb.DuckDBPyConnection | None = None
        try:
            with duckdb.connect(str(self.database_path)) as connection:
                connection.execute("BEGIN TRANSACTION")
                
                self._delete_existing_run(connection, run_id)
                
                connection.execute(
                    INSERT_RUN,
                    build_run_row(
                        run_id=run_id,
                        experiment=experiment,
                        pipeline=pipeline,
                        metadata=metadata,
                    ),
                )
                
                for result in results:
                    connection.execute(
                        INSERT_SAMPLE,
                        build_sample_row(run_id=run_id, result=result),
                    )
                    
                    retrieved_chunk_rows = build_retrieved_chunk_rows(
                        run_id=run_id,
                        result=result,
                    )
                    
                    if retrieved_chunk_rows:
                        connection.executemany(
                            INSERT_RETRIEVED_CHUNK,
                            retrieved_chunk_rows,
                        )
                    
                    generated_answer_rows = build_generated_answer_row(
                        run_id=run_id,
                        result=result,
                    )
                    if generated_answer_rows:
                        connection.execute(
                            INSERT_GENERATED_ANSWER,
                            generated_answer_rows
                        )
                    
                    connection.execute(
                        INSERT_METRIC_SCORE,
                        build_metric_row(run_id=run_id, result=result),
                    )
                    
                    failure_rows = build_failure_rows(run_id=run_id, result=result)
                    if failure_rows:
                        connection.executemany(
                            INSERT_FAILURE_LABEL,
                            failure_rows,
                        )

                connection.execute("COMMIT")
                
        except duckdb.Error as exc:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            raise ResultsStoreError(
                f"Failed to persist run `{run_id}`: {exc}"
            ) from exc
    
    def fetch_run(self, run_id: str) -> list[dict[str, Any]]:
        """
        Return persisted sample-level records for one run.
        :param run_id:
        :return:
        """
        try:
            with duckdb.connect(str(self.database_path)) as connection:
                cursor = connection.execute(FETCH_RUN, [run_id])
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description]
        except duckdb.Error as exc:
            raise ResultsStoreError(
                f"Failed to fetch run `{run_id}`: {exc}"
            ) from exc
        
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def rescore_run(
        self,
        *,
        run_id: str,
        retrieval_k: int | None = None,
    ) -> dict[str, int]:
        """
        Recompute deterministic metrics and failure labels for one persisted run.
        """

        if retrieval_k is not None and retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1.")

        self.initialize()
        sample_rows = self.fetch_run(run_id)
        if not sample_rows:
            raise ResultsStoreError(f"Run not found or has no samples: {run_id}")

        retrieved_by_sample_id = self._fetch_retrieved_chunks_by_sample_id(run_id)
        metric_rows: list[tuple[Any, ...]] = []
        failure_rows: list[tuple[Any, ...]] = []

        for row in sample_rows:
            sample = _row_to_sample(row)
            retrieved_chunks = retrieved_by_sample_id.get(sample.sample_id, [])
            generated_answer = _row_to_generated_answer(row)
            k = retrieval_k or max(1, len(retrieved_chunks))
            retrieval_metrics = score_retrieval(sample, retrieved_chunks, k=k)
            generation_metrics = score_generation(sample, generated_answer)
            failures = classify_failures(
                sample,
                retrieved_chunks,
                generated_answer=generated_answer,
                context_was_used=_context_was_used(row),
                hallucination_score=(
                    generation_metrics.hallucination
                    if generation_metrics is not None
                    else None
                ),
                partial_answer_score=(
                    1.0 - generation_metrics.faithfulness
                    if generation_metrics is not None
                    and generation_metrics.faithfulness is not None
                    else None
                ),
                retrieval_k=k,
            )

            metric_rows.append(
                (
                    run_id,
                    sample.sample_id,
                    retrieval_metrics.precision_at_k,
                    retrieval_metrics.recall_at_k,
                    retrieval_metrics.mrr,
                    retrieval_metrics.ndcg,
                    generation_metrics.faithfulness if generation_metrics else None,
                    generation_metrics.relevance if generation_metrics else None,
                    generation_metrics.hallucination if generation_metrics else None,
                    generation_metrics.bert_score if generation_metrics else None,
                )
            )
            failure_rows.extend(
                (run_id, sample.sample_id, failure.value)
                for failure in failures
            )

        connection: duckdb.DuckDBPyConnection | None = None
        try:
            with duckdb.connect(str(self.database_path)) as connection:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(DELETE_METRIC_SCORES_BY_RUN_ID, [run_id])
                connection.execute(DELETE_FAILURE_LABELS_BY_RUN_ID, [run_id])
                connection.executemany(INSERT_METRIC_SCORE, metric_rows)
                if failure_rows:
                    connection.executemany(INSERT_FAILURE_LABEL, failure_rows)
                connection.execute("COMMIT")
        except duckdb.Error as exc:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            raise ResultsStoreError(
                f"Failed to rescore run `{run_id}`: {exc}"
            ) from exc

        return {
            "sample_count": len(sample_rows),
            "metric_count": len(metric_rows),
            "failure_label_count": len(failure_rows),
        }

    def _fetch_retrieved_chunks_by_sample_id(
        self,
        run_id: str,
    ) -> dict[str, list[RetrievedChunk]]:
        try:
            with duckdb.connect(str(self.database_path)) as connection:
                cursor = connection.execute(FETCH_RETRIEVED_CHUNKS_FOR_RUN, [run_id])
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description]
        except duckdb.Error as exc:
            raise ResultsStoreError(
                f"Failed to fetch retrieved chunks for `{run_id}`: {exc}"
            ) from exc

        by_sample_id: dict[str, list[RetrievedChunk]] = {}
        for row in rows:
            payload = dict(zip(columns, row, strict=True))
            retrieved = _row_to_retrieved_chunk(payload)
            by_sample_id.setdefault(str(payload["sample_id"]), []).append(retrieved)

        return by_sample_id
    
    def _delete_existing_run(
            self,
            connection: duckdb.DuckDBPyConnection,
            run_id: str,
    ) -> None:
        connection.execute(DELETE_FAILURE_LABELS_BY_RUN_ID, [run_id])
        connection.execute(DELETE_RETRIEVED_CHUNKS_BY_RUN_ID, [run_id])
        connection.execute(DELETE_METRIC_SCORES_BY_RUN_ID, [run_id])
        connection.execute(DELETE_GENERATED_ANSWERS_BY_RUN_ID, [run_id])
        connection.execute(DELETE_SAMPLES_BY_RUN_ID, [run_id])
        connection.execute(DELETE_RUNS_BY_RUN_ID, [run_id])


def _row_to_sample(row: dict[str, Any]) -> EvalSample:
    metadata = _json_object(row.get("sample_metadata_json"))
    return EvalSample(
        sample_id=str(row["sample_id"]),
        question=str(row["question"]),
        reference_answer=row.get("reference_answer"),
        answer_aliases=list(metadata.pop("answer_aliases", [])),
        choices=list(metadata.pop("choices", [])),
        question_type=str(row["question_type"]),
        source_dataset=str(row["source_dataset"]),
        source_split=row.get("source_split"),
        source_id=metadata.pop("source_id", None),
        evidence_chunk_ids=list(metadata.pop("evidence_chunk_ids", [])),
        evidence_spans=list(metadata.pop("evidence_spans", [])),
        is_answerable=bool(row["is_answerable"]),
        metadata=metadata,
    )


def _row_to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    metadata = _json_object(row.get("metadata_json"))
    chunk_metadata = metadata.get("chunk_metadata")
    retrieved_metadata = metadata.get("retrieved_metadata")
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            text=str(metadata.get("text", "")),
            start_char=metadata.get("start_char"),
            end_char=metadata.get("end_char"),
            metadata=chunk_metadata if isinstance(chunk_metadata, dict) else {},
        ),
        rank=int(row["rank"]),
        score=float(row["score"]),
        retriever_name=str(row["retriever_name"]),
        metadata=retrieved_metadata if isinstance(retrieved_metadata, dict) else {},
    )


def _row_to_generated_answer(row: dict[str, Any]) -> GeneratedAnswer | None:
    if row.get("answer") is None:
        return None

    metadata = _json_object(row.get("answer_metadata_json"))
    return GeneratedAnswer(
        sample_id=str(row["sample_id"]),
        answer=str(row["answer"]),
        model_name=str(row["model_name"]),
        prompt_tokens=row.get("prompt_tokens"),
        completion_tokens=row.get("completion_tokens"),
        latency_ms=row.get("latency_ms"),
        cost_usd=float(row.get("cost_usd") or 0.0),
        metadata=metadata,
    )


def _context_was_used(row: dict[str, Any]) -> bool | None:
    final_context = _json_object(row.get("final_context_json"))
    if not final_context:
        return None

    chunks = final_context.get("chunks")
    if isinstance(chunks, list):
        return bool(chunks)

    rendered_text = final_context.get("rendered_text")
    if isinstance(rendered_text, str):
        return bool(rendered_text.strip())

    return None


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}
