from __future__ import annotations

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
from rag_evaluator.schemas import EvalResult


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
