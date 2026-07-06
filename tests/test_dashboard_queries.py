from __future__ import annotations

from pathlib import Path

import duckdb

from rag_evaluator.dashboard.data import _format_run_id_query
from rag_evaluator.dashboard.queries import (
    LOAD_COST_LATENCY_COMPARISON,
    LOAD_FAILURE_BREAKDOWN,
    LOAD_QUESTION_TYPE_BREAKDOWN,
    LOAD_RETRIEVED_CHUNKS,
    LOAD_RUN_COMPARISON,
    LOAD_RUN_SUMMARY,
    LOAD_RUNS,
)
from rag_evaluator.persistence import DuckDBResultsStore


def test_dashboard_queries_execute_against_empty_initialized_database(tmp_path: Path) -> None:
    database_path = tmp_path / "results.duckdb"
    DuckDBResultsStore(database_path=database_path).initialize()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute(LOAD_RUNS).fetchall() == []
        assert connection.execute(LOAD_RUN_SUMMARY, ["missing-run"]).fetchall() == []
        assert connection.execute(LOAD_RETRIEVED_CHUNKS, ["missing-run", "sample"]).fetchall() == []

        run_id_queries = [
            LOAD_RUN_COMPARISON,
            LOAD_QUESTION_TYPE_BREAKDOWN,
            LOAD_FAILURE_BREAKDOWN,
            LOAD_COST_LATENCY_COMPARISON,
        ]
        for query in run_id_queries:
            formatted_query = _format_run_id_query(query, ["missing-run"])
            assert connection.execute(formatted_query, ["missing-run"]).fetchall() == []
