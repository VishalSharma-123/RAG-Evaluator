from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from rag_evaluator.dashboard.queries import (
    LOAD_COST_LATENCY_COMPARISON,
    LOAD_FAILURE_BREAKDOWN,
    LOAD_QUESTION_TYPE_BREAKDOWN,
    LOAD_RETRIEVED_CHUNKS,
    LOAD_RUN_COMPARISON,
    LOAD_RUN_SUMMARY,
    LOAD_RUNS,
)

DEFAULT_DATABASE_PATH = Path("storage/results.duckdb")

def load_runs(database_path: str | Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(LOAD_RUNS).df()

def load_run_summary(
        run_id: str,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(LOAD_RUN_SUMMARY, [run_id]).df()

def load_failure_breakdown(
        run_ids: str | list[str],
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    if not normalized_run_ids:
        return pd.DataFrame()

    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(
            _format_run_id_query(LOAD_FAILURE_BREAKDOWN, normalized_run_ids),
            normalized_run_ids,
        ).df()

def load_run_comparison(
        run_ids: list[str],
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    if not normalized_run_ids:
        return pd.DataFrame()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            _format_run_id_query(LOAD_RUN_COMPARISON, normalized_run_ids),
            normalized_run_ids,
        ).df()

def load_question_type_breakdown(
        run_ids: list[str],
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    if not normalized_run_ids:
        return pd.DataFrame()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            _format_run_id_query(LOAD_QUESTION_TYPE_BREAKDOWN, normalized_run_ids),
            normalized_run_ids,
        ).df()

def load_cost_latency_comparison(
        run_ids: list[str],
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    if not normalized_run_ids:
        return pd.DataFrame()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            _format_run_id_query(LOAD_COST_LATENCY_COMPARISON, normalized_run_ids),
            normalized_run_ids,
        ).df()

def load_retrieved_chunks(
        run_id: str,
        sample_id: str,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(LOAD_RETRIEVED_CHUNKS, [run_id, sample_id]).df()


def _normalize_run_ids(run_ids: str | list[str]) -> list[str]:
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    return [run_id for run_id in dict.fromkeys(run_ids) if run_id]


def _format_run_id_query(query: str, run_ids: list[str]) -> str:
    placeholders = ", ".join("?" for _ in run_ids)
    return query.format(run_id_placeholders=placeholders)
