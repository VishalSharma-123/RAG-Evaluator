from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from rag_evaluator.dashboard.queries import (
    LOAD_FAILURE_BREAKDOWN,
    LOAD_RETRIEVED_CHUNKS,
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
        run_id: str,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(LOAD_FAILURE_BREAKDOWN, [run_id]).df()

def load_retrieved_chunks(
        run_id: str,
        sample_id: str,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only = True) as connection:
        return connection.execute(LOAD_RETRIEVED_CHUNKS, [run_id, sample_id]).df()