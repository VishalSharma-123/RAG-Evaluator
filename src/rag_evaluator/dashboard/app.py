from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_evaluator.dashboard.breakdown import (
    render_failure_breakdown,
    render_question_breakdown,
)
from rag_evaluator.dashboard.data import (
    DEFAULT_DATABASE_PATH,
    load_failure_breakdown,
    load_run_summary,
    load_runs,
)
from rag_evaluator.dashboard.drilldown import render_sample_drilldown
from rag_evaluator.dashboard.overview import (
    render_run_overview,
    render_run_selector,
    render_summary_metrics,
)


def main() -> None:
    st.set_page_config(page_title="RAG Evaluator Dashboard", layout="wide")
    st.title("RAG Evaluator Dashboard")
    
    database_path_input = st.sidebar.text_input(
        "DuckDB Path",
        value=str(os.environ.get("RAG_EVALUATOR_DATABASE_PATH", DEFAULT_DATABASE_PATH)),
    )
    database_path = Path(database_path_input)
    
    if not database_path.exists():
        st.error(f"DuckDB file not found: {database_path}")
        return
    
    runs_df = load_runs(database_path)
    selected_run_id = render_run_selector(runs_df)
    if selected_run_id is None:
        return
    
    render_run_overview(runs_df, selected_run_id)
    
    summary_df = load_run_summary(
        selected_run_id,
        database_path=database_path,
    )
    failure_df = load_failure_breakdown(
        selected_run_id,
        database_path=database_path,
    )
    
    render_summary_metrics(summary_df)
    render_question_breakdown(summary_df)
    render_failure_breakdown(failure_df)
    render_sample_drilldown(
        summary_df,
        run_id=selected_run_id,
        database_path=database_path,
    )

if __name__ == "__main__":
    main()
    
