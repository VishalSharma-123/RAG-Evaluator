from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_evaluator.dashboard.breakdown import (
    render_cost_latency_comparison,
    render_failure_breakdown,
    render_question_type_comparison,
    render_retrieval_generation_separation,
)
from rag_evaluator.dashboard.data import (
    DEFAULT_DATABASE_PATH,
    load_cost_latency_comparison,
    load_failure_breakdown,
    load_question_type_breakdown,
    load_run_comparison,
    load_run_summary,
    load_runs,
)
from rag_evaluator.dashboard.drilldown import render_sample_drilldown
from rag_evaluator.dashboard.overview import (
    render_run_comparison,
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
    selected_run_ids, primary_run_id = render_run_selector(runs_df)
    if primary_run_id is None:
        return
    
    comparison_df = load_run_comparison(
        selected_run_ids,
        database_path=database_path,
    )
    question_type_df = load_question_type_breakdown(
        selected_run_ids,
        database_path=database_path,
    )
    failure_df = load_failure_breakdown(
        selected_run_ids,
        database_path=database_path,
    )
    cost_latency_df = load_cost_latency_comparison(
        selected_run_ids,
        database_path=database_path,
    )
    
    summary_df = load_run_summary(
        primary_run_id,
        database_path=database_path,
    )

    tabs = st.tabs(
        [
            "Run Comparison",
            "Question Types",
            "Retrieval vs Generation",
            "Failures",
            "Cost & Latency",
            "Drill-down",
        ]
    )
    with tabs[0]:
        render_run_overview(runs_df, primary_run_id)
        render_summary_metrics(summary_df)
        render_run_comparison(comparison_df)
    with tabs[1]:
        render_question_type_comparison(question_type_df)
    with tabs[2]:
        render_retrieval_generation_separation(question_type_df)
    with tabs[3]:
        render_failure_breakdown(failure_df)
    with tabs[4]:
        render_cost_latency_comparison(cost_latency_df)
    with tabs[5]:
        render_sample_drilldown(
            summary_df,
            run_id=primary_run_id,
            database_path=database_path,
        )

if __name__ == "__main__":
    main()
    
