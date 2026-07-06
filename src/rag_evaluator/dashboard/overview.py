from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def render_run_selector(runs_df: pd.DataFrame) -> tuple[list[str], str | None]:
    if runs_df.empty:
        st.warning("No runs found")
        return [], None
    
    run_options = runs_df["run_id"].tolist()
    selected_run_ids = st.sidebar.multiselect(
        "Compare Runs",
        run_options,
        default=run_options[: min(3, len(run_options))],
    )
    if not selected_run_ids:
        st.info("Select at least one run to compare.")
        return [], None

    primary_run_id = st.sidebar.selectbox("Primary Run", selected_run_ids)
    return selected_run_ids, primary_run_id

def render_run_overview(runs_df: pd.DataFrame, selected_run_id: str) -> None:
    selected_row = runs_df.loc[runs_df["run_id"] == selected_run_id].iloc[0]
    
    st.subheader("Run Overview")
    st.write(f"Experiment: {selected_row['experiment_name']}")
    st.write(f"Pipeline: {selected_row['pipeline_name']}")
    st.write(f"Config Hash: {selected_row['config_hash']}")
    if "run_status" in selected_row and pd.notna(selected_row["run_status"]):
        st.write(f"Status: `{selected_row['run_status']}`")
    
    metadata_json = selected_row.get("metadata_json")
    if metadata_json:
        with st.expander("Run Metadata"):
            try:
                st.json(json.loads(metadata_json))
            except json.JSONDecodeError:
                st.text(metadata_json)

def render_summary_metrics(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        st.info("No sample records found for this run.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(summary_df))
    col2.metric("Avg Recall@k", f"{summary_df['recall_at_k'].fillna(0).mean():.3f}")
    col3.metric("Avg Faithfulness", f"{summary_df['faithfulness'].fillna(0).mean():.3f}")
    col4.metric("Avg Latency (ms)", f"{summary_df['latency_ms'].fillna(0).mean():.1f}")


def render_run_comparison(comparison_df: pd.DataFrame) -> None:
    st.subheader("Run Comparison")

    if comparison_df.empty:
        st.info("No comparison records found for the selected runs.")
        return

    display_columns = [
        "run_id",
        "experiment_name",
        "pipeline_name",
        "run_status",
        "sample_count",
        "avg_recall_at_k",
        "avg_mrr",
        "avg_ndcg",
        "avg_faithfulness",
        "avg_relevance",
        "avg_hallucination",
        "total_cost_usd",
        "avg_latency_ms",
        "prompt_tokens",
        "completion_tokens",
    ]
    available_columns = [
        column for column in display_columns if column in comparison_df.columns
    ]
    st.dataframe(
        comparison_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )
