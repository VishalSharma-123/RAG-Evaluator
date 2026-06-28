from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def render_run_selector(runs_df: pd.DataFrame) -> str | None:
    if runs_df.empty:
        st.warning("No runs found")
        return None
    
    run_options = runs_df["run_id"].tolist()
    return st.sidebar.selectbox("Run ID", run_options)

def render_run_overview(runs_df: pd.DataFrame, selected_run_id: str) -> None:
    selected_row = runs_df.loc[runs_df["run_id"] == selected_run_id].iloc[0]
    
    st.subheader("Run Overview")
    st.write(f"Experiment: {selected_row['experiment_name']}")
    st.write(f"Pipeline: {selected_row['pipeline_name']}")
    st.write(f"Config Hash: {selected_row['config_hash']}")
    
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
