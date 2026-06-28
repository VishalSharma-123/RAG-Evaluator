from __future__ import annotations

import pandas as pd
import streamlit as st


def render_question_breakdown(summary_df: pd.DataFrame) -> None:
    st.subheader("Metric Breakdown by Question type")
    
    if summary_df.empty:
        st.info("No sample records found for this run")
        return
    
    grouped = (
        summary_df.groupby("question_type", dropna=False)[
            [
                "precision_at_k",
                "recall_at_k",
                "mrr",
                "ndcg",
                "faithfulness",
                "relevance",
                "hallucination",
                "bert_score",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    
    st.dataframe(grouped, use_container_width=True)

def render_failure_breakdown(failure_df: pd.DataFrame) -> None:
    st.subheader("Metric Breakdown by Failure type")
    
    if failure_df.empty:
        st.info("No sample failure records found for this run")
        return
    
    st.dataframe(failure_df, use_container_width=True)
    st.bar_chart(failure_df.set_index("failure_mode"))
