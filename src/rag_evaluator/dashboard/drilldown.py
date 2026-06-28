from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_evaluator.dashboard.data import load_retrieved_chunks


def render_sample_drilldown(
        summary_df: pd.DataFrame,
        *,
        run_id: str,
        database_path: str | Path,
) -> None:
    st.subheader("Per Sample drilldown")
    
    if summary_df.empty:
        st.info("No summary data found")
        return
    
    sample_id = st.selectbox("Sample ID", summary_df["sample_id"].tolist())
    sample_row = summary_df.loc[summary_df["sample_id"] == sample_id].iloc[0]

    st.write(f"Question: {sample_row['question']}")
    st.write(f"Question Type: `{sample_row['question_type']}`")
    st.write(f"Dataset: `{sample_row['source_dataset']}`")
    st.write(f"Reference Answer: {sample_row['reference_answer']}")
    st.write(f"Generated Answer: {sample_row['answer']}")
    
    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg",
        "faithfulness",
        "relevance",
        "hallucination",
        "bert_score",
    ]
    st.dataframe(
        pd.DataFrame([sample_row[metric_columns].to_dict()]),
        use_container_width=True,
    )
    
    retrieved_chunks_df = load_retrieved_chunks(
        run_id,
        sample_id,
        database_path=database_path,
    )
    if not retrieved_chunks_df.empty:
        st.write("Retrieved Chunks")
        st.dataframe(retrieved_chunks_df, use_container_width=True)
