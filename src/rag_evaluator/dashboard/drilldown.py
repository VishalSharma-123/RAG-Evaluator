from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from rag_evaluator.dashboard.data import load_retrieved_chunks


def render_sample_drilldown(
        summary_df: pd.DataFrame,
        *,
        run_id: str,
        database_path: str | Path,
) -> None:
    st.subheader("Per Sample Drilldown")
    
    if summary_df.empty:
        st.info("No summary data found.")
        return
    
    sample_id = st.selectbox("Sample ID", summary_df["sample_id"].tolist())
    sample_row = summary_df.loc[summary_df["sample_id"] == sample_id].iloc[0]
    
    st.write(f"Question: {sample_row['question']}")
    st.write(f"Question Type: `{sample_row['question_type']}`")
    st.write(f"Dataset: `{sample_row['source_dataset']}`")
    st.write(f"Answerable: `{sample_row['is_answerable']}`")
    st.write(f"Reference Answer: {sample_row.get('reference_answer') or '[None]'}")
    st.write(f"Generated Answer: {sample_row.get('answer') or '[No answer]'}")
    
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
    available_metric_columns = [
        column for column in metric_columns if column in summary_df.columns
    ]
    if available_metric_columns:
        st.write("Metrics")
        st.dataframe(
            pd.DataFrame([sample_row[available_metric_columns].to_dict()]),
            use_container_width=True,
        )
    
    sample_metadata = _parse_json_field(sample_row.get("sample_metadata_json"))
    answer_metadata = _parse_json_field(sample_row.get("answer_metadata_json"))
    result_metadata = {}
    
    if isinstance(answer_metadata, dict):
        result_metadata = _as_dict(answer_metadata.get("result_metadata"))

    if sample_metadata:
        st.write("Sample Metadata")
        st.json(sample_metadata)

    if result_metadata:
        st.write("Result Metadata")
        st.json(result_metadata)

    question_type_signals = _as_dict(result_metadata.get("question_type_signals"))
    if question_type_signals:
        st.write("Question Type Signals")
        st.json(question_type_signals)

    answer_metadata_without_result = (
        dict(answer_metadata) if isinstance(answer_metadata, dict) else {}
    )
    answer_metadata_without_result.pop("result_metadata", None)
    if answer_metadata_without_result:
        st.write("Answer Metadata")
        st.json(answer_metadata_without_result)

    retrieved_chunks_df = load_retrieved_chunks(
        run_id,
        sample_id,
        database_path=database_path,
    )
    if not retrieved_chunks_df.empty:
        rendered_chunks_df = retrieved_chunks_df.copy()
        if "metadata_json" in rendered_chunks_df.columns:
            rendered_chunks_df["metadata_json"] = rendered_chunks_df["metadata_json"].apply(
                _parse_json_field
            )

        st.write("Retrieved Chunks")
        st.dataframe(rendered_chunks_df, use_container_width=True)

def _parse_json_field(value: Any) -> dict[str, Any] | list[Any] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}

    return {"raw": value}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
