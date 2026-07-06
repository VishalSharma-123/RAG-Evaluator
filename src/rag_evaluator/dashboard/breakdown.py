from __future__ import annotations

import pandas as pd
import streamlit as st


def render_question_breakdown(summary_df: pd.DataFrame) -> None:
    st.subheader("Metric Breakdown by Question Type")
    
    if summary_df.empty:
        st.info("No sample records found for this run.")
        return

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
    available_metric_columns = [column for column in metric_columns if column in summary_df.columns]

    if not available_metric_columns:
        st.info("No metric columns available for question-type breakdown.")
        return
    
    grouped = (
        summary_df.groupby("question_type", dropna=False)[available_metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )

    st.dataframe(grouped, use_container_width=True)


def render_question_type_comparison(breakdown_df: pd.DataFrame) -> None:
    st.subheader("Metric Breakdown by Question Type")

    if breakdown_df.empty:
        st.info("No question-type metric records found for the selected runs.")
        return

    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_retrieval_generation_separation(breakdown_df: pd.DataFrame) -> None:
    st.subheader("Retrieval vs Generation")

    if breakdown_df.empty:
        st.info("No metric records found for the selected runs.")
        return

    retrieval_columns = [
        "run_id",
        "question_type",
        "sample_count",
        "avg_precision_at_k",
        "avg_recall_at_k",
        "avg_mrr",
        "avg_ndcg",
    ]
    generation_columns = [
        "run_id",
        "question_type",
        "sample_count",
        "avg_faithfulness",
        "avg_relevance",
        "avg_hallucination",
        "avg_bert_score",
    ]

    left, right = st.columns(2)
    with left:
        st.write("Retrieval Metrics")
        st.dataframe(
            breakdown_df[_available_columns(breakdown_df, retrieval_columns)],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.write("Generation Metrics")
        st.dataframe(
            breakdown_df[_available_columns(breakdown_df, generation_columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_failure_breakdown(failure_df: pd.DataFrame) -> None:
    st.subheader("Failure Breakdown")
    
    if failure_df.empty:
        st.info("No sample failure records found for this run.")
        return
    
    st.dataframe(failure_df, use_container_width=True, hide_index=True)
    chart_df = failure_df.pivot_table(
        index="failure_mode",
        columns="run_id",
        values="failure_count",
        aggfunc="sum",
        fill_value=0,
    )
    st.bar_chart(chart_df)


def render_cost_latency_comparison(cost_latency_df: pd.DataFrame) -> None:
    st.subheader("Cost & Latency")

    if cost_latency_df.empty:
        st.info("No cost or latency records found for the selected runs.")
        return

    st.dataframe(cost_latency_df, use_container_width=True, hide_index=True)

    chart_columns = [
        column
        for column in ["total_cost_usd", "avg_latency_ms"]
        if column in cost_latency_df.columns
    ]
    if chart_columns:
        st.bar_chart(cost_latency_df.set_index("run_id")[chart_columns])


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]
