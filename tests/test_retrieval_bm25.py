from __future__ import annotations

from rag_evaluator.retrieval.bm25 import BM25Retriever


def test_bm25_retriever_prefers_lexical_match(make_chunk) -> None:
    retriever = BM25Retriever(
        chunks=[
            make_chunk(chunk_id="doc:chunk:0", text="retrieval augmented generation"),
            make_chunk(chunk_id="doc:chunk:1", text="dashboard metrics and charts"),
        ]
    )

    results = retriever.retrieve("dashboard")

    assert results[0].chunk.chunk_id == "doc:chunk:1"


def test_bm25_retriever_returns_empty_for_empty_chunk_list() -> None:
    retriever = BM25Retriever(chunks=[])

    assert retriever.retrieve("query") == []
