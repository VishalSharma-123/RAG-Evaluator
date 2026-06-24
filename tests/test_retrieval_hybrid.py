from __future__ import annotations

from rag_evaluator.ingestion.stores import InMemoryVectorStore
from rag_evaluator.retrieval.bm25 import BM25Retriever
from rag_evaluator.retrieval.hybrid import HybridRetriever
from rag_evaluator.retrieval.vector import VectorRetriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in query else [0.0, 1.0]


def test_hybrid_retriever_combines_vector_and_bm25(make_chunk) -> None:
    chunks = [
        make_chunk(chunk_id="doc:chunk:0", text="alpha retrieval"),
        make_chunk(chunk_id="doc:chunk:1", text="beta charts"),
    ]
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(embedder=FakeEmbedder(), store=store, default_top_k=2),
        bm25_retriever=BM25Retriever(chunks=chunks, default_top_k=2),
        default_top_k=2,
    )

    results = retriever.retrieve("alpha", top_k=2)

    assert results[0].chunk.chunk_id == "doc:chunk:0"
    assert results[0].metadata["fusion"] == "reciprocal_rank_fusion"
