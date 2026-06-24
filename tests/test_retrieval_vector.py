from __future__ import annotations

from rag_evaluator.ingestion.stores import InMemoryVectorStore
from rag_evaluator.retrieval.vector import VectorRetriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in query else [0.0, 1.0]


def test_vector_retriever_uses_embedder_and_store(make_chunk) -> None:
    store = InMemoryVectorStore()
    chunks = [
        make_chunk(chunk_id="doc:chunk:0", text="alpha"),
        make_chunk(chunk_id="doc:chunk:1", text="beta"),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    retriever = VectorRetriever(embedder=FakeEmbedder(), store=store, default_top_k=2)

    results = retriever.retrieve("alpha")

    assert results[0].chunk.chunk_id == "doc:chunk:0"
