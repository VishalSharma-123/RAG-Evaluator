from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.ingestion.embedders import Embedder
from rag_evaluator.ingestion.stores import VectorStore
from rag_evaluator.retrieval.base import Retriever
from rag_evaluator.schemas import RetrievedChunk


@dataclass(frozen=True)
class VectorRetriever(Retriever):
    """
    Dense vector retriever backed by a VectorStore.
    """

    embedder: Embedder
    store: VectorStore
    default_top_k: int = 10
    retriever_name: str = "vector"

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed the query and retrieve nearest chunks from the vector store.
        """
        k = top_k or self.default_top_k

        if k < 1:
            raise ValueError("top_k must be >= 1")

        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self.embedder.embed_query(query)

        return self.store.search(
            query_embedding,
            top_k=k,
            retriever_name=self.retriever_name,
        )
