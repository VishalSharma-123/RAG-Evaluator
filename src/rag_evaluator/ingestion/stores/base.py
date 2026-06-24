from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_evaluator.ingestion.embedders import Embedding
from rag_evaluator.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    """
    Base interface for vector stores.
    """

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """
        Add chunks and corresponding embeddings to the store.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int,
        retriever_name: str = "vector",
    ) -> list[RetrievedChunk]:
        """
        Search for nearest chunks by query embedding.
        """
        raise NotImplementedError
