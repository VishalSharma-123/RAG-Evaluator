from __future__ import annotations

from abc import ABC, abstractmethod

from rag_evaluator.schemas import RetrievedChunk


class Retriever(ABC):
    """
    Base interface for retrieval strategies.
    """

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve chunks for a query.
        """
        raise NotImplementedError
