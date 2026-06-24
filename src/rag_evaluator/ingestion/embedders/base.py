from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

Embedding = list[float]


class Embedder(ABC):
    """
    Base interface for text embedders.
    """

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts.
        """
        raise NotImplementedError

    def embed_query(self, query: str) -> Embedding:
        """
        Embed a single query string.
        """
        return self.embed_texts([query])[0]
