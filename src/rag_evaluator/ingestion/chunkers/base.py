from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.schemas import Chunk


@dataclass(frozen=True)
class SourceDocument:
    """
    Raw source document before chunking.
    """

    document_id: str
    text: str
    metadata: dict[str, object] | None = None


class Chunker(ABC):
    """
    Base interface for document chunkers.
    """

    @abstractmethod
    def chunk(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        """
        Split documents into chunks.
        """
        raise NotImplementedError
