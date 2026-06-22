from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.chunkers.base import Chunker, SourceDocument
from rag_evaluator.ingestion.chunkers.utils import chunk_id
from rag_evaluator.schemas import Chunk


@dataclass(frozen=True)
class FixedSizeChunker(Chunker):
    """
    Character-based fixed-size chunker with overlap.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            text = document.text
            start = 0
            chunk_index = 0
            step = self.chunk_size - self.chunk_overlap

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]

                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id(document.document_id, chunk_index),
                            document_id=document.document_id,
                            text=chunk_text,
                            start_char=start,
                            end_char=end,
                            metadata=document.metadata or {},
                        )
                    )
                    chunk_index += 1

                start += step

        return chunks
