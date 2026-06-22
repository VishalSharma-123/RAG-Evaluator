from __future__ import annotations

from rag_evaluator.ingestion.chunkers.base import Chunker
from rag_evaluator.ingestion.chunkers.fixed import FixedSizeChunker
from rag_evaluator.ingestion.chunkers.late import LateChunker
from rag_evaluator.ingestion.chunkers.semantic import SemanticChunker
from rag_evaluator.ingestion.chunkers.sentence import SentenceChunker


def build_chunker(
    *,
    chunker_type: str,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
) -> Chunker:
    """
    Build a chunker by config type.
    """

    if chunker_type == "fixed":
        return FixedSizeChunker(
            chunk_size=chunk_size or 512,
            chunk_overlap=chunk_overlap,
        )

    if chunker_type == "sentence":
        return SentenceChunker(
            chunk_size=chunk_size or 512,
            chunk_overlap_sentences=chunk_overlap,
        )

    if chunker_type == "semantic":
        return SemanticChunker(
            chunk_size=chunk_size or 512,
            chunk_overlap_sentences=chunk_overlap,
        )

    if chunker_type == "late":
        return LateChunker(
            chunk_size=chunk_size or 512,
            context_sentences=chunk_overlap,
        )

    raise ValueError(f"Unknown chunker type: {chunker_type}")
