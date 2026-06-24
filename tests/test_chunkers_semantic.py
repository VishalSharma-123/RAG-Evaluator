from __future__ import annotations

from rag_evaluator.ingestion.chunkers.base import SourceDocument
from rag_evaluator.ingestion.chunkers.semantic import SemanticChunker


def test_semantic_chunker_adds_strategy_metadata() -> None:
    text = "Cats chase mice. Cats like milk. Quantum fields are complex."
    chunker = SemanticChunker(chunk_size=25, chunk_overlap_sentences=0, similarity_threshold=0.1)

    chunks = chunker.chunk([SourceDocument(document_id="doc", text=text)])

    assert chunks
    assert all(chunk.metadata["chunking_strategy"] == "semantic" for chunk in chunks)
