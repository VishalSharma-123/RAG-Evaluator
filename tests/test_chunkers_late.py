from __future__ import annotations

from rag_evaluator.ingestion.chunkers.base import SourceDocument
from rag_evaluator.ingestion.chunkers.late import LateChunker


def test_late_chunker_records_anchor_and_expanded_metadata() -> None:
    text = "One short sentence. Two more words here. Three wraps this up."
    chunker = LateChunker(chunk_size=20, context_sentences=1)

    chunks = chunker.chunk([SourceDocument(document_id="doc", text=text)])

    assert chunks
    first = chunks[0]
    assert first.metadata["chunking_strategy"] == "late"
    assert first.metadata["anchor_start_char"] >= first.start_char
    assert first.metadata["anchor_end_char"] <= first.end_char
