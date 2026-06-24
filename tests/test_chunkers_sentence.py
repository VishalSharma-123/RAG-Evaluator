from __future__ import annotations

from rag_evaluator.ingestion.chunkers.base import SourceDocument
from rag_evaluator.ingestion.chunkers.sentence import SentenceChunker


def test_sentence_chunker_groups_sentences_by_size() -> None:
    text = "Alpha one. Beta two. Gamma three."
    chunker = SentenceChunker(chunk_size=20, chunk_overlap_sentences=0)

    chunks = chunker.chunk([SourceDocument(document_id="doc", text=text)])

    assert len(chunks) == 3
    assert chunks[0].text.startswith("Alpha one.")
    assert chunks[1].text.strip().startswith("Beta")
    assert chunks[2].text.strip().startswith("Gamma")
