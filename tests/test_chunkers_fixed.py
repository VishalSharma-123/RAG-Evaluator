from __future__ import annotations

from rag_evaluator.ingestion.chunkers.base import SourceDocument
from rag_evaluator.ingestion.chunkers.fixed import FixedSizeChunker


def test_fixed_chunker_produces_stable_chunk_ids() -> None:
    chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.chunk([SourceDocument(document_id="doc", text="abcdefghij")])

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc:chunk:0",
        "doc:chunk:1",
        "doc:chunk:2",
    ]
    assert chunks[0].text == "abcde"
    assert chunks[1].start_char == 4
