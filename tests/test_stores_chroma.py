from __future__ import annotations

from pathlib import Path

from rag_evaluator.ingestion.stores.chroma import ChromaVectorStore


def test_chroma_store_round_trip(tmp_path: Path, make_chunk) -> None:
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory=tmp_path,
    )
    chunks = [
        make_chunk(chunk_id="doc:chunk:0", text="alpha"),
        make_chunk(chunk_id="doc:chunk:1", text="beta", start_char=10, end_char=14),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=1)

    assert results[0].chunk.chunk_id == "doc:chunk:0"
    assert results[0].chunk.document_id == "doc"
