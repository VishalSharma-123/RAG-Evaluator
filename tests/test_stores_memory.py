from __future__ import annotations

import pytest

from rag_evaluator.ingestion.stores.memory import InMemoryVectorStore


def test_memory_store_ranks_by_cosine_similarity(make_chunk) -> None:
    store = InMemoryVectorStore()
    chunks = [
        make_chunk(chunk_id="doc:chunk:0", text="alpha"),
        make_chunk(chunk_id="doc:chunk:1", text="beta"),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=1)

    assert results[0].chunk.chunk_id == "doc:chunk:0"
    assert results[0].rank == 1


def test_memory_store_rejects_mismatched_lengths(make_chunk) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="same length"):
        store.add([make_chunk()], [])
