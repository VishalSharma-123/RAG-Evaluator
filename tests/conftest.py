from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_evaluator.schemas import (
    Chunk,
    EvalSample,
    QuestionType,
    RetrievedChunk,
    RetrievalMetrics,
)


@pytest.fixture
def make_sample() -> Callable[..., EvalSample]:
    def _make_sample(**overrides: object) -> EvalSample:
        data: dict[str, object] = {
            "sample_id": "sample-1",
            "question": "What is RAG?",
            "reference_answer": "Retrieval augmented generation.",
            "question_type": QuestionType.FACTOID,
            "source_dataset": "unit",
            "source_split": "test",
            "evidence_chunk_ids": ["doc:chunk:0"],
        }
        data.update(overrides)
        return EvalSample.model_validate(data)

    return _make_sample


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    def _make_chunk(**overrides: object) -> Chunk:
        data: dict[str, object] = {
            "chunk_id": "doc:chunk:0",
            "document_id": "doc",
            "text": "Retrieval augmented generation uses retrieved context.",
            "start_char": 0,
            "end_char": 54,
        }
        data.update(overrides)
        return Chunk.model_validate(data)

    return _make_chunk


@pytest.fixture
def make_retrieved_chunk(make_chunk: Callable[..., Chunk]) -> Callable[..., RetrievedChunk]:
    def _make_retrieved_chunk(**overrides: object) -> RetrievedChunk:
        chunk = overrides.pop("chunk", make_chunk())
        data: dict[str, object] = {
            "chunk": chunk,
            "rank": 1,
            "score": 1.0,
            "retriever_name": "vector",
        }
        data.update(overrides)
        return RetrievedChunk.model_validate(data)

    return _make_retrieved_chunk


@pytest.fixture
def retrieval_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=1.0,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg=1.0,
    )
