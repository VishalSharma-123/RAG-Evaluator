from __future__ import annotations

from rag_evaluator.config import PipelineConfig
from rag_evaluator.ingestion.chunkers import SourceDocument
from rag_evaluator.runner import run_single_pipeline
from rag_evaluator.schemas import Chunk, QuestionType, RetrievedChunk


class _FakeChunker:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, documents) -> list[Chunk]:
        return self._chunks


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _FakeVectorStore:
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.chunks = chunks
        self.embeddings = embeddings


class _FakeRetriever:
    def __init__(self, retrieved_chunks: list[RetrievedChunk]) -> None:
        self._retrieved_chunks = retrieved_chunks

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedChunk]:
        return self._retrieved_chunks[:top_k]


def test_run_single_pipeline_persists_question_type_signals(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    sample = make_sample(
        question_type=QuestionType.COMPARATIVE,
        question="Which city is larger, Paris or Berlin?",
        reference_answer="Paris is larger than Berlin.",
        metadata={"comparison_targets": ["Paris", "Berlin"]},
    )
    chunk = make_chunk(
        text="Paris is larger than Berlin.",
        metadata={"source": "unit"},
    )
    retrieved_chunk = make_retrieved_chunk(chunk=chunk)

    monkeypatch.setattr(
        "rag_evaluator.runner.build_chunker",
        lambda **kwargs: _FakeChunker([chunk]),
    )
    monkeypatch.setattr(
        "rag_evaluator.runner.build_embedder",
        lambda config: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "rag_evaluator.runner.build_vector_store",
        lambda **kwargs: _FakeVectorStore(),
    )
    monkeypatch.setattr(
        "rag_evaluator.runner.build_retriever",
        lambda **kwargs: _FakeRetriever([retrieved_chunk]),
    )

    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 1},
            "generator": {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
            "judge": {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
        }
    )

    output = run_single_pipeline(
        pipeline=pipeline,
        samples=[sample],
        documents=[SourceDocument(document_id="doc", text=chunk.text)],
    )

    assert len(output.results) == 1
    result = output.results[0]
    assert result.generation_metrics is not None
    assert "question_type_signals" in result.metadata
    assert result.metadata["question_type_signals"]["performed_comparison"] is True
    assert result.metadata["question_type_signals"]["covered_key_entities"] is True
