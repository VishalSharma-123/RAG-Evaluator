from __future__ import annotations

from types import SimpleNamespace

from rag_evaluator.config import PipelineConfig
from rag_evaluator.execution.fallbacks import SimpleExtractiveGenerator
from rag_evaluator.execution.runner import run_sample
from rag_evaluator.execution.runtime import build_pipeline_runtime
from rag_evaluator.execution.types import PipelineRuntime
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge


def test_build_pipeline_runtime_uses_reranker_factory(monkeypatch, make_chunk) -> None:
    captured: dict[str, object] = {}

    fake_chunk = make_chunk(chunk_id="doc:chunk:1", text="chunk text")
    fake_runtime_reranker = SimpleNamespace(
        configured_type="none",
        implementation_name="test",
        implemented=True,
    )

    class FakeChunker:
        def chunk(self, documents):
            captured["documents"] = list(documents)
            return [fake_chunk]

    class FakeRetriever:
        def retrieve(self, question: str, *, top_k: int):
            captured["retrieval_top_k"] = top_k
            return []

    monkeypatch.setattr(
        "rag_evaluator.execution.runtime.build_chunker",
        lambda **kwargs: FakeChunker(),
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.runtime.build_reranker",
        lambda pipeline: fake_runtime_reranker,
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.runtime.build_retriever",
        lambda **kwargs: FakeRetriever(),
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.runtime.build_generator",
        lambda pipeline: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.runtime.build_judge",
        lambda pipeline: SimpleNamespace(),
    )

    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "bm25", "top_k": 4},
        }
    )

    runtime = build_pipeline_runtime(
        pipeline=pipeline,
        documents=[SimpleNamespace(document_id="doc", text="doc text")],
    )

    assert runtime.chunks == [fake_chunk]
    assert runtime.reranker is fake_runtime_reranker
    assert runtime.metadata["reranker_type"] == "none"
    assert captured["documents"][0].document_id == "doc"


def test_run_sample_uses_retriever_top_k_when_reranker_top_k_is_missing(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
):
    chunk = make_chunk(chunk_id="doc:chunk:1")
    retrieved = [
        make_retrieved_chunk(chunk=chunk, rank=1, score=0.9),
        make_retrieved_chunk(
            chunk=make_chunk(chunk_id="doc:chunk:2", text="second chunk"),
            rank=2,
            score=0.8,
        ),
    ]

    class RecordingReranker:
        configured_type = "none"
        implementation_name = "recording"
        implemented = True

        def rerank(self, sample, retrieved_chunks, *, top_k: int):
            assert top_k == 2
            return retrieved_chunks[:top_k]

    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 2},
            "reranker": {"type": "none"},
        }
    )

    runtime = PipelineRuntime(
        pipeline=pipeline,
        chunks=[chunk],
        retriever=SimpleNamespace(retrieve=lambda question, *, top_k: retrieved),
        reranker=RecordingReranker(),
        generator=SimpleExtractiveGenerator(),
        judge=HeuristicJudge(),
    )

    result = run_sample(runtime=runtime, sample=make_sample())

    assert len(result.retrieved_chunks) == 2
    assert result.final_context.chunks == [chunk, retrieved[1].chunk]
    assert result.metadata["retrieval_gold_strategy"] == "exact_evidence_id"
    assert result.metadata["resolved_gold_chunk_ids"] == ["doc:chunk:0"]
    assert result.metadata["retrieved_relevance_flags"] == [False, False]
