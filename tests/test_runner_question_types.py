from __future__ import annotations

from rag_evaluator.config import PipelineConfig
from rag_evaluator.execution.fallbacks import SimpleExtractiveGenerator
from rag_evaluator.execution.runner import run_pipeline_variant
from rag_evaluator.execution.types import PipelineRuntime
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge
from rag_evaluator.schemas import Chunk, QuestionType, RetrievedChunk


class _FakeRetriever:
    def __init__(self, retrieved_chunks: list[RetrievedChunk]) -> None:
        self._retrieved_chunks = retrieved_chunks

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedChunk]:
        del question
        return self._retrieved_chunks[:top_k]


class _PassThroughReranker:
    configured_type = "none"
    implementation_name = "pass_through"
    implemented = False

    def rerank(self, sample, retrieved_chunks, *, top_k: int):
        del sample
        return retrieved_chunks[:top_k]


class _SelectSecondChunkReranker:
    configured_type = "cross_encoder"
    implementation_name = "test_selector"
    implemented = True

    def rerank(self, sample, retrieved_chunks, *, top_k: int):
        del sample
        return retrieved_chunks[1:2][:top_k]


def _build_pipeline() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 2},
            "generator": {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
            "judge": {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
        }
    )


def test_run_pipeline_variant_persists_question_type_signals(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
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
    runtime = PipelineRuntime(
        pipeline=_build_pipeline(),
        chunks=[chunk],
        retriever=_FakeRetriever([make_retrieved_chunk(chunk=chunk)]),
        reranker=_PassThroughReranker(),
        generator=SimpleExtractiveGenerator(),
        judge=HeuristicJudge(),
    )

    output = run_pipeline_variant(runtime=runtime, samples=[sample])

    assert len(output.results) == 1
    result = output.results[0]
    assert result.generation_metrics is not None
    assert result.final_context.rendered_text == "[1] Paris is larger than Berlin."
    assert result.final_context_chunks == [chunk]
    assert "question_type_signals" in result.metadata
    assert result.metadata["question_type_signals"]["performed_comparison"] is True
    assert result.metadata["question_type_signals"]["covered_key_entities"] is True


def test_run_pipeline_variant_separates_retrieved_chunks_from_final_context(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
) -> None:
    first_chunk = make_chunk(
        chunk_id="doc:chunk:0",
        text="Background context that does not answer the question.",
    )
    second_chunk = make_chunk(
        chunk_id="doc:chunk:1",
        text="Retrieval augmented generation uses retrieved context.",
    )
    retrieved_chunks = [
        make_retrieved_chunk(chunk=first_chunk, rank=1, score=0.9),
        make_retrieved_chunk(chunk=second_chunk, rank=2, score=0.8),
    ]
    sample = make_sample(
        evidence_chunk_ids=["doc:chunk:1"],
        reference_answer="Retrieval augmented generation uses retrieved context.",
    )
    runtime = PipelineRuntime(
        pipeline=_build_pipeline(),
        chunks=[first_chunk, second_chunk],
        retriever=_FakeRetriever(retrieved_chunks),
        reranker=_SelectSecondChunkReranker(),
        generator=SimpleExtractiveGenerator(),
        judge=HeuristicJudge(),
    )

    output = run_pipeline_variant(runtime=runtime, samples=[sample])

    result = output.results[0]
    assert [item.chunk.chunk_id for item in result.retrieved_chunks] == [
        "doc:chunk:0",
        "doc:chunk:1",
    ]
    assert result.final_context.rendered_text == "[1] Retrieval augmented generation uses retrieved context."
    assert [chunk.chunk_id for chunk in result.final_context_chunks] == ["doc:chunk:1"]
    assert result.generated_answer is not None
    assert result.generated_answer.answer == second_chunk.text
