from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_evaluator.question_types.signals import build_context_text, token_overlap_ratio
from rag_evaluator.schemas import Chunk, EvalSample, RetrievedChunk
from rag_evaluator.scoring.engine.base import ChunkRelevanceFn, ChunkRelevanceScorer
from rag_evaluator.scoring.engine.types import ChunkRelevanceScore, ChunkRelevanceStrategy


@dataclass(frozen=True)
class DefaultChunkRelevanceScorer(ChunkRelevanceScorer):
    """
    Deterministic chunk relevance scorer used by the engine layer.
    """

    llm_judge: ChunkRelevanceFn | None = None

    def score(
        self,
        sample: EvalSample,
        retrieved_chunk: RetrievedChunk,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> float | None:
        return self.score_chunk(
            sample,
            retrieved_chunk,
            context_chunks=context_chunks,
            metadata=metadata,
        ).overall_score

    def score_chunk(
        self,
        sample: EvalSample,
        retrieved_chunk: RetrievedChunk,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> ChunkRelevanceScore:
        metadata = metadata or {}
        chunk = retrieved_chunk.chunk
        evidence_chunk_ids = set(sample.evidence_chunk_ids)

        exact_match = chunk.chunk_id in evidence_chunk_ids
        span_overlap = _chunk_overlaps_evidence_spans(sample, chunk)
        semantic_similarity = _semantic_similarity(sample, chunk, context_chunks=context_chunks)
        llm_score = (
            self.llm_judge(sample, retrieved_chunk, context_chunks, metadata)
            if self.llm_judge is not None
            else None
        )

        strategies: dict[str, float | bool | None] = {
            ChunkRelevanceStrategy.EXACT_EVIDENCE_ID.value: exact_match,
            ChunkRelevanceStrategy.EVIDENCE_SPAN_OVERLAP.value: span_overlap,
            ChunkRelevanceStrategy.SEMANTIC_SIMILARITY.value: semantic_similarity,
            ChunkRelevanceStrategy.LLM_JUDGE.value: llm_score,
        }

        return ChunkRelevanceScore(
            chunk_id=chunk.chunk_id,
            strategies=strategies,
            overall_score=_combine_scores(
                exact_match=exact_match,
                span_overlap=span_overlap,
                semantic_similarity=semantic_similarity,
                llm_score=llm_score,
            ),
            metadata={
                "retriever_name": retrieved_chunk.retriever_name,
                "rank": retrieved_chunk.rank,
                "retrieval_score": retrieved_chunk.score,
                **metadata,
            },
        )

    def score_batch(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkRelevanceScore]:
        return [
            self.score_chunk(
                sample,
                retrieved_chunk,
                context_chunks=context_chunks,
                metadata=metadata,
            )
            for retrieved_chunk in retrieved_chunks
        ]


def score_chunk_relevance(
    sample: EvalSample,
    retrieved_chunk: RetrievedChunk,
    *,
    context_chunks: list[Chunk],
    llm_judge: ChunkRelevanceFn | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChunkRelevanceScore:
    """
    Score one retrieved chunk against the sample's evidence and context.
    """
    scorer = DefaultChunkRelevanceScorer(llm_judge=llm_judge)
    return scorer.score_chunk(
        sample,
        retrieved_chunk,
        context_chunks=context_chunks,
        metadata=metadata,
    )


def score_chunk_relevance_batch(
    sample: EvalSample,
    retrieved_chunks: list[RetrievedChunk],
    *,
    context_chunks: list[Chunk],
    llm_judge: ChunkRelevanceFn | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkRelevanceScore]:
    """
    Score all retrieved chunks for one sample.
    """
    scorer = DefaultChunkRelevanceScorer(llm_judge=llm_judge)
    return scorer.score_batch(
        sample,
        retrieved_chunks,
        context_chunks=context_chunks,
        metadata=metadata,
    )


def _chunk_overlaps_evidence_spans(sample: EvalSample, chunk: Chunk) -> bool:
    for span in sample.evidence_spans:
        if span.chunk_id is not None and span.chunk_id == chunk.chunk_id:
            return True

        if span.document_id != chunk.document_id:
            continue

        if chunk.start_char is None or chunk.end_char is None:
            continue

        if span.end_char <= chunk.start_char:
            continue

        if span.start_char >= chunk.end_char:
            continue

        return True

    return False


def _semantic_similarity(
    sample: EvalSample,
    chunk: Chunk,
    *,
    context_chunks: list[Chunk],
) -> float:
    evidence_text = _evidence_text(sample, context_chunks=context_chunks)

    if evidence_text.strip():
        return token_overlap_ratio(evidence_text, chunk.text)

    if sample.reference_answer:
        return token_overlap_ratio(sample.reference_answer, chunk.text)

    return token_overlap_ratio(sample.question, chunk.text)


def _evidence_text(
    sample: EvalSample,
    *,
    context_chunks: list[Chunk],
) -> str:
    span_texts = [
        span.text.strip()
        for span in sample.evidence_spans
        if span.text is not None and span.text.strip()
    ]
    if span_texts:
        return " ".join(span_texts)

    context_text = build_context_text(context_chunks)
    if context_text.strip():
        return context_text

    if sample.reference_answer:
        return sample.reference_answer

    return sample.question


def _combine_scores(
    *,
    exact_match: bool,
    span_overlap: bool,
    semantic_similarity: float,
    llm_score: float | None,
) -> float:
    values: list[float] = [
        1.0 if exact_match else 0.0,
        1.0 if span_overlap else 0.0,
        max(0.0, min(1.0, semantic_similarity)),
    ]

    if llm_score is not None:
        values.append(max(0.0, min(1.0, llm_score)))

    if not values:
        return 0.0

    return sum(values) / len(values)
