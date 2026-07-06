from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rag_evaluator.schemas import (
    Chunk,
    EvalSample,
    FailureMode,
    FinalContext,
    GeneratedAnswer,
    GenerationMetrics,
    RetrievalMetrics,
    RetrievedChunk,
)


class ChunkRelevanceStrategy(StrEnum):
    EXACT_EVIDENCE_ID = "exact_evidence_id"
    EVIDENCE_SPAN_OVERLAP = "evidence_span_overlap"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"

@dataclass(frozen=True)
class ChunkRelevanceScore:
    chunk_id: str
    strategies: dict[str, float | bool | None] = field(default_factory=dict)
    overall_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ScoringRequest:
    run_id: str
    sample: EvalSample
    retrieved_chunks: list[RetrievedChunk]
    final_context: FinalContext = field(default_factory=FinalContext)
    generated_answer: GeneratedAnswer | None = None
    retrieval_k: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> EvalSample:
        """
        Backward-compatible alias for older callers.
        """
        return self.sample

@dataclass(frozen=True)
class ScoredSample:
    run_id: str
    sample: EvalSample
    retrieved_chunks: list[RetrievedChunk]
    final_context: FinalContext
    generated_answer: GeneratedAnswer | None
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics | None = None
    chunk_relevance: list[ChunkRelevanceScore] = field(default_factory=list)
    retrieval_failure_modes: list[FailureMode] = field(default_factory=list)
    generation_failure_modes: list[FailureMode] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_context_chunks(self) -> list[Chunk]:
        return self.final_context.chunks
