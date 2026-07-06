from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
from rag_evaluator.scoring.engine.types import ChunkRelevanceScore, ScoredSample


class ChunkRelevanceScorer(ABC):
    @abstractmethod
    def score_chunk(
            self,
            sample: EvalSample,
            retrieved_chunk: RetrievedChunk,
            *,
            context_chunks: list[Chunk],
            metadata: dict[str, Any] | None = None,
    ) -> ChunkRelevanceScore:
        raise NotImplementedError

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

class SampleScorer(ABC):
    @abstractmethod
    def score_sample(
        self,
        *,
        run_id: str,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        final_context: FinalContext,
        generated_answer: GeneratedAnswer | None,
        retrieval_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScoredSample:
        raise NotImplementedError

class FailureClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        context_was_used: bool | None = None,
        hallucination_score: float | None = None,
        partial_answer_score: float | None = None,
        retrieval_k: int | None = None,
    ) -> list[FailureMode]:
        raise NotImplementedError


class GenerationScorer(ABC):
    @abstractmethod
    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics | None:
        raise NotImplementedError


class RetrievalScorer(ABC):
    @abstractmethod
    def score(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        k: int,
    ) -> RetrievalMetrics:
        raise NotImplementedError


ChunkRelevanceFn = Callable[
    [EvalSample, RetrievedChunk, list[Chunk], dict[str, Any] | None],
    float | None,
]
