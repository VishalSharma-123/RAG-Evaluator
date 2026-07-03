from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.execution.types import Reranker
from rag_evaluator.generation.base import Generator
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, RetrievedChunk


@dataclass(frozen=True)
class SimpleExtractiveGenerator(Generator):
    """
    Deterministic generator used for smoke tests and provider fallbacks.
    """

    model_name: str = "simple_extractive_generator"

    def generate(
        self,
        sample: EvalSample,
        context_chunks: list[Chunk],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        start_time = time.perf_counter()

        if context_chunks:
            answer = context_chunks[0].text
        elif sample.is_answerable:
            answer = ""
        else:
            answer = "I don't know"

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        return GeneratedAnswer(
            sample_id=sample.sample_id,
            answer=answer,
            model_name=self.model_name,
            latency_ms=latency_ms,
            cost_usd=0.0,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class FallbackGenerator(Generator):
    """
    Wrap a primary generator and fall back cleanly when it is not yet usable.
    """

    primary: Generator
    fallback: Generator = field(default_factory=SimpleExtractiveGenerator)
    fallback_reason: str = "configured_generator_not_implemented"
    configured_model: str | None = None

    def generate(
        self,
        sample: EvalSample,
        context_chunks: list[Chunk],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        base_metadata = metadata or {}

        try:
            answer = self.primary.generate(
                sample,
                context_chunks,
                metadata=base_metadata,
            )
        except NotImplementedError:
            answer = self.fallback.generate(
                sample,
                context_chunks,
                metadata={
                    **base_metadata,
                    "generator_fallback_reason": self.fallback_reason,
                    "configured_generator_model": self.configured_model,
                },
            )
        else:
            answer = answer.model_copy(
                update={
                    "metadata": {
                        **base_metadata,
                        **answer.metadata,
                    }
                }
            )

        return answer


@dataclass(frozen=True)
class PassThroughReranker(Reranker):
    """
    Reranker placeholder that preserves retrieval ordering.
    """

    configured_type: str
    implementation_name: str = "pass_through"
    implemented: bool = False

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        del sample
        selected = retrieved_chunks[:top_k]
        reranked: list[RetrievedChunk] = []

        for rank, item in enumerate(selected, start=1):
            reranked.append(item.model_copy(update={"rank": rank}))

        return reranked
