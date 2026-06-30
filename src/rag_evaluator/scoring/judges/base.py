from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics


class JudgeScoringError(ValueError):
    """
    Raised when judge-based scoring fails or returns invalid output.
    """


class GenerationJudge(ABC):
    """
    Base interface for generation judges.
    """

    @abstractmethod
    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        """
        Score one generated answer against the sample and retrieved context.
        """
        raise NotImplementedError
