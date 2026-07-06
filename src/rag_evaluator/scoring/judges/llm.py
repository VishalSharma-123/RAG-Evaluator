from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics
from rag_evaluator.scoring.judges.base import GenerationJudge
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge
from rag_evaluator.scoring.judges.service import LLMJudgeService


@dataclass(frozen=True)
class LLMJudge(GenerationJudge):
    """
    Shared base for provider-backed LLM judges.

    Until live judge prompts are implemented, this falls back to the heuristic
    judge so callers can adopt the class hierarchy without breaking behavior.
    """

    config: LLMConfig
    heuristic_judge: HeuristicJudge = field(default_factory=HeuristicJudge)
    service: LLMJudgeService = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service",
            LLMJudgeService(config=self.config, heuristic_judge=self.heuristic_judge),
        )

    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        return self.service.score_with_fallback(
            sample,
            generated_answer,
            context_chunks=context_chunks,
            metadata=metadata,
        )

    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider name handled by this judge implementation.
        """
        raise NotImplementedError
