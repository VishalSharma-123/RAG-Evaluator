from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.config import LLMProvider
from rag_evaluator.scoring.judges.llm import LLMJudge


@dataclass(frozen=True)
class OpenAIJudge(LLMJudge):
    """
    Judge wrapper for OpenAI-backed scoring.

    Today this reuses the deterministic heuristic path. Later, this can be
    replaced with an OpenAI-backed judge prompt without changing callers.
    """

    def __post_init__(self) -> None:
        if self.config.provider != LLMProvider.OPENAI:
            raise ValueError("OpenAIJudge requires config.provider to be `openai`.")

    def provider_name(self) -> str:
        return LLMProvider.OPENAI.value
