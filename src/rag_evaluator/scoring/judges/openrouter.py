from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.config import LLMProvider
from rag_evaluator.scoring.judges.llm import LLMJudge


@dataclass(frozen=True)
class OpenRouterJudge(LLMJudge):
    """
    Judge wrapper for OpenRouter-backed scoring.
    """
    
    def __post_init__(self) -> None:
        if self.config.provider != LLMProvider.OPENROUTER:
            raise ValueError("OpenRouterJudge requires config.provider to be `openrouter`.")
        super().__post_init__()
    
    def provider_name(self) -> str:
        return LLMProvider.OPENROUTER.value
