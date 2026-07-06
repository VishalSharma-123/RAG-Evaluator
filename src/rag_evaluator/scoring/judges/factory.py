from __future__ import annotations

from rag_evaluator.config import LLMConfig, LLMProvider
from rag_evaluator.scoring.judges.base import GenerationJudge
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge
from rag_evaluator.scoring.judges.openai import OpenAIJudge
from rag_evaluator.scoring.judges.openrouter import OpenRouterJudge


def build_judge(config: LLMConfig) -> GenerationJudge:
    """
    Build the judge implementation for a configured LLM Provider
    :param config:
    :return:
    """
    if config.provider in {LLMProvider.OPENAI, LLMProvider.OPENROUTER}:
        if config.provider == LLMProvider.OPENAI:
            return OpenAIJudge(config=config)

        return OpenRouterJudge(config=config)

    return HeuristicJudge()
