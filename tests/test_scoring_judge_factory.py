from __future__ import annotations

from rag_evaluator.config import LLMConfig
from rag_evaluator.scoring.judges import OpenAIJudge, OpenRouterJudge, build_judge
from rag_evaluator.scoring.judges.service import LLMJudgeService


def test_build_judge_returns_openai_judge_for_openai_provider() -> None:
    judge = build_judge(
        LLMConfig.model_validate(
            {
                "provider": "openai",
                "model": "gpt-5-mini",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    assert isinstance(judge, OpenAIJudge)
    assert isinstance(judge, LLMJudgeService) is False


def test_build_judge_returns_openrouter_judge_for_openrouter_provider() -> None:
    judge = build_judge(
        LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    assert isinstance(judge, OpenRouterJudge)


def test_build_judge_accepts_any_openrouter_model() -> None:
    judge = build_judge(
        LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "some/custom-model",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    assert isinstance(judge, OpenRouterJudge)
