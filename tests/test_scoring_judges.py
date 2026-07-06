from __future__ import annotations

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import GeneratedAnswer, QuestionType
from rag_evaluator.scoring.judges import HeuristicJudge, OpenAIJudge, OpenRouterJudge


def test_heuristic_judge_scores_answerable_sample(make_sample, make_chunk) -> None:
    judge = HeuristicJudge()
    sample = make_sample(
        question_type=QuestionType.FACTOID,
        question="What is the capital of France?",
        reference_answer="Paris",
    )
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    metrics = judge.score(
        sample,
        generated_answer,
        context_chunks=[make_chunk(text="Paris is the capital of France.")],
    )

    assert metrics.faithfulness == 1.0
    assert metrics.hallucination == 0.0


def test_openai_judge_rejects_non_openai_provider() -> None:
    with pytest.raises(ValueError, match="config.provider"):
        OpenAIJudge(
            config=LLMConfig.model_validate(
                {
                    "provider": "openrouter",
                    "model": "gpt-5-mini",
                }
            )
        )


def test_openrouter_judge_rejects_non_openrouter_provider() -> None:
    with pytest.raises(ValueError, match="openrouter"):
        OpenRouterJudge(
            config=LLMConfig.model_validate(
                {
                    "provider": "openai",
                    "model": "gpt-5-mini",
                }
            )
        )
