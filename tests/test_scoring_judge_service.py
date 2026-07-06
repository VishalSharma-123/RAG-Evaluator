from __future__ import annotations

from typing import Any

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import GeneratedAnswer, QuestionType
from rag_evaluator.scoring.judges.base import JudgeScoringError
from rag_evaluator.scoring.judges.service import LLMJudgeService
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.types import ProviderGenerationResult


class FakeProvider:
    def __init__(self, result: ProviderGenerationResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderGenerationResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata,
            }
        )
        return self._result


def test_llm_judge_service_scores_with_provider_response(monkeypatch: pytest.MonkeyPatch, make_sample, make_chunk) -> None:
    fake_provider = FakeProvider(
        ProviderGenerationResult(
            content='{"faithfulness": 0.75, "relevance": 0.8, "hallucination": 0.1, "bert_score": 0.6}',
            response_metadata={"provider": "openrouter", "id": "resp-1"},
        )
    )
    monkeypatch.setattr(
        "rag_evaluator.scoring.judges.service.build_synthetic_provider",
        lambda config: fake_provider,
    )

    service = LLMJudgeService(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "metadata": {"api_key": "secret"},
            }
        )
    )
    sample = make_sample(question_type=QuestionType.FACTOID)
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    metrics = service.score(
        sample,
        generated_answer,
        context_chunks=[make_chunk(text="Paris is the capital of France.")],
        metadata={"pipeline_name": "unit"},
    )

    assert metrics.faithfulness == 0.75
    assert metrics.relevance == 0.8
    assert metrics.hallucination == 0.1
    assert metrics.bert_score == 0.6
    assert len(fake_provider.calls) == 1
    assert "Question:" in fake_provider.calls[0]["user_prompt"]
    assert fake_provider.calls[0]["metadata"] == {"pipeline_name": "unit"}


def test_llm_judge_service_wraps_provider_errors(monkeypatch: pytest.MonkeyPatch, make_sample) -> None:
    class ExplodingProvider:
        def generate_json(self, **kwargs: Any) -> ProviderGenerationResult:
            raise SyntheticProviderError("boom")

    monkeypatch.setattr(
        "rag_evaluator.scoring.judges.service.build_synthetic_provider",
        lambda config: ExplodingProvider(),
    )

    service = LLMJudgeService(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "metadata": {"api_key": "secret"},
            }
        )
    )
    sample = make_sample()
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    with pytest.raises(JudgeScoringError, match="Judge request failed"):
        service.score(sample, generated_answer, context_chunks=[])
