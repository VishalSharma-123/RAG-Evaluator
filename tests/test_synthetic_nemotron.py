from __future__ import annotations

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import QuestionType
from rag_evaluator.synthetic.errors import SyntheticGenerationError
from rag_evaluator.synthetic.models.nemotron import NemotronSyntheticGenerator


def test_nemotron_synthetic_generator_rejects_non_nemotron_model() -> None:
    with pytest.raises(SyntheticGenerationError, match="Unsupported model family"):
        NemotronSyntheticGenerator(
            config=LLMConfig.model_validate(
                {
                    "provider": "openai",
                    "model": "gpt-test",
                }
            )
        )


def test_nemotron_synthetic_generator_delegates_to_service(
    make_chunk,
    make_sample,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, *, provider) -> None:
            captured["provider"] = provider

        def generate_samples(self, chunks, *, question_types=None, max_samples=None, metadata=None):
            captured["chunks"] = chunks
            captured["question_types"] = question_types
            captured["max_samples"] = max_samples
            captured["metadata"] = metadata
            return [make_sample()]

    fake_provider = object()

    monkeypatch.setattr(
        "rag_evaluator.synthetic.models.nemotron.build_synthetic_provider",
        lambda config: fake_provider,
    )
    monkeypatch.setattr(
        "rag_evaluator.synthetic.models.nemotron.SyntheticGenerationService",
        FakeService,
    )

    generator = NemotronSyntheticGenerator(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
            }
        )
    )

    samples = generator.generate_samples(
        [make_chunk()],
        question_types=[QuestionType.FACTOID],
        max_samples=3,
        metadata={"pipeline": "synthetic"},
    )

    assert len(samples) == 1
    assert captured["provider"] is fake_provider
    assert captured["question_types"] == [QuestionType.FACTOID]
    assert captured["max_samples"] == 3
    assert captured["metadata"] == {"pipeline": "synthetic"}
