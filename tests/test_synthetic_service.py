from __future__ import annotations

import pytest

from rag_evaluator.schemas import QuestionType
from rag_evaluator.synthetic.errors import SyntheticGenerationError, SyntheticValidationError
from rag_evaluator.synthetic.providers.base import LLMProviderClient
from rag_evaluator.synthetic.service import SyntheticGenerationService
from rag_evaluator.synthetic.types import ProviderGenerationResult


class FakeProvider(LLMProviderClient):
    def __init__(self, content: str) -> None:
        self.content = content

    def generate_json(self, *, system_prompt: str, user_prompt: str, metadata=None) -> ProviderGenerationResult:
        return ProviderGenerationResult(content=self.content)


def test_synthetic_generation_service_rejects_empty_chunks() -> None:
    service = SyntheticGenerationService(provider=FakeProvider('{"samples": []}'))

    with pytest.raises(SyntheticGenerationError, match="At least one chunk"):
        service.generate_samples([])


def test_synthetic_generation_service_merges_metadata(make_chunk) -> None:
    service = SyntheticGenerationService(
        provider=FakeProvider(
            """
            {
              "samples": [
                {
                  "sample_id": "s1",
                  "question": "What is the capital of France?",
                  "reference_answer": "Paris",
                  "question_type": "factoid",
                  "evidence_chunk_ids": ["doc:chunk:0"],
                  "is_answerable": true,
                  "metadata": {"source": "model"}
                }
              ]
            }
            """
        )
    )

    samples = service.generate_samples(
        [make_chunk(text="Paris is the capital of France.")],
        question_types=[QuestionType.FACTOID],
        metadata={"pipeline": "synthetic"},
    )

    assert samples[0].metadata == {
        "pipeline": "synthetic",
        "source": "model",
    }


def test_synthetic_generation_service_enforces_grounding(make_chunk) -> None:
    service = SyntheticGenerationService(
        provider=FakeProvider(
            """
            {
              "samples": [
                {
                  "sample_id": "s1",
                  "question": "What is the capital of France?",
                  "reference_answer": "Berlin",
                  "question_type": "factoid",
                  "evidence_chunk_ids": ["doc:chunk:0"],
                  "is_answerable": true,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    with pytest.raises(SyntheticValidationError, match="not grounded"):
        service.generate_samples(
            [make_chunk(text="Paris is the capital of France.")],
            question_types=[QuestionType.FACTOID],
        )
