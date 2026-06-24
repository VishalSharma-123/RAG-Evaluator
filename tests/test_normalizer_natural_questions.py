from __future__ import annotations

from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.normalizers.general.natural_questions import (
    NaturalQuestionsNormalizer,
)


def test_natural_questions_normalizer_extracts_question_text_from_dict() -> None:
    normalizer = NaturalQuestionsNormalizer(
        DatasetConfig(
            name="natural_questions",
            source=DatasetSource.HUGGINGFACE,
            dataset_name="google-research-datasets/natural_questions",
            split="validation",
        )
    )

    sample = normalizer.normalize_record(
        {
            "id": "123",
            "question": {"text": "who won the first nobel prize in physics"},
            "short_answers": [{"text": "Wilhelm Conrad Rontgen"}],
        },
        index=0,
        split="validation",
    )

    assert sample.question == "who won the first nobel prize in physics"
    assert sample.reference_answer == "Wilhelm Conrad Rontgen"
