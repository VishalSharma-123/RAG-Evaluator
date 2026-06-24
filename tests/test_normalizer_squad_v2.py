from __future__ import annotations

from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.normalizers.general.squad_v2 import SquadV2Normalizer
from rag_evaluator.schemas import QuestionType


def test_squad_v2_normalizer_builds_answerable_sample() -> None:
    normalizer = SquadV2Normalizer(
        DatasetConfig(name="squad_v2", source=DatasetSource.HUGGINGFACE, split="validation")
    )

    sample = normalizer.normalize_record(
        {
            "id": "q1",
            "question": "What is RAG?",
            "title": "RAG",
            "context": "RAG stands for retrieval augmented generation.",
            "answers": {
                "text": ["retrieval augmented generation", "RAG"],
                "answer_start": [15, 15],
            },
        },
        index=0,
        split="validation",
    )

    assert sample.question_type == QuestionType.FACTOID
    assert sample.reference_answer == "retrieval augmented generation"
    assert sample.answer_aliases == ["RAG"]
    assert sample.evidence_spans[0].start_char == 15


def test_squad_v2_normalizer_builds_unanswerable_sample() -> None:
    normalizer = SquadV2Normalizer(
        DatasetConfig(name="squad_v2", source=DatasetSource.HUGGINGFACE, split="validation")
    )

    sample = normalizer.normalize_record(
        {
            "id": "q2",
            "question": "What is missing?",
            "answers": {"text": [], "answer_start": []},
        },
        index=0,
        split="validation",
    )

    assert sample.question_type == QuestionType.UNANSWERABLE
    assert sample.reference_answer is None
    assert sample.is_answerable is False
