from __future__ import annotations

from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.normalizers.multihop.hotpot_qa import HotpotQANormalizer
from rag_evaluator.schemas import QuestionType


def test_hotpot_qa_normalizer_collects_supporting_titles() -> None:
    normalizer = HotpotQANormalizer(
        DatasetConfig(name="hotpot_qa", source=DatasetSource.HUGGINGFACE, split="validation")
    )

    sample = normalizer.normalize_record(
        {
            "_id": "hp-1",
            "question": "Which city hosted both events?",
            "answer": "Paris",
            "supporting_facts": [["Paris", 0], ["France", 1]],
            "context": [],
            "level": "medium",
            "type": "bridge",
        },
        index=0,
        split="validation",
    )

    assert sample.question_type == QuestionType.MULTI_HOP
    assert sample.metadata["supporting_fact_titles"] == ["Paris", "France"]
