from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class TwoWikiMultiHopQANormalizer(DatasetNormalizer):
    """
    Normalizer for 2WikiMultiHopQA records.
    """

    dataset_key = "two_wiki_multihop_qa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "_id")
        answer = record.get("answer")

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=str(answer) if answer is not None else None,
            question_type=QuestionType.MULTI_HOP,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                context=record.get("context"),
                evidence=record.get("evidence") or record.get("evidences"),
                supporting_facts=record.get("supporting_facts"),
                type=record.get("type"),
            ),
        )
