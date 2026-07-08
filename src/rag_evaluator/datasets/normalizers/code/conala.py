from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class ConalaNormalizer(DatasetNormalizer):
    """
    Normalizer for CoNaLa records.
    """
    dataset_key = "conala"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "question_id", "id")
        question = record.get("intent") or record.get("rewritten_intent")
        answer = record.get("snippet")

        sample_id = (
            self.sample_id(split, index)
            if source_id is None
            else f"{self.config.name}:{split}:{source_id}:{index}"
        )

        return EvalSample(
            sample_id=sample_id,
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="code",
                rewritten_intent=record.get("rewritten_intent"),
            ),
        )
