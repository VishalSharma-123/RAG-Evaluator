from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class FinQANormalizer(DatasetNormalizer):
    """
    Normalizer for FinQA financial QA records.
    """

    dataset_key = "finqa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "uid")
        answer = record.get("answer") or record.get("qa", {}).get("exe_ans")
        question = record.get("question") or record.get("qa", {}).get("question")

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="finance",
                table=record.get("table"),
                pre_text=record.get("pre_text"),
                post_text=record.get("post_text"),
                program=record.get("program"),
                qa=record.get("qa"),
            ),
        )
