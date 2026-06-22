from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class PubMedQANormalizer(DatasetNormalizer):
    """
    Normalizer for PubMedQA biomedical QA records.
    """

    dataset_key = "pubmedqa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "pubid", "id")
        answer = record.get("final_decision") or record.get("long_answer")

        answer_aliases: list[str] = []
        if record.get("long_answer") and record.get("long_answer") != answer:
            answer_aliases.append(str(record["long_answer"]))

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=str(answer) if answer is not None else None,
            answer_aliases=answer_aliases,
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="medical",
                context=record.get("context"),
                final_decision=record.get("final_decision"),
                long_answer=record.get("long_answer"),
            ),
        )

