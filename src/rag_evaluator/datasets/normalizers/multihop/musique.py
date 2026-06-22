from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class MusiqueNormalizer(DatasetNormalizer):
    """
    Normalizer for MuSiQue hard multi-hop QA records.
    """

    dataset_key = "musique"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "paragraph_id")
        aliases = record.get("answer_aliases") or []
        answer = record.get("answer") or (aliases[0] if aliases else None)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=str(answer) if answer is not None else None,
            answer_aliases=[str(alias) for alias in aliases],
            question_type=QuestionType.MULTI_HOP,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                paragraphs=record.get("paragraphs"),
                question_decomposition=record.get("question_decomposition"),
            ),
        )
