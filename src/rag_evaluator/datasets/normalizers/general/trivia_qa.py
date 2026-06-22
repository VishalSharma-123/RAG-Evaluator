from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class TriviaQANormalizer(DatasetNormalizer):
    """
    Normalizer for TriviaQA factoid QA records.
    """

    dataset_key = "trivia_qa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "question_id", "id")
        answer_obj = record.get("answer") or {}
        aliases = answer_obj.get("aliases") or []
        answer = answer_obj.get("value") or (aliases[0] if aliases else None)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=str(answer) if answer is not None else None,
            answer_aliases=[str(alias) for alias in aliases],
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                entity_pages=record.get("entity_pages"),
                search_results=record.get("search_results"),
            ),
        )
