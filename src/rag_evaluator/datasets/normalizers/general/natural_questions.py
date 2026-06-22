from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class NaturalQuestionsNormalizer(DatasetNormalizer):
    """
    Normalizer for Natural Questions factoid QA records.
    """

    dataset_key = "natural_questions"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "example_id")
        question = record.get("question") or record.get("question_text")
        answer = self._first_answer(record)

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
                document=record.get("document"),
                long_answer=record.get("long_answer"),
                annotations=record.get("annotations"),
            ),
        )

    def _first_answer(self, record: dict[str, Any]) -> str | None:
        for key in ("short_answers", "long_answer", "annotations"):
            value = record.get(key)
            answer = self._first_text(value)
            if answer is not None:
                return answer

        return None

    def _first_text(self, value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, str) and value:
            return value

        if isinstance(value, list):
            for item in value:
                answer = self._first_text(item)
                if answer is not None:
                    return answer

        if isinstance(value, dict):
            for nested in value.values():
                answer = self._first_text(nested)
                if answer is not None:
                    return answer

        return None
