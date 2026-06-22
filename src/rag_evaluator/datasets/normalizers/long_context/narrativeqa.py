from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class NarrativeQANormalizer(DatasetNormalizer):
    """
    Normalizer for NarrativeQA long-context QA records.
    """

    dataset_key = "narrativeqa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "document_id")
        question = self._text(record.get("question"))
        answers = record.get("answers") or []
        answer_texts = [self._text(answer) for answer in answers]
        answer_texts = [answer for answer in answer_texts if answer]
        answer = answer_texts[0] if answer_texts else None

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=answer,
            answer_aliases=answer_texts[1:],
            question_type=QuestionType.ABSTRACTIVE,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                document=record.get("document"),
                summary=record.get("summary"),
                raw_answers=answers,
            ),
        )

    def _text(self, value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            for key in ("text", "answer", "question"):
                if value.get(key):
                    return str(value[key])

        return str(value)
