from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class MedQANormalizer(DatasetNormalizer):
    """
    Normalizer for MedQA / USMLE medical QA records.
    """

    dataset_key = "med_qa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "qid", "question_id")
        question = record.get("question") or self._question_from_alt_fields(record)
        choices = self._choices(record.get("options")) or self._choices_from_alt_fields(record)
        answer = record.get("answer")

        if answer is None:
            answer_idx = record.get("answer_idx", record.get("label"))
            answer = self._answer_from_idx(answer_idx, choices)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            choices=choices,
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="medical",
                answer_idx=record.get("answer_idx"),
                meta_info=record.get("meta_info"),
            ),
        )

    def _choices(self, options: Any) -> list[str]:
        if options is None:
            return []

        if isinstance(options, list):
            return [str(option) for option in options]

        if isinstance(options, dict):
            return [str(value) for _, value in sorted(options.items())]

        return []

    def _choices_from_alt_fields(self, record: dict[str, Any]) -> list[str]:
        endings = [
            record.get("ending0"),
            record.get("ending1"),
            record.get("ending2"),
            record.get("ending3"),
        ]
        return [str(choice) for choice in endings if choice is not None]

    def _question_from_alt_fields(self, record: dict[str, Any]) -> str | None:
        stem = record.get("sent1")
        extra = record.get("sent2")

        if stem is None:
            return None

        if extra:
            return f"{stem} {extra}".strip()

        return str(stem)

    def _answer_from_idx(self, answer_idx: Any, choices: list[str]) -> str | None:
        if answer_idx is None:
            return None

        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
            return choices[answer_idx]

        if isinstance(answer_idx, str):
            normalized = answer_idx.strip()

            if normalized.isdigit():
                index = int(normalized)
                if 0 <= index < len(choices):
                    return choices[index]

            letter_index = ord(normalized.upper()) - ord("A")
            if 0 <= letter_index < len(choices):
                return choices[letter_index]

        return None
