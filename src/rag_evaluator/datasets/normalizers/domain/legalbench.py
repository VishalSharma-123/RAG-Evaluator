from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class LegalBenchNormalizer(DatasetNormalizer):
    """
    Normalizer for LegalBench legal reasoning records.
    """

    dataset_key = "legalbench"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "idx", "example_id")
        question = (
            record.get("question")
            or record.get("prompt")
            or record.get("text")
            or record.get("input")
        )
        answer = (
            record.get("answer")
            or record.get("label")
            or record.get("target")
            or record.get("output")
        )
        choices = self._choices(record)

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
                domain="legal",
                task=record.get("task"),
                subset=record.get("subset"),
                raw_label=record.get("label"),
            ),
        )

    def _choices(self, record: dict[str, Any]) -> list[str]:
        choices = record.get("choices") or record.get("options")

        if choices is None:
            return []

        if isinstance(choices, list):
            return [str(choice) for choice in choices]

        if isinstance(choices, dict):
            return [str(value) for _, value in sorted(choices.items())]

        return []
