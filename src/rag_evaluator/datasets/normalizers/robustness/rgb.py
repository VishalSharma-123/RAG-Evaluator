from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class RGBNormalizer(DatasetNormalizer):
    """
    Normalizer for RGB RAG robustness benchmark records.
    """

    dataset_key = "rgb"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "qid", "question_id")
        question = record.get("question") or record.get("query")
        answer = record.get("answer") or record.get("ground_truth")

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            answer_aliases=self._aliases(record),
            question_type=self._question_type(record),
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            evidence_chunk_ids=self._evidence_ids(record),
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="robustness",
                contexts=record.get("contexts") or record.get("documents"),
                noise=record.get("noise"),
                raw_type=record.get("type"),
            ),
        )

    def _aliases(self, record: dict[str, Any]) -> list[str]:
        aliases = record.get("answer_aliases") or record.get("aliases") or []

        if isinstance(aliases, list):
            return [str(alias) for alias in aliases]

        return []

    def _evidence_ids(self, record: dict[str, Any]) -> list[str]:
        ids = record.get("evidence_chunk_ids") or record.get("evidence_ids") or []

        if isinstance(ids, list):
            return [str(item) for item in ids]

        if ids:
            return [str(ids)]

        return []

    def _question_type(self, record: dict[str, Any]) -> QuestionType:
        raw_type = str(record.get("type") or "").lower()

        if "adversarial" in raw_type or "noise" in raw_type:
            return QuestionType.ADVERSARIAL

        return QuestionType.FACTOID
