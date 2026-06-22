from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class BeirNormalizer(DatasetNormalizer):
    """
    Normalizer for BEIR-style retrieval benchmark query records.
    """

    dataset_key = "beir"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "query_id", "qid", "id", "_id")
        question = record.get("query") or record.get("question") or record.get("text")
        answer = record.get("answer")
        evidence_ids = self._evidence_ids(record)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            question_type=self.config.question_type or QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            evidence_chunk_ids=evidence_ids,
            is_answerable=True,
            metadata=self.metadata(
                corpus_id=record.get("corpus_id"),
                doc_id=record.get("doc_id"),
                relevant_docs=record.get("relevant_docs"),
                raw=record,
            ),
        )

    def _evidence_ids(self, record: dict[str, Any]) -> list[str]:
        for key in ("evidence_chunk_ids", "relevant_docs", "positive_doc_ids", "doc_id", "corpus_id"):
            value = record.get(key)
            if value is None:
                continue

            if isinstance(value, list):
                return [str(item) for item in value]

            if isinstance(value, dict):
                return [str(item) for item in value.keys()]

            return [str(value)]

        return []
