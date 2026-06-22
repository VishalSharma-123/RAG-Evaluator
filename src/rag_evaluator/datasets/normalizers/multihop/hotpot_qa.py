from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class HotpotQANormalizer(DatasetNormalizer):
    """Normalizer for HotpotQA multi-hop QA records."""

    dataset_key = "hotpot_qa"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "_id")
        supporting_facts = record.get("supporting_facts") or {}
        titles = self._supporting_titles(supporting_facts)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=str(record.get("answer")) if record.get("answer") is not None else None,
            question_type=QuestionType.MULTI_HOP,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=record.get("answer") is not None,
            metadata=self.metadata(
                supporting_fact_titles=titles,
                supporting_facts=supporting_facts,
                context=record.get("context"),
                level=record.get("level"),
                type=record.get("type"),
            ),
        )

    def _supporting_titles(self, supporting_facts: Any) -> list[str]:
        if isinstance(supporting_facts, dict):
            titles = supporting_facts.get("title") or []
            return [str(title) for title in titles]

        if isinstance(supporting_facts, list):
            titles: list[str] = []
            for fact in supporting_facts:
                if isinstance(fact, (list, tuple)) and fact:
                    titles.append(str(fact[0]))
                elif isinstance(fact, dict) and fact.get("title") is not None:
                    titles.append(str(fact["title"]))
            return titles

        return []
