from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class RagasNormalizer(DatasetNormalizer):
    """
    Normalizer for RAGAS-generated synthetic testset records.
    """

    dataset_key = "ragas"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "sample_id")
        question = self._question(record)
        answer = self._reference_answer(record)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=question,
            reference_answer=str(answer) if answer is not None else None,
            answer_aliases=self._aliases(record),
            question_type=self._question_type(record),
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            evidence_chunk_ids=self._evidence_ids(record),
            is_answerable=self._is_answerable(record, answer),
            metadata=self.metadata(
                domain="synthetic",
                retrieved_contexts=self._string_list(record.get("retrieved_contexts")),
                reference_contexts=self._string_list(record.get("reference_contexts")),
                retrieved_context_ids=self._string_list(record.get("retrieved_context_ids")),
                reference_context_ids=self._string_list(record.get("reference_context_ids")),
                contexts=self._string_list(record.get("contexts")),
                ground_truth=record.get("ground_truth"),
                response=record.get("response"),
                multi_responses=self._string_list(record.get("multi_responses")),
                rubrics=record.get("rubrics"),
                persona_name=record.get("persona_name"),
                query_style=record.get("query_style"),
                query_length=record.get("query_length"),
                synthesizer_name=record.get("synthesizer_name"),
                evolution_type=record.get("evolution_type"),
                question_type_raw=record.get("question_type") or record.get("type"),
            ),
        )

    def _question(self, record: dict[str, Any]) -> str:
        value = record.get("question")
        if value is None:
            value = record.get("user_input")

        if isinstance(value, str) and value.strip():
            return value

        raise ValueError("RAGAS record is missing a valid question/user_input string.")

    def _reference_answer(self, record: dict[str, Any]) -> str | None:
        for key in ("reference", "reference_answer", "answer"):
            value = record.get(key)
            if value is None:
                continue

            if isinstance(value, str):
                return value

            return str(value)

        return None

    def _aliases(self, record: dict[str, Any]) -> list[str]:
        return self._string_list(record.get("answer_aliases"))

    def _evidence_ids(self, record: dict[str, Any]) -> list[str]:
        for key in (
            "reference_context_ids",
            "retrieved_context_ids",
            "evidence_chunk_ids",
            "context_ids",
        ):
            ids = self._string_list(record.get(key))
            if ids:
                return ids

        return []

    def _is_answerable(self, record: dict[str, Any], answer: Any) -> bool:
        if "is_answerable" in record:
            return bool(record["is_answerable"])

        return answer is not None

    def _question_type(self, record: dict[str, Any]) -> QuestionType:
        raw_type = " ".join(
            str(value).lower()
            for value in (
                record.get("question_type"),
                record.get("evolution_type"),
                record.get("type"),
                record.get("synthesizer_name"),
                record.get("query_style"),
            )
            if value
        )

        if "multi_hop" in raw_type or "multihop" in raw_type or "multi-hop" in raw_type:
            return QuestionType.MULTI_HOP
        if "abstract" in raw_type:
            return QuestionType.ABSTRACTIVE
        if "comparative" in raw_type:
            return QuestionType.COMPARATIVE
        if "adversarial" in raw_type:
            return QuestionType.ADVERSARIAL
        if "unanswerable" in raw_type:
            return QuestionType.UNANSWERABLE

        return QuestionType.FACTOID

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item) for item in value if item is not None]

        return [str(value)]
