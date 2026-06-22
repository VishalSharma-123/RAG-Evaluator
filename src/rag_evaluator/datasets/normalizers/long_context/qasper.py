from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class QasperNormalizer(DatasetNormalizer):
    """
    Normalizer for QASPER long-document scientific QA records.
    """

    dataset_key = "qasper"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "paper_id")
        question, question_id, answers = self._qa_fields(record)
        answer = self._first_answer(answers)

        return EvalSample(
            sample_id=self.sample_id(split, index, question_id or source_id),
            question=str(question),
            reference_answer=answer,
            question_type=QuestionType.ABSTRACTIVE,
            source_dataset=self.config.name,
            source_split=split,
            source_id=question_id or source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                title=record.get("title"),
                abstract=record.get("abstract"),
                full_text=record.get("full_text"),
                answers=answers,
                qas=record.get("qas"),
            ),
        )

    def _qa_fields(self, record: dict[str, Any]) -> tuple[str | None, str | None, list[Any]]:
        if "question" in record:
            return record.get("question"), self.source_id(record, "question_id", "id"), record.get("answers") or []

        qas = record.get("qas")
        if not isinstance(qas, dict):
            return None, None, []

        questions = qas.get("question") or []
        question_ids = qas.get("question_id") or []
        answers = qas.get("answers") or []

        question = questions[0] if questions else None
        question_id = str(question_ids[0]) if question_ids else None
        first_answers = answers[0].get("answer") if answers and isinstance(answers[0], dict) else []

        return question, question_id, first_answers

    def _first_answer(self, answers: list[Any]) -> str | None:
        for answer in answers:
            if not isinstance(answer, dict):
                continue

            free_form = answer.get("free_form_answer")
            if free_form:
                return str(free_form)

            extractive_spans = answer.get("extractive_spans") or []
            if extractive_spans:
                return str(extractive_spans[0])

            yes_no = answer.get("yes_no")
            if yes_no is not None:
                return str(yes_no)

        return None
