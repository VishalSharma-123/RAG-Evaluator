from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class QualityNormalizer(DatasetNormalizer):
    """
    Normalizer for QuALITY long-context multiple-choice QA records.
    """

    dataset_key = "quality"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "question_id", "article_id")
        parsed = self._parse_quality_prompt(record)
        choices = parsed["choices"]
        answer_index = record.get("gold_label")
        answer = self._answer_from_index(answer_index, choices)
        if answer is None and parsed["answer_text"] is not None:
            answer = parsed["answer_text"]

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=parsed["question"],
            reference_answer=answer,
            choices=choices,
            question_type=QuestionType.ABSTRACTIVE,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                article=record.get("article") or parsed["article"],
                article_id=record.get("article_id"),
                gold_label=answer_index,
                pid=record.get("pid"),
            ),
        )

    def _answer_from_index(self, answer_index: Any, choices: list[str]) -> str | None:
        if isinstance(answer_index, int) and 0 <= answer_index < len(choices):
            return choices[answer_index]

        if isinstance(answer_index, str) and answer_index.isdigit():
            index = int(answer_index)
            if 0 <= index < len(choices):
                return choices[index]

        return None

    def _parse_quality_prompt(self, record: dict[str, Any]) -> dict[str, Any]:
        question = record.get("question")
        choices = [str(option) for option in (record.get("options") or [])]
        article = record.get("article")
        answer_text = record.get("output")

        if question is not None:
            return {
                "question": str(question),
                "choices": choices,
                "article": article,
                "answer_text": str(answer_text) if answer_text is not None else None,
            }

        prompt = str(record.get("input") or "")
        sections = prompt.split("\n\n", 1)
        question_block = sections[0] if sections else prompt
        article_block = sections[1] if len(sections) > 1 else None

        lines = [line.rstrip() for line in question_block.splitlines() if line.strip()]
        parsed_question = lines[0] if lines else prompt.strip()
        parsed_choices = [line[4:].strip() for line in lines[1:] if line.startswith("(") and len(line) > 4]

        return {
            "question": parsed_question,
            "choices": parsed_choices,
            "article": article_block,
            "answer_text": str(answer_text) if answer_text is not None else None,
        }
