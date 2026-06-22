from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, EvidenceSpan, QuestionType


class CuadNormalizer(DatasetNormalizer):
    """
    Normalizer for CUAD legal contract QA records.
    """

    dataset_key = "cuad"

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id")
        answers = record.get("answers") or {}
        answer_texts = answers.get("text") or []
        is_answerable = len(answer_texts) > 0

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(record.get("question")),
            reference_answer=answer_texts[0] if is_answerable else None,
            answer_aliases=answer_texts[1:],
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            evidence_spans=self._answer_spans(record),
            is_answerable=is_answerable,
            metadata=self.metadata(
                domain="legal",
                title=record.get("title"),
                context=record.get("context"),
            ),
        )
    
    def _answer_spans(self, record: dict[str, Any]) -> list[EvidenceSpan]:
        answers = record.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        source_id = self.source_id(record, "id") or "cuad_context"
        
        spans: list[EvidenceSpan] = []
        
        for index, text in enumerate(texts):
            if index >= len(starts):
                continue
            
            start = starts[index]
            if not isinstance(start, int):
                continue
            
            spans.append(
                EvidenceSpan(
                    document_id=source_id,
                    start_char=start,
                    end_char=start + len(text),
                    text=text,
                )
            )
        
        return spans
        
    