from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.schemas import EvalSample, QuestionType


class CodeSearchNetNormalizer(DatasetNormalizer):
    """
    Normalizer for CodeSearchNet records
    """
    
    dataset_key = "codesearchnet"
    
    def normalize_record(
            self,
            record: dict[str, Any],
            *,
            index: int,
            split: str,
    ) -> EvalSample:
        source_id = self.source_id(record, "id", "doc_id", "func_name", "url")
        question = (
            record.get("func_documentation_string")
            or record.get("docstring")
            or record.get("query")
            or record.get("func_name")
        )
        answer = record.get("func_code_string") or record.get("code") or record.get("code_tokens")
        
        if isinstance(answer, list):
            answer = " ".join(str(token) for token in answer)

        return EvalSample(
            sample_id=self.sample_id(split, index, source_id),
            question=str(question),
            reference_answer=str(answer) if answer is not None else None,
            question_type=QuestionType.FACTOID,
            source_dataset=self.config.name,
            source_split=split,
            source_id=source_id,
            is_answerable=answer is not None,
            metadata=self.metadata(
                domain="code",
                language=record.get("language"),
                repository_name=record.get("repository_name"),
                func_name=record.get("func_name"),
                url=record.get("url"),
            ),
        )