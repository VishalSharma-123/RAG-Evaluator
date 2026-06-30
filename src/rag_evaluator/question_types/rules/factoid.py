from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.rules.common import (
    build_base_score_signals,
    require_answerable,
    require_minimum_evidence_chunks,
)
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk


@dataclass(frozen=True)
class FactoidRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use factoid for direct, specific, evidence-grounded questions with concise answers.",
            "A factoid question should usually be answerable from one chunk or one local span.",
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        require_answerable(sample)
        require_minimum_evidence_chunks(sample, 1)

    def score_answer(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
        *,
        context_chunks: list[Chunk],
    ) -> TypeScoreSignals:
        return build_base_score_signals(
            sample,
            generated_answer,
            context_chunks=context_chunks,
        )

    def classify_failures(
        self,
        sample: EvalSample,
        retrieved_chunk: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        type_signals: TypeScoreSignals | None = None,
    ) -> list[FailureMode]:
        return []
