from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.rules.common import (
    answer_text,
    build_base_score_signals,
    build_context_text,
    require_answerable,
    require_metadata_key,
    require_minimum_evidence_chunks,
)
from rag_evaluator.question_types.signals import (
    token_overlap_ratio,
)
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk


@dataclass(frozen=True)
class AdversarialRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use adversarial for answerable questions that are intentionally tricky, misleading, or distraction-heavy.",  # noqa: E501
            "An adversarial question must still have a clear evidence-grounded answer.",
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        require_answerable(sample)
        require_minimum_evidence_chunks(sample, 1)
        require_metadata_key(sample, "adversarial_pattern")

    def score_answer(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
        *,
        context_chunks: list[Chunk],
    ) -> TypeScoreSignals:
        signals = build_base_score_signals(
            sample,
            generated_answer,
            context_chunks=context_chunks,
        )
        answer = answer_text(generated_answer)
        context_text = build_context_text(context_chunks)
        misleading_overlap = token_overlap_ratio(context_text, answer) if answer and context_text else 0.0  # noqa: E501

        return TypeScoreSignals(
            grounded_in_context=signals.grounded_in_context,
            grounded_in_reference=signals.grounded_in_reference,
            used_multiple_evidence_chunks=signals.used_multiple_evidence_chunks,
            abstained_correctly=signals.abstained_correctly,
            difficulty_mismatch=misleading_overlap > 0.7 and not signals.grounded_in_reference,
            metadata={
                **signals.metadata,
                "adversarial_pattern": sample.metadata.get("adversarial_pattern"),
            },
        )

    def classify_failures(
        self,
        sample: EvalSample,
        retrieved_chunk: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        type_signals: TypeScoreSignals | None = None,
    ) -> list[FailureMode]:
        if type_signals is None or not type_signals.difficulty_mismatch:
            return []
        return [FailureMode.ADVERSARIAL_MISREAD]
