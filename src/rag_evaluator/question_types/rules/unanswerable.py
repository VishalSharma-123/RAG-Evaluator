from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.signals import answer_text, looks_like_abstention
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.synthetic.errors import SyntheticValidationError


@dataclass(frozen=True)
class UnanswerableRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use unanswerable only when the provided chunks do not contain enough information to answer the question.",  # noqa: E501
            "Unanswerable samples must require abstention instead of guessing.",
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        if sample.is_answerable:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} with question_type 'unanswerable' must set "
                "is_answerable to false."
            )

        if sample.reference_answer is not None:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} with question_type 'unanswerable' must set "
                "reference_answer to null."
            )

        if sample.evidence_chunk_ids:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} with question_type 'unanswerable' must use "
                "an empty evidence_chunk_ids list."
            )

    def score_answer(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
        *,
        context_chunks: list[Chunk],
    ) -> TypeScoreSignals:
        answer = answer_text(generated_answer)
        return TypeScoreSignals(
            abstained_correctly=looks_like_abstention(answer) if answer else False,
            metadata={"expected_behavior": "abstain"},
        )

    def classify_failures(
        self,
        sample: EvalSample,
        retrieved_chunk: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        type_signals: TypeScoreSignals | None = None,
    ) -> list[FailureMode]:
        if type_signals is not None and type_signals.abstained_correctly:
            return []
        return [FailureMode.UNANSWERABLE_FAIL]
