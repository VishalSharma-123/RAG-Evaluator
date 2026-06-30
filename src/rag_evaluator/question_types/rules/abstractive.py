from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.rules.common import (
    build_base_score_signals,
    build_context_text,
    evidence_chunks_for_sample,
    is_grounded_in_text,
    require_answerable,
    require_minimum_evidence_chunks,
)
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.synthetic.errors import SyntheticValidationError


@dataclass(frozen=True)
class AbstractiveRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use abstractive for grounded synthesis or summary answers written in natural paraphrased form.",  # noqa: E501
            "Abstractive answers may combine facts from the evidence but should remain faithful to the chunks.",  # noqa: E501
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        require_answerable(sample)
        require_minimum_evidence_chunks(sample, 1)

        evidence_chunks = evidence_chunks_for_sample(sample, available_chunks_by_id)
        evidence_text = build_context_text(evidence_chunks)
        if evidence_text and sample.reference_answer:
            if not is_grounded_in_text(
                sample.reference_answer,
                evidence_text,
                min_overlap_ratio=0.4,
            ):
                raise SyntheticValidationError(
                    f"Sample {sample.sample_id} with question_type 'abstractive' must remain "
                    "grounded in the cited evidence."
                )

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
            context_overlap_ratio=0.4,
            reference_overlap_ratio=0.3,
            chunk_overlap_ratio=0.4,
        )
        return TypeScoreSignals(
            grounded_in_context=signals.grounded_in_context,
            grounded_in_reference=signals.grounded_in_reference,
            used_multiple_evidence_chunks=signals.used_multiple_evidence_chunks,
            abstained_correctly=signals.abstained_correctly,
            metadata={"abstractive": True},
        )

    def classify_failures(
        self,
        sample: EvalSample,
        retrieved_chunk: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        type_signals: TypeScoreSignals | None = None,
    ) -> list[FailureMode]:
        if type_signals is None or type_signals.grounded_in_context:
            return []
        return [FailureMode.ABSTRACTIVE_UNGROUNDED]
