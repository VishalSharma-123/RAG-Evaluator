from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.rules.common import (
    build_base_score_signals,
    count_grounding_chunks,
    evidence_chunks_for_sample,
    require_answerable,
    require_minimum_evidence_chunks,
)
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.synthetic.errors import SyntheticValidationError


@dataclass(frozen=True)
class MultiHopRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use multi_hop only when answering requires combining information from at least two chunks.",  # noqa: E501
            "Do not label a question multi_hop if one chunk alone is sufficient.",
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        require_answerable(sample)
        require_minimum_evidence_chunks(sample, 2)

        evidence_chunks = evidence_chunks_for_sample(sample, available_chunks_by_id)
        if evidence_chunks and sample.reference_answer:
            grounding_count = count_grounding_chunks(sample.reference_answer, evidence_chunks)
            if grounding_count < 2:
                raise SyntheticValidationError(
                    f"Sample {sample.sample_id} with question_type 'multi_hop' must require "
                    "multiple evidence chunks to support the reference_answer."
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
        )
        return TypeScoreSignals(
            grounded_in_context=signals.grounded_in_context,
            grounded_in_reference=signals.grounded_in_reference,
            used_multiple_evidence_chunks=signals.used_multiple_evidence_chunks,
            abstained_correctly=signals.abstained_correctly,
            metadata={
                **signals.metadata,
                "requires_multiple_evidence_chunks": True,
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
        if type_signals is None or type_signals.used_multiple_evidence_chunks:
            return []
        return [FailureMode.MULTI_HOP_INCOMPLETE]
