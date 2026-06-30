from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.question_types.base import QuestionTypeRule, TypeScoreSignals
from rag_evaluator.question_types.rules.common import (
    answer_text,
    build_base_score_signals,
    require_answerable,
    require_metadata_key,
    require_minimum_evidence_chunks,
)
from rag_evaluator.question_types.signals import (
    has_comparison_language,
    tokenize,
)
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.synthetic.errors import SyntheticValidationError


@dataclass(frozen=True)
class ComparativeRule(QuestionTypeRule):
    def prompt_instructions(self) -> list[str]:
        return [
            "Use comparative only when the question compares at least two entities, facts, or conditions from the evidence.",  # noqa: E501
            "Comparative answers should cover both sides of the comparison, not just one.",
        ]

    def validate_method(
        self,
        sample: EvalSample,
        *,
        available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        require_answerable(sample)
        require_minimum_evidence_chunks(sample, 1)
        require_metadata_key(sample, "comparison_targets")

        if not has_comparison_language(sample.question):
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} with question_type 'comparative' must use "
                "comparison-oriented language in the question."
            )

        targets = sample.metadata.get("comparison_targets")
        if not isinstance(targets, list) or len(targets) < 2:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} with question_type 'comparative' must include "
                "at least two comparison_targets."
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
        answer = answer_text(generated_answer)
        targets = sample.metadata.get("comparison_targets", [])
        answer_tokens = tokenize(answer)

        covered_targets = 0
        for target in targets:
            if isinstance(target, str) and tokenize(target) & answer_tokens:
                covered_targets += 1

        return TypeScoreSignals(
            grounded_in_context=signals.grounded_in_context,
            grounded_in_reference=signals.grounded_in_reference,
            used_multiple_evidence_chunks=signals.used_multiple_evidence_chunks,
            performed_comparison=has_comparison_language(answer) if answer else False,
            covered_key_entities=covered_targets >= 2,
            abstained_correctly=signals.abstained_correctly,
            metadata={
                **signals.metadata,
                "covered_comparison_targets": covered_targets,
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
        if type_signals is None:
            return []
        if type_signals.performed_comparison and type_signals.covered_key_entities:
            return []
        return [FailureMode.COMPARISON_INCOMPLETE]
