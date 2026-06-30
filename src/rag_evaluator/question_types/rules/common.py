from __future__ import annotations

from rag_evaluator.question_types.base import TypeScoreSignals
from rag_evaluator.question_types.signals import (
    answer_text,
    build_context_text,
    count_grounding_chunks,
    is_grounded_in_text,
    looks_like_abstention,
)
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer
from rag_evaluator.synthetic.errors import SyntheticValidationError


def require_metadata_key(sample: EvalSample, key: str) -> None:
    value = sample.metadata.get(key)
    if value in (None, "", [], {}):
        raise SyntheticValidationError(
            f"Sample {sample.sample_id} with question_type "
            f"{sample.question_type.value!r} must include metadata[{key!r}]."
        )


def require_answerable(sample: EvalSample) -> None:
    if not sample.is_answerable:
        raise SyntheticValidationError(
            f"Sample {sample.sample_id} with question_type "
            f"{sample.question_type.value!r} must be answerable."
        )


def require_minimum_evidence_chunks(sample: EvalSample, minimum_chunks: int) -> None:
    if len(sample.evidence_chunk_ids) < minimum_chunks:
        raise SyntheticValidationError(
            f"Sample {sample.sample_id} with question_type "
            f"{sample.question_type.value!r} must cite at least {minimum_chunks} "
            f"evidence_chunk_id{'s' if minimum_chunks != 1 else ''}."
        )


def evidence_chunks_for_sample(
    sample: EvalSample,
    available_chunks_by_id: dict[str, Chunk] | None,
) -> list[Chunk]:
    if available_chunks_by_id is None:
        return []

    return [
        available_chunks_by_id[chunk_id]
        for chunk_id in sample.evidence_chunk_ids
        if chunk_id in available_chunks_by_id
    ]


def build_base_score_signals(
    sample: EvalSample,
    generated_answer: GeneratedAnswer | None,
    *,
    context_chunks: list[Chunk],
    context_overlap_ratio: float = 0.6,
    reference_overlap_ratio: float = 0.6,
    chunk_overlap_ratio: float = 0.6,
) -> TypeScoreSignals:
    answer = answer_text(generated_answer)
    context_text = build_context_text(context_chunks)
    reference_text = sample.reference_answer or ""

    return TypeScoreSignals(
        grounded_in_context=(
            is_grounded_in_text(answer, context_text, min_overlap_ratio=context_overlap_ratio)
            if answer
            else False
        ),
        grounded_in_reference=(
            is_grounded_in_text(
                answer,
                reference_text,
                min_overlap_ratio=reference_overlap_ratio,
            )
            if answer and reference_text
            else False
        ),
        used_multiple_evidence_chunks=(
            count_grounding_chunks(
                answer,
                context_chunks,
                min_overlap_ratio=chunk_overlap_ratio,
            ) >= 2
            if answer
            else False
        ),
        abstained_correctly=looks_like_abstention(answer) if answer else False,
        metadata={},
    )
