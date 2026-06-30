from __future__ import annotations

from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.question_types.signals import build_context_text, is_grounded_in_text
from rag_evaluator.schemas import Chunk, EvalSample, QuestionType
from rag_evaluator.synthetic.errors import SyntheticValidationError


def validate_synthetic_sample(
        sample: EvalSample,
        *,
        available_chunk_ids: set[str],
        available_chunks_by_id: dict[str, Chunk] | None = None,
        allowed_question_types: set[QuestionType] | None = None
) -> None:
    """
    Validate one synthetic EvalSample against framework rules.
    :param available_chunks_by_id:
    :param allowed_question_types:
    :param sample:
    :param available_chunk_ids:
    :return:
    """
    if allowed_question_types is not None and sample.question_type not in allowed_question_types:
        allowed_values = ", ".join(
            sorted(question_type.value for question_type in allowed_question_types)
        )
        raise SyntheticValidationError(
            f"Sample {sample.sample_id} returned unsupported question_type "
            f"{sample.question_type.value!r}. Allowed types: {allowed_values}."
        )
    
    _validate_answerability_shape(sample)
    _validate_known_evidence_chunk_ids(sample, available_chunk_ids)
    
    if sample.is_answerable and available_chunks_by_id is not None:
        evidence_text = build_context_text(
            _resolve_evidence_chunks(
                sample.evidence_chunk_ids,
                available_chunks_by_id = available_chunks_by_id,
            )
        )
        
        if not is_grounded_in_text(sample.reference_answer or "", evidence_text):
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} reference_answer is not grounded in the cited evidence."
            )
    
    rule = get_question_type_rule(sample.question_type)
    rule.validate_method(
        sample,
        available_chunks_by_id=available_chunks_by_id
    )

def validate_synthetic_samples_batch(
        samples: list[EvalSample],
        *,
        available_chunk_ids: set[str],
        available_chunks_by_id: dict[str, Chunk] | None = None,
        allowed_question_types: set[QuestionType] | None = None
) -> list[EvalSample]:
    """
    Validate a batch of synthetic EvalSample records.
    :param available_chunks_by_id:
    :param samples:
    :param available_chunk_ids:
    :param allowed_question_types:
    :return:
    """
    for sample in samples:
        validate_synthetic_sample(
            sample,
            available_chunk_ids=available_chunk_ids,
            available_chunks_by_id=available_chunks_by_id,
            allowed_question_types=allowed_question_types
        )
    
    return samples

def _validate_answerability_shape(sample: EvalSample) -> None:
    if sample.is_answerable:
        if not sample.reference_answer or not sample.reference_answer.strip():
            raise SyntheticValidationError(
                "Answerable synthetic samples must include a non-empty reference answer."
            )

        if not sample.evidence_chunk_ids:
            raise SyntheticValidationError(
                "Answerable synthetic samples must include at least one evidence_chunk_id."
            )

        if sample.question_type == QuestionType.UNANSWERABLE:
            raise SyntheticValidationError(
                "Answerable synthetic samples cannot use question_type `unanswerable`."
            )
        return

    if sample.question_type != QuestionType.UNANSWERABLE:
        raise SyntheticValidationError(
            "Non-answerable synthetic samples must use question_type `unanswerable`."
        )

    if sample.reference_answer is not None:
        raise SyntheticValidationError(
            "Non-answerable synthetic samples must set reference_answer to null."
        )

    if sample.evidence_chunk_ids:
        raise SyntheticValidationError(
            "Non-answerable synthetic samples must use an empty evidence_chunk_ids list."
        )

def _validate_known_evidence_chunk_ids(
    sample: EvalSample,
    available_chunk_ids: set[str],
) -> None:
    for chunk_id in sample.evidence_chunk_ids:
        if chunk_id not in available_chunk_ids:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} references unknown evidence_chunk_id {chunk_id!r}."
            )

def _resolve_evidence_chunks(
    evidence_chunk_ids: list[str],
    *,
    available_chunks_by_id: dict[str, Chunk],
) -> list[Chunk]:
    return [
        available_chunks_by_id[chunk_id]
        for chunk_id in evidence_chunk_ids
        if chunk_id in available_chunks_by_id
    ]
