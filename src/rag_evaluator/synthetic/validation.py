from __future__ import annotations

import re

from rag_evaluator.schemas import Chunk
from rag_evaluator.schemas import EvalSample, QuestionType
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
    else:
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
    
    for chunk_id in sample.evidence_chunk_ids:
        if chunk_id not in available_chunk_ids:
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} references unknown evidence_chunk_id "
                f"{chunk_id!r}."
            )

    if sample.is_answerable and available_chunks_by_id is not None:
        evidence_text = _build_evidence_text(
            sample.evidence_chunk_ids,
            available_chunks_by_id=available_chunks_by_id,
        )
        if not _is_answer_grounded(
            sample.reference_answer or "",
            evidence_text,
        ):
            raise SyntheticValidationError(
                f"Sample {sample.sample_id} reference_answer is not grounded in the cited evidence."
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


def _build_evidence_text(
        evidence_chunk_ids: list[str],
        *,
        available_chunks_by_id: dict[str, Chunk],
) -> str:
    texts: list[str] = []

    for chunk_id in evidence_chunk_ids:
        chunk = available_chunks_by_id.get(chunk_id)
        if chunk is not None and chunk.text.strip():
            texts.append(chunk.text)

    return " ".join(texts)


def _is_answer_grounded(reference_answer: str, evidence_text: str) -> bool:
    normalized_answer = _normalize_text(reference_answer)
    normalized_evidence = _normalize_text(evidence_text)

    if not normalized_answer or not normalized_evidence:
        return False

    if normalized_answer in normalized_evidence:
        return True

    answer_tokens = set(normalized_answer.split())
    evidence_tokens = set(normalized_evidence.split())
    if not answer_tokens or not evidence_tokens:
        return False

    overlap = len(answer_tokens & evidence_tokens)
    overlap_ratio = overlap / len(answer_tokens)
    return overlap_ratio >= 0.6


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
