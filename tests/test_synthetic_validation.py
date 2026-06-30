from __future__ import annotations

import pytest

from rag_evaluator.schemas import QuestionType
from rag_evaluator.synthetic.errors import SyntheticValidationError
from rag_evaluator.synthetic.validation import (
    validate_synthetic_sample,
    validate_synthetic_samples_batch,
)


def test_validate_synthetic_sample_accepts_grounded_answer(make_sample, make_chunk) -> None:
    sample = make_sample(reference_answer="Paris", evidence_chunk_ids=["doc:chunk:0"])
    chunk = make_chunk(text="Paris is the capital of France.")

    validate_synthetic_sample(
        sample,
        available_chunk_ids={chunk.chunk_id},
        available_chunks_by_id={chunk.chunk_id: chunk},
        allowed_question_types={QuestionType.FACTOID},
    )


def test_validate_synthetic_sample_rejects_ungrounded_answer(make_sample, make_chunk) -> None:
    sample = make_sample(reference_answer="Berlin", evidence_chunk_ids=["doc:chunk:0"])
    chunk = make_chunk(text="Paris is the capital of France.")

    with pytest.raises(SyntheticValidationError, match="not grounded"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={chunk.chunk_id},
            available_chunks_by_id={chunk.chunk_id: chunk},
            allowed_question_types={QuestionType.FACTOID},
        )


def test_validate_synthetic_sample_rejects_unknown_chunk_id(make_sample) -> None:
    sample = make_sample(evidence_chunk_ids=["missing"])

    with pytest.raises(SyntheticValidationError, match="unknown evidence_chunk_id"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={"doc:chunk:0"},
            allowed_question_types={QuestionType.FACTOID},
        )


def test_validate_synthetic_sample_rejects_invalid_unanswerable_shape(make_sample) -> None:
    sample = make_sample(
        question_type=QuestionType.UNANSWERABLE,
        is_answerable=False,
        reference_answer=None,
        evidence_chunk_ids=["doc:chunk:0"],
    )

    with pytest.raises(SyntheticValidationError, match="empty evidence_chunk_ids"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={"doc:chunk:0"},
            allowed_question_types={QuestionType.UNANSWERABLE},
        )


def test_validate_synthetic_samples_batch_returns_samples(make_sample, make_chunk) -> None:
    sample = make_sample(reference_answer="Paris", evidence_chunk_ids=["doc:chunk:0"])
    chunk = make_chunk(text="Paris is the capital of France.")

    validated = validate_synthetic_samples_batch(
        [sample],
        available_chunk_ids={chunk.chunk_id},
        available_chunks_by_id={chunk.chunk_id: chunk},
        allowed_question_types={QuestionType.FACTOID},
    )

    assert validated == [sample]


def test_validate_synthetic_sample_rejects_multihop_with_single_chunk(make_sample, make_chunk) -> None:
    sample = make_sample(
        question_type=QuestionType.MULTI_HOP,
        reference_answer="Paris",
        evidence_chunk_ids=["doc:chunk:0"],
    )
    chunk = make_chunk(text="Paris is the capital of France.")

    with pytest.raises(SyntheticValidationError, match="at least 2 evidence_chunk_ids"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={chunk.chunk_id},
            available_chunks_by_id={chunk.chunk_id: chunk},
            allowed_question_types={QuestionType.MULTI_HOP},
        )


def test_validate_synthetic_sample_rejects_comparative_without_targets(make_sample, make_chunk) -> None:
    sample = make_sample(
        question="Which city is larger, Paris or Berlin?",
        question_type=QuestionType.COMPARATIVE,
        reference_answer="Paris is larger.",
        evidence_chunk_ids=["doc:chunk:0"],
        metadata={},
    )
    chunk = make_chunk(text="Paris is larger than Berlin.")

    with pytest.raises(SyntheticValidationError, match="comparison_targets"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={chunk.chunk_id},
            available_chunks_by_id={chunk.chunk_id: chunk},
            allowed_question_types={QuestionType.COMPARATIVE},
        )


def test_validate_synthetic_sample_rejects_adversarial_without_pattern(make_sample, make_chunk) -> None:
    sample = make_sample(
        question_type=QuestionType.ADVERSARIAL,
        reference_answer="Paris",
        evidence_chunk_ids=["doc:chunk:0"],
        metadata={},
    )
    chunk = make_chunk(text="Paris is the capital of France.")

    with pytest.raises(SyntheticValidationError, match="adversarial_pattern"):
        validate_synthetic_sample(
            sample,
            available_chunk_ids={chunk.chunk_id},
            available_chunks_by_id={chunk.chunk_id: chunk},
            allowed_question_types={QuestionType.ADVERSARIAL},
        )
