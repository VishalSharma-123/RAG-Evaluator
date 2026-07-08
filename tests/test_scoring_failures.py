from __future__ import annotations

from rag_evaluator.schemas import EvidenceSpan, FailureMode, GeneratedAnswer, QuestionType
from rag_evaluator.scoring.failures import classify_failures


def test_classify_failures_adds_context_ignored_hallucination_and_partial_answer(
    make_sample,
    make_retrieved_chunk,
) -> None:
    sample = make_sample(
        question_type=QuestionType.FACTOID,
        reference_answer="Paris",
        evidence_chunk_ids=["doc:chunk:0"],
    )
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Completely unrelated answer",
        model_name="unit-test",
    )

    failures = classify_failures(
        sample,
        [make_retrieved_chunk()],
        generated_answer=generated_answer,
        context_was_used=False,
        hallucination_score=0.9,
        partial_answer_score=0.8,
    )

    assert FailureMode.CONTEXT_IGNORED in failures
    assert FailureMode.HALLUCINATION in failures
    assert FailureMode.PARTIAL_ANSWER in failures


def test_classify_failures_deduplicates_overlapping_modes(make_sample, make_retrieved_chunk) -> None:
    sample = make_sample(
        question_type=QuestionType.FACTOID,
        evidence_chunk_ids=["doc:chunk:0"],
    )
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    failures = classify_failures(
        sample,
        [make_retrieved_chunk(chunk=make_retrieved_chunk().chunk.model_copy(update={"chunk_id": "other:chunk:1"}))],
        generated_answer=generated_answer,
        retrieval_k=1,
    )

    assert failures == [FailureMode.RETRIEVAL_MISS]


def test_classify_failures_uses_resolved_span_gold_for_rank(
    make_sample,
    make_retrieved_chunk,
    make_chunk,
) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[EvidenceSpan(document_id="doc", start_char=40, end_char=50)],
    )
    retrieved = [
        make_retrieved_chunk(
            chunk=make_chunk(chunk_id="doc:chunk:0", start_char=0, end_char=10),
            rank=1,
        ),
        make_retrieved_chunk(
            chunk=make_chunk(chunk_id="doc:chunk:1", start_char=45, end_char=60),
            rank=2,
        ),
    ]

    failures = classify_failures(sample, retrieved, retrieval_k=1)

    assert failures == [FailureMode.RETRIEVAL_RANK]


def test_classify_failures_skips_retrieval_modes_when_gold_unavailable(
    make_sample,
    make_retrieved_chunk,
) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[],
        reference_answer=None,
    )

    failures = classify_failures(sample, [make_retrieved_chunk()], retrieval_k=1)

    assert FailureMode.RETRIEVAL_MISS not in failures
    assert FailureMode.RETRIEVAL_RANK not in failures
