from __future__ import annotations

from rag_evaluator.schemas import FailureMode, GeneratedAnswer, QuestionType
from rag_evaluator.scoring.failures import classify_failures


def test_classify_failures_adds_multi_hop_incomplete(make_sample, make_retrieved_chunk) -> None:
    sample = make_sample(
        question_type=QuestionType.MULTI_HOP,
        reference_answer="Paris and France",
        evidence_chunk_ids=["doc:chunk:0", "doc:chunk:1"],
    )
    retrieved_chunks = [
        make_retrieved_chunk(chunk=make_retrieved_chunk().chunk.model_copy(update={"chunk_id": "doc:chunk:0"})),
        make_retrieved_chunk(
            chunk=make_retrieved_chunk().chunk.model_copy(
                update={"chunk_id": "doc:chunk:1", "text": "France is in Europe."}
            ),
            rank=2,
        ),
    ]
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    failures = classify_failures(
        sample,
        retrieved_chunks,
        generated_answer=generated_answer,
    )

    assert FailureMode.MULTI_HOP_INCOMPLETE in failures


def test_classify_failures_adds_unanswerable_fail_for_non_abstention(make_sample) -> None:
    sample = make_sample(
        question_type=QuestionType.UNANSWERABLE,
        is_answerable=False,
        reference_answer=None,
        evidence_chunk_ids=[],
    )
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Paris",
        model_name="unit-test",
    )

    failures = classify_failures(
        sample,
        [],
        generated_answer=generated_answer,
    )

    assert failures == [FailureMode.UNANSWERABLE_FAIL]
