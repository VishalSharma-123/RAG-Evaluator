from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_evaluator.schemas import (
    EvalResult,
    EvidenceSpan,
    FailureMode,
    FinalContext,
)


def test_eval_sample_accepts_valid_payload(make_sample) -> None:
    sample = make_sample()

    assert sample.sample_id == "sample-1"
    assert sample.evidence_chunk_ids == ["doc:chunk:0"]


def test_eval_sample_rejects_extra_fields(make_sample) -> None:
    payload = make_sample().model_dump()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError):
        make_sample().__class__.model_validate(payload)


def test_evidence_span_rejects_negative_offsets() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(
            document_id="doc",
            start_char=-1,
            end_char=4,
        )


def test_chunk_accepts_optional_offsets(make_chunk) -> None:
    chunk = make_chunk(start_char=None, end_char=None)

    assert chunk.start_char is None
    assert chunk.end_char is None


def test_eval_result_contains_failure_modes(
    make_sample,
    make_retrieved_chunk,
    retrieval_metrics,
) -> None:
    result = EvalResult(
        run_id="run-1",
        sample=make_sample(),
        retrieved_chunks=[make_retrieved_chunk()],
        retrieval_metrics=retrieval_metrics,
        failure_modes=[FailureMode.RETRIEVAL_MISS],
    )

    assert result.failure_modes == [FailureMode.RETRIEVAL_MISS]


def test_eval_result_accepts_legacy_final_context_chunks(
    make_sample,
    make_chunk,
    make_retrieved_chunk,
    retrieval_metrics,
) -> None:
    chunk = make_chunk()
    result = EvalResult.model_validate(
        {
            "run_id": "run-1",
            "sample": make_sample().model_dump(mode="json"),
            "retrieved_chunks": [make_retrieved_chunk().model_dump(mode="json")],
            "final_context_chunks": [chunk.model_dump(mode="json")],
            "retrieval_metrics": retrieval_metrics.model_dump(mode="json"),
        }
    )

    assert result.final_context_chunks == [chunk]
    assert result.final_context.chunks == [chunk]
    assert result.final_context.rendered_text == ""


def test_eval_result_exposes_structured_final_context(
    make_sample,
    make_chunk,
    make_retrieved_chunk,
    retrieval_metrics,
) -> None:
    chunk = make_chunk(text="Structured context.")
    result = EvalResult(
        run_id="run-1",
        sample=make_sample(),
        retrieved_chunks=[make_retrieved_chunk()],
        final_context=FinalContext(
            chunks=[chunk],
            rendered_text="[1] Structured context.",
            metadata={"source": "reranker"},
        ),
        retrieval_metrics=retrieval_metrics,
    )

    assert result.final_context_chunks == [chunk]
    assert result.final_context.rendered_text == "[1] Structured context."
    assert result.final_context.metadata == {"source": "reranker"}


def test_failure_mode_supports_question_type_specific_values() -> None:
    assert FailureMode.MULTI_HOP_INCOMPLETE == "MULTI_HOP_INCOMPLETE"
    assert FailureMode.COMPARISON_INCOMPLETE == "COMPARISON_INCOMPLETE"
    assert FailureMode.ABSTRACTIVE_UNGROUNDED == "ABSTRACTIVE_UNGROUNDED"
    assert FailureMode.ADVERSARIAL_MISREAD == "ADVERSARIAL_MISREAD"
