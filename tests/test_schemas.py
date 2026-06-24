from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_evaluator.schemas import Chunk, EvalResult, EvidenceSpan, FailureMode


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


def test_eval_result_contains_failure_modes(make_sample, make_retrieved_chunk, retrieval_metrics) -> None:
    result = EvalResult(
        run_id="run-1",
        sample=make_sample(),
        retrieved_chunks=[make_retrieved_chunk()],
        retrieval_metrics=retrieval_metrics,
        failure_modes=[FailureMode.RETRIEVAL_MISS],
    )

    assert result.failure_modes == [FailureMode.RETRIEVAL_MISS]
