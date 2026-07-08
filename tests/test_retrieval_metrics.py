from __future__ import annotations

import pytest

from rag_evaluator.schemas import EvidenceSpan
from rag_evaluator.scoring.retrieval import score_retrieval, score_retrieval_batch


def test_score_retrieval_computes_expected_metrics(make_sample, make_retrieved_chunk, make_chunk) -> None:
    sample = make_sample(evidence_chunk_ids=["doc:chunk:0", "doc:chunk:2"])
    retrieved = [
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:1"), rank=1, score=0.9),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:0"), rank=2, score=0.8),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:2"), rank=3, score=0.7),
    ]

    metrics = score_retrieval(sample, retrieved, k=3)

    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.ndcg is not None


def test_score_retrieval_batch_returns_metrics_by_sample_id(make_sample, make_retrieved_chunk) -> None:
    sample = make_sample()
    batch = score_retrieval_batch(
        [sample],
        {sample.sample_id: [make_retrieved_chunk()]},
        k=1,
    )

    assert sample.sample_id in batch
    assert batch[sample.sample_id].precision_at_k == 1.0


def test_score_retrieval_uses_character_span_overlap(make_sample, make_retrieved_chunk, make_chunk) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[
            EvidenceSpan(document_id="doc", start_char=10, end_char=20),
        ],
    )
    retrieved = [
        make_retrieved_chunk(
            chunk=make_chunk(chunk_id="doc:chunk:1", start_char=0, end_char=12),
            rank=1,
        ),
    ]

    metrics = score_retrieval(sample, retrieved, k=1)

    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0


def test_score_retrieval_uses_evidence_text_overlap(make_sample, make_retrieved_chunk, make_chunk) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[
            EvidenceSpan(
                document_id="doc",
                start_char=0,
                end_char=0,
                text="retrieved context grounds the answer",
            ),
        ],
    )
    retrieved = [
        make_retrieved_chunk(
            chunk=make_chunk(
                chunk_id="doc:chunk:1",
                document_id="other",
                text="The retrieved context grounds the answer with details.",
                start_char=20,
                end_char=80,
            ),
            rank=1,
        ),
    ]

    metrics = score_retrieval(sample, retrieved, k=1)

    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0


def test_score_retrieval_uses_answer_text_fallback(make_sample, make_retrieved_chunk, make_chunk) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[],
        reference_answer="Retrieval augmented generation.",
    )
    retrieved = [
        make_retrieved_chunk(
            chunk=make_chunk(text="Retrieval augmented generation uses retrieved context."),
            rank=1,
        ),
    ]

    metrics = score_retrieval(sample, retrieved, k=1)

    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0


def test_score_retrieval_returns_null_metrics_when_gold_unavailable(
    make_sample,
    make_retrieved_chunk,
) -> None:
    sample = make_sample(
        evidence_chunk_ids=[],
        evidence_spans=[],
        reference_answer=None,
    )

    metrics = score_retrieval(sample, [make_retrieved_chunk()], k=1)

    assert metrics.precision_at_k is None
    assert metrics.recall_at_k is None
    assert metrics.mrr is None
    assert metrics.ndcg is None
