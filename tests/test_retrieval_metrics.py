from __future__ import annotations

import pytest

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
