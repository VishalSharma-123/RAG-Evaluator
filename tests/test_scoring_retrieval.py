from __future__ import annotations

import pytest

from rag_evaluator.scoring.retrieval import score_retrieval


def test_score_retrieval_rejects_invalid_k(make_sample, make_retrieved_chunk) -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="k must be >= 1"):
        score_retrieval(sample, [make_retrieved_chunk()], k=0)


def test_score_retrieval_returns_zeros_when_no_gold_chunks(make_sample, make_retrieved_chunk) -> None:
    sample = make_sample(evidence_chunk_ids=[])

    metrics = score_retrieval(sample, [make_retrieved_chunk()], k=1)

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg == 0.0
