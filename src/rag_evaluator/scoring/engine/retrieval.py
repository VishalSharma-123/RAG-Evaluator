from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.schemas import EvalSample, RetrievalMetrics, RetrievedChunk
from rag_evaluator.scoring.engine.base import RetrievalScorer


def _normalize_k(retrieved_chunks: Sequence[RetrievedChunk], k: int | None) -> int:
    if k is None:
        return max(1, len(retrieved_chunks))
    if k < 1:
        raise ValueError("k must be >= 1.")
    return k

def score_retrieval_metrics(
        sample: EvalSample,
        retrieved_chunks: Sequence[RetrievedChunk],
        *,
        k: int | None = None,
) -> RetrievalMetrics:
    """
    Score retrieval metrics for one sample with safe tok-k normalization
    :param sample:
    :param retrieved_chunks:
    :param k:
    :return:
    """
    normalized_k = _normalize_k(retrieved_chunks, k)
    from rag_evaluator.scoring.retrieval import score_retrieval as _score_retrieval

    return _score_retrieval(sample, retrieved_chunks, k=normalized_k)

def score_retrieval_metrics_batch(
        samples: Sequence[EvalSample],
        retrieved_by_sample_id: dict[str, Sequence[RetrievedChunk]],
        *,
        k: int | None = None,
) -> dict[str, RetrievalMetrics]:
    """
    Score retrieval metrics for a batch of samples keyed by sample_id
    :param samples:
    :param retrieved_by_sample_id:
    :param k:
    :return:
    """
    if k is not None and k < 1:
        raise ValueError("k must be >= 1.")

    batch_k = k
    if batch_k is None:
        batch_k = max(1, max((len(retrieved_chunks) for retrieved_chunks in retrieved_by_sample_id.values()), default=0))
    
    from rag_evaluator.scoring.retrieval import (
        score_retrieval_batch as _score_retrieval_batch,
    )

    return _score_retrieval_batch(
        samples,
        retrieved_by_sample_id,
        k=batch_k,
    )

@dataclass(frozen=True)
class DefaultRetrievalScorer(RetrievalScorer):
    """
    Concrete retrieval scorer used by the scoring engine.
    """
    
    def score(
            self,
            sample: EvalSample,
            retrieved_chunks: Sequence[RetrievedChunk],
            *,
            k: int
    ) -> RetrievalMetrics:
        return score_retrieval_metrics(sample, retrieved_chunks, k=k)
