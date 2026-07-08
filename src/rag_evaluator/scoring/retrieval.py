from __future__ import annotations

import math
from collections.abc import Sequence

from rag_evaluator.schemas import EvalSample, RetrievalMetrics, RetrievedChunk
from rag_evaluator.scoring.engine.chunk_relevance import resolve_retrieval_gold


def score_retrieval(
        sample: EvalSample,
        retrieved_chunks: Sequence[RetrievedChunk],
        *,
        k: int
) -> RetrievalMetrics:
    """
    Compute retrieval metrics for one EvalSample.
    :param sample:
    :param retrieved_chunks:
    :param k:
    :return:
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    top_k = list(retrieved_chunks[:k])
    resolution = resolve_retrieval_gold(sample, list(retrieved_chunks))

    if resolution.strategy == "unavailable":
        return RetrievalMetrics(
            precision_at_k=None,
            recall_at_k=None,
            mrr=None,
            ndcg=None,
        )

    relevant_flags = resolution.relevant_flags[: len(top_k)]
    hits = sum(relevant_flags)
    total_relevant = _total_relevant(resolution.resolved_gold_chunk_ids, relevant_flags)

    return RetrievalMetrics(
        precision_at_k=hits/k,
        recall_at_k=hits/total_relevant,
        mrr=_mrr(relevant_flags),
        ndcg=_ndcg(relevant_flags, total_relevant=total_relevant),
    )

def score_retrieval_batch(
        samples: Sequence[EvalSample],
        retrieved_by_sample_id: dict[str, Sequence[RetrievedChunk]],
        *,
        k: int,
) -> dict[str, RetrievalMetrics]:
    """
    Compute retrieval metrics for multiple samples keyed by sample_id.
    :param samples:
    :param retrieved_by_sample_id:
    :param k:
    :return:
    """
    
    return {
        sample.sample_id: score_retrieval(
            sample,
            retrieved_by_sample_id.get(sample.sample_id, []),
            k=k,
        )
        for sample in samples
    }

def _mrr(relevant_flags: Sequence[bool]) -> float:
    for index, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            return 1.0 / index

    return 0.0

def _total_relevant(resolved_gold_chunk_ids: Sequence[str], relevant_flags: Sequence[bool]) -> int:
    if resolved_gold_chunk_ids:
        return len(set(resolved_gold_chunk_ids))
    return max(1, sum(relevant_flags))

def _ndcg(relevant_flags: Sequence[bool], *, total_relevant: int) -> float:
    dcg = 0.0
    
    for index, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            dcg += 1.0 / math.log2(index + 1)

    ideal_relevant = min(total_relevant, len(relevant_flags))
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant + 1))

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg
