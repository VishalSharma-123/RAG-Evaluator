from __future__ import annotations

import math
from collections.abc import Sequence

from rag_evaluator.schemas import EvalSample, RetrievalMetrics, RetrievedChunk


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

    gold_chunk_ids = set(sample.evidence_chunk_ids)
    top_k = list(retrieved_chunks[:k])

    if not gold_chunk_ids:
        return RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg=0.0,
        )

    retrieved_ids = [retrieved.chunk.chunk_id for retrieved in top_k]
    relevant_flags = [chunk_id in gold_chunk_ids for chunk_id in retrieved_ids]
    hits = sum(relevant_flags)

    return RetrievalMetrics(
        precision_at_k=hits/k,
        recall_at_k=hits/len(gold_chunk_ids),
        mrr=_mrr(relevant_flags),
        ndcg=_ndcg(relevant_flags, total_relevant=len(gold_chunk_ids)),
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
