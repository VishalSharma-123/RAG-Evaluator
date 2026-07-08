from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_evaluator.schemas import EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.scoring.engine.base import FailureClassifier
from rag_evaluator.scoring.engine.chunk_relevance import resolve_retrieval_gold


@dataclass(frozen=True)
class FailureBreakdown:
    """
    Split failure modes into retrieval-specific and generation-specific buckets.
    """
    
    retrieval: list[FailureMode] = field(default_factory=list)
    generation: list[FailureMode] = field(default_factory=list)
    
    @property
    def all(self) -> list[FailureMode]:
        return _duplicate_failures([*self.retrieval, *self.generation])


@dataclass(frozen=True)
class DefaultFailureClassifier(FailureClassifier):
    """
    Default failure classifier for the scoring engine.
    """
    
    def classify(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        context_was_used: bool | None = None,
        hallucination_score: float | None = None,
        partial_answer_score: float | None = None,
        retrieval_k: int | None = None,
    ) -> list[FailureMode]:
        breakdown = self.classify_breakdown(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
            context_was_used=context_was_used,
            hallucination_score=hallucination_score,
            partial_answer_score=partial_answer_score,
            retrieval_k=retrieval_k,
        )
        return breakdown.all
    
    def classify_breakdown(
            self,
            sample: EvalSample,
            retrieved_chunks: Sequence[RetrievedChunk],
            *,
            generated_answer: GeneratedAnswer | None = None,
            context_was_used: bool | None = None,
            hallucination_score: float | None = None,
            partial_answer_score: float | None = None,
            retrieval_k: int | None = None,
    ) -> FailureBreakdown:
        retrieval_failure = classify_retrieval_failures(
            sample,
            retrieved_chunks,
            retrieval_k=retrieval_k,
        )
        
        generation_failures = classify_generation_failures(
            sample,
            retrieved_chunks,
            generated_answer=generated_answer,
            context_was_used=context_was_used,
            hallucination_score=hallucination_score,
            partial_answer_score=partial_answer_score,
            retrieval_k=retrieval_k,
        )
        
        return FailureBreakdown(
            retrieval=retrieval_failure,
            generation=generation_failures,
        )

def classify_retrieval_failures(
        sample: EvalSample,
        retrieved_chunks: Sequence[RetrievedChunk],
        *,
        retrieval_k: int | None = None,
) -> list[FailureMode]:
    """
    Return retrieval-specific failure modes only.
    :param sample:
    :param retrieved_chunks:
    :param retrieval_k:
    :return:
    """
    if not sample.is_answerable:
        return []

    resolution = resolve_retrieval_gold(sample, list(retrieved_chunks))
    if resolution.strategy == "unavailable":
        return []
    
    failures: list[FailureMode] = []
    
    if not any(resolution.relevant_flags):
        failures.append(FailureMode.RETRIEVAL_MISS)
    elif retrieval_k is not None:
        if not any(resolution.relevant_flags[:retrieval_k]):
            failures.append(FailureMode.RETRIEVAL_RANK)
    
    return failures

def classify_generation_failures(
        sample: EvalSample,
        retrieved_chunks: Sequence[RetrievedChunk],
        *,
        generated_answer: GeneratedAnswer | None = None,
        context_was_used: bool | None = None,
        hallucination_score: float | None = None,
        partial_answer_score: float | None = None,
        retrieval_k: int | None = None,
) -> list[FailureMode]:
    """
    Return generation and question-type-specific failure modes only.
    :param sample:
    :param retrieved_chunks:
    :param generated_answer:
    :param context_was_used:
    :param hallucination_score:
    :param partial_answer_score:
    :param retrieval_k:
    :return:
    """
    from rag_evaluator.scoring.failures import classify_failures as _classify_failures

    all_failures = _classify_failures(
        sample=sample,
        retrieved_chunks=retrieved_chunks,
        generated_answer=generated_answer,
        context_was_used=context_was_used,
        hallucination_score=hallucination_score,
        partial_answer_score=partial_answer_score,
        retrieval_k=retrieval_k,
    )
    
    retrieval_failures = set(
        classify_retrieval_failures(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            retrieval_k=retrieval_k,
        )
    )
    
    return [failure for failure in all_failures if failure not in retrieval_failures]


def _duplicate_failures(failures: Sequence[FailureMode]) -> list[FailureMode]:
    seen: set[FailureMode] = set()
    deduped: list[FailureMode] = []
    
    for failure in failures:
        if failure in seen:
            continue
        seen.add(failure)
        deduped.append(failure)
    
    return deduped
