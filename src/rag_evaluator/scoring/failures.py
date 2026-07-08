from __future__ import annotations

from collections.abc import Sequence

from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk
from rag_evaluator.scoring.engine.chunk_relevance import resolve_retrieval_gold


def classify_failures(
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
    Classify typed failure modes for one evaluated sample.
    :param sample:
    :param retrieved_chunks:
    :param generated_answer:
    :param context_was_used:
    :param hallucination_score:
    :param partial_answer_score:
    :param retrieval_k:
    :return:
    """
    
    failures: list[FailureMode] = []
    resolution = resolve_retrieval_gold(sample, list(retrieved_chunks))

    if sample.is_answerable and resolution.strategy != "unavailable":
        if not any(resolution.relevant_flags):
            failures.append(FailureMode.RETRIEVAL_MISS)
        elif retrieval_k is not None:
            if not any(resolution.relevant_flags[:retrieval_k]):
                failures.append(FailureMode.RETRIEVAL_RANK)

    if generated_answer is not None:
        if not sample.is_answerable and not _looks_like_abstention(generated_answer.answer):
            failures.append(FailureMode.UNANSWERABLE_FAIL)

        if context_was_used is False:
            failures.append(FailureMode.CONTEXT_IGNORED)

        if hallucination_score is not None and hallucination_score > 0.5:
            failures.append(FailureMode.HALLUCINATION)

        if partial_answer_score is not None and partial_answer_score > 0.5:
            failures.append(FailureMode.PARTIAL_ANSWER)

        failures.extend(
            _classify_question_type_failures(
                sample,
                retrieved_chunks,
                generated_answer=generated_answer,
            )
        )

    return _deduplicate_failures(failures)


def _deduplicate_failures(failures: Sequence[FailureMode]) -> list[FailureMode]:
    seen: set[FailureMode] = set()
    deduped: list[FailureMode] = []

    for failure in failures:
        if failure not in seen:
            seen.add(failure)
            deduped.append(failure)

    return deduped


def _looks_like_abstention(answer: str) -> bool:
    normalized = answer.lower().strip()
    abstentions = {
        "i don't know",
        "i do not know",
        "unknown",
        "not enough information",
        "insufficient information",
        "cannot be determined",
        "can't be determined",
    }
    return normalized in abstentions


def _classify_question_type_failures(
    sample: EvalSample,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    generated_answer: GeneratedAnswer,
) -> list[FailureMode]:
    rule = get_question_type_rule(sample.question_type)
    context_chunks: list[Chunk] = [retrieved.chunk for retrieved in retrieved_chunks]
    type_signals = rule.score_answer(
        sample,
        generated_answer,
        context_chunks=context_chunks,
    )

    return rule.classify_failures(
        sample,
        list(retrieved_chunks),
        generated_answer=generated_answer,
        type_signals=type_signals,
    )
