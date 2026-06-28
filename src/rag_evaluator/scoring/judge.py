from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics


class JudgeScoringError(ValueError):
    """
    Raised when judge-based scoring fails or returns invalid output.
    """


class GenerationJudge(ABC):
    """
    Base interface for generation judges.
    """

    @abstractmethod
    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        """
        Score one generated answer against the sample and retrieved context.
        """
        raise NotImplementedError


class HeuristicJudge(GenerationJudge):
    """
    Deterministic judge that scores answers using reference/context overlap.

    This is functional today and can be reused by model-family wrappers until
    a true LLM-as-judge path is added.
    """

    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        answer = generated_answer.answer.strip()
        if not answer:
            return GenerationMetrics(
                faithfulness=0.0,
                relevance=0.0,
                hallucination=1.0,
                bert_score=0.0,
            )

        context_text = " ".join(chunk.text for chunk in context_chunks if chunk.text.strip())
        reference = (sample.reference_answer or "").strip()

        relevance = _answer_relevance(sample.question, answer)
        faithfulness = _faithfulness(sample, answer, context_text)
        hallucination = _hallucination_score(sample, answer, context_text)
        bert_score = _reference_overlap(reference, answer) if reference else None

        return GenerationMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            bert_score=bert_score,
        )


@dataclass(frozen=True)
class NemotronJudge(HeuristicJudge):
    """
    Judge wrapper for the approved nvidia/nemotron family.

    Today this reuses the deterministic heuristic scoring path. Later, this can
    be swapped to a Nemotron-backed judge prompt without changing callers.
    """

    config: LLMConfig

    def __post_init__(self) -> None:
        if not self.config.model.startswith("nvidia/nemotron"):
            raise ValueError("NemotronJudge requires a model in the `nvidia/nemotron` family.")


def _faithfulness(sample: EvalSample, answer: str, context_text: str) -> float:
    if not sample.is_answerable:
        return 1.0 if _looks_like_abstention(answer) else 0.0

    if not context_text.strip():
        return 0.0

    context_overlap = _reference_overlap(context_text, answer)

    if sample.reference_answer:
        reference_overlap = _reference_overlap(sample.reference_answer, answer)
        return (context_overlap + reference_overlap) / 2.0

    return context_overlap


def _hallucination_score(sample: EvalSample, answer: str, context_text: str) -> float:
    if not answer.strip():
        return 1.0

    if not sample.is_answerable:
        return 0.0 if _looks_like_abstention(answer) else 1.0

    if not context_text.strip():
        return 1.0

    return 1.0 - _reference_overlap(context_text, answer)


def _answer_relevance(question: str, answer: str) -> float:
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)

    if not question_tokens or not answer_tokens:
        return 0.0

    overlap = len(question_tokens & answer_tokens)
    return overlap / len(question_tokens)


def _reference_overlap(reference: str, answer: str) -> float:
    reference_tokens = _tokenize(reference)
    answer_tokens = _tokenize(answer)

    if not reference_tokens or not answer_tokens:
        return 0.0

    overlap = len(reference_tokens & answer_tokens)
    return overlap / len(reference_tokens)


def _looks_like_abstention(answer: str) -> bool:
    normalized = answer.lower().strip()
    abstentions = {
        "i don't know",
        "i do not know",
        "unknown",
        "not enough information",
        "insufficient information",
    }
    return normalized in abstentions


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}
