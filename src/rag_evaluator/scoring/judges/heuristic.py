from __future__ import annotations

from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.question_types.signals import extract_question_keywords, token_overlap_ratio
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics
from rag_evaluator.scoring.judges.base import GenerationJudge


class HeuristicJudge(GenerationJudge):
    """
    Deterministic judge that scores answers using question-type-aware heuristics.
    """

    def score(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, object] | None = None,
    ) -> GenerationMetrics:
        answer = generated_answer.answer.strip()
        if not answer:
            return GenerationMetrics(
                faithfulness=0.0,
                relevance=0.0,
                hallucination=1.0,
                bert_score=0.0,
            )

        rule = get_question_type_rule(sample.question_type)
        type_signals = rule.score_answer(
            sample,
            generated_answer,
            context_chunks=context_chunks,
        )

        relevance = _answer_relevance(sample.question, answer, type_signals=type_signals)
        faithfulness = _faithfulness(sample, answer, type_signals=type_signals)
        hallucination = _hallucination_score(sample, answer, type_signals=type_signals)
        bert_score = _bert_score(sample, answer, type_signals=type_signals)

        return GenerationMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            bert_score=bert_score,
        )


def _answer_relevance(question: str, answer: str, *, type_signals: object) -> float:
    question_keywords = extract_question_keywords(question)
    answer_tokens = set(answer.lower().split())

    if not question_keywords or not answer_tokens:
        return 0.0

    overlap = len(question_keywords & answer_tokens)
    relevance = overlap / len(question_keywords)

    if getattr(type_signals, "performed_comparison", False) and not getattr(
        type_signals,
        "covered_key_entities",
        False,
    ):
        relevance *= 0.5

    return relevance


def _faithfulness(sample: EvalSample, answer: str, *, type_signals: object) -> float:
    if not sample.is_answerable:
        return 1.0 if getattr(type_signals, "abstained_correctly", False) else 0.0

    if sample.question_type.value == "multi_hop":
        if getattr(type_signals, "grounded_in_reference", False) and getattr(
            type_signals,
            "used_multiple_evidence_chunks",
            False,
        ):
            return 1.0
        if getattr(type_signals, "grounded_in_reference", False):
            return 0.5
        return 0.0

    if sample.question_type.value == "comparative":
        if getattr(type_signals, "grounded_in_reference", False) and getattr(
            type_signals,
            "covered_key_entities",
            False,
        ):
            return 1.0
        if getattr(type_signals, "grounded_in_reference", False):
            return 0.5
        return 0.0

    if sample.question_type.value == "abstractive":
        if getattr(type_signals, "grounded_in_context", False):
            return 1.0 if getattr(type_signals, "grounded_in_reference", False) else 0.7
        return 0.0

    if sample.question_type.value == "adversarial":
        if getattr(type_signals, "difficulty_mismatch", False):
            return 0.0
        if getattr(type_signals, "grounded_in_reference", False):
            return 1.0
        if getattr(type_signals, "grounded_in_context", False):
            return 0.5
        return 0.0

    if sample.reference_answer:
        return token_overlap_ratio(sample.reference_answer, answer)

    return 0.0


def _hallucination_score(sample: EvalSample, answer: str, *, type_signals: object) -> float:
    if not sample.is_answerable:
        return 0.0 if getattr(type_signals, "abstained_correctly", False) else 1.0

    if sample.question_type.value == "adversarial" and getattr(
        type_signals,
        "difficulty_mismatch",
        False,
    ):
        return 1.0

    if sample.question_type.value == "abstractive":
        return 0.0 if getattr(type_signals, "grounded_in_context", False) else 1.0

    if sample.question_type.value == "multi_hop":
        if getattr(type_signals, "grounded_in_reference", False) and getattr(
            type_signals,
            "used_multiple_evidence_chunks",
            False,
        ):
            return 0.0
        if getattr(type_signals, "grounded_in_reference", False):
            return 0.5
        return 1.0

    if sample.question_type.value == "comparative":
        if getattr(type_signals, "grounded_in_reference", False) and getattr(
            type_signals,
            "covered_key_entities",
            False,
        ):
            return 0.0
        if getattr(type_signals, "grounded_in_reference", False):
            return 0.5
        return 1.0

    if sample.reference_answer:
        return 1.0 - token_overlap_ratio(sample.reference_answer, answer)

    return 1.0


def _bert_score(sample: EvalSample, answer: str, *, type_signals: object) -> float | None:
    if not sample.reference_answer:
        return None

    if sample.question_type.value == "abstractive":
        return max(
            token_overlap_ratio(sample.reference_answer, answer),
            0.7 if getattr(type_signals, "grounded_in_context", False) else 0.0,
        )

    return token_overlap_ratio(sample.reference_answer, answer)
