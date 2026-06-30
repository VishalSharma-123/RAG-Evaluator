from __future__ import annotations

from collections.abc import Sequence

from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.question_types.signals import (
    extract_question_keywords,
    looks_like_abstention,
    token_overlap_ratio,
)
from rag_evaluator.schemas import EvalSample, GeneratedAnswer, GenerationMetrics


def score_generation(
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
) -> GenerationMetrics | None:
    """
    Compute generation-only metrics for one EvalSample.
    :param sample:
    :param generated_answer:
    :return:
    """
    if generated_answer is None:
        return None
    
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
        context_chunks=[]
    )
    
    relevance = _answer_relevance(sample, answer, type_signals=type_signals)
    faithfulness = _faithfulness(sample, answer, type_signals=type_signals)
    hallucination = _hallucination_score(sample, answer, type_signals=type_signals)
    bert_score = _bert_score(sample, answer, type_signals=type_signals)

    return GenerationMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        hallucination=hallucination,
        bert_score=bert_score,
    )
def score_generation_batch(
        samples: Sequence[EvalSample],
        generated_by_sample_id: dict[str, GeneratedAnswer | None],
) -> dict[str, GenerationMetrics | None]:
    """
    Compute generation metrics for multiple samples keyed by sample_id.
    :param samples:
    :param generated_by_sample_id:
    :return:
    """
    return {
        sample.sample_id: score_generation(
            sample,
            generated_by_sample_id.get(sample.sample_id),
        )
        for sample in samples
    }

def _answer_relevance(
        sample: EvalSample,
        answer: str,
        *,
        type_signals,
) -> float:
    if not answer.strip():
        return 0.0
    
    question_keywords = extract_question_keywords(sample.question)
    answer_tokens = set(answer.lower().split())
    
    if not question_keywords or not answer_tokens:
        return 0.0
    
    overlap = len(question_keywords & answer_tokens)
    relevance = overlap / len(question_keywords)
    
    if sample.question_type.value == "comparative" and not type_signals.covered_key_entities:
        relevance *= 0.5
    
    return relevance

def _faithfulness(sample: EvalSample, answer: str, *, type_signals) -> float:
    if not answer.strip():
        return 0.0
    
    if not sample.is_answerable:
        return 1.0 if type_signals.abstained_correctly else 0.0
    
    if sample.question_type.value == "multi_hop":
        if type_signals.grounded_in_reference and type_signals.used_multiple_evidence_chunks:
            return 1.0
        if type_signals.grounded_in_reference:
            return 0.5
        return 0.0
    
    if sample.question_type.value == "comparative":
        if type_signals.grounded_in_reference and type_signals.covered_key_entities:
            return 1.0
        if type_signals.grounded_in_reference:
            return 0.5
        return 0.0
    
    if sample.question_type.value == "abstractive":
        if type_signals.grounded_in_context:
            return 1.0 if type_signals.grounded_in_reference else 0.7
        return 0.0
    
    if sample.question_type.value == "adversarial":
        if type_signals.difficulty_mismatch:
            return 0.0
        if type_signals.grounded_in_reference:
            return 1.0
        if type_signals.grounded_in_context:
            return 0.5
        return 0.0
    
    if sample.reference_answer:
        return token_overlap_ratio(sample.reference_answer, answer)
    
    return 0.0

def _hallucination_score(sample: EvalSample, answer: str, *, type_signals) -> float:
    if not answer.strip():
        return 1.0
    
    if not sample.is_answerable:
        return 0.0 if type_signals.abstained_correctly else 1.0
    
    if sample.question_type.value == "adversarial" and type_signals.difficulty_mismatch:
        return 1.0
    
    if sample.question_type.value == "abstractive":
        return 0.0 if type_signals.grounded_in_context else 1.0
    
    if sample.question_type.value == "multi_hop":
        if type_signals.grounded_in_reference and type_signals.used_multiple_evidence_chunks:
            return 0.0
        if type_signals.grounded_in_reference:
            return 0.5
        return 1.0
    
    if sample.question_type.value == "comparative":
        if type_signals.grounded_in_reference and type_signals.covered_key_entities:
            return 0.0
        if type_signals.grounded_in_reference:
            return 0.5
        return 1.0
    
    if sample.reference_answer:
        return 1.0 - token_overlap_ratio(sample.reference_answer, answer)
    
    return 1.0

def _bert_score(sample: EvalSample, answer: str, *, type_signals) -> float:
    if not sample.reference_answer:
        return None
    
    if not sample.is_answerable and looks_like_abstention(answer):
        return None
    
    if sample.question_type.value == "abstractive":
        return (
            max(
                token_overlap_ratio(sample.reference_answer, answer),
                0.7 if type_signals.grounded_in_context else 0.0,
            )
        )
    
    return token_overlap_ratio(sample.reference_answer, answer)