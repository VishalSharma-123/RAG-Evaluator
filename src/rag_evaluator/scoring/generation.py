from __future__ import annotations

from collections.abc import Sequence

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
    
    reference = (sample.reference_answer or "").strip()

    relevance = _answer_relevance(sample, answer)
    faithfulness = _faithfulness(sample, answer)
    hallucination = _hallucination_score(sample, answer)
    bert_score = _reference_overlap(reference, answer) if reference else None

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
        answer: str
) -> float:
    if not answer.strip():
        return 0.0
    
    question_tokens = _tokenize(sample.question)
    answer_tokens = _tokenize(answer)
    
    if not question_tokens or not answer_tokens:
        return 0.0
    
    overlap = len(question_tokens & answer_tokens)
    return overlap / len(question_tokens)

def _faithfulness(sample: EvalSample, answer: str) -> float:
    if not answer.strip():
        return 0.0
    
    if not sample.reference_answer:
        return 0.0 if sample.is_answerable else _unanswerable_alignment(answer)
    
    return _reference_overlap(sample.reference_answer, answer)

def _hallucination_score(sample: EvalSample, answer: str) -> float:
    if not answer.strip():
        return 0.0
    
    if not sample.is_answerable:
        return 0.0 if _looks_like_abstention(answer) else 1.0
    
    if not sample.reference_answer:
        return 1.0
    
    overlap = _reference_overlap(sample.reference_answer, answer)
    return 1.0 - overlap

def _unanswerable_alignment(answer: str) -> str:
    return 1.0 if _looks_like_abstention(answer) else 0.0

def _looks_like_abstention(answer: str) -> bool:
    normalized = answer.lower().strip()
    abstentions = [
        "i don't know",
        "i do not know",
        "unknown",
        "not enough information",
        "insufficient information",
    ]
    
    return normalized in abstentions

def _reference_overlap(reference: str, answer: str) -> float:
    reference_tokens = _tokenize(reference)
    answer_tokens = _tokenize(answer)
    
    if not reference_tokens or not answer_tokens:
        return 0.0
    
    overlap = len(reference_tokens & answer_tokens)
    return overlap / len(reference_tokens)

def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}

