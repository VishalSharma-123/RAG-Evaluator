from __future__ import annotations

import re
from collections.abc import Iterable

from rag_evaluator.schemas import Chunk, GeneratedAnswer


def normalize_text(text: str) -> str:
    """
    Lowercase and collapse text into a toke-friendly normalized form.
    :param text:
    :return:
    """
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def tokenize(text: str) -> set[str]:
    """
    Tokenize normalized text into a unique token set.
    :param text:
    :return:
    """
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return {token for token in normalized.split() if token}

def token_overlap_ratio(reference_text: str, candidate_text: str) -> float:
    """
    Compute token overlap.
    :param reference_text:
    :param candidate_text:
    :return:
    """
    reference_tokens = tokenize(reference_text)
    candidate_tokens = tokenize(candidate_text)
    
    if not reference_tokens or not candidate_tokens:
        return 0.0
    
    overlap = len(reference_tokens & candidate_tokens)
    return overlap / len(reference_tokens)

def build_context_text(chunks: Iterable[Chunk]) -> str:
    """
    Join non-empty chunk text into one deterministic context string.
    :param chunks:
    :return:
    """
    texts = [chunk.text.strip() for chunk in chunks if chunk.text.strip()]
    return " ".join(texts)

def looks_like_abstention(answer: str) -> bool:
    """
    Detect simple abstention-style responses for unanswerable answers.
    :param answer:
    :return:
    """
    normalized = normalize_text(answer)
    abstentions = {
        "i don t know",
        "i do not know",
        "unknown",
        "not enough information",
        "insufficient information",
        "cannot be determined",
        "can t be determined",
    }
    return normalized in abstentions

def answer_text(generated_answer: GeneratedAnswer | None) -> str:
    """
    Safely extract stripped answer text from a generated answer.
    :param generated_answer:
    :return:
    """
    if generated_answer is None:
        return ""
    return generated_answer.answer.strip()

def is_grounded_in_text(
        answer: str,
        evidence_text: str,
        *,
        min_overlap_ratio: float = 0.6,
) -> bool:
    """
    Check whether an answer is sufficiently grounded in a body of evidence.
    :param answer:
    :param evidence_text:
    :param min_overlap_ratio:
    :return:
    """
    normalized_answer = normalize_text(answer)
    normalized_evidence = normalize_text(evidence_text)

    if not normalized_answer or not normalized_evidence:
        return False

    if normalized_answer in normalized_evidence:
        return True

    return token_overlap_ratio(answer, evidence_text) >= min_overlap_ratio

def count_grounding_chunks(
        answer: str,
        chunks: Iterable[Chunk],
        *,
        min_overlap_ratio: float = 0.6,
) -> int:
    """
    Count how many chunks individually ground the answer.
    :param answer:
    :param chunks:
    :param min_overlap_ratio:
    :return:
    """
    grounded = 0
    for chunk in chunks:
        if is_grounded_in_text(answer, chunk.text, min_overlap_ratio=min_overlap_ratio):
            grounded += 1
    return grounded

def extract_question_keywords(question: str) -> set[str]:
    """
    Extract a simple keyword set from a question for heuristic relevance checks.
    :param question:
    :return:
    """
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
    return {token for token in tokenize(question) if token not in stopwords}

def has_comparison_language(text: str) -> bool:
    """
    Heuristic detection for comparison-style phrasing.
    :param text:
    :return:
    """
    normalized = normalize_text(text)
    comparison_markers = (
        "compare",
        "compared",
        "comparison",
        "better",
        "worse",
        "higher",
        "lower",
        "more",
        "less",
        "than",
        "versus",
        "vs",
        "difference",
        "different",
        "similar",
    )
    return any(marker in normalized.split() for marker in comparison_markers)