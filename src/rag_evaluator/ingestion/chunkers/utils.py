from __future__ import annotations

import re


def chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:chunk:{chunk_index}"


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    sentences: list[tuple[str, int, int]] = []
    pattern = re.compile(r"[^.!?]+[.!?]*\s*", re.MULTILINE)

    for match in pattern.finditer(text):
        sentence = match.group(0)
        if not sentence.strip():
            continue
        sentences.append((sentence, match.start(), match.end()))

    return sentences


def sentence_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_tokens(left))
    right_tokens = set(_normalize_tokens(right))

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / len(union)


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
