from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.retrieval.base import Retriever
from rag_evaluator.schemas import Chunk, RetrievedChunk


@dataclass(frozen=True)
class BM25Retriever(Retriever):
    """
    Sparse BM25 retriever over in-memory chunks.
    """

    chunks: Sequence[Chunk]
    default_top_k: int = 10
    retriever_name: str = "bm25"

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve chunks using BM25 lexical matching.
        """
        k = top_k or self.default_top_k

        if k < 1:
            raise ValueError("top_k must be >= 1")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not self.chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25Retriever requires `rank-bm25`. "
                "Install it with: python -m pip install -e '.[retrieval]'"
            ) from exc

        tokenized_corpus = [_tokenize(chunk.text) for chunk in self.chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        query_terms = set(tokenized_query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (
                float(item[1]),
                _overlap_count(tokenized_corpus[item[0]], query_terms),
            ),
            reverse=True,
        )

        return [
            RetrievedChunk(
                chunk=self.chunks[chunk_index],
                rank=rank,
                score=float(score),
                retriever_name=self.retriever_name,
            )
            for rank, (chunk_index, score) in enumerate(ranked[:k], start=1)
        ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _overlap_count(tokens: list[str], query_terms: set[str]) -> int:
    return sum(1 for token in tokens if token in query_terms)
