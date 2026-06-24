from __future__ import annotations

from collections.abc import Sequence

from rag_evaluator.config import RetrieverConfig, RetrieverType
from rag_evaluator.ingestion.embedders import Embedder
from rag_evaluator.ingestion.stores import VectorStore
from rag_evaluator.retrieval.base import Retriever
from rag_evaluator.retrieval.bm25 import BM25Retriever
from rag_evaluator.retrieval.hybrid import HybridRetriever
from rag_evaluator.retrieval.vector import VectorRetriever
from rag_evaluator.schemas import Chunk


def build_retriever(
    *,
    config: RetrieverConfig,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    chunks: Sequence[Chunk] | None = None,
) -> Retriever:
    """
    Build a retriever from config and available retrieval dependencies.
    """
    if config.type == RetrieverType.VECTOR:
        if embedder is None:
            raise ValueError("Vector retrieval requires an embedder.")

        if vector_store is None:
            raise ValueError("Vector retrieval requires a vector store.")

        return VectorRetriever(
            embedder=embedder,
            store=vector_store,
            default_top_k=config.top_k,
        )

    if config.type == RetrieverType.BM25:
        if chunks is None:
            raise ValueError("BM25 retrieval requires chunks.")

        return BM25Retriever(
            chunks=chunks,
            default_top_k=config.top_k,
        )

    if config.type == RetrieverType.HYBRID:
        if embedder is None:
            raise ValueError("Hybrid retrieval requires an embedder.")

        if vector_store is None:
            raise ValueError("Hybrid retrieval requires a vector store.")

        if chunks is None:
            raise ValueError("Hybrid retrieval requires chunks.")

        vector_retriever = VectorRetriever(
            embedder=embedder,
            store=vector_store,
            default_top_k=config.top_k,
        )
        bm25_retriever = BM25Retriever(
            chunks=chunks,
            default_top_k=config.top_k,
        )

        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            default_top_k=config.top_k,
            rrf_k=config.metadata.get("rrf_k", 60),
        )

    raise ValueError(f"Unsupported retriever type: {config.type}")
