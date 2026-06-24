from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.retrieval.base import Retriever
from rag_evaluator.retrieval.bm25 import BM25Retriever
from rag_evaluator.retrieval.vector import VectorRetriever
from rag_evaluator.schemas import RetrievedChunk


@dataclass(frozen=True)
class HybridRetriever(Retriever):
    """
    Hybrid dense+sparse retriever using reciprocal rank fusion.
    """

    vector_retriever: VectorRetriever
    bm25_retriever: BM25Retriever
    default_top_k: int = 10
    retriever_name: str = "hybrid"
    rrf_k: int = 60

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve chunks using dense and sparse retrievers, then fuse by rank.
        """
        k = top_k or self.default_top_k

        if k < 1:
            raise ValueError("top_k must be >= 1")

        vector_results = self.vector_retriever.retrieve(query, top_k=k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=k)

        fused: dict[str, tuple[RetrievedChunk, float, dict[str, float | int]]] = {}

        for source_name, results in (
            ("vector", vector_results),
            ("bm25", bm25_results),
        ):
            for result in results:
                chunk_id = result.chunk.chunk_id
                rrf_score = 1.0 / (self.rrf_k + result.rank)

                if chunk_id not in fused:
                    fused[chunk_id] = (
                        result,
                        0.0,
                        {
                            "vector_rank": 0,
                            "vector_score": 0.0,
                            "bm25_rank": 0,
                            "bm25_score": 0.0,
                        },
                    )

                representative, current_score, metadata = fused[chunk_id]
                metadata[f"{source_name}_rank"] = result.rank
                metadata[f"{source_name}_score"] = float(result.score)
                fused[chunk_id] = (representative, current_score + rrf_score, metadata)

        ranked = sorted(
            fused.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievedChunk(
                chunk=result.chunk,
                rank=rank,
                score=fused_score,
                retriever_name=self.retriever_name,
                metadata={
                    **metadata,
                    "fusion": "reciprocal_rank_fusion",
                    "rrf_k": self.rrf_k,
                },
            )
            for rank, (result, fused_score, metadata) in enumerate(ranked[:k], start=1)
        ]
