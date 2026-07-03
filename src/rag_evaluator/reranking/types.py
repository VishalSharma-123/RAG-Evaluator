from __future__ import annotations

from rag_evaluator.schemas import EvalSample, RetrievedChunk


class Reranker:
    """
    Minimal reranker protocol used by the execution layer.
    """

    configured_type: str
    implementation_name: str
    implemented: bool

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError