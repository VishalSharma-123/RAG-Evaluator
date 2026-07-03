from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.reranking.types import Reranker
from rag_evaluator.schemas import EvalSample, RetrievedChunk


@dataclass(frozen=True)
class CohereReranker(Reranker):
    """
    Cohere API-backed reranker.
    """

    configured_type: str
    model_name: str
    api_key_env: str = "COHERE_API_KEY"
    base_url: str | None = None
    client_name: str | None = None
    timeout: float | None = None
    implementation_name: str = "cohere"
    implemented: bool = True
    _client: Any | None = field(default=None, init=False, repr=False, compare=False)

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not retrieved_chunks:
            return []

        client = self._load_client()
        response = client.rerank(
            model=self.model_name,
            query=sample.question,
            documents=[item.chunk.text for item in retrieved_chunks],
            top_n=top_k,
        )
        results = list(response.results)
        ordered_results = sorted(
            results,
            key=lambda result: (-float(result.relevance_score), int(result.index)),
        )
        selected_indices = [int(result.index) for result in ordered_results[:top_k]]

        return [
            retrieved_chunks[index].model_copy(update={"rank": rank})
            for rank, index in enumerate(selected_indices, start=1)
        ]

    def _load_client(self) -> Any:
        client = self._client
        if client is not None:
            return client

        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "CohereReranker requires `cohere`. "
                "Install it with: python -m pip install -e '.[llm]'"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")

        client = cohere.ClientV2(
            api_key=api_key,
            base_url=self.base_url,
            client_name=self.client_name,
            timeout=self.timeout,
        )
        object.__setattr__(self, "_client", client)
        return client
