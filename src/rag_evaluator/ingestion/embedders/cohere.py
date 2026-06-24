from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.embedders.base import Embedder, Embedding


@dataclass(frozen=True)
class CohereEmbedder(Embedder):
    """
    Cohere API embedder.
    """

    model_name: str
    batch_size: int = 32
    api_key_env: str = "COHERE_API_KEY"
    input_type: str = "search_document"

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts with Cohere.
        """
        if not texts:
            return []

        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "CohereEmbedder requires `cohere`. "
                "Install it after adding `cohere` to project dependencies."
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")

        client = cohere.ClientV2(api_key=api_key)
        embeddings: list[Embedding] = []

        for batch in self._batches(texts):
            response = client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self.input_type,
            )
            embeddings.extend(response.embeddings)

        return embeddings

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        return [
            list(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
