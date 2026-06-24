from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.embedders.base import Embedder, Embedding


@dataclass(frozen=True)
class OpenAIEmbedder(Embedder):
    """
    OpenAI-compatible text embedder.
    """

    model_name: str
    batch_size: int = 32
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts with an OpenAI-compatible embedding API.
        """
        if not texts:
            return []

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbedder requires `openai`. "
                "Install it with: python -m pip install -e '.[llm]'"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        embeddings: list[Embedding] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)

        return embeddings
