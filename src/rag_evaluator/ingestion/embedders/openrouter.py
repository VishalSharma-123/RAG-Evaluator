from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_evaluator.ingestion.embedders.base import Embedder, Embedding


@dataclass(frozen=True)
class OpenRouterEmbedder(Embedder):
    """
    OpenRouter embedder using the embeddings endpoint directly.
    """

    model_name: str
    batch_size: int = 32
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1/embeddings"
    input_type: str = "text"

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")

        embeddings: list[Embedding] = []

        for batch in self._batches(texts):
            payload = {
                "model": self.model_name,
                "input": [self._build_input_item(text) for text in batch],
                "encoding_format": "float",
            }
            response_data = self._post_embeddings(payload, api_key)
            embeddings.extend(item["embedding"] for item in response_data["data"])

        return embeddings

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        return [
            list(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]

    def _build_input_item(self, text: str) -> str | dict[str, object]:
        if self.input_type == "text":
            return text

        if self.input_type == "multimodal_text":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ]
            }

        raise ValueError(f"Unsupported OpenRouter input_type: {self.input_type}")

    def _post_embeddings(
        self,
        payload: dict[str, object],
        api_key: str,
    ) -> dict[str, object]:
        request = Request(
            url=self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter embeddings request failed with HTTP {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter embeddings request failed: {exc.reason}") from exc

        data = json.loads(body)
        if "data" not in data:
            raise ValueError("OpenRouter embeddings response did not include `data`.")

        return data
