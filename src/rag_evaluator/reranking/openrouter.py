from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from rag_evaluator.reranking.types import Reranker
from rag_evaluator.schemas import EvalSample, RetrievedChunk


@dataclass(frozen=True)
class OpenRouterReranker(Reranker):
    """
    OpenRouter-backed reranker that uses the dedicated rerank endpoint.
    """

    configured_type: str
    model_name: str
    api_key: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1/rerank"
    timeout_seconds: float = 60.0
    http_referer: str | None = None
    app_name: str | None = None
    implementation_name: str = "openrouter"
    implemented: bool = True

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not retrieved_chunks:
            return []

        response_payload = self._request_rerank(
            query=sample.question,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
        )
        ranked_items = self._extract_ranked_items(response_payload)
        selected_indices = self._select_indices(ranked_items, candidate_count=len(retrieved_chunks))

        selected_indices = selected_indices[:top_k]
        return [
            retrieved_chunks[index].model_copy(update={"rank": rank})
            for rank, index in enumerate(selected_indices, start=1)
        ]

    def _request_rerank(
        self,
        *,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [item.chunk.text for item in retrieved_chunks],
            "top_n": top_k,
        }

        try:
            response = requests.post(
                self._base_url(),
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(
                f"OpenRouter rerank request failed with HTTP error: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenRouter rerank request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("OpenRouter rerank response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("OpenRouter rerank response had an unexpected shape.")

        return payload

    def _extract_ranked_items(self, response_payload: dict[str, Any]) -> list[dict[str, Any]]:
        items = response_payload.get("results")
        if items is None:
            items = response_payload.get("data")

        if not isinstance(items, list):
            raise ValueError(
                "OpenRouter rerank response did not contain a results list."
            )

        parsed_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    "OpenRouter rerank response contained a non-object result item."
                )

            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            if index is None or score is None:
                raise ValueError(
                    "OpenRouter rerank response items must include index and score fields."
                )

            try:
                parsed_index = int(index)
                parsed_score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "OpenRouter rerank response contained invalid index or score values."
                ) from exc

            parsed_items.append(
                {
                    "index": parsed_index,
                    "score": parsed_score,
                }
            )

        return parsed_items

    def _select_indices(
        self,
        ranked_items: list[dict[str, Any]],
        *,
        candidate_count: int,
    ) -> list[int]:
        if len(ranked_items) > candidate_count:
            raise ValueError(
                "OpenRouter rerank response returned more items than candidate chunks."
            )

        seen: set[int] = set()
        ordered = sorted(
            ranked_items,
            key=lambda item: (-item["score"], item["index"]),
        )

        selected_indices: list[int] = []
        for item in ordered:
            index = item["index"]
            if index < 0 or index >= candidate_count:
                raise ValueError(
                    f"OpenRouter rerank response returned invalid candidate index: {index}"
                )
            if index in seen:
                raise ValueError(
                    f"OpenRouter rerank response returned duplicate candidate index: {index}"
                )
            seen.add(index)
            selected_indices.append(index)

        return selected_indices

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

        http_referer = self.http_referer or os.getenv("OPENROUTER_HTTP_REFERER")
        app_name = self.app_name or os.getenv("OPENROUTER_APP_NAME")

        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_name:
            headers["X-Title"] = app_name

        return headers

    def _api_key(self) -> str:
        api_key = self.api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")
        return api_key

    def _base_url(self) -> str:
        return (
            self.base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1/rerank"
        )
