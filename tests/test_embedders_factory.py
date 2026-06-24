from __future__ import annotations

import pytest

from rag_evaluator.config import EmbedderConfig, EmbedderProvider
from rag_evaluator.ingestion.embedders import (
    CohereEmbedder,
    OpenAIEmbedder,
    OpenRouterEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)


def test_embedder_factory_builds_supported_embedders() -> None:
    assert isinstance(
        build_embedder(EmbedderConfig(provider=EmbedderProvider.BGE, model="BAAI/bge-small-en-v1.5")),
        SentenceTransformerEmbedder,
    )
    assert isinstance(
        build_embedder(EmbedderConfig(provider=EmbedderProvider.OPENAI, model="text-embedding-3-small")),
        OpenAIEmbedder,
    )
    assert isinstance(
        build_embedder(EmbedderConfig(provider=EmbedderProvider.OPENROUTER, model="nvidia/test")),
        OpenRouterEmbedder,
    )
    assert isinstance(
        build_embedder(EmbedderConfig(provider=EmbedderProvider.COHERE, model="embed-english-v3.0")),
        CohereEmbedder,
    )


def test_openai_embedder_requires_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")

    with pytest.raises(ValueError, match="Missing API key environment variable"):
        embedder.embed_texts(["hello"])


def test_openrouter_embedder_rejects_unknown_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    embedder = OpenRouterEmbedder(model_name="nvidia/test", input_type="bad")

    with pytest.raises(ValueError, match="Unsupported OpenRouter input_type"):
        embedder.embed_texts(["hello"])


def test_cohere_embedder_is_present_for_future_api_testing() -> None:
    embedder = CohereEmbedder(model_name="embed-english-v3.0")

    assert embedder.model_name == "embed-english-v3.0"
