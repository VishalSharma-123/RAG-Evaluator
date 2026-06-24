from __future__ import annotations

from rag_evaluator.config import EmbedderConfig, EmbedderProvider
from rag_evaluator.ingestion.embedders.base import Embedder
from rag_evaluator.ingestion.embedders.cohere import CohereEmbedder
from rag_evaluator.ingestion.embedders.openai import OpenAIEmbedder
from rag_evaluator.ingestion.embedders.openrouter import OpenRouterEmbedder
from rag_evaluator.ingestion.embedders.sentence_transformer import SentenceTransformerEmbedder


def build_embedder(config: EmbedderConfig) -> Embedder:
    """
    Build an embedder from config.
    """
    if config.provider == EmbedderProvider.BGE:
        return SentenceTransformerEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
            show_progress_bar=config.metadata.get("show_progress_bar", False),
        )

    if config.provider == EmbedderProvider.OPENAI:
        return OpenAIEmbedder(
            model_name=config.model or config.metadata.get("model"),
            batch_size=config.batch_size,
            api_key_env=config.metadata.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.metadata.get("base_url"),
        )

    if config.provider == EmbedderProvider.OPENROUTER:
        return OpenRouterEmbedder(
            model_name=config.model or config.metadata.get("model"),
            batch_size=config.batch_size,
            api_key_env=config.metadata.get("api_key_env", "OPENROUTER_API_KEY"),
            base_url=config.metadata.get("base_url", "https://openrouter.ai/api/v1/embeddings"),
            input_type=config.metadata.get("input_type", "text"),
        )

    if config.provider == EmbedderProvider.COHERE:
        return CohereEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
            api_key_env=config.metadata.get("api_key_env", "COHERE_API_KEY"),
            input_type=config.metadata.get("input_type", "search_document"),
        )

    raise ValueError(f"Unknown embedder provider: {config.provider}")
