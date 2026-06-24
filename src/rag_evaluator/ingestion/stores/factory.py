from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_evaluator.ingestion.stores.base import VectorStore
from rag_evaluator.ingestion.stores.chroma import ChromaVectorStore
from rag_evaluator.ingestion.stores.memory import InMemoryVectorStore


def build_vector_store(
        *,
        provider: str = "memory",
        collection_name: str | None = None,
        persist_directory: str | Path = "storage/chroma",
        metadata: dict[str, Any] | None = None,
) -> VectorStore:
    """
    Build a vector store with the provided name.
    :param provider:
    :param collection_name:
    :param persist_directory:
    :param metadata:
    :return:
    """
    
    normalized_provider = provider.lower().strip()
    metadata = metadata or {}
    
    if normalized_provider == "memory":
        return InMemoryVectorStore()
    
    if normalized_provider == "chroma":
        resolved_collection_name = collection_name or metadata.get("collection_name")
        if not resolved_collection_name:
            raise ValueError("collection_name is required for ChromaVectorStore")
        
        return ChromaVectorStore(
            collection_name=resolved_collection_name,
            persist_directory = metadata.get("persist_directory", persist_directory),
        )

    raise ValueError(f"Invalid vector store provider {provider}")
