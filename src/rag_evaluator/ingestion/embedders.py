from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.config import EmbedderConfig, EmbedderProvider

Embedding  = list[float]

class Embedder(ABC):
    """
    Base interface for text embedders.
    """
    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts.
        :param texts:
        :return:
        """
        raise NotImplementedError
    
    def embed_query(self, query:str) -> Embedding:
        """
        Embed a single query string.
        :param query:
        :return:
        """
        return self.embed_texts([query])[0]


@dataclass(frozen=True)
class SentenceTransformerEmbedder(Embedder):
    """
    Sentence-transformers based local embedder for BGE-style models.
    """
    
    model_name: str
    batch_size: int = 32
    
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts with sentence-transformers.
        :param texts:
        :return:
        """
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformersEmbedder required `sentence-transformers`"
                "Install it with: python -m pip install -e '.[retrieval]'"
            ) from exc
        
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            list(texts),
            batch_size= self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        
        return [embedding.tolist() for embedding in embeddings]

@dataclass(frozen=True)
class OpenAIEmbedder(Embedder):
    """
    Open-AI compatible API embedder placeholder.
    """
    
    model_name: str
    batch_size: int = 32
    
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts with an OpenAI-compatible embedding API.
        :param texts:
        :return:
        """
        raise NotImplementedError("OpenAI embedding support is not implemented yet.")

@dataclass(frozen=True)
class CohereEmbedder(Embedder):
    """
    Cohere embedder placeholder.
    """
    
    model_name: str
    batch_size: int = 32
    
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        raise NotImplementedError("Cohere embedding support is not implemented yet.")

def build_embedder(config: EmbedderConfig) -> Embedder:
    """
    Build an embedder from config.
    :param config:
    :return:
    """
    
    if config.provider == EmbedderProvider.BGE:
        return SentenceTransformerEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    
    if config.provider == EmbedderProvider.OPENAI:
        return OpenAIEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    
    if config.provider == EmbedderProvider.COHERE:
        return CohereEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    
    raise ValueError(f"Unknown embedder provider: {config.provider}")