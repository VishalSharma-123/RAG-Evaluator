from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_evaluator.ingestion.embedders import Embedding
from rag_evaluator.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    """
    Base interface for vector stores.
    """
    
    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """
        Add chunks and corresponding embeddings to the store.
        :param chunks:
        :param embeddings:
        :return:
        """
        raise NotImplementedError
    
    @abstractmethod
    def search(self,
               query_embedding: Embedding,
               *,
               top_k: int,
               retriever_name: str = "vector"
        ) -> list[RetrievedChunk]:
        """
        Search for nearest chunks by query embedding.
        :param query_embedding:
        :param top_k:
        :param retriever_name:
        :return:
        """
        raise NotImplementedError

@dataclass
class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory cosine-similarity vector store for smoke tests..
    """
    
    _chunks: list[Chunk] = field(default_factory=list)
    _embeddings: list[Embedding] = field(default_factory=list)
    
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """
        Add chunks and corresponding embeddings to in-memory index.
        :param chunks:
        :param embeddings:
        :return:
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length.")
        
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not embedding:
                raise ValueError(f"Emtpy embedding for chunk_id={chunk.chunk_id}")
            
            self._chunks.append(chunk)
            self._embeddings.append(embedding)
    
    def search(self,
               query_embeddig: Embedding,
               *,
               top_k: int,
               retriever_name: str = "vector"
        ) -> list[RetrievedChunk]:
        """
        Return the top-k chunks ranked by cosine similarity.
        :param query_embeddig:
        :param top_k:
        :param retriever_name:
        :return:
        """
        if top_k < 1:
            raise ValueError("top_k must be greater than 1.")
        
        if not query_embeddig:
            raise ValueError("query_embedding must not be empty")
        
        scored: list[tuple[float, Chunk]] = []
        
        for chunk, embedding in zip(self._chunks, self._embeddings):
            score = _cosine_similarity(query_embeddig, embedding)
            scored.append((score, chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [
            RetrievedChunk(
                chunk= chunk,
                rank=index,
                score=score,
                retriever_name=retriever_name,
            )
            for index, (score, chunk) in enumerate(scored[:top_k], start = 1)
        ]

def _cosine_similarity(left: Embedding, right: Embedding) -> float:
    if len(left) != len(right) :
        raise ValueError("Embeddings must have same dimensions.")
    
    left_norm = math.sqrt(sum(value*value for value in left))
    right_norm = math.sqrt(sum(value*value for value in right))
    
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    
    return dot_product / (left_norm * right_norm)