from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from transformers.modeling_utils import VLMS

from rag_evaluator.config import RetrieverConfig, RetrieverType
from rag_evaluator.ingestion.embedders import Embedder
from rag_evaluator.ingestion.stores import VectorStore
from rag_evaluator.schemas import Chunk, RetrievedChunk


class Retriever(ABC):
    """
    Base interface for retrieval strategies.
    """
    
    @abstractmethod
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve chunks for a query.
        :param query:
        :param top_k:
        :return:
        """
        raise NotImplementedError()

@dataclass(frozen=True)
class VectorRetriever(Retriever):
    """
    Dense vector retriever backed by a VectorStore
    """
    
    embedder: Embedder
    store: VectorStore
    default_top_k: int = 10
    retriever_name: str = "vector"
    
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed the query and retrieve nearest chunks from the vector store.
        :param query:
        :param top_k:
        :return:
        """
        
        k = top_k or self.default_top_k
        
        if k < 1:
            raise ValueError("top_k must be >= 1")
        
        query_embedding = self.embedder.embed_query(query)
        
        return self.store.search(
            query_embedding,
            top_k=k,
            retriever_name=self.retriever_name,
        )

@dataclass(frozen=True)
class BM25Retriever(Retriever):
    """
    Sparse BM25 retriever placeholder.
    """
    
    chunks: Sequence[Chunk]
    default_top_k: int = 10
    retriever_name: str = "bm25"
    
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retriever chunks using BM25
        :param query:
        :param top_k:
        :return:
        """
        raise NotImplementedError("BM25 Retriever not implemented")

@dataclass(frozen=True)
class HybridRetriever(Retriever):
    """
    Hybrid dense+sparse retriever placeholder.
    """
    
    vector_retriever: VectorRetriever
    bm25_retriever: BM25Retriever
    default_top_k: int = 10
    retriever_name: str = "hybrid"
    
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retriever chunks using Hybrid dense+sparse strategy.
        :param query:
        :param top_k:
        :return:
        """
        raise NotImplementedError("Hybrid Retriever not implemented")

def build_retriever(
        *,
        config: RetrieverConfig,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        chunks: Sequence[Chunk] | None = None,
) -> Retriever:
    """
    Build a retriever from config and available retrieval dependencies.
    :param config:
    :param embedder:
    :param vector_store:
    :param chunks:
    :return:
    """
    
    if config.type == RetrieverType.VECTOR:
        if embedder is None:
            raise ValueError("Vector retrieval requires an embedder.")
        
        if vector_store is None:
            raise ValueError("Vector retrieval requires a vector store.")
        
        return VectorRetriever(
            embedder=embedder,
            store=vector_store,
            default_top_k=config.top_k,
        )
    
    if config.type == RetrieverType.BM25:
        if chunks is None:
            raise ValueError("BM25 retrieval requires a chunks.")
        
        return BM25Retriever(
            chunks=chunks,
            default_top_k=config.top_k,
        )
    
    if config.type == RetrieverType.HYBRID:
        if embedder is None:
            raise ValueError("Hybrid retrieval requires an embedder.")
        
        if vector_store is None:
            raise ValueError("Hybrid retrieval requires a vector store.")
        
        if chunks is None:
            raise ValueError("Hybrid retrieval requires a chunks.")
        
        vector_retriever = VectorRetriever(
            embedder=embedder,
            store=vector_store,
            default_top_k=config.top_k,
        )
        bm25_retriever = BM25Retriever(
            chunks=chunks,
            default_top_k=config.top_k,
        )
        
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            default_top_k=config.top_k,
        )
    
    raise ValueError(f"Unsupported retriever type: {config.type}")