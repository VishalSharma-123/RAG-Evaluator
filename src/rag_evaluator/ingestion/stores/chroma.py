from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_evaluator.ingestion.embedders import Embedding
from rag_evaluator.ingestion.stores.base import VectorStore
from rag_evaluator.schemas import Chunk, RetrievedChunk


@dataclass(frozen=True)
class ChromaVectorStore(VectorStore):
    """
    Persistent Chroma DB Vector storage.
    """
    
    collection_name: str
    persist_directory: str | Path = "storage/chroma"
    
    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("Collection name is required")
        
        object.__setattr__(self, "persist_directory", Path(self.persist_directory))
    
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """
        Add chunks and corresponding embeddings.
        :param chunks:
        :param embeddings:
        :return:
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Length of chunks and embeddings do not match")
        
        if not chunks:
            return
        
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        normalized_embeddings: list[Embedding] = []
        
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not embedding:
                raise ValueError(f"Empty embedding for chunk_id = {chunk.chunk_id}")
        
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(_chunk_to_metadata(chunk))
            normalized_embeddings.append(embedding)

        collection = self._collection()
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=normalized_embeddings,
            metadatas=metadatas,
        )
    
    def search(
            self,
            query_embedding: Embedding,
            *,
            top_k: int,
            retriever_name: str = "chroma"
    ) -> list[RetrievedChunk]:
        """
        Return top-k chunks ranked by Chroma distance.
        :param query_embedding:
        :param top_k:
        :param retriever_name:
        :return:
        """
        if top_k < 1:
            raise ValueError("Top k must be greater than or equal to 1")
        
        if not query_embedding:
            raise ValueError("Empty query embedding")
        
        collection = self._collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        
        retrieved_chunks: list[RetrievedChunk] = []
        
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            chunk = _metadata_to_chunk(
                chunk_id=chunk_id,
                text=document or "",
                metadata=metadata or {},
            )
            
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    rank=rank,
                    score=_distance_to_score(float(distance)),
                    retriever_name=retriever_name,
                    metadata={
                        "distance": float(distance),
                        "store": "chroma",
                        "collection_name": self.collection_name,
                    },
                )
            )
        
        return retrieved_chunks

    def _collection(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires `chromadb`. Please install"
            ) from exc
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path = str(self.persist_directory))
        return client.get_or_create_collection(
            name = self.collection_name,
            metadata = {
                "hnsw:space": "cosine",
            },
        )
    
def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata = _sanitize_metadata(chunk.metadata)
    
    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
    )
    
    return metadata

def _metadata_to_chunk(
        *,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
) -> Chunk:
    clean_metadata = dict(metadata)
    
    stored_chunk_id = str(clean_metadata.pop("chunk_id", chunk_id))
    document_id = str(clean_metadata.pop("document_id", ""))
    start_char = clean_metadata.pop("start_char", None)
    end_char = clean_metadata.pop("end_char", None)
    
    return Chunk(
        chunk_id=stored_chunk_id,
        document_id=document_id,
        text=text,
        start_char=_optional_int(start_char),
        end_char=_optional_int(end_char),
        metadata=clean_metadata,
    )

def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata values must be scalar primitives
    :param metadata:
    :return:
    """
    sanitized: dict[str, Any] = {}
    
    for key, value in metadata.items():
        if value is None:
            continue
        
        if isinstance(value, str|int|float|bool):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    
    return sanitized

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    
    return int(value)

def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))
        
        
    
