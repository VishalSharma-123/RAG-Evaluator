from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.ingestion.embedders.base import Embedder, Embedding


@dataclass(frozen=True)
class SentenceTransformerEmbedder(Embedder):
    """
    Sentence-transformers based local embedder for BGE-style models.
    """

    model_name: str
    batch_size: int = 32
    show_progress_bar: bool = False
    _model: Any | None = field(default=None, init=False, repr=False, compare=False)

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        """
        Embed a batch of texts with sentence-transformers.
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )

        return [embedding.tolist() for embedding in embeddings]

    def _load_model(self) -> Any:
        model = self._model
        if model is not None:
            return model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires `sentence-transformers`. "
                "Install it with: python -m pip install -e '.[retrieval]'"
            ) from exc

        model = SentenceTransformer(self.model_name)
        object.__setattr__(self, "_model", model)
        return model
