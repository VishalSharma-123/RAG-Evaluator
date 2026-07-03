from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.reranking.types import Reranker
from rag_evaluator.schemas import EvalSample, RetrievedChunk


@dataclass(frozen=True)
class CrossEncoderReranker(Reranker):
    """
    Local cross-encoder reranker backed by sentence-transformers.
    """

    configured_type: str
    model_name: str
    batch_size: int = 32
    device: str | None = None
    show_progress_bar: bool = False
    max_length: int | None = None
    trust_remote_code: bool = False
    cache_dir: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    implementation_name: str = "cross_encoder"
    implemented: bool = True
    _model: Any | None = field(default=None, init=False, repr=False, compare=False)

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not retrieved_chunks:
            return []

        model = self._load_model()
        pairs = [(sample.question, item.chunk.text) for item in retrieved_chunks]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        score_list = _coerce_score_list(scores)
        if len(score_list) != len(retrieved_chunks):
            raise ValueError(
                "Cross-encoder reranker returned a score count that does not match "
                "the number of candidate chunks."
            )

        ordered_indices = sorted(
            range(len(retrieved_chunks)),
            key=lambda index: (-score_list[index], index),
        )
        selected_indices = ordered_indices[:top_k]

        return [
            retrieved_chunks[index].model_copy(update={"rank": rank})
            for rank, index in enumerate(selected_indices, start=1)
        ]

    def _load_model(self) -> Any:
        model = self._model
        if model is not None:
            return model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires `sentence-transformers`. "
                "Install it with: python -m pip install -e '.[retrieval]'"
            ) from exc

        model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        object.__setattr__(self, "_model", model)
        return model


def _coerce_score_list(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        values = scores.tolist()
    else:
        values = scores

    if isinstance(values, list):
        return [float(value) for value in values]

    return [float(values)]
