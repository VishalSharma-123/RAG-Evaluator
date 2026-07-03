from __future__ import annotations

import os
from dataclasses import dataclass

from rag_evaluator.config import PipelineConfig, RerankerType
from rag_evaluator.reranking.cohere import CohereReranker
from rag_evaluator.reranking.cross_encoder import CrossEncoderReranker
from rag_evaluator.reranking.openrouter import OpenRouterReranker
from rag_evaluator.reranking.types import Reranker
from rag_evaluator.schemas import EvalSample, RetrievedChunk


@dataclass(frozen=True)
class PassThroughReranker(Reranker):
    """
    Reranker placeholder that preserves retrieval ordering.
    """

    configured_type: str
    implementation_name: str = "pass_through"
    implemented: bool = False

    def rerank(
        self,
        sample: EvalSample,
        retrieved_chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        del sample
        selected = retrieved_chunks[:top_k]
        reranked: list[RetrievedChunk] = []

        for rank, item in enumerate(selected, start=1):
            reranked.append(item.model_copy(update={"rank": rank}))

        return reranked


def build_reranker(pipeline: PipelineConfig) -> Reranker:
    """
    Build the reranker used by execution-time context assembly.
    """
    reranker_type = pipeline.reranker.type

    if reranker_type == RerankerType.NONE:
        return PassThroughReranker(configured_type=reranker_type.value)

    model_name = _resolve_model_name(pipeline)

    if reranker_type == RerankerType.CROSS_ENCODER:
        return CrossEncoderReranker(
            configured_type=reranker_type.value,
            model_name=model_name,
            batch_size=int(pipeline.reranker.metadata.get("batch_size", 32)),
            device=pipeline.reranker.metadata.get("device"),
            show_progress_bar=bool(
                pipeline.reranker.metadata.get("show_progress_bar", False)
            ),
            max_length=pipeline.reranker.metadata.get("max_length"),
            trust_remote_code=bool(
                pipeline.reranker.metadata.get("trust_remote_code", False)
            ),
            cache_dir=pipeline.reranker.metadata.get("cache_dir"),
            revision=pipeline.reranker.metadata.get("revision"),
            local_files_only=bool(
                pipeline.reranker.metadata.get("local_files_only", False)
            ),
        )

    if reranker_type == RerankerType.COHERE:
        return CohereReranker(
            configured_type=reranker_type.value,
            model_name=model_name,
            api_key_env=str(
                pipeline.reranker.metadata.get("api_key_env", "COHERE_API_KEY")
            ),
            base_url=pipeline.reranker.metadata.get("base_url"),
            client_name=pipeline.reranker.metadata.get("client_name"),
            timeout=pipeline.reranker.metadata.get("timeout"),
        )

    if reranker_type == RerankerType.OPENROUTER:
        return OpenRouterReranker(
            configured_type=reranker_type.value,
            model_name=model_name,
            api_key=pipeline.reranker.metadata.get("api_key"),
            api_key_env=str(
                pipeline.reranker.metadata.get("api_key_env", "OPENROUTER_API_KEY")
            ),
            base_url=_resolve_openrouter_base_url(pipeline),
            timeout_seconds=_resolve_timeout_seconds(pipeline),
            http_referer=pipeline.reranker.metadata.get("http_referer"),
            app_name=pipeline.reranker.metadata.get("app_name"),
        )

    raise NotImplementedError(
        f"Reranker type `{reranker_type.value}` is not implemented yet."
    )


def _resolve_model_name(pipeline: PipelineConfig) -> str:
    model_name = pipeline.reranker.model or pipeline.reranker.metadata.get("model")

    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()

    raise ValueError(
        "Reranker configuration requires a model name for implemented reranker types."
    )


def _resolve_openrouter_base_url(pipeline: PipelineConfig) -> str:
    base_url = (
        pipeline.reranker.metadata.get("base_url")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1/rerank"
    )
    return str(base_url)


def _resolve_timeout_seconds(pipeline: PipelineConfig) -> float:
    raw_timeout = pipeline.reranker.metadata.get("timeout_seconds", 60.0)
    try:
        return float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid OpenRouter reranker timeout_seconds value: {raw_timeout!r}"
        ) from exc
