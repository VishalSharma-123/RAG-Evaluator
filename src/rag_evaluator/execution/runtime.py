from __future__ import annotations

from typing import Any

from rag_evaluator.config import LLMProvider, PipelineConfig, RetrieverType
from rag_evaluator.execution.types import PipelineRuntime
from rag_evaluator.generation.base import Generator
from rag_evaluator.generation.chat_completion import ChatCompletionGenerator
from rag_evaluator.ingestion.chunkers import SourceDocument, build_chunker
from rag_evaluator.ingestion.embedders import Embedder, build_embedder
from rag_evaluator.ingestion.stores import VectorStore, build_vector_store
from rag_evaluator.reranking.factory import build_reranker
from rag_evaluator.retrieval import build_retriever
from rag_evaluator.scoring.judges import HeuristicJudge, NemotronJudge, OpenAIJudge
from rag_evaluator.scoring.judges.base import GenerationJudge


def build_pipeline_runtime(
    pipeline: PipelineConfig,
    documents: list[SourceDocument],
) -> PipelineRuntime:
    chunker = build_chunker(
        chunker_type=pipeline.chunker.type.value,
        chunk_size=pipeline.chunker.chunk_size,
        chunk_overlap=pipeline.chunker.chunk_overlap,
    )
    chunks = chunker.chunk(documents)

    embedder = _build_optional_embedder(pipeline)
    vector_store = _build_optional_vector_store(pipeline, chunks=chunks, embedder=embedder)
    retriever = build_retriever(
        config=pipeline.retriever,
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )

    return PipelineRuntime(
        pipeline=pipeline,
        chunks=chunks,
        retriever=retriever,
        reranker=build_reranker(pipeline),
        generator=build_generator(pipeline),
        judge=build_judge(pipeline),
        metadata={
            "chunk_count": len(chunks),
            "retriever_type": pipeline.retriever.type.value,
            "reranker_type": pipeline.reranker.type.value,
            "generator_model": pipeline.generator.model,
            "judge_model": pipeline.judge.model,
        },
    )


def build_generator(pipeline: PipelineConfig) -> Generator:
    return ChatCompletionGenerator(config=pipeline.generator)


def build_judge(pipeline: PipelineConfig) -> GenerationJudge:
    if pipeline.judge.provider == LLMProvider.OPENAI:
        return OpenAIJudge(config=pipeline.judge)

    if pipeline.judge.model.startswith("nvidia/nemotron"):
        return NemotronJudge(config=pipeline.judge)

    return HeuristicJudge()


def _build_optional_embedder(pipeline: PipelineConfig) -> Embedder | None:
    if pipeline.retriever.type == RetrieverType.BM25:
        return None

    return build_embedder(pipeline.embedder)


def _build_optional_vector_store(
    pipeline: PipelineConfig,
    *,
    chunks: list[Any],
    embedder: Embedder | None,
) -> VectorStore | None:
    if pipeline.retriever.type == RetrieverType.BM25:
        return None

    if embedder is None:
        raise ValueError("Retriever requires an embedder but none was built.")

    chunk_embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store = build_vector_store(
        provider=pipeline.store.provider.value,
        collection_name=pipeline.store.collection_name,
        persist_directory=pipeline.store.persist_directory,
        metadata=pipeline.store.metadata,
    )
    vector_store.add(chunks, chunk_embeddings)
    return vector_store
