from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag_evaluator.execution.runtime import build_pipeline_runtime
from rag_evaluator.execution.types import (
    FinalContext,
    PipelineRunOutput,
    PipelineRuntime,
    SampleExecutionArtifacts,
)
from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.schemas import Chunk, EvalResult, EvalSample, GeneratedAnswer
from rag_evaluator.scoring.failures import classify_failures
from rag_evaluator.scoring.engine.chunk_relevance import resolve_retrieval_gold
from rag_evaluator.scoring.retrieval import score_retrieval


def run_pipeline_variant(
    *,
    runtime: PipelineRuntime,
    samples: list[EvalSample],
) -> PipelineRunOutput:
    results = [run_sample(runtime=runtime, sample=sample) for sample in samples]

    return PipelineRunOutput(
        pipeline=runtime.pipeline,
        chunks=runtime.chunks,
        results=results,
        runtime_metadata={
            **runtime.metadata,
            "sample_count": len(samples),
        },
    )


def run_single_pipeline(
    *,
    pipeline,
    samples: list[EvalSample],
    documents,
) -> PipelineRunOutput:
    runtime = build_pipeline_runtime(
        pipeline=pipeline,
        documents=documents,
    )
    return run_pipeline_variant(runtime=runtime, samples=samples)


def run_sample(
    *,
    runtime: PipelineRuntime,
    sample: EvalSample,
) -> EvalResult:
    artifacts = _execute_sample(runtime=runtime, sample=sample)
    generated_answer = _generate_answer(
        runtime=runtime,
        sample=sample,
        artifacts=artifacts,
    )

    retrieval_metrics = score_retrieval(
        sample,
        artifacts.retrieved_chunks,
        k=runtime.pipeline.retriever.top_k,
    )
    retrieval_gold = resolve_retrieval_gold(sample, artifacts.retrieved_chunks)
    generation_metrics = runtime.judge.score(
        sample,
        generated_answer,
        context_chunks=artifacts.final_context.chunks,
        metadata={
            "pipeline_name": runtime.pipeline.name,
            "judge_model": runtime.pipeline.judge.model,
            "final_context_chunk_ids": [
                chunk.chunk_id for chunk in artifacts.final_context.chunks
            ],
        },
    )
    question_type_signals = get_question_type_rule(sample.question_type).score_answer(
        sample,
        generated_answer,
        context_chunks=artifacts.final_context.chunks,
    )
    failure_modes = classify_failures(
        sample,
        artifacts.retrieved_chunks,
        generated_answer=generated_answer,
        context_was_used=bool(artifacts.final_context.chunks),
        hallucination_score=generation_metrics.hallucination,
        partial_answer_score=(
            1.0 - generation_metrics.faithfulness
            if generation_metrics.faithfulness is not None
            else None
        ),
        retrieval_k=runtime.pipeline.retriever.top_k,
    )

    return EvalResult(
        run_id=runtime.pipeline.name,
        sample=sample,
        retrieved_chunks=artifacts.retrieved_chunks,
        final_context=artifacts.final_context,
        generated_answer=generated_answer,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        failure_modes=failure_modes,
        metadata={
            "pipeline_name": runtime.pipeline.name,
            "source_pipeline_name": runtime.pipeline.metadata.get(
                "source_pipeline_name",
                runtime.pipeline.name,
            ),
            "generator_model": runtime.pipeline.generator.model,
            "judge_name": runtime.judge.__class__.__name__,
            "judge_model": runtime.pipeline.judge.model,
            "question_type_signals": asdict(question_type_signals),
            "retrieved_chunk_ids": [
                item.chunk.chunk_id for item in artifacts.retrieved_chunks
            ],
            "retrieval_gold_strategy": retrieval_gold.strategy,
            "resolved_gold_chunk_ids": retrieval_gold.resolved_gold_chunk_ids,
            "retrieved_relevance_flags": retrieval_gold.relevant_flags,
            "final_context_chunk_ids": [
                chunk.chunk_id for chunk in artifacts.final_context.chunks
            ],
            "reranker": artifacts.reranker_metadata,
        },
    )


def _execute_sample(
    *,
    runtime: PipelineRuntime,
    sample: EvalSample,
) -> SampleExecutionArtifacts:
    retrieved_chunks = runtime.retriever.retrieve(
        sample.question,
        top_k=runtime.pipeline.retriever.top_k,
    )
    reranked_chunks = runtime.reranker.rerank(
        sample,
        retrieved_chunks,
        top_k=runtime.pipeline.reranker.top_k or runtime.pipeline.retriever.top_k,
    )
    reranker_metadata = {
        "configured_type": runtime.reranker.configured_type,
        "implementation": runtime.reranker.implementation_name,
        "implemented": runtime.reranker.implemented,
        "selected_chunk_ids": [item.chunk.chunk_id for item in reranked_chunks],
    }

    return SampleExecutionArtifacts(
        retrieved_chunks=retrieved_chunks,
        final_context=FinalContext(
            chunks=[item.chunk for item in reranked_chunks],
            rendered_text=_render_context_text([item.chunk for item in reranked_chunks]),
            metadata={"reranker": reranker_metadata},
        ),
        reranker_metadata=reranker_metadata,
    )


def _generate_answer(
    *,
    runtime: PipelineRuntime,
    sample: EvalSample,
    artifacts: SampleExecutionArtifacts,
) -> GeneratedAnswer:
    return runtime.generator.generate(
        sample,
        artifacts.final_context.chunks,
        metadata=_build_generation_metadata(runtime=runtime, artifacts=artifacts),
    )


def _build_generation_metadata(
    *,
    runtime: PipelineRuntime,
    artifacts: SampleExecutionArtifacts,
) -> dict[str, Any]:
    return {
        "pipeline_name": runtime.pipeline.name,
        "generator_model": runtime.pipeline.generator.model,
        "generator_provider": runtime.pipeline.generator.provider.value,
        "final_context_chunk_ids": [
            chunk.chunk_id for chunk in artifacts.final_context.chunks
        ],
        "final_context_text": artifacts.final_context.rendered_text,
        "reranker": artifacts.reranker_metadata,
    }


def _render_context_text(context_chunks: list[Chunk]) -> str:
    if not context_chunks:
        return ""

    return "\n\n".join(
        f"[{index}] {chunk.text}"
        for index, chunk in enumerate(context_chunks, start=1)
    )
