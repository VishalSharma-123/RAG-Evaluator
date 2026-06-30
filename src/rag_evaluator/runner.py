from __future__ import annotations

from dataclasses import asdict
import time
from dataclasses import dataclass

from rag_evaluator.config import PipelineConfig
from rag_evaluator.ingestion.chunkers import SourceDocument, build_chunker
from rag_evaluator.ingestion.embedders import build_embedder
from rag_evaluator.ingestion.stores import build_vector_store
from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.retrieval import build_retriever
from rag_evaluator.schemas import Chunk, EvalResult, EvalSample, GeneratedAnswer
from rag_evaluator.scoring.failures import classify_failures
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge
from rag_evaluator.scoring.retrieval import score_retrieval


@dataclass(frozen=True)
class PipelineRunOutput:
    """
    Raw outputs from running one pipeline over an eval suite.
    """
    
    chunks: list[Chunk]
    results: list[EvalResult]

class SimpleExtractiveGenerator:
    """
    Minimal generator for smoke tests that returns top retrieved context.
    """
    
    model_name: str = "simple_extractive_generator"
    
    def generate(self, sample: EvalSample, context_chunks: list[Chunk]) -> GeneratedAnswer:
        """
        Generate a placeholder answer from retrieved context.
        :param sample:
        :param context_chunks:
        :return:
        """
        
        start_time = time.perf_counter()
        
        if context_chunks:
            answer = context_chunks[0].text
        elif sample.is_answerable:
            answer = ""
        else:
            answer = "I dont know"
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        return GeneratedAnswer(
            sample_id = sample.sample_id,
            answer=answer,
            model_name=self.model_name,
            latency_ms=latency_ms,
            cost_usd=0.0
        )

def run_single_pipeline(
        *,
        pipeline: PipelineConfig,
        samples: list[EvalSample],
        documents: list[SourceDocument],
) -> PipelineRunOutput:
    """
    Run one local smoke-test RAG pipeline.
    :param pipeline:
    :param samples:
    :param documents:
    :return:
    """
    chunker = build_chunker(
        chunker_type=pipeline.chunker.type.value,
        chunk_size=pipeline.chunker.chunk_size,
        chunk_overlap=pipeline.chunker.chunk_overlap,
    )
    chunks = chunker.chunk(documents)
    
    embedder = build_embedder(pipeline.embedder)
    chunk_embedding = embedder.embed_texts([chunk.text for chunk in chunks])
    
    vector_store = build_vector_store(
        provider=pipeline.store.provider.value,
        collection_name=pipeline.store.collection_name,
        persist_directory=pipeline.store.persist_directory,
        metadata=pipeline.store.metadata,
    )
    vector_store.add(chunks, chunk_embedding)
    
    retriever = build_retriever(
        config=pipeline.retriever,
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )
    
    generator = SimpleExtractiveGenerator()
    judge = HeuristicJudge()
    results: list[EvalResult] = []
    
    for sample in samples:
        retrieved_chunks = retriever.retrieve(
            sample.question,
            top_k=pipeline.retriever.top_k,
        )
        context_chunks = [retrieved.chunk for retrieved in retrieved_chunks]
        generated_answer = generator.generate(sample, context_chunks)
        
        retrieval_metrics = score_retrieval(
            sample,
            retrieved_chunks,
            k=pipeline.retriever.top_k,
        )
        
        generation_metrics = judge.score(
            sample,
            generated_answer,
            context_chunks=context_chunks,
            metadata = {
                "pipeline_name": pipeline.name,
                "judge_model": pipeline.judge.model,
            }
        )
        question_type_signals = get_question_type_rule(sample.question_type).score_answer(
            sample,
            generated_answer,
            context_chunks=context_chunks,
        )
        
        failure_modes = classify_failures(
            sample,
            retrieved_chunks,
            generated_answer=generated_answer,
            hallucination_score=generation_metrics.hallucination,
            partial_answer_score=(
                1.0 - generation_metrics.faithfulness
                if generation_metrics.faithfulness is not None
                else None
            ),
            retrieval_k=pipeline.retriever.top_k,
        )
        
        results.append(
            EvalResult(
                run_id = pipeline.name,
                sample=sample,
                retrieved_chunks=retrieved_chunks,
                generated_answer=generated_answer,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                failure_modes=failure_modes,
                metadata={
                    "pipeline_name": pipeline.name,
                    "generator_name": generator.model_name,
                    "judge_name": judge.__class__.__name__,
                    "judge_model": pipeline.judge.model,
                    "question_type_signals": asdict(question_type_signals),
                }
            ))
        
    return PipelineRunOutput(
        chunks=chunks,
        results=results,
    )
