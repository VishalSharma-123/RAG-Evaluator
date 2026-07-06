from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.question_types.registry import get_question_type_rule
from rag_evaluator.schemas import (
    Chunk,
    EvalResult,
    EvalSample,
    FinalContext,
    GeneratedAnswer,
    RetrievedChunk,
)
from rag_evaluator.scoring.engine.chunk_relevance import (
    DefaultChunkRelevanceScorer,
)
from rag_evaluator.scoring.engine.failures import DefaultFailureClassifier
from rag_evaluator.scoring.engine.generation import DefaultGenerationScorer
from rag_evaluator.scoring.engine.retrieval import DefaultRetrievalScorer
from rag_evaluator.scoring.engine.types import ChunkRelevanceScore, ScoredSample, ScoringRequest


@dataclass(frozen=True)
class ScoringEngine:
    """
    Orchestrator retrieval scoring, generation config, chunk relevance scoring,
    and failure classification for one evaluated sample.
    """
    
    retrieval_scorer: DefaultRetrievalScorer = field(default_factory=DefaultRetrievalScorer)
    generation_scorer: DefaultGenerationScorer = field(default_factory=DefaultGenerationScorer)
    chunk_relevance_scorer: DefaultChunkRelevanceScorer = field(default_factory=DefaultChunkRelevanceScorer)
    failure_classifier: DefaultFailureClassifier = field(default_factory=DefaultFailureClassifier)
    
    def score_sample(
            self,
            *,
            run_id: str,
            sample: EvalSample,
            retrieved_chunks: list[RetrievedChunk],
            final_context: FinalContext | None = None,
            generated_answer: GeneratedAnswer | None = None,
            retrieval_k: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ScoredSample:
        final_context = final_context or FinalContext()
        metadata = metadata or {}
        if retrieval_k is None:
            retrieval_k = len(retrieved_chunks)
        
        retrieval_metrics = self.retrieval_scorer.score(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            k=retrieval_k,
        )
        generation_metrics = self.generation_scorer.score(
            sample=sample,
            generated_answer=generated_answer,
            context_chunks=final_context.chunks,
            metadata=metadata,
        )
        chunk_relevance = self.chunk_relevance_scorer.score_batch(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            context_chunks=final_context.chunks,
            metadata=metadata,
        )
        
        breakdown = self.failure_classifier.classify_breakdown(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
            context_was_used=bool(final_context.chunks),
            hallucination_score=(
                generation_metrics.hallucination if generation_metrics is not None else None
            ),
            partial_answer_score=(
                1.0 - generation_metrics.faithfulness
                if generation_metrics is not None and generation_metrics.faithfulness is not None
                else None
            ),
            retrieval_k=retrieval_k,
        )
        
        question_type_signals = self._build_question_type_signals(
            sample=sample,
            generated_answer=generated_answer,
            context_chunks=final_context.chunks,
        )
        
        return ScoredSample(
            run_id=run_id,
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            final_context=final_context,
            generated_answer=generated_answer,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            chunk_relevance=chunk_relevance,
            retrieval_failure_modes=breakdown.retrieval,
            generation_failure_modes=breakdown.generation,
            failure_modes=breakdown.all,
            metadata={
                **metadata,
                "question_type_signals": question_type_signals,
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved_chunks],
                "final_context_chunk_ids": [chunk.chunk_id for chunk in final_context.chunks],
                "chunk_relevance": [
                    {
                        "chunk_id": item.chunk_id,
                        "strategies":item.strategies,
                        "overall_score":item.overall_score,
                        "metadata":item.metadata,
                    }
                    for item in chunk_relevance
                ],
                "retrieval_failure_modes": [item.value for item in breakdown.retrieval],
                "generation_failure_modes": [item.value for item in breakdown.generation],
            },
        )
    
    def score_batch(
            self,
            *,
            run_id: str,
            samples: list[EvalSample],
            retrieved_by_sample_id: dict[str, list[RetrievedChunk]],
            final_context_by_sample_id: dict[str, FinalContext] | None = None,
            generated_by_sample_id: dict[str, GeneratedAnswer | None] | None = None,
            retrieval_k: int | None = None,
            metadata_by_sample_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[ScoredSample]:
        final_context_by_sample_id = final_context_by_sample_id or {}
        generated_by_sample_id = generated_by_sample_id or {}
        metadata_by_sample_id = metadata_by_sample_id or {}
        
        return [
            self.score_sample(
                run_id=run_id,
                sample=sample,
                retrieved_chunks=retrieved_by_sample_id.get(sample.sample_id, []),
                final_context=final_context_by_sample_id.get(sample.sample_id),
                generated_answer=generated_by_sample_id.get(sample.sample_id),
                retrieval_k=retrieval_k,
                metadata=metadata_by_sample_id.get(sample.sample_id, {}),
            )
            for sample in samples
        ]
    
    def score_request(self, request: ScoringRequest) -> ScoredSample:
        return self.score_sample(
            run_id=request.run_id,
            sample=request.sample,
            retrieved_chunks=request.retrieved_chunks,
            final_context=request.final_context,
            generated_answer=request.generated_answer,
            retrieval_k=request.retrieval_k,
            metadata=request.metadata,
        )
    
    def score_chunk_relevance(
            self,
            sample: EvalSample,
            retrieved_chunk: RetrievedChunk,
            *,
            context_chunks: list[Chunk],
            metadata: dict[str, Any] | None = None,
    ) -> ChunkRelevanceScore:
        return self.chunk_relevance_scorer.score_chunk(
            sample=sample,
            retrieved_chunk=retrieved_chunk,
            context_chunks=context_chunks,
            metadata=metadata,
        )
    
    def _build_question_type_signals(
            self,
            sample: EvalSample,
            generated_answer: GeneratedAnswer | None,
            *,
            context_chunks: list[Chunk],
    ) -> dict[str, Any]:
        rule = get_question_type_rule(sample.question_type)
        type_signals = rule.score_answer(
            sample=sample,
            generated_answer=generated_answer,
            context_chunks=context_chunks,
        )
        return dict(type_signals.metadata)
    

def to_eval_result(scored_sample: ScoredSample) -> EvalResult:
    """
    Convert a scored sample into the existing persistence-friendly EvalResult.
    :param scored_sample:
    :return:
    """
    return EvalResult(
        run_id=scored_sample.run_id,
        sample=scored_sample.sample,
        retrieved_chunks=scored_sample.retrieved_chunks,
        final_context=scored_sample.final_context,
        generated_answer=scored_sample.generated_answer,
        retrieval_metrics=scored_sample.retrieval_metrics,
        generation_metrics=scored_sample.generation_metrics,
        failure_modes=scored_sample.failure_modes,
        metadata=scored_sample.metadata,
    )
