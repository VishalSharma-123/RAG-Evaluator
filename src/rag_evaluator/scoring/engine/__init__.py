from __future__ import annotations

from rag_evaluator.scoring.engine.aggregation import (
    ScoringEngine,
    to_eval_result,
)
from rag_evaluator.scoring.engine.base import (
    ChunkRelevanceFn,
    ChunkRelevanceScorer,
    FailureClassifier,
    GenerationScorer,
    RetrievalScorer,
    SampleScorer,
)
from rag_evaluator.scoring.engine.chunk_relevance import (
    DefaultChunkRelevanceScorer,
    score_chunk_relevance,
    score_chunk_relevance_batch,
)
from rag_evaluator.scoring.engine.failures import (
    DefaultFailureClassifier,
    FailureBreakdown,
    classify_generation_failures,
    classify_retrieval_failures,
)
from rag_evaluator.scoring.engine.generation import (
    DefaultGenerationScorer,
    score_generation_metrics,
    score_generation_metrics_batch,
)
from rag_evaluator.scoring.engine.retrieval import (
    DefaultRetrievalScorer,
    score_retrieval_metrics,
    score_retrieval_metrics_batch,
)
from rag_evaluator.scoring.engine.types import (
    ChunkRelevanceScore,
    ChunkRelevanceStrategy,
    ScoredSample,
    ScoringRequest,
)

__all__ = [
    "ChunkRelevanceFn",
    "ChunkRelevanceScore",
    "ChunkRelevanceScorer",
    "ChunkRelevanceStrategy",
    "DefaultChunkRelevanceScorer",
    "DefaultFailureClassifier",
    "DefaultGenerationScorer",
    "DefaultRetrievalScorer",
    "FailureBreakdown",
    "FailureClassifier",
    "GenerationScorer",
    "RetrievalScorer",
    "SampleScorer",
    "ScoredSample",
    "ScoringEngine",
    "ScoringRequest",
    "classify_generation_failures",
    "classify_retrieval_failures",
    "score_chunk_relevance",
    "score_chunk_relevance_batch",
    "score_generation_metrics",
    "score_generation_metrics_batch",
    "score_retrieval_metrics",
    "score_retrieval_metrics_batch",
    "to_eval_result",
]