from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics
from rag_evaluator.scoring.engine.base import GenerationScorer
from rag_evaluator.scoring.generation import score_generation as _score_generation
from rag_evaluator.scoring.generation import score_generation_batch as _score_generation_batch


def score_generation_metrics(
        sample: EvalSample,
        generated_answer: GeneratedAnswer | None,
) -> GenerationMetrics | None:
    """
    Score generation metrics for one sample.
    :param sample:
    :param generated_answer:
    :return:
    """
    return _score_generation(sample, generated_answer)

def score_generation_metrics_batch(
        samples: Sequence[EvalSample],
        generated_by_sample_id: dict[str, GeneratedAnswer | None],
) -> dict[str, GenerationMetrics | None]:
    """
    Score generation metrics for a batch of samples keyed by sample_id.
    :param samples:
    :param generated_by_sample_id:
    :return:
    """
    return _score_generation_batch(samples, generated_by_sample_id)

@dataclass(frozen=True)
class DefaultGenerationScorer(GenerationScorer):
    """
    Default generation scorer used by the scoring engine.
    """
    
    def score(
            self,
            sample: EvalSample,
            generated_answer: GeneratedAnswer | None,
            *,
            context_chunks: list[Chunk],
            metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics | None:
        del context_chunks, metadata
        return score_generation_metrics(sample, generated_answer)
