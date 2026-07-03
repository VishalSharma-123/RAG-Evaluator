from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.config import PipelineConfig
from rag_evaluator.generation.base import Generator
from rag_evaluator.reranking.types import Reranker
from rag_evaluator.retrieval.base import Retriever
from rag_evaluator.schemas import (
    Chunk,
    EvalResult,
    FinalContext,
    RetrievedChunk,
)
from rag_evaluator.scoring.judges.base import GenerationJudge


@dataclass(frozen=True)
class PipelineRuntime:
    """
    Prepared runtime for one pipeline variant.
    """

    pipeline: PipelineConfig
    chunks: list[Chunk]
    retriever: Retriever
    reranker: Reranker
    generator: Generator
    judge: GenerationJudge
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleExecutionArtifacts:
    """
    Structured intermediate artifacts for one sample execution.
    """

    retrieved_chunks: list[RetrievedChunk]
    final_context: FinalContext
    reranker_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_context_chunks(self) -> list[Chunk]:
        return self.final_context.chunks


@dataclass(frozen=True)
class PipelineRunOutput:
    """
    Outputs from running one pipeline variant over an eval suite.
    """

    pipeline: PipelineConfig
    chunks: list[Chunk]
    results: list[EvalResult]
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRunOutput:
    """
    Outputs from running all pipeline variants for one experiment.
    """

    experiment_name: str
    pipeline_runs: list[PipelineRunOutput]
