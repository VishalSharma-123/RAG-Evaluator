from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_evaluator.config import ExperimentConfig
from rag_evaluator.ingestion.chunkers import SourceDocument
from rag_evaluator.schemas import EvalSample


@dataclass(frozen=True)
class DatasetLoadSummary:
    """
    Sample count loaded for one configured dataset.
    """

    dataset_name: str
    sample_count: int


@dataclass(frozen=True)
class ExperimentInputs:
    """
    Fully prepared inputs for experiment execution.
    """

    experiment: ExperimentConfig
    samples: list[EvalSample]
    documents: list[SourceDocument]
    datasets: list[DatasetLoadSummary] = field(default_factory=list)


@dataclass(frozen=True)
class PersistedPipelineRunSummary:
    """
    Persistence outcome for one executed pipeline variant.
    """

    pipeline_name: str
    run_id: str
    result_count: int
    chunk_count: int
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRunSummary:
    """
    Structured summary returned by the experiment application service.
    """

    experiment_name: str
    config_path: Path
    database_path: Path
    sample_count: int
    document_count: int
    datasets: list[DatasetLoadSummary]
    pipeline_runs: list[PersistedPipelineRunSummary]


@dataclass(frozen=True)
class SyntheticGenerationSummary:
    """
    Structured summary returned by synthetic generation orchestration.
    """

    chunks_path: Path
    output_path: Path
    provider: str
    model: str
    chunk_count: int
    sample_count: int
