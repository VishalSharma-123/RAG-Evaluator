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


@dataclass(frozen=True)
class IndexBuildPipelineSummary:
    """
    Structured summary for one built retrieval index.
    """

    pipeline_name: str
    chunk_count: int
    store_provider: str
    collection_name: str | None
    persist_directory: Path | None
    retriever_type: str


@dataclass(frozen=True)
class IndexBuildSummary:
    """
    Structured summary returned by index-building orchestration.
    """

    experiment_name: str
    config_path: Path
    document_count: int
    sample_count: int
    pipeline_indexes: list[IndexBuildPipelineSummary]


@dataclass(frozen=True)
class ScoreRunSummary:
    """
    Structured summary returned by deterministic persisted-run rescoring.
    """

    run_id: str
    database_path: Path
    sample_count: int
    metric_count: int
    failure_label_count: int


@dataclass(frozen=True)
class DashboardLaunchSummary:
    """
    Structured summary for dashboard launch command.
    """

    database_path: Path
    app_path: Path
    return_code: int
