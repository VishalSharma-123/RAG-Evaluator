from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.schemas import EvalResult


class ResultsStoreError(ValueError):
    """
    Raised when results persistence fails.
    """


class ResultsStore(ABC):
    """
    Base interface for experiment result persistence.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the backing store schema and required objects.
        """
        raise NotImplementedError

    @abstractmethod
    def write_run(
        self,
        *,
        run_id: str,
        experiment: ExperimentConfig,
        pipeline: PipelineConfig,
        results: list[EvalResult],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist one evaluated run and its per-sample results.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_run(self, run_id: str) -> list[dict[str, Any]]:
        """
        Return persisted records for one run.
        """
        raise NotImplementedError
