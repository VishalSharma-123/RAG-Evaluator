from __future__ import annotations

from abc import ABC, abstractmethod

from rag_evaluator.datasets.config import DatasetConfig
from rag_evaluator.schemas import EvalSample


class DatasetAdapter(ABC):
    """
    Base interface for dataset adapters that produce EvalSample records.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> list[EvalSample]:
        """
        Load source records and normalize them into EvalSample objects.
        """
        raise NotImplementedError
