from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.datasets.config import DatasetConfig
from rag_evaluator.schemas import EvalSample


class DatasetNormalizer(ABC):
    """
    Base interface for raw dataset record normalizers.
    """
    
    dataset_key: str
    
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
    
    @abstractmethod
    def normalize_record(
            self,
            record: dict[str, Any],
            *,
            index: int,
            split: str
    ) -> EvalSample:
        """
        Build a stable sample ID for a normalized record.
        :param record:
        :param index:
        :param split:
        :return:
        """
        raise NotImplementedError
    
    def sample_id(self, split: str,index: int, source_id: Any = None) -> str:
        """
        Build a stable sample ID for a normalized record.
        :param split:
        :param index:
        :param source_id:
        :return:
        """
        if source_id is not None:
            return f"{self.config.name}:{split}:{source_id}"
        
        return f"{self.config.name}:{split}:{index}"
    
    def source_id(self, record: dict[str, Any], *candidate_keys: str) -> str | None:
        """
        Extract a source ID from the first matching candidate key.
        :param record:
        :param candidate_keys:
        :return:
        """
        
        for key in candidate_keys:
            value = record.get(key)
            
            if value is not None:
                return str(value)
        
        return None
    
    def metadata(self, **values: Any) -> dict[str, Any]:
        """
        Build a common metadata for a normalized record.
        :param values:
        :return:
        """
        metadata: dict[str, Any] = {
            "dataset_name": self.config.dataset_name,
            "dataset_config": self.config.dataset_config,
            "domain": self.config.domain,
        }
        
        metadata.update(values)
        return metadata
